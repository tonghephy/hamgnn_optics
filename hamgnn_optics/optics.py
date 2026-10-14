from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.linalg import eigh

HARTREE_TO_EV = 2.0 * 13.605698066


BOHR_TO_ANGSTROM = 1.0 / 1.8897259886


TENSOR_LABELS = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")


@dataclass(frozen=True)
class RealSpaceMatrices:
    """Absolute r(R), lattice and H/S blocks with explicit units and filling."""

    hr: dict
    sr: dict
    rr: dict
    lattice: np.ndarray
    occupied_bands: int
    h_unit: str = "hartree"
    length_unit: str = "bohr"
    spin_degeneracy: int = 2


def correct_position_operator(rr, sr, centers):
    """Convert row-centered OpenMX r to absolute r, exactly once in assembly."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("Orbital centers must have shape (norbs, 3)")
    corrected = {}
    for key, block in rr.items():
        if key not in sr or block.shape != (len(centers), len(centers), 3):
            raise ValueError(f"Incompatible position/overlap blocks at R={key}")
        corrected[key] = np.asarray(block) + centers[:, None, :] * np.asarray(sr[key])[..., None]
    return corrected


def assemble_matrices(graph, prediction, settings, occupied_bands):
    """Compact padded on/off-site blocks using the original orbital ordering."""
    from pymatgen.core.periodic_table import Element
    if __package__:
        from .graph_data_gen import orbital_indices
    else:
        from hamgnn_optics.graph_data_gen import orbital_indices

    symbols = [Element.from_Z(int(z)).symbol for z in graph["z"]]
    local = orbital_indices(symbols, settings)
    offsets = np.r_[0, np.cumsum([len(indices) for indices in local])]
    norbs = int(offsets[-1])
    nao = settings["nao_max"]
    natoms = len(symbols)
    edges = graph["edge_index"]
    translations = [(0, 0, 0)] * natoms + [tuple(map(int, row)) for row in graph["cell_shift"]]
    keys = sorted(set(translations))
    hr = {key: np.zeros((norbs, norbs), dtype=np.complex128) for key in keys}
    sr = {key: np.zeros((norbs, norbs), dtype=np.complex128) for key in keys}
    rr = {key: np.zeros((norbs, norbs, 3), dtype=np.complex128) for key in keys}
    overlap = np.concatenate((graph["Son"], graph["Soff"])).reshape(-1, nao, nao)
    position = np.concatenate((graph["Ron"], graph["Roff"])).reshape(-1, nao, nao, 3)
    prediction = np.asarray(prediction).reshape(natoms + edges.shape[1], nao, nao)
    pairs = [(atom, atom) for atom in range(natoms)] + list(edges.T)
    for index, ((source, target), key) in enumerate(zip(pairs, translations)):
        rows, cols = local[source], local[target]
        row_slice = slice(offsets[source], offsets[source + 1])
        col_slice = slice(offsets[target], offsets[target + 1])
        selection = np.ix_(rows, cols)
        hr[key][row_slice, col_slice] += prediction[index][selection]
        sr[key][row_slice, col_slice] += overlap[index][selection]
        rr[key][row_slice, col_slice] += position[index][rows[:, None], cols[None, :], :]
    centers = np.repeat(graph["pos"], np.diff(offsets), axis=0)
    rr = correct_position_operator(rr, sr, centers)
    return RealSpaceMatrices(hr, sr, rr, graph["cell"].reshape(3, 3), occupied_bands)


@dataclass(frozen=True)
class OpticalResult:
    """Frequency-dependent linear optical response."""

    energy_ev: np.ndarray
    conductivity: np.ndarray
    dielectric: np.ndarray
    absorption_cm1: np.ndarray


def _hermitian_from_upper(matrix):
    upper = np.triu(np.asarray(matrix, dtype=np.complex128))
    return upper + np.triu(upper, 1).conj().T


def _fourier_upper(blocks, translations, phases, multiplier=None):
    dim = next(iter(blocks.values())).shape[0]
    result = np.zeros((dim, dim), dtype=np.complex128)
    for index, (translation, phase) in enumerate(zip(translations, phases)):
        factor = 1.0 if multiplier is None else multiplier[index]
        result += phase * factor * np.triu(np.asarray(blocks[translation]))
    return result


def _matrices_at_k(hr, sr, rr, keys, r_cart_angstrom, k_fractional):
    """Return H, S, their Cartesian derivatives, and r at one k point."""
    phases = np.exp(2j * np.pi * (np.asarray(keys, dtype=float) @ k_fractional))
    hk = _hermitian_from_upper(_fourier_upper(hr, keys, phases))
    sk = _hermitian_from_upper(_fourier_upper(sr, keys, phases))
    dh = np.empty((3, hk.shape[0], hk.shape[1]), dtype=np.complex128)
    ds = np.empty_like(dh)
    rk = np.empty_like(dh)
    for axis in range(3):
        derivative_factor = 1j * r_cart_angstrom[:, axis]
        dh[axis] = _hermitian_from_upper(
            _fourier_upper(hr, keys, phases, derivative_factor)
        )
        ds[axis] = _hermitian_from_upper(
            _fourier_upper(sr, keys, phases, derivative_factor)
        )

        r_blocks = {key: np.asarray(rr[key])[..., axis] for key in keys}
        completed = np.triu(_fourier_upper(r_blocks, keys, phases))
        rows, cols = np.tril_indices(hk.shape[0])
        rhs = (completed + 1j * ds[axis]).conj().T
        completed[rows, cols] = rhs[rows, cols]
        rk[axis] = completed
    return hk, sk, dh, ds, rk


def _velocity_in_band_basis(eigenvalues, eigenvectors, dh, ds, rk):
    velocity = np.empty_like(dh)
    energy_difference = eigenvalues[:, None] - eigenvalues[None, :]
    for axis in range(3):
        c_dh_c = eigenvectors.conj().T @ dh[axis] @ eigenvectors
        c_ds_c = eigenvectors.conj().T @ ds[axis] @ eigenvectors
        c_r_c = eigenvectors.conj().T @ rk[axis] @ eigenvectors
        upper = (
            c_dh_c
            - eigenvalues[:, None] * c_ds_c
            + 1j * energy_difference * c_r_c
        )
        velocity[axis] = _hermitian_from_upper(upper)
    return velocity


def calculate_optical_properties(
    hr_dict,
    sr_dict,
    rr_dict,
    lattice_matrix,
    occupied_bands,
    *,
    kmesh=(20, 20, 20),
    omega=(0.0, 5.0),
    domega=0.05,
    eta=0.01,
    h_unit="hartree",
    length_unit="bohr",
    spin_degeneracy=2,
):
    """Calculate the Lorentzian Kubo optical response.

    The defaults match the native matrices assembled below: H(R) in Hartree,
    lattice/r(R) in Bohr. Tensor outputs are shaped ``(n_omega, 3, 3)``.
    """
    lattice = np.asarray(lattice_matrix, dtype=float)
    mesh = tuple(int(value) for value in kmesh)
    if len(omega) != 2:
        raise ValueError("omega must contain start and end energies")
    if domega <= 0 or eta <= 0 or omega[1] < omega[0]:
        raise ValueError("require domega > 0, eta > 0, and omega[1] >= omega[0]")
    if spin_degeneracy not in (1, 2):
        raise ValueError("spin_degeneracy must be 1 or 2")

    h_name = str(h_unit).lower()
    if h_name not in {"ha", "hartree", "ev", "electronvolt"}:
        raise ValueError("h_unit must be 'hartree' or 'eV'")
    h_scale = HARTREE_TO_EV if h_name in {"ha", "hartree"} else 1.0
    length_name = str(length_unit).lower()
    if length_name not in {"bohr", "au", "a.u.", "angstrom", "ang", "a"}:
        raise ValueError("length_unit must be 'bohr' or 'angstrom'")
    length_scale = (
        BOHR_TO_ANGSTROM if length_name in {"bohr", "au", "a.u."} else 1.0
    )

    keys = sorted(hr_dict)
    hr = {key: np.asarray(hr_dict[key], complex) * h_scale for key in keys}
    sr = {key: np.asarray(sr_dict[key], complex) for key in keys}
    rr = {key: np.asarray(rr_dict[key], complex) * length_scale for key in keys}
    lattice_angstrom = lattice * length_scale
    r_cart_angstrom = np.asarray(keys, dtype=float) @ lattice_angstrom

    energy_ev = np.arange(
        float(omega[0]), float(omega[1]) + 0.5 * float(domega), float(domega)
    )
    conductivity = np.zeros((3, 3, len(energy_ev)), dtype=np.complex128)
    dielectric = np.zeros_like(conductivity)
    axes = [np.arange(n, dtype=float) / n for n in mesh]
    kpoints = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    max_transition = float(omega[0]) + int(len(energy_ev) * 1.5) * float(domega)

    for ik, kpoint in enumerate(kpoints):
        hk, sk, dh, ds, rk = _matrices_at_k(
            hr, sr, rr, keys, r_cart_angstrom, kpoint
        )
        try:
            eigenvalues, eigenvectors = eigh(hk, sk, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"overlap matrix is not positive definite at k={kpoint.tolist()}"
            ) from exc
        velocity = _velocity_in_band_basis(eigenvalues, eigenvectors, dh, ds, rk)

        occ = slice(0, int(occupied_bands))
        emp = slice(int(occupied_bands), None)
        band_gap = eigenvalues[int(occupied_bands)] - eigenvalues[int(occupied_bands) - 1]
        if band_gap <= 1.0e-10:
            raise ValueError(
                "optical calculation requires an insulator/semiconductor; "
                f"band gap is {band_gap:.6g} eV at k={kpoint.tolist()}"
            )
        delta_all = eigenvalues[occ, None] - eigenvalues[None, emp]
        active = np.abs(delta_all) <= max_transition
        if np.any(active):
            delta = delta_all[active]
            de = delta[:, None]
            photon = energy_ev[None, :]
            denominator1 = (de + photon) ** 2 + eta**2
            denominator2 = (-de + photon) ** 2 + eta**2
            fac1 = eta / de / denominator1
            fac2 = (de + photon) / de / denominator1
            fac3 = eta / de / denominator2
            fac4 = (-de + photon) / de / denominator2
            for alpha in range(3):
                v_alpha = velocity[alpha, occ, emp][active]
                for beta in range(3):
                    vv = v_alpha * velocity[beta, emp, occ].T[active]
                    conductivity[alpha, beta] += np.sum(
                        -(fac1 + 1j * fac2) * vv[:, None]
                        - (fac3 + 1j * fac4) * vv.conj()[:, None], axis=0
                    )
                    dielectric[alpha, beta] += np.sum(
                        -(1j * fac1 - fac2) * vv[:, None] / -de
                        - (1j * fac3 - fac4) * vv.conj()[:, None] / de, axis=0
                    )
    resistance_quantum = 25812.80745
    epsilon0 = 8.854187817e-12
    hbar = 1.05457182e-34
    electronvolt = 1.60217662e-19
    volume_angstrom3 = abs(float(np.linalg.det(lattice_angstrom)))
    prefactor = (
        spin_degeneracy * 2.0 * np.pi / resistance_quantum
        / volume_angstrom3 / len(kpoints) * 1.0e10
    )
    conductivity *= prefactor
    dielectric *= prefactor * hbar / epsilon0 / electronvolt
    dielectric[0, 0] += 1.0
    dielectric[1, 1] += 1.0
    dielectric[2, 2] += 1.0

    conductivity = np.moveaxis(conductivity, -1, 0)
    dielectric = np.moveaxis(dielectric, -1, 0)
    diagonal = np.diagonal(dielectric, axis1=1, axis2=2)
    eps_real = diagonal.real.mean(axis=1)
    eps_imag = diagonal.imag.mean(axis=1)
    refractive_term = np.maximum(
        (np.hypot(eps_real, eps_imag) - eps_real) / 2.0, 0.0
    )
    absorption = (
        4.0 * np.pi * energy_ev / 1239.841984 * 1.0e7
        * np.sqrt(refractive_term)
    )
    return OpticalResult(energy_ev, conductivity, dielectric, absorption)


def write_optical_results(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_sigma = result.conductivity.reshape(len(result.energy_ev), 9)
    flat_epsilon = result.dielectric.reshape(len(result.energy_ev), 9)
    header = "omega(eV) " + " ".join(TENSOR_LABELS)
    for name, values in (
        ("optical_conductivity_real_part.dat", flat_sigma.real),
        ("optical_conductivity_imag_part.dat", flat_sigma.imag),
        ("dielectric_function_real_part.dat", flat_epsilon.real),
        ("dielectric_function_imag_part.dat", flat_epsilon.imag),
    ):
        np.savetxt(
            output_dir / name,
            np.column_stack((result.energy_ev, values)),
            header=header,
            comments="# ",
            fmt="%.8e",
        )
    np.savetxt(
        output_dir / "absorption.csv",
        np.column_stack((result.energy_ev, result.absorption_cm1)),
        delimiter=",",
        header="Energy(eV),Absorption(cm^-1)",
        comments="",
    )
    return output_dir

def main(argv=None):
    """Assemble matrices from a HamGNN prediction and graph, then calculate optics."""
    import argparse

    from pymatgen.core.periodic_table import Element

    if __package__:
        from .data import SPECIES_DATA
    else:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
        from hamgnn_optics.data import SPECIES_DATA

    parser = argparse.ArgumentParser(description="Calculate optics from HamGNN prediction and graph data")
    parser.add_argument("--config", required=True, help="Standalone optics YAML configuration")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    prediction_file = Path(config["prediction_file"]).expanduser()
    graph_dir = Path(config["graph_dir"]).expanduser()
    output = Path(config.get("output", "results/optics")).expanduser()
    if not prediction_file.is_absolute():
        prediction_file = base / prediction_file
    if not graph_dir.is_absolute():
        graph_dir = base / graph_dir
    if not output.is_absolute():
        output = base / output
    prediction_file = prediction_file.resolve()
    graph_dir = graph_dir.resolve()
    output = output.resolve()
    hamgnn = {"nao_max": 26, "orbital_indices": {}, **config.get("hamgnn", {})}
    optics = {
        "kmesh": [20, 20, 20], "energy_range_ev": [0.0, 5.0],
        "energy_step_ev": 0.05, "broadening_ev": 0.01,
        "occupied_bands": None, **config.get("optics", {}),
    }

    with np.load(graph_dir / "graph_arrays.npz", allow_pickle=False) as archive:
        graph = {key: archive[key].copy() for key in archive.files}
    prediction = np.load(prediction_file)
    occupied = optics["occupied_bands"]
    if occupied is None:
        symbols = [Element.from_Z(int(number)).symbol for number in graph["z"]]
        occupied = sum(SPECIES_DATA[symbol]["valence_electrons"] for symbol in symbols) // 2
    matrices = assemble_matrices(graph, prediction, hamgnn, occupied)
    output.mkdir(parents=True, exist_ok=True)
    result = calculate_optical_properties(
        matrices.hr, matrices.sr, matrices.rr, matrices.lattice, matrices.occupied_bands,
        kmesh=optics["kmesh"], omega=optics["energy_range_ev"],
        domega=optics["energy_step_ev"], eta=optics["broadening_ev"],
        h_unit=matrices.h_unit, length_unit=matrices.length_unit,
        spin_degeneracy=matrices.spin_degeneracy,
    )
    write_optical_results(result, output)
    print(f"Optical results: {output}")
    return output


if __name__ == "__main__":
    main()
