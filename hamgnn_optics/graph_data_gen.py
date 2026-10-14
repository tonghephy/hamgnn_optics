"""OpenMX reader and HamGNN graph-data generation entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from pymatgen.core import Structure

if __package__:
    from .data import BASIS_DATA
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from hamgnn_optics.data import BASIS_DATA


# Keep this coordinate conversion matched to the OpenMX graph-data convention.
GRAPH_BOHR_TO_ANGSTROM = 0.5291772490000065


def orbital_indices(symbols, settings):
    """Return the established padded orbital ordering, with explicit overrides."""
    registry = BASIS_DATA
    basis = {**registry.get(str(settings["nao_max"]), {}), **settings["orbital_indices"]}
    result = []
    for symbol in symbols:
        if symbol not in basis:
            raise ValueError(f"No orbital map for {symbol}; configure hamgnn.orbital_indices")
        indices = np.asarray(basis[symbol])
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
                or np.any(indices < 0) or np.any(indices >= settings["nao_max"])
                or len(np.unique(indices)) != len(indices) or np.any(np.diff(indices) <= 0)):
            raise ValueError(f"Orbital indices for {symbol} must be unique, increasing integers within nao_max")
        result.append(indices.astype(int))
    return result


def _integer_array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or not np.all(np.isfinite(array)) or np.any(array != np.rint(array)):
        raise ValueError(f"{name} must be an integer array with shape {shape}")
    return array.astype(np.int64)


def build_graph(openmx_data, structure, settings):
    """Build NumPy graph fields from OpenMX reader data, using H0 as target H."""
    symbols = [site.specie.symbol for site in structure]
    local_maps = orbital_indices(symbols, settings)
    nao = settings["nao_max"]
    natoms = len(symbols)
    edge_raw = np.asarray(openmx_data["edge_index"])
    nedges = edge_raw.shape[1]
    edges = _integer_array(edge_raw, (2, nedges), "edge_index")
    shifts = _integer_array(openmx_data["cell_shift"], (nedges, 3), "cell_shift")
    inverse = _integer_array(openmx_data["inv_edge_idx"], (nedges,), "inv_edge_idx")
    pos = np.asarray(openmx_data["pos"], dtype=float)
    neighbor_shift = np.asarray(openmx_data["nbr_shift"], dtype=float).reshape(nedges, 3)
    lattice_bohr = np.asarray(structure.lattice.matrix) / GRAPH_BOHR_TO_ANGSTROM

    def expand(on, off, name, vector=False):
        if len(on) != natoms or len(off) != nedges:
            raise ValueError(f"{name} block count does not match graph topology")
        tail = (3,) if vector else ()
        blocks = np.zeros((natoms + nedges, nao * nao, *tail), dtype=np.float32)
        pairs = [(atom, atom) for atom in range(natoms)] + list(edges.T)
        for index, ((source, target), block) in enumerate(zip(pairs, [*on, *off])):
            rows, cols = local_maps[source], local_maps[target]
            values = np.asarray(block, dtype=float)
            shape = (len(rows) * len(cols), *tail)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} block {index} must have shape {shape} and finite values")
            padded_indices = (rows[:, None] * nao + cols[None, :]).reshape(-1)
            blocks[index, padded_indices] = values
        return blocks

    h0 = expand(openmx_data["Hon"][0], openmx_data["Hoff"][0], "H0")
    overlap = expand(openmx_data["Son"], openmx_data["Soff"], "S")
    position = expand(openmx_data["Ron"], openmx_data["Roff"], "r", vector=True)
    angular = expand(openmx_data["Lon"], openmx_data["Loff"], "L", vector=True)
    return {
        "z": np.array([site.specie.Z for site in structure], dtype=np.int64),
        "cell": lattice_bohr[None].astype(np.float32), "pos": pos.astype(np.float32),
        "total_energy": np.array(0.0, dtype=np.float32),
        "node_counts": np.array([natoms], dtype=np.int64),
        "edge_index": edges, "inv_edge_idx": inverse,
        "nbr_shift": neighbor_shift.astype(np.float32), "cell_shift": shifts,
        "hamiltonian": h0.copy(), "overlap": overlap,
        "Hon": h0[:natoms], "Hoff": h0[natoms:],
        "Hon0": h0[:natoms].copy(), "Hoff0": h0[natoms:].copy(),
        "Son": overlap[:natoms], "Soff": overlap[natoms:],
        "Ron": position[:natoms], "Roff": position[natoms:],
        "Lon": angular[:natoms], "Loff": angular[natoms:],
        "doping_charge": np.array([0.0], dtype=np.float32),
    }


def write_graph(graph, workdir):
    """Write NumPy graph arrays and the PyG archive consumed by HamGNN."""
    import torch
    from torch_geometric.data import Data

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    arrays = workdir / "graph_arrays.npz"
    graph_path = workdir / "graph_data.npz"
    np.savez_compressed(arrays, **graph)
    fields = {key: torch.from_numpy(np.asarray(value).copy()) for key, value in graph.items()}
    fields["total_energy"] = float(np.asarray(fields["total_energy"]))
    np.savez_compressed(graph_path, graph={0: Data(**fields)})
    return graph_path

def read_openmx(settings, scfout, workdir):
    """Read the matched postprocessor's binary format into matrix-block data."""
    workdir = Path(workdir).resolve()
    json_path = workdir / "HS.json"
    subprocess.run([settings["reader"], str(Path(scfout).resolve())], cwd=workdir)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    required = {"pos", "edge_index", "inv_edge_idx", "nbr_shift", "cell_shift",
                "Hon", "Hoff", "Son", "Soff", "Ron", "Roff", "Lon", "Loff"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Reader output lacks {sorted(missing)}; use the matching optical postprocessor/reader")
    return data

def generate_graph(openmx_data, structure, settings, output_dir):
    """Build and serialize one HamGNN graph from parsed OpenMX reader data."""
    graph = build_graph(openmx_data, structure, settings)
    graph_path = write_graph(graph, output_dir)
    return graph, graph_path

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate HamGNN graph files from OpenMX reader output"
    )
    parser.add_argument("--config", required=True, help="Graph-generation YAML configuration")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    structure_path = Path(config["structure_file"]).expanduser()
    output_dir = Path(config.get("output_dir", "graph_data")).expanduser()
    if not structure_path.is_absolute():
        structure_path = base / structure_path
    if not output_dir.is_absolute():
        output_dir = base / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    structure = Structure.from_file(structure_path)
    settings = config.get("hamgnn", {})
    settings.setdefault("nao_max", 26)
    settings.setdefault("orbital_indices", {})
    if config.get("reader_output_file"):
        reader_path = Path(config["reader_output_file"]).expanduser()
        if not reader_path.is_absolute():
            reader_path = base / reader_path
        openmx_data = json.loads(reader_path.read_text(encoding="utf-8"))
    else:
        openmx_settings = config["openmx"]
        scfout_path = Path(openmx_settings["scfout_file"]).expanduser()
        reader = Path(openmx_settings["reader"]).expanduser()
        if not scfout_path.is_absolute():
            scfout_path = base / scfout_path
        if not reader.is_absolute():
            reader = base / reader
        with tempfile.TemporaryDirectory(prefix="openmx-reader-", dir=output_dir) as temp_dir:
            openmx_data = read_openmx(
                {"reader": str(reader)}, scfout_path, temp_dir
            )
    graph, graph_path = generate_graph(openmx_data, structure, settings, output_dir)
    print(f"Generated graph: {graph_path}")
    print(f"Graph arrays: {Path(output_dir).resolve() / 'graph_arrays.npz'}")
    print(f"Atoms: {len(graph['z'])}; edges: {graph['edge_index'].shape[1]}")
    return graph_path


if __name__ == "__main__":
    main()
