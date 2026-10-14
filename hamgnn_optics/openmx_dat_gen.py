"""Standalone generator for an OpenMX postprocessing input deck."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from pymatgen.core import Structure

if __package__:
    from .data import SPECIES_DATA
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from hamgnn_optics.data import SPECIES_DATA


def resolve_species_settings(symbols, settings):
    """Add built-in basis, pseudopotential, and valence settings for the .dat file."""
    species = list(dict.fromkeys(symbols))
    missing = [symbol for symbol in species if symbol not in SPECIES_DATA]
    if missing:
        raise ValueError(f"Elements absent from the built-in OpenMX species table: {', '.join(missing)}")
    resolved = dict(settings)
    for field in ("basis", "pseudopotentials", "valence_electrons"):
        resolved[field] = {symbol: SPECIES_DATA[symbol][field] for symbol in species}
    return resolved


def write_input(structure, settings, workdir):
    """Write an OpenMX postprocessing input using the built-in element table."""
    symbols = [site.specie.symbol for site in structure]
    settings = resolve_species_settings(symbols, settings)
    species = list(dict.fromkeys(symbols))
    definitions = []
    valences = {}
    for symbol in species:
        basis = settings["basis"][symbol]
        pseudopotential = settings["pseudopotentials"][symbol]
        valence = settings["valence_electrons"][symbol]
        definitions.append(f"{symbol} {basis} {pseudopotential}")
        valences[symbol] = valence
    data_path = str(settings["data_path"]).replace("\\", "/")
    if any(char.isspace() for char in data_path) or '"' in data_path:
        raise ValueError("This OpenMX input parser requires data_path without whitespace or quotes")
    lines = [
        "# OpenMX basis-matrix preparation for HamGNN",
        "System.CurrrentDirectory ./", "System.Name struct",
        f"DATA.PATH {data_path}", "level.of.stdout 1", "level.of.fileout 1",
        "HS.fileout on", "postprocess 1", "scf.XcType GGA-PBE",
        "scf.SpinPolarization Off", "scf.partialCoreCorrection on",
        "scf.ElectronicTemperature 300.0",
        f"scf.energycutoff {settings['energy_cutoff_ry']}", "scf.maxIter 200",
        "scf.EigenvalueSolver Band", "scf.Kgrid 1 1 1", "MD.Type Nomd",
        f"Species.Number {len(species)}", "<Definition.of.Atomic.Species",
        *definitions, "Definition.of.Atomic.Species>",
        f"Atoms.Number {len(symbols)}", "Atoms.SpeciesAndCoordinates.Unit Ang",
        "<Atoms.SpeciesAndCoordinates",
    ]
    for index, (symbol, position) in enumerate(zip(symbols, structure.cart_coords), 1):
        coordinates = " ".join(f"{item:.12f}" for item in position)
        lines.append(f"{index} {symbol} {coordinates} {valences[symbol]/2:.8f} {valences[symbol]/2:.8f}")
    lines.extend(["Atoms.SpeciesAndCoordinates>", "Atoms.UnitVectors.Unit Ang", "<Atoms.UnitVectors"])
    lines.extend(" ".join(f"{item:.12f}" for item in row) for row in structure.lattice.matrix)
    lines.append("Atoms.UnitVectors>")
    path = Path(workdir) / "struct.dat"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate an OpenMX .dat input for HamGNN graph preparation")
    parser.add_argument("--config", required=True, help="OpenMX input-generation YAML")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    structure_path = Path(config["structure_file"]).expanduser()
    data_path = Path(config["data_path"]).expanduser()
    output_dir = Path(config.get("output_dir", "openmx_input")).expanduser()
    if not structure_path.is_absolute():
        structure_path = base / structure_path
    if not data_path.is_absolute():
        data_path = base / data_path
    if not output_dir.is_absolute():
        output_dir = base / output_dir

    structure = Structure.from_file(structure_path)
    settings = {
        "data_path": data_path,
        "energy_cutoff_ry": config.get("energy_cutoff_ry", 200.0),
    }
    output = write_input(structure, settings, output_dir)
    print(f"OpenMX input: {output}")
    return output


if __name__ == "__main__":
    main()
