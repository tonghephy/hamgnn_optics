# hamgnn_optics

[Chinese documentation](README.zh-CN.md)

A package for calculating linear optical properties from Hamiltonians predicted by [HamGNN](https://github.com/QuantumLab-ZY/HamGNN). It uses an OpenMX atomic orbital basis together with real-space overlap and position-operator matrices to calculate optical conductivity, the dielectric tensor, and isotropic absorption.

The parallel implementation is currently being prepared for public release and will be made available in a future update.

## Contents

1. [Overview](#overview)
2. [Requirements and installation](#requirements-and-installation)
3. [Generate OpenMX input](#generate-openmx-input)
4. [Generate graph files from OpenMX results](#generate-graph-files-from-openmx-results)
5. [Run HamGNN prediction](#run-hamgnn-prediction)
6. [Calculate optical properties](#calculate-optical-properties)
7. [Output files](#output-files)
8. [Input conventions](#input-conventions)

## Overview

The package provides three scripts that can be run independently:

| Script | Configuration | Purpose |
| --- | --- | --- |
| `hamgnn_optics/openmx_dat_gen.py` | `openmx_dat.yaml` | Generate an OpenMX `.dat` input from a structure |
| `hamgnn_optics/graph_data_gen.py` | `graph_data_gen.yaml` | Generate graph files from OpenMX reader output |
| `hamgnn_optics/optics.py` | `optics.yaml` | Calculate optical properties from a predicted Hamiltonian and its graph |

The complete data flow is:

```text
Structure file (e.g., Si.cif)
    ↓ openmx_dat_gen.py
struct.dat
    ↓ Run OpenMX postprocess separately
overlap.scfout
    ↓ graph_data_gen.py calls read_openmx or reads an existing HS.json
graph_data.npz + graph_arrays.npz
    ↓ Run HamGNN separately, using graph_data.npz
prediction_hamiltonian.npy
    ↓ optics.py also reads the corresponding graph_arrays.npz
Optical conductivity, dielectric tensor, and absorption coefficient
```

YAML templates for the three scripts are included in `hamgnn_optics/`: [`openmx_dat_gen.yaml`](hamgnn_optics/openmx_dat_gen.yaml), [`graph_data_gen.yaml`](hamgnn_optics/graph_data_gen.yaml), and [`optics.yaml`](hamgnn_optics/optics.yaml). Replace the `/path/to/...` values before running a script. These YAML files are included when the Python package is built.

**You must install HamGNN yourself, prepare the model configuration and checkpoint, and run prediction separately. Installing `hamgnn_optics` does not install HamGNN.** You must also run OpenMX separately. If you already have a predicted Hamiltonian and its corresponding `graph_arrays.npz`, you can proceed directly to optical calculations.

## Requirements and installation

### Python dependencies

The package requires Python ≥ 3.10 and:

- NumPy ≥ 1.24
- SciPy ≥ 1.10
- PyYAML ≥ 6
- pymatgen ≥ 2023.8

Generating `graph_data.npz` additionally requires **PyTorch and PyTorch Geometric**. Install these yourself using versions compatible with your HamGNN environment. They are not installed automatically by this package.

From the project root containing `pyproject.toml`, run:

```bash
python -m pip install -e .
```

Alternatively, create the basic Conda environment supplied with the project:

```bash
conda env create -f environment.yml
conda activate hamgnn-optics
```

You still need to install PyTorch, PyTorch Geometric, and HamGNN separately. Follow the installation instructions in the [HamGNN repository](https://github.com/QuantumLab-ZY/HamGNN). If you use separate environments, transfer the graph files and prediction results between them.

### OpenMX tools and data

For the input and graph preparation steps, prepare:

- An OpenMX basis and pseudopotential directory, such as `DFT_DATA19`, containing `PAO/` and `VPS/`.
- An `openmx_postprocess` executable that generates the required matrices.
- A `read_openmx` executable compatible with the binary output format.

**The reader must export the position-operator matrices `Ron/Roff` for optical calculations.** The graph generator reads these JSON fields: `pos`, `edge_index`, `inv_edge_idx`, `nbr_shift`, `cell_shift`, `Hon/Hoff`, `Son/Soff`, `Ron/Roff`, and `Lon/Loff`. A reader that exports only the Hamiltonian and overlap matrices cannot supply all the optical input required here.

Follow the compilation instructions for your HamGNN/OpenMX interface and use a matching postprocessor and reader.

## Generate OpenMX input

The examples below use `Si.cif` as an example input; this repository does not include that structure file. Provide your own structure file and run the commands from the project root. Relative paths in a YAML file are resolved relative to **the directory containing that YAML file**. Copy the relevant YAML template from `hamgnn_optics/` to your working directory before editing it.

Edit `openmx_dat.yaml`:

```yaml
structure_file: Si.cif
data_path: /absolute/path/to/DFT_DATA19
output_dir: results/openmx_input
energy_cutoff_ry: 200.0
```

Run:

```bash
python hamgnn_optics/openmx_dat_gen.py --config openmx_dat.yaml
```

This writes `results/openmx_input/struct.dat`. Structure formats are handled by pymatgen, including CIF and POSCAR.

| Parameter | Description | Default |
| --- | --- | --- |
| `structure_file` | Path to the structure file | Required |
| `data_path` | OpenMX basis and pseudopotential directory | Required |
| `output_dir` | Directory for the generated `.dat` file | `openmx_input` |
| `energy_cutoff_ry` | OpenMX grid cutoff energy in Ry | `200.0` |

Element-specific basis names, PBE19 pseudopotential names, and valence electron counts come from `SPECIES_DATA` in `hamgnn_optics/data.py`. For example, Si uses `Si7.0-s2p2d1` and `Si_PBE19`. The files under `data_path` must match these names. The current input generator requires `data_path` to contain no whitespace or quotation marks.

Next, run OpenMX postprocess yourself. The following is a Linux example; replace the executable path:

```bash
cd results/openmx_input
/absolute/path/to/openmx_postprocess struct.dat
cd ../..
```

The remaining examples assume the matrix output is `results/openmx_input/overlap.scfout`. Use the actual output path produced by your OpenMX tools.

## Generate graph files from OpenMX results

### Option 1: Read an existing HS.json

If you have already run the reader to create `HS.json`, edit `graph_data_gen.yaml`:

```yaml
structure_file: Si.cif
reader_output_file: results/openmx_input/HS.json
output_dir: results/graph_data

hamgnn:
  nao_max: 26
  orbital_indices: {}
```

### Option 2: Call the reader on a scfout file

Replace the configuration above with the following, omitting `reader_output_file`:

```yaml
structure_file: Si.cif
output_dir: results/graph_data

openmx:
  reader: /absolute/path/to/read_openmx
  scfout_file: results/openmx_input/overlap.scfout

hamgnn:
  nao_max: 26
  orbital_indices: {}
```

The script runs the reader in a temporary working directory, reads the generated `HS.json`, and writes the graph files. If both `reader_output_file` and `openmx` are specified, it uses `reader_output_file`.

For either option, run:

```bash
python hamgnn_optics/graph_data_gen.py --config graph_data_gen.yaml
```

The output directory contains:

| File | Purpose |
| --- | --- |
| `graph_data.npz` | PyG graph archive used for HamGNN prediction |
| `graph_arrays.npz` | NumPy arrays used to assemble the optical matrices |

`hamgnn.nao_max` is the maximum number of orbitals in the padded model representation and must match the model. `orbital_indices` specifies the indices of each element's actual orbitals within that representation. An empty dictionary selects the built-in `BASIS_DATA` mapping from `data.py`. Use the same settings for graph generation and optical calculations, and ensure that the actual OpenMX basis matches the mapping.

## Run HamGNN prediction

**Run this step in your own HamGNN installation.** Prepare a trained model and its original configuration for the chosen OpenMX basis and orbital ordering.

Keep the network architecture settings consistent with the checkpoint. Update the following fields in the model YAML to set the input paths and prediction mode. This is a partial example, not a complete model configuration:

```yaml
dataset_params:
  graph_data_path: /absolute/path/to/results/graph_data
  batch_size: 1
  split_file: null

profiler_params:
  train_dir: /absolute/path/to/results/hamgnn

setup:
  stage: test
  checkpoint_path: /absolute/path/to/model.ckpt
  load_from_checkpoint: true
  resume: false
  accelerator: cpu
  num_gpus: null
```

The accompanying HamGNN version provides this prediction command:

```bash
HamGNN2.0 --config model_config.yaml
```

For another version, follow its own command and configuration format. After prediction, locate `prediction_hamiltonian.npy` and enter its actual path as `prediction_file` in the optical configuration. The output path may contain a subdirectory such as `version_0`.

The current optical matrix assembly handles **one structure at a time**. The prediction must use the same atom, edge, and orbital ordering as the graph. Run prediction on the generated `graph_data.npz` and retain its paired `graph_arrays.npz`.

## Calculate optical properties

Edit `optics.yaml`, for example:

```yaml
prediction_file: results/hamgnn/version_0/prediction_hamiltonian.npy
graph_dir: results/graph_data
output: results/optics

hamgnn:
  nao_max: 26
  orbital_indices: {}

optics:
  kmesh: [20, 20, 20]
  energy_range_ev: [0.0, 5.0]
  energy_step_ev: 0.05
  broadening_ev: 0.01
  occupied_bands: null
```

Run:

```bash
python hamgnn_optics/optics.py --config optics.yaml
```

`graph_dir` must contain `graph_arrays.npz`. The current script loads the predicted Hamiltonian as a numerical `.npy` array; do not use a `.npz` archive with named fields as `prediction_file`.

| Parameter | Description | Default |
| --- | --- | --- |
| `prediction_file` | Path to the predicted Hamiltonian `.npy` file | Required |
| `graph_dir` | Directory containing `graph_arrays.npz` | Required |
| `output` | Optical results directory | `results/optics` |
| `hamgnn.nao_max` | Maximum number of orbitals in the padded model representation | `26` |
| `hamgnn.orbital_indices` | Element-specific orbital index overrides, matching graph generation | `{}` |
| `optics.kmesh` | Uniform k-point grid dimensions | `[20, 20, 20]` |
| `optics.energy_range_ev` | Start and end photon energies in eV | `[0.0, 5.0]` |
| `optics.energy_step_ev` | Photon energy spacing in eV | `0.05` |
| `optics.broadening_ev` | Lorentzian broadening parameter in eV | `0.01` |
| `optics.occupied_bands` | Number of occupied bands at each k point | `null` |

When `occupied_bands` is `null`, the program uses the integer quotient of the total valence electron count divided by two. A structure with two Si atoms has eight valence electrons, giving `occupied_bands: 4`. Specify the occupied band count explicitly when the system requires a different electron filling.

All three scripts also support `python -m hamgnn_optics.<module_name> --config <configuration_file>`. A small k-point grid can be used for an initial calculation; then assess convergence with respect to the grid and broadening.

## Output files

Results are written to the `output` directory specified in `optics.yaml`:

| File | Contents and units |
| --- | --- |
| `optical_conductivity_real_part.dat` | Real optical conductivity, S/m |
| `optical_conductivity_imag_part.dat` | Imaginary optical conductivity, S/m |
| `dielectric_function_real_part.dat` | Real relative dielectric tensor, dimensionless |
| `dielectric_function_imag_part.dat` | Imaginary relative dielectric tensor, dimensionless |
| `absorption.csv` | Isotropic absorption coefficient, cm⁻¹ |

In each tensor `.dat` file, the first column is photon energy in eV. The remaining nine columns are ordered as:

```text
xx xy xz yx yy yz zx zy zz
```

The two columns of `absorption.csv` are `Energy(eV)` and `Absorption(cm^-1)`. The absorption coefficient is calculated from the average of the three diagonal components of the dielectric tensor.

## Input conventions

- The current calculation is intended for non-spin-polarized insulators or semiconductors without SOC, using a spin degeneracy of two. A positive gap between occupied and unoccupied bands is required at the sampled k points.
- Predicted Hamiltonians are interpreted in Hartree. The graph lattice, atomic positions, and position-operator matrices are interpreted in Bohr. Convert data from other sources to these units before use.
- Use the same structure, atom ordering, lattice orientation, edge ordering, and orbital mapping for the OpenMX results, graphs, and predictions.
- Reader-provided `Ron/Roff` are interpreted relative to the center of the row orbital. During assembly, the optical script adds the atomic-position correction multiplied by the overlap matrix. Do not apply this correction twice.
- The Hamiltonian stored in `graph_data.npz` comes from the OpenMX reader. Optical calculations use the predicted Hamiltonian specified by `prediction_file` and obtain overlap, position, and other graph data from `graph_arrays.npz`.

Project files:

```text
hamgnn_optics/
├── openmx_dat_gen.py   # Generate OpenMX .dat input
├── graph_data_gen.py   # Read OpenMX data and generate graphs
├── optics.py           # Assemble real-space matrices and calculate optics
├── data.py             # Built-in species, basis, and orbital mapping tables
└── __init__.py
```


#   H a m G N N _ o p t i c s  
 