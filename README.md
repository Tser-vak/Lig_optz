# Ligand 3D Generation & Force Field Optimization... 🧪Needs some fixies🧪

A robust, high-throughput Python pipeline for converting 2D/3D ligand structures (SDF) into energy-optimized 3D geometries (PDB). 

This script is specifically designed for large-scale virtual screening preparations. It uses **RDKit** for cheminformatics and features a robust, timeout-protected multiprocessing engine using **Pebble** to ensure the pipeline never silently hangs on "pathological" molecules.

## ✨ Key Features

* **Automated Scientific Workflow:** Handles sanitization, hydrogen addition, ETKDG 3D embedding, and Force Field minimization automatically.
* **Smart Force Field Fallbacks:** Defaults to the highly accurate MMFF94 (Merck Molecular Force Field) for small organics, but seamlessly falls back to UFF (Universal Force Field) if unusual elements are encountered.
* **Timeout Protection:** Prevents the dreaded "silent freeze." If a highly constrained molecule causes the math minimizer to enter an infinite loop, the script forcefully terminates that specific worker, logs the failure, and continues the batch.
* **Hardware-Aware Parallelism:** Automatically detects your system hardware and defaults to using `(Total CPU Cores) - 3` to keep your machine responsive while processing thousands of compounds.

## 🛠️ Installation

You will need an environment with **RDKit** installed. It is highly recommended to use `conda` or `mamba` to install RDKit, followed by `pip` for the process management and progress bar libraries.

```bash
# 1. Create a new conda environment with RDKit
conda create -c conda-forge -n ligand_prep rdkit python=3.10
conda activate ligand_prep

# 2. Install the multiprocessing and progress bar libraries
pip install pebble tqdm
```

## 🚀 Usage

Place your input `.sdf` files in a folder (default is `./ligands`) and run the script.

### Basic Run

Will process all SDFs in the `ligands/` folder and output optimized PDBs to `optimized_ligands/`.

```bash
python Optimizing_ligands.py
```
## ⚙️ Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | Path | `ligands/` | Directory containing your input `.sdf` files. |
| `--output` | Path | `optimized_ligands/` | Directory where optimized `.pdb` files will be saved. |
| `--workers` | Int | All Cores - 3 | Number of parallel CPU processes to run. Set to `1` for sequential debugging. |
| `--timeout` | Int | `60` | Maximum seconds allowed per molecule before forcefully terminating the worker. |
| `--ff` | String | `MMFF94` | Preferred force field. Choices are `MMFF94` or `UFF`. |
| `--iters` | Int | `2000` | Maximum optimization iterations for the force field minimizer. |
| `--attempts` | Int | `50` | Maximum retry attempts for ETKDG 3D embedding. |
| `--no-Hs` | Flag | `False` | Add this flag to skip adding explicit hydrogens (Not recommended for FF minimization). |
| `--log` | Path | `ligand_pipeline.log` | Path to the main pipeline log file. |
| `--failed` | Path | `failed_ligands.log` | Path to the log file recording molecules that failed or timed out. |

## 🧪 The Scientific Workflow Under the Hood

1. **Molecule Sanitization:** The script reads the SDF and validates standard chemical properties (valency, ring systems, aromaticity).
2. **Protonation:** Adds explicit hydrogens. This is chemically essential because force fields rely on explicit atoms to calculate Van der Waals forces and electrostatics accurately.
3. **3D Embedding (ETKDG):** Generates initial 3D coordinates using the Experimental-Torsion Knowledge Distance Geometry algorithm. It attempts modern ETKDGv3, falls back to v2, and uses random coordinates as a last resort.
4. **Energy Minimization:** Refines the initial geometry using physics-based force fields to find the nearest local energy minimum.

## 📝 Logging

The script outputs a clean `tqdm` progress bar to the console and routes all detailed debug information to `ligand_pipeline.log`. Any molecule that fails embedding, optimization, or hits the timeout limit will be cleanly recorded in `failed_ligands.log` so you can easily review problem compounds later.
