# Ligand 3D Generation & Force Field Optimization

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

## 🚀 Usage

Place your input `.sdf` files in a folder (default is `./ligands`) and run the script.

### Basic Run

Will process all SDFs in the `ligands/` folder and output optimized PDBs to `optimized_ligands/`.

```bash
python Optimizing_ligands.py
