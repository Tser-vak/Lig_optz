"""
Ligand 3D Generation and Force Field Optimization Pipeline
----------------------------------------------------------
This script automates the conversion of 2D/3D ligand structures (SDF) 
into optimized 3D geometries (PDB). 

Scientific Workflow:
1.  Molecule Sanitization: Validates chemical structures (valency, aromaticity).
2.  Hydrogen Addition: Essential for accurate force field energy calculations.
3.  3D Embedding: Uses ETKDG (Experimental-Torsion Knowledge Distance Geometry) 
    to generate realistic initial 3D conformations.
4.  Force Field Optimization: Minimizes the molecule's energy using MMFF94 
    (Merck Molecular Force Field) or UFF (Universal Force Field) as a fallback.

Designed for high-throughput screening prep using multiprocessing.
"""


import os
import logging
import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from tqdm import tqdm
from rdkit import Chem, rdBase 
from rdkit.Chem import  rdmolfiles, rdDistGeom, rdForceFieldHelpers
from rdkit.Chem import AllChem

# Silence RDKit warnings to keep the tqdm progress bar clean
rdBase.DisableLog("rdApp.*")
# pyright: reportAttributeAccessIssue=false


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

def setup_logger(log_file: Path) -> logging.Logger:
    """
    Configures a file-only logger for clean console output.
    - File: Records detailed debug information and errors (DEBUG).
    """
    logger = logging.getLogger("LigandPipeline")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    # File handler (keeping all logs here)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Configuration Data Class
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Centralized configuration for the optimization pipeline.
    Using dataclasses makes the parameters easy to pass around and manage.
    """
    input_dir: Path
    output_dir: Path
    log_file: Path = Path("ligand_pipeline.log")
    failed_log: Path = Path("failed_ligands.log")
    num_workers: int = 4          # Number of CPU cores to use
    embed_attempts: int = 50      # How many times to try embedding before giving up
    ff_max_iters: int = 2000      # Max steps for the energy minimizer
    force_field: str = "MMFF94"   # Preferred force field (MMFF94 is best for small organics)
    add_hydrogens: bool = True    # Explicit hydrogens are required for 3D physics


# ---------------------------------------------------------------------------
# Result Container
# ---------------------------------------------------------------------------

@dataclass
class MoleculeResult:
    """Encapsulates the outcome of a single molecule's processing run."""
    filename: str
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    num_atoms: int = 0
    converged: bool = False
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
# Core Molecule Processor
# ---------------------------------------------------------------------------

class MoleculeProcessor:
    """
    The 'Engine' of the pipeline. Handles the transformation from SDF to PDB.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def process(self, sdf_path: Path) -> MoleculeResult:
        """Executes the full optimization sequence for one file."""
        t0 = time.perf_counter()
        result = MoleculeResult(filename=sdf_path.name, success=False)

        try:
            # 1. Load the molecule
            mol = self._read_molecule(sdf_path)
            
            # 2. Prepare (Sanitize and Add Hydrogens)
            mol = self._prepare_molecule(mol)
            
            # 3. Generate initial 3D coordinates (Embedding)
            mol = self._generate_conformer(mol, sdf_path.name)
            
            # 4. Refine geometry using Physics (Force Field)
            converged = self._optimize_geometry(mol, sdf_path.name)

            # 5. Save the final structure
            output_path = self._write_pdb(mol, sdf_path.stem)

            result.success = True
            result.output_path = output_path
            result.num_atoms = mol.GetNumAtoms()
            result.converged = converged

        except _PipelineError as e:
            result.error = str(e)
        except Exception as e:
            result.error = f"Unexpected scientific/system error: {e}"

        result.elapsed = time.perf_counter() - t0
        return result

    # ------------------------------------------------------------------
    # Step-by-Step Implementation
    # ------------------------------------------------------------------

    def _read_molecule(self, sdf_path: Path) -> Chem.Mol:
        """Reads SDF files. removeHs=False keeps existing H's for consistency."""
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol = next((m for m in supplier if m is not None), None)
        if mol is None:
            raise _PipelineError(f"RDKit could not parse {sdf_path.name}")
        return mol

    def _prepare_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Sanitizes the molecule (checks valency) and adds hydrogens.
        Force fields like MMFF94 require all atoms (including H) to calculate 
        electrostatics and Van der Waals forces correctly.
        """
        try:
            Chem.SanitizeMol(mol) # Safety check for valid chemistry
            if self.config.add_hydrogens:
                mol = Chem.AddHs(mol)
            return mol
        except Exception as e:
            raise _PipelineError(f"Preparation failed: {e}")

    def _generate_conformer(self, mol: Chem.Mol, name: str) -> Chem.Mol:
        """
        Generates 3D coordinates using the ETKDG algorithm.
        We use a 'Waterfall' fallback strategy:
        1. ETKDGv3: The modern standard (includes better ring handling).
        2. ETKDGv2: Older but often robust.
        3. Random Coordinates: A last resort for highly strained systems.
        """
        # Stage 1: Modern ETKDG
        params = rdDistGeom.ETKDGv3()
        for seed in range(self.config.embed_attempts):
            params.randomSeed = seed
            if AllChem.EmbedMolecule(mol, params) != -1:
                return mol

        # Stage 2: Fallback to v2
        params_v2 = rdDistGeom.ETKDGv2()
        for seed in range(self.config.embed_attempts):
            params_v2.randomSeed = seed
            if AllChem.EmbedMolecule(mol, params_v2) != -1:
                return mol

        # Stage 3: Random Coords (Desperation mode)
        params_rand = rdDistGeom.ETKDGv3()
        params_rand.useRandomCoords = True
        for seed in range(self.config.embed_attempts):
            params_rand.randomSeed = seed
            if AllChem.EmbedMolecule(mol, params_rand) != -1:
                return mol

        raise _PipelineError(f"3D embedding failed for {name} after multiple attempts.")

    def _optimize_geometry(self, mol: Chem.Mol, name: str) -> bool:
        """
        Minimizes the potential energy of the molecule.
        MMFF94 (Merck Molecular Force Field) is parameterized for small organic drugs.
        UFF (Universal Force Field) covers almost the entire periodic table and 
        serves as our fallback for organometallics or unusual elements.
        """
        if self.config.force_field == "MMFF94":
            converged = self._try_mmff94(mol)
            if converged is not None:
                return converged

        # Fallback to UFF if MMFF94 lacks parameters for this atom type
        result = rdForceFieldHelpers.UFFOptimizeMolecule(mol, maxIters=self.config.ff_max_iters)
        if result == -1:
            raise _PipelineError(f"UFF optimization failed for {name}")
        return result == 0

    def _try_mmff94(self, mol: Chem.Mol) -> Optional[bool]:
        """Helper to run MMFF94. Returns None if molecule can't be parameterized."""
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        if props is None:
            return None # Molecule has atom types not recognized by MMFF94
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=0)
        if ff is None:
            return None
        return ff.Minimize(maxIts=self.config.ff_max_iters) == 0

    def _write_pdb(self, mol: Chem.Mol, stem: str) -> Path:
        """Writes the final 3D coordinates to a PDB file."""
        output_path = self.config.output_dir / f"{stem}_optimized.pdb"
        rdmolfiles.MolToPDBFile(mol, str(output_path))
        return output_path


# ---------------------------------------------------------------------------
# Execution Logic
# ---------------------------------------------------------------------------

class _PipelineError(Exception):
    """Custom exception for controlled pipeline failures."""

class LigandBatchRunner:
    """
    Manages the batch processing of multiple molecules in parallel.
    Parallelism is critical in cheminformatics to handle thousands of compounds.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = setup_logger(config.log_file)

    def run(self) -> None:
        """Entry point for the batch process."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        sdf_files = sorted(self.config.input_dir.glob("*.sdf"))
        
        if not sdf_files:
            self.logger.warning(f"No SDF files found in {self.config.input_dir}")
            return

        self.logger.info(f"Starting pipeline for {len(sdf_files)} molecules...")
        t_start = time.perf_counter()

        # Execute in parallel
        results = self._run_parallel(sdf_files)

        # Final Report
        self._report(results, len(sdf_files), time.perf_counter() - t_start)

    def _run_parallel(self, sdf_files: list[Path]) -> list[MoleculeResult]:
        """Distributes work across CPU cores with a progress bar."""
        results = []
        total = len(sdf_files)
        
        if self.config.num_workers == 1:
            # Sequential mode (easier debugging)
            for path in tqdm(sdf_files, desc="Optimizing Ligands", unit="mol"):
                result = _process_one(path, self.config)
                self._log_result(result)
                results.append(result)
            return results

        # Parallel mode
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {executor.submit(_process_one, p, self.config): p for p in sdf_files}
            
            # tqdm wrap around as_completed for the parallel progress bar
            with tqdm(total=total, desc="Optimizing Ligands (Parallel)", unit="mol") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    self._log_result(result)
                    results.append(result)
                    pbar.update(1)
                    
        return results

    def _log_result(self, result: MoleculeResult) -> None:
        """Logs individual molecule success/failure."""
        if result.success:
            self.logger.info(f"[OK] {result.filename} | converged={result.converged}")
        else:
            self.logger.error(f"[FAIL] {result.filename} | {result.error}")
            with open(self.config.failed_log, "a") as f:
                f.write(f"{result.filename}: {result.error}\n")

    def _report(self, results: list[MoleculeResult], total: int, elapsed: float) -> None:
        """Prints a summary of the entire run."""
        successes = sum(1 for r in results if r.success)
        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline Complete: {successes}/{total} succeeded.")
        self.logger.info(f"Total time: {elapsed:.2f} seconds.")
        self.logger.info("=" * 60)

def _process_one(sdf_path: Path, config: PipelineConfig) -> MoleculeResult:
    """Top-level function required for Windows multiprocessing compatibility."""
    return MoleculeProcessor(config).process(sdf_path)

# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def parse_args() -> PipelineConfig:
    """Parses command line arguments and returns a configuration object."""
    parser = argparse.ArgumentParser(
        description="Scientific Ligand Optimizer: 3D generation and force field minimization."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("test_data"),
        help="Input directory with SDF files"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("new_output"),
        help="Output directory for PDB files"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel worker processes (1 = sequential mode)"
    )
    parser.add_argument(
        "--iters", type=int, default=2000,
        help="Force field minimisation iterations"
    )
    parser.add_argument(
        "--attempts", type=int, default=50,
        help="ETKDG embedding attempts per molecule"
    )
    parser.add_argument(
        "--ff", type=str, default="MMFF94", choices=["MMFF94", "UFF"],
        help="Preferred force field (MMFF94 falls back to UFF automatically)"
    )
    parser.add_argument(
        "--no-Hs", action="store_true",
        help="Skip adding hydrogens before embedding (NOT recommended)"
    )
    parser.add_argument(
        "--log", type=Path, default=Path("ligand_pipeline.log"),
        help="Main log file path"
    )
    parser.add_argument(
        "--failed", type=Path, default=Path("failed_ligands.log"),
        help="Failed molecules log path"
    )

    args = parser.parse_args()

    return PipelineConfig(
        input_dir      = args.input,
        output_dir     = args.output,
        log_file       = args.log,
        failed_log     = args.failed,
        num_workers    = args.workers,
        embed_attempts = args.attempts,
        ff_max_iters   = args.iters,
        force_field    = args.ff,
        add_hydrogens  = not args.no_Hs,
    )

if __name__ == "__main__":
    config = parse_args()
    LigandBatchRunner(config).run()
