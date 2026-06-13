"""
utils.py
Logging, result saving, convergence plotting, terminal summary.
"""

import logging
import json
import os
from datetime import datetime
from dataclasses import asdict


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-20s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def save_result(result, output_dir="data/results"):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{output_dir}/{result.molecule_name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    logging.getLogger(__name__).info(f"Result saved → {path}")
    return path


def plot_convergence(result, save_dir="data"):
    """Save convergence plot as PNG. No display needed on Pi."""
    try:
        import matplotlib
        matplotlib.use("Agg")          # no display needed on Pi
        import matplotlib.pyplot as plt

        if not result.convergence_history:
            return

        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result.convergence_history, color="#1D9E75", linewidth=1.5)
        ax.set_xlabel("VQE Iteration")
        ax.set_ylabel("Energy (Hartree)")
        ax.set_title(
            f"VQE Convergence — {result.molecule_name}\n"
            f"Final: {result.ground_state_energy_hartree:.6f} Ha | "
            f"{result.num_qubits} qubits | {result.runtime_seconds:.1f}s"
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = f"{save_dir}/{result.molecule_name}_convergence.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logging.getLogger(__name__).info(f"Plot saved → {path}")

    except Exception as e:
        logging.getLogger(__name__).warning(f"Plot failed (non-critical): {e}")


def print_result_summary(result):
    sep = "=" * 58
    status = "✓ SUCCESS" if result.success else "✗ FAILED"
    print(f"\n{sep}")
    print(f"  QMolSim Phase 1 — VQE Result          {status}")
    print(sep)
    print(f"  Molecule     : {result.molecule_name}")
    print(f"  SMILES       : {result.smiles or 'N/A (geometry mode)'}")
    print(f"  Energy (Ha)  : {result.ground_state_energy_hartree:.8f}")
    print(f"  Energy (eV)  : {result.ground_state_energy_ev:.4f}")
    print(f"  Qubits used  : {result.num_qubits}")
    print(f"  Parameters   : {result.num_parameters}")
    print(f"  Iterations   : {result.num_iterations}")
    print(f"  Runtime      : {result.runtime_seconds:.1f}s")
    print(f"  Optimizer    : {result.optimizer_used}")
    print(f"  Ansatz       : {result.ansatz_type}")
    print(f"  Basis set    : {result.basis_set}")
    if result.error_message:
        print(f"  Error        : {result.error_message}")
    print(f"{sep}\n")
