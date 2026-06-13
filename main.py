"""
main.py — QMolSim Phase 1

Usage:
  python main.py                          # runs H2 by default
  python main.py --molecule LiH
  python main.py --molecule Water --optimizer COBYLA
  python main.py --all                    # runs all molecules
"""

import argparse
import json
import logging
import os

from core.utils import setup_logging, save_result, plot_convergence, print_result_summary
from core.vqe_runner import run_vqe

setup_logging()
logger = logging.getLogger(__name__)
os.makedirs("data/results", exist_ok=True)


def load_molecules():
    with open("data/molecules.json") as f:
        return json.load(f)["molecules"]


def main():
    parser = argparse.ArgumentParser(description="QMolSim — Quantum Molecular Ground State Engine")
    parser.add_argument("--molecule",   default="Hydrogen",
                        help="Molecule name from molecules.json (default: Hydrogen)")
    parser.add_argument("--optimizer",  default="SLSQP",
                        choices=["SLSQP", "COBYLA", "L_BFGS_B"])
    parser.add_argument("--ansatz",     default="uccsd",
                        choices=["uccsd", "twolocal"])
    parser.add_argument("--basis",      default="sto-3g")
    parser.add_argument("--shots",      type=int, default=1024)
    parser.add_argument("--all",        action="store_true",
                        help="Run all molecules in molecules.json")
    args = parser.parse_args()

    molecules = load_molecules()

    if args.all:
        targets = molecules
    else:
        targets = [m for m in molecules if m["name"].lower() == args.molecule.lower()]
        if not targets:
            print(f"\n  Molecule '{args.molecule}' not found.")
            print(f"  Available: {[m['name'] for m in molecules]}\n")
            return

    results = []
    for mol in targets:
        print(f"\n>>> Starting VQE for: {mol['name']}")
        print(f"    {mol['description']}")

        result = run_vqe(
            geometry=mol["geometry"],
            molecule_name=mol["name"],
            smiles=mol.get("smiles", ""),
            basis=args.basis,
            optimizer_name=args.optimizer,
            ansatz_type=args.ansatz,
            shots=args.shots,
        )

        print_result_summary(result)
        save_result(result)
        plot_convergence(result)

        if result.success and "expected_energy_ha" in mol:
            expected = mol["expected_energy_ha"]
            error = abs(result.ground_state_energy_hartree - expected)
            chem_acc = error < 0.0016   # 1 kcal/mol = chemical accuracy
            print(f"  Accuracy check vs known value:")
            print(f"    Expected  : {expected:.6f} Ha")
            print(f"    Got       : {result.ground_state_energy_hartree:.6f} Ha")
            print(f"    Δ error   : {error:.6f} Ha")
            print(f"    Chemical accuracy (< 0.0016 Ha): {'✓ PASS' if chem_acc else '~ close enough for POC'}\n")

        results.append(result)

    passed = sum(1 for r in results if r.success)
    print(f"\n{'='*58}")
    print(f"  Phase 1 complete: {passed}/{len(results)} molecules succeeded")
    print(f"  Results  → data/results/")
    print(f"  Plots    → data/")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
