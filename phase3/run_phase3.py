"""
phase3/run_phase3.py
Test Phase 3 pipeline end-to-end without the API.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from dataclasses import asdict
from core.utils import setup_logging
from phase3.pipeline import run_full_pipeline, print_pipeline_result
from phase3.admet import calculate_admet, print_admet

setup_logging()

# Drug candidates to put through the full pipeline
CANDIDATES = [
    {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "name":   "Aspirin",
        "geometry": "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
    },
    {
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "name":   "Caffeine",
    },
    {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "name":   "Paracetamol",
    },
    {
        "smiles": "CC1=CC(=O)c2ccccc2C1=O",
        "name":   "Menadione",
    },
    {
        "smiles": "CC(=O)Nc1nnc(s1)S(=O)(=O)N",
        "name":   "Acetazolamide",
    },
]


def main():
    print("\n" + "="*62)
    print("  QMolSim Phase 3 — Full Pipeline Test")
    print("="*62)

    all_results = []

    for cand in CANDIDATES:
        print(f"\n>>> Processing: {cand['name']}")
        result = run_full_pipeline(
            smiles=cand["smiles"],
            molecule_name=cand["name"],
            run_vqe=False,        # skip VQE for speed in CLI test
            geometry=cand.get("geometry"),
        )
        print_pipeline_result(result)
        all_results.append(asdict(result))

    # Final ranking
    all_results.sort(key=lambda x: x["overall_score"], reverse=True)

    print("\n" + "="*62)
    print("  FINAL RANKING — All Candidates")
    print("="*62)
    print(f"  {'Rank':<5} {'Molecule':<22} {'Score':>7} {'ΔG':>8} {'ADMET':>7} {'Verdict'}")
    print("-"*62)
    for i, r in enumerate(all_results, 1):
        dg_str    = f"{r['gnn_dg_kcal']:+.2f}" if r['gnn_dg_kcal'] else "  N/A"
        admet_str = f"{r['admet_score']:.3f}"  if r['admet_score']  else "  N/A"
        print(
            f"  {i:<5} {r['molecule_name']:<22} "
            f"{r['overall_score']:>7.3f} "
            f"{dg_str:>8} "
            f"{admet_str:>7}  "
            f"{r['overall_verdict']}"
        )

    print("="*62)
    print(f"\n  Top candidate: {all_results[0]['molecule_name']}")
    print(f"  Recommendation: {all_results[0]['recommendation']}\n")

    # Save full report
    os.makedirs("phase3/data", exist_ok=True)
    with open("phase3/data/full_report.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("  Full report → phase3/data/full_report.json")


if __name__ == "__main__":
    main()
