"""
phase2/run_phase2.py
QMolSim Phase 2 — GNN Binding Affinity Predictor

Usage:
  python phase2/run_phase2.py               # train + predict
  python phase2/run_phase2.py --predict-only # load saved model + predict
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import argparse
import numpy as np
from datetime import datetime

from core.utils import setup_logging
from phase2.models.graph import smiles_to_graph
from phase2.models.gnn import MolGNN
from phase2.training.trainer import train

setup_logging()
logger = logging.getLogger(__name__)

MODEL_PATH = "phase2/data/model.json"
RESULTS_PATH = "phase2/data/predictions.json"

# Drug candidates to screen after training
# These are real pharmacophore scaffolds relevant to anti-cancer / anti-infective targets


SCREEN_CANDIDATES = [
    {"smiles": "CC1=C(C=CC(=C1)NS(=O)(=O)C2=CC=C(C=C2)N)C",  "name": "Sulfonamide-scaffold"},
    {"smiles": "O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1O",              "name": "Hydroxybenzamide-F"},
    {"smiles": "CC(=O)Nc1ccc(cc1)S(=O)(=O)N",                 "name": "Sulfacetamide"},
    {"smiles": "c1ccc(cc1)C(=O)Nc2ccccc2O",                   "name": "Salicylanilide"},
    {"smiles": "CC(=O)Nc1nnc(s1)S(=O)(=O)N",                  "name": "Acetazolamide"},
    {"smiles": "O=C(O)c1ccc(Cl)cc1",                          "name": "4-Chlorobenzoic"},
    {"smiles": "CC(C)(C)c1ccc(cc1)C(=O)Nc2ccc(cc2)O",         "name": "tBu-HBA"},
    {"smiles": "O=C(Nc1ccccc1)c1ccc(F)cc1",                   "name": "Fluorobenzanilide"},
    {"smiles": "CC1=CC(=O)c2ccccc2C1=O",                      "name": "Menadione"},
    {"smiles": "O=C(O)c1ccc(NC(=O)c2ccccc2)cc1",              "name": "Benzamidobenzoic"},
]




def load_training_data():
    with open("phase2/data/drug_molecules.json") as f:
        raw = json.load(f)["molecules"]

    graphs = []
    failed = 0
    for mol in raw:
        g = smiles_to_graph(mol["smiles"], label_dg=mol["dg"])
        if g is not None:
            graphs.append(g)
        else:
            logger.warning(f"Failed to parse: {mol['name']} ({mol['smiles']})")
            failed += 1

    logger.info(f"Loaded {len(graphs)} molecules ({failed} failed)")
    return graphs


def train_model(graphs):
    rng = np.random.default_rng(42)
    rng.shuffle(graphs)
    split = int(len(graphs) * 0.8)
    train_graphs = graphs[:split]
    val_graphs   = graphs[split:]

    model = MolGNN(hidden_dim=64, seed=42)   # 64 dim for Pi speed
    history = train(
        model=model,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        epochs=80,
        lr=0.01,
        save_path=MODEL_PATH,
    )
    return model, history


def screen_molecules(model):
    print("\n" + "="*65)
    print("  QMolSim Phase 2 — Drug Candidate Screening")
    print("="*65)
    print(f"  {'Molecule':<28} {'ΔG (kcal/mol)':>14} {'QED':>8} {'Verdict':>12}")
    print("-"*65)

    results = []
    for cand in SCREEN_CANDIDATES:
        pred = model.predict(cand["smiles"])
        if pred.get("error"):
            print(f"  {cand['name']:<28} {'ERROR':>14}")
            continue

        dg  = pred["dg"]
        qed = pred["qed"]

        # Verdict logic:
        # Good drug candidate: ΔG < -6 (strong binding) AND QED > 0.5 (drug-like)
        if dg < -6.0 and qed > 0.5:
            verdict = "★ PROMISING"
        elif dg < -5.0:
            verdict = "◆ MODERATE"
        else:
            verdict = "  WEAK"

        print(f"  {cand['name']:<28} {dg:>+14.2f} {qed:>8.3f} {verdict:>12}")
        results.append({**pred, "name": cand["name"], "verdict": verdict.strip()})

    print("="*65)
    print(f"  Promising candidates: {sum(1 for r in results if 'PROMISING' in r.get('verdict',''))}")
    print(f"  Screened: {len(results)} molecules")
    print("="*65 + "\n")

    return results


def save_results(results, history=None):
    os.makedirs("phase2/data", exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "phase": "2 — GNN Binding Affinity",
        "predictions": results,
        "training_history": history,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved → {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip training, load saved model")
    args = parser.parse_args()

    if args.predict_only and os.path.exists(MODEL_PATH):
        logger.info("Loading saved model...")
        model = MolGNN.load(MODEL_PATH)
        history = None
    else:
        logger.info("Loading training data...")
        graphs = load_training_data()

        logger.info("Training GNN...")
        model, history = train_model(graphs)

    results = screen_molecules(model)
    save_results(results, history)

    print(f"  Results → {RESULTS_PATH}")
    print(f"  Model   → {MODEL_PATH}")


if __name__ == "__main__":
    main()
