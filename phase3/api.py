"""
phase3/api.py — QMolSim REST API v2.0
Fast path: GNN + ADMET
On-demand: VQE (IBM Quantum hardware or local simulator fallback)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import logging
from dataclasses import asdict
from flask import Flask, request, jsonify, render_template

from core.utils import setup_logging
from phase3.admet import calculate_admet
from phase3.pipeline import run_full_pipeline
from phase2.models.gnn import MolGNN

setup_logging(logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

GNN_MODEL = None
GNN_MODEL_PATH = "phase2/data/model.json"

VQE_GEOMETRIES = {
    "[H][H]":  "H 0 0 0; H 0 0 0.735",
    "H":       "H 0 0 0; H 0 0 0.735",
    "O":       "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
    "CCO":     "C -0.748 0 0; C 0.748 0 0; O 1.172 1.21 0; H -1.18 1.02 0; H -1.18 -0.51 0.88; H -1.18 -0.51 -0.88; H 1.18 -1.02 0; H 2.14 1.21 0",
    "C":       "C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629",
    "[LiH]":   "Li 0 0 0; H 0 0 1.6",
}


def clean(d):
    """Recursively replace NaN/Inf with None for safe JSON."""
    if isinstance(d, dict):
        return {k: clean(v) for k, v in d.items()}
    if isinstance(d, list):
        return [clean(v) for v in d]
    if isinstance(d, float) and (math.isnan(d) or math.isinf(d)):
        return None
    return d


def get_model():
    global GNN_MODEL
    if GNN_MODEL is None:
        GNN_MODEL = MolGNN.load(GNN_MODEL_PATH) if os.path.exists(GNN_MODEL_PATH) else MolGNN(hidden_dim=64)
    return GNN_MODEL


@app.route("/", methods=["GET"])
def dashboard():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "QMolSim API",
        "version": "2.0.0",
        "phases": ["GNN", "ADMET", "VQE-on-demand"],
        "ibm_configured": True,
        "model_loaded": os.path.exists(GNN_MODEL_PATH),
    })


@app.route("/admet/<path:smiles>", methods=["GET"])
def admet_only(smiles):
    try:
        return jsonify(clean(asdict(calculate_admet(smiles))))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/gnn/<path:smiles>", methods=["GET"])
def gnn_only(smiles):
    try:
        return jsonify(clean(get_model().predict(smiles)))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/screen", methods=["POST"])
def screen_single():
    data = request.get_json(silent=True)
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing 'smiles'"}), 400
    try:
        result = run_full_pipeline(
            smiles=data["smiles"],
            molecule_name=data.get("name", "candidate"),
            run_vqe=False,
        )
        return jsonify(clean(asdict(result)))
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500


@app.route("/vqe", methods=["POST"])
def run_vqe_endpoint():
    """
    On-demand VQE. Tries IBM Quantum hardware first,
    falls back to local AerSimulator automatically.
    """
    data = request.get_json(silent=True)
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing 'smiles'"}), 400

    smiles   = data["smiles"]
    name     = data.get("name", "candidate")
    use_ibm  = data.get("use_ibm", True)
    geometry = data.get("geometry") or VQE_GEOMETRIES.get(smiles)

    if not geometry:
        return jsonify({
            "error": (
                "No geometry available for this molecule. "
                "VQE supported for: H₂ ([H][H]), Water (O), "
                "Ethanol (CCO), Methane (C), LiH ([LiH])."
            ),
            "supported_smiles": list(VQE_GEOMETRIES.keys()),
            "vqe_ran": True,
            "vqe_success": False,
        }), 400

    try:
        logger.warning(f"VQE requested: {name} ({smiles}) | ibm={use_ibm}")
        result = run_full_pipeline(
            smiles=smiles,
            molecule_name=name,
            run_vqe=True,
            geometry=geometry,
            use_ibm=use_ibm,
        )
        d = clean(asdict(result))
        return jsonify({
            "vqe_success":    d["vqe_success"],
            "vqe_ran":        d["vqe_ran"],
            "vqe_energy_ha":  d["vqe_energy_ha"],
            "vqe_energy_ev":  d["vqe_energy_ev"],
            "vqe_qubits":     d["vqe_qubits"],
            "vqe_runtime_s":  d["vqe_runtime_s"],
            "overall_score":  d["overall_score"],
            "overall_verdict":d["overall_verdict"],
            "quantum_note":   d["quantum_note"],
            "recommendation": d["recommendation"],
            "backend_used":   "IBM Quantum" if use_ibm else "AerSimulator",
        })
    except Exception as e:
        logger.exception(e)
        return jsonify({
            "error": str(e),
            "vqe_ran": True,
            "vqe_success": False,
        }), 500


@app.route("/screen/batch", methods=["POST"])
def screen_batch():
    data = request.get_json(silent=True)
    if not data or "molecules" not in data:
        return jsonify({"error": "Missing 'molecules'"}), 400
    results = []
    for mol in data["molecules"]:
        smiles = mol.get("smiles", "")
        if not smiles:
            continue
        try:
            r = run_full_pipeline(smiles=smiles, molecule_name=mol.get("name", smiles[:20]))
            results.append(clean(asdict(r)))
        except Exception as e:
            results.append({"smiles": smiles, "error": str(e)})
    results.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
    return jsonify({"count": len(results), "results": results,
                    "top_candidate": results[0] if results else None})


@app.route("/screen/compare", methods=["POST"])
def screen_compare():
    panel = [
        {"smiles": "CC(=O)Oc1ccccc1C(=O)O",        "name": "Aspirin"},
        {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "name": "Caffeine"},
        {"smiles": "CC(=O)Nc1ccc(O)cc1",            "name": "Paracetamol"},
        {"smiles": "CC1=CC(=O)c2ccccc2C1=O",        "name": "Menadione"},
        {"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",   "name": "Ibuprofen"},
        {"smiles": "CC(=O)Nc1nnc(s1)S(=O)(=O)N",   "name": "Acetazolamide"},
    ]
    results = [clean(asdict(run_full_pipeline(smiles=m["smiles"], molecule_name=m["name"]))) for m in panel]
    results.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
    return jsonify(results)


if __name__ == "__main__":
    print("\n  QMolSim API v2.0")
    print("  POST /screen        — GNN + ADMET (fast)")
    print("  POST /vqe           — quantum validation (IBM or simulator)")
    print("  POST /screen/batch  — batch screening")
    print("  POST /screen/compare\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
