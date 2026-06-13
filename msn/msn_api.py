"""

msn/msn_api.py — QMolSim MSN Labs Edition
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.database import init_db
from auth.auth_routes import auth_bp
from auth.auth import require_auth, current_user, audit_log
from auth.storage import save_analysis, get_analyses, create_project, get_projects
import json
import math
import logging
from dataclasses import asdict
from flask import Flask, request, jsonify, render_template_string, send_file


from core.utils import setup_logging
from msn.msn_pipeline import profile_api, screen_impurities, analyze_polymorphs


MODEL_PATH = "msn/data/ml_models.json"
setup_logging(logging.WARNING)
logger = logging.getLogger(__name__)
app = Flask(__name__)
# Register auth blueprint
app.register_blueprint(auth_bp)

# Init database on startup
init_db()

def clean(d):
    if isinstance(d, dict):  return {k: clean(v) for k, v in d.items()}
    if isinstance(d, list):  return [clean(v) for v in d]
    if isinstance(d, float) and (math.isnan(d) or math.isinf(d)): return None
    return d


def load_portfolio():
    with open("msn/data/msn_molecules.json") as f:
        return json.load(f)

@app.route("/executive", methods=["GET"])
def executive_dashboard():
    with open("msn/templates/executive.html") as f:
        return f.read()

@app.route("/msn", methods=["GET"])
def msn_dashboard():
    with open("msn/templates/msn.html") as f:
        return f.read()

@app.route("/msn/structure/<path:smiles>", methods=["GET"])
def molecule_structure(smiles):
    """Returns base64 SVG image of molecule structure."""
    try:
        from core.mol_image import smiles_to_image_base64
        img = smiles_to_image_base64(smiles, width=400, height=280)
        return jsonify({"image": img, "smiles": smiles})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/msn/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "QMolSim MSN Edition",
        "version": "1.0.0",
        "modules": ["API Profiler", "Impurity Screener", "Polymorph Analyzer", "PDF Reports"],
    })


@app.route("/msn/profile", methods=["POST"])
@require_auth("analyst")
def api_profile():
    data = request.get_json(silent=True)
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing 'smiles'"}), 400
    try:
        report = profile_api(
            smiles=data["smiles"],
            name=data.get("name", "API"),
            category=data.get("category", "Unknown"),
        )
        result = clean(asdict(report))

        # Save to database automatically
        u = current_user()
        if u:
            save_analysis(
                user_id=u["id"],
                smiles=data["smiles"],
                molecule_name=data.get("name","API"),
                analysis_type="api_profile",
                result=result,
                project_id=data.get("project_id"),
            )
            audit_log(u["id"], u["username"], "API_PROFILE",
                      resource="molecules",
                      resource_id=data.get("name","API"),
                      details=f"Score: {result.get('api_score')}")
        return jsonify(result)
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500


@app.route("/msn/impurity", methods=["POST"])
def impurity_screen():
    data = request.get_json(silent=True)
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing 'smiles'"}), 400
    try:
        report = screen_impurities(
            parent_smiles=data["smiles"],
            parent_name=data.get("name", "API"),
            impurity_smiles_list=data.get("impurities"),
        )
        return jsonify(clean(asdict(report)))
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500


@app.route("/msn/polymorph", methods=["POST"])
def polymorph_analyze():
    data = request.get_json(silent=True)
    if not data or "forms" not in data:
        return jsonify({"error": "Missing 'forms'"}), 400
    try:
        result = analyze_polymorphs(
            api_name=data.get("name", "API"),
            forms=data["forms"],
            use_vqe=data.get("use_vqe", False),
        )
        return jsonify(clean(asdict(result)))
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500


@app.route("/msn/portfolio", methods=["GET"])
def full_portfolio():
    portfolio = load_portfolio()
    results = []
    for mol in portfolio["api_portfolio"]:
        try:
            profile  = profile_api(smiles=mol["smiles"], name=mol["name"], category=mol["category"])
            impurity = screen_impurities(parent_smiles=mol["smiles"], parent_name=mol["name"],
                                         impurity_smiles_list=mol.get("known_impurities") or None)
            results.append({
                "name":     mol["name"],
                "category": mol["category"],
                "profile":  clean(asdict(profile)),
                "impurity": clean(asdict(impurity)),
            })
        except Exception as e:
            results.append({"name": mol["name"], "error": str(e)})
    results.sort(key=lambda x: x.get("profile", {}).get("api_score", 0), reverse=True)
    return jsonify({"count": len(results), "portfolio": results})


@app.route("/msn/report", methods=["POST"])
def generate_report():
    """Generate PDF report for any molecule."""
    data = request.get_json(silent=True)
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing 'smiles'"}), 400

    try:
        from reports.pdf_generator import generate_api_report
        from datetime import datetime

        smiles   = data["smiles"]
        name     = data.get("name", "API")
        category = data.get("category", "Unknown")

        # Run all modules
        profile  = profile_api(smiles=smiles, name=name, category=category)
        impurity = screen_impurities(parent_smiles=smiles, parent_name=name)

        # Optional polymorph
        poly_data = None
        if data.get("polymorph_forms"):
            poly_result = analyze_polymorphs(
                api_name=name,
                forms=data["polymorph_forms"],
            )
            poly_data = clean(asdict(poly_result))

        # Generate PDF
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = name.replace(" ", "_").replace("/", "_")
        path      = f"reports/{safe_name}_{ts}.pdf"

        generate_api_report(
            profile_data=clean(asdict(profile)),
            impurity_data=clean(asdict(impurity)),
            polymorph_data=poly_data,
            vqe_data=data.get("vqe_data"),
            output_path=path,
        )

        return send_file(
            os.path.abspath(path),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"QMolSim_{safe_name}_Report.pdf",
        )

    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500


@app.route("/msn/report/portfolio", methods=["GET"])
def portfolio_report():
    """Generate PDF reports for ALL molecules in the MSN portfolio."""
    from reports.pdf_generator import generate_api_report
    from datetime import datetime
    import zipfile

    portfolio = load_portfolio()
    generated = []

    for mol in portfolio["api_portfolio"]:
        try:
            profile  = profile_api(smiles=mol["smiles"], name=mol["name"], category=mol["category"])
            impurity = screen_impurities(parent_smiles=mol["smiles"], parent_name=mol["name"])
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = mol["name"].replace(" ", "_")
            path      = f"reports/{safe_name}_{ts}.pdf"
            generate_api_report(
                profile_data=clean(asdict(profile)),
                impurity_data=clean(asdict(impurity)),
                output_path=path,
            )
            generated.append(path)
            logger.warning(f"Generated: {path}")
        except Exception as e:
            logger.error(f"Failed {mol['name']}: {e}")

    # Zip all PDFs together
    from datetime import datetime
    zip_path = f"reports/MSN_Portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for p in generated:
            zf.write(p, os.path.basename(p))

    return send_file(
        os.path.abspath(zip_path),
        mimetype="application/zip",
        as_attachment=True,
        download_name="QMolSim_MSN_Portfolio_Reports.zip",
    )

@app.route("/msn/batch", methods=["POST"])
def batch_upload():
    """
    POST /msn/batch
    Accepts: multipart/form-data with 'file' (CSV)
    OR: JSON with {"csv": "smiles,name\n..."}
    Returns: ranked results JSON
    """
    from msn.batch_engine import process_batch_csv

    csv_content = None

    # Handle file upload
    if 'file' in request.files:
        f = request.files['file']
        csv_content = f.read().decode('utf-8')
    # Handle JSON with csv string
    elif request.is_json:
        data = request.get_json()
        csv_content = data.get("csv","")
    else:
        return jsonify({"error": "Send CSV file or JSON with 'csv' field"}), 400

    if not csv_content:
        return jsonify({"error": "Empty CSV"}), 400

    try:
        result = process_batch_csv(csv_content)
        return jsonify(clean(result))
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500


@app.route("/msn/batch/download", methods=["POST"])
def batch_download_csv():
    """Download batch results as ranked CSV."""
    from msn.batch_engine import process_batch_csv, results_to_csv
    from flask import Response

    csv_content = None
    if 'file' in request.files:
        csv_content = request.files['file'].read().decode('utf-8')
    elif request.is_json:
        csv_content = request.get_json().get("csv","")

    if not csv_content:
        return jsonify({"error": "Empty CSV"}), 400

    try:
        result = process_batch_csv(csv_content)
        output_csv = results_to_csv(result["results"])
        return Response(
            output_csv,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=QMolSim_Ranked_Results.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/msn/manufacturing", methods=["POST"])
def manufacturing_score():
    from msn.manufacturing_score import calculate_manufacturing_score
    from dataclasses import asdict
    data = request.get_json(silent=True)
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing 'smiles'"}), 400
    try:
        # Always auto-fetch full profile so all factors score correctly
        smiles   = data["smiles"]
        name     = data.get("name", "API")
        profile  = data.get("profile")
        if not profile:
            full = profile_api(smiles=smiles, name=name, category="Unknown")
            profile = asdict(full)
        result = calculate_manufacturing_score(
            smiles=smiles,
            api_name=name,
            profile_data=profile,
        )
        return jsonify(clean(asdict(result)))
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500

@app.route("/msn/roadmap", methods=["POST"])
def development_roadmap():
    """
    POST /msn/roadmap
    Body: {"smiles":"...","name":"Metformin"}
    Returns full CMC development roadmap.
    """
    from msn.roadmap_engine import generate_roadmap
    from msn.manufacturing_score import calculate_manufacturing_score
    from dataclasses import asdict
    data = request.get_json(silent=True)
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing 'smiles'"}), 400
    try:
        smiles  = data["smiles"]
        name    = data.get("name", "API")
        profile = profile_api(smiles=smiles, name=name, category="Unknown")
        pd      = asdict(profile)
        mfg     = calculate_manufacturing_score(smiles=smiles, api_name=name, profile_data=pd)
        roadmap = generate_roadmap(
            api_name=name,
            profile_data=pd,
            manufacturing_score=mfg.total_score,
        )
        return jsonify(clean(asdict(roadmap)))
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500
@app.route("/msn/copilot", methods=["POST"])
def regulatory_copilot():
    """
    POST /msn/copilot
    Body: {
      "message": "Why is Atorvastatin high risk?",
      "history": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}],
      "molecule_profile": {...optional...}
    }
    """
    from msn.ai_copilot import get_copilot_response
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message'"}), 400
    try:
        response = get_copilot_response(
            user_message=data["message"],
            conversation_history=data.get("history", []),
            molecule_profile=data.get("molecule_profile"),
        )
        return jsonify({"response": response, "role": "assistant"})
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500

@app.route("/msn/benchmark", methods=["POST"])
def competitive_benchmark():
    """
    POST /msn/benchmark
    Body: {"smiles":"...","name":"Metformin"}
    Returns competitive benchmarking vs 20 industry APIs.
    """
    from msn.benchmarking import benchmark_api
    from msn.manufacturing_score import calculate_manufacturing_score
    from dataclasses import asdict
    data = request.get_json(silent=True)
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing 'smiles'"}), 400
    try:
        smiles  = data["smiles"]
        name    = data.get("name","API")
        profile = profile_api(smiles=smiles, name=name, category="Unknown")
        pd      = asdict(profile)
        mfg     = calculate_manufacturing_score(smiles=smiles, api_name=name, profile_data=pd)
        result  = benchmark_api(
            api_name=name,
            profile_data=pd,
            mfg_score=mfg.total_score,
        )
        return jsonify(clean(asdict(result)))
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500
@app.route("/msn/analyses", methods=["GET"])
@require_auth("viewer")
def list_analyses():
    u = current_user()
    analyses = get_analyses(
        user_id=u["id"] if u["role"] not in ["admin","manager"] else None,
        analysis_type=request.args.get("type"),
        limit=int(request.args.get("limit", 50)),
    )
    return jsonify({"count": len(analyses), "analyses": analyses})


@app.route("/msn/projects", methods=["GET"])
@require_auth("viewer")
def list_projects():
    u = current_user()
    return jsonify(get_projects(u["id"]))


@app.route("/msn/projects", methods=["POST"])
@require_auth("analyst")
def create_proj():
    data = request.get_json(silent=True) or {}
    if not data.get("name"):
        return jsonify({"error": "Missing 'name'"}), 400
    u = current_user()
    pid = create_project(data["name"], data.get("description",""), u["id"])
    audit_log(u["id"], u["username"], "PROJECT_CREATED",
              resource="projects", resource_id=str(pid),
              details=data["name"])
    return jsonify({"project_id": pid, "name": data["name"]}), 201
@app.route("/login", methods=["GET"])
def login_page():
    with open("msn/templates/login.html") as f:
        return f.read()
@app.route("/msn/ml/predict", methods=["POST"])
@require_auth("analyst")
def ml_predict():
    """
    POST /msn/ml/predict
    ML-based predictions replacing rule-based estimates.
    Body: {"smiles":"...","name":"..."}
    """
    from msn.ml_models import get_models
    data = request.get_json(silent=True)
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing 'smiles'"}), 400
    try:
        models = get_models()
        result = models.predict(data["smiles"])
        u = current_user()
        if u:
            audit_log(u["id"], u["username"], "ML_PREDICT",
                      resource="molecules",
                      resource_id=data.get("name","unknown"),
                      details=f"mfg={result.get('ml_mfg_complexity')} tox={result.get('ml_toxicity_label')}")
        return jsonify(clean(result))
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500


@app.route("/msn/ml/compare", methods=["POST"])
@require_auth("analyst")
def ml_compare():
    """
    Compare ML predictions vs rule-based for same molecule.
    Shows improvement from data-driven approach.
    Body: {"smiles":"...","name":"..."}
    """
    from msn.ml_models import get_models
    data = request.get_json(silent=True)
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing 'smiles'"}), 400
    try:
        smiles = data["smiles"]
        name   = data.get("name","API")

        # Rule-based
        rule_profile = profile_api(smiles=smiles, name=name, category="Unknown")
        rule_mfg     = __import__("msn.manufacturing_score",
                                   fromlist=["calculate_manufacturing_score"]
                       ).calculate_manufacturing_score(
                           smiles=smiles, api_name=name,
                           profile_data=asdict(rule_profile))

        # ML-based
        ml = get_models().predict(smiles)

        return jsonify(clean({
            "smiles": smiles,
            "name":   name,
            "rule_based": {
                "stability":        rule_profile.hydrolysis_risk,
                "solubility":       rule_profile.aqueous_solubility,
                "toxicity":         rule_profile.mutagenicity_risk,
                "mfg_complexity":   rule_mfg.total_score,
                "mfg_band":         rule_mfg.complexity_band,
            },
            "ml_based": {
                "stability":        ml["ml_stability_label"],
                "solubility":       ml["ml_solubility_label"],
                "toxicity":         ml["ml_toxicity_label"],
                "mfg_complexity":   ml["ml_mfg_complexity"],
                "mfg_band":         ml["ml_mfg_band"],
            },
            "ml_scores": {
                "stability_score":   ml["ml_stability"],
                "solubility_score":  ml["ml_solubility"],
                "toxicity_score":    ml["ml_toxicity_risk"],
                "formulation_score": ml["ml_formulation_risk"],
            },
            "model_metrics": ml["model_metrics"],
            "method": "Comparison: Rule-based vs ML (trained on 40 real APIs)",
        }))
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500


@app.route("/msn/ml/status", methods=["GET"])
@require_auth("viewer")
def ml_status():
    from msn.ml_models import get_models
    models = get_models()
    return jsonify({
        "trained": models.trained,
        "metrics": models.metrics,
        "model_path": MODEL_PATH if os.path.exists("msn/data/ml_models.json") else "not saved",
        "training_size": 40,
        "features": ["mw","logp","hbd","hba","tpsa","rotatable_bonds",
                     "chiral_centers","aromatic_rings","lipinski_flags",
                     "interaction_terms"],
        "models": ["stability","solubility","toxicity_risk",
                   "formulation_risk","manufacturing_complexity"],
    })

@app.route("/msn/portfolio/analyze", methods=["POST"])
@require_auth("analyst")
def portfolio_analyze():
    """
    POST /msn/portfolio/analyze
    Full portfolio analysis with ML, risk heatmap, budget forecast.
    Body: {"molecules":[{"smiles":"...","name":"...","category":"..."}], "use_ml":true}
    Or send nothing to use built-in MSN portfolio.
    """
    from msn.portfolio_engine import analyze_portfolio
    data = request.get_json(silent=True) or {}
    molecules = data.get("molecules")

    if not molecules:
        # Use built-in MSN portfolio
        port = load_portfolio()
        molecules = [
            {"smiles":m["smiles"],"name":m["name"],"category":m["category"],"status":"active"}
            for m in port.get("api_portfolio",[])
        ]

    use_ml = data.get("use_ml", True)
    try:
        result  = analyze_portfolio(molecules, use_ml=use_ml)
        d       = clean(asdict(result))
        u = current_user()
        if u:
            audit_log(u["id"], u["username"], "PORTFOLIO_ANALYSIS",
                      details=f"APIs={result.total_apis} health={result.portfolio_health_score}")
        return jsonify(d)
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500


@app.route("/msn/portfolio/heatmap", methods=["GET"])
@require_auth("viewer")
def portfolio_heatmap():
    """Quick risk heatmap data for the executive dashboard."""
    from msn.portfolio_engine import analyze_portfolio
    port = load_portfolio()
    molecules = [
        {"smiles":m["smiles"],"name":m["name"],"category":m["category"]}
        for m in port.get("api_portfolio",[])
    ]
    result = analyze_portfolio(molecules, use_ml=False)
    return jsonify({
        "heatmap":          result.risk_heatmap,
        "health_score":     result.portfolio_health_score,
        "risk_counts":      {"low":result.low_risk,"moderate":result.moderate_risk,"high":result.high_risk},
        "budget_forecast":  result.budget_forecast,
        "workload_forecast":result.workload_forecast,
    })


@app.route("/msn/twin", methods=["POST"])
@require_auth("analyst")
def digital_twin():
    """
    POST /msn/twin
    Digital twin simulation.
    Body: {
      "smiles": "...",
      "name": "Atorvastatin",
      "scenarios": [
        {"name":"HCl Salt + Nanosized", "salt_form":"HCl", "particle_size":"nanosized",
         "packaging":"amber_glass", "storage":"cool_2_8", "polymorph":"form_I"},
        {"name":"Amorphous + Blister", "salt_form":"free_base", "particle_size":"spray_dried",
         "packaging":"blister_alu", "storage":"controlled_25", "polymorph":"amorphous"}
      ]
    }
    """
    from msn.digital_twin import run_digital_twin
    data = request.get_json(silent=True)
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing 'smiles'"}), 400
    if not data.get("scenarios"):
        # Default scenarios if none provided
        data["scenarios"] = [
            {"name":"HCl Salt + Micronized + Blister",
             "salt_form":"HCl","particle_size":"micronized",
             "packaging":"blister_alu","storage":"controlled_25","polymorph":"form_I"},
            {"name":"Amorphous + Nanosized + Amber Glass",
             "salt_form":"free_base","particle_size":"nanosized",
             "packaging":"amber_glass","storage":"cool_2_8","polymorph":"amorphous"},
            {"name":"Co-crystal + Standard + HDPE",
             "salt_form":"co_crystal","particle_size":"standard",
             "packaging":"hdpe_bottle","storage":"room_temp","polymorph":"form_II"},
            {"name":"Mesylate Salt + Spray Dried + Blister",
             "salt_form":"mesylate","particle_size":"spray_dried",
             "packaging":"blister_alu","storage":"controlled_25","polymorph":"form_I"},
        ]
    try:
        result = run_digital_twin(
            smiles=data["smiles"],
            api_name=data.get("name","API"),
            scenarios=data["scenarios"],
        )
        d = clean(asdict(result))
        u = current_user()
        if u:
            audit_log(u["id"], u["username"], "DIGITAL_TWIN",
                      resource="molecules",
                      resource_id=data.get("name","unknown"),
                      details=f"scenarios={len(data['scenarios'])} best={result.best_scenario}")
            save_analysis(u["id"], data["smiles"], data.get("name","API"),
                          "digital_twin", d)
        return jsonify(d)
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500


@app.route("/msn/twin/options", methods=["GET"])
@require_auth("viewer")
def twin_options():
    """Returns all available options for digital twin simulation."""
    from msn.digital_twin import (SALT_FORM_IMPACTS, PARTICLE_SIZE_IMPACTS,
                                   PACKAGING_IMPACTS, STORAGE_IMPACTS,
                                   POLYMORPH_IMPACTS)
    return jsonify({
        "salt_forms":    list(SALT_FORM_IMPACTS.keys()),
        "particle_sizes":list(PARTICLE_SIZE_IMPACTS.keys()),
        "packaging":     list(PACKAGING_IMPACTS.keys()),
        "storage":       list(STORAGE_IMPACTS.keys()),
        "polymorphs":    list(POLYMORPH_IMPACTS.keys()),
    })

if __name__ == "__main__":
    print("\n  QMolSim MSN Edition v1.1")
    print("  GET  http://0.0.0.0:5001/msn")
    print("  POST http://0.0.0.0:5001/msn/profile")
    print("  POST http://0.0.0.0:5001/msn/impurity")
    print("  POST http://0.0.0.0:5001/msn/polymorph")
    print("  GET  http://0.0.0.0:5001/msn/portfolio")
    print("  POST http://0.0.0.0:5001/msn/report       — single molecule PDF")
    print("  GET  http://0.0.0.0:5001/msn/report/portfolio — ALL molecules ZIP\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
