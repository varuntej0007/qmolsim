"""
msn/batch_engine.py
Batch CSV upload engine.
Input:  CSV with columns: smiles, name, category (category optional)
Output: Ranked list with full profile for each molecule
"""

import csv
import io
import logging
import time
from dataclasses import asdict
from typing import List, Dict

logger = logging.getLogger(__name__)


def process_batch_csv(csv_content: str, max_molecules: int = 500) -> Dict:
    """
    Process a CSV string of molecules.
    Returns ranked results with full API profile.
    """
    start = time.time()
    rows  = []

    # Parse CSV
    reader = csv.DictReader(io.StringIO(csv_content))
    headers = [h.lower().strip() for h in (reader.fieldnames or [])]

    # Accept flexible column names
    smiles_col = next((h for h in headers if 'smiles' in h), None)
    name_col   = next((h for h in headers if 'name' in h), None)
    cat_col    = next((h for h in headers if 'cat' in h or 'class' in h or 'type' in h), None)

    if not smiles_col:
        return {"error": "CSV must have a 'smiles' column", "count": 0, "results": []}

    for row in reader:
        smiles = row.get(smiles_col, "").strip()
        if not smiles:
            continue
        rows.append({
            "smiles":   smiles,
            "name":     row.get(name_col, smiles[:15]) if name_col else smiles[:15],
            "category": row.get(cat_col, "Unknown") if cat_col else "Unknown",
        })
        if len(rows) >= max_molecules:
            break

    if not rows:
        return {"error": "No valid SMILES found in CSV", "count": 0, "results": []}

    # Process each molecule
    from msn.msn_pipeline import profile_api, screen_impurities
    results  = []
    errors   = []
    total    = len(rows)

    logger.info(f"Batch processing {total} molecules...")

    for i, mol in enumerate(rows):
        try:
            profile  = profile_api(
                smiles=mol["smiles"],
                name=mol["name"],
                category=mol["category"],
            )
            impurity = screen_impurities(
                parent_smiles=mol["smiles"],
                parent_name=mol["name"],
            )
            d = asdict(profile)
            results.append({
                "rank":           0,
                "name":           mol["name"],
                "smiles":         mol["smiles"],
                "category":       mol["category"],
                "api_score":      d["api_score"],
                "regulatory_readiness": d["regulatory_readiness"],
                "mw":             d["mw"],
                "logp":           d["logp"],
                "aqueous_solubility": d["aqueous_solubility"],
                "lipinski_pass":  d["lipinski_pass"],
                "hydrolysis_risk":d["hydrolysis_risk"],
                "oxidation_risk": d["oxidation_risk"],
                "mutagenicity_risk": d["mutagenicity_risk"],
                "genotox_alerts": d["genotox_alerts"],
                "chiral_centers": d["chiral_centers"],
                "dmf_flags":      d["dmf_flags"],
                "ich_concern":    impurity.ich_q3b_concern,
                "impurity_count": impurity.total_flagged,
                "synthesis_complexity": d["synthesis_complexity"].split("—")[0].strip(),
            })
        except Exception as e:
            errors.append({"name": mol["name"], "smiles": mol["smiles"], "error": str(e)})

    # Sort by api_score descending
    results.sort(key=lambda x: x["api_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    runtime = round(time.time() - start, 2)
    logger.info(f"Batch done: {len(results)} success, {len(errors)} errors in {runtime}s")

    # Summary stats
    scores    = [r["api_score"] for r in results]
    high_risk = [r for r in results if r["genotox_alerts"] or r["mutagenicity_risk"]=="HIGH"]
    reg_ready = [r for r in results if r["api_score"] >= 0.80]
    bcs4      = [r for r in results if r["aqueous_solubility"] == "VERY LOW"]

    return {
        "count":          len(results),
        "errors":         len(errors),
        "runtime_s":      runtime,
        "results":        results,
        "error_list":     errors,
        "summary": {
            "total_processed":  total,
            "success":          len(results),
            "avg_score":        round(sum(scores)/len(scores), 3) if scores else 0,
            "max_score":        round(max(scores), 3) if scores else 0,
            "min_score":        round(min(scores), 3) if scores else 0,
            "high_risk_count":  len(high_risk),
            "reg_ready_count":  len(reg_ready),
            "bcs4_count":       len(bcs4),
            "genotox_count":    sum(1 for r in results if r["genotox_alerts"]),
        },
        "top_5":    results[:5],
        "high_risk":high_risk[:10],
    }


def results_to_csv(results: List[Dict]) -> str:
    """Convert batch results back to CSV for download."""
    if not results:
        return ""
    output = io.StringIO()
    fields = [
        "rank","name","smiles","category","api_score",
        "mw","logp","aqueous_solubility","lipinski_pass",
        "hydrolysis_risk","oxidation_risk","mutagenicity_risk",
        "genotox_alerts","chiral_centers","ich_concern",
        "impurity_count","synthesis_complexity","regulatory_readiness",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for r in results:
        row = dict(r)
        row["genotox_alerts"] = "|".join(r.get("genotox_alerts",[]))
        writer.writerow(row)
    return output.getvalue()
