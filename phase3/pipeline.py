"""
phase3/pipeline.py
Full QMolSim pipeline — GNN + ADMET fast path, VQE on-demand.
"""

import logging
import os
import math
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    smiles: str
    molecule_name: str
    timestamp: str
    vqe_energy_ha: Optional[float]
    vqe_energy_ev: Optional[float]
    vqe_qubits: int
    vqe_runtime_s: float
    vqe_success: bool
    vqe_ran: bool
    gnn_dg_kcal: Optional[float]
    gnn_qed: Optional[float]
    gnn_verdict: str
    admet_score: Optional[float]
    admet_verdict: str
    lipinski_pass: bool
    veber_pass: bool
    gi_absorption: str
    bbb_likely: bool
    tox_alerts: List[str]
    pains_alerts: int
    mw: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rotatable_bonds: int
    overall_score: float
    overall_verdict: str
    recommendation: str
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quantum_note: str = ""


def safe_float(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _score_and_verdict(gnn_dg, admet_score, gnn_qed):
    score_parts = []
    if gnn_dg is not None:
        dg_norm = max(0.0, min(1.0, (-gnn_dg - 3.0) / 7.0))
        score_parts.append(0.40 * dg_norm)
    if admet_score is not None:
        score_parts.append(0.35 * admet_score)
    if gnn_qed is not None:
        score_parts.append(0.25 * gnn_qed)
    overall = sum(score_parts) if score_parts else 0.0
    if overall >= 0.65:
        verdict = "★★ STRONG CANDIDATE"
        rec = "Advance to wet-lab validation"
    elif overall >= 0.45:
        verdict = "★ CANDIDATE"
        rec = "Optimise scaffold, re-screen"
    elif overall >= 0.25:
        verdict = "◆ MARGINAL"
        rec = "Structural modification needed"
    else:
        verdict = "✗ REJECT"
        rec = "Do not advance"
    return round(overall, 3), verdict, rec


def _build_explanation(result):
    reasons = []
    warnings = []
    dg = result.gnn_dg_kcal
    if dg is not None:
        if dg < -7.0:
            reasons.append(f"Strong predicted binding affinity (ΔG = {dg:.2f} kcal/mol) — exceeds typical drug-target threshold of -7.0")
        elif dg < -6.0:
            reasons.append(f"Good predicted binding affinity (ΔG = {dg:.2f} kcal/mol) — within promising range")
        elif dg < -5.0:
            reasons.append(f"Moderate binding affinity (ΔG = {dg:.2f} kcal/mol) — may need scaffold optimisation")
        else:
            warnings.append(f"Weak predicted binding affinity (ΔG = {dg:.2f} kcal/mol) — unlikely to engage target effectively")
    if result.admet_score is not None:
        if result.admet_score >= 0.85:
            reasons.append(f"Excellent ADMET profile (score {result.admet_score:.3f}) — strong drug-like properties")
        elif result.admet_score >= 0.65:
            reasons.append(f"Acceptable ADMET profile (score {result.admet_score:.3f})")
        else:
            warnings.append(f"Poor ADMET profile (score {result.admet_score:.3f}) — bioavailability concerns")
    if result.lipinski_pass:
        reasons.append("Passes Lipinski Rule of 5 — orally bioavailable candidate")
    else:
        warnings.append(f"Fails Lipinski RO5 (MW={result.mw} Da, LogP={result.logp}) — poor oral bioavailability predicted")
    if result.veber_pass:
        reasons.append("Passes Veber rules — good intestinal permeability predicted")
    else:
        warnings.append("Fails Veber rules (too many rotatable bonds or high TPSA)")
    if result.bbb_likely:
        reasons.append("Blood-brain barrier penetration likely — relevant for CNS targets")
    if result.gi_absorption == "HIGH":
        reasons.append("High GI absorption predicted")
    else:
        warnings.append("Low GI absorption predicted — formulation challenges expected")
    if not result.tox_alerts:
        reasons.append("No structural toxicity alerts detected")
    else:
        for t in result.tox_alerts:
            warnings.append(f"Toxicity alert: {t} — review before advancing")
    if result.pains_alerts == 0:
        reasons.append("No PAINS alerts — unlikely to be a false positive in assays")
    else:
        warnings.append(f"{result.pains_alerts} PAINS alert(s) — risk of pan-assay interference")
    if result.gnn_qed and result.gnn_qed > 0.6:
        reasons.append(f"High drug-likeness score (QED = {result.gnn_qed:.3f})")
    elif result.gnn_qed and result.gnn_qed < 0.4:
        warnings.append(f"Low drug-likeness score (QED = {result.gnn_qed:.3f})")
    if result.vqe_ran and result.vqe_success:
        quantum_note = (
            f"Quantum VQE computed ground state energy: "
            f"{result.vqe_energy_ha:.6f} Ha ({result.vqe_energy_ev:.2f} eV) "
            f"using {result.vqe_qubits} qubits in {result.vqe_runtime_s:.1f}s on IBM Quantum. "
            f"Quantum validation increases confidence in electronic structure prediction."
        )
    elif result.vqe_ran and not result.vqe_success:
        quantum_note = "Quantum VQE attempted but did not converge for this molecule size. GNN prediction stands."
    else:
        quantum_note = "Quantum validation not yet run. Click Run Quantum Validation for ground-state energy analysis using IBM Quantum."
    return reasons, warnings, quantum_note


def run_full_pipeline(
    smiles: str,
    molecule_name: str = "candidate",
    run_vqe: bool = False,
    geometry: str = None,
    use_ibm: bool = False,
    ibm_backend: str = "ibm_brisbane",
    vqe_basis: str = "sto-3g",
    gnn_model_path: str = "phase2/data/model.json",
) -> PipelineResult:

    ts = datetime.now().isoformat()
    logger.info(f"=== PIPELINE START: {molecule_name} ===")

    # Phase 2: GNN
    gnn_dg = None
    gnn_qed = None
    gnn_verdict = "N/A"
    try:
        from phase2.models.gnn import MolGNN
        model = MolGNN.load(gnn_model_path) if os.path.exists(gnn_model_path) else MolGNN(hidden_dim=64)
        pred = model.predict(smiles)
        if not pred.get("error"):
            gnn_dg = pred["dg"]
            gnn_qed = pred["qed"]
            if gnn_dg < -6.0 and gnn_qed > 0.5:
                gnn_verdict = "★ PROMISING"
            elif gnn_dg < -5.0:
                gnn_verdict = "◆ MODERATE"
            else:
                gnn_verdict = "WEAK"
        logger.info(f"Phase 2 done | dg={gnn_dg:.2f}")
    except Exception as e:
        logger.warning(f"Phase 2 failed: {e}")

    # Phase 3: ADMET
    admet_score = None
    admet_verdict = "N/A"
    lipinski = veber = bbb = False
    gi_abs = "N/A"
    tox_alerts = []
    pains = 0
    mw = logp = tpsa = 0.0
    hbd = hba = rot = 0
    try:
        from phase3.admet import calculate_admet
        admet = calculate_admet(smiles)
        admet_score = admet.admet_score
        admet_verdict = admet.admet_verdict
        lipinski = admet.lipinski_pass
        veber = admet.veber_pass
        gi_abs = admet.gi_absorption
        bbb = admet.bbb_likely
        tox_alerts = admet.tox_alerts
        pains = admet.pains_alerts
        mw = admet.mw
        logp = admet.logp
        hbd = admet.hbd
        hba = admet.hba
        tpsa = admet.tpsa
        rot = admet.rotatable_bonds
        logger.info(f"Phase 3 done | admet={admet_score:.3f}")
    except Exception as e:
        logger.warning(f"Phase 3 failed: {e}")

    overall, verdict, rec = _score_and_verdict(gnn_dg, admet_score, gnn_qed)

    # Phase 1: VQE on-demand
    vqe_energy_ha = None
    vqe_energy_ev = None
    vqe_qubits = 0
    vqe_runtime = 0.0
    vqe_success = False
    vqe_ran = False

    if run_vqe and geometry:
        vqe_ran = True
        try:
            if use_ibm:
                from core.ibm_vqe import run_ibm_vqe
                vqe_result = run_ibm_vqe(
                    geometry=geometry,
                    molecule_name=molecule_name,
                    smiles=smiles,
                    backend_name=ibm_backend,
                )
            else:
                from core.vqe_runner import run_vqe as _run_vqe
                vqe_result = _run_vqe(
                    geometry=geometry,
                    molecule_name=molecule_name,
                    smiles=smiles,
                    basis=vqe_basis,
                )
            vqe_energy_ha = safe_float(vqe_result.ground_state_energy_hartree)
            vqe_energy_ev = safe_float(vqe_result.ground_state_energy_ev)
            vqe_qubits = vqe_result.num_qubits
            vqe_runtime = vqe_result.runtime_seconds
            vqe_success = vqe_result.success
            if vqe_success:
                overall = min(1.0, overall + 0.05)
                if overall >= 0.65:
                    verdict = "★★ STRONG CANDIDATE"
                    rec = "Advance to wet-lab validation"
            logger.info(f"Phase 1 done | energy={vqe_energy_ha} Ha")
        except Exception as e:
            logger.warning(f"Phase 1 failed: {e}")

    result = PipelineResult(
        smiles=smiles,
        molecule_name=molecule_name,
        timestamp=ts,
        vqe_energy_ha=vqe_energy_ha,
        vqe_energy_ev=vqe_energy_ev,
        vqe_qubits=vqe_qubits,
        vqe_runtime_s=safe_float(vqe_runtime) or 0.0,
        vqe_success=vqe_success,
        vqe_ran=vqe_ran,
        gnn_dg_kcal=safe_float(gnn_dg),
        gnn_qed=safe_float(gnn_qed),
        gnn_verdict=gnn_verdict,
        admet_score=safe_float(admet_score),
        admet_verdict=admet_verdict,
        lipinski_pass=lipinski,
        veber_pass=veber,
        gi_absorption=gi_abs,
        bbb_likely=bbb,
        tox_alerts=tox_alerts,
        pains_alerts=pains,
        mw=mw, logp=logp, hbd=hbd, hba=hba,
        tpsa=tpsa, rotatable_bonds=rot,
        overall_score=round(overall, 3),
        overall_verdict=verdict,
        recommendation=rec,
    )
    result.reasons, result.warnings, result.quantum_note = _build_explanation(result)
    logger.info(f"=== PIPELINE DONE: {molecule_name} | score={overall:.3f} ===")
    return result
