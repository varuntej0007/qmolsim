"""
msn/ai_copilot.py
AI Regulatory Copilot — chat interface powered by Claude API.
Answers CMC, regulatory, and manufacturing questions about any API.
Context-aware: knows the molecule's full profile.
"""

import logging
import json
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert CMC (Chemistry, Manufacturing and Controls) regulatory consultant 
with 20+ years of experience in pharmaceutical API development. You specialize in:
- ICH guidelines (Q1-Q11, M7, S2, E6)
- FDA, EMA, and CDSCO regulatory requirements
- API manufacturing and process chemistry
- Stability studies, impurity qualification, polymorphism
- Drug Master File (DMF) preparation
- Indian pharmaceutical manufacturing (MSN Laboratories context)

You are embedded in QMolSim, an AI-powered pharmaceutical intelligence platform.
When answering questions:
1. Be specific and actionable — give real regulatory citations
2. Use plain business language alongside technical terms
3. Quantify timelines and costs in Indian context (INR, months)
4. Flag critical risks clearly
5. Keep answers concise but complete — 3-5 sentences max per point
6. Always mention the relevant ICH guideline

If molecule profile data is provided, use it to give molecule-specific answers.
Never make up data — if unsure, say so and recommend consulting a qualified RA professional."""


def build_context(molecule_profile: Optional[dict] = None) -> str:
    """Build context string from molecule profile for the copilot."""
    if not molecule_profile:
        return ""

    p = molecule_profile
    ctx_parts = [
        f"Current molecule under analysis: {p.get('name','Unknown API')}",
        f"Formula: {p.get('formula','—')} | MW: {p.get('mw','—')} Da | LogP: {p.get('logp','—')}",
        f"Solubility: {p.get('aqueous_solubility','—')} | BCS: {p.get('solubility_class','—')}",
        f"Lipinski: {'PASS' if p.get('lipinski_pass') else 'FAIL'} | Veber: {'PASS' if p.get('veber_pass') else 'FAIL'}",
        f"Hydrolysis risk: {p.get('hydrolysis_risk','—')} | Oxidation risk: {p.get('oxidation_risk','—')}",
        f"Photosensitive: {p.get('photosensitive','—')} | Hygroscopic: {p.get('hygroscopic_risk','—')}",
        f"Chiral centers: {p.get('chiral_centers',0)} | Mutagenicity risk: {p.get('mutagenicity_risk','—')}",
        f"Genotoxic alerts: {', '.join(p.get('genotox_alerts',[])) or 'None'}",
        f"Reactive groups: {', '.join(p.get('reactive_groups',[])) or 'None'}",
        f"API Score: {p.get('api_score','—')} | Regulatory readiness: {p.get('regulatory_readiness','—')}",
        f"DMF flags: {' | '.join(p.get('dmf_flags',[])) or 'None'}",
    ]
    return "\n".join(ctx_parts)


def get_copilot_response(
    user_message: str,
    conversation_history: List[Dict],
    molecule_profile: Optional[dict] = None,
) -> str:
    """
    Get AI regulatory copilot response using Claude API.
    Falls back to rule-based responses if API unavailable.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()

        # Build system prompt with molecule context
        system = SYSTEM_PROMPT
        ctx = build_context(molecule_profile)
        if ctx:
            system += f"\n\nMolecule context for this session:\n{ctx}"

        # Build messages
        messages = []
        for msg in conversation_history[-10:]:  # last 10 for context window
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        messages.append({
            "role": "user",
            "content": user_message
        })

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    except Exception as e:
        logger.warning(f"Claude API failed: {e} — using rule-based fallback")
        return _rule_based_response(user_message, molecule_profile)


def _rule_based_response(query: str, profile: Optional[dict] = None) -> str:
    """
    Rule-based fallback responses for common regulatory questions.
    Works without internet/API access.
    """
    q = query.lower()
    name = profile.get("name", "this API") if profile else "this API"

    if any(x in q for x in ["genotox", "ames", "mutagenic", "ich m7"]):
        alerts = profile.get("genotox_alerts", []) if profile else []
        if alerts:
            return (
                f"**ICH M7 Genotoxic Alert for {name}**\n\n"
                f"Structural alerts detected: {', '.join(alerts)}.\n\n"
                f"Required actions per ICH M7:\n"
                f"1. Bacterial reverse mutation assay (Ames test) — 6-8 weeks, ₹8-15 Lakhs\n"
                f"2. In vitro mammalian cell gene mutation assay — 8-10 weeks, ₹12-20 Lakhs\n"
                f"3. TDI (Threshold of Toxicological Concern) calculation — 1.5 μg/day limit\n"
                f"4. Dedicated manufacturing equipment to prevent cross-contamination\n\n"
                f"This is URGENT — cannot proceed to clinical phase without clearance."
            )
        return (
            f"No genotoxic structural alerts detected for {name}. "
            f"Standard ICH M7 risk assessment still required — document that no alerting structures are present. "
            f"Include in DMF Section 3.2.S.3.2 (Impurities)."
        )

    elif any(x in q for x in ["stability", "shelf life", "ich q1", "degradation"]):
        hydrolysis = profile.get("hydrolysis_risk","LOW") if profile else "LOW"
        return (
            f"**Stability Programme for {name} (ICH Q1A(R2))**\n\n"
            f"Required studies:\n"
            f"1. Forced degradation (acid, base, oxidative, thermal, photolytic) — 6-8 weeks\n"
            f"2. Accelerated: 40°C/75%RH for 6 months — mandatory\n"
            f"3. Long-term: 25°C/60%RH for 12-24 months — establishes shelf life\n"
            f"{'4. Humidity stress testing — HIGH PRIORITY (hydrolysis risk detected)' if hydrolysis=='HIGH' else '4. Standard humidity conditions acceptable'}\n\n"
            f"Recommended shelf life target: 24-36 months. "
            f"Accelerated data supports 2-year shelf life claim if no significant degradation observed."
        )

    elif any(x in q for x in ["solubility", "bcs", "dissolution", "bioavailability"]):
        sol = profile.get("aqueous_solubility","MODERATE") if profile else "MODERATE"
        bcs = profile.get("solubility_class","") if profile else ""
        return (
            f"**Solubility & Bioavailability Assessment for {name}**\n\n"
            f"Classification: {sol} solubility — {bcs}\n\n"
            + (
                "BCS Class IV is the most challenging — both solubility and permeability limited.\n"
                "Recommended strategies:\n"
                "1. Amorphous solid dispersion (hot-melt extrusion or spray drying)\n"
                "2. Nanoparticle formulation — particle size reduction to <200nm\n"
                "3. Co-crystal screening — may improve both solubility and stability\n"
                "4. Bioequivalence study required — cannot rely on dissolution for BE waiver\n"
                "Estimated formulation development: 12-18 months, ₹30-80 Lakhs"
                if sol == "VERY LOW" else
                "Standard dissolution testing required. "
                "BCS biowaiver possible if permeability confirmed. "
                "USP <711> dissolution method development recommended."
            )
        )

    elif any(x in q for x in ["chiral", "stereocenter", "enantiomer", "racemic"]):
        chiral = profile.get("chiral_centers",0) if profile else 0
        return (
            f"**Chirality Assessment for {name}**\n\n"
            f"{chiral} chiral center(s) detected.\n\n"
            f"Regulatory requirements:\n"
            f"1. FDA requires specification for enantiomeric purity (ICH Q6A)\n"
            f"2. Chiral HPLC method development and validation mandatory\n"
            f"3. Evaluate: asymmetric synthesis vs racemate vs chiral resolution\n"
            f"4. Racemisation study under process and storage conditions required\n"
            f"5. If racemic: justify that inactive enantiomer is safe (ICH M4S)\n\n"
            f"Estimated chiral method development: 4-8 weeks, ₹5-10 Lakhs"
        )

    elif any(x in q for x in ["dmf", "drug master file", "submission", "regulatory"]):
        return (
            f"**Drug Master File (DMF) for {name}**\n\n"
            f"Type II DMF required for APIs supplied to US market (FDA 21 CFR 314.420).\n\n"
            f"Required sections:\n"
            f"1. Administrative information and introduction\n"
            f"2. Drug substance characterisation (structure, physicochemical properties)\n"
            f"3. Manufacture (process, controls, validation)\n"
            f"4. Characterisation (impurities, specifications, analytical methods)\n"
            f"5. Stability data (ICH Q1A minimum 12 months long-term)\n\n"
            f"Timeline: 18-30 months from API identification to DMF submission.\n"
            f"Estimated preparation cost: ₹5-15 Lakhs (documentation + regulatory consultant)"
        )

    elif any(x in q for x in ["impurit", "ich q3", "qualification", "threshold"]):
        return (
            f"**Impurity Qualification for {name} (ICH Q3A/Q3B)**\n\n"
            f"Identification thresholds (ICH Q3A for API):\n"
            f"• >0.10% or 1.0 mg/day intake (whichever is lower) — identify\n"
            f"• >0.15% or 1.0 mg/day — qualify\n\n"
            f"Qualification requirements:\n"
            f"1. Structure elucidation (NMR, MS, IR)\n"
            f"2. Biological safety assessment or literature review\n"
            f"3. If no data available: 90-day toxicity study\n"
            f"4. Genotoxicity screen per ICH M7 for all unknown impurities\n\n"
            f"Recommendation: run ICH M7 in silico assessment first — "
            f"eliminates need for wet lab testing if no alerts found."
        )

    elif any(x in q for x in ["qbd", "quality by design", "qtpp", "cqa", "cpp", "design space"]):
        return (
            f"**Quality by Design (QbD) Assessment for {name}**\n\n"
            f"QTPP:\n"
            f"• Assay >99.5%\n"
            f"• Controlled impurity profile\n"
            f"• Stable polymorphic form\n"
            f"• Regulatory compliance\n\n"
            f"CQAs:\n"
            f"• Assay\n"
            f"• Related substances\n"
            f"• Residual solvents\n"
            f"• Water content\n"
            f"• Particle size\n"
            f"• Polymorphic form\n\n"
            f"CPPs:\n"
            f"• Reaction temperature\n"
            f"• pH\n"
            f"• Crystallisation rate\n"
            f"• Drying conditions\n"
            f"• Solvent composition\n\n"
            f"Design space should be established using DOE studies according to ICH Q8/Q9/Q10."
        )

    elif any(x in q for x in ["manufacturing", "500 kg", "commercial scale", "process development", "scale up"]):
        return (
            f"**Commercial Manufacturing Strategy for {name}**\n\n"
            f"Scale-up path:\n"
            f"1. Lab scale\n"
            f"2. Pilot batches\n"
            f"3. Engineering batches\n"
            f"4. Three commercial validation batches\n\n"
            f"Critical Process Parameters:\n"
            f"• Temperature\n"
            f"• Mixing efficiency\n"
            f"• Solvent ratio\n"
            f"• Crystallisation profile\n"
            f"• Drying endpoint\n\n"
            f"Likely impurity pathways:\n"
            f"• Starting material carryover\n"
            f"• Side reactions\n"
            f"• Oxidation\n"
            f"• Residual solvents\n\n"
            f"FDA process validation requires three successful consecutive commercial batches."
        )

    elif any(x in q for x in ["fda inspection", "pai", "pre approval inspection", "inspection readiness"]):
        return (
            f"**FDA Pre-Approval Inspection Readiness for {name}**\n\n"
            f"Inspection focus areas:\n"
            f"1. Data integrity\n"
            f"2. Batch records\n"
            f"3. Deviations\n"
            f"4. CAPA\n"
            f"5. Change control\n"
            f"6. Method validation\n"
            f"7. Process validation\n"
            f"8. Cleaning validation\n"
            f"9. Stability programme\n"
            f"10. Supplier qualification\n\n"
            f"All SOPs, validation reports, audit trails and training records must be inspection ready."
        )


    elif any(x in q for x in ["cost", "budget", "price", "investment", "crore", "lakh"]):
        return (
            f"**Development Cost Estimate for {name}**\n\n"
            f"Typical API development cost breakdown (Indian context):\n"
            f"• Characterisation & analytical methods: ₹20-40 Lakhs\n"
            f"• Stability studies (full ICH package): ₹25-50 Lakhs\n"
            f"• Impurity studies & qualification: ₹15-30 Lakhs\n"
            f"• Process development & validation: ₹40-100 Lakhs\n"
            f"• DMF preparation & submission: ₹5-15 Lakhs\n"
            f"• Regulatory consulting: ₹10-25 Lakhs\n\n"
            f"**Total estimate: ₹115-260 Lakhs (₹1.15-2.6 Crore)**\n"
            f"Timeline: 18-30 months to DMF submission."
        )

    elif any(x in q for x in ["polymorph", "crystal", "solid form", "amorphous"]):
        return (
            f"**Polymorphism Assessment for {name} (ICH Q6A)**\n\n"
            f"Polymorphism screening is mandatory when:\n"
            f"• API is poorly soluble (BCS Class II or IV)\n"
            f"• Multiple solid forms detected during development\n"
            f"• Bioavailability differences between batches observed\n\n"
            f"Recommended screening:\n"
            f"1. Solvent-mediated transformation (10 solvents minimum)\n"
            f"2. Thermal methods (DSC, TGA, hot-stage microscopy)\n"
            f"3. XRPD characterisation of all forms\n"
            f"4. Thermodynamic stability ranking (VQE quantum simulation available in QMolSim)\n\n"
            f"Manufacturing must specify and control the correct polymorph — "
            f"wrong form causes bioavailability failure and regulatory rejection."
        )

    else:
        return (
            f"I can help you with regulatory and CMC questions about {name}. "
            f"Try asking about:\n\n"
            f"• **Stability** — ICH Q1A study design, shelf life\n"
            f"• **Impurities** — ICH Q3A/M7 qualification, genotox\n"
            f"• **Solubility** — BCS classification, formulation strategy\n"
            f"• **Chirality** — stereocenter control, chiral methods\n"
            f"• **DMF** — Drug Master File preparation and submission\n"
            f"• **Polymorphism** — crystal form screening and control\n"
            f"• **Cost & timeline** — development budget estimates\n\n"
            f"Ask a specific question and I will give you actionable guidance."
        )
