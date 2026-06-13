# QMolSim — Quantum-Enhanced Pharmaceutical Intelligence Platform

> Running on Raspberry Pi 5 — Edge quantum computing meets pharmaceutical manufacturing intelligence

## What it does

QMolSim combines quantum chemistry (VQE on IBM Quantum), AI/ML (Graph Neural Networks), and regulatory intelligence to help pharmaceutical companies like MSN Laboratories make faster, better decisions about their API portfolio.

## Platform

### Drug Discovery API (Port 5000)
- VQE quantum ground state energy calculation (IBM ibm_fez)
- GNN binding affinity prediction
- ADMET screening
- On-demand quantum validation with real IBM job IDs

### MSN Manufacturing Intelligence API (Port 5001)
- **API Quality Profiler** — full physicochemical + regulatory DMF readiness
- **Impurity Screener** — ICH Q3B/M7 compliant degradant identification
- **Polymorph Analyzer** — crystal form stability ranking (VQE-powered)
- **Batch CSV Upload** — screen 500 molecules at once
- **Manufacturing Complexity Score** — 0-100 scale with cost/timeline estimates
- **Development Roadmap Engine** — auto-generate ICH-compliant CMC study plans
- **AI Regulatory Copilot** — Claude-powered CMC Q&A
- **Competitive Benchmarking** — percentile rankings vs 20 industry reference APIs
- **PDF Report Generator** — professional DMF-ready reports
- **Executive Dashboard** — portfolio health KPIs and charts

## Quick Start

```bash
# Clone
git clone https://github.com/varuntej0007/qmolsim
cd qmolsim

# Install dependencies
pip install -r requirements.txt

# Run everything
python run_all.py
```

Then open:
- `http://localhost:5001/msn` — MSN Manufacturing Dashboard
- `http://localhost:5001/executive` — Executive Portfolio Dashboard
- `http://localhost:5000` — Drug Discovery Dashboard

## Docker

```bash
docker-compose up
```

## Hardware

- **Edge device**: Raspberry Pi 5 8GB (ARM32)
- **Quantum backend**: IBM Quantum ibm_fez (156 qubits)
- **Local simulation**: Qiskit AerSimulator (fallback)

## Tech Stack

- Quantum: Qiskit 2.3.1, qiskit-nature 0.8.0, PySCF 2.13.1, IBM Quantum Runtime
- AI/ML: Numpy-native GNN (no PyTorch — ARM32 compatible), RDKit
- Backend: Flask 3.x, ReportLab
- Frontend: Vanilla JS, Chart.js

## MSN Labs Use Cases

1. Screen entire API portfolio for regulatory risk in minutes
2. Generate ICH-compliant DMF study plans automatically  
3. Identify genotoxic impurity risks before wet lab investment
4. Rank crystal polymorphs by quantum-computed stability
5. Benchmark manufacturing complexity vs industry averages

---
Built by Varun Tej | Malla Reddy University, Hyderabad | 2026
