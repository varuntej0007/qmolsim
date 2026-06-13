"""
vqe_runner.py
VQE execution — compatible with qiskit-aer 0.17.2 (EstimatorV2 API)
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP, COBYLA, L_BFGS_B
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import StatevectorEstimator

from core.molecule import build_qubit_hamiltonian
from core.circuit import build_uccsd_ansatz, build_twolocal_ansatz

logger = logging.getLogger(__name__)
algorithm_globals.random_seed = 42

HARTREE_TO_EV = 27.2114


@dataclass
class VQEResult:
    molecule_name: str
    smiles: str
    ground_state_energy_hartree: float
    ground_state_energy_ev: float
    num_qubits: int
    num_parameters: int
    num_iterations: int
    convergence_history: List[float]
    runtime_seconds: float
    optimizer_used: str
    ansatz_type: str
    basis_set: str
    eigenvalue: float
    success: bool
    error_message: Optional[str] = None


def run_vqe(
    geometry: str,
    molecule_name: str = "molecule",
    smiles: str = "",
    basis: str = "sto-3g",
    mapper_type: str = "parity",
    optimizer_name: str = "SLSQP",
    ansatz_type: str = "uccsd",
    shots: int = 1024,
    max_iterations: int = 500,
) -> VQEResult:
    convergence_history = []
    start_time = time.time()
    logger.info(f"=== VQE START: {molecule_name} ===")

    try:
        # ── 1. Build Hamiltonian ──────────────────────────────────
        qubit_op, num_particles, num_spatial_orbitals, problem = build_qubit_hamiltonian(
            geometry=geometry,
            basis=basis,
            mapper_type=mapper_type,
        )
        num_qubits = qubit_op.num_qubits
        logger.info(f"Hamiltonian: {num_qubits} qubits")

        # ── 2. Build ansatz ───────────────────────────────────────
        if ansatz_type == "uccsd":
            ansatz = build_uccsd_ansatz(
                num_spatial_orbitals=num_spatial_orbitals,
                num_particles=num_particles,
                mapper_type=mapper_type,
            )
        else:
            ansatz = build_twolocal_ansatz(num_qubits=num_qubits)

        num_params = ansatz.num_parameters

        # ── 3. Optimizer ──────────────────────────────────────────
        optimizers = {
            "SLSQP":    SLSQP(maxiter=max_iterations),
            "COBYLA":   COBYLA(maxiter=max_iterations),
            "L_BFGS_B": L_BFGS_B(maxiter=max_iterations),
        }
        optimizer = optimizers.get(optimizer_name, SLSQP(maxiter=max_iterations))

        # ── 4. Convergence callback ───────────────────────────────
        iteration_count = [0]

        def callback(eval_count, params, mean, std):
            iteration_count[0] += 1
            convergence_history.append(float(mean))
            if iteration_count[0] % 10 == 0:
                logger.info(f"  iter {iteration_count[0]:4d} | energy: {mean:.8f} Ha")

        # ── 5. StatevectorEstimator — exact, no shots needed ──────
        # Works with qiskit-algorithms VQE and qiskit 2.x natively
        estimator = StatevectorEstimator()

        # ── 6. Run VQE ────────────────────────────────────────────
        logger.info(f"Running VQE | optimizer={optimizer_name} | params={num_params}")

        vqe = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            callback=callback,
        )

        result = vqe.compute_minimum_eigenvalue(operator=qubit_op)

        runtime = time.time() - start_time
        energy_ha = float(result.eigenvalue.real)
        energy_ev = energy_ha * HARTREE_TO_EV

        logger.info(f"=== VQE DONE: {molecule_name} ===")
        logger.info(f"  Energy : {energy_ha:.8f} Ha | {energy_ev:.4f} eV")
        logger.info(f"  Iters  : {iteration_count[0]} | Runtime: {runtime:.1f}s")

        return VQEResult(
            molecule_name=molecule_name,
            smiles=smiles,
            ground_state_energy_hartree=energy_ha,
            ground_state_energy_ev=energy_ev,
            num_qubits=num_qubits,
            num_parameters=num_params,
            num_iterations=iteration_count[0],
            convergence_history=convergence_history,
            runtime_seconds=runtime,
            optimizer_used=optimizer_name,
            ansatz_type=ansatz_type,
            basis_set=basis,
            eigenvalue=float(result.eigenvalue.real),
            success=True,
        )

    except Exception as e:
        runtime = time.time() - start_time
        logger.error(f"VQE failed for {molecule_name}: {e}", exc_info=True)
        return VQEResult(
            molecule_name=molecule_name,
            smiles=smiles,
            ground_state_energy_hartree=float("nan"),
            ground_state_energy_ev=float("nan"),
            num_qubits=0,
            num_parameters=0,
            num_iterations=0,
            convergence_history=convergence_history,
            runtime_seconds=runtime,
            optimizer_used=optimizer_name,
            ansatz_type=ansatz_type,
            basis_set=basis,
            eigenvalue=float("nan"),
            success=False,
            error_message=str(e),
        )
