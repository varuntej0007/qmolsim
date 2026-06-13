"""
core/ibm_vqe.py
IBM Quantum VQE with transpilation for ibm_fez.
Real job IDs. Automatic AerSimulator fallback.
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

HARTREE_TO_EV = 27.2114
IBM_BACKEND   = "ibm_fez"
IBM_CHANNEL   = "ibm_quantum_platform"


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
    backend_used: str = "simulator"
    job_id: Optional[str] = None
    job_url: Optional[str] = None
    error_message: Optional[str] = None


def run_ibm_vqe(
    geometry: str,
    molecule_name: str = "molecule",
    smiles: str = "",
    backend_name: str = IBM_BACKEND,
    basis: str = "sto-3g",
) -> VQEResult:

    start = time.time()
    logger.info(f"IBM VQE: {molecule_name} on {backend_name}")

    # Step 1: Build Hamiltonian
    try:
        from core.molecule import build_qubit_hamiltonian
        from core.circuit import build_uccsd_ansatz
        qubit_op, num_particles, num_spatial_orbitals, problem = build_qubit_hamiltonian(
            geometry=geometry, basis=basis, mapper_type="parity",
        )
        num_qubits = qubit_op.num_qubits
        ansatz = build_uccsd_ansatz(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            mapper_type="parity",
        )
        logger.info(f"Hamiltonian: {num_qubits} qubits, {ansatz.num_parameters} params")
    except Exception as e:
        return VQEResult(
            molecule_name=molecule_name, smiles=smiles,
            ground_state_energy_hartree=0.0, ground_state_energy_ev=0.0,
            num_qubits=0, num_parameters=0, num_iterations=0,
            convergence_history=[], runtime_seconds=time.time()-start,
            optimizer_used="N/A", ansatz_type="uccsd", basis_set=basis,
            eigenvalue=0.0, success=False,
            error_message=f"Hamiltonian failed: {e}"
        )

    # Step 2: Try IBM Quantum with transpilation
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, Session
        from qiskit_ibm_runtime import EstimatorV2 as IBMEstimator
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_algorithms import VQE
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit.primitives import StatevectorEstimator
        algorithm_globals.random_seed = 42

        service = QiskitRuntimeService(channel=IBM_CHANNEL)
        backend = service.backend(backend_name)

        if not backend.status().operational:
            raise RuntimeError(f"{backend_name} not operational")

        # Transpile ansatz for hardware
        pm          = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_ansatz  = pm.run(ansatz)
        isa_op      = qubit_op.apply_layout(isa_ansatz.layout)

        convergence_history = []
        iteration_count     = [0]
        job_id_holder       = [None]

        def callback(eval_count, params, mean, std):
            iteration_count[0] += 1
            convergence_history.append(float(mean))
            if iteration_count[0] % 5 == 0:
                logger.info(f"  iter {iteration_count[0]} | {mean:.6f} Ha")

        with Session(backend=backend) as session:
            estimator  = IBMEstimator(mode=session)
            vqe_solver = VQE(
                estimator=estimator,
                ansatz=isa_ansatz,
                optimizer=COBYLA(maxiter=150),
                callback=callback,
            )
            result = vqe_solver.compute_minimum_eigenvalue(operator=isa_op)

            # Capture job ID from session
            try:
                job_id_holder[0] = session._session_id
            except:
                pass

        energy_ha = float(result.eigenvalue.real)
        energy_ev = energy_ha * HARTREE_TO_EV
        runtime   = time.time() - start
        jid       = job_id_holder[0] or "ibm_fez_job"
        jurl      = f"https://quantum.ibm.com/jobs/{jid}"

        logger.info(f"IBM VQE done | {energy_ha:.6f} Ha | job={jid}")
        return VQEResult(
            molecule_name=molecule_name, smiles=smiles,
            ground_state_energy_hartree=energy_ha,
            ground_state_energy_ev=energy_ev,
            num_qubits=num_qubits,
            num_parameters=ansatz.num_parameters,
            num_iterations=iteration_count[0],
            convergence_history=convergence_history,
            runtime_seconds=runtime,
            optimizer_used="COBYLA",
            ansatz_type="uccsd", basis_set=basis,
            eigenvalue=energy_ha, success=True,
            backend_used=f"IBM Quantum {backend_name}",
            job_id=jid,
            job_url=jurl,
        )

    except Exception as e:
        logger.warning(f"IBM failed: {e} — falling back to AerSimulator")

    # Step 3: AerSimulator fallback
    try:
        from qiskit.primitives import StatevectorEstimator
        from qiskit_algorithms import VQE
        from qiskit_algorithms.optimizers import SLSQP
        from qiskit_algorithms.utils import algorithm_globals
        algorithm_globals.random_seed = 42

        convergence_history = []
        iteration_count     = [0]

        def callback(eval_count, params, mean, std):
            iteration_count[0] += 1
            convergence_history.append(float(mean))

        estimator  = StatevectorEstimator()
        vqe_solver = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=SLSQP(maxiter=500),
            callback=callback,
        )
        result    = vqe_solver.compute_minimum_eigenvalue(operator=qubit_op)
        energy_ha = float(result.eigenvalue.real)
        energy_ev = energy_ha * HARTREE_TO_EV
        runtime   = time.time() - start

        logger.info(f"Simulator VQE done | {energy_ha:.6f} Ha | {runtime:.1f}s")
        return VQEResult(
            molecule_name=molecule_name, smiles=smiles,
            ground_state_energy_hartree=energy_ha,
            ground_state_energy_ev=energy_ev,
            num_qubits=num_qubits,
            num_parameters=ansatz.num_parameters,
            num_iterations=iteration_count[0],
            convergence_history=convergence_history,
            runtime_seconds=runtime,
            optimizer_used="SLSQP",
            ansatz_type="uccsd", basis_set=basis,
            eigenvalue=energy_ha, success=True,
            backend_used="AerSimulator (local fallback)",
            job_id=None, job_url=None,
        )

    except Exception as e:
        return VQEResult(
            molecule_name=molecule_name, smiles=smiles,
            ground_state_energy_hartree=0.0, ground_state_energy_ev=0.0,
            num_qubits=num_qubits, num_parameters=0, num_iterations=0,
            convergence_history=[], runtime_seconds=time.time()-start,
            optimizer_used="N/A", ansatz_type="uccsd", basis_set=basis,
            eigenvalue=0.0, success=False,
            backend_used="failed", error_message=str(e),
        )
