"""
circuit.py
VQE ansatz circuit builder.
UCCSD = chemically motivated, accurate.
TwoLocal = hardware-efficient fallback for larger molecules.
"""

import logging
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
from qiskit.circuit.library import TwoLocal

logger = logging.getLogger(__name__)


def build_uccsd_ansatz(
    num_spatial_orbitals: int,
    num_particles: tuple,
    mapper_type: str = "parity",
):
    """
    UCCSD ansatz with Hartree-Fock initial state.
    Best for molecules up to ~10 qubits on AerSimulator.
    """
    if mapper_type == "parity":
        mapper = ParityMapper(num_particles=num_particles)
    else:
        mapper = JordanWignerMapper()

    hf_state = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
    )

    ansatz = UCCSD(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=hf_state,
    )

    logger.info(
        f"UCCSD ansatz | qubits={ansatz.num_qubits} | "
        f"parameters={ansatz.num_parameters}"
    )
    return ansatz


def build_twolocal_ansatz(num_qubits: int, reps: int = 2):
    """
    Hardware-efficient fallback. Use when molecule > 10 qubits.
    """
    ansatz = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=reps,
    )
    logger.info(
        f"TwoLocal ansatz | qubits={num_qubits} | "
        f"parameters={ansatz.num_parameters}"
    )
    return ansatz
