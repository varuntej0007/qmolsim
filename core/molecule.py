"""
molecule.py
SMILES string ? qubit Hamiltonian for VQE
qiskit-nature 0.8.0 + numpy 1.26.4 compatible
"""

import logging
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.units import DistanceUnit

logger = logging.getLogger(__name__)


def smiles_to_geometry(smiles: str) -> str:
    """
    Convert SMILES string to XYZ geometry string using RDKit 3D embedding.
    Returns geometry string compatible with PySCFDriver.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        raise RuntimeError(f"3D embedding failed for: {smiles}")

    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()

    parts = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        parts.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")

    geometry = "; ".join(parts)
    logger.info(f"SMILES '{smiles}' ? {len(parts)} atoms")
    return geometry


def build_qubit_hamiltonian(
    geometry: str,
    basis: str = "sto-3g",
    mapper_type: str = "parity",
    freeze_core: bool = True,
):
    """
    geometry     : XYZ string e.g. 'H 0 0 0; H 0 0 0.735'
    basis        : 'sto-3g' fastest, 'sto3g' also works
    mapper_type  : 'parity' (fewer qubits) or 'jordan_wigner'
    freeze_core  : True reduces qubit count significantly

    Returns: (qubit_op, num_particles, num_spatial_orbitals, problem)
    """
    logger.info(f"Building Hamiltonian | basis={basis} | mapper={mapper_type} | freeze_core={freeze_core}")

    driver = PySCFDriver(
        atom=geometry,
        basis=basis,
        unit=DistanceUnit.ANGSTROM,
    )

    problem = driver.run()

    if freeze_core:
        transformer = FreezeCoreTransformer()
        problem = transformer.transform(problem)

    hamiltonian = problem.hamiltonian.second_q_op()
    num_particles = problem.num_particles
    num_spatial_orbitals = problem.num_spatial_orbitals

    if mapper_type == "parity":
        mapper = ParityMapper(num_particles=num_particles)
    else:
        mapper = JordanWignerMapper()

    qubit_op = mapper.map(hamiltonian)

    logger.info(
        f"Hamiltonian ready | qubits={qubit_op.num_qubits} | "
        f"particles={num_particles} | orbitals={num_spatial_orbitals}"
    )
    return qubit_op, num_particles, num_spatial_orbitals, problem
