"""
Methods for computing the symbolic form of the inertial parameter least-squares data
matrix.
"""

import logging
import pickle
import time

from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sympy

from pydrake.all import Evaluate, Expression, MultibodyForces_, from_sympy, to_sympy
from tqdm import tqdm

from robot_payload_id.symbolic import eval_expression_mat
from robot_payload_id.utils import ArmPlantComponents, JointData, SymJointStateVariables


def symbolic_to_numeric_data_matrix(
    state_variables: SymJointStateVariables, joint_data: JointData, W_sym: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts symbolic data matrix to numeric data matrix.

    Args:
        state_variables (SymJointStateVariables): The symbolic joint state variables.
        joint_data (JointData): The joint data to use for substituting values.
        W_sym (np.ndarray): The symbolic data matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the numeric data matrix and
            the numeric joint torques.
    """
    num_joints = joint_data.joint_positions.shape[1]
    num_timesteps = len(joint_data.sample_times_s)
    num_lumped_params = num_joints * 10
    W_data = np.zeros((num_timesteps * num_joints, num_lumped_params))
    tau_data = joint_data.joint_torques.flatten()

    for i in tqdm(range(num_timesteps), desc="Computing W_data from W_sym"):
        sym_to_val = {}
        for j in range(num_joints):
            sym_to_val[state_variables.q[j]] = joint_data.joint_positions[i, j]
            sym_to_val[state_variables.q_dot[j]] = joint_data.joint_velocities[i, j]
            sym_to_val[state_variables.q_ddot[j]] = joint_data.joint_accelerations[i, j]
        W_data[i * num_joints : (i + 1) * num_joints, :] = Evaluate(W_sym, sym_to_val)

    return W_data, tau_data


def make_memo_pickable(memo: dict) -> Dict[str, str]:
    # Preserve sympy to Drake direction
    pickable_memo = {}
    for key, val in memo.items():
        if isinstance(key, sympy.Symbol):
            pickable_memo[key] = str(val)
    return pickable_memo


def pickle_to_file(obj, file_path: Path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def pickle_symbolic_data_matrix(
    W_sym: np.ndarray,
    state_variables: SymJointStateVariables,
    save_dir: Path,
) -> None:
    """Pickles the symbolic data matrix to disk.

    Args:
        W_sym (np.np.ndarray): The symbolic data matrix.
        state_variables (SymJointStateVariables): The symbolic joint state variables.
        save_dir (Path): The directory to save the symbolic data matrix to. The
            directory will be created if it does not exist.
    """
    start_time = time.time()
    save_dir.mkdir(exist_ok=True)
    state_variable_names = SymJointStateVariables(
        q=[var.get_name() for var in state_variables.q],
        q_dot=[var.get_name() for var in state_variables.q_dot],
        q_ddot=[var.get_name() for var in state_variables.q_ddot],
        tau=[var.get_name() for var in state_variables.tau],
    )
    pickle_to_file(state_variable_names, save_dir / "state_variable_names.pkl")
    # NOTE: Could parallelise this if needed
    for i, row in tqdm(enumerate(W_sym), total=len(W_sym), desc="Saving to disk (row)"):
        for j, expression in tqdm(
            enumerate(row), total=len(row), desc="    Saving to disk (column)"
        ):
            memo = {}
            expression_sympy = to_sympy(expression, memo=memo)
            pickle_to_file(expression_sympy, save_dir / f"W_{i}_{j}_sympy.pkl")
            pickle_to_file(make_memo_pickable(memo), save_dir / f"W_{i}_{j}_memo.pkl")
    logging.info(f"Time to save to disk: {timedelta(seconds=time.time() - start_time)}")


def pickle_load(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_symbolic_data_matrix(
    dir_path: Path,
    sym_state_variables: SymJointStateVariables,
    num_joints: int,
    num_params: int,
) -> np.ndarray:
    """Loads the symbolic data matrix from disk. The symbolic data matrix is assumed to
    have been saved using `pickle_symbolic_data_matrix`.

    Args:
        dir_path (Path): The directory containing the symbolic data matrix.
        sym_state_variables (SymJointStateVariables): The symbolic joint state
            variables.
        num_joints (int): The number of joints.
        num_params (int): The number of parameters.

    Returns:
        np.ndarray: The symbolic data matrix.
    """
    name_to_var = {}
    for q, q_dot, q_ddot, tau in zip(
        sym_state_variables.q,
        sym_state_variables.q_dot,
        sym_state_variables.q_ddot,
        sym_state_variables.tau,
        strict=True,
    ):
        name_to_var[q.get_name()] = q
        name_to_var[q_dot.get_name()] = q_dot
        name_to_var[q_ddot.get_name()] = q_ddot
        name_to_var[tau.get_name()] = tau

    W_sym = np.empty((num_joints, num_params), dtype=Expression)
    for i in tqdm(range(num_joints), total=num_joints, desc="Loading W_sym (joints)"):
        for j in tqdm(
            range(num_params),
            total=num_params,
            desc="   Loading W_sym (params)",
            leave=False,
        ):
            memo = pickle_load(dir_path / f"W_{i}_{j}_memo.pkl")
            for key, val in memo.items():
                memo[key] = name_to_var[val]
            expression_sympy = pickle_load(dir_path / f"W_{i}_{j}_sympy.pkl")
            expression_drake = from_sympy(expression_sympy, memo=memo)
            W_sym[i, j] = (
                expression_drake
                if isinstance(expression_drake, Expression)
                else Expression(expression_drake)
            )
    return W_sym


def extract_symbolic_data_matrix_Wensing_trick(
    symbolic_plant_components: ArmPlantComponents,
) -> np.ndarray:
    """
    Wensing's trick for computing W_sym by setting lumped parameters equal to one at a
    time. Requires the inverse dynamics to be computed in terms of the lumped
    parameters.

    Args:
        symbolic_plant_components (ArmPlantComponents): The symbolic plant components.

    Returns:
        np.ndarray: The symbolic data matrix.
    """
    # Compute the symbolic torques
    forces = MultibodyForces_[Expression](symbolic_plant_components.plant)
    symbolic_plant_components.plant.CalcForceElementsContribution(
        symbolic_plant_components.plant_context, forces
    )
    sym_torques = symbolic_plant_components.plant.CalcInverseDynamics(
        symbolic_plant_components.plant_context,
        symbolic_plant_components.state_variables.q_ddot.T,
        forces,
    )

    # Compute the symbolic data matrix
    sym_parameters_arr = np.concatenate(
        [
            params.get_lumped_param_list()
            for params in symbolic_plant_components.parameters
        ]
    )
    W_column_vectors = []
    start_time = time.time()
    for i in tqdm(
        range(len(sym_parameters_arr)),
        total=len(sym_parameters_arr),
        desc="Computing W_sym (parameter loop)",
    ):
        param_values = np.zeros(len(sym_parameters_arr))
        param_values[i] = 1.0
        W_column_vector = []
        expression: Expression
        for expression in tqdm(
            sym_torques, desc="    Computing W_sym (column loop)", leave=False
        ):
            W_column_vector.append(
                expression.EvaluatePartial(dict(zip(sym_parameters_arr, param_values)))
            )
        W_column_vectors.append(W_column_vector)
    W_sym = np.hstack(W_column_vectors)
    logging.info(
        f"Time to compute W_sym: { timedelta(seconds=time.time() - start_time)}"
    )

    return W_sym


def extract_symbolic_data_matrix(
    symbolic_plant_components: ArmPlantComponents,
    simplify: bool = False,
) -> np.ndarray:
    """Uses symbolic differentiation to compute the symbolic data matrix. Requires the
    inverse dynamics to be computed in terms of the lumped parameters.

    Args:
        symbolic_plant_components (ArmPlantComponents): The symbolic plant components.
        simplify (bool, optional): Whether to simplify the symbolic expressions using
            sympy before computing the Jacobian.

    Returns:
        np.ndarray: The symbolic data matrix.
    """
    # Compute the symbolic torques
    forces = MultibodyForces_[Expression](symbolic_plant_components.plant)
    symbolic_plant_components.plant.CalcForceElementsContribution(
        symbolic_plant_components.plant_context, forces
    )
    sym_torques = symbolic_plant_components.plant.CalcInverseDynamics(
        symbolic_plant_components.plant_context,
        symbolic_plant_components.state_variables.q_ddot.T,
        forces,
    )

    # Compute the symbolic data matrix
    # NOTE: Could parallelise this if needed
    sym_parameters_arr = np.concatenate(
        [
            params.get_lumped_param_list()
            for params in symbolic_plant_components.parameters
        ]
    )
    W_sym: List[Expression] = []
    expression: Expression
    start_time = time.time()
    for expression in tqdm(sym_torques, desc="Computing W_sym"):
        if simplify:
            memo = {}
            expression_sympy = to_sympy(expression, memo=memo)
            # NOTE: This simplification step is very slow. We might be able to do better
            # by manually choosing the simplification methods to use (e.g. 'expand'
            # might suffice)
            simplified_expression_sympy = sympy.simplify(expression_sympy)
            simplified_expression: Expression = from_sympy(
                simplified_expression_sympy, memo=memo
            )
            W_sym.append(simplified_expression.Jacobian(sym_parameters_arr))
        else:
            W_sym.append(expression.Jacobian(sym_parameters_arr))
    W_sym = np.vstack(W_sym)
    logging.info(
        f"Time to compute W_sym: {timedelta(seconds=time.time() - start_time)}"
    )

    return W_sym


def reexpress_symbolic_data_matrix(
    W_sym: np.ndarray,
    sym_state_variables: SymJointStateVariables,
    joint_data: JointData,
) -> np.ndarray:
    """Re-expresses the symbolic data matrix using different symbolic variables.

    Args:
        W_sym (np.ndarray): The symbolic data matrix that will be re-expressed.
        sym_state_variables (SymJointStateVariables): The symbolic joint state
            variables contained in `W_sym`.
        joint_data (JointData): The joint data to use for substituting values of type
            `Expression`. Only the joint positions, velocities, and accelerations are
            used.

    Returns:
        np.ndarray: The re-expressed symbolic data matrix.
    """
    num_timesteps, num_joints = joint_data.joint_positions.shape
    W_sym_new = np.empty((num_timesteps, W_sym.shape[1]), dtype=Expression)
    for i in tqdm(
        range(num_timesteps),
        total=num_timesteps,
        desc="Creating data matrix from traj samples (time steps)",
    ):
        sym_to_val = {}
        for j in range(num_joints):
            sym_to_val[sym_state_variables.q[j]] = joint_data.joint_positions[i, j]
            sym_to_val[sym_state_variables.q_dot[j]] = joint_data.joint_velocities[i, j]
            sym_to_val[sym_state_variables.q_ddot[j]] = joint_data.joint_accelerations[
                i, j
            ]

        for m in tqdm(
            range(num_joints),
            total=num_joints,
            desc="    Substituting W_sym (column)",
            leave=False,
        ):
            for n in tqdm(
                range(W_sym.shape[1]),
                total=W_sym.shape[1],
                desc="        Substituting W_sym (row)",
                leave=False,
            ):
                W_sym_new[i * num_joints + m, n] = W_sym[m, n].Substitute(sym_to_val)
    return W_sym_new


def remove_structurally_unidentifiable_columns(
    W_sym: np.ndarray, symbolic_vars: np.ndarray, tolerance: float = 1e-12
) -> np.ndarray:
    """Removes structurally unidentifiable columns from a symbolic data matrix. The
    methods works by evaluating the symbolic expressions with random values and
    performing QR decomposition on the resulting numeric data matrix.

    Args:
        W_sym (np.ndarray): The symbolic data matrix.
        symbolic_vars (np.ndarray): The symbolic variables in `W_sym`.
        tolerance (float, optional): The tolerance for removing unidentifiable columns.

    Returns:
        np.ndarray: The symbolic data matrix with unidentifiable columns removed.
    """
    # Evaluate with random values
    W_numeric = eval_expression_mat(
        W_sym,
        symbolic_vars,
        np.random.uniform(low=1, high=100, size=len(symbolic_vars)),
    )
    _, R = np.linalg.qr(W_numeric)
    identifiable = np.abs(np.diag(R)) > tolerance
    return W_sym[:, identifiable]
