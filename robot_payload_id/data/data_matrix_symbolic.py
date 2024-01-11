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


def extract_symbolic_data_matrix_Wensing_trick(
    symbolic_plant_components: ArmPlantComponents,
) -> np.ndarray:
    """
    Wensing's trick for computing W_sym by setting lumped parameters equal to one at a
    time. This doesn't work as Drake doesn't simplify the expressions and thus throws a
    division by zero error for terms such as m * hx/m when setting m = 0.
    Simplifying using sympy should be possible but faces the same slowness issues as
    `extract_data_matrix_symbolic`.
    NOTE: This method is currently experimental and should be used with care.
    TODO: Clean up + add documentation.
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
    for i in range(len(sym_parameters_arr)):
        param_values = np.zeros(len(sym_parameters_arr))
        param_values[i] = 1.0
        W_column_vector = []
        expression: Expression
        for expression in sym_torques:
            W_column_vector.append(
                expression.EvaluatePartial(dict(zip(sym_parameters_arr, param_values)))
            )
        W_column_vectors.append(W_column_vector)
    W_sym = np.hstack(W_column_vectors)

    return W_sym


def extract_symbolic_data_matrix(
    symbolic_plant_components: ArmPlantComponents,
    simplify: bool = False,
) -> np.ndarray:
    """Uses symbolic differentiation to compute the symbolic data matrix.

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
            # Cancel simplification should be sufficient
            simplified_expression_sympy = sympy.cancel(expression_sympy)
            simplified_expression: Expression = from_sympy(
                simplified_expression_sympy, memo=memo
            )
            W_sym.append(simplified_expression.Jacobian(sym_parameters_arr))
        else:
            W_sym.append(expression.Jacobian(sym_parameters_arr))
    W_sym = np.vstack(W_sym)
    logging.info(
        f"Time to compute W_sym: { timedelta(seconds=time.time() - start_time)}"
    )

    return W_sym
