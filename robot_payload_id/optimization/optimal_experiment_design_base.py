import enum

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from numpy import ndarray
from pydrake.all import MakeVectorVariable, ModelInstanceIndex, MultibodyPlant

import wandb

from robot_payload_id.data import (
    extract_symbolic_data_matrix,
    load_symbolic_data_matrix,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.symbolic import create_symbolic_plant
from robot_payload_id.utils import SymJointStateVariables


class CostFunction(enum.Enum):
    CONDITION_NUMBER = "condition_number"
    CONDITION_NUMBER_AND_D_OPTIMALITY = "condition_number_and_d_optimality"
    CONDITION_NUMBER_AND_E_OPTIMALITY = "condition_number_and_e_optimality"

    def __str__(self):
        return self.value


class ExcitationTrajectoryOptimizerBase(ABC):
    """
    The abstract base class for excitation trajectory optimizers.
    """

    def __init__(
        self,
        num_joints: int,
        cost_function: CostFunction,
        plant: MultibodyPlant,
        robot_model_instance_idx: ModelInstanceIndex,
    ):
        """
        Args:
            num_joints (int): The number of joints in the arm.
            cost_function (CostFunction): The cost function to use. NOTE: Currently only
                supports condition number.
            plant (MultibodyPlant): The plant to use for adding constraints.
            robot_model_instance_idx (ModelInstanceIndex): The model instance index of
                the robot. Used for adding constraints.
        """
        self._num_joints = num_joints
        self._num_params = num_joints * 10
        self._cost_function = cost_function
        self._plant = plant
        self._robot_model_instance_idx = robot_model_instance_idx

    @abstractmethod
    def optimize(self) -> Any:
        """Optimizes the trajectory parameters.

        Returns:
            The optimized trajectory parameters.
        """
        raise NotImplementedError

    def _obtain_symbolic_data_matrix_and_state_vars(
        self,
        data_matrix_dir_path: Optional[Path] = None,
        model_path: Optional[str] = None,
    ) -> Tuple[ndarray, SymJointStateVariables]:
        assert not (
            data_matrix_dir_path is None and model_path is None
        ), "Must provide either data matrix dir path or model path!"

        if data_matrix_dir_path is None:  # Compute W_sym
            arm_components = create_arm(
                arm_file_path=model_path, num_joints=self._num_joints, time_step=0.0
            )
            sym_plant_components = create_symbolic_plant(
                arm_components=arm_components, use_lumped_parameters=True
            )
            sym_state_variables = sym_plant_components.state_variables
            W_sym = extract_symbolic_data_matrix(
                symbolic_plant_components=sym_plant_components
            )
        else:  # Load W_sym
            q_var = MakeVectorVariable(self._num_joints, "q")
            q_dot_var = MakeVectorVariable(self._num_joints, "\dot{q}")
            q_ddot_var = MakeVectorVariable(self._num_joints, "\ddot{q}")
            tau_var = MakeVectorVariable(self._num_joints, "\tau")
            sym_state_variables = SymJointStateVariables(
                q=q_var, q_dot=q_dot_var, q_ddot=q_ddot_var, tau=tau_var
            )
            W_sym = load_symbolic_data_matrix(
                dir_path=data_matrix_dir_path,
                sym_state_variables=sym_state_variables,
                num_joints=self._num_joints,
                num_params=self._num_params,
            )
        return W_sym, sym_state_variables


def condition_number_cost(W_dataTW_data_numeric: np.ndarray) -> float:
    """Computes the condition number of the matrix W_dataTW_data_numeric.

    Args:
        W_dataTW_data_numeric (np.ndarray): The matrix W_dataTW_data in numeric form.

    Returns:
        float: The condition number of the matrix W_dataTW_data_numeric or np.inf if
            the matrix is not positive definite.
    """
    eigenvalues, _ = np.linalg.eigh(W_dataTW_data_numeric)
    min_eig_idx = np.argmin(eigenvalues)
    max_eig_idx = np.argmax(eigenvalues)
    min_eig = eigenvalues[min_eig_idx]
    max_eig = eigenvalues[max_eig_idx]

    if min_eig <= 0:
        return np.inf

    condition_number = max_eig / min_eig
    wandb.log({"condition_number": condition_number})
    return condition_number


def condition_number_and_d_optimality_cost(
    W_dataTW_data_numeric: np.ndarray,
    d_optimality_weight: float = 0.1,
    log_det_scaling: float = 1e-10,
) -> float:
    """Computes the cost of the condition number and d-optimality.

    Args:
        W_dataTW_data_numeric (np.ndarray): The matrix W_dataTW_data in numeric form.
        d_optimality_weight (float, optional): The weight of the d-optimality term.
        log_det_scaling (float, optional): The scaling factor for the log determinant.
            This is used to prevent the log determinant from reaching infinity. The
            default value seems to work well for the iiwa.

    Returns:
        float: The cost of the condition number and d-optimality or np.inf if the matrix
            is not positive definite.
    """
    eigenvalues, _ = np.linalg.eigh(W_dataTW_data_numeric)
    min_eig_idx = np.argmin(eigenvalues)
    max_eig_idx = np.argmax(eigenvalues)
    min_eig = eigenvalues[min_eig_idx]
    max_eig = eigenvalues[max_eig_idx]

    if min_eig <= 0:
        return np.inf

    condition_number = max_eig / min_eig
    # NOTE: The scaling is needed to prevent the log_det from reaching inf
    log_det = np.log(np.prod(eigenvalues * log_det_scaling))
    d_optimality = -log_det

    cost = condition_number + d_optimality_weight * d_optimality
    wandb.log({"condition_number": condition_number, "d_optimality": d_optimality})
    return cost


def condition_number_and_e_optimality_cost(
    W_dataTW_data_numeric: np.ndarray, e_optimality_weight: float = 1e-3
) -> float:
    """Computes the cost of the condition number and e-optimality.

    Args:
        W_dataTW_data_numeric (np.ndarray): The matrix W_dataTW_data in numeric form.
        e_optimality_weight (float, optional): The weight of the e-optimality term.

    Returns:
        float: The cost of the condition number and e-optimality or np.inf if the matrix
            is not positive definite.
    """
    eigenvalues, _ = np.linalg.eigh(W_dataTW_data_numeric)
    min_eig_idx = np.argmin(eigenvalues)
    max_eig_idx = np.argmax(eigenvalues)
    min_eig = eigenvalues[min_eig_idx]
    max_eig = eigenvalues[max_eig_idx]

    if min_eig <= 0:
        return np.inf

    condition_number = max_eig / min_eig
    e_optimality = -min_eig

    cost = condition_number + e_optimality_weight * e_optimality
    wandb.log({"condition_number": condition_number, "e_optimality": e_optimality})
    return cost
