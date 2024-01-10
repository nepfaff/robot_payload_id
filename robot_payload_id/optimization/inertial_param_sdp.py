import logging

from typing import List, Optional, Tuple

import numpy as np

from pydrake.all import (
    CommonSolverOption,
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    SolverOptions,
)

from robot_payload_id.utils import JointParameters


def construct_psuedo_inertias(variables: List[JointParameters]) -> List[np.ndarray]:
    pseudo_inertias = []
    for i in range(len(variables)):
        mass = np.array([[variables[i].m]])
        inertia = variables[i].get_inertia_matrix()
        density_weighted_2nd_moment_matrix = (
            0.5 * np.trace(inertia) * np.eye(3) - inertia
        )
        density_weighted_1st_moment = np.array(
            [variables[i].hx, variables[i].hy, variables[i].hz]
        ).reshape((3, 1))
        pseudo_inertias.append(
            np.block(
                [
                    [
                        density_weighted_2nd_moment_matrix,
                        density_weighted_1st_moment,
                    ],
                    [
                        density_weighted_1st_moment.T,
                        mass,
                    ],
                ]
            )
        )
    return pseudo_inertias


def solve_inertial_param_sdp(
    num_links: int,
    W_data: np.ndarray,
    tau_data: np.ndarray,
    identifiable: Optional[np.ndarray] = None,
    regularization_weight: float = 0.0,
    solver_kPrintToConsole: bool = False,
) -> Tuple[MathematicalProgram, MathematicalProgramResult, np.ndarray, np.ndarray]:
    """Solves the inertial parameter SDP with a quadratic cost, a parameter
    regularization term, and inertial parameter feasibility constraints (pseudo inertias
    being positive definite).

    Args:
        num_links (int): The number of links in the robot.
        W_data (np.ndarray): The data matrix.
        tau_data (np.ndarray): The joint torque data.
        identifiable (np.ndarray, optional): A boolean array of shape
            (num_lumped_params,) where True indicates that the corresponding parameter
            is identifiable. If None, all parameters are assumed to be identifiable.
        regularization_weight (float, optional): The parameter regularization weight.
        solver_kPrintToConsole (bool, optional): Whether to print solver output.

    Returns:
        Tuple[MathematicalProgram, MathematicalProgramResult, np.ndarray, np.ndarray]: A
            tuple containing the MathematicalProgram, MathematicalProgramResult, an
            array of the variable names, and an array of the MathematicalProgram
            variables. The variable names and variables only contain the identifiable
            parameters.
    """
    prog = MathematicalProgram()

    # Create decision variables
    variables: List[JointParameters] = []
    for i in range(num_links):
        variables.append(
            JointParameters(
                m=prog.NewContinuousVariables(1, f"m{i}")[0],
                hx=prog.NewContinuousVariables(1, f"hx{i}")[0],
                hy=prog.NewContinuousVariables(1, f"hy{i}")[0],
                hz=prog.NewContinuousVariables(1, f"hz{i}")[0],
                Ixx=prog.NewContinuousVariables(1, f"Ixx{i}")[0],
                Ixy=prog.NewContinuousVariables(1, f"Ixy{i}")[0],
                Ixz=prog.NewContinuousVariables(1, f"Ixz{i}")[0],
                Iyy=prog.NewContinuousVariables(1, f"Iyy{i}")[0],
                Iyz=prog.NewContinuousVariables(1, f"Iyz{i}")[0],
                Izz=prog.NewContinuousVariables(1, f"Izz{i}")[0],
            )
        )

    variable_list = np.concatenate([var.get_lumped_param_list() for var in variables])
    variable_names = np.array([var.get_name() for var in variable_list])

    # Optionally remove unidentifiable parameters
    if identifiable is not None:
        W_data = W_data[:, identifiable]
        variable_list = variable_list[identifiable]
        variable_names = variable_names[identifiable]

        logging.info(
            "Condition number after removing unidentifiable params: "
            + f"{np.linalg.cond(W_data)}"
        )

    # Objective
    prog.Add2NormSquaredCost(A=W_data, b=tau_data, vars=variable_list)

    # Regularization
    # TODO: Replace this with geometric regularization
    prog.AddQuadraticCost(
        regularization_weight * variable_list.T @ variable_list, is_convex=True
    )

    # Inertial parameter feasibility constraints
    pseudo_inertias = construct_psuedo_inertias(variables)
    for pseudo_inertia in pseudo_inertias:
        prog.AddPositiveSemidefiniteConstraint(pseudo_inertia - 1e-6 * np.identity(4))

    options = None
    if solver_kPrintToConsole:
        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)

    result = Solve(prog=prog, solver_options=options)
    return prog, result, variable_names, variable_list
