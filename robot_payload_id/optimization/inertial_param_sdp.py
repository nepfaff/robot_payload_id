import logging

from typing import List, Optional, Tuple

import numpy as np

from pydrake.all import (
    CommonSolverOption,
    DecomposeAffineExpressions,
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    SolverOptions,
)

from robot_payload_id.utils import JointParameters


def add_entropic_divergence_regularization(
    prog: MathematicalProgram,
    pseudo_inertias: List[np.ndarray],
    pseudo_inertias_guess: List[np.ndarray],
    regularization_weight: float,
) -> None:
    """Adds entropic divergence regularization to the inertial parameter SDP.
    Entropic divergence is from Eq. (19) of "Geometric Robot Dynamic Identification:
    A Convex Programming Approach" (https://ieeexplore.ieee.org/document/8922724):
    d_M(ϕ, ϕ₀)² = d_F(P || P₀) = -log(|P|) + tr(P₀⁻¹ P), where ℙ(4) represents
    pseudo-inertias.
    See https://github.com/alex07143/Geometric-Robot-DynID/blob/592e64c5a9/Functions/Identification/ID_Entropic.m#L67
    for the original MATLAB implementation.

    Args:
        prog (MathematicalProgram): The MathematicalProgram to add the cost to.
        pseudo_inertias (List[np.ndarray]): The pseudo-inertias to regularize.
        pseudo_inertias_guess (List[np.ndarray]): The pseudo-inertias initial guess.
        regularization_weight (float): The regularization weight.
    """
    for pseudo_inertia, pseudo_inertia_guess in zip(
        pseudo_inertias, pseudo_inertias_guess
    ):
        # The AddMaximizeLogDeterminantCost cost is -∑ᵢt(i)
        # We remove the cost and add it back with scalling
        cost, t, _ = prog.AddMaximizeLogDeterminantCost(pseudo_inertia)
        prog.RemoveCost(cost)
        prog.AddLinearCost(-regularization_weight * np.sum(t))

        prog.AddCost(
            regularization_weight
            * np.trace(np.linalg.inv(pseudo_inertia_guess) @ pseudo_inertia)
        )


def solve_inertial_param_sdp(
    num_links: int,
    W_data: np.ndarray,
    tau_data: np.ndarray,
    base_param_mapping: Optional[np.ndarray] = None,
    regularization_weight: float = 0.0,
    params_guess: Optional[List[JointParameters]] = None,
    use_euclidean_regularization: bool = False,
    identify_rotor_inertia: bool = False,
    identify_reflected_inertia: bool = True,
    identify_viscous_friction: bool = True,
    identify_dynamic_dry_friction: bool = True,
    payload_only=False,
    solver_kPrintToConsole: bool = False,
) -> Tuple[
    MathematicalProgram, MathematicalProgramResult, np.ndarray, np.ndarray, np.ndarray
]:
    """Solves the inertial parameter SDP with a quadratic cost, a parameter
    regularization term, and inertial parameter feasibility constraints (pseudo inertias
    being positive definite).

    Args:
        num_links (int): The number of links in the robot.
        W_data (np.ndarray): The data matrix.
        tau_data (np.ndarray): The joint torque data.
        base_param_mapping (np.ndarray, optional): The base parameter mapping matrix
            that maps the full parameters to the identifiable parameters. It corresponds
            to the part of V in the SVD that corresponds to non-zero singular values. If
            None, all parameters are assumed to be identifiable.
        regularization_weight (float, optional): The parameter regularization weight.
        params_guess (List[JointParameters], optional): A guess of the inertial
            parameters to use for regularization. Required if `regularization_weight` is
            non-zero.
        use_euclidean_regularization (bool, optional): Whether to use euclidean
            regularization instead of entropic divergence regularization.
        identify_rotor_inertia (bool, optional): Whether to identify the rotor inertia.
        identify_reflected_inertia (bool, optional): Whether to identify the reflected
            inertia. NOTE: It is not possible to identify both the rotor inertia and the
            reflected inertia at the same time as they are the same parameter in
            different forms (reflected inertia = rotor inertia * gear ratio^2).
        identify_viscous_friction (bool, optional): Whether to identify the viscous
            friction.
        identify_dynamic_dry_friction (bool, optional): Whether to identify the dynamic
            dry friction.
        payload_only (bool, optional): Whether to only include the 10 inertial
            parameters of the last link. These are the parameters that we care about
            for payload identification.
        solver_kPrintToConsole (bool, optional): Whether to print solver output.

    Returns:
        Tuple[MathematicalProgram, MathematicalProgramResult, np.ndarray, np.ndarray
            np.ndarray]: A tuple containing the MathematicalProgram,
            MathematicalProgramResult, an array of the variable names, an array of the
            MathematicalProgram variables, and an array of the MathematicalProgram base
            variables. The variable names and variables contain all the parameters, even
            the non-identifiable ones, while the base variables contain only the
            identifiable parameters (can be linear combinations of other parameters).
    """
    assert regularization_weight == 0.0 or params_guess is not None, (
        "Ground truth inertial parameters are required if the regularization weight is "
        "non-zero."
    )
    assert not (
        identify_rotor_inertia and identify_reflected_inertia
    ), "Cannot identify both rotor inertia and reflected inertia."

    prog = MathematicalProgram()

    # Create decision variables
    variables: List[JointParameters] = []
    for i in range(num_links):
        if payload_only and i < num_links - 1:
            # Skip all but the last link
            continue

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
                rotor_inertia=prog.NewContinuousVariables(1, f"rotor_inertia{i}")[0]
                if identify_rotor_inertia and not payload_only
                else None,
                reflected_inertia=prog.NewContinuousVariables(
                    1, f"reflected_inertia{i}"
                )[0]
                if identify_reflected_inertia and not payload_only
                else None,
                viscous_friction=prog.NewContinuousVariables(1, f"viscous_friction{i}")[
                    0
                ]
                if identify_viscous_friction and not payload_only
                else None,
                dynamic_dry_friction=prog.NewContinuousVariables(
                    1, f"dynamic_dry_friction{i}"
                )[0]
                if identify_dynamic_dry_friction and not payload_only
                else None,
            )
        )

    variable_vec = np.concatenate([var.get_lumped_param_list() for var in variables])
    variable_names = np.array([var.get_name() for var in variable_vec])

    # Optionally remove unidentifiable parameters
    if base_param_mapping is not None:
        base_variable_vec = variable_vec.T @ base_param_mapping

        logging.info(
            "Condition number after removing unidentifiable params: "
            + f"{np.linalg.cond(W_data)}"
        )
    else:
        base_variable_vec = variable_vec

        logging.info(f"Condition number: {np.linalg.cond(W_data)}")

    # Create decision variables z = x.T @ base_param_mapping to preserve good
    # conditioning
    z = prog.NewContinuousVariables(base_variable_vec.shape[0], "z")
    # Normalize cost to achieve similar scaling between the cost and linear
    # equality constraints (improves numerics and is required for solvability)
    num_datapoints = len(W_data) // num_links
    prog.AddQuadraticCost(
        2 * W_data.T @ W_data / num_datapoints,
        -2 * tau_data.T @ W_data / num_datapoints,
        tau_data.T @ tau_data / num_datapoints,
        vars=z,
        is_convex=True,
    )

    # Add constraint that z = x.T @ base_param_mapping
    A, _, x = DecomposeAffineExpressions(base_variable_vec)  # z = Ax
    num_base_params = len(z)
    prog.AddLinearEqualityConstraint(
        np.hstack([np.eye(num_base_params), -A]),
        np.zeros(num_base_params),
        np.concatenate([z, x]),
    )

    # Regularization towards initial parameter guess
    pseudo_inertias = [var.get_pseudo_inertia_matrix() for var in variables]
    if use_euclidean_regularization:
        variables_guess = np.concatenate(
            [var.get_lumped_param_list() for var in params_guess]
        )
        prog.AddQuadraticCost(
            regularization_weight
            * (variables_guess - variable_vec).T
            @ (variables_guess - variable_vec),
            is_convex=True,
        )
    else:
        pseudo_inertias_guess = [
            var.get_pseudo_inertia_matrix() for var in params_guess
        ]
        add_entropic_divergence_regularization(
            prog=prog,
            pseudo_inertias=pseudo_inertias,
            pseudo_inertias_guess=pseudo_inertias_guess,
            regularization_weight=regularization_weight,
        )

        # Use euclidean regularization for the non-inertia parameters.
        # TODO: Treat non-inertial params as elements of the set of positive scalars and
        # add the corresponding entropic divergence regularization (see part V, part a
        # in "Geometric Robot Dynamic Identification: A Convex Programming Approach").
        # This is helpful for ensuring equal scaling between the inertial and
        # non-inertial parameter regularization terms.
        non_inertia_params = []
        non_inertia_params_guess = []
        if not payload_only:
            if identify_rotor_inertia:
                non_inertia_params += [var.rotor_inertia for var in variables]
                non_inertia_params_guess += [var.rotor_inertia for var in params_guess]
            if identify_reflected_inertia:
                non_inertia_params += [var.reflected_inertia for var in variables]
                non_inertia_params_guess += [
                    var.reflected_inertia for var in params_guess
                ]
            if identify_viscous_friction:
                non_inertia_params += [var.viscous_friction for var in variables]
                non_inertia_params_guess += [
                    var.viscous_friction for var in params_guess
                ]
            if identify_dynamic_dry_friction:
                non_inertia_params += [var.dynamic_dry_friction for var in variables]
                non_inertia_params_guess += [
                    var.dynamic_dry_friction for var in params_guess
                ]
            if len(non_inertia_params) > 0:
                non_inertia_params = np.asarray(non_inertia_params)
                non_inertia_params_guess = np.asarray(non_inertia_params_guess)
                prog.AddQuadraticCost(
                    regularization_weight
                    * (non_inertia_params_guess - non_inertia_params).T
                    @ (non_inertia_params_guess - non_inertia_params),
                    is_convex=True,
                )

    # Inertial parameter feasibility constraints
    for pseudo_inertia in pseudo_inertias:
        prog.AddPositiveSemidefiniteConstraint(pseudo_inertia - 1e-6 * np.identity(4))

    # Reflected rotor inertia feasibility constraints
    if identify_rotor_inertia and not payload_only:
        rotor_inertias = np.array([var.rotor_inertia for var in variables])
        for rotor_inertia in rotor_inertias:
            prog.AddConstraint(rotor_inertia >= 0)

    # Reflected inertia feasibility constraints
    if identify_reflected_inertia and not payload_only:
        reflected_inertias = np.array([var.reflected_inertia for var in variables])
        for reflected_inertia in reflected_inertias:
            prog.AddConstraint(reflected_inertia >= 0)

    # Viscous friction feasibility constraints
    if identify_viscous_friction and not payload_only:
        viscous_frictions = np.array([var.viscous_friction for var in variables])
        for viscous_friction in viscous_frictions:
            prog.AddConstraint(viscous_friction >= 0)

    # Dynamic dry friction feasibility constraints
    if identify_dynamic_dry_friction and not payload_only:
        dynamic_dry_frictions = np.array(
            [var.dynamic_dry_friction for var in variables]
        )
        for dynamic_dry_friction in dynamic_dry_frictions:
            prog.AddConstraint(dynamic_dry_friction >= 0)

    options = None
    if solver_kPrintToConsole:
        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)

    logging.info("Starting to solve SDP...")
    result = Solve(prog=prog, solver_options=options)
    return prog, result, variable_names, variable_vec, base_variable_vec
