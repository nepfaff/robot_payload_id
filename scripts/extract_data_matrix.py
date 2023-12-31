from typing import List, Optional

import numpy as np

from drake_torch_sys_id_test import add_model_iiwa
from pydrake.all import (
    AutoDiffXd,
    DecomposeLumpedParameters,
    Evaluate,
    Expression,
    ForceElementIndex,
    MathematicalProgram,
    MultibodyForces_,
    MultibodyPlant,
    Solve,
)
from tqdm import tqdm

from robot_payload_id.environment import create_arm
from robot_payload_id.symbolic import create_autodiff_plant, create_symbolic_plant
from robot_payload_id.utils import (
    ArmComponents,
    ArmPlantComponents,
    JointData,
    JointParameters,
    SymJointStateVariables,
)


def symbolic_decomposition_with_dynamic_substitution(
    sym_plant_components: ArmPlantComponents, joint_data: JointData
):
    """Algorithm 1 from Andy's thesis."""
    num_timesteps = len(joint_data.sample_times_s)
    num_links = sym_plant_components.state_variables.q.shape[-1]
    num_lumped_params = (
        51  # num_links * 10  # TODO: True in theory but maybe not in practice
    )
    W_data = np.zeros((num_timesteps * num_links, num_lumped_params))
    # tau_data = np.zeros((num_timesteps * num_links, 1))
    tau_data = joint_data.joint_torques.flatten()

    plant = sym_plant_components.plant
    state_variables = sym_plant_components.state_variables
    for i in tqdm(range(num_timesteps)):
        # Substitute symbolic variables with numeric values
        new_context = sym_plant_components.plant_context.Clone()
        plant.SetPositions(new_context, joint_data.joint_positions[i])
        plant.SetVelocities(new_context, joint_data.joint_velocities[i])

        # Calculate inverse dynamics
        forces = MultibodyForces_[Expression](plant)
        plant.CalcForceElementsContribution(new_context, forces)
        sym_torques = plant.CalcInverseDynamics(
            new_context, joint_data.joint_accelerations[i], forces
        )

        # Decompose symbolic expressions
        sym_parameters_arr = np.concatenate(
            [params.get_base_param_list() for params in sym_plant_components.parameters]
        )
        W, alpha, w0 = DecomposeLumpedParameters(sym_torques, sym_parameters_arr)

        try:
            W_data[i * num_links : (i + 1) * num_links, :] = Evaluate(W, {})
            alpha_sym = alpha
        except:
            print("W_data shape:", W_data.shape)
            print("W shape:", W.shape)
            print("i:", i)
            print("------------------")
            continue
        # tau_data[i * num_links : (i + 1) * num_links] = Evaluate(sym_torques, {})

    print(f"Condition number: {np.linalg.cond(W_data)}")
    alpha_fit = np.linalg.lstsq(W_data, tau_data, rcond=None)[0]
    print(f"alpha: {alpha}")
    print(f"alpha_fit: {alpha_fit}")

    return W_data, alpha_sym, tau_data


def extract_data_matrix_symbolic(
    prog: Optional[MathematicalProgram] = None,
    use_implicit_dynamics: bool = False,
    use_w0: bool = False,
    use_one_link_arm: bool = False,
):
    assert not use_w0 or use_implicit_dynamics, "Can only use w0 with implicit dynamics"

    urdf_path = (
        "./models/one_link_arm.urdf" if use_one_link_arm else "./models/iiwa.dmd.yaml"
    )
    num_joints = 1 if use_one_link_arm else 7
    time_step = 0 if use_implicit_dynamics else 1e-3

    arm_components = create_arm(
        arm_file_path=urdf_path, num_joints=num_joints, time_step=time_step
    )
    sym_plant_components = create_symbolic_plant(
        arm_components=arm_components, prog=prog
    )

    sym_parameters_arr = np.concatenate(
        [params.get_base_param_list() for params in sym_plant_components.parameters]
    )
    if use_implicit_dynamics:
        derivatives = (
            sym_plant_components.plant_context.Clone().get_mutable_continuous_state()
        )
        derivatives.SetFromVector(
            np.hstack(
                (
                    0 * sym_plant_components.state_variables.q_dot,
                    sym_plant_components.state_variables.q_ddot,
                )
            )
        )
        residual = sym_plant_components.plant.CalcImplicitTimeDerivativesResidual(
            sym_plant_components.plant_context, derivatives
        )
        W_sym, alpha_sym, w0_sym = DecomposeLumpedParameters(
            residual[int(len(residual) / 2) :], sym_parameters_arr
        )
    else:
        forces = MultibodyForces_[Expression](sym_plant_components.plant)
        sym_plant_components.plant.CalcForceElementsContribution(
            sym_plant_components.plant_context, forces
        )
        sym_torques = sym_plant_components.plant.CalcInverseDynamics(
            sym_plant_components.plant_context,
            sym_plant_components.state_variables.q_ddot.T,
            forces,
        )
        W_sym, alpha_sym, w0_sym = DecomposeLumpedParameters(
            sym_torques, sym_parameters_arr
        )

    print("W:\n", W_sym)
    print("alpha:\n", alpha_sym)
    print("w0:\n", w0_sym)

    # Substitute data values and compute least squares fit
    joint_data = get_data(num_q=num_joints, plant=arm_components.plant)
    num_timesteps = len(joint_data.sample_times_s)
    num_lumped_params = num_joints * (3 if use_one_link_arm else 10)
    W_data = np.zeros((num_timesteps * num_joints, num_lumped_params))
    w0_data = np.zeros(num_timesteps * num_joints)
    tau_data = joint_data.joint_torques.flatten()

    state_variables = sym_plant_components.state_variables
    for i in tqdm(range(num_timesteps)):
        sym_to_val = {}
        for j in range(num_joints):
            sym_to_val[state_variables.q[j]] = joint_data.joint_positions[i, j]
            sym_to_val[state_variables.q_dot[j]] = joint_data.joint_velocities[i, j]
            sym_to_val[state_variables.q_ddot[j]] = joint_data.joint_accelerations[i, j]
            if use_implicit_dynamics:
                sym_to_val[state_variables.tau[j]] = joint_data.joint_torques[i, j]
        W_data[i * num_joints : (i + 1) * num_joints, :] = Evaluate(W_sym, sym_to_val)
        if use_w0:
            w0_data[i * num_joints : (i + 1) * num_joints] = Evaluate(
                w0_sym, sym_to_val
            )

    print(f"Condition number: {np.linalg.cond(W_data)}")

    if use_w0:
        alpha_fit = np.linalg.lstsq(W_data, -w0_data, rcond=None)[0]
    else:
        alpha_fit = np.linalg.lstsq(W_data, tau_data, rcond=None)[0]
    print(f"alpha_fit: {alpha_fit}")

    return W_data, alpha_sym, tau_data, sym_plant_components


def get_data(plant, num_q, N=100, add_noise: bool = False) -> JointData:
    dim_q = (N, num_q)
    q = 1.0 * np.random.uniform(size=dim_q)
    v = 3.0 * np.random.uniform(size=dim_q)
    vd = 5.0 * np.random.uniform(size=dim_q)
    tau_gt = np.empty(dim_q)

    context = plant.CreateDefaultContext()
    for i, (q_curr, v_curr, v_dot_curr) in enumerate(zip(q, v, vd)):
        plant.SetPositions(context, q_curr)
        plant.SetVelocities(context, v_curr)
        forces = MultibodyForces_(plant)
        plant.CalcForceElementsContribution(context, forces)
        tau_gt[i] = plant.CalcInverseDynamics(context, v_dot_curr, forces)

    if add_noise:
        q += 0.001 * np.random.normal(size=dim_q)
        v += 0.01 * np.random.normal(size=dim_q)
        vd += 0.05 * np.random.normal(size=dim_q)

    joint_data = JointData(
        joint_positions=q,
        joint_velocities=v,
        joint_accelerations=vd,
        joint_torques=tau_gt,
        sample_times_s=np.arange(N) * 1e-3,
    )
    return joint_data


def compute_joint_data_from_traj_params(
    plant, num_timesteps: int, a: np.ndarray, b: np.ndarray
) -> JointData:
    # qi(t) = ai * sin(ω*i*t) + bi
    # qi_dot(t) = ai * ω * i * cos(ω*i*t)
    # qi_ddot(t) = ai * (ω*i)**2 * cos(ω*i*t)

    omega = 0.5
    q = np.zeros((num_timesteps, len(a)), dtype=AutoDiffXd)
    q_dot = np.zeros((num_timesteps, len(a)), dtype=AutoDiffXd)
    q_ddot = np.zeros((num_timesteps, len(a)), dtype=AutoDiffXd)
    for t in range(num_timesteps):
        for i in range(len(a)):
            q[t, i] = a[i] * np.sin(omega * (1 + i) * t) + b[i]
            q_dot[t, i] = a[i] * omega * (1 + i) * np.cos(omega * (1 + i) * t)
            q_ddot[t, i] = a[i] * ((omega * (1 + i)) ** 2) * np.cos(omega * (1 + i) * t)

    tau_gt = np.empty((num_timesteps, len(a)))
    context = plant.CreateDefaultContext()
    for i, (q_curr, v_curr, v_dot_curr) in enumerate(zip(q, q_dot, q_ddot)):
        plant.SetPositions(context, q_curr)
        plant.SetVelocities(context, v_curr)
        forces = MultibodyForces_(plant)
        plant.CalcForceElementsContribution(context, forces)
        tau_gt[i] = plant.CalcInverseDynamics(context, v_dot_curr, forces)

    joint_data = JointData(
        joint_positions=q,
        joint_velocities=q_dot,
        joint_accelerations=q_ddot,
        joint_torques=tau_gt,
        sample_times_s=np.arange(num_timesteps) * 1e-3,
    )
    return joint_data


def create_data_matrix_from_traj_samples(
    W_sym: np.ndarray,
    sym_state_variables: SymJointStateVariables,
    q: np.ndarray,
    q_dot: np.ndarray,
    q_ddot: np.ndarray,
) -> np.ndarray:
    num_joints = q.shape[1]
    W_data = np.empty((len(q), W_sym.shape[1]), dtype=Expression)
    for i in range(len(q)):
        sym_to_val = {}
        for j in range(num_joints):
            sym_to_val[sym_state_variables.q[j]] = q[i, j]
            sym_to_val[sym_state_variables.q_dot[j]] = q_dot[i, j]
            sym_to_val[sym_state_variables.q_ddot[j]] = q_ddot[i, j]

        for m in range(num_joints):
            for n in range(W_sym.shape[1]):
                W_data[i * num_joints + m, n] = W_sym[m, n].Substitute(sym_to_val)
    return W_data


def extract_data_matrix_andy_iteration(prog: Optional[MathematicalProgram] = None):
    num_q = 7
    plant = MultibodyPlant(time_step=1e-3)
    bodies = add_model_iiwa(plant)
    # bodies = add_model_one_link_arm(plant)
    plant.Finalize()
    joint_data = get_data(num_q=num_q, plant=plant)
    arm_components = ArmComponents(
        num_joints=num_q,
        diagram=None,
        plant=plant,
        trajectory_source=None,
        state_logger=None,
        commanded_torque_logger=None,
        meshcat=None,
        meshcat_visualizer=None,
    )
    sym_plant_components = create_symbolic_plant(
        arm_components=arm_components, prog=prog
    )

    W_data, alpha_sym, tau_data = symbolic_decomposition_with_dynamic_substitution(
        sym_plant_components=sym_plant_components, joint_data=joint_data
    )
    return W_data, alpha_sym, tau_data, sym_plant_components


def extract_data_matrix_autodiff(
    use_one_link_arm: bool = False, num_data_points: int = 100
):
    num_joints = 1 if use_one_link_arm else 7
    time_step = 0
    urdf_path = (
        "./models/one_link_arm.urdf" if use_one_link_arm else "./models/iiwa.dmd.yaml"
    )
    arm_components = create_arm(
        arm_file_path=urdf_path, num_joints=num_joints, time_step=time_step
    )
    # joint_data = get_data(
    #     num_q=num_joints, plant=arm_components.plant, N=num_data_points, add_noise=True
    # )
    joint_data = compute_joint_data_from_traj_params(
        plant=arm_components.plant,
        num_timesteps=num_data_points,
        a=-40.2049 * np.ones(num_joints),
        b=20.8 * np.zeros(num_joints),
    )

    ad_plant_components = create_autodiff_plant(arm_components=arm_components)

    ad_parameters_arr = np.concatenate(
        [params.get_lumped_param_list() for params in ad_plant_components.parameters]
    )

    num_timesteps = len(joint_data.sample_times_s)
    num_lumped_params = num_joints * 10
    W_data = np.zeros((num_timesteps * num_joints, num_lumped_params))
    # w0_data = np.zeros(num_timesteps * num_joints)
    tau_data = joint_data.joint_torques.flatten()

    for i in tqdm(range(num_timesteps)):
        # Set joint data
        ad_plant_components.plant.get_actuation_input_port().FixValue(
            ad_plant_components.plant_context, joint_data.joint_torques[i]
        )
        ad_plant_components.plant.SetPositions(
            ad_plant_components.plant_context, joint_data.joint_positions[i]
        )
        ad_plant_components.plant.SetVelocities(
            ad_plant_components.plant_context, joint_data.joint_velocities[i]
        )

        # Implicit dynamics
        # derivatives = (
        #     ad_plant_components.plant_context.Clone().get_mutable_continuous_state()
        # )
        # print("derivatives", derivatives.get_mutable_vector())
        # print("size", derivatives.size())
        # derivatives.SetFromVector(
        #     np.hstack(
        #         (
        #             0 * joint_data.joint_positions[i],
        #             joint_data.joint_accelerations[i],
        #         )
        #     )
        # )
        # residual = ad_plant_components.plant.CalcImplicitTimeDerivativesResidual(
        #     ad_plant_components.plant_context, derivatives
        # )
        # implicit_dynamics = residual[int(len(residual) / 2) :]

        # Inverse dynamics
        forces = MultibodyForces_[AutoDiffXd](ad_plant_components.plant)
        ad_plant_components.plant.CalcForceElementsContribution(
            ad_plant_components.plant_context, forces
        )
        sym_torques = ad_plant_components.plant.CalcInverseDynamics(
            ad_plant_components.plant_context,
            joint_data.joint_accelerations[i],
            forces,
        )

        # Differentiate w.r.t. parameters
        sym_torques_derivative = np.vstack(
            [joint_torques.derivatives() for joint_torques in sym_torques]
        )
        W_data[i * num_joints : (i + 1) * num_joints, :] = sym_torques_derivative

    print(f"Condition number: {np.linalg.cond(W_data)}")
    alpha_fit = np.linalg.lstsq(W_data, tau_data, rcond=None)[0]
    print(f"alpha_fit: {alpha_fit}")

    return W_data, tau_data, ad_plant_components


def main_symbolic():
    prog = MathematicalProgram()

    # NOTE: All methods achieve the same results for the one link arm but the pure
    # symbolic methods does not scale to the iiwa arm
    # W_data, alpha_sym, tau_data, sym_plant_components = extract_data_matrix_symbolic(
    #     prog=prog
    # )
    (
        W_data,
        alpha_sym,
        tau_data,
        sym_plant_components,
    ) = extract_data_matrix_andy_iteration(prog=prog)

    # Construct pseudo inertia
    pseudo_inertias = []
    params = sym_plant_components.parameters
    for i in range(7):
        inertia = params[i].get_inertia_matrix()
        density_weighted_2nd_moment_matrix = (
            0.5 * np.trace(inertia) * np.eye(3) - inertia
        )
        density_weighted_1st_moment = params[i].m * np.array(
            [params[i].cx, params[i].cy, params[i].cz]
        )
        pseudo_inertias.append(
            np.block(
                [
                    [
                        density_weighted_2nd_moment_matrix,
                        density_weighted_1st_moment.reshape((3, 1)),
                    ],
                    [
                        density_weighted_1st_moment.reshape((3, 1)).T,
                        np.array([[params[i].m]]),
                    ],
                ]
            )
        )

    # Solve constrained SDP
    # Group lumped parameters to ensure convexity
    lumped_vars = prog.NewContinuousVariables(len(alpha_sym), "alpha")
    for lumped_var, old_var in zip(lumped_vars, alpha_sym, strict=True):
        # TODO: Want to define this to be true instead of adding a constraint (constraint not linear!)
        prog.AddLinearEqualityConstraint(lumped_var == old_var)

    # NOTE: This fails as vars appear in combinations (lumped parameters)
    prog.Add2NormSquaredCost(A=W_data, b=tau_data, vars=lumped_vars)
    for pseudo_inertia in pseudo_inertias:
        # NOTE: This fails due to nonlinearities (lumped parameters)
        prog.AddPositiveSemidefiniteConstraint(pseudo_inertia - 1e-3 * np.identity(4))


def solve_sdp(
    num_links, W_data, tau_data, identifiable, inverse_covariance_matrix=None
):
    prog = MathematicalProgram()

    # Create decision variables
    variables: List[JointParameters] = []
    for i in range(num_links):
        m = prog.NewContinuousVariables(1, f"m{i}")[0]
        hx = prog.NewContinuousVariables(1, f"hx{i}")[0]
        hy = prog.NewContinuousVariables(1, f"hy{i}")[0]
        hz = prog.NewContinuousVariables(1, f"hz{i}")[0]
        Ixx = prog.NewContinuousVariables(1, f"Ixx{i}")[0]
        Ixy = prog.NewContinuousVariables(1, f"Ixy{i}")[0]
        Ixz = prog.NewContinuousVariables(1, f"Ixz{i}")[0]
        Iyy = prog.NewContinuousVariables(1, f"Iyy{i}")[0]
        Iyz = prog.NewContinuousVariables(1, f"Iyz{i}")[0]
        Izz = prog.NewContinuousVariables(1, f"Izz{i}")[0]

        variables.append(
            JointParameters(
                m=m,
                hx=hx,
                hy=hy,
                hz=hz,
                Ixx=Ixx,
                Ixy=Ixy,
                Ixz=Ixz,
                Iyy=Iyy,
                Iyz=Iyz,
                Izz=Izz,
            )
        )

    # Construct pseudo inertia
    pseudo_inertias = []
    for i in range(num_links):
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

    # Solve constrained SDP
    variable_list = np.concatenate([var.get_lumped_param_list() for var in variables])
    variable_names = np.array([var.get_name() for var in variable_list])

    if identifiable is not None:
        # Remove unidentifiable parameters
        W_data = W_data[:, identifiable]
        variable_list = variable_list[identifiable]
        variable_names = variable_names[identifiable]

        print(
            f"Condition number after removing unidentifiable params: {np.linalg.cond(W_data)}"
        )

    if inverse_covariance_matrix is None:
        prog.Add2NormSquaredCost(A=W_data, b=tau_data, vars=variable_list)
        # print(
        #     (W_data @ variable_list - tau_data).T @ (W_data @ variable_list - tau_data)
        # )
    else:
        prog.AddQuadraticCost(
            (W_data @ variable_list - tau_data).T
            @ inverse_covariance_matrix
            @ (W_data @ variable_list - tau_data),
            is_convex=True,
        )
        # print(
        #     (W_data @ variable_list - tau_data).T
        #     @ inverse_covariance_matrix
        #     @ (W_data @ variable_list - tau_data)
        # )
    # TODO: Replace this with geometric regularization
    regularization_weight = 1e-3
    prog.AddQuadraticCost(
        regularization_weight * variable_list.T @ variable_list, is_convex=True
    )
    for pseudo_inertia in pseudo_inertias:
        prog.AddPositiveSemidefiniteConstraint(pseudo_inertia - 1e-6 * np.identity(4))

    from pydrake.all import CommonSolverOption, SolverOptions

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)

    result = Solve(prog, None, options)
    if result.is_success():
        var_sol_dict = dict(zip(variable_names, result.GetSolution(variable_list)))
        print("SDP result:\n", var_sol_dict)

        ## NOTE: All these inverse covariance matrix lead to worse results than without
        # scaling (numerics get worse not better...)

        # # Compute error covariance matrix
        # alpha = result.GetSolution(variable_list)
        # error = W_data @ alpha - tau_data
        # simga_squared = error.T @ error / (len(tau_data) - len(alpha))
        # # Can't compute error_covariance as W is not full column rank as not all
        # # parameters are identifiable and thus W.T @ W is not invertible
        # # error_covariance = simga_squared * np.linalg.inv(W_data.T @ W_data)
        # inverse_error_covariance = W_data.T @ W_data / simga_squared

        # Compute sample covariance matrix from https://www.sciencedirect.com/topics/mathematics/sample-covariance-matrix
        # alpha = result.GetSolution(variable_list)
        # error = (W_data @ alpha - tau_data).reshape((-1, 1))
        # mean_error = np.mean(error)
        # normalized_error = error - mean_error
        # sample_covariance = (normalized_error @ normalized_error.T) / len(error)
        # inverse_error_covariance = np.linalg.inv(sample_covariance)

        # From ChatGPT, similar to "Inertial Parameter Identification in Robotics: A Survey"
        # Covariance matrix diagonal with diagonal variances being equal to the squared
        # error
        # alpha = result.GetSolution(variable_list)
        # error = W_data @ alpha - tau_data
        # sample_covariance = np.zeros((len(tau_data), len(tau_data)))
        # for i in range(len(tau_data)):
        #     sample_covariance[i, i] = error[i] ** 2
        # inverse_error_covariance = np.linalg.inv(sample_covariance)

        # Sample covariance matrix associated with the torque observations (Optimal
        # excitation trajectories for mechanical system identification)
        # tau_data_mean = np.mean(tau_data)
        # tau_data_normalized = (tau_data - tau_data_mean).reshape((-1, 1))
        # sample_covariance = (tau_data_normalized @ tau_data_normalized.T) / (len(tau_data)-1)
        # inverse_error_covariance = np.linalg.inv(sample_covariance)

        # Diagonal sample covariance matrix associated with the torque observations
        # sample_covariance = np.zeros((len(tau_data), len(tau_data)))
        # tau_data_mean = np.mean(tau_data)
        # tau_data_normalized = tau_data - tau_data_mean
        # for i in range(len(tau_data)):
        #     sample_covariance[i, i] = tau_data_normalized[i] ** 2
        # inverse_error_covariance = np.linalg.inv(sample_covariance)

        # Diagonal error covariance matrix from "Inertial Parameter Identification in
        # Robotics: A Survey" and repeating it for each observation
        alpha = result.GetSolution(variable_list)
        errors = W_data @ alpha - tau_data
        errors_per_joint = errors.reshape(
            (-1, num_links)
        ).T  # Shape (num_links, num_timesteps)
        num_samples = len(tau_data) / num_links
        variance_per_link = np.empty(num_links)
        for i in range(num_links):
            variance_per_link[i] = np.linalg.norm(errors_per_joint[i]) / (
                num_samples - len(alpha)
            )
        variances = np.repeat(variance_per_link, num_samples)
        inverse_error_covariance = np.diag(1 / variances)

        # print(inverse_error_covariance)
    else:
        print("Failed to solve")
        print(prog)
        print(result.get_solution_result())
        print(
            result.get_solver_details().rescode,
            result.get_solver_details().solution_status,
        )

    name = (
        "iiwa_id_sdp_latex.txt"
        if inverse_covariance_matrix is None
        else "iiwa_id_sdp_rescaled_latex.txt"
    )
    with open(name, "w") as f:
        f.write(prog.ToLatex())

    # Singular values of W_data
    # print(np.linalg.eigvalsh(W_data.T @ W_data))

    return alpha, inverse_error_covariance


def main_ad():
    use_one_link_arm = True
    remove_unidentifiable_params = True
    num_data_points = 50000
    num_links = 1 if use_one_link_arm else 7
    W_data, tau_data, _ = extract_data_matrix_autodiff(
        use_one_link_arm=use_one_link_arm, num_data_points=num_data_points
    )

    identifiable = None
    if remove_unidentifiable_params:
        # Identify the identifiable parameters using the QR decomposition
        _, R = np.linalg.qr(W_data)
        tolerance = 1e-12
        identifiable = np.abs(np.diag(R)) > tolerance

    print("Solving the first time")
    alpha, inverse_error_covariance = solve_sdp(
        num_links=num_links, W_data=W_data, tau_data=tau_data, identifiable=identifiable
    )

    # print("Solving the second time")
    # alpha1, inverse_error_covariance1 = solve_sdp(
    #     num_links=num_links,
    #     W_data=W_data,
    #     tau_data=tau_data,
    #     identifiable=identifiable,
    #     inverse_covariance_matrix=inverse_error_covariance,
    # )


if __name__ == "__main__":
    # extract_data_matrix_autodiff(use_one_link_arm=False)
    # extract_data_matrix_symbolic(use_one_link_arm=True)
    main_ad()
