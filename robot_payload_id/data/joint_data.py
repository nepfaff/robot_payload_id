from typing import Optional

import numpy as np

from pydrake.all import (
    AutoDiffXd,
    JacobianWrtVariable,
    MultibodyForces_,
    MultibodyPlant,
    SpatialForce,
)
from tqdm import tqdm

from robot_payload_id.utils import (
    ArmPlantComponents,
    FourierSeriesTrajectoryAttributes,
    JointData,
)


def generate_random_joint_data(
    plant: MultibodyPlant, num_joints: int, num_datapoints: int, add_noise: bool = False
) -> JointData:
    """Generates random joint data for a given plant.

    Args:
        plant (MultibodyPlant): The plant to generate data for.
        num_joints (int): The number of joints in the plant.
        num_datapoints (int): The number of datapoints to generate.
        add_noise (bool, optional): Whether to add zero-mean Gaussian noise to the data.

    Returns:
        JointData: The generated joint data.
    """
    dim_q = (num_datapoints, num_joints)
    q = 1.0 * np.random.uniform(size=dim_q)
    v = 3.0 * np.random.uniform(size=dim_q)
    vd = 5.0 * np.random.uniform(size=dim_q)

    # Compute the corresponding torques
    tau_gt = np.empty(dim_q)
    context = plant.CreateDefaultContext()
    for i, (q_curr, v_curr, v_dot_curr) in tqdm(
        enumerate(zip(q, v, vd)), total=num_datapoints, desc="Generating random data"
    ):
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
        sample_times_s=np.arange(num_datapoints) * 1e-3,
    )
    return joint_data


def compute_autodiff_joint_data_from_simple_sinusoidal_traj_params(
    num_timesteps: int,
    timestep: float,
    a: np.ndarray,
    b: np.ndarray,
    omega: float,
    plant: Optional[MultibodyPlant] = None,
) -> JointData:
    """Generates autodiff joint data from the following simple sinusoidal trajectory
    parameterization:
        qi(t) = ai * sin(ω*i*t) + bi
        qi_dot(t) = ai * ω * i * cos(ω*i*t)
        qi_ddot(t) = ai * (ω*i)**2 * cos(ω*i*t)

    Args:
        num_timesteps (int): The number of datapoints to generate.
        timestep (float): The timestep between datapoints.
        a (np.ndarray): The `a` parameters for the trajectory of type AutoDiffXd and
            shape (num_joints,).
        b (np.ndarray): The `b` parameters for the trajectory of type AutoDiffXd and
            shape (num_joints,).
        omega (float): The frequency of the trajectory.
        plant (MultibodyPlant, optional): The plant to generate data for. If None, then
            the torques will be set to zero.

    Returns:
        JointData: The generated joint data.
    """
    # Compute the joint positions, velocities, and accelerations
    q = np.zeros((num_timesteps, len(a)), dtype=AutoDiffXd)
    q_dot = np.zeros((num_timesteps, len(a)), dtype=AutoDiffXd)
    q_ddot = np.zeros((num_timesteps, len(a)), dtype=AutoDiffXd)
    for t in range(num_timesteps):
        for i in range(len(a)):
            time = t * timestep
            q[t, i] = a[i] * np.sin(omega * (1 + i) * time) + b[i]
            q_dot[t, i] = a[i] * omega * (1 + i) * np.cos(omega * (1 + i) * time)
            q_ddot[t, i] = (
                a[i] * ((omega * (1 + i)) ** 2) * np.cos(omega * (1 + i) * time)
            )

    # Compute the corresponding torques
    tau_gt = np.zeros((num_timesteps, len(a)))
    if plant is not None:
        context = plant.CreateDefaultContext()
        for i, (q_curr, v_curr, v_dot_curr) in tqdm(
            enumerate(zip(q, q_dot, q_ddot)),
            total=num_timesteps,
            desc="Generating sinusoidal data",
        ):
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
        sample_times_s=np.arange(num_timesteps) * timestep,
    )
    return joint_data


def compute_autodiff_joint_data_from_fourier_series_traj_params(
    num_timesteps: int,
    time_horizon: float,
    traj_attrs: FourierSeriesTrajectoryAttributes,
    plant: Optional[MultibodyPlant] = None,
    use_progress_bar: bool = True,
) -> JointData:
    """Generates autodiff joint data from the following Fourier series trajectory
    parameterization (see "Optimal robot excitation and identification", Equation 11):
        qᵢ(t) = ∑ₗ₌₁ᴺᵢ (aₗⁱ / (ωₙ l) sin(ωₙ lt) - bₗⁱ / (ωₙ l) cos(ωₙ lt)) + qᵢ₀
        q̇ᵢ(t) = ∑ₗ₌₁ᴺᵢ (aₗⁱ cos(ωₙ lt) + bₗⁱ sin(ωₙ lt))
        q̈ᵢ(t) = ∑ₗ₌₁ᴺᵢ (-aₗⁱ ωₙ l sin(ωₙ lt) + bₗⁱ ωₙ l cos(ωₙ lt))

    Args:
        num_timesteps (int): The number of datapoints to generate.
        time_horizon (float): The time horizon/ duration of the trajectory. The sampling
            time step is computed as time_horizon / num_timesteps.
        traj_attrs (FourierSeriesTrajectoryAttributes): The Fourier series trajectory
            parameters. A recommended value for omega is 0.2 * np.pi.
        plant (MultibodyPlant, optional): The plant to generate data for. If None, then
            the torques will be set to zero.
        use_progress_bar (bool, optional): Whether to use a progress bar.

    Returns:
        JointData: The generated joint data.
    """
    a = traj_attrs.a_values
    b = traj_attrs.b_values
    q0 = traj_attrs.q0_values
    omega = traj_attrs.omega

    # Compute the joint positions, velocities, and accelerations
    num_joints, num_terms = a.shape
    q = np.zeros((num_timesteps, num_joints), dtype=AutoDiffXd)
    q_dot = np.zeros((num_timesteps, num_joints), dtype=AutoDiffXd)
    q_ddot = np.zeros((num_timesteps, num_joints), dtype=AutoDiffXd)
    num_terms = a.shape[1]
    times = np.linspace(0, time_horizon, num_timesteps, endpoint=True)
    for t in tqdm(
        range(num_timesteps),
        total=num_timesteps,
        desc="Generating joint data from traj params.",
        disable=not use_progress_bar,
    ):
        for i in range(num_joints):
            time = times[t]
            q[t, i] = (
                sum(
                    (
                        a[i][l - 1] / (omega * l) * np.sin(omega * l * time)
                        - b[i][l - 1] / (omega * l) * np.cos(omega * l * time)
                    )
                    for l in range(1, num_terms + 1)
                )
                + q0[i]
            )
            q_dot[t, i] = sum(
                (
                    a[i][l - 1] * np.cos(omega * l * time)
                    + b[i][l - 1] * np.sin(omega * l * time)
                )
                for l in range(1, num_terms + 1)
            )
            q_ddot[t, i] = sum(
                (
                    -a[i][l - 1] * omega * l * np.sin(omega * l * time)
                    + b[i][l - 1] * omega * l * np.cos(omega * l * time)
                )
                for l in range(1, num_terms + 1)
            )

    # Compute the corresponding torques
    tau_gt = np.zeros((num_timesteps, len(a)))
    if plant is not None:
        context = plant.CreateDefaultContext()
        for i, (q_curr, v_curr, v_dot_curr) in tqdm(
            enumerate(zip(q, q_dot, q_ddot)),
            total=num_timesteps,
            desc="Generating GT torque data.",
            disable=not use_progress_bar,
        ):
            plant.SetPositions(context, q_curr)
            plant.SetVelocities(context, v_curr)
            forces = MultibodyForces_(plant)
            plant.CalcForceElementsContribution(context, forces)
            tau_gt[i] = plant.CalcInverseDynamics(context, v_dot_curr, forces)

    sample_delta = time_horizon / num_timesteps
    joint_data = JointData(
        joint_positions=q,
        joint_velocities=q_dot,
        joint_accelerations=q_ddot,
        joint_torques=tau_gt,
        sample_times_s=np.arange(num_timesteps) * sample_delta,
    )
    return joint_data


def compute_autodiff_joint_data_from_fourier_series_traj_params1(
    num_timesteps: int,
    time_horizon: float,
    traj_attrs: FourierSeriesTrajectoryAttributes,
    plant: Optional[MultibodyPlant] = None,
    reflected_inertias: Optional[np.ndarray] = None,
    viscous_frictions: Optional[np.ndarray] = None,
    dynamic_dry_frictions: Optional[np.ndarray] = None,
    use_progress_bar: bool = True,
) -> JointData:
    """Generates autodiff joint data from the following Fourier series trajectory
    parameterization (It is a slight modificaiton of
    "Optimal robot excitation and identification", Equation 11):
        qᵢ(t) = ∑ₗ₌₁ᴺᵢ (aₗⁱ sin(ωₙ lt) + bₗⁱ cos(ωₙ lt)) + qᵢ₀
        q̇ᵢ(t) = ∑ₗ₌₁ᴺᵢ (aₗⁱ ωₙ l cos(ωₙ lt) - bₗⁱ ωₙ l sin(ωₙ lt))
        q̈ᵢ(t) = ∑ₗ₌₁ᴺᵢ (-aₗⁱ ωₙ^2 l^2 sin(ωₙ lt) - bₗⁱ ωₙ^2 l^2 cos(ωₙ lt))

    Args:
        num_timesteps (int): The number of datapoints to generate.
        time_horizon (float): The time horizon/ duration of the trajectory. The sampling
            time step is computed as time_horizon / num_timesteps.
        traj_attrs (FourierSeriesTrajectoryAttributes): The Fourier series trajectory
            parameters. A recommended value for omega is 0.3 * np.pi.
        plant (MultibodyPlant, optional): The plant to generate data for. If None, then
            the torques will be set to zero.
        reflected_inertias (np.ndarray, optional): The reflected inertias of the joints.
            If None, then reflected inertias are not added to the torques.
        viscous_frictions (np.ndarray, optional): The viscous frictions of the joints.
            If None, then viscous frictions are not added to the torques.
        dynamic_dry_frictions (np.ndarray, optional): The dynamic dry frictions of the
            joints. If None, then dynamic dry frictions are not added to the torques.
        use_progress_bar (bool, optional): Whether to use a progress bar.

    Returns:
        JointData: The generated joint data.
    """
    a = traj_attrs.a_values
    b = traj_attrs.b_values
    q0 = traj_attrs.q0_values
    omega = traj_attrs.omega

    # Compute the joint positions, velocities, and accelerations
    num_joints, num_terms = a.shape
    q = np.zeros((num_timesteps, num_joints), dtype=AutoDiffXd)
    q_dot = np.zeros((num_timesteps, num_joints), dtype=AutoDiffXd)
    q_ddot = np.zeros((num_timesteps, num_joints), dtype=AutoDiffXd)
    times = np.linspace(0, time_horizon, num_timesteps, endpoint=True)
    for t in tqdm(
        range(num_timesteps),
        total=num_timesteps,
        desc="Generating joint data from traj params.",
        disable=not use_progress_bar,
    ):
        for i in range(num_joints):
            time = times[t]
            q[t, i] = (
                sum(
                    (
                        a[i][l - 1] * np.sin(omega * l * time)
                        + b[i][l - 1] * np.cos(omega * l * time)
                    )
                    for l in range(1, num_terms + 1)
                )
                + q0[i]
            )
            q_dot[t, i] = sum(
                (
                    a[i][l - 1] * omega * l * np.cos(omega * l * time)
                    - b[i][l - 1] * omega * l * np.sin(omega * l * time)
                )
                for l in range(1, num_terms + 1)
            )
            q_ddot[t, i] = sum(
                (
                    -a[i][l - 1] * omega * l * omega * l * np.sin(omega * l * time)
                    - b[i][l - 1] * omega * l * omega * l * np.cos(omega * l * time)
                )
                for l in range(1, num_terms + 1)
            )

    # Compute the corresponding torques
    tau_gt = np.zeros((num_timesteps, len(a)))
    if plant is not None:
        context = plant.CreateDefaultContext()
        for i, (q_curr, v_curr, v_dot_curr) in tqdm(
            enumerate(zip(q, q_dot, q_ddot)),
            total=num_timesteps,
            desc="Generating GT torque data.",
            disable=not use_progress_bar,
        ):
            plant.SetPositions(context, q_curr)
            plant.SetVelocities(context, v_curr)
            forces = MultibodyForces_(plant)
            plant.CalcForceElementsContribution(context, forces)
            tau_gt[i] = plant.CalcInverseDynamics(context, v_dot_curr, forces)

            if reflected_inertias is not None:
                tau_gt[i] += reflected_inertias[i] * joint_data.joint_accelerations[i]
            if viscous_frictions is not None:
                tau_gt[i] += viscous_frictions[i] * joint_data.joint_velocities[i]
            if dynamic_dry_frictions is not None:
                tau_gt[i] += dynamic_dry_frictions[i] * np.sign(
                    joint_data.joint_velocities[i]
                )

    sample_delta = time_horizon / num_timesteps
    joint_data = JointData(
        joint_positions=q,
        joint_velocities=q_dot,
        joint_accelerations=q_ddot,
        joint_torques=tau_gt,
        sample_times_s=np.arange(num_timesteps) * sample_delta,
    )
    return joint_data


def compute_ft_sensor_measurements(
    arm_plant_components: ArmPlantComponents,
    joint_data: JointData,
    ft_sensor_frame_name: str,
) -> JointData:
    """TODO"""
    plant = arm_plant_components.plant
    context = arm_plant_components.plant_context

    measurement_frame = plant.GetFrameByName(ft_sensor_frame_name)
    X_WF = measurement_frame.CalcPoseInWorld(context)
    X_FW = X_WF.inverse()

    ft_sensor_weldjoint = plant.GetJointByName("ft_sensor_weldjoint")
    ft_sensor_weldjoint_idx = ft_sensor_weldjoint.index()

    # Compute transform from frame Jc on child body C to measurement frame F.
    Jc_frame = ft_sensor_weldjoint.frame_on_child()
    X_WJc = Jc_frame.CalcPoseInWorld(context)
    X_FJc = X_FW @ X_WJc
    R_FJc = X_FJc.rotation()

    ft_sensor_measurements = np.empty((joint_data.joint_positions.shape[0], 6))
    for i in range(joint_data.joint_positions.shape[0]):
        # Set joint data
        plant.SetPositions(context, joint_data.joint_positions[i])
        plant.SetVelocities(context, joint_data.joint_velocities[i])

        # Compute the reaction force F_CJc_Jc on the child body C at the last joint's
        # child frame Jc.
        F_CJc_Jc: SpatialForce = plant.get_reaction_forces_output_port().Eval(context)[
            ft_sensor_weldjoint_idx
        ]

        # Invert using action-reaction principle.
        f_CJc_Jc = -F_CJc_Jc.translational()
        tau_CJc_Jc = -F_CJc_Jc.rotational()

        # Express in sensor frame.
        f_CJc_F = R_FJc @ f_CJc_Jc
        tau_CJc_F = R_FJc @ tau_CJc_Jc

        ft_sensor_measurement = np.concatenate([f_CJc_F, tau_CJc_F])
        ft_sensor_measurements[i] = ft_sensor_measurement

    return JointData(
        joint_positions=joint_data.joint_positions,
        joint_velocities=joint_data.joint_velocities,
        joint_accelerations=joint_data.joint_accelerations,
        joint_torques=joint_data.joint_torques,
        ft_sensor_measurements=ft_sensor_measurements,
        sample_times_s=joint_data.sample_times_s,
    )
