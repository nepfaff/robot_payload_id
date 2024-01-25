from typing import Optional

import numpy as np

from pydrake.all import AutoDiffXd, MultibodyForces_, MultibodyPlant
from tqdm import tqdm

from robot_payload_id.utils import JointData


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
        sample_times_s=np.arange(num_timesteps) * 1e-3,
    )
    return joint_data


def compute_autodiff_joint_data_from_fourier_series_traj_params(
    num_timesteps: int,
    time_horizon: float,
    a: np.ndarray,
    b: np.ndarray,
    q0: np.ndarray,
    omega: float = 0.2 * np.pi,
    plant: Optional[MultibodyPlant] = None,
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
        a (np.ndarray): The `a` parameters for the trajectory of type AutoDiffXd and
            shape (num_joints, num_fourier_terms).
        b (np.ndarray): The `b` parameters for the trajectory of type AutoDiffXd and
            shape (num_joints, num_fourier_terms).
        q0 (np.ndarray): The `q0` parameters for the trajectory of type AutoDiffXd and
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
    num_terms = a.shape[1]
    times = np.linspace(0, time_horizon, num_timesteps)
    for t in range(num_timesteps):
        for i in range(len(a)):
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
        sample_times_s=np.arange(num_timesteps) * 1e-3,
    )
    return joint_data


def compute_autodiff_joint_data_from_fourier_series_traj_params1(
    num_timesteps: int,
    time_horizon: float,
    a: np.ndarray,
    b: np.ndarray,
    q0: np.ndarray,
    omega: float = 0.3 * np.pi,
    plant: Optional[MultibodyPlant] = None,
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
        a (np.ndarray): The `a` parameters for the trajectory of type AutoDiffXd and
            shape (num_joints, num_fourier_terms).
        b (np.ndarray): The `b` parameters for the trajectory of type AutoDiffXd and
            shape (num_joints, num_fourier_terms).
        q0 (np.ndarray): The `q0` parameters for the trajectory of type AutoDiffXd and
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
    num_terms = a.shape[1]
    times = np.linspace(0, time_horizon, num_timesteps)
    for t in range(num_timesteps):
        for i in range(len(a)):
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
        sample_times_s=np.arange(num_timesteps) * 1e-3,
    )
    return joint_data
