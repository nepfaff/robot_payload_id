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
    plant: MultibodyPlant,
    num_timesteps: int,
    timestep: float,
    a: np.ndarray,
    b: np.ndarray,
    omega: float,
) -> JointData:
    """Generates autodiff joint data from the following simple sinusoidal trajectory
    parameterization:
        qi(t) = ai * sin(ω*i*t) + bi
        qi_dot(t) = ai * ω * i * cos(ω*i*t)
        qi_ddot(t) = ai * (ω*i)**2 * cos(ω*i*t)

    Args:
        plant (MultibodyPlant): The plant to generate data for.
        num_timesteps (int): The number of datapoints to generate.
        timestep (float): The timestep between datapoints.
        a (np.ndarray): The `a` parameters for the trajectory of type AutoDiffXd and
            shape (num_joints,).
        b (np.ndarray): The `b` parameters for the trajectory of type AutoDiffXd and
            shape (num_joints,).
        omega (float): The frequency of the trajectory.

    Returns:
        JointData: The generated joint data.
    """

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
    tau_gt = np.empty((num_timesteps, len(a)))
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
