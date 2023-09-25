from typing import List

import numpy as np

from pydrake.all import VectorLog

from .dataclasses import JointData


def gather_joint_log_data(state_log: VectorLog, torque_log: VectorLog) -> JointData:
    """Gathers joint data by reading the position/ velocity logs and estimating
    accelerations using finite differences.

    Args:
        state_log (VectorLog): The robot state logs that contains information
        about joint positions and joint velocities.
        torque_log (VectorLog): The robot torque logs that contains information
        about joint torques.

    Returns:
        JointData: The joint position, velocity, and acceleration data.
    """
    num_joints = len(state_log.data()) // 2
    sample_times_s = state_log.sample_times()
    joint_positions = state_log.data()[:num_joints, :]
    joint_velocities = state_log.data()[num_joints:, :]

    # Estimate accelerations using finite differences
    joint_accelerations = np.zeros((num_joints, len(sample_times_s)))
    for i in range(num_joints):
        joint_accelerations[i, :] = np.gradient(joint_velocities[i, :], sample_times_s)

    return JointData(
        joint_positions=joint_positions.T,
        joint_velocities=joint_velocities.T,
        joint_accelerations=joint_accelerations.T,
        joint_torques=torque_log.data().T,
        sample_times_s=sample_times_s,
    )


def merge_joint_datas(joint_datas: List[JointData]) -> JointData:
    """Merges multiple JointData objects along the time sample dimension.

    Args:
        joint_datas (List[JointData]): The JointData objects to merge.

    Returns:
        JointData: The merged joint data.
    """
    return JointData(
        joint_positions=np.vstack([data.joint_positions for data in joint_datas]),
        joint_velocities=np.vstack([data.joint_velocities for data in joint_datas]),
        joint_accelerations=np.vstack(
            [data.joint_accelerations for data in joint_datas]
        ),
        joint_torques=np.vstack([data.joint_torques for data in joint_datas]),
        sample_times_s=np.hstack([data.sample_times_s for data in joint_datas]),
    )
