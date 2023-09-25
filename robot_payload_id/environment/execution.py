from pathlib import Path
from typing import List, Optional

import numpy as np

from pydrake.all import PiecewisePolynomial, Simulator

from robot_payload_id.utils import (
    ArmComponents,
    JointData,
    gather_joint_log_data,
    merge_joint_datas,
)


def generate_sinusoidal_traj(
    initial_positions: List[float],
    base_frequency_idx: int,
    amplitude: float = 1.0,
    duration_s: float = 10.0,
    timestep_s: float = 0.01,
) -> PiecewisePolynomial:
    """Generates a sinusoidal joint space trajectory for a robotic manipulator, where
    every joint moves at a different frequency.

    Args:
        initial_positions (List[float]): The initial joint positions. The length of the
        lists corresponds to the number of joints.
        base_frequency_idx (int): The base frequency index to use.
        amplitude (float, optional): The sinusoidal amplitude.
        duration_s (float, optional): The trajectory duration in seconds.
        timestep_s (float, optional): The time step/ sample period for converting the
        continuous trajectory into a piecewise polynomial.

    Returns:
        PiecewisePolynomial: The generated trajectory.
    """
    num_joints = len(initial_positions)
    sample_times_s = np.arange(0, duration_s, timestep_s)
    joint_positions = []
    for i in range(num_joints):
        joint_positions.append(
            initial_positions[i]
            + amplitude * np.sin((base_frequency_idx + i) * 0.5 * sample_times_s)
        )
    joint_positions = np.array(joint_positions).reshape((num_joints, -1))
    return PiecewisePolynomial.CubicShapePreserving(
        sample_times_s, joint_positions, zero_end_point_derivatives=True
    )


def generate_joint_data(
    arm_components: ArmComponents,
    joint_trajectory: PiecewisePolynomial,
    trajectory_duration_s: float,
) -> JointData:
    """Generate joint data by executing the `joint_trajectory` on the robot.

    Args:
        arm_components (ArmComponents): The components of the robotic system that is
        used for the data collection.
        joint_trajectory (PiecewisePolynomial): The joint space trajectory to execute.
        trajectory_duration_s (float): The duration of the joint space trajectory in
        seconds.

    Returns:
        JointData: The generated joint data.
    """
    arm_components.trajectory_source.UpdateTrajectory(joint_trajectory)

    simulator = Simulator(arm_components.diagram)
    plant_context = arm_components.plant.GetMyContextFromRoot(
        simulator.get_mutable_context()
    )

    # Set starting positions (On a real robot, we need a controller to get us here)
    arm_components.plant.SetPositions(
        plant_context,
        arm_components.plant.GetModelInstanceByName("arm"),
        np.array(joint_trajectory.value(0)),
    )
    simulator.AdvanceTo(trajectory_duration_s)

    state_log = arm_components.state_logger.FindLog(simulator.get_context())
    torque_log = arm_components.commanded_torque_logger.FindLog(simulator.get_context())
    data = gather_joint_log_data(state_log, torque_log)

    return data


def collect_joint_data(
    arm_components: ArmComponents,
    initial_joint_positions: List[float],
    num_trajectories: int,
    sinusoidal_amplitude: float,
    trajectory_duration_s: float,
    log_dir_path: Optional[Path] = None,
) -> JointData:
    """Collects joint data by executing a joint-wise sinusoidal trajectory on the
    robotic arm.

    Args:
        arm_components (ArmComponents): The components of the robotic arm system.
        initial_joint_positions (List[float]): The robot's initial joint positions.
        num_trajectories (int): The number of different trajectories to execute for
        data collection.
        sinusoidal_amplitude (float): The amplitude of the sinusoidal trajectories.
        trajectory_duration_s (float): The duration in seconds of each trajectory.
        log_dir_path (Optional[Path], optional): The path of the directory to log the
        meshcat recordings to (one recording per executed trajectory). The recordings
        are only saved if a path is specified.

    Returns:
        JointData: The collected joint data.
    """
    joint_datas = []
    for i in range(num_trajectories):
        q_traj = generate_sinusoidal_traj(
            initial_positions=initial_joint_positions,
            amplitude=sinusoidal_amplitude,
            duration_s=trajectory_duration_s,
            base_frequency_idx=i,
        )

        arm_components.meshcat_visualizer.StartRecording()
        data = generate_joint_data(
            arm_components=arm_components,
            joint_trajectory=q_traj,
            trajectory_duration_s=trajectory_duration_s,
        )

        arm_components.meshcat_visualizer.StopRecording()
        arm_components.meshcat_visualizer.PublishRecording()
        if log_dir_path is not None:
            html = arm_components.meshcat.StaticHtml()
            with open(log_dir_path / f"meshcat_{i}.html", "w") as f:
                f.write(html)

        joint_datas.append(data)

    merged_joint_data = merge_joint_datas(joint_datas)
    return merged_joint_data
