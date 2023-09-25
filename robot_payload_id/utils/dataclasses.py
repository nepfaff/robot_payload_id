from dataclasses import dataclass
from typing import List, Union

import numpy as np
import pydrake.symbolic as sym

from pydrake.all import (
    Context,
    Diagram,
    Meshcat,
    MeshcatVisualizer,
    MultibodyPlant,
    TrajectorySource,
    VectorLogSink,
)


@dataclass
class JointData:
    """
    A dataclass for joint data. For all properties, N refers to the number of joints and
    T refers to the number of samples.
    """

    joint_positions: np.ndarray
    """The measured joint positions of shape (T, N)."""
    joint_velocities: np.ndarray
    """The measured joint velocities of shape (T, N)."""
    joint_accelerations: np.ndarray
    """The estimated joint accelerations of shape (T, N)."""
    joint_torques: np.ndarray
    """The measured joint torques of shape (T, N)."""
    sample_times_s: np.ndarray
    """The sample times in seconds of shape (T,)."""


@dataclass
class SymJointStateVariables:
    """
    A dataclass for symbolic joint state variables.
    """

    q: np.ndarray
    """Variables for the joint positions."""
    q_dot: np.ndarray
    """Variables for the joint velocities."""
    q_ddot: np.ndarray
    """Variables for the joint accelerations."""
    tau: np.ndarray
    """Variables for the joint torques."""


@dataclass
class JointParameters:
    """
    A dataclass for joint inertial parameters. All parameters should either by symbolic
    or numeric but not mixed. Some of the parameters are optional as not every system
    will have all possible parameters.
    """

    m: Union[sym.Variable, float]
    cx: Union[sym.Variable, float, None] = None
    cy: Union[sym.Variable, float, None] = None
    cz: Union[sym.Variable, float, None] = None
    Gxx: Union[sym.Variable, float, None] = None
    Gxy: Union[sym.Variable, float, None] = None
    Gxz: Union[sym.Variable, float, None] = None
    Gyy: Union[sym.Variable, float, None] = None
    Gyz: Union[sym.Variable, float, None] = None
    Gzz: Union[sym.Variable, float, None] = None

    def get_param_list(self) -> List[sym.Variable]:
        param_list = [self.m]
        if self.cx is not None:
            param_list.append(self.cx)
        if self.cy is not None:
            param_list.append(self.cy)
        if self.cz is not None:
            param_list.append(self.cz)
        if self.Gxx is not None:
            param_list.append(self.Gxx)
        if self.Gxy is not None:
            param_list.append(self.Gxy)
        if self.Gxz is not None:
            param_list.append(self.Gxz)
        if self.Gyy is not None:
            param_list.append(self.Gyy)
        if self.Gyz is not None:
            param_list.append(self.Gyz)
        if self.Gzz is not None:
            param_list.append(self.Gzz)
        return param_list


@dataclass
class ArmComponents:
    """
    A dataclass that contains all the robotic arm system components.
    """

    num_joints: int
    diagram: Diagram
    plant: MultibodyPlant
    trajectory_source: TrajectorySource
    state_logger: VectorLogSink
    commanded_torque_logger: VectorLogSink
    meshcat: Meshcat
    meshcat_visualizer: MeshcatVisualizer


@dataclass
class SymbolicArmPlantComponents:
    """
    A dataclass that contains everything that goes with a symbolic plant of a robotics
    arm.
    """

    plant: MultibodyPlant
    """The symbolic plant."""
    plant_context: Context
    """The context of the symbolic plant."""
    state_variables: SymJointStateVariables
    """The symbolic plant state variables."""
    parameters: List[JointParameters]
    """The symbolic plant parameters."""
