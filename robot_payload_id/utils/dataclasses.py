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
    """
    Variables for the joint positions of shape (N,) where N is the number of joints.
    """
    q_dot: np.ndarray
    """Variables for the joint velocities of shape (N,)."""
    q_ddot: np.ndarray
    """Variables for the joint accelerations of shape (N,)."""
    tau: np.ndarray
    """Variables for the joint torques of shape (N,)."""


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

    # Lumped parameters that the dynamics are linear in (h=mc, I=mG)
    hx: Union[sym.Variable, float, None] = None
    hy: Union[sym.Variable, float, None] = None
    hz: Union[sym.Variable, float, None] = None
    Ixx: Union[sym.Variable, float, None] = None
    Ixy: Union[sym.Variable, float, None] = None
    Ixz: Union[sym.Variable, float, None] = None
    Iyy: Union[sym.Variable, float, None] = None
    Iyz: Union[sym.Variable, float, None] = None
    Izz: Union[sym.Variable, float, None] = None

    def get_base_param_list(self) -> List[sym.Variable]:
        """
        Returns a list of Drake's base inertial parameters for the joint.
        The output is of form [m, cx, cy, cz, Gxx, Gxy, Gxz, Gyy, Gyz, Gzz], where m is
        the mass, cx, cy, and cz are the center of mass, and Gxx, Gxy, Gxz, Gyy, Gyz,
        and Gzz are the rotational unit inertia matrix elements. Note that elements
        that are None are not included in the output.
        """
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

    def get_lumped_param_list(self) -> List[sym.Variable]:
        """
        Returns a list of the lumped parameters that the dynamics are linear in for the
        joint.
        The output is of form [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz], where m is
        the mass, hx, hy, and hz are the mass times the center of mass, and Ixx, Ixy,
        Ixz, Iyy, Iyz, and Izz are the rotational inertia matrix elements. Note that
        elements that are None are not included in the output.
        """
        param_list = [self.m]
        if self.hx is not None:
            param_list.append(self.hx)
        if self.hy is not None:
            param_list.append(self.hy)
        if self.hz is not None:
            param_list.append(self.hz)
        if self.Ixx is not None:
            param_list.append(self.Ixx)
        if self.Ixy is not None:
            param_list.append(self.Ixy)
        if self.Ixz is not None:
            param_list.append(self.Ixz)
        if self.Iyy is not None:
            param_list.append(self.Iyy)
        if self.Iyz is not None:
            param_list.append(self.Iyz)
        if self.Izz is not None:
            param_list.append(self.Izz)
        return param_list

    def get_inertia_matrix(self) -> np.ndarray:
        """
        Returns the rotational inertia matrix of the joint.
        """
        return self.m * np.array(
            [
                [self.Gxx, self.Gxy, self.Gxz],
                [self.Gxy, self.Gyy, self.Gyz],
                [self.Gxz, self.Gyz, self.Gzz],
            ]
        )


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
