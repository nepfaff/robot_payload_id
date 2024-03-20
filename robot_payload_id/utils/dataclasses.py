import os

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pydrake.symbolic as sym

from pydrake.all import (
    BsplineBasis,
    BsplineTrajectory,
    Context,
    Diagram,
    Meshcat,
    MeshcatVisualizer,
    MultibodyPlant,
    Trajectory,
    TrajectorySource,
    VectorLogSink,
)

import wandb

from .inertia import change_inertia_reference_points_with_parallel_axis_theorem


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
    ft_sensor_measurements: Optional[np.ndarray] = None
    """The force-torque sensor measurements of shape (T, 6) and form
    [f_x, f_y, f_z, tau_x, tau_y, tau_z]."""

    def remove_duplicate_samples(self) -> "JointData":
        """
        Removes duplicate samples from the joint data. Returns a new JointData object
        rather than modifying the current one.
        """
        _, unique_indices = np.unique(self.sample_times_s, return_index=True)
        return JointData(
            joint_positions=self.joint_positions[unique_indices],
            joint_velocities=self.joint_velocities[unique_indices],
            joint_accelerations=self.joint_accelerations[unique_indices],
            joint_torques=self.joint_torques[unique_indices],
            sample_times_s=self.sample_times_s[unique_indices],
            ft_sensor_measurements=(
                None
                if self.ft_sensor_measurements is None
                else self.ft_sensor_measurements[unique_indices]
            ),
        )

    def save_to_disk(self, path: Path) -> None:
        """Saves the joint data to disk.

        Args:
            path: The path to save the joint data to.
        """
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "joint_positions.npy", self.joint_positions)
        np.save(path / "joint_velocities.npy", self.joint_velocities)
        np.save(path / "joint_accelerations.npy", self.joint_accelerations)
        np.save(path / "joint_torques.npy", self.joint_torques)
        np.save(path / "sample_times_s.npy", self.sample_times_s)
        if self.ft_sensor_measurements is not None:
            np.save(path / "ft_sensor_measurements.npy", self.ft_sensor_measurements)

    @classmethod
    def load_from_disk(cls, path: Path) -> "JointData":
        """Loads the joint data from disk.

        Args:
            path: The path to load the joint data from.

        Returns:
            The joint data.
        """
        ft_sensor_measurements_path = path / "ft_sensor_measurements.npy"
        return cls(
            joint_positions=np.load(path / "joint_positions.npy"),
            joint_velocities=np.load(path / "joint_velocities.npy"),
            joint_accelerations=np.load(path / "joint_accelerations.npy"),
            joint_torques=np.load(path / "joint_torques.npy"),
            sample_times_s=np.load(path / "sample_times_s.npy"),
            ft_sensor_measurements=(
                np.load(ft_sensor_measurements_path)
                if ft_sensor_measurements_path.exists()
                else None
            ),
        )

    @classmethod
    def from_trajectory(
        cls, trajectory: Trajectory, sample_times_s: np.ndarray
    ) -> "JointData":
        """
        Creates a JointData object from a Trajectory object.

        Args:
            trajectory: The trajectory object.
            sample_times_s: The sample times in seconds of shape (T,).

        Returns:
            The joint data.
        """
        return cls(
            joint_positions=np.asarray(
                [trajectory.value(t) for t in sample_times_s]
            ).squeeze(),
            joint_velocities=np.asarray(
                [trajectory.EvalDerivative(t, 1) for t in sample_times_s]
            ).squeeze(),
            joint_accelerations=np.asarray(
                [trajectory.EvalDerivative(t, 2) for t in sample_times_s]
            ).squeeze(),
            joint_torques=np.nan * np.ones_like(sample_times_s),
            sample_times_s=sample_times_s,
        )


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
    rotor_inertia: Union[sym.Variable, float, None] = None
    reflected_inertia: Union[sym.Variable, float, None] = None
    viscous_friction: Union[sym.Variable, float, None] = None
    dynamic_dry_friction: Union[sym.Variable, float, None] = None

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
        The output is of form [m, cx, cy, cz, Gxx, Gxy, Gxz, Gyy, Gyz, Gzz,
        rotor_inertia, viscous_friction, dynamic_dry_friction], where m is the mass, cx,
        cy, and cz are the center of mass, and Gxx, Gxy, Gxz, Gyy, Gyz, and Gzz are the
        rotational unit inertia matrix elements. Note that elements that are None are
        not included in the output.
        """
        assert not (
            self.rotor_inertia is not None and self.reflected_inertia is not None
        ), "Only one of rotor_inertia and reflected_inertia can be specified."

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
        if self.rotor_inertia is not None:
            param_list.append(self.rotor_inertia)
        if self.reflected_inertia is not None:
            param_list.append(self.reflected_inertia)
        if self.viscous_friction is not None:
            param_list.append(self.viscous_friction)
        if self.dynamic_dry_friction is not None:
            param_list.append(self.dynamic_dry_friction)
        return param_list

    def get_lumped_param_list(self) -> List[sym.Variable]:
        """
        Returns a list of the lumped parameters that the dynamics are linear in for the
        joint.
        The output is of form [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz,
        rotor_inertia, viscous_friction, dynamic_dry_friction], where m is the mass, hx,
        hy, and hz are the mass times the center of mass, and Ixx, Ixy, Ixz, Iyy, Iyz,
        and Izz are the rotational inertia matrix elements. Note that elements that are
        None are not included in the output.
        """
        assert not (
            self.rotor_inertia is not None and self.reflected_inertia is not None
        ), "Only one of rotor_inertia and reflected_inertia can be specified."

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
        if self.rotor_inertia is not None:
            param_list.append(self.rotor_inertia)
        if self.reflected_inertia is not None:
            param_list.append(self.reflected_inertia)
        if self.viscous_friction is not None:
            param_list.append(self.viscous_friction)
        if self.dynamic_dry_friction is not None:
            param_list.append(self.dynamic_dry_friction)
        return param_list

    def get_com(self) -> np.ndarray:
        """
        Returns the center of mass of the joint.
        """
        if self.cx is not None and self.cy is not None and self.cz is not None:
            return np.array([self.cx, self.cy, self.cz])
        elif self.hx is not None and self.hy is not None and self.hz is not None:
            return np.array([self.hx, self.hy, self.hz]) / self.m
        raise NotImplementedError(
            "Currently only supporting all center of mass or all mass times center "
            + "of mass."
        )

    def get_inertia_matrix(self) -> np.ndarray:
        """
        Returns the rotational inertia matrix of the joint.
        """
        if (
            self.Ixx is not None
            and self.Iyy is not None
            and self.Izz is not None
            and self.Ixy is not None
            and self.Ixz is not None
            and self.Iyz is not None
        ):
            return np.array(
                [
                    [self.Ixx, self.Ixy, self.Ixz],
                    [self.Ixy, self.Iyy, self.Iyz],
                    [self.Ixz, self.Iyz, self.Izz],
                ]
            )
        elif (
            self.Gxx is not None
            and self.Gyy is not None
            and self.Gzz is not None
            and self.Gxy is not None
            and self.Gxz is not None
            and self.Gyz is not None
        ):
            return self.m * np.array(
                [
                    [self.Gxx, self.Gxy, self.Gxz],
                    [self.Gxy, self.Gyy, self.Gyz],
                    [self.Gxz, self.Gyz, self.Gzz],
                ]
            )
        raise NotImplementedError(
            "Currently only supporting all rotational inertia or all rotational "
            + "unit inertia."
        )

    def get_pseudo_inertia_matrix(self) -> np.ndarray:
        """
        Returns the pseudo-inertia matrix of the joint.

        Pseudo-inertia ℙ(4):
            P(m, c, Iₘ) = [ Σ  h ]
                          [ hᵀ m ]
        where
            Iₘ is I_BBo_B (about link origin)/ rotational inertia
            c is p_BBcm (center of mass)
            Σ = 1/2 tr(Iₘ) eye(3) - Iₘ is the covariance/ density weighted 2nd moment
            h = m c the density weighted 1st moment
        """
        rotational_inertia = self.get_inertia_matrix()
        density_weighted_2nd_moment_matrix = (
            0.5 * np.trace(rotational_inertia) * np.eye(3) - rotational_inertia
        )
        if self.hx is not None and self.hy is not None and self.hz is not None:
            density_weighted_1st_moment = np.array([self.hx, self.hy, self.hz]).reshape(
                (3, 1)
            )
        elif self.cx is not None and self.cy is not None and self.cz is not None:
            density_weighted_1st_moment = self.m * np.array(
                [self.cx, self.cy, self.cz]
            ).reshape((3, 1))
        else:
            raise NotImplementedError(
                "Currently only supporting all center of mass or all mass times center "
                + "of mass."
            )
        return np.block(
            [
                [density_weighted_2nd_moment_matrix, density_weighted_1st_moment],
                [density_weighted_1st_moment.T, self.m],
            ]
        )

    def perturb(self, perturb_scale: float) -> None:
        """
        Perturbs the inertial parameters of the joint by a random scale factor.

        Args:
            perturb_scale: The maximum scale factor to perturb the parameters by.
        """
        # Get center of mass
        if self.cx is not None and self.cy is not None and self.cz is not None:
            com = np.array([self.cx, self.cy, self.cz])
            is_base_params = True
        elif self.hx is not None and self.hy is not None and self.hz is not None:
            com = np.array([self.hx, self.hy, self.hz]) / self.m
            is_base_params = False
        else:
            raise NotImplementedError(
                "Currently only supporting all center of mass or all mass times center "
                + "of mass."
            )

        # Express inertia with respect to center of mass
        rot_inertias_cm = change_inertia_reference_points_with_parallel_axis_theorem(
            I_BBa_B=self.get_inertia_matrix(),
            m_B=np.array([self.m]),
            p_BaBb_B=com,
            Ba_is_Bcm=False,
        )

        # Perturb inertial parameters
        # NOTE: Could also rotate inertias and add random point-mass noise
        mass_scale = 1.0 + perturb_scale * np.random.uniform()
        self.m *= mass_scale
        rot_inertias_cm *= mass_scale

        # Re-express inertia with respect to link origin
        rot_inertias = change_inertia_reference_points_with_parallel_axis_theorem(
            I_BBa_B=rot_inertias_cm,
            m_B=np.array([self.m]),
            p_BaBb_B=com,
            Ba_is_Bcm=True,
        )
        if is_base_params:
            self.Ixx, self.Iyy, self.Izz = rot_inertias.diagonal()
            self.Ixy, self.Ixz, self.Iyz = (
                rot_inertias[0, 1],
                rot_inertias[0, 2],
                rot_inertias[1, 2],
            )
        else:
            self.hx, self.hy, self.hz = self.m * com
            self.Gxx, self.Gyy, self.Gzz = rot_inertias.diagonal()
            self.Gxy, self.Gxz, self.Gyz = (
                rot_inertias[0, 1],
                rot_inertias[0, 2],
                rot_inertias[1, 2],
            ) / self.m


@dataclass
class ArmComponents:
    """
    A dataclass that contains all the robotic arm system components.
    """

    num_joints: int
    plant: MultibodyPlant
    diagram: Diagram = None
    trajectory_source: TrajectorySource = None
    state_logger: VectorLogSink = None
    commanded_torque_logger: VectorLogSink = None
    meshcat: Meshcat = None
    meshcat_visualizer: MeshcatVisualizer = None


@dataclass
class ArmPlantComponents:
    """
    A dataclass that contains everything that goes with a plant of a robotics arm.
    """

    plant: MultibodyPlant
    """The plant."""
    plant_context: Optional[Context] = None
    """The context of the plant."""
    parameters: Optional[List[JointParameters]] = None
    """The plant parameters."""
    state_variables: Optional[SymJointStateVariables] = None
    """The symbolic plant state variables."""


@dataclass
class BsplineTrajectoryAttributes:
    """A data class to hold the attributes of a B-spline trajectory."""

    spline_order: int
    """The order of the B-spline basis to use."""
    control_points: np.ndarray
    """The control points of the B-spline trajectory of shape
    (num_joints, num_control_points)."""
    knots: np.ndarray
    """The knots of the B-spline basis of shape (num_knots,)."""

    @classmethod
    def from_bspline_trajectory(cls, traj: BsplineTrajectory) -> None:
        """Sets the attributes from a B-spline trajectory."""
        assert traj.start_time() == 0.0, "Trajectory must start at time 0!"
        return cls(
            spline_order=traj.basis().order(),
            control_points=traj.control_points(),
            knots=np.array(traj.basis().knots()) * traj.end_time(),
        )

    def log(self, logging_path: Optional[Path] = None) -> None:
        """Logs the B-spline trajectory attributes to wandb. If logging_path is not
        None, then the attributes are also saved to disk."""
        if wandb.run is not None:
            # NOTE: This overwrites the previous log
            np.save(
                os.path.join(wandb.run.dir, "spline_order.npy"),
                np.array([self.spline_order]),
            )
            np.save(
                os.path.join(wandb.run.dir, "control_points.npy"), self.control_points
            )
            np.save(os.path.join(wandb.run.dir, "knots.npy"), self.knots)
        if logging_path is not None:
            np.save(logging_path / "spline_order.npy", np.array([self.spline_order]))
            np.save(logging_path / "control_points.npy", self.control_points)
            np.save(logging_path / "knots.npy", self.knots)

    @classmethod
    def load(cls, path: Path) -> "BsplineTrajectoryAttributes":
        """Loads the B-spline trajectory attributes from disk."""
        return cls(
            spline_order=int(np.load(path / "spline_order.npy")[0]),
            control_points=np.load(path / "control_points.npy"),
            knots=np.load(path / "knots.npy"),
        )

    def to_bspline_trajectory(self) -> BsplineTrajectory:
        """Converts the attributes to a B-spline trajectory."""
        return BsplineTrajectory(
            basis=BsplineBasis(order=self.spline_order, knots=self.knots),
            control_points=self.control_points,
        )


@dataclass
class FourierSeriesTrajectoryAttributes:
    """A data class to hold the attributes of a finite Fourier series trajectory."""

    a_values: int
    """The `a` parameters of shape (num_joints, num_fourier_terms)."""
    b_values: np.ndarray
    """The `b` parameters of shape (num_joints, num_fourier_terms)."""
    q0_values: np.ndarray
    """The `q0` parameters of shape (num_joints,)."""
    omega: float
    """The frequency of the trajectory in radians."""

    def log(self, logging_path: Optional[Path] = None) -> None:
        """Logs the trajectory attributes to wandb. If logging_path is not None, then
        the attributes are also saved to disk."""
        if wandb.run is not None:
            # NOTE: This overwrites the previous log
            np.save(os.path.join(wandb.run.dir, "a_value.npy"), self.a_values)
            np.save(os.path.join(wandb.run.dir, "b_value.npy"), self.b_values)
            np.save(os.path.join(wandb.run.dir, "q0_value.npy"), self.q0_values)
            np.save(os.path.join(wandb.run.dir, "omega.npy"), np.array([self.omega]))
        if logging_path is not None:
            np.save(logging_path / "a_value.npy", self.a_values)
            np.save(logging_path / "b_value.npy", self.b_values)
            np.save(logging_path / "q0_value.npy", self.q0_values)
            np.save(logging_path / "omega.npy", np.array([self.omega]))

    @classmethod
    def from_flattened_data(
        cls,
        a_values: np.ndarray,
        b_values: np.ndarray,
        q0_values: np.ndarray,
        omega: float,
        num_joints: int,
    ) -> "FourierSeriesTrajectoryAttributes":
        """Creates a FourierSeriesTrajectoryAttributes object from flattened data."""
        return cls(
            a_values=a_values.reshape((num_joints, -1), order="F"),
            b_values=b_values.reshape((num_joints, -1), order="F"),
            q0_values=q0_values,
            omega=omega,
        )

    def to_flattened_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Converts the attributes to flattened data.

        Returns: A tuple (a_values, b_values, q0_values, omega) of flattened arrays.
        """
        return (
            self.a_values.flatten(order="F"),
            self.b_values.flatten(order="F"),
            self.q0_values,
            self.omega,
        )

    @classmethod
    def load(
        cls, path: Path, num_joints: Optional[int] = None
    ) -> "FourierSeriesTrajectoryAttributes":
        """Loads the trajectory attributes from disk."""
        a_values = np.load(path / "a_value.npy")
        b_values = np.load(path / "b_value.npy")
        q0_values = np.load(path / "q0_value.npy")
        omega = float(np.load(path / "omega.npy")[0])

        if len(a_values.shape) == 1:
            assert (
                num_joints is not None
            ), "num_joints must be provided when loading flattened data!"
            return cls.from_flattened_data(
                a_values=a_values,
                b_values=b_values,
                q0_values=q0_values,
                omega=omega,
                num_joints=num_joints,
            )

        return cls(
            a_values=a_values, b_values=b_values, q0_values=q0_values, omega=omega
        )
