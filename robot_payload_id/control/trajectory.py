import copy

import numpy as np

from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from iiwa_setup.motion_planning import (
    plan_unconstrained_gcs_path_start_to_goal,
    reparameterize_with_toppra,
)
from pydrake.all import (
    CompositeTrajectory,
    Context,
    DerivativeTrajectory,
    DiscreteValues,
    LeafSystem,
    PathParameterizedTrajectory,
    PiecewisePolynomial,
    Trajectory,
    TrajectorySource,
)

from robot_payload_id.utils import FourierSeriesTrajectoryAttributes


class FourierSeriesTrajectory(Trajectory):
    """
    Represents the following Fourier series trajectory:
    qᵢ(t) = ∑ₗ₌₁ᴺᵢ (aₗⁱ sin(ωₙ lt) + bₗⁱ cos(ωₙ lt)) + qᵢ₀
    q̇ᵢ(t) = ∑ₗ₌₁ᴺᵢ (aₗⁱ ωₙ l cos(ωₙ lt) - bₗⁱ ωₙ l sin(ωₙ lt))
    q̈ᵢ(t) = ∑ₗ₌₁ᴺᵢ (-aₗⁱ ωₙ^2 l^2 sin(ωₙ lt) - bₗⁱ ωₙ^2 l^2 cos(ωₙ lt))
    """

    def __init__(
        self,
        traj_attrs: FourierSeriesTrajectoryAttributes,
        time_horizon: float,
        traj_start_time: float = 0.0,
    ):
        """
        Args:
            traj_attrs: The Fourier series trajectory attributes.
            time_horizon: The time horizon of the trajectory in seconds.
            traj_start_time: The start time of the trajectory in seconds.
        """
        super().__init__()

        self._traj_attrs = traj_attrs
        self._time_horizon = time_horizon
        self._traj_start_time = traj_start_time
        self._a = traj_attrs.a_values
        self._b = traj_attrs.b_values
        self._q0 = traj_attrs.q0_values
        self._omega = traj_attrs.omega
        self._num_positions, self._num_terms = self._a.shape

        # Used for computing the positions, velocities, and accelerations
        self._l_values = np.arange(1, self._num_terms + 1)
        self._omega_l = self._omega * self._l_values

    def _compute_positions(self, time: np.ndarray) -> np.ndarray:
        """qᵢ(t) = ∑ₗ₌₁ᴺᵢ (aₗⁱ sin(ωₙ lt) + bₗⁱ cos(ωₙ lt)) + qᵢ₀"""
        cos_part = np.cos(self._omega_l * time)
        sin_part = np.sin(self._omega_l * time)
        return (
            np.einsum("ij,j->i", self._a, sin_part)
            + np.einsum("ij,j->i", self._b, cos_part)
            + self._q0
        )

    def _compute_velocities(self, time: np.ndarray) -> np.ndarray:
        """q̇ᵢ(t) = ∑ₗ₌₁ᴺᵢ (aₗⁱ ωₙ l cos(ωₙ lt) - bₗⁱ ωₙ l sin(ωₙ lt))"""
        cos_part = self._omega_l * np.cos(self._omega_l * time)
        sin_part = self._omega_l * np.sin(self._omega_l * time)
        return np.einsum("il,l->i", self._a, cos_part) - np.einsum(
            "il,l->i", self._b, sin_part
        )

    def _compute_accelerations(self, time: np.ndarray) -> np.ndarray:
        """q̈ᵢ(t) = ∑ₗ₌₁ᴺᵢ (-aₗⁱ ωₙ^2 l^2 sin(ωₙ lt) - bₗⁱ ωₙ^2 l^2 cos(ωₙ lt))"""
        sin_part = ((self._omega_l) ** 2) * np.sin(self._omega_l * time)
        cos_part = ((self._omega_l) ** 2) * np.cos(self._omega_l * time)
        return np.einsum("il,l->i", -self._a, sin_part) + np.einsum(
            "il,l->i", -self._b, cos_part
        )

    def rows(self) -> int:
        return self._num_positions

    def cols(self) -> int:
        return 1

    def start_time(self) -> float:
        return self._traj_start_time

    def end_time(self) -> float:
        return self._traj_start_time + self._time_horizon

    def value(self, time: float) -> np.ndarray:
        return self.DoEvalDerivative(time, derivative_order=0)

    def do_has_derivative(self):
        return True

    def DoEvalDerivative(self, time: float, derivative_order: int) -> np.ndarray:
        assert derivative_order < 3
        traj_time = time - self._traj_start_time
        if derivative_order == 0:
            return self._compute_positions(traj_time)
        if derivative_order == 1:
            return self._compute_velocities(traj_time)
        if derivative_order == 2:
            return self._compute_accelerations(traj_time)

    def DoMakeDerivative(self, derivative_order: int) -> "FourierSeriesTrajectory":
        return DerivativeTrajectory(nominal=self, derivative_order=derivative_order)

    def Clone(self) -> "FourierSeriesTrajectory":
        return FourierSeriesTrajectory(
            traj_attrs=copy.deepcopy(self._traj_attrs),
            time_horizon=self._time_horizon,
            traj_start_time=self._traj_start_time,
        )


class ExcitationTrajectorySourceInitializer(LeafSystem):
    """
    A system that initializes the trajectory source with the start and excitation
    trajectories. The start trajectory is an unconstrained GCS trajectory from the start
    positions to the excitation trajectory start positions.
    """

    def __init__(
        self,
        station: IiwaHardwareStationDiagram,
        excitaiton_traj: Trajectory,
        traj_source: TrajectorySource,
        start_traj_limit_fraction: float = 0.2,
    ):
        """
        Args:
            station: The IIWA hardware station diagram.
            excitaiton_traj: The excitation trajectory.
            traj_source: The trajectory source to initialize.
            start_traj_limit_fraction: The fraction of the velocity and acceleration
                limits to use when retiming the start trajectory with Toppra.
        """
        super().__init__()

        self._station = station
        self._excitation_traj = excitaiton_traj
        self._traj_source = traj_source
        self._start_traj_limit_fraction = start_traj_limit_fraction

        num_joint_positions = station.get_iiwa_controller_plant().num_positions()
        self._iiwa_position_measured_input_port = self.DeclareVectorInputPort(
            "iiwa.position_measured", num_joint_positions
        )
        self.DeclareInitializationDiscreteUpdateEvent(self._initialize_discrete_state)

    def _initialize_discrete_state(
        self, context: Context, discrete_values: DiscreteValues
    ) -> None:
        iiwa_controller_plant = self._station.get_iiwa_controller_plant()
        q_current = self._iiwa_position_measured_input_port.Eval(context)

        # Stay at the current pose for 5s before moving.
        pause_traj_end_time = 5.0
        initial_pause_traj = PiecewisePolynomial.ZeroOrderHold(
            breaks=[0.0, pause_traj_end_time],
            samples=np.stack([q_current, q_current], axis=1),
        )

        start_traj = plan_unconstrained_gcs_path_start_to_goal(
            plant=iiwa_controller_plant,
            q_start=q_current,
            q_goal=self._excitation_traj.value(0.0),
        )
        start_traj_retimed = reparameterize_with_toppra(
            trajectory=start_traj,
            plant=iiwa_controller_plant,
            velocity_limits=np.min(
                [
                    np.abs(iiwa_controller_plant.GetVelocityLowerLimits()),
                    np.abs(iiwa_controller_plant.GetVelocityUpperLimits()),
                ],
                axis=0,
            )
            * self._start_traj_limit_fraction,
            acceleration_limits=np.min(
                [
                    np.abs(iiwa_controller_plant.GetAccelerationLowerLimits()),
                    np.abs(iiwa_controller_plant.GetAccelerationUpperLimits()),
                ],
                axis=0,
            )
            * self._start_traj_limit_fraction,
        )

        # Delay the start trajectory to start after the pause trajectory ends.
        start_traj_time = PiecewisePolynomial().FirstOrderHold(
            [0.0, start_traj_retimed.end_time()],
            [[0.0, start_traj_retimed.end_time()]],
        )
        start_traj_time.shiftRight(pause_traj_end_time)
        start_traj_delayed = PathParameterizedTrajectory(
            path=start_traj_retimed, time_scaling=start_traj_time
        )

        # Delay the excitation trajectory to start after the start trajectory ends.
        self._excitation_traj_start_time = start_traj_delayed.end_time()
        excitation_traj_time = PiecewisePolynomial().FirstOrderHold(
            [0.0, self._excitation_traj.end_time()],
            [[0.0, self._excitation_traj.end_time()]],
        )
        excitation_traj_time.shiftRight(self._excitation_traj_start_time)
        excitation_traj_delayed = PathParameterizedTrajectory(
            path=self._excitation_traj, time_scaling=excitation_traj_time
        )

        self._combined_traj = CompositeTrajectory(
            [initial_pause_traj, start_traj_delayed, excitation_traj_delayed]
        )
        self._traj_source.UpdateTrajectory(self._combined_traj)

    def get_excitation_traj_start_time(self) -> float:
        return self._excitation_traj_start_time

    def get_end_time(self) -> float:
        return self._combined_traj.end_time()
