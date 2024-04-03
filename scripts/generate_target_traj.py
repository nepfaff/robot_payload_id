### Script for generating a target trajectory for moving between two waypoints.
### This trajectory is used for testing payload parameter excitation while staying close
### to a desired trajectory.

import argparse
import os

from pathlib import Path

import numpy as np

from create_bspline_traj_from_fourier_series import construct_and_sample_traj_sym
from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from iiwa_setup.motion_planning import (
    plan_unconstrained_gcs_path_start_to_goal,
    reparameterize_with_toppra,
)
from manipulation.station import LoadScenario
from pydrake.all import KinematicTrajectoryOptimization, Solve

from robot_payload_id.utils import BsplineTrajectoryAttributes, JointData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario_path",
        type=str,
        default="models/iiwa_scenario.yaml",
        help="Path to the scenario file. This must contain an iiwa model named 'iiwa'.",
    )
    parser.add_argument(
        "--start_q",
        type=float,
        nargs="+",
        default=[-1.57, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0],
        help="The starting joint positions.",
    )
    parser.add_argument(
        "--end_q",
        type=float,
        nargs="+",
        default=[1.57, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0],
        help="The goal joint positions.",
    )
    parser.add_argument(
        "--vel_limit_fraction",
        type=float,
        default=0.75,
        help="The fraction of the velocity limits to use for reparameterization.",
    )
    parser.add_argument(
        "--acc_limit_fraction",
        type=float,
        default=0.75,
        help="The fraction of the acceleration limits to use for reparameterization.",
    )
    parser.add_argument(
        "--num_control_points",
        type=int,
        default=30,
        help="The number of control points to use for the B-spline trajectory.",
    )
    parser.add_argument(
        "--spline_order",
        type=int,
        default=4,
        help="The order of the B-spline basis functions.",
    )
    parser.add_argument(
        "--sample_period_s",
        type=float,
        default=1e-3,
        help="The period at which to sample the trajectory for fitting the B-spline.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="The directory to save the B-spline trajectory to.",
    )
    args = parser.parse_args()
    scenario_path = args.scenario_path
    start_q = args.start_q
    end_q = args.end_q
    vel_limit_fraction = args.vel_limit_fraction
    acc_limit_fraction = args.acc_limit_fraction
    num_control_points = args.num_control_points
    spline_order = args.spline_order
    sample_period_s = args.sample_period_s
    save_dir = args.save_dir

    scenario = LoadScenario(filename=scenario_path)
    has_wsg = "wsg" in scenario.model_drivers.keys()
    station: IiwaHardwareStationDiagram = IiwaHardwareStationDiagram(
        scenario=scenario,
        has_wsg=has_wsg,
        use_hardware=False,
        control_mode=scenario.model_drivers["iiwa"].control_mode,
        package_xmls=[os.path.abspath("models/package.xml")],
    )
    iiwa_controller_plant = station.get_iiwa_controller_plant()
    num_positions = iiwa_controller_plant.num_positions()

    # Generate the trajectory
    traj = plan_unconstrained_gcs_path_start_to_goal(
        plant=iiwa_controller_plant,
        q_start=start_q,
        q_goal=end_q,
    )
    assert traj is not None, "Failed to plan a trajectory."
    traj_retimed = reparameterize_with_toppra(
        trajectory=traj,
        plant=iiwa_controller_plant,
        velocity_limits=np.min(
            [
                np.abs(iiwa_controller_plant.GetVelocityLowerLimits()),
                np.abs(iiwa_controller_plant.GetVelocityUpperLimits()),
            ],
            axis=0,
        )
        * vel_limit_fraction,
        acceleration_limits=np.min(
            [
                np.abs(iiwa_controller_plant.GetAccelerationLowerLimits()),
                np.abs(iiwa_controller_plant.GetAccelerationUpperLimits()),
            ],
            axis=0,
        )
        * acc_limit_fraction,
    )

    traj_duration = traj_retimed.end_time()

    # Fit a B-spline to the trajectory
    num_sample_points = int(traj_duration // sample_period_s)
    joint_data_gt: JointData = JointData.from_trajectory(
        trajectory=traj_retimed,
        sample_times_s=np.linspace(0, traj_duration, num_sample_points),
    )
    trajopt = KinematicTrajectoryOptimization(
        num_positions=num_positions,
        num_control_points=num_control_points,
        spline_order=spline_order,
        duration=traj_duration,
    )
    trajopt.AddPositionBounds(
        lb=iiwa_controller_plant.GetPositionLowerLimits(),
        ub=iiwa_controller_plant.GetPositionUpperLimits(),
    )
    trajopt.AddVelocityBounds(
        lb=iiwa_controller_plant.GetVelocityLowerLimits(),
        ub=iiwa_controller_plant.GetVelocityUpperLimits(),
    )
    trajopt.AddAccelerationBounds(
        lb=iiwa_controller_plant.GetAccelerationLowerLimits(),
        ub=iiwa_controller_plant.GetAccelerationUpperLimits(),
    )
    # The jerk limits are rather arbitrary and are mainly there to prevent
    # acceleration discontinuities that lead to commanded torque discontinuities
    # when using inverse dynamics control
    trajopt.AddJerkBounds(lb=-np.full(num_positions, 25), ub=np.full(num_positions, 25))

    prog = trajopt.get_mutable_prog()
    joint_data_sym = construct_and_sample_traj_sym(
        var_values=prog.decision_variables(),
        num_timesteps=num_sample_points,
        num_control_points=num_control_points,
        num_joints=num_positions,
        trajopt=trajopt,
    )
    for pos, pos_gt in zip(
        joint_data_sym.joint_positions,
        joint_data_gt.joint_positions,
    ):
        prog.AddQuadraticCost((pos - pos_gt).T @ (pos - pos_gt))

    print("Solving the problem...")
    result = Solve(prog)
    print(
        "Optimal cost (increase number of control points to decrease): "
        + f"{result.get_optimal_cost()}"
    )

    # Save the trajectory
    save_dir.mkdir(parents=True, exist_ok=True)
    scaled_knots = np.array(trajopt.basis().knots()) * traj_duration
    control_points = result.GetSolution(trajopt.control_points())
    BsplineTrajectoryAttributes(
        spline_order=trajopt.basis().order(),
        control_points=control_points,
        knots=scaled_knots,
    ).log(save_dir)


if __name__ == "__main__":
    main()
