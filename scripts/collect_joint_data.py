import argparse
import logging
import os

from pathlib import Path

import numpy as np

from iiwa_setup.controllers import (
    InverseDynamicsControllerWithGravityCompensationCancellation,
)
from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    BsplineBasis,
    BsplineTrajectory,
    ConstantVectorSource,
    Demultiplexer,
    DiagramBuilder,
    MeshcatVisualizer,
    PiecewisePolynomial,
    Simulator,
    TrajectorySource,
    VectorLogSink,
)

from robot_payload_id.control import (
    ExcitationTrajectorySourceInitializer,
    FourierSeriesTrajectory,
)
from robot_payload_id.utils import (
    BsplineTrajectoryAttributes,
    FourierSeriesTrajectoryAttributes,
    JointData,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario_path",
        type=str,
        required=True,
        help="Path to the scenario file. This must contain an iiwa model named 'iiwa'.",
    )
    parser.add_argument(
        "--traj_parameter_path",
        type=Path,
        required=True,
        help="Path to the trajectory parameter folder. The folder must contain "
        + "'a_value.npy', 'b_value.npy', and 'q0_value.npy' or 'control_points.npy', "
        + "'knots.npy', and 'spline_order.npy'.",
    )
    parser.add_argument(
        "--save_data_path",
        type=Path,
        help="Path to save the data to.",
    )
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )
    parser.add_argument(
        "--time_horizon",
        type=float,
        default=10.0,
        help="The time horizon/ duration of the trajectory. Only used for Fourier "
        + "series trajectories.",
    )
    parser.add_argument(
        "--only_log_excitation_traj_data",
        action="store_true",
        help="Whether to only log data during excitation trajectory execution rather "
        + "than during the entire runtime.",
    )
    parser.add_argument(
        "--duration_to_remove_at_start",
        type=float,
        default=1.0,
        help="The duration to remove from the start of the trajectory. Only used if "
        + "`only_log_excitation_traj_data` is True. This is useful as there might be "
        + "significant noise when transitioning from the slow start trajectory to the "
        + "fast excitation trajectory.",
    )
    parser.add_argument(
        "--html_path",
        type=Path,
        default=None,
        help="Path to save the html file of the visualization.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    scenario_path = args.scenario_path
    traj_parameter_path = args.traj_parameter_path
    save_data_path = args.save_data_path
    use_hardware = args.use_hardware
    time_horizon = args.time_horizon
    only_log_excitation_traj_data = args.only_log_excitation_traj_data
    duration_to_remove_at_start = args.duration_to_remove_at_start
    html_path = args.html_path

    builder = DiagramBuilder()
    scenario = LoadScenario(filename=scenario_path)
    has_wsg = "wsg" in scenario.model_drivers.keys()
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "iiwa_hardware_station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            has_wsg=has_wsg,
            use_hardware=use_hardware,
            control_mode=scenario.model_drivers["iiwa"].control_mode,
            package_xmls=[os.path.abspath("models/package.xml")],
        ),
    )

    # Load trajectory parameters
    is_fourier_series = os.path.exists(traj_parameter_path / "a_value.npy")
    if is_fourier_series:
        traj_attrs = FourierSeriesTrajectoryAttributes.load(traj_parameter_path)
        excitation_traj = FourierSeriesTrajectory(
            traj_attrs=traj_attrs,
            time_horizon=time_horizon,
        )
    else:
        traj_attrs = BsplineTrajectoryAttributes.load(traj_parameter_path)
        excitation_traj = BsplineTrajectory(
            basis=BsplineBasis(order=traj_attrs.spline_order, knots=traj_attrs.knots),
            control_points=traj_attrs.control_points,
        )

    # Placeholder trajectory
    traj_source: TrajectorySource = builder.AddNamedSystem(
        "trajectory_source",
        TrajectorySource(
            trajectory=PiecewisePolynomial.ZeroOrderHold(
                [0.0, 1.0], np.zeros((len(excitation_traj.value(0.0)), 2))
            ),
            output_derivative_order=2,
        ),
    )
    traj_source_initializer: ExcitationTrajectorySourceInitializer = (
        builder.AddNamedSystem(
            "trajectory_source_initializer",
            ExcitationTrajectorySourceInitializer(
                station=station,
                excitaiton_traj=excitation_traj,
                traj_source=traj_source,
            ),
        )
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        traj_source_initializer.GetInputPort("iiwa.position_measured"),
    )

    # Add controller
    controler_plant = station.get_iiwa_controller_plant()
    num_positions = controler_plant.num_positions()
    controller = builder.AddNamedSystem(
        "controller",
        InverseDynamicsControllerWithGravityCompensationCancellation(
            station=station,
            scenario=scenario,
            kp_gains=np.full(7, 600),
            damping_ratios=np.full(7, 0.2),
        ),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"),
        controller.GetInputPort("iiwa.state_estimated"),
    )
    desired_state_acceleration_demux: Demultiplexer = builder.AddNamedSystem(
        "desired_state_acceleration_demux",
        Demultiplexer(output_ports_sizes=[num_positions * 2, num_positions]),
    )
    builder.Connect(
        traj_source.get_output_port(), desired_state_acceleration_demux.get_input_port()
    )
    builder.Connect(
        desired_state_acceleration_demux.get_output_port(0),
        controller.GetInputPort("iiwa.desired_state"),
    )
    builder.Connect(
        desired_state_acceleration_demux.get_output_port(1),
        controller.GetInputPort("iiwa.desired_accelerations"),
    )
    builder.Connect(controller.get_output_port(), station.GetInputPort("iiwa.torque"))

    # TODO: Decide what to do there. Do we want to open for 5s and then grip tightly?
    # If yes, then force control would be better than position control.
    if has_wsg:
        wsg_const_pos_source: ConstantVectorSource = builder.AddNamedSystem(
            "wsg_position_source", ConstantVectorSource(source_value=0.05 * np.ones(1))
        )
        builder.Connect(
            wsg_const_pos_source.get_output_port(), station.GetInputPort("wsg.position")
        )

    # Add data loggers
    desired_state_demux = builder.AddNamedSystem(
        "desired_state_demux", Demultiplexer(output_ports_sizes=[7, 7])
    )
    builder.Connect(
        desired_state_acceleration_demux.get_output_port(0),
        desired_state_demux.get_input_port(),
    )
    commanded_position_logger: VectorLogSink = builder.AddNamedSystem(
        "commanded_position_logger", VectorLogSink(num_positions)
    )
    measured_position_logger: VectorLogSink = builder.AddNamedSystem(
        "measured_position_logger", VectorLogSink(num_positions)
    )
    measured_velocity_logger: VectorLogSink = builder.AddNamedSystem(
        "measured_velocity_logger", VectorLogSink(num_positions)
    )
    measured_torque_logger: VectorLogSink = builder.AddNamedSystem(
        "measured_torque_logger", VectorLogSink(num_positions)
    )
    builder.Connect(
        desired_state_demux.get_output_port(0),
        commanded_position_logger.get_input_port(),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        measured_position_logger.get_input_port(),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.velocity_estimated"),
        measured_velocity_logger.get_input_port(),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.torque_measured"),
        measured_torque_logger.get_input_port(),
    )

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    # Build and setup simulation
    diagram = builder.Build()
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)
    visualizer.StartRecording()
    simulator.Initialize()

    simulation_end_margin = 1.0
    simulator.AdvanceTo(traj_source_initializer.get_end_time() + simulation_end_margin)

    # Save data
    visualizer.StopRecording()
    visualizer.PublishRecording()
    if html_path is not None:
        html = station.internal_meshcat.StaticHtml()
        with open(html_path, "w") as f:
            f.write(html)

    commanded_position_data = (
        commanded_position_logger.FindLog(simulator.get_context()).data().T
    )
    measured_position_data = (
        measured_position_logger.FindLog(simulator.get_context()).data().T
    )
    measured_velocity_data = (
        measured_velocity_logger.FindLog(simulator.get_context()).data().T
    )
    measured_torque_data = (
        measured_torque_logger.FindLog(simulator.get_context()).data().T
    )
    sample_times_s = measured_position_logger.FindLog(
        simulator.get_context()
    ).sample_times()

    if only_log_excitation_traj_data:
        # Only keep data during excitation trajectory execution
        data_start_time = (
            traj_source_initializer.get_excitation_traj_start_time()
        ) + duration_to_remove_at_start
        excitation_traj_end_time = traj_source_initializer.get_end_time()
        excitation_traj_start_idx = np.argmax(sample_times_s >= data_start_time)
        data_end_time = np.argmax(sample_times_s >= excitation_traj_end_time)
        commanded_position_data = commanded_position_data[
            excitation_traj_start_idx:data_end_time
        ]
        measured_position_data = measured_position_data[
            excitation_traj_start_idx:data_end_time
        ]
        measured_velocity_data = measured_velocity_data[
            excitation_traj_start_idx:data_end_time
        ]
        measured_torque_data = measured_torque_data[
            excitation_traj_start_idx:data_end_time
        ]
        sample_times_s = sample_times_s[excitation_traj_start_idx:data_end_time]
        # Shift sample times to start at 0
        sample_times_s -= sample_times_s[0]

    # Save data
    if save_data_path is not None:
        joint_data = JointData(
            joint_positions=measured_position_data,
            joint_velocities=measured_velocity_data,
            joint_accelerations=np.array([np.nan]),
            joint_torques=measured_torque_data,
            sample_times_s=sample_times_s,
        )
        joint_data.save_to_disk(save_data_path)

    # Print tracking statistics
    print(
        "Mean absolute position tracking error per joint:",
        np.mean(np.abs(commanded_position_data - measured_position_data), axis=0),
    )
    print(
        "Max absolute position tracking error per joint:",
        np.max(np.abs(commanded_position_data - measured_position_data), axis=0),
    )


if __name__ == "__main__":
    main()
