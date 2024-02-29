import argparse
import logging
import os

from pathlib import Path

import numpy as np

from iiwa_setup.iiwa import IiwaHardwareStationDiagram
from manipulation.station import LoadScenario
from pydrake.all import (
    Adder,
    ApplySimulatorConfig,
    BsplineBasis,
    BsplineTrajectory,
    ConstantVectorSource,
    Demultiplexer,
    DiagramBuilder,
    Gain,
    InverseDynamics,
    InverseDynamicsController,
    MeshcatVisualizer,
    MultibodyPlant,
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
    filter_time_series_data,
    get_parser,
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
        required=True,
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
        help="The time horizon/ duration of the trajectory.",
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
    html_path = args.html_path

    builder = DiagramBuilder()
    scenario = LoadScenario(filename=scenario_path)
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "iiwa_hardware_station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            has_wsg=True,  # TODO: This script should work with and without gripper, based on the scenario file.
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

    # Add the iiwa controller
    # TODO: Add this entire controller system to `iiwa_setup` (inverse dynamics control
    # with gravity compensation cancellation).
    controler_plant = station.get_iiwa_controller_plant()
    num_positions = controler_plant.num_positions()
    torque_adder: Adder = builder.AddNamedSystem(
        "torque_adder", Adder(2, num_positions)
    )
    kp = np.full(num_positions, 600)
    damping_ratio = 0.2
    inverse_dynamics_controller: InverseDynamicsController = builder.AddNamedSystem(
        "inverse_dynamics_controller",
        InverseDynamicsController(
            controler_plant,
            kp=kp,
            ki=[1] * num_positions,
            kd=2 * damping_ratio * np.sqrt(kp),
            has_reference_acceleration=True,
        ),
    )
    state_acceleration_demux: Demultiplexer = builder.AddNamedSystem(
        "state_acceleration_demux",
        Demultiplexer(output_ports_sizes=[num_positions * 2, num_positions]),
    )
    builder.Connect(
        traj_source.get_output_port(), state_acceleration_demux.get_input_port()
    )
    builder.Connect(
        state_acceleration_demux.get_output_port(0),
        inverse_dynamics_controller.get_input_port_desired_state(),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"),
        inverse_dynamics_controller.get_input_port_estimated_state(),
    )
    builder.Connect(
        state_acceleration_demux.get_output_port(1),
        inverse_dynamics_controller.get_input_port_desired_acceleration(),
    )
    builder.Connect(
        inverse_dynamics_controller.get_output_port_control(),
        torque_adder.get_input_port(0),
    )

    # Cancel out the iiwa control box's internal gravity compensation.
    # This assumes that the iiwa control box does not know about the gripper and thus
    # only considers the arm in its gravity compensation.
    iiwa_only_controller_plant = MultibodyPlant(
        time_step=scenario.plant_config.time_step
    )
    # TODO: Shouldn't be loading a fixed model here!
    iiwa_only_controller_plant_parser = get_parser(iiwa_only_controller_plant)
    iiwa_only_controller_plant_parser.AddModelsFromUrl(
        "package://drake/manipulation/models/iiwa_description/iiwa7/iiwa7_no_collision.sdf"
    )
    iiwa_only_controller_plant.WeldFrames(
        iiwa_only_controller_plant.world_frame(),
        iiwa_only_controller_plant.GetFrameByName("iiwa_link_0"),
    )
    iiwa_only_controller_plant.Finalize()
    gravity_compensation: InverseDynamics = builder.AddNamedSystem(
        "gravity_compensation",
        InverseDynamics(
            plant=iiwa_only_controller_plant,
            mode=InverseDynamics.InverseDynamicsMode.kGravityCompensation,
        ),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"),
        gravity_compensation.get_input_port_estimated_state(),
    )
    negater: Gain = builder.AddNamedSystem("negater", Gain(k=-1, size=num_positions))
    builder.Connect(
        gravity_compensation.get_output_port(),
        negater.get_input_port(),
    )
    builder.Connect(
        negater.get_output_port(),
        torque_adder.get_input_port(1),
    )

    builder.Connect(torque_adder.get_output_port(), station.GetInputPort("iiwa.torque"))

    # TODO: Decide what to do there. Do we want to open for 5s and then grip tightly?
    # If yes, then force control would be better than position control.
    wsg_const_pos_source: ConstantVectorSource = builder.AddNamedSystem(
        "wsg_position_source", ConstantVectorSource(source_value=0.05 * np.ones(1))
    )
    builder.Connect(
        wsg_const_pos_source.get_output_port(), station.GetInputPort("wsg.position")
    )

    # Add data loggers
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

    simulator.AdvanceTo(traj_source_initializer.get_end_time() + 1.0)

    # Save data
    visualizer.StopRecording()
    visualizer.PublishRecording()
    if html_path is not None:
        html = station.internal_meshcat.StaticHtml()
        with open(html_path, "w") as f:
            f.write(html)

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

    # Estimate accelerations using finite differences
    sample_period = sample_times_s[1] - sample_times_s[0]
    joint_accelerations = np.gradient(measured_velocity_data, sample_period, axis=0)
    fs_hz = 1.0 / sample_period
    filtered_joint_accelerations = filter_time_series_data(
        data=joint_accelerations,
        order=10,
        cutoff_freq_hz=5,
        fs_hz=fs_hz,
    )

    # Filter torque data
    filtered_tau_measured = filter_time_series_data(
        data=measured_torque_data,
        order=10,
        cutoff_freq_hz=5,
        fs_hz=fs_hz,
    )

    # Save data
    joint_data = JointData(
        joint_positions=measured_position_data,
        joint_velocities=measured_velocity_data,
        joint_accelerations=filtered_joint_accelerations,
        joint_torques=filtered_tau_measured,
        sample_times_s=sample_times_s,
    )
    joint_data.save_to_disk(save_data_path)


if __name__ == "__main__":
    main()
