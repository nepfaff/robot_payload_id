import numpy as np

from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import load_scenario
from pydrake.all import DiagramBuilder, MeshcatPoseSliders, Simulator

from robot_payload_id.environment import HardwareStationDiagram, IiwaForwardKinematics


def main():
    use_hardware = False
    has_wsg = False

    scenario_data = (
        """
    directives:
    - add_directives:
        file: package://manipulation/iiwa_and_wsg.dmd.yaml
    plant_config:
        time_step: 0.005
        contact_model: "hydroelastic"
        discrete_contact_solver: "sap"
    model_drivers:
        iiwa: !IiwaDriver
            hand_model_name: wsg
        wsg: !SchunkWsgDriver {}
    """
        if has_wsg
        else """
    directives:
    - add_directives:
        file: package://robot_payload_id/iiwa.dmd.yaml
    plant_config:
        # For some reason, this requires a small timestep
        time_step: 0.0001
        contact_model: "hydroelastic"
        discrete_contact_solver: "sap"
    model_drivers:
        iiwa: !IiwaDriver {}
    """
    )

    builder = DiagramBuilder()

    scenario = load_scenario(data=scenario_data)
    station: HardwareStationDiagram = builder.AddNamedSystem(
        "station",
        HardwareStationDiagram(
            scenario=scenario, has_wsg=has_wsg, use_hardware=use_hardware
        ),
    )
    controller_plant = station.internal_station.GetSubsystemByName(
        "iiwa.controller"
    ).get_multibody_plant_for_control()
    differential_ik = AddIiwaDifferentialIK(
        builder,
        controller_plant,
        frame=controller_plant.GetFrameByName("iiwa_link_7"),
    )
    builder.Connect(
        differential_ik.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"),
        differential_ik.GetInputPort("robot_state"),
    )

    # Set up teleop widgets
    teleop = builder.AddSystem(
        MeshcatPoseSliders(
            station.internal_meshcat,
            lower_limit=[0, -0.5, -np.pi, -0.6, -0.8, 0.0],
            upper_limit=[2 * np.pi, np.pi, np.pi, 0.8, 0.3, 1.1],
        )
    )
    builder.Connect(
        teleop.get_output_port(), differential_ik.GetInputPort("X_WE_desired")
    )
    iiwa_forward_kinematics = builder.AddSystem(
        IiwaForwardKinematics(station.internal_station.GetSubsystemByName("plant"))
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_commanded"),
        iiwa_forward_kinematics.get_input_port(),
    )
    builder.Connect(iiwa_forward_kinematics.get_output_port(), teleop.get_input_port())
    if has_wsg:
        wsg_teleop = builder.AddSystem(WsgButton(station.internal_meshcat))
        builder.Connect(
            wsg_teleop.get_output_port(0), station.GetInputPort("wsg.position")
        )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Stop Simulation")
    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)
    station.internal_meshcat.DeleteButton("Stop Simulation")


if __name__ == "__main__":
    main()
