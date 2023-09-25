import numpy as np

from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK, ExtractBodyPose
from manipulation.station import MakeHardwareStation, load_scenario
from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import MeshcatPoseSliders


def main():
    # NOTE: This currently doesn't work on the real hardware
    hardware = False

    scenario_data = """
    directives:
    - add_directives:
        file: package://manipulation/iiwa_and_wsg.dmd.yaml
    model_drivers:
        iiwa: !IiwaDriver
            hand_model_name: wsg
        wsg: !SchunkWsgDriver {}
    """

    meshcat = StartMeshcat()
    builder = DiagramBuilder()

    scenario = load_scenario(data=scenario_data)
    station = builder.AddSystem(
        MakeHardwareStation(scenario, meshcat, hardware=hardware)
    )
    controller_plant = station.GetSubsystemByName(
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
    meshcat.DeleteAddedControls()
    teleop = builder.AddSystem(
        MeshcatPoseSliders(
            meshcat,
            lower_limit=[0, -0.5, -np.pi, -0.6, -0.8, 0.0],
            upper_limit=[2 * np.pi, np.pi, np.pi, 0.8, 0.3, 1.1],
        )
    )
    builder.Connect(
        teleop.get_output_port(), differential_ik.GetInputPort("X_WE_desired")
    )
    # NOTE: This is using "Cheat Ports". For it to work on hardware, we would need to
    # construct the initial pose from the HardwareStation outputs
    plant = station.GetSubsystemByName("plant")
    ee_pose = builder.AddSystem(
        ExtractBodyPose(
            station.GetOutputPort("body_poses"),
            plant.GetBodyByName("iiwa_link_7").index(),
        )
    )
    builder.Connect(station.GetOutputPort("body_poses"), ee_pose.get_input_port())
    builder.Connect(ee_pose.get_output_port(), teleop.get_input_port())
    wsg_teleop = builder.AddSystem(WsgButton(meshcat))
    builder.Connect(wsg_teleop.get_output_port(0), station.GetInputPort("wsg.position"))

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.get_mutable_context()
    simulator.set_target_realtime_rate(1.0)

    meshcat.AddButton("Stop Simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
    meshcat.DeleteButton("Stop Simulation")


if __name__ == "__main__":
    main()
