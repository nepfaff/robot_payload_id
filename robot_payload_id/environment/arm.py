import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseDynamicsController,
    LogVectorOutput,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    PiecewisePolynomial,
    StartMeshcat,
    TrajectorySource,
)

from robot_payload_id.utils import ArmComponents, get_parser


def create_arm(
    arm_file_path: str, num_joints: int, time_step: float = 0.0
) -> ArmComponents:
    """Creates a robotic arm system.

    Args:
        arm_file_path (str): The URDF or SDFormat file of the robotic arm.
        num_joints (int): The number of joints of the robotic arm.
        time_step (float, optional): The time step to use for the plant. Defaults to 0.0.

    Returns:
        ArmComponents: The components of the robotic arm system.
    """

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
    parser = get_parser(plant)

    # Add arm
    arm = parser.AddModels(arm_file_path)[0]
    plant.RenameModelInstance(arm, "arm")
    plant.Finalize()

    placeholder_trajectory = PiecewisePolynomial(np.zeros((num_joints, 1)))
    trajectory_source = builder.AddSystem(
        TrajectorySource(placeholder_trajectory, output_derivative_order=1)
    )

    # Add Controller
    controller_plant = MultibodyPlant(time_step)
    controller_parser = Parser(controller_plant)
    controller_parser.AddModels(arm_file_path)
    controller_plant.Finalize()
    arm_controller = builder.AddSystem(
        InverseDynamicsController(
            controller_plant,
            kp=[100] * num_joints,
            kd=[10] * num_joints,
            ki=[1] * num_joints,
            has_reference_acceleration=False,
        )
    )
    arm_controller.set_name("arm_controller")
    builder.Connect(
        plant.get_state_output_port(arm),
        arm_controller.get_input_port_estimated_state(),
    )
    builder.Connect(
        arm_controller.get_output_port_control(), plant.get_actuation_input_port(arm)
    )
    builder.Connect(
        trajectory_source.get_output_port(),
        arm_controller.get_input_port_desired_state(),
    )

    # Meshcat
    meshcat = StartMeshcat()
    if num_joints < 3:
        meshcat.Set2dRenderMode()
    meshcat_visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    state_logger = LogVectorOutput(plant.get_state_output_port(), builder)
    commanded_torque_logger = LogVectorOutput(
        arm_controller.get_output_port_control(), builder
    )

    diagram = builder.Build()

    return ArmComponents(
        num_joints=num_joints,
        diagram=diagram,
        plant=plant,
        trajectory_source=trajectory_source,
        state_logger=state_logger,
        commanded_torque_logger=commanded_torque_logger,
        meshcat=meshcat,
        meshcat_visualizer=meshcat_visualizer,
    )
