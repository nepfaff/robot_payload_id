from manipulation.station import MakeHardwareStation, Scenario
from pydrake.all import (
    AbstractValue,
    Context,
    Demultiplexer,
    Diagram,
    DiagramBuilder,
    LeafSystem,
    MultibodyPlant,
    Multiplexer,
    RigidTransform,
    StartMeshcat,
)


def forward_kinematics(plant: MultibodyPlant, plant_context: Context) -> RigidTransform:
    """Computes the pose of the iiwa link 7 based on the joint positions stored in the
    context.

    Args:
        plant (MultibodyPlant): The plant that contains the iiwa.
        plant_context (Context): The context that contains the joint positions of the
        iiwa.

    Returns:
        RigidTransform: The pose of the iiwa link 7 in the world frame.
    """
    link_7 = plant.GetBodyByName("iiwa_link_7")
    gripper_frame = link_7.body_frame()
    X_WG = gripper_frame.CalcPoseInWorld(plant_context)
    return X_WG


class IiwaForwardKinematics(LeafSystem):
    """
    A system that takes the iiwa positions as input and outputs the pose of the iiwa
    link 7 in the world frame.
    """

    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self._plant = plant

        self.DeclareVectorInputPort("iiwa_positions", 7)
        self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()), self._calc_output
        )

    def _calc_output(self, context: Context, output: AbstractValue) -> None:
        iiwa_positions = self.get_input_port().Eval(context)
        plant_context = self._plant.CreateDefaultContext()
        self._plant.SetPositions(
            plant_context, self._plant.GetModelInstanceByName("iiwa"), iiwa_positions
        )
        X_WG = forward_kinematics(self._plant, plant_context)
        output.get_mutable_value().set(X_WG.rotation(), X_WG.translation())


class HardwareStationDiagram(Diagram):
    """
    Consists of an "internal" and and "external" hardware station. The "internal"
    station mirrors the "external" station and is always simulated. One can think of
    the "internal" station as the internal system model. The "external" station
    represents the hardware or a simulated version of it. Having two stations is
    important as only the "internal" station will contain a plant, etc. when the
    "external" station represents hardware and thus only contains LCM logic.
    """

    def __init__(self, scenario: Scenario, use_hardware: bool):
        super().__init__()

        builder = DiagramBuilder()

        # Internal Station
        self.internal_meshcat = StartMeshcat()
        self.internal_station = builder.AddNamedSystem(
            "internal_station",
            MakeHardwareStation(
                scenario,
                meshcat=self.internal_meshcat,
                hardware=False,
            ),
        )

        # External Station
        self.external_meshcat = StartMeshcat()
        self._external_station = builder.AddNamedSystem(
            "external_station",
            MakeHardwareStation(scenario, self.external_meshcat, hardware=use_hardware),
        )

        # Connect the output of external station to the input of internal station
        builder.Connect(
            self._external_station.GetOutputPort("iiwa.position_commanded"),
            self.internal_station.GetInputPort("iiwa.position"),
        )
        wsg_state_demux = builder.AddSystem(Demultiplexer(2, 1))
        builder.Connect(
            self._external_station.GetOutputPort("wsg.state_measured"),
            wsg_state_demux.get_input_port(),
        )
        builder.Connect(
            wsg_state_demux.get_output_port(0),
            self.internal_station.GetInputPort("wsg.position"),
        )

        # Export internal station ports
        builder.ExportOutput(
            self.internal_station.GetOutputPort("body_poses"), "body_poses"
        )
        # Export external station ports
        builder.ExportInput(
            self._external_station.GetInputPort("iiwa.position"), "iiwa.position"
        )
        builder.ExportInput(
            self._external_station.GetInputPort("wsg.position"), "wsg.position"
        )
        builder.ExportOutput(
            self._external_station.GetOutputPort("iiwa.position_measured"),
            "iiwa.position_measured",
        )
        builder.ExportOutput(
            self._external_station.GetOutputPort("iiwa.velocity_estimated"),
            "iiwa.velocity_estimated",
        )
        # Export external state output
        iiwa_state_mux = builder.AddSystem(Multiplexer([7, 7]))
        builder.Connect(
            self._external_station.GetOutputPort("iiwa.position_measured"),
            iiwa_state_mux.get_input_port(0),
        )
        builder.Connect(
            self._external_station.GetOutputPort("iiwa.velocity_estimated"),
            iiwa_state_mux.get_input_port(1),
        )
        builder.ExportOutput(iiwa_state_mux.get_output_port(), "iiwa.state_estimated")

        builder.BuildInto(self)
