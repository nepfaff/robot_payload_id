import pathlib

from pathlib import Path

import hydra

from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from robot_payload_id.environment import collect_joint_data, create_arm
from robot_payload_id.symbolic import calc_lumped_parameters, create_symbolic_plant


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath("..", "config")),
    version_base=None,
)
def main(cfg: OmegaConf):
    # Add log dir to config
    with open_dict(cfg):
        cfg.log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print("Config:\n", cfg)

    log_dir_path = Path(cfg.log_dir_path)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    arm_components = create_arm(
        arm_file_path=cfg.urdf_path, num_joints=cfg.num_joints, time_step=cfg.time_step
    )
    sym_plant_components = create_symbolic_plant(arm_components=arm_components)

    joint_data = collect_joint_data(
        arm_components=arm_components,
        initial_joint_positions=cfg.initial_joint_positions,
        num_trajectories=cfg.num_trajectories,
        sinusoidal_amplitude=cfg.sinusoidal_amplitude,
        trajectory_duration_s=cfg.trajectory_duration_s,
        log_dir_path=log_dir_path,
    )

    alpha_sym, alpha_estimated, alpha_gt = calc_lumped_parameters(
        sym_arm_plant_components=sym_plant_components,
        joint_data=joint_data,
        gt_parameters=[instantiate(params) for params in cfg.gt_parameters],
    )
    for fit, gt, sym in zip(alpha_estimated, alpha_gt, alpha_sym):
        print(f"Estimated {sym.to_string()}: {fit} \t GT: {gt}")


if __name__ == "__main__":
    main()
