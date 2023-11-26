"""
Script for identifying a robot model through SGD on reparameterized inertial parameters.
"""

import pickle
import time

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from robot_payload_id.eric_id.drake_torch_dynamics_test import torch_uniform
from robot_payload_id.eric_id.drake_torch_sys_id import (
    DynamicsModel,
    DynamicsModelTrajectoryLoss,
)
from robot_payload_id.eric_id.drake_torch_sys_id_test import (
    make_dyn_model,
    param_in,
    perturb_inertial_params,
)
from robot_payload_id.utils import (
    SystemIdDataset,
    filter_time_series_data,
    split_dataset_into_train_val_test,
)


def save_checkpoint(
    dynamics_model: DynamicsModel, checkpoint_dir: Path, epoch: int
) -> None:
    checkpoint_dir.mkdir(exist_ok=True)

    torch.save(
        dynamics_model.state_dict(),
        checkpoint_dir / f"dynamics_model_checkpoint_{epoch}.pt",
    )

    masses, coms, rot_inertias = dynamics_model.inertial_params()
    inertial_param_dict = {
        "masses": masses.detach().numpy(),
        "coms": coms.detach().numpy(),
        "rot_inertias": rot_inertias.detach().numpy(),
    }
    pickle.dump(
        inertial_param_dict, open(checkpoint_dir / f"inertial_params_{epoch}.pkl", "wb")
    )


@torch.no_grad()
@hydra.main(
    config_path=str(Path(__file__).parent.joinpath("..", "config")),
    version_base=None,
)
def main(cfg: OmegaConf):
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    OmegaConf.register_new_resolver("get_class", hydra.utils.get_class)

    # Add log dir to config
    with open_dict(cfg):
        cfg.log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log_dir = Path(cfg.log_dir)
    print("Config:\n", cfg)

    wandb.init(
        project="robot_payload_id",
        name=f"{cfg.wandb.name} ({log_dir.parent.name}/{log_dir.name})",
        dir=cfg.log_dir,
        config=OmegaConf.to_container(cfg),
        mode=cfg.wandb.mode,
    )

    torch.manual_seed(0)

    dyn_model, calc_mean_inertial_params_dist = make_dyn_model(
        add_model=cfg.dynamics_model.add_model_func,
        timestep=cfg.dynamics_model.timestep,
        inertial_param_cls=cfg.dynamics_model.inertial_param_cls,
    )
    print(f"GT params:\n{dyn_model.inertial_params()}")

    if cfg.training.regularize_towards_GT:
        # Regularize towards the ground-truth parameters
        dyn_model_loss = DynamicsModelTrajectoryLoss(
            model=dyn_model,
            gamma=cfg.training.regularizer_weight,
        )

    # Load data
    data_dir = Path(cfg.data.loading.data_dir)
    system_id_state_logs = np.load(data_dir / cfg.data.loading.state_logs_file_name)
    system_id_torque_logs = np.load(
        data_dir / cfg.data.loading.measured_torque_logs_file_name
    )
    system_id_log_sample_times = np.load(
        data_dir / cfg.data.loading.sample_times_file_name
    )
    num_joints = system_id_state_logs.shape[-1] // 2
    joint_positions = system_id_state_logs[:, :num_joints]
    joint_velocities = system_id_state_logs[:, num_joints:]

    # Estimate accelerations using finite differences
    joint_accelerations = np.zeros((len(system_id_log_sample_times), num_joints))
    for i in range(num_joints):
        joint_accelerations[:, i] = np.gradient(joint_velocities[:, i])

    fs_hz = 1.0 / (system_id_log_sample_times[1] - system_id_log_sample_times[0])
    filtered_joint_accelerations = (
        filter_time_series_data(
            data=joint_accelerations,
            order=cfg.data.preprocessing.acceleration_filter_order,
            cutoff_freq_hz=cfg.data.preprocessing.acceleration_filter_cutoff_hz,
            fs_hz=fs_hz,
        )
        if cfg.data.preprocessing.filter_accelerations
        else joint_accelerations
    )
    filtered_tau_measured = (
        filter_time_series_data(
            data=system_id_torque_logs,
            order=cfg.data.preprocessing.measured_torque_filter_order,
            cutoff_freq_hz=cfg.data.preprocessing.measured_torque_filter_cutoff_hz,
            fs_hz=fs_hz,
        )
        if cfg.data.preprocessing.filter_measured_torques
        else system_id_torque_logs
    )

    # Log data
    np.save(log_dir / "q.npy", joint_positions)
    np.save(log_dir / "v.npy", joint_velocities)
    np.save(log_dir / "vd.npy", filtered_joint_accelerations)
    np.save(log_dir / "tau_measured.npy", filtered_tau_measured)
    if cfg.data.preprocessing.filter_accelerations:
        np.save(log_dir / "vd_unfiltered.npy", joint_accelerations)
    if cfg.data.preprocessing.filter_measured_torques:
        np.save(log_dir / "tau_measured_unfiltered.npy", system_id_torque_logs)

    q = torch.tensor(joint_positions)
    v = torch.tensor(joint_velocities)
    vd = torch.tensor(filtered_joint_accelerations)
    tau_measured = torch.tensor(filtered_tau_measured)

    # Create data loaders
    dataset = SystemIdDataset(q, v, vd, filtered_tau_measured)
    train_dataset, val_dataset, test_dataset = split_dataset_into_train_val_test(
        dataset,
        train_ratio=cfg.data.splits.train_ratio,
        val_ratio=cfg.data.splits.val_ratio,
        shuffle=cfg.data.splits.shuffle_before_split,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=len(train_dataset))

    # Perturb all model parameters by a small amount
    perturb_inertial_params(
        dyn_model.inertial_params,
        perturb_scale=cfg.dynamics_model.initial_inertial_perturb_scale,
    )
    inertial_params_list = list(dyn_model.inertial_params.parameters())
    for param in dyn_model.parameters():
        if param_in(param, inertial_params_list):
            continue
        param.data += cfg.dynamics_model.initial_inertial_perturb_scale * torch_uniform(
            param.shape
        )

    if not cfg.training.regularize_towards_GT:
        # Regularize towards the initial guess
        dyn_model_loss = DynamicsModelTrajectoryLoss(
            model=dyn_model,
            gamma=cfg.training.regularizer_weight,
        )

    opt = torch.optim.Adam(dyn_model.parameters(), lr=cfg.training.learning_rate)
    losses = []
    val_tau_losses = []
    test_tau_losses = []
    loss_dicts = []
    dists = []

    print(f"Initial params:\n{dyn_model.inertial_params()}")
    start_time = time.time()
    with torch.set_grad_enabled(True):
        for epoch in tqdm(range(cfg.training.num_epochs), desc="Epoch"):
            current_epoch_losses = []
            current_epoch_loss_dicts = []
            for data in tqdm(train_dataloader, leave=False, desc="  Batch"):
                loss, loss_dict = dyn_model_loss(
                    data["q"], data["v"], data["vd"], data["tau"]
                )
                wandb_logs_iter = {
                    f"{key}_loss_iter": loss_dict[key] for key in loss_dict.keys()
                }
                wandb_logs_iter["combined_loss_iter"] = loss.detach().item()
                wandb.log(wandb_logs_iter)
                current_epoch_losses.append(loss.detach().item())
                current_epoch_loss_dicts.append(loss_dict)

                loss.backward()
                opt.step()
                opt.zero_grad()

            # Record mean train loss
            losses.append(np.mean(current_epoch_losses))
            mean_loss_dict = {
                key: sum(d[key] for d in current_epoch_loss_dicts)
                / len(current_epoch_loss_dicts)
                for key in current_epoch_loss_dicts[0].keys()
            }
            wandb_logs_epoch = {
                f"{key}_loss_epoch": mean_loss_dict[key]
                for key in mean_loss_dict.keys()
            }
            wandb_logs_epoch["combined_loss_epoch"] = losses[-1]

            loss_dicts.append(mean_loss_dict)

            # Record val and test loss
            val_data = next(iter(val_dataloader))
            _, val_loss_dict = dyn_model_loss(
                val_data["q"], val_data["v"], val_data["vd"], val_data["tau"]
            )
            val_tau_losses.append(val_loss_dict["tau"])
            wandb_logs_epoch["val_tau_loss_epoch"] = val_tau_losses[-1]
            test_data = next(iter(test_dataloader))
            _, test_loss_dict = dyn_model_loss(
                test_data["q"], test_data["v"], test_data["vd"], test_data["tau"]
            )
            test_tau_losses.append(test_loss_dict["tau"])
            wandb_logs_epoch["test_tau_loss_epoch"] = test_tau_losses[-1]

            # Record distance from ground-truth parameters.
            dist = calc_mean_inertial_params_dist().item()
            dists.append(dist)
            wandb_logs_epoch["entropic_divergence_epoch"] = dist

            wandb.log(wandb_logs_epoch)

            if (
                epoch > 0 and epoch % cfg.training.save_checkpoint_every_n_epochs == 0
            ) or epoch == cfg.training.num_epochs - 1:
                save_checkpoint(
                    dynamics_model=dyn_model,
                    checkpoint_dir=log_dir / "checkpoints",
                    epoch=epoch,
                )

    print(f"Final params:\n{dyn_model.inertial_params()}")

    final_loss, final_loss_dict = dyn_model_loss(q, v, vd, tau_measured)
    print(f"Initial loss: {losses[0]}, Final loss: {final_loss}")
    wandb.log(
        {
            "initial_loss": losses[0],
            "final_loss": final_loss,
            "optimization_time_s": time.time() - start_time,
        }
    )

    # Plot losses to visualize convergence
    _, axs = plt.subplots(nrows=2)
    plt.sca(axs[0])
    loss_keys = list(final_loss_dict.keys())
    plt.plot(losses, linewidth=3)
    for key in loss_keys:
        plt.plot([loss_dict[key] for loss_dict in loss_dicts])
    plt.plot(val_tau_losses)
    plt.plot(test_tau_losses)
    plt.legend(["sum"] + loss_keys + ["tau_val", "tau_test"])
    plt.ylabel("Loss")
    plt.sca(axs[1])
    plt.plot(dists)
    plt.ylabel("Entropic Divergence")
    plt.xlabel("Epoch")
    wandb.log({"combined_loss_plot": wandb.Image(plt)})


if __name__ == "__main__":
    main()
