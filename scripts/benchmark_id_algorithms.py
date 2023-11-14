import csv

from dataclasses import dataclass
from typing import Callable, List

import torch

from drake_torch_sys_id_test import add_model_iiwa, add_model_one_link_arm
from pydrake.all import Body, MultibodyPlant, Parser
from tqdm import tqdm

from robot_payload_id.eric_id.containers import dict_items_zip
from robot_payload_id.eric_id.dair_pll_inertia import parallel_axis_theorem
from robot_payload_id.eric_id.drake_torch_dynamics import (
    InertialEntropicDivergence,
    InertialParameter,
    LogCholeskyComInertialParameter,
    LogCholeskyInertialParameter,
    LogCholeskyLinAlgInertialParameter,
    PseudoInertiaCholeskyInertialParameter,
    RawInertialParameter,
    VectorInertialParameter,
)
from robot_payload_id.eric_id.drake_torch_dynamics_test import torch_uniform
from robot_payload_id.eric_id.drake_torch_sys_id import (
    DynamicsModel,
    DynamicsModelTrajectoryLoss,
)


@dataclass
class IDCombination:
    inertial_param_cls: InertialParameter
    add_model_func: Callable[[MultibodyPlant], List[Body]]
    use_regularization: bool
    use_identity_initial_guess: bool
    initial_guess_perturb_scale: float
    noise_scale: float


ID_COMBINATIONS: List[IDCombination] = []

for inertial_param_cls in [
    RawInertialParameter,
    VectorInertialParameter,
    PseudoInertiaCholeskyInertialParameter,
    LogCholeskyInertialParameter,
    LogCholeskyLinAlgInertialParameter,
    LogCholeskyComInertialParameter,
]:
    for add_model_func in [
        # add_model_one_link_arm,
        add_model_iiwa,
    ]:
        for noise_scale in [0.0, 0.5, 1.0, 2.0, 5.0]:
            for use_regularization in [False, True]:
                for initial_guess_perturb_scale in [0.0, 1e-10, 0.3, 0.4, 0.5]:
                    use_identity_initial_guess = (
                        True if initial_guess_perturb_scale == 0.0 else False
                    )
                    ID_COMBINATIONS.append(
                        IDCombination(
                            inertial_param_cls=inertial_param_cls,
                            add_model_func=add_model_func,
                            use_regularization=use_regularization,
                            use_identity_initial_guess=use_identity_initial_guess,
                            initial_guess_perturb_scale=initial_guess_perturb_scale,
                            noise_scale=noise_scale,
                        )
                    )


def make_dyn_model(add_model, inertial_param_cls):
    plant = MultibodyPlant(time_step=0.001)
    bodies = add_model(plant)
    plant.Finalize()
    dyn_model = DynamicsModel.from_plant(
        plant, bodies, inertial_param_cls=inertial_param_cls
    )
    masses_gt, coms_gt, rot_inertias_gt = dyn_model.inertial_params()
    inertial_params_dist_gt = InertialEntropicDivergence(
        masses_gt, coms_gt, rot_inertias_gt
    )

    @torch.no_grad()
    def calc_mean_inertial_params_dist():
        masses, coms, rot_inertias = dyn_model.inertial_params()
        return inertial_params_dist_gt(masses, coms, rot_inertias).mean()

    return dyn_model, calc_mean_inertial_params_dist


def param_in(p_check, params):
    # Possible bug in PyTorch. See testing below.
    ids = [id(p) for p in params]
    return id(p_check) in ids


def copy_parameters(*, dest, src):
    dest_params = dict(dest.named_parameters())
    src_params = dict(src.named_parameters())
    items_iter = dict_items_zip(dest_params, src_params)
    for _, (dest_param, src_param) in items_iter:
        dest_param.data[:] = src_param.data


def perturb_inertial_params(inertial_params, *, perturb_scale):
    masses, coms, rot_inertias = inertial_params()
    rot_inertias_cm = parallel_axis_theorem(rot_inertias, masses, coms)
    # TODO(eric.cousineau): Shift CoM?
    N = masses.shape[0]
    mass_scale = 1.0 + perturb_scale * torch_uniform(N)
    masses *= mass_scale
    # TODO(eric.cousineau): Should rotate inertia as well? Add random
    # point-mass noise per Lee et al?
    rot_inertias_cm *= mass_scale.reshape((-1, 1, 1))
    rot_inertias = parallel_axis_theorem(
        rot_inertias_cm,
        masses,
        coms,
        Ba_is_Bcm=False,
    )
    # Reconstruct and remap parameters.
    inertial_param_cls = type(inertial_params)
    perturbed = inertial_param_cls(masses, coms, rot_inertias)
    copy_parameters(dest=inertial_params, src=perturbed)


def is_psd(mat: torch.Tensor) -> bool:
    return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real >= 0).all())


def are_inertial_params_feasible(
    masses: torch.Tensor, rot_inertias: torch.Tensor
) -> bool:
    # NOTE: Also require the CoM to be within the convex hull of the body
    for mass in masses:
        if mass <= 0.0:
            return False
    for rot_inertia in rot_inertias:
        if not is_psd(rot_inertia):
            return False
        eigvals = torch.linalg.eigvals(rot_inertia).real
        if not (
            eigvals[0] <= eigvals[1] + eigvals[2]
            and eigvals[1] <= eigvals[0] + eigvals[2]
            and eigvals[2] <= eigvals[0] + eigvals[1]
        ):
            return False
    return True


@torch.no_grad()
def main():
    # Parameters
    # TODO: Make these configurable with Hydra
    lr = 1e-2
    regularizer_weight = 1e-2
    num_epochs = 200
    csv_path = "id_benchmark.csv"

    for i, id_combination in tqdm(
        enumerate(ID_COMBINATIONS), total=len(ID_COMBINATIONS), desc="ID combinations"
    ):
        torch.manual_seed(0)

        dyn_model, calc_mean_inertial_params_dist = make_dyn_model(
            add_model=id_combination.add_model_func,
            inertial_param_cls=id_combination.inertial_param_cls,
        )
        gt_params = dyn_model.inertial_params()

        # Generate excitaiton trajectory
        num_q = dyn_model.inverse_dynamics.num_velocities()
        N = 1000
        dim_q = (N, num_q)
        q = 1.0 * torch_uniform(dim_q)
        v = 3.0 * torch_uniform(dim_q)
        vd = 5.0 * torch_uniform(dim_q)
        tau_gt = dyn_model(q, v, vd)

        # Set initial guess
        if id_combination.use_identity_initial_guess:
            num_bodies = len(dyn_model.inverse_dynamics.inertial_params()[0])
            dyn_model.inverse_dynamics.inertial_params = (
                id_combination.inertial_param_cls(
                    masses=torch.Tensor([1.0]).expand(num_bodies),
                    coms=torch.zeros((1, 3)).expand((num_bodies, 3)),
                    rot_inertias=torch.eye(3).unsqueeze(0).expand((num_bodies, 3, 3)),
                )
            )
        else:
            # Perturb all model parameters a small amount.
            perturb_inertial_params(
                dyn_model.inertial_params,
                perturb_scale=id_combination.initial_guess_perturb_scale,
            )
            inertial_params_list = list(dyn_model.inertial_params.parameters())
            for param in dyn_model.parameters():
                if param_in(param, inertial_params_list):
                    continue
                param.data += (
                    id_combination.initial_guess_perturb_scale
                    * torch_uniform(param.shape)
                )

        # This initializes the entropic divergence regularizer to the current
        # (disturbed) parameters
        dyn_model_loss = DynamicsModelTrajectoryLoss(
            model=dyn_model,
            gamma=id_combination.use_regularization * regularizer_weight,
        )

        # Add some slight (arbitrary) noise to simulate measurement errors/ unmodelled
        # dynamics
        q += id_combination.noise_scale * 0.001 * torch_uniform(dim_q)
        v += id_combination.noise_scale * 0.01 * torch_uniform(dim_q)
        vd += id_combination.noise_scale * 0.05 * torch_uniform(dim_q)

        # Show that we can decrease from here.
        opt = torch.optim.Adam(dyn_model.parameters(), lr=lr)
        losses = []
        loss_dicts = []
        dists = []

        try:
            with torch.set_grad_enabled(True):
                for _ in tqdm(range(num_epochs), leave=False, desc="Epochs"):
                    # Record loss
                    loss, loss_dict = dyn_model_loss(q, v, vd, tau_gt)
                    losses.append(loss.detach().item())
                    loss_dicts.append(loss_dict)
                    # Record distance from ground-truth parameters
                    dist = calc_mean_inertial_params_dist().item()
                    dists.append(dist)
                    # Optimize
                    loss.backward()
                    opt.step()
                    opt.zero_grad()

            # Analyze basic trends.
            final_loss, final_loss_dict = dyn_model_loss(q, v, vd, tau_gt)
            losses.append(final_loss.item())
            loss_dicts.append(final_loss_dict)
            final_dist = calc_mean_inertial_params_dist()
            dists.append(final_dist.item())

            final_masses, final_coms, final_rot_inertias = dyn_model.inertial_params()

            inertial_params_dist_gt = InertialEntropicDivergence(*gt_params)
            entropic_divergence_to_gt = (
                inertial_params_dist_gt(final_masses, final_coms, final_rot_inertias)
                .mean()
                .item()
            )

            success = True
        except:
            final_loss = torch.Tensor([float("nan")])
            entropic_divergence_to_gt = float("nan")
            final_masses = torch.Tensor([float("nan")])
            final_coms = torch.Tensor([float("nan")])
            final_rot_inertias = torch.Tensor([float("nan")])
            success = False

        info = {
            "inertial_param_cls": id_combination.inertial_param_cls.__name__,
            "add_model_func": id_combination.add_model_func.__name__,
            "use_regularization": id_combination.use_regularization,
            "use_identity_initial_guess": id_combination.use_identity_initial_guess,
            "initial_guess_perturb_scale": id_combination.initial_guess_perturb_scale,
            "noise_scale": id_combination.noise_scale,
            "final_loss": final_loss.item(),
            "entropic_divergence_to_gt": entropic_divergence_to_gt,
            "are_final_params_feasible": are_inertial_params_feasible(
                masses=final_masses, rot_inertias=final_rot_inertias
            ),
            "success": success,
        }

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            column_names = info.keys()
            if i == 0:
                writer.writerow(column_names)
            writer.writerow([info[name] for name in column_names])


if __name__ == "__main__":
    main()
