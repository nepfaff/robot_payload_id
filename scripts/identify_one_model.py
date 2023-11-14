import matplotlib.pyplot as plt
import torch

from drake_torch_sys_id_test import add_model_iiwa, add_model_one_link_arm
from pydrake.all import MultibodyPlant, Parser
from tqdm import tqdm

from robot_payload_id.eric_id.containers import dict_items_zip
from robot_payload_id.eric_id.dair_pll_inertia import parallel_axis_theorem
from robot_payload_id.eric_id.drake_torch_dynamics import (
    InertialEntropicDivergence,
    LogCholeskyComInertialParameter,
    LogCholeskyInertialParameter,
    LogCholeskyLinAlgInertialParameter,
    PseudoInertiaCholeskyInertialParameter,
    RawInertialParameter,
    VectorInertialParameter,
    get_candidate_sys_id_bodies,
)
from robot_payload_id.eric_id.drake_torch_dynamics_test import torch_uniform
from robot_payload_id.eric_id.drake_torch_sys_id import (
    DynamicsModel,
    DynamicsModelTrajectoryLoss,
)

VISUALIZE = True
# TODO: Compare different parameterizations empirically
DEFAULT_INERTIA_CLS = LogCholeskyInertialParameter


def make_dyn_model(add_model, inertial_param_cls=DEFAULT_INERTIA_CLS):
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


@torch.no_grad()
def main():
    # Parameters
    add_model = add_model_iiwa  # add_model_one_link_arm
    lr = 1e-2
    regularizer_weight = 1e-2  # TODO: Investigate whether this is too high or whether we are better without it
    num_epochs = 200
    perturb_scale = 0.5
    inertial_param_cls = DEFAULT_INERTIA_CLS
    noise_scale = 1.0

    dyn_model, calc_mean_inertial_params_dist = make_dyn_model(
        add_model,
        inertial_param_cls=inertial_param_cls,
    )
    print(f"GT params:\n{dyn_model.inertial_params()}")

    torch.manual_seed(0)

    num_q = dyn_model.inverse_dynamics.num_velocities()
    N = 1000
    dim_q = (N, num_q)
    q = 1.0 * torch_uniform(dim_q)
    v = 3.0 * torch_uniform(dim_q)
    vd = 5.0 * torch_uniform(dim_q)
    tau_gt = dyn_model(q, v, vd)

    # Perturb all model parameters a small amount.
    # TODO: Try initializing to some positive mass, zero CoM, and identity inertia
    perturb_inertial_params(dyn_model.inertial_params, perturb_scale=perturb_scale)
    inertial_params_list = list(dyn_model.inertial_params.parameters())
    for param in dyn_model.parameters():
        if param_in(param, inertial_params_list):
            continue
        param.data += perturb_scale * torch_uniform(param.shape)
    # num_bodies = len(dyn_model.inverse_dynamics.inertial_params()[0])
    # dyn_model.inverse_dynamics.inertial_params = inertial_param_cls(
    #     masses=torch.Tensor([1.0]).expand(num_bodies),
    #     coms=torch.zeros((1, 3)).expand((num_bodies, 3)),
    #     rot_inertias=torch.eye(3).unsqueeze(0).expand((num_bodies, 3, 3)),
    # )

    # NOTE: This initializes the entropic divergence regularizer to the current
    # (disturbed) parameters
    dyn_model_loss = DynamicsModelTrajectoryLoss(
        model=dyn_model,
        gamma=regularizer_weight,
    )

    # Add some slight (arbitrary) noise to simulate measurement errors/ unmodelled
    # dynamics
    q += noise_scale * 0.001 * torch_uniform(dim_q)
    v += noise_scale * 0.01 * torch_uniform(dim_q)
    vd += noise_scale * 0.05 * torch_uniform(dim_q)

    # Show that we can decrease from here.
    opt = torch.optim.Adam(dyn_model.parameters(), lr=lr)
    losses = []
    loss_dicts = []
    dists = []

    print(f"Initial params:\n{dyn_model.inertial_params()}")
    with torch.set_grad_enabled(True):
        for _ in tqdm(range(num_epochs)):
            # Record loss.
            loss, loss_dict = dyn_model_loss(q, v, vd, tau_gt)
            losses.append(loss.detach().item())
            loss_dicts.append(loss_dict)
            # Record distance from ground-truth parameters.
            dist = calc_mean_inertial_params_dist().item()
            dists.append(dist)
            # Optimize.
            loss.backward()
            opt.step()
            opt.zero_grad()
    print(f"Final params:\n{dyn_model.inertial_params()}")

    # Analyze basic trends.
    final_loss, final_loss_dict = dyn_model_loss(q, v, vd, tau_gt)
    losses.append(final_loss.item())
    loss_dicts.append(final_loss_dict)
    final_dist = calc_mean_inertial_params_dist()
    dists.append(final_dist.item())

    if VISUALIZE:
        _, axs = plt.subplots(nrows=2)

        plt.sca(axs[0])
        loss_keys = list(final_loss_dict.keys())
        plt.plot(losses, linewidth=3)
        for key in loss_keys:
            plt.plot([loss_dict[key] for loss_dict in loss_dicts])
        plt.legend(["sum"] + loss_keys)
        plt.ylabel("Loss")

        plt.sca(axs[1])
        plt.plot(dists)
        plt.ylabel("Entropic Divergence")
        plt.xlabel("Epoch")
        plt.show()

    print(f"Initial loss: {losses[0]}, Final loss: {final_loss}")


if __name__ == "__main__":
    main()
