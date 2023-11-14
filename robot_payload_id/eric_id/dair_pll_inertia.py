"""
Code adapted from:
https://github.com/DAIRLab/dair_pll/blob/e109191687/dair_pll/inertia.py

Note: If any improvements are made here, we should upstream them as much as
possible.


BSD 3-Clause License

Copyright (c) 2022, Dynamic Autonomy and Intelligent Robotics Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Excerpts from documentation:

    * ``pi_cm`` is an intuitive formatting of these 10 Dof as a standard vector in
      :math:`\mathbb{R}^{10}` as::

        [m, m * p_x, m * p_y, m * p_z, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz]

    * ``pi_o`` is nearly the same as ``pi_cm`` except that the moments of inertia
      are about the body's origin :math:`Bo` instead of the center of mass.

    * ``theta`` is a format designed for underlying smooth, unconstrained,
      and non-degenerate parameterization of rigid body inertia. For a body,
      any value in :math:`\mathbb{R}^{10}` for ``theta`` can be mapped to a
      valid and non-degenerate set of inertial terms as follows::

        theta == [alpha, d_1, d_2, d_3, s_12, s_23, s_13, t_1, t_2, t_3]
        s == [s_12, s_23, s_13]
        t == [t_1, t_2, t_3]
        pi_o == [
            t \\cdot t + 1,
            t_1 * exp(d_1),
            t_1 * s_12 + t_2 * exp(d_2),
            t_1 * s_13 + t_2 * s_23 + t_3 * exp(d_3),
            s \\cdot s + exp(2 * d_2) + exp(2 * d_3),
            s_13 ** 2 + s_23 ** 2 + exp(2 * d_1) + exp(2 * d_3),
            s_12 ** 2 + exp(2 * d_1) + exp(2 * d_2),
            -s_12 * exp(d_1),
            -s_13 * exp(d_1),
            -s_12 * s_13 - s_23 * exp(d_2)
        ]

    An original derivation and characterization of ``theta`` can be found in
    Rucker and Wensing [1]_. Note that Drake orders the inertial off-diagonal
    terms as ``[Ixy Ixz Iyz]``, whereas the original paper [1]_ uses
    ``[Ixy Iyz Ixz]``; thus the index ordering here is slightly different.

.. [1] C. Rucker and P. M. Wensing, "Smooth Parameterization of Rigid-Body
    Inertia", IEEE RA-L 2020, https://doi.org/10.1109/LRA.2022.3144517
"""

from typing import List

import torch

from torch import Tensor

# File: tensor_utils.py


def deal(dealing_tensor: Tensor, dim: int = 0, keep_dim: bool = False) -> List[Tensor]:
    """Converts dim of tensor to list.

    Example:
        Let ``t`` be a 3-dimensional tensor of shape ``(3,5,3)`` such that::

            t[:, i, :] == torch.eye(3).

        Then ``deal(t, dim=1)`` returns a list of 5 ``(3,3)`` identity tensors,
        and ``deal(t, dim=1, keep_dim=True)`` returns a list of ``(3,1,3)``
        tensors.

    Args:
        dealing_tensor: ``(n_0, ..., n_dim, ..., n_{k-1})`` shaped tensor.
        dim: tensor dimension to deal, ``-k <= dim <= k-1``.
        keep_dim: whether to squeeze list items along ``dim``.

    Returns:
        List of dealt sub-tensors of shape ``(..., n_{dim-1}, {n_dim+1}, ...)``
        or ``(..., n_{dim-1}, 1, {n_dim+1}, ...)``.
    """
    tensor_list = torch.split(dealing_tensor, 1, dim=dim)
    if keep_dim:
        return tensor_list
    return [tensor_i.squeeze(dim) for tensor_i in tensor_list]


def skew_symmetric(vectors: Tensor) -> Tensor:
    r"""Converts vectors in :math:`\mathbb{R}^3` into skew-symmetric form.

    Converts vector(s) :math:`v` in ``vectors`` into skew-symmetric matrix:

    .. math::

        S(v) = -S(v)^T = \begin{bmatrix} 0 & -v_3 & v_2 \\
        v_3 & 0 & -v_1 \\
        -v_2 & v_1 & 0 \end{bmatrix}

    Args:
        vectors: ``(*, 3)`` vector(s) to convert to matrices

    Returns:
        ``(*, 3, 3)`` skew-symmetric matrices :math:`S(v)`
    """
    # pylint: disable=E1103
    zero = torch.zeros_like(vectors[..., 0])

    # pylint: disable=E1103
    row_1 = torch.stack((zero, -vectors[..., 2], vectors[..., 1]), -1)
    row_2 = torch.stack((vectors[..., 2], zero, -vectors[..., 0]), -1)
    row_3 = torch.stack((-vectors[..., 1], vectors[..., 0], zero), -1)

    return torch.stack((row_1, row_2, row_3), -2)


def symmetric_offdiagonal(vectors: Tensor) -> Tensor:
    r"""Converts vectors in :math:`\mathbb{R}^3` into symmetric off-diagonal
    form.  This is the same as skew symmetric except for the skew negative
    signs.

    Converts vector(s) :math:`v` in ``vectors`` into symmetric matrix:

    .. math::

        S(v) = S(v)^T = \begin{bmatrix}
            0 & v_3 & v_2 \\
            v_3 & 0 & v_1 \\
            v_2 & v_1 & 0
        \end{bmatrix}

    Args:
        vectors: ``(*, 3)`` vector(s) to convert to matrices

    Returns:
        ``(*, 3, 3)`` symmetric matrices :math:`S(v)`
    """
    # pylint: disable=E1103
    zero = torch.zeros_like(vectors[..., 0])

    # pylint: disable=E1103
    row_1 = torch.stack((zero, vectors[..., 2], vectors[..., 1]), -1)
    row_2 = torch.stack((vectors[..., 2], zero, vectors[..., 0]), -1)
    row_3 = torch.stack((vectors[..., 1], vectors[..., 0], zero), -1)

    return torch.stack((row_1, row_2, row_3), -2)


# File: inertia.py


def parallel_axis_theorem(
    I_BBa_B: Tensor, m_B: Tensor, p_BaBb_B: Tensor, Ba_is_Bcm: bool = True
) -> Tensor:
    """Converts an inertia matrix represented from one reference point to that
    represented from another reference point.  One of these reference points
    must be the center of mass.

    The parallel axis theorem states [2]:

    .. math::

        I_R = I_C - m_{tot} [d]^2


    ...for :math:`I_C` as the inertia matrix about the center of mass,
    :math:`I_R` as the moment of inertia about a point :math:`R` defined as
    :math:`R = C + d`, and :math:`m_{tot}` as the total mass of the body.  The
    brackets in :math:`[d]` indicate the skew-symmetric matrix formed from the
    vector :math:`d`.

    [2] https://en.wikipedia.org/wiki/Moment_of_inertia#Parallel_axis_theorem

    Args:
        I_BBa_B: ``(*, 3, 3)`` inertia matrices.
        m_B: ``(*)`` masses.
        p_BaBb_B: ``(*, 3)`` displacement from current frame to new frame.
        Ba_is_Bcm: ``True`` if the provided I_BBa_B is from the perspective of
          the CoM, ``False`` if from the perspective of the origin.

    Returns:
        ``(*, 3, 3)`` inertia matrices with changed reference point.
    """
    d_squared = skew_symmetric(p_BaBb_B) @ skew_symmetric(p_BaBb_B)
    term = d_squared * m_B.view((-1, 1, 1))

    if Ba_is_Bcm:
        return I_BBa_B - term
    else:
        return I_BBa_B + term


def inertia_matrix_from_vector(I_BBa_B_vec: Tensor) -> Tensor:
    r"""Converts vectorized inertia vector of the following order into an
    inertia matrix:

    .. math::

        [I_{xx}, I_{yy}, I_{zz}, I_{xy}, I_{xz}, I_{yz}]
        \Rightarrow
        \begin{bmatrix}
            I_{xx} & I_{xy} & I_{xz} \\
            I_{xy} & I_{yy} & I_{yz} \\
            I_{xz} & I_{yz} & I_{zz}
        \end{bmatrix}

    Args:
        I_BBa_B_vec: ``(*, 6)`` vectorized inertia parameters.

    Returns:
        ``(*, 3, 3)`` inertia matrix.
    """
    # Put Ixx, Iyy, Izz on the diagonals.
    diags = torch.diag_embed(I_BBa_B_vec[:, :3])

    # Put Ixy, Ixz, Iyz on the off-diagonals.
    off_diags = symmetric_offdiagonal(I_BBa_B_vec[:, 3:].flip(1))

    return diags + off_diags


def inertia_vector_from_matrix(I_BBa_B_mat: Tensor) -> Tensor:
    r"""Converts inertia matrix into vectorized inertia vector of the following
    order:

    .. math::

        \begin{bmatrix}
            I_{xx} & I_{xy} & I_{xz} \\
            I_{xy} & I_{yy} & I_{yz} \\
            I_{xz} & I_{yz} & I_{zz}
        \end{bmatrix}
        \Rightarrow
        [I_{xx}, I_{yy}, I_{zz}, I_{xy}, I_{xz}, I_{yz}]

    Args:
        I_BBa_B_mat: ``(*, 3, 3)`` inertia matrix.

    Returns:
        ``(*, 6)`` vectorized inertia parameters.
    """
    # Grab Ixx, Iyy, Izz on the diagonals.
    firsts = I_BBa_B_mat.diagonal(dim1=1, dim2=2)

    # Grab Ixy, Ixz, Iyz on the off-diagonals individually.
    ixys = I_BBa_B_mat[:, 0, 1].reshape(-1, 1)
    ixzs = I_BBa_B_mat[:, 0, 2].reshape(-1, 1)
    iyzs = I_BBa_B_mat[:, 1, 2].reshape(-1, 1)

    return torch.cat((firsts, ixys, ixzs, iyzs), dim=1)


class InertialParameterConverter:
    """Utility class for transforming between inertial parameterizations."""

    @staticmethod
    def theta_to_pi_o(theta: Tensor) -> Tensor:
        """Converts batch of ``theta`` parameters to ``pi_o`` parameters.

        Args:
            theta: ``(*, 10)`` ``theta``-type parameterization.

        Returns:
            ``(*, 10)`` ``pi_o``-type parameterization.
        """
        (alpha, d_1, d_2, d_3, s_12, s_23, s_13, t_1, t_2, t_3) = deal(theta, -1)

        s_dot_s = (theta[..., 4:7] * theta[..., 4:7]).sum(dim=-1)
        t_dot_t = (theta[..., 7:10] * theta[..., 7:10]).sum(dim=-1)

        # pylint: disable=E1103
        scaled_pi_elements = (
            t_dot_t + 1,
            t_1 * torch.exp(d_1),
            t_1 * s_12 + t_2 * torch.exp(d_2),
            t_1 * s_13 + t_2 * s_23 + t_3 * torch.exp(d_3),
            s_dot_s + torch.exp(2 * d_2) + torch.exp(2 * d_3),
            s_13 * s_13 + s_23 * s_23 + torch.exp(2 * d_1) + torch.exp(2 * d_3),
            s_12 * s_12 + torch.exp(2 * d_1) + torch.exp(2 * d_2),
            -s_12 * torch.exp(d_1),
            -s_13 * torch.exp(d_1),
            -s_12 * s_13 - s_23 * torch.exp(d_2),
        )

        # pylint: disable=E1103
        return torch.exp(2 * alpha).unsqueeze(-1) * torch.stack(
            scaled_pi_elements, dim=-1
        )

    @staticmethod
    def pi_o_to_theta(pi_o: Tensor) -> Tensor:
        """Converts batch of ``pi_o`` parameters to ``theta`` parameters.

        Implements hand-derived local inverse of standard mapping from Rucker
        and Wensing.

        This function inverts :py:meth:`theta_to_pi_o` for valid ``pi_o``.

        Args:
            pi_o: ``(*, 10)`` ``pi_o``-type parameterization.

        Returns:
            ``(*, 10)`` ``theta``-type parameterization.
        """

        # exp(alpha)exp(d_1)
        # pylint: disable=E1103
        exp_alpha_exp_d_1 = torch.sqrt(
            0.5 * (pi_o[..., 5] + pi_o[..., 6] - pi_o[..., 4])
        )

        # exp(alpha)s_12
        exp_alpha_s_12 = -pi_o[..., 7] / exp_alpha_exp_d_1

        # exp(alpha)s_13
        exp_alpha_s_13 = -pi_o[..., 8] / exp_alpha_exp_d_1

        # exp(alpha)exp(d_2)
        # pylint: disable=E1103
        exp_alpha_exp_d_2 = torch.sqrt(
            pi_o[..., 6] - exp_alpha_exp_d_1**2 - exp_alpha_s_12**2
        )

        # exp(alpha)s_23
        exp_alpha_s_23 = (
            -pi_o[..., 9] - exp_alpha_s_12 * exp_alpha_s_13
        ) / exp_alpha_exp_d_2

        # exp(alpha)exp(d3)
        # pylint: disable=E1103
        exp_alpha_exp_d_3 = torch.sqrt(
            pi_o[..., 5]
            - exp_alpha_exp_d_1**2
            - exp_alpha_s_13**2
            - exp_alpha_s_23**2
        )

        # exp(alpha)t_1
        exp_alpha_t_1 = pi_o[..., 1] / exp_alpha_exp_d_1

        # exp(alpha)t_2
        exp_alpha_t_2 = (
            pi_o[..., 2] - exp_alpha_t_1 * exp_alpha_s_12
        ) / exp_alpha_exp_d_2

        # exp(alpha)t_3
        exp_alpha_t_3 = (
            pi_o[..., 3]
            - exp_alpha_t_1 * exp_alpha_s_13
            - exp_alpha_t_2 * exp_alpha_s_23
        ) / exp_alpha_exp_d_3

        # exp(alpha)
        # pylint: disable=E1103
        exp_alpha = torch.sqrt(
            pi_o[..., 0] - exp_alpha_t_1**2 - exp_alpha_t_2**2 - exp_alpha_t_3**2
        ).unsqueeze(-1)

        alpha = torch.log(exp_alpha)
        d_vector = torch.log(
            torch.stack((exp_alpha_exp_d_1, exp_alpha_exp_d_2, exp_alpha_exp_d_3), -1)
            / exp_alpha
        )
        s_and_t = (
            torch.stack(
                (
                    exp_alpha_s_12,
                    exp_alpha_s_23,
                    exp_alpha_s_13,
                    exp_alpha_t_1,
                    exp_alpha_t_2,
                    exp_alpha_t_3,
                ),
                -1,
            )
            / exp_alpha
        )
        theta = torch.cat((alpha, d_vector, s_and_t), -1)
        assert torch.isfinite(theta).all()
        return theta
