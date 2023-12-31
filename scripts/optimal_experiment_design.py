import pickle

from concurrent import futures
from pathlib import Path
from typing import Tuple

import nevergrad as ng
import numpy as np
import sympy

from pydrake.all import (
    AutoDiffXd,
    CommonSolverOption,
    Expression,
    ExpressionCost,
    ExpressionKind,
    MakeVectorVariable,
    MathematicalProgram,
    MathematicalProgramResult,
    MultibodyForces_,
    SnoptSolver,
    Solve,
    SolverOptions,
    from_sympy,
    log,
)
from tqdm import tqdm

from robot_payload_id.environment import create_arm
from robot_payload_id.symbolic import create_autodiff_plant
from robot_payload_id.utils import ArmPlantComponents, SymJointStateVariables


def pickle_load(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_W_sym(
    dir_path: Path,
    sym_state_variables: SymJointStateVariables,
    num_joints: int,
    num_params: int,
) -> np.ndarray:
    name_to_var = {}
    for q, q_dot, q_ddot, tau in zip(
        sym_state_variables.q,
        sym_state_variables.q_dot,
        sym_state_variables.q_ddot,
        sym_state_variables.tau,
        strict=True,
    ):
        name_to_var[q.get_name()] = q
        name_to_var[q_dot.get_name()] = q_dot
        name_to_var[q_ddot.get_name()] = q_ddot
        name_to_var[tau.get_name()] = tau

    W_sym = np.empty((num_joints, num_params), dtype=Expression)
    for i in range(num_joints):
        for j in range(num_params):
            memo = pickle_load(dir_path / f"W_{i}_{j}_memo.pkl")
            for key, val in memo.items():
                memo[key] = name_to_var[val]
            expression_sympy = pickle_load(dir_path / f"W_{i}_{j}_sympy.pkl")
            expression_drake = from_sympy(expression_sympy, memo=memo)
            W_sym[i, j] = (
                expression_drake
                if isinstance(expression_drake, Expression)
                else Expression(expression_drake)
            )
    return W_sym


def compute_joint_params_from_traj_params(
    num_timesteps: int, a: np.ndarray, b: np.ndarray
):
    # qi(t) = ai * sin(ω*i*t) + bi
    # qi_dot(t) = ai * ω * i * cos(ω*i*t)
    # qi_ddot(t) = ai * (ω*i)**2 * cos(ω*i*t)

    omega = 0.5
    q = np.zeros((num_timesteps, len(a)), dtype=AutoDiffXd)
    q_dot = np.zeros((num_timesteps, len(a)), dtype=AutoDiffXd)
    q_ddot = np.zeros((num_timesteps, len(a)), dtype=AutoDiffXd)
    for t in range(num_timesteps):
        for i in range(len(a)):
            q[t, i] = a[i] * np.sin(omega * (1 + i) * t) + b[i]
            q_dot[t, i] = a[i] * omega * (1 + i) * np.cos(omega * (1 + i) * t)
            q_ddot[t, i] = a[i] * ((omega * (1 + i)) ** 2) * np.cos(omega * (1 + i) * t)
    return q, q_dot, q_ddot


def create_data_matrix_from_traj_samples(
    W_sym: np.ndarray,
    sym_state_variables: SymJointStateVariables,
    q: np.ndarray,
    q_dot: np.ndarray,
    q_ddot: np.ndarray,
) -> np.ndarray:
    num_joints = q.shape[1]
    W_data = np.empty((len(q), W_sym.shape[1]), dtype=Expression)
    for i in range(len(q)):
        sym_to_val = {}
        for j in range(num_joints):
            sym_to_val[sym_state_variables.q[j]] = q[i, j]
            sym_to_val[sym_state_variables.q_dot[j]] = q_dot[i, j]
            sym_to_val[sym_state_variables.q_ddot[j]] = q_ddot[i, j]

        for m in range(num_joints):
            for n in range(W_sym.shape[1]):
                W_data[i * num_joints + m, n] = W_sym[m, n].Substitute(sym_to_val)
    return W_data


def eval_expression_mat(
    expression_mat: np.ndarray, symbolic_vars: np.ndarray, var_vals: np.ndarray
) -> np.ndarray:
    var_val_mapping = dict(zip(symbolic_vars, var_vals))
    evaluated_mat = np.empty(expression_mat.shape)
    for i in range(expression_mat.shape[0]):
        for j in range(expression_mat.shape[1]):
            evaluated_mat[i, j] = expression_mat[i, j].Evaluate(var_val_mapping)
    return evaluated_mat


def eval_expression_mat_derivative(
    expression_mat: np.ndarray, symbolic_vars: np.ndarray, var_vals: np.ndarray
) -> np.ndarray:
    derivatives = np.empty((len(symbolic_vars), *expression_mat.shape))
    for k in range(len(symbolic_vars)):
        var_val_mapping = dict(zip(np.delete(symbolic_vars, k), np.delete(var_vals, k)))
        for i in range(expression_mat.shape[0]):
            for j in range(expression_mat.shape[1]):
                expr: Expression = expression_mat[i, j]
                # Partial derivative of expr w.r.t. symbolic_vars[k]
                derivatives[k, i, j] = (
                    expr.EvaluatePartial(var_val_mapping)
                    .Differentiate(symbolic_vars[k])
                    .Evaluate({symbolic_vars[k]: var_vals[k]})
                )
    return derivatives


# def expression_mat_to_autodiff_mat(
#     expression_mat: np.ndarray, symbolic_vars: np.ndarray, ad_vars: np.ndarray
# ) -> np.ndarray:
#     # Symbolic vars and ad vals must have same order!
#     ad_var_values = np.array([ad_var.value() for ad_var in ad_vars])
#     ad_mat = np.empty(expression_mat.shape, dtype=AutoDiffXd)
#     for i in range(expression_mat.shape[0]):
#         for j in range(expression_mat.shape[1]):
#             expr: Expression = expression_mat[i, j]
#             expr_val = expr.Evaluate(dict(zip(symbolic_vars, ad_var_values)))
#             gradients = []
#             for k in range(len(ad_vars)):
#                 gradients.append(
#                     expr.EvaluatePartial(
#                         dict(
#                             zip(
#                                 np.delete(symbolic_vars, k), np.delete(ad_var_values, k)
#                             )
#                         )
#                     )
#                     .Differentiate(symbolic_vars[k])
#                     .Evaluate({symbolic_vars[k]: ad_var_values[k]})
#                 )
#             ad_mat[i, j] = AutoDiffXd(expr_val, gradients)
#     return ad_mat


def remove_structurally_unidentifiable(
    data_matrix: np.ndarray, symbolic_vars: np.ndarray
) -> np.ndarray:
    """TODO: This needs more work to also deal with vars, etc."""
    # Evaluate with random values
    data_matrix_numeric = eval_expression_mat(
        data_matrix,
        symbolic_vars,
        np.random.uniform(low=1, high=10, size=len(symbolic_vars)),
    )
    _, R = np.linalg.qr(data_matrix_numeric)
    tolerance = 1e-12
    identifiable = np.abs(np.diag(R)) > tolerance
    return data_matrix[:, identifiable]


def optimize_traj(use_one_link_arm: bool = False):
    num_joints = 1 if use_one_link_arm else 7
    data_matrix_dir_path = Path(
        "symbolic_data_matrix_one_link_arm"
        if use_one_link_arm
        else "symbolic_data_matrix_iiwa"
    )

    q_var = MakeVectorVariable(num_joints, "q")
    q_dot_var = MakeVectorVariable(num_joints, "\dot{q}")
    q_ddot_var = MakeVectorVariable(num_joints, "\ddot{q}")
    tau_var = MakeVectorVariable(num_joints, "\tau")
    sym_state_variables = SymJointStateVariables(
        q=q_var, q_dot=q_dot_var, q_ddot=q_ddot_var, tau=tau_var
    )

    W_sym = load_W_sym(
        dir_path=data_matrix_dir_path,
        sym_state_variables=sym_state_variables,
        num_joints=num_joints,
        num_params=num_joints * 10,
    )

    prog = MathematicalProgram()
    a = prog.NewContinuousVariables(num_joints, "a")
    b = prog.NewContinuousVariables(num_joints, "b")
    symbolic_vars = np.concatenate([a, b])

    q, q_dot, q_ddot = compute_joint_params_from_traj_params(
        num_timesteps=100, a=a, b=b
    )
    W_data = create_data_matrix_from_traj_samples(
        W_sym=W_sym,
        sym_state_variables=sym_state_variables,
        q=q,
        q_dot=q_dot,
        q_ddot=q_ddot,
    )

    # Remove structurally unidentifiable columns to prevent SolutionResult.kUnbounded
    W_data = remove_structurally_unidentifiable(W_data, symbolic_vars)

    W_dataTW_data = W_data.T @ W_data

    # A-Optimality
    # NOTE: Can't symbolically compute matrix inverse for matrices bigger than 4x4
    # prog.AddCost(np.trace(W_dataTW_data_inv))

    # D-Optimality
    # NOTE: This doesn't seem to work atm as the det is 0, logdet is inf
    # prog.AddCost(-log_determinant(W_dataTW_data))

    # Can't use AddMaximizeLogDeterminantCost because it requires W_dataTW_data to be
    # polynomial to ensure convexity. We don't care about the convexity
    # prog.AddMaximizeLogDeterminantCost(W_dataTW_data)

    def condition_number_cost_with_gradient(vars):
        # Assumes that vars are of type AutoDiffXd
        var_values = [var.value() for var in vars]
        W_dataTW_data_numeric = eval_expression_mat(
            W_dataTW_data, symbolic_vars, var_values
        )
        eigenvalues, eigenvectors = np.linalg.eigh(W_dataTW_data_numeric)
        min_eig_idx = np.argmin(eigenvalues)
        max_eig_idx = np.argmax(eigenvalues)
        min_eig = eigenvalues[min_eig_idx]
        max_eig = eigenvalues[max_eig_idx]
        assert min_eig > 0, "Minimum eigenvalue must be greater than zero!"
        condition_number = max_eig / min_eig

        W_dataTW_data_derivatives = eval_expression_mat_derivative(
            W_dataTW_data, symbolic_vars, var_values
        )
        condition_number_derivatives = np.empty(len(vars))
        # Based on first order perturbation theory
        # Same as in https://github.com/SNURobotics/optimal-excitation/blob/master/apps/optimal_excitation/functions_KUKA/getCondNumber.m#L57-L58
        # with difference that they name their min eigenvalue max and vice versa
        for i in range(len(vars)):
            condition_number_derivatives[i] = (
                (1 / min_eig)
                * eigenvectors[max_eig_idx].T
                @ W_dataTW_data_derivatives[i]
                @ eigenvectors[max_eig_idx]
            ) - (
                (max_eig / min_eig**2)
                * eigenvectors[min_eig_idx].T
                @ W_dataTW_data_derivatives[i]
                @ eigenvectors[min_eig_idx]
            )

        print(
            vars[0].value(),
            vars[1].value(),
            max_eig,
            min_eig,
            condition_number,
            condition_number_derivatives,
        )
        return AutoDiffXd(condition_number, condition_number_derivatives)

    prog.AddCost(condition_number_cost_with_gradient, vars=symbolic_vars)

    # Prevents zero min eigenvalue
    # TODO: Find a better solution
    prog.AddConstraint(a[0] >= 1e-1)
    prog.AddConstraint(b[0] >= 1e-1)

    # Need some initial guess for reasonable results
    # TODO: How to determine a good guess?
    prog.SetInitialGuess(a, 38 * np.ones(num_joints))
    prog.SetInitialGuess(b, 2 * np.ones(num_joints))

    solver = SnoptSolver()
    snopt = solver.solver_id()
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    # prog.SetSolverOption(snopt, "Major Optimality Tolerance", 1e-1)
    # prog.SetSolverOption(snopt, "Minor Optimality Tolerance", 1e-1)
    prog.SetSolverOption(snopt, "Print file", "snopt.out")
    # options.SetOption(snopt, "Major print level", 1)
    res: MathematicalProgramResult = solver.Solve(prog)
    # res: MathematicalProgramResult = Solve(prog)
    if res.is_success():
        print("a:", res.GetSolution(a))
        print("b:", res.GetSolution(b))
        print("solver:", res.get_solver_id().name())
    else:
        print("Failed to solve")
        print(prog)
        print("Solution result:", res.get_solution_result())
        print("Snopt info:", res.get_solver_details().info)
        print("Final loss:", res.get_optimal_cost())
        print("a:", res.GetSolution(a))
        print("b:", res.GetSolution(b))


def optimize_traj_black_box(use_one_link_arm: bool = False):
    num_joints = 1 if use_one_link_arm else 7
    data_matrix_dir_path = Path(
        "symbolic_data_matrix_one_link_arm"
        if use_one_link_arm
        else "symbolic_data_matrix_iiwa"
    )

    q_var = MakeVectorVariable(num_joints, "q")
    q_dot_var = MakeVectorVariable(num_joints, "\dot{q}")
    q_ddot_var = MakeVectorVariable(num_joints, "\ddot{q}")
    tau_var = MakeVectorVariable(num_joints, "\tau")
    sym_state_variables = SymJointStateVariables(
        q=q_var, q_dot=q_dot_var, q_ddot=q_ddot_var, tau=tau_var
    )

    W_sym = load_W_sym(
        dir_path=data_matrix_dir_path,
        sym_state_variables=sym_state_variables,
        num_joints=num_joints,
        num_params=num_joints * 10,
    )

    a = MakeVectorVariable(num_joints, "a")
    b = MakeVectorVariable(num_joints, "b")
    symbolic_vars = np.concatenate([a, b])

    q, q_dot, q_ddot = compute_joint_params_from_traj_params(
        num_timesteps=100, a=a, b=b
    )
    W_data = create_data_matrix_from_traj_samples(
        W_sym=W_sym,
        sym_state_variables=sym_state_variables,
        q=q,
        q_dot=q_dot,
        q_ddot=q_ddot,
    )

    # Remove structurally unidentifiable columns to prevent SolutionResult.kUnbounded
    W_data = remove_structurally_unidentifiable(W_data, symbolic_vars)

    W_dataTW_data = W_data.T @ W_data

    def condition_number_cost(var_values):
        W_dataTW_data_numeric = eval_expression_mat(
            W_dataTW_data, symbolic_vars, var_values
        )
        eigenvalues, _ = np.linalg.eigh(W_dataTW_data_numeric)
        min_eig_idx = np.argmin(eigenvalues)
        max_eig_idx = np.argmax(eigenvalues)
        min_eig = eigenvalues[min_eig_idx]
        max_eig = eigenvalues[max_eig_idx]

        if min_eig <= 0:
            return np.inf

        condition_number = max_eig / min_eig
        return condition_number

    def condition_number_and_d_optimality_cost(var_values):
        W_dataTW_data_numeric = eval_expression_mat(
            W_dataTW_data, symbolic_vars, var_values
        )
        eigenvalues, _ = np.linalg.eigh(W_dataTW_data_numeric)
        min_eig_idx = np.argmin(eigenvalues)
        max_eig_idx = np.argmax(eigenvalues)
        min_eig = eigenvalues[min_eig_idx]
        max_eig = eigenvalues[max_eig_idx]

        if min_eig <= 0:
            return np.inf

        condition_number = max_eig / min_eig
        log_det = np.log(np.prod(eigenvalues))
        d_optimality = -log_det

        d_optimality_weight = 1e-1
        cost = condition_number + d_optimality_weight * d_optimality
        return cost

    def condition_number_and_e_optimality_cost(var_values):
        W_dataTW_data_numeric = eval_expression_mat(
            W_dataTW_data, symbolic_vars, var_values
        )
        eigenvalues, _ = np.linalg.eigh(W_dataTW_data_numeric)
        min_eig_idx = np.argmin(eigenvalues)
        max_eig_idx = np.argmax(eigenvalues)
        min_eig = eigenvalues[min_eig_idx]
        max_eig = eigenvalues[max_eig_idx]

        if min_eig <= 0:
            return np.inf

        condition_number = max_eig / min_eig
        e_optimality = -min_eig

        e_optimality_weight = 1e-3
        cost = condition_number + e_optimality_weight * e_optimality
        return cost

    # NOTE: Cost function must be pickable for parallelization
    # optimizer = ng.optimizers.NGOpt(parametrization=2, budget=500000, num_workers=16)
    # # Optimize in parallel
    # with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
    #     recommendation = optimizer.minimize(
    #         condition_number_cost, executor=executor, batch_mode=True
    #     )

    optimizer = ng.optimizers.NGOpt(parametrization=2, budget=1000)
    recommendation = optimizer.minimize(condition_number_and_e_optimality_cost)
    print("Final param values", recommendation.value)
    print("Final loss", recommendation.loss)


if __name__ == "__main__":
    # optimize_traj(use_one_link_arm=True)
    optimize_traj_black_box(use_one_link_arm=True)
