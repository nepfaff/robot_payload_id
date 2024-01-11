"""Script for confirming that the gradients of the condition number are correct."""

import numpy as np

from pydrake.all import (
    AutoDiffXd,
    CommonSolverOption,
    Expression,
    MathematicalProgram,
    MathematicalProgramResult,
    SnoptSolver,
    SolverOptions,
)

from robot_payload_id.symbolic import (
    eval_expression_mat,
    eval_expression_mat_derivative,
)


def optimize_cond():
    prog = MathematicalProgram()
    a = prog.NewContinuousVariables(1, "a")
    b = prog.NewContinuousVariables(1, "b")
    symbolic_vars = np.concatenate([a, b])

    W_sym = np.array(
        [
            [Expression(a[0]) * 200, Expression(a[0])],
            [Expression(0), Expression(b[0])],
        ]
    )
    W_sym = W_sym.T @ W_sym

    def condition_number_cost_with_gradient(vars):
        # Assumes that vars are of type AutoDiffXd
        var_values = [var.value() for var in vars]
        W_dataTW_data_numeric = eval_expression_mat(W_sym, symbolic_vars, var_values)
        eigenvalues, eigenvectors = np.linalg.eigh(W_dataTW_data_numeric)
        min_eig_idx = np.argmin(eigenvalues)
        max_eig_idx = np.argmax(eigenvalues)
        min_eig = eigenvalues[min_eig_idx]
        max_eig = eigenvalues[max_eig_idx]
        assert min_eig > 0, "Minimum eigenvalue must be greater than zero!"
        condition_number = max_eig / min_eig

        W_dataTW_data_derivatives = eval_expression_mat_derivative(
            W_sym, symbolic_vars, var_values
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
    prog.SetInitialGuess(a, 2 * np.ones(1))
    prog.SetInitialGuess(b, 1e-3 * np.ones(1))

    solver = SnoptSolver()
    snopt = solver.solver_id()
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    # options.SetOption(snopt, "Major print level", 1)
    res: MathematicalProgramResult = solver.Solve(prog)
    # res: MathematicalProgramResult = Solve(prog)
    if res.is_success():
        print("a:", res.GetSolution(a))
        print("b:", res.GetSolution(b))
    else:
        print("Failed to solve")
        print(prog)
        print(res.get_solution_result())
        print(res.get_optimal_cost())
        print("Snopt info:", res.get_solver_details().info)


if __name__ == "__main__":
    optimize_cond()
