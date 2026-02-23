import numpy as np
import matplotlib.pyplot as plt


class SeidelSolver:
    def __init__(self, A, b, tol=1e-10, max_iter=1000):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self._validate_input()

    def _validate_input(self):
        n, m = self.A.shape
        if n != m:
            raise ValueError("Матрица A должна быть квадратной.")
        if self.b.shape[0] != n:
            raise ValueError("Размерность b должна совпадать с размером A.")
        if np.any(np.isclose(np.diag(self.A), 0.0)):
            raise ValueError("На диагонали A есть нули, метод Зейделя неприменим.")
        if self.tol <= 0:
            raise ValueError("tol должен быть положительным.")

    def solve(self, x0=None, return_history=False):
        n = self.A.shape[0]
        x = np.zeros(n, dtype=float) if x0 is None else np.array(x0, dtype=float)

        diff_history = []
        residual_history = []

        for iteration in range(1, self.max_iter + 1):
            x_prev = x.copy()
            for i in range(n):
                a_sum = np.dot(self.A[i], x)
                x[i] = (self.b[i] - (a_sum - A[i, i] * x[i])) / self.A[i, i]

            iter_diff = np.linalg.norm(x - x_prev, ord=np.inf)
            residual = np.linalg.norm(self.A @ x - self.b, ord=np.inf)
            diff_history.append(iter_diff)
            residual_history.append(residual)

            if iter_diff < self.tol:
                if return_history:
                    return x, iteration, diff_history, residual_history
                return x, iteration

        raise RuntimeError(
            f"Метод не сошёлся за {self.max_iter} итераций. "
            f"Попробуйте увеличить max_iter или изменить систему."
        )


def plot_convergence(diff_history, residual_history):
    iterations = np.arange(1, len(diff_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, diff_history, label="||x_k - x_(k-1)||_inf")
    plt.plot(iterations, residual_history, label="||A x_k - b||_inf")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Gauss-Seidel: iteration difference and residual")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def build_T(n):
    return np.eye(n) - 0.25 * np.eye(n, k=1) - 0.25 * np.eye(n, k=-1)


def build_D(n):
    return -0.25 * np.eye(n)


def build_A(n):
    T = build_T(n)
    D = build_D(n)
    Z = np.zeros((n, n))

    blocks = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(T)
            elif abs(i - j) == 1:
                row.append(D)
            else:
                row.append(Z)
        blocks.append(row)

    return np.block(blocks)


if __name__ == "__main__":
    x_sol2 = [2.75, 3.5, 2.75, 3.5, 4.5, 3.5, 2.75, 3.5, 2.75]

    n = 30

    A = build_A(n)
    print('A =', A)
    b = np.ones(n * n)
    print('b =', b)

    solver = SeidelSolver(A, b, tol=1e-12, max_iter=5000)
    solution, iters, diff_hist, residual_hist = solver.solve(return_history=True)

    print("Решение x:", solution)
    print("Итераций:", iters)
    plot_convergence(diff_hist, residual_hist)