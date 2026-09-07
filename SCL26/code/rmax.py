"""
solve.py
========

Given a K-regular graph specified as a graph6 string, solve the pair of
self-consistent fixed-point equations

    R_par   = F_par(R_par, 0)      F_par(R_par, 0)   = 1 / K_uv(R_par, 0)
    R_perp  = F_perp(0, R_perp)    F_perp(0, R_perp) = 1 / K_vv(0, R_perp)

Background
----------
For a K-regular graph on N nodes with adjacency matrix A, define

    Q     = A^2 - (K^2 / N) * J          (J = the all-ones matrix)
    M_uu  = Q
    M_uv  = c*I - Q/N
    M_vv  = (1 - 2/N)*c*I + Q/N^2

where c = K(N-K)/N = diag(Q). On a regular graph every diagonal entry of A^2
equals K, so diag(Q) really is the constant c*I and no Hadamard products are
needed anywhere in building these matrices.

Two structural facts follow from Q @ 1 = 0:

  * At R_perp = 0 (beta = 0), x is forced to be a multiple of the all-ones
    vector, so K_uu(R_par, 0) is identically zero and equation 1 collapses
    to a one-dimensional optimisation over a single scalar alpha in
    [-R_par, R_par].

  * At R_par = 0 (alpha = 0), equation 2 is a genuine (N-1)-dimensional,
    non-convex maximisation of  s2(y)^T M_vv s2(y)  over
    {y : 1^T y = 0,  ||y||_2 <= R_perp}, solved here by projected-gradient
    ascent (Adam) from several restarts.

Usage
-----
    python solve.py "<graph6 string>"
"""

from __future__ import annotations

import sys
import numpy as np
import networkx as nx
from scipy.optimize import brentq

# ============================================================================
# 1. Graph construction & validation
# ============================================================================


class RegularGraph:
    """A connected K-regular graph, together with the matrices needed to
    solve the two fixed-point equations."""

    def __init__(self, graph6: str):
        text = graph6.strip()
        if text.startswith(">>graph6<<"):
            text = text[len(">>graph6<<") :]
        G = nx.from_graph6_bytes(text.encode("ascii"))

        self.A = nx.to_numpy_array(G)
        self.N = G.number_of_nodes()
        self.K = self._check_connected_and_regular(G)
        self.c = self.K * (self.N - self.K) / self.N  # diag(Q)
        self.u_star = 1.0 / self.K
        self.Q, self.M_uv, self.M_vv = self._build_matrices()
        eigvals = np.linalg.eigvalsh(self.A)
        self.Mpp = self.K * np.sqrt(np.sum((self.K - eigvals[:-1]) ** (-2)))

    @staticmethod
    def _check_connected_and_regular(G: nx.Graph) -> int:
        if not nx.is_connected(G):
            raise ValueError("graph6 input must describe a connected graph")
        degrees = {deg for _, deg in G.degree()}
        if len(degrees) != 1:
            raise ValueError(f"graph must be regular; found degrees {sorted(degrees)}")
        return degrees.pop()

    def _build_matrices(self):
        """Q = A^2 - (K^2/N) J, then M_uv and M_vv follow directly from the
        diag(Q) = c*I identity -- no Hadamard products required."""
        N, K, c = self.N, self.K, self.c
        I, J = np.eye(N), np.ones((N, N))
        Q = self.A @ self.A - (K**2 / N) * J
        M_uv = c * I - Q / N
        M_vv = (1 - 2.0 / N) * c * I + Q / N**2
        return Q, M_uv, M_vv


# ============================================================================
# 2. Generic numerical utilities (shared by both equations)
# ============================================================================


def maximize_scalar(f, lo: float, hi: float, grid_points: int = 400, polish_iters: int = 100) -> float:
    """Robust 1-D maximiser for a (possibly mildly multimodal) function on
    [lo, hi]: a coarse grid scan locates the right neighbourhood, then a
    golden-section search polishes the maximiser within it."""
    grid = np.linspace(lo, hi, grid_points + 1)
    values = np.array([f(x) for x in grid])
    best = int(np.argmax(values))
    cell = (hi - lo) / grid_points
    a, b = max(lo, grid[best] - cell), min(hi, grid[best] + cell)

    invphi = (np.sqrt(5.0) - 1) / 2
    x1, x2 = b - invphi * (b - a), a + invphi * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(polish_iters):
        if f1 < f2:
            a, x1, f1 = x1, x2, f2
            x2 = a + invphi * (b - a)
            f2 = f(x2)
        else:
            b, x2, f2 = x2, x1, f1
            x1 = b - invphi * (b - a)
            f1 = f(x1)
    return x1 if f1 >= f2 else x2


def solve_fixed_point_equation(K_of_R, lo: float, hi: float, growth: float = 2.0, max_expand: int = 60):
    """Solve R = 1 / K_of_R(R) by Brent's method, expanding [lo, hi] outward
    until the root is bracketed. h(R) = R - 1/K_of_R(R) is increasing in R
    (more budget R can only raise K, which lowers F = 1/K), so a sign change
    in h always exists once the bracket is wide enough. Returns (R*, residual)."""

    def h(R: float) -> float:
        k = K_of_R(R)
        return R - (1.0 / k if k > 0 else np.inf)

    f_lo, f_hi = h(lo), h(hi)
    expansions = 0
    while f_lo * f_hi > 0:
        expansions += 1
        if expansions > max_expand:
            raise RuntimeError("could not bracket root")
        if f_lo > 0:
            hi, f_hi = lo, f_lo
            lo /= growth
        else:
            lo, f_lo = hi, f_hi
            hi *= growth
        f_lo, f_hi = h(lo), h(hi)

    R_star = brentq(h, lo, hi, xtol=1e-10, rtol=1e-12)
    return R_star, h(R_star)


# ============================================================================
# 3. Equation 1 -- the 1-D problem at R_perp = 0
# ============================================================================


def K_uv_para(R_par: float, g: RegularGraph) -> float:
    """K_uv(R_par, 0) = c*N * max_{|alpha| <= R_par} [
           sech^4(z) + (4 u(alpha)^2 / N) tanh^2(z) sech^4(z) ]
       where z = alpha / sqrt(N),  u(alpha) = u_star + sqrt(R_par^2 - alpha^2).

    K_uu(R_par, 0) is identically zero here (Q @ 1 = 0 and x is a multiple
    of the all-ones vector when beta = 0), so this scalar maximisation is
    the entire content of equation 1.
    """

    def objective(alpha: float) -> float:
        u = g.u_star + np.sqrt(max(R_par**2 - alpha**2, 0.0))
        t = np.tanh(alpha / np.sqrt(g.N))
        sech4 = (1.0 - t**2) ** 2
        return sech4 + (4.0 * u**2 / g.N) * t**2 * sech4

    alpha_star = maximize_scalar(objective, -R_par, R_par) if R_par > 0 else 0.0
    return g.Mpp * np.sqrt(g.c * g.N * objective(alpha_star))


# ============================================================================
# 4. Equation 2 -- the (N-1)-D problem at R_par = 0
# ============================================================================


def project_to_ball(y: np.ndarray, R: float) -> np.ndarray:
    """Project y onto the feasible set {1^T y = 0, ||y||_2 <= R}."""
    y = y - y.mean()
    norm = np.linalg.norm(y)
    return y * (R / norm) if norm > R > 0 else y


def s2_quadratic_form(y: np.ndarray, g: RegularGraph):
    """f(y) = s2(y)^T M_vv s2(y) and its gradient, where
    s2(y) = -2 tanh(y) sech^2(y)  and  s2'(y) = 2 sech^2(y) (2 tanh^2(y) - sech^2(y))."""
    t = np.tanh(y)
    sech2 = 1.0 - t**2
    s2 = -2.0 * t * sech2
    Mv = g.M_vv @ s2
    f = s2 @ Mv
    ds2 = 2.0 * sech2 * (2.0 * t**2 - sech2)
    grad = 2.0 * ds2 * Mv
    return f, grad


def top_eigenvector_excluding_uniform(g: RegularGraph) -> np.ndarray:
    """The eigenvector of M_vv restricted to {1^T y = 0} with the largest
    eigenvalue -- used as a warm start for K_vv_perp. The all-ones vector
    is an exact eigenvector of M_vv (since Q @ 1 = 0), so it is identified
    by maximal overlap and excluded; the chosen eigenvector is then
    explicitly deflated against it as a guard against near-degenerate
    eigenvalues, where eigh's basis choice within the degenerate subspace
    could otherwise leave a small residual uniform component."""
    eigvals, eigvecs = np.linalg.eigh(g.M_vv)
    uniform = np.ones(g.N) / np.sqrt(g.N)
    uniform_idx = int(np.argmax(np.abs(eigvecs.T @ uniform)))
    candidate_idx = max((i for i in range(g.N) if i != uniform_idx), key=lambda i: eigvals[i])
    v = eigvecs[:, candidate_idx]
    v = v - uniform * (uniform @ v)
    return v / np.linalg.norm(v)


def K_vv_perp(R_perp: float, g: RegularGraph, n_restarts: int = 8, iters: int = 400, seed: int = 0) -> float:
    """K_vv(0, R_perp) = u_*^2 * max_{y _|_ 1, ||y|| <= R_perp} s2(y)^T M_vv s2(y),
    found by projected-gradient ascent (Adam) from several restarts: the
    top eigenvector of M_vv restricted to {1^T y = 0} as a warm start
    (exact, via eigh on the explicit matrix we already built), plus a
    handful of random restarts to guard against local maxima."""
    if R_perp == 0.0:
        return 0.0

    v_top = top_eigenvector_excluding_uniform(g)
    rng = np.random.default_rng(seed)
    starts = [R_perp * v_top, -R_perp * v_top]
    for _ in range(n_restarts):
        z = rng.normal(size=g.N)
        z -= z.mean()
        z *= R_perp * rng.uniform(0.3, 1.0) / np.linalg.norm(z)
        starts.append(z)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = 0.05 * max(R_perp, 1e-3)

    best_f = -np.inf
    for y0 in starts:
        y = project_to_ball(y0.copy(), R_perp)
        m, v = np.zeros(g.N), np.zeros(g.N)
        for t in range(1, iters + 1):
            _, grad = s2_quadratic_form(y, g)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            step = lr * (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps)
            y = project_to_ball(y + step, R_perp)
        f_final, _ = s2_quadratic_form(y, g)
        best_f = max(best_f, f_final)

    return g.Mpp * np.sqrt(g.u_star**2 * best_f)


# ============================================================================
# 5. Mathematical integrity checks
# ============================================================================


def verify_integrity(g: RegularGraph, R_par: float, par_residual: float, R_perp: float, perp_residual: float) -> None:
    """Cross-checks on both the matrix construction and the solved roots:
    structural identities that must hold exactly for a correctly built
    K-regular graph, plus fixed-point residuals that must vanish at a
    correctly solved root."""
    issues = []

    if not np.allclose(g.A, g.A.T):
        issues.append("A is not symmetric")
    if np.any(np.diag(g.A) != 0):
        issues.append("A has self-loops")
    if np.any((g.A != 0) & (g.A != 1)):
        issues.append("A is not a 0/1 matrix")

    scale = max(g.c, 1.0)
    q_ones_residual = np.abs(g.Q @ np.ones(g.N)).max()
    if q_ones_residual > 1e-6 * scale:
        issues.append(f"Q @ 1 != 0 (max residual {q_ones_residual:.3e})")

    diag_spread = np.std(np.diag(g.Q))
    if diag_spread > 1e-6 * scale:
        issues.append(f"diag(Q) is not constant (std {diag_spread:.3e})")

    if abs(par_residual) > 1e-6:
        issues.append(f"R_par fixed-point residual too large: {par_residual:.3e}")
    if abs(perp_residual) > 1e-3:
        issues.append(f"R_perp fixed-point residual too large: {perp_residual:.3e}")

    if issues:
        raise AssertionError("mathematical integrity checks failed:\n  " + "\n  ".join(issues))


# ============================================================================
# 6. Entry point
# ============================================================================


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python solve.py <graph6 string>", file=sys.stderr)
        sys.exit(1)

    try:
        g = RegularGraph(sys.argv[1])
        # print(f"N={g.N} K={g.K} c={g.c:.8f}")

        R_para, para_residual = solve_fixed_point_equation(lambda R: K_uv_para(R, g), lo=1e-6, hi=10.0)
        # print(f"R_para* = {R_para:.8f}   (F_para(R*,0) = {1.0 / K_uv_para(R_para, g):.8f})")

        R_perp, perp_residual = solve_fixed_point_equation(lambda R: K_vv_perp(R, g), lo=1e-3, hi=5.0)
        # print(f"R_perp* = {R_perp:.8f}   (F_perp(0,R*) = {1.0 / K_vv_perp(R_perp, g):.8f})")

        verify_integrity(g, R_para, para_residual, R_perp, perp_residual)
        # print("integrity checks passed.")
        print(f"{R_para:.6f} {R_perp:.6f}")
    except (ValueError, RuntimeError, AssertionError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
