/*
 * solve.cpp
 *
 * C++ translation of rmax.py using Eigen.
 * Solves the pair of self-consistent fixed-point equations for a K-regular graph.
 */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <stdexcept>
#include <functional>
#include <iomanip>
#include <Eigen/Dense>

// ============================================================================
// 1. Graph construction & validation
// ============================================================================

class RegularGraph {
public:
    Eigen::MatrixXd A;
    int N;
    int K;
    double c;
    double u_star;
    double Mpp;
    Eigen::MatrixXd Q, M_uv, M_vv;

    RegularGraph(const std::string& graph6) {
        std::string text = graph6;
        
        // Strip ">>graph6<<"
        std::string prefix = ">>graph6<<";
        if (text.compare(0, prefix.length(), prefix) == 0) {
            text = text.substr(prefix.length());
        }
        
        // Strip trailing whitespace
        while (!text.empty() && std::isspace(text.back())) {
            text.pop_back();
        }

        parse_graph6(text);
        K = check_connected_and_regular();
        c = static_cast<double>(K) * (N - K) / N;
        u_star = 1.0 / K;
        
        build_matrices();
        
        // Eigenvalues for Mpp
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
        Eigen::VectorXd eigvals = es.eigenvalues();
        
        double sum_inv_sq = 0.0;
        // Skip the largest eigenvalue (which is exactly K for connected K-regular graph)
        for (int i = 0; i < N - 1; ++i) {
            sum_inv_sq += 1.0 / std::pow(K - eigvals[i], 2);
        }
        Mpp = K * std::sqrt(sum_inv_sq);
    }

private:
    void parse_graph6(const std::string& text) {
        if (text.empty()) throw std::invalid_argument("Empty graph6 string");

        int i = 0;
        if (text[i] - 63 != 63) {
            N = text[i++] - 63;
        } else {
            if (text.length() < 4) throw std::invalid_argument("Invalid graph6 length");
            N = ((text[1] - 63) << 12) | ((text[2] - 63) << 6) | (text[3] - 63);
            i += 4;
        }

        A = Eigen::MatrixXd::Zero(N, N);
        int bit_idx = 5;
        for (int col = 1; col < N; ++col) {
            for (int row = 0; row < col; ++row) {
                if (i >= text.length()) throw std::invalid_argument("graph6 string too short");
                
                if ((text[i] - 63) & (1 << bit_idx)) {
                    A(row, col) = 1.0;
                    A(col, row) = 1.0;
                }
                
                if (bit_idx == 0) {
                    bit_idx = 5;
                    i++;
                } else {
                    bit_idx--;
                }
            }
        }
    }

    int check_connected_and_regular() {
        Eigen::VectorXd degrees = A.rowwise().sum();
        int first_deg = static_cast<int>(std::round(degrees[0]));
        
        for (int i = 1; i < N; ++i) {
            if (static_cast<int>(std::round(degrees[i])) != first_deg) {
                throw std::invalid_argument("graph must be regular");
            }
        }
        
        // Check connectivity (simple BFS)
        std::vector<bool> visited(N, false);
        std::vector<int> q = {0};
        visited[0] = true;
        int count = 1;
        while (!q.empty()) {
            int u = q.back();
            q.pop_back();
            for (int v = 0; v < N; ++v) {
                if (A(u, v) == 1.0 && !visited[v]) {
                    visited[v] = true;
                    q.push_back(v);
                    count++;
                }
            }
        }
        if (count != N) {
            throw std::invalid_argument("graph6 input must describe a connected graph");
        }
        
        return first_deg;
    }

    void build_matrices() {
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N, N);
        Eigen::MatrixXd J = Eigen::MatrixXd::Ones(N, N);
        
        Q = A * A - (std::pow(K, 2) / N) * J;
        M_uv = c * I - Q / N;
        M_vv = (1.0 - 2.0 / N) * c * I + Q / std::pow(N, 2);
    }
};

// ============================================================================
// 2. Generic numerical utilities (shared by both equations)
// ============================================================================

double maximize_scalar(const std::function<double(double)>& f, double lo, double hi, int grid_points = 400, int polish_iters = 100) {
    double max_val = -std::numeric_limits<double>::infinity();
    int best = 0;
    double cell = (hi - lo) / grid_points;
    std::vector<double> grid(grid_points + 1);
    
    for (int i = 0; i <= grid_points; ++i) {
        grid[i] = lo + i * cell;
        double val = f(grid[i]);
        if (val > max_val) {
            max_val = val;
            best = i;
        }
    }
    
    double a = std::max(lo, grid[best] - cell);
    double b = std::min(hi, grid[best] + cell);

    double invphi = (std::sqrt(5.0) - 1.0) / 2.0;
    double x1 = b - invphi * (b - a);
    double x2 = a + invphi * (b - a);
    double f1 = f(x1);
    double f2 = f(x2);

    for (int i = 0; i < polish_iters; ++i) {
        if (f1 < f2) {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + invphi * (b - a);
            f2 = f(x2);
        } else {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - invphi * (b - a);
            f1 = f(x1);
        }
    }
    return (f1 >= f2) ? x1 : x2;
}

// C++ Implementation of Brent's method (equivalent to scipy.optimize.brentq)
double brentq(const std::function<double(double)>& f, double a, double b, double xtol = 1e-10, double rtol = 1e-12, int maxiter = 100) {
    double fa = f(a), fb = f(b);
    if (fa * fb > 0.0) throw std::runtime_error("Root not bracketed");
    
    double c = a, fc = fa, d = 0.0, e = 0.0;
    
    for (int iter = 0; iter < maxiter; ++iter) {
        if (std::abs(fc) < std::abs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }
        
        double tol1 = 2.0 * rtol * std::abs(b) + 0.5 * xtol;
        double m = 0.5 * (c - b);
        
        if (std::abs(m) <= tol1 || fb == 0.0) return b;
        
        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            double s = fb / fa, p, q, r;
            if (a == c) {
                p = 2.0 * m * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                r = fb / fc;
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            p = std::abs(p);
            
            double min1 = 3.0 * m * q - std::abs(tol1 * q);
            double min2 = std::abs(e * q);
            
            if (2.0 * p < (min1 < min2 ? min1 : min2)) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        } else {
            d = m;
            e = m;
        }
        
        a = b;
        fa = fb;
        if (std::abs(d) > tol1) b += d;
        else b += (m > 0.0 ? tol1 : -tol1);
        fb = f(b);
        
        if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
            c = a;
            fc = fa;
            e = d = b - a;
        }
    }
    return b;
}

std::pair<double, double> solve_fixed_point_equation(const std::function<double(double)>& K_of_R, double lo, double hi, double growth = 2.0, int max_expand = 60) {
    auto h = [&](double R) {
        double k = K_of_R(R);
        return R - (k > 0.0 ? 1.0 / k : std::numeric_limits<double>::infinity());
    };

    double f_lo = h(lo);
    double f_hi = h(hi);
    int expansions = 0;
    
    while (f_lo * f_hi > 0.0) {
        expansions++;
        if (expansions > max_expand) {
            throw std::runtime_error("could not bracket root");
        }
        if (f_lo > 0.0) {
            hi = lo;
            f_hi = f_lo;
            lo /= growth;
        } else {
            lo = hi;
            f_lo = f_hi;
            hi *= growth;
        }
        f_lo = h(lo);
        f_hi = h(hi);
    }

    double R_star = brentq(h, lo, hi);
    return {R_star, h(R_star)};
}

// ============================================================================
// 3. Equation 1 -- the 1-D problem at R_perp = 0
// ============================================================================

double K_uv_para(double R_par, const RegularGraph& g) {
    auto objective = [&](double alpha) {
        double inner = std::max(R_par * R_par - alpha * alpha, 0.0);
        double u = g.u_star + std::sqrt(inner);
        double t = std::tanh(alpha / std::sqrt(g.N));
        double sech4 = std::pow(1.0 - t * t, 2);
        return sech4 + (4.0 * u * u / g.N) * t * t * sech4;
    };

    double alpha_star = (R_par > 0.0) ? maximize_scalar(objective, -R_par, R_par) : 0.0;
    return g.Mpp * std::sqrt(g.c * g.N * objective(alpha_star));
}

// ============================================================================
// 4. Equation 2 -- the (N-1)-D problem at R_par = 0
// ============================================================================

Eigen::VectorXd project_to_ball(Eigen::VectorXd y, double R) {
    y.array() -= y.mean();
    double norm = y.norm();
    if (norm > R && R > 0.0) {
        y *= (R / norm);
    }
    return y;
}

void s2_quadratic_form(const Eigen::VectorXd& y, const RegularGraph& g, double& f, Eigen::VectorXd& grad) {
    Eigen::ArrayXd t = y.array().tanh();
    Eigen::ArrayXd sech2 = 1.0 - t.square();
    Eigen::VectorXd s2 = (-2.0 * t * sech2).matrix();
    
    Eigen::VectorXd Mv = g.M_vv * s2;
    f = s2.dot(Mv);
    
    Eigen::ArrayXd ds2 = 2.0 * sech2 * (2.0 * t.square() - sech2);
    grad = 2.0 * ds2 * Mv.array();
}

Eigen::VectorXd top_eigenvector_excluding_uniform(const RegularGraph& g) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(g.M_vv);
    Eigen::VectorXd eigvals = es.eigenvalues();
    Eigen::MatrixXd eigvecs = es.eigenvectors();

    Eigen::VectorXd uniform = Eigen::VectorXd::Constant(g.N, 1.0 / std::sqrt(g.N));
    
    int uniform_idx = -1;
    double max_overlap = -1.0;
    for (int i = 0; i < g.N; ++i) {
        double overlap = std::abs(eigvecs.col(i).dot(uniform));
        if (overlap > max_overlap) {
            max_overlap = overlap;
            uniform_idx = i;
        }
    }

    int candidate_idx = -1;
    double max_val = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < g.N; ++i) {
        if (i != uniform_idx && eigvals[i] > max_val) {
            max_val = eigvals[i];
            candidate_idx = i;
        }
    }

    Eigen::VectorXd v = eigvecs.col(candidate_idx);
    v -= uniform * uniform.dot(v);
    return v / v.norm();
}

double K_vv_perp(double R_perp, const RegularGraph& g, int n_restarts = 8, int iters = 400, unsigned int seed = 0) {
    if (R_perp == 0.0) return 0.0;

    Eigen::VectorXd v_top = top_eigenvector_excluding_uniform(g);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> norm_dist(0.0, 1.0);
    std::uniform_real_distribution<double> unif_dist(0.3, 1.0);

    std::vector<Eigen::VectorXd> starts;
    starts.push_back(R_perp * v_top);
    starts.push_back(-R_perp * v_top);

    for (int i = 0; i < n_restarts; ++i) {
        Eigen::VectorXd z(g.N);
        for (int j = 0; j < g.N; ++j) z[j] = norm_dist(rng);
        z.array() -= z.mean();
        z *= R_perp * unif_dist(rng) / z.norm();
        starts.push_back(z);
    }

    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    double lr = 0.05 * std::max(R_perp, 1e-3);
    double best_f = -std::numeric_limits<double>::infinity();

    for (const auto& y0 : starts) {
        Eigen::VectorXd y = project_to_ball(y0, R_perp);
        Eigen::VectorXd m = Eigen::VectorXd::Zero(g.N);
        Eigen::VectorXd v = Eigen::VectorXd::Zero(g.N);

        for (int t = 1; t <= iters; ++t) {
            double f;
            Eigen::VectorXd grad;
            s2_quadratic_form(y, g, f, grad);
            
            m = beta1 * m + (1.0 - beta1) * grad;
            v = beta2 * v + (1.0 - beta2) * grad.array().square().matrix();
            
            double b1_t = 1.0 - std::pow(beta1, t);
            double b2_t = 1.0 - std::pow(beta2, t);
            
            Eigen::VectorXd m_hat = m / b1_t;
            Eigen::VectorXd v_hat = v / b2_t;
            
            Eigen::VectorXd step = lr * (m_hat.array() / (v_hat.array().sqrt() + eps)).matrix();
            y = project_to_ball(y + step, R_perp);
        }
        
        double f_final;
        Eigen::VectorXd dummy_grad;
        s2_quadratic_form(y, g, f_final, dummy_grad);
        if (f_final > best_f) best_f = f_final;
    }

    return g.Mpp * std::sqrt(std::pow(g.u_star, 2) * best_f);
}

// ============================================================================
// 5. Mathematical integrity checks
// ============================================================================

void verify_integrity(const RegularGraph& g, double R_par, double par_residual, double R_perp, double perp_residual) {
    std::vector<std::string> issues;

    if (!g.A.isApprox(g.A.transpose(), 1e-10)) {
        issues.push_back("A is not symmetric");
    }
    
    if (g.A.diagonal().cwiseAbs().maxCoeff() != 0) {
        issues.push_back("A has self-loops");
    }
    
    // Check 0/1 logic strictly
    for (int i = 0; i < g.N; ++i) {
        for (int j = 0; j < g.N; ++j) {
            if (g.A(i,j) != 0.0 && g.A(i,j) != 1.0) {
                issues.push_back("A is not a 0/1 matrix");
                goto done_01_check;
            }
        }
    }
done_01_check:

    double scale = std::max(g.c, 1.0);
    double q_ones_residual = (g.Q * Eigen::VectorXd::Ones(g.N)).cwiseAbs().maxCoeff();
    
    if (q_ones_residual > 1e-6 * scale) {
        issues.push_back("Q @ 1 != 0");
    }

    Eigen::VectorXd diag = g.Q.diagonal();
    double mean_diag = diag.mean();
    double diag_spread = std::sqrt((diag.array() - mean_diag).square().sum() / g.N);
    
    if (diag_spread > 1e-6 * scale) {
        issues.push_back("diag(Q) is not constant");
    }

    if (std::abs(par_residual) > 1e-6) {
        issues.push_back("R_par fixed-point residual too large");
    }
    if (std::abs(perp_residual) > 1e-3) {
        issues.push_back("R_perp fixed-point residual too large");
    }

    if (!issues.empty()) {
        std::string err = "mathematical integrity checks failed:\n";
        for (const auto& issue : issues) err += "  " + issue + "\n";
        throw std::runtime_error(err);
    }
}

// ============================================================================
// 6. Entry point
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: ./solve <graph6 string>\n";
        return 1;
    }

    try {
        RegularGraph g(argv[1]);

        auto par_result = solve_fixed_point_equation([&g](double R) { return K_uv_para(R, g); }, 1e-6, 10.0);
        double R_para = par_result.first;
        double para_residual = par_result.second;

        auto perp_result = solve_fixed_point_equation([&g](double R) { return K_vv_perp(R, g); }, 1e-3, 5.0);
        double R_perp = perp_result.first;
        double perp_residual = perp_result.second;

        verify_integrity(g, R_para, para_residual, R_perp, perp_residual);

        std::cout << std::fixed << std::setprecision(6) << R_para << " " << R_perp << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}