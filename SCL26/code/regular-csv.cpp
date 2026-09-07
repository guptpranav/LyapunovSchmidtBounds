#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <omp.h>

#include <Eigen/Dense>
#include <nlopt.hpp>

const int N_RESTARTS = 4;
const double OPT_TOL = 1e-6;
const std::vector<std::string> OBJECTIVES = {"uu", "uv", "vv"};

// Memory structure allocated ONCE per optimization chain to avoid heap contention
struct OptData {
    double r_para;
    double r_perp;
    std::string objective;
    double u_star;
    const Eigen::MatrixXd &W;
    const Eigen::MatrixXd &M;
    int N;

    // Pre-allocated scratch buffers
    Eigen::VectorXd x_vec;
    Eigen::VectorXd th;
    Eigen::VectorXd s1;
    Eigen::VectorXd s2;
    Eigen::VectorXd s3;
    Eigen::VectorXd M_s1;
    Eigen::VectorXd M_s2;
    Eigen::VectorXd dK_dx;
};

struct ObjResult {
    double r_para;
    double r_perp;
    double K_max;
    double alpha_max;
    std::vector<double> parcel_beta;
    double u_max;
};

struct TaskResult {
    double r_para;
    double r_perp;
    double K_uu, K_uv, K_vv;
    ObjResult uu, uv, vv;
};

double compute_u(double alpha, double r_parallel, double u_star) {
    double val = (r_parallel * r_parallel) - (alpha * alpha);
    return u_star + std::sqrt(std::max(0.0, val));
}

// Analytical Constraint Gradient
double beta_constraint(const std::vector<double> &x, std::vector<double> &grad, void *data) {
    OptData *d = static_cast<OptData *>(data);
    if (!grad.empty()) {
        grad[0] = 0.0; // d/dalpha = 0
        for (int i = 1; i < d->N; ++i) {
            grad[i] = 2.0 * x[i]; // d/dbeta_i = 2 * beta_i
        }
    }
    double sum_sq = 0.0;
    for (size_t i = 1; i < x.size(); ++i) {
        sum_sq += x[i] * x[i];
    }
    return sum_sq - (d->r_perp * d->r_perp);
}

// Analytical Objective Gradient Engine
double zetas_obj(const std::vector<double> &x, std::vector<double> &grad, void *data) {
    OptData *d = static_cast<OptData *>(data);
    int N = d->N;
    double alpha = x[0];
    
    Eigen::Map<const Eigen::VectorXd> beta(x.data() + 1, N - 1);
    
    // Compute x without allocations
    d->x_vec.setConstant(alpha / std::sqrt(N));
    d->x_vec.noalias() += d->W * beta;

    double r_diff = d->r_para * d->r_para - alpha * alpha;
    double sqrt_r = (r_diff > 0.0) ? std::sqrt(r_diff) : 0.0;
    double u = d->u_star + sqrt_r;

    d->th = d->x_vec.array().tanh();
    d->s1 = 1.0 - d->th.array().square();
    d->s2 = -2.0 * d->th.array() * d->s1.array();

    d->M_s1.noalias() = d->M * d->s1;
    d->M_s2.noalias() = d->M * d->s2;

    double s1_M_s1 = d->s1.dot(d->M_s1);
    double s2_M_s2 = d->s2.dot(d->M_s2);

    double Kval = 0.0;
    if (d->objective == "uu") {
        Kval = (u * u / (N * N)) * s2_M_s2 + (2.0 / N) * s1_M_s1;
    } else if (d->objective == "uv") {
        Kval = (u * u / N) * s2_M_s2 + s1_M_s1;
    } else if (d->objective == "vv") {
        Kval = u * u * s2_M_s2;
    }

    // Execute analytical backend if gradient parameter is requested by SLSQP
    if (!grad.empty()) {
        d->s3 = 2.0 * d->s1.array() * (3.0 * d->th.array().square() - 1.0);

        if (d->objective == "uu") {
            d->dK_dx = (u * u / (N * N)) * 2.0 * d->s3.array() * d->M_s2.array() 
                     + (2.0 / N) * 2.0 * d->s2.array() * d->M_s1.array();
        } else if (d->objective == "uv") {
            d->dK_dx = (u * u / N) * 2.0 * d->s3.array() * d->M_s2.array() 
                     + 2.0 * d->s2.array() * d->M_s1.array();
        } else if (d->objective == "vv") {
            d->dK_dx = u * u * 2.0 * d->s3.array() * d->M_s2.array();
        }

        double dK_du = 0.0;
        if (d->objective == "uu") dK_du = (2.0 * u / (N * N)) * s2_M_s2;
        else if (d->objective == "uv") dK_du = (2.0 * u / N) * s2_M_s2;
        else if (d->objective == "vv") dK_du = 2.0 * u * s2_M_s2;

        double du_dalpha = (r_diff > 1e-12) ? (-alpha / sqrt_r) : 0.0;
        double dK_dalpha = dK_du * du_dalpha + d->dK_dx.sum() / std::sqrt(N);
        
        grad[0] = -dK_dalpha; 

        Eigen::Map<Eigen::VectorXd> grad_beta(grad.data() + 1, N - 1);
        grad_beta.noalias() = -d->W.transpose() * d->dK_dx;
    }

    return -Kval;
}

ObjResult optimize_objective(double r_para, double r_perp, const std::string& obj, std::mt19937& rng, 
                             double u_star, const Eigen::MatrixXd& W, const Eigen::MatrixXd& M, int N) {
    
    OptData opt_data = {r_para, r_perp, obj, u_star, W, M, N};
    opt_data.x_vec.resize(N); opt_data.th.resize(N); opt_data.s1.resize(N);
    opt_data.s2.resize(N); opt_data.s3.resize(N); opt_data.M_s1.resize(N);
    opt_data.M_s2.resize(N); opt_data.dK_dx.resize(N);

    std::vector<double> lb(N, -HUGE_VAL);
    std::vector<double> ub(N, HUGE_VAL);
    lb[0] = -r_para; ub[0] = r_para;

    double best_fun = HUGE_VAL;
    std::vector<double> best_params;

    // Switch to Gradient-Based SLSQP
    nlopt::opt opt(nlopt::LD_SLSQP, N);
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    opt.set_min_objective(zetas_obj, &opt_data);
    opt.add_inequality_constraint(beta_constraint, &opt_data, 1e-7);
    opt.set_ftol_abs(OPT_TOL);
    opt.set_maxeval(512); // Safety iteration cap matching SciPy defaults

    std::uniform_real_distribution<double> u_dist_alpha(-r_para, r_para);
    std::uniform_real_distribution<double> u_dist_beta(0.0, r_perp);
    std::normal_distribution<double> n_dist(0.0, 1.0);

    for (int restart = 0; restart < N_RESTARTS; ++restart) {
        std::vector<double> x(N, 0.0);
        x[0] = (r_para > 0) ? u_dist_alpha(rng) : 0.0;

        if (r_perp > 0) {
            double norm = 0.0;
            for (int i = 1; i < N; ++i) {
                x[i] = n_dist(rng);
                norm += x[i] * x[i];
            }
            norm = std::sqrt(norm) + 1e-12;
            double beta_mag = u_dist_beta(rng);
            for (int i = 1; i < N; ++i) {
                x[i] = (x[i] / norm) * beta_mag;
            }
        }

        double minf;
        try {
            nlopt::result res = opt.optimize(x, minf);
            bool feasible = (std::abs(x[0]) <= r_para + 1e-5);
            double b_norm = 0.0;
            for(int i=1; i<N; ++i) b_norm += x[i]*x[i];
            feasible = feasible && (std::sqrt(b_norm) <= r_perp + 1e-5);

            if (feasible && minf < best_fun) {
                best_fun = minf;
                best_params = x;
            }
        } catch (std::exception &e) {
            continue;
        }
    }

    if (best_params.empty()) {
        best_params.assign(N, 0.0);
        std::vector<double> empty_grad;
        best_fun = zetas_obj(best_params, empty_grad, &opt_data);
    }

    ObjResult res;
    res.r_para = r_para; res.r_perp = r_perp;
    res.K_max = std::sqrt(-best_fun);
    res.alpha_max = best_params[0];
    res.parcel_beta = std::vector<double>(best_params.begin() + 1, best_params.end());
    res.u_max = compute_u(best_params[0], r_para, u_star);
    return res;
}

Eigen::MatrixXd g6toA(const std::string& g6) {
    if (g6.empty())
        throw std::invalid_argument("Empty graph6 string");

    // Only supports n <= 62.
    unsigned char first = static_cast<unsigned char>(g6[0]);
    if (first == '~')
        throw std::invalid_argument("Only graph6 with <= 62 vertices is supported.");

    int n = first - 63;
    if (n < 0 || n > 62)
        throw std::invalid_argument("Invalid graph6 vertex count.");

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);

    int charPos = 1;
    int bitPos = 5;

    auto nextBit = [&]() -> int {
        if (charPos >= static_cast<int>(g6.size()))
            throw std::invalid_argument("Truncated graph6 string.");

        unsigned char value = static_cast<unsigned char>(g6[charPos]) - 63;
        int bit = (value >> bitPos) & 1;

        if (--bitPos < 0) {
            bitPos = 5;
            ++charPos;
        }

        return bit;
    };

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (nextBit()) {
                A(i, j) = 1.0;
                A(j, i) = 1.0;
            }
        }
    }

    return A;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <g6> <RPR_MAX> <RPR_STEP> <RPP_MAX> <RPP_STEP>\n";
        return 1;
    }

    std::string g6 = argv[1];
    double rpr_max = std::stod(argv[2]); double rpr_step = std::stod(argv[3]);
    double rpp_max = std::stod(argv[4]); double rpp_step = std::stod(argv[5]);

    Eigen::MatrixXd A = g6toA(g6);
    int N = A.rows();

    Eigen::VectorXd degrees = A.rowwise().sum();
    int K = std::round(degrees(0));
    double U_STAR = 1.0 / K;

    std::vector<std::pair<double, double>> grid;
    for (double r_para = rpr_step; r_para <= rpr_max + 1e-9; r_para += rpr_step) {
        for (double r_perp = rpp_step; r_perp <= rpp_max + 1e-9; r_perp += rpp_step) {
            grid.push_back({r_para, r_perp});
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    Eigen::MatrixXd evecs = es.eigenvectors().rowwise().reverse();
    Eigen::MatrixXd W = evecs.rightCols(N - 1);

    Eigen::MatrixXd J = Eigen::MatrixXd::Ones(N, N);
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(N, N) - J / N;
    Eigen::MatrixXd Q = A * A - ((K * K) / (double)N) * J;

    Eigen::MatrixXd M_uu = Q;
    Eigen::MatrixXd M_uv = (P.array() * Q.array()).matrix();
    Eigen::MatrixXd M_vv = (Q.array() * P.array().square()).matrix();

    std::cout << "\n--- Starting Gradient-Informed SLSQP Solver ---\n";
    std::cout << "Grid Size: " << grid.size() << " | Parallel Worker Threads: " << omp_get_max_threads() << "\n";

    std::vector<TaskResult> all_results(grid.size());

    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < grid.size(); ++i) {
        double r_para = grid[i].first;
        double r_perp = grid[i].second;
        std::mt19937 rng(1337 + omp_get_thread_num() + i);

        TaskResult tr;
        tr.r_para = r_para; tr.r_perp = r_perp;
        tr.uu = optimize_objective(r_para, r_perp, "uu", rng, U_STAR, W, M_uu, N);
        tr.uv = optimize_objective(r_para, r_perp, "uv", rng, U_STAR, W, M_uv, N);
        tr.vv = optimize_objective(r_para, r_perp, "vv", rng, U_STAR, W, M_vv, N);
        
        tr.K_uu = tr.uu.K_max; tr.K_uv = tr.uv.K_max; tr.K_vv = tr.vv.K_max;
        all_results[i] = tr;
    }

    // auto write_obj_csv = [&](const std::string& filename, const std::string& obj_type) {
    //     std::ofstream f(filename);
    //     f << "R_parallel,R_perp,K_max,alpha_max,beta_max,u_max\n";
    //     for (const auto& tr : all_results) {
    //         const ObjResult* obj = (obj_type == "uu") ? &tr.uu : ((obj_type == "uv") ? &tr.uv : &tr.vv);
    //         f << std::fixed << std::setprecision(3) << obj->r_para << "," << obj->r_perp << ",";
    //         f << std::fixed << std::setprecision(6) << obj->K_max << "," << obj->alpha_max << ",\"[[";
    //         for (size_t j = 0; j < obj->parcel_beta.size(); ++j) {
    //             f << obj->parcel_beta[j] << (j == obj->parcel_beta.size() - 1 ? "" : ",");
    //         }
    //         f << "]]\"," << obj->u_max << "\n";
    //     }
    // };

    // write_obj_csv("Kuu.csv", "uu");
    // write_obj_csv("Kuv.csv", "uv");
    // write_obj_csv("Kvv.csv", "vv");

    std::ofstream fks("Ks.csv");
    fks << "R_parallel,R_perp,K_uu,K_uv,K_vv\n";
    for (const auto& tr : all_results) {
        fks << std::fixed << std::setprecision(3) << tr.r_para << "," << tr.r_perp << ",";
        fks << std::fixed << std::setprecision(6) << tr.K_uu << "," << tr.K_uv << "," << tr.K_vv << "\n";
    }
    
    std::cout << "------ Execution complete. CSVs written ------";

    // double rprmax = 0.0;
    // double rppmax = 0.0;

    // Eigen::VectorXd eigvals = es.eigenvalues();

    // double Mpp = std::sqrt(K * ((K - eigvals.head(eigvals.size() - 1).array()).inverse().square()).sum());

    // for (const auto &tr : all_results)
    // {
    //     double rpr = std::min(tr.r_para, 1.0 / (Mpp * (tr.K_uv + std::sqrt(tr.K_uu * tr.K_vv))));
    //     if (rpr > rprmax) {rprmax = rpr;}

    //     double rpp = std::min(tr.r_perp, 1.0 / (Mpp * tr.K_vv));
    //     if (rpp > rppmax) {rppmax = rpp;}
    // }

    // std::cout << rprmax << "\t" << rppmax;
    
    return 0;
}