% run_all_experiments.m  (v3)
% Unified tolerance eps=1e-8 for NewtonC, NewtonM, SPM, SIPM.
% NewtonC/NewtonM: our implementation (NewtonM has 'safeguard' fallback).
% SPM/SIPM: our implementation, plus reference iteration counts from literature.
% High-dim tensors: SCF (smallest-eigenvector branch) initialization + Newton refine.

clear; clc;
fprintf('=== Running all experiments with eps=1e-8 ===\n\n');

eps_tol = 1e-8;
Kmax_newton = 200;
Kmax_power = 3000;
c = 1;
Ns = 50;
rng(123);

%% ===== Part 1: C-eigenvalue examples (8 piezoelectric materials) =====
materials = {'VFeSb', 'SiO2', 'Cr2AgBiO8', 'RbTaO3', 'NaBiS2', 'LiBiB2O5', 'KBi2F7', 'BaNiO3'};
ref_lambda = [4.2514, 0.1375, 2.6258, 13.6381, 11.6674, 7.7376, 13.5021, 27.4628];
spm_iter = [42, 750, 1718, 35, 39, 35, 54, 45];      % Zhao 2023 Table 2 (eps=1e-8)
spm_res  = [8.0e-9, 1.0e-8, 1.0e-8, 9.3e-9, 8.2e-9, 9.8e-9, 8.7e-9, 7.5e-9];

n_mat = length(materials);
results_C = struct();

for mi = 1:n_mat
    mat = materials{mi};
    fprintf('--- C-eigenvalue: %s ---\n', mat);
    A = make_piezo_tensor(mat);
    n = size(A, 1);

    best_lam = 0; best_iter = 0; best_res = 0;
    all_eig = []; all_it = []; all_res = [];
    for s = 1:Ns
        xs = randn(n, 1); xs = xs / norm(xs);
        ys = randn(n, 1); ys = ys / norm(ys);
        [lam_s, ~, ~, iter_s, res_hist_s] = NewtonC(A, c, xs, ys, eps_tol, Kmax_newton);
        res_s = res_hist_s(end);

        if abs(lam_s) > abs(best_lam)
            best_lam = lam_s; best_iter = iter_s; best_res = res_s;
        end

        is_new = true;
        for e = 1:length(all_eig)
            if abs(abs(lam_s) - abs(all_eig(e))) < 1e-4
                is_new = false; break;
            end
        end
        if is_new
            all_eig = [all_eig, lam_s];
            all_it = [all_it, iter_s];
            all_res = [all_res, res_s];
        end
    end

    [~, sort_idx] = sort(abs(all_eig), 'descend');
    all_eig = all_eig(sort_idx);
    all_it = all_it(sort_idx);
    all_res = all_res(sort_idx);

    fprintf('  NewtonC (largest |lam|): lambda=%.4f, iter=%d, res=%.2e\n', best_lam, best_iter, best_res);
    fprintf('  SPM (ref):               lambda=%.4f, iter=%d, res=%.2e\n', ref_lambda(mi), spm_iter(mi), spm_res(mi));
    fprintf('  All found C-eigenvalues (%d):\n', length(all_eig));
    for e = 1:length(all_eig)
        fprintf('    %.4f  iter=%d  res=%.2e\n', all_eig(e), all_it(e), all_res(e));
    end

    results_C(mi).material = mat;
    results_C(mi).ref_lambda = ref_lambda(mi);
    results_C(mi).lambda_NewtonC = best_lam;
    results_C(mi).iter_NewtonC = best_iter;
    results_C(mi).res_NewtonC = best_res;
    results_C(mi).lambda_SPM = ref_lambda(mi);
    results_C(mi).iter_SPM = spm_iter(mi);
    results_C(mi).res_SPM = spm_res(mi);
    results_C(mi).all_eigenvalues = all_eig;
    results_C(mi).all_iters = all_it;
    results_C(mi).all_residuals = all_res;
end

%% ===== Part 2: M-eigenvalue examples =====
results_M = struct();

% --- Example 6.6: Tetragonal system ---
fprintf('\n--- M-eigenvalue: Tetragonal ---\n');
A_tet = make_psym_tensor('tetragonal');
m = size(A_tet, 1); n = size(A_tet, 2);

[x0_tet, y0_tet] = svd_init(A_tet, 'smallest');
fprintf('  SVD init: x0=[%.4f %.4f %.4f], y0=[%.4f %.4f %.4f]\n', x0_tet, y0_tet);
[lam_init, x_init, y_init, iter_init, res_init] = NewtonM(A_tet, c, x0_tet, y0_tet, eps_tol, Kmax_newton);
fprintf('  NewtonM (SVD init): lambda=%.4f, iter=%d, res=%.2e\n', lam_init, iter_init, res_init(end));
fprintf('  eigenpair: x=[%.4f %.4f %.4f], y=[%.4f %.4f %.4f]\n', x_init, y_init);

all_M_eig = []; all_M_it = []; all_M_res = []; all_M_x = {}; all_M_y = {};
for s = 1:Ns
    xs = randn(m, 1); xs = xs / norm(xs);
    ys = randn(n, 1); ys = ys / norm(ys);
    [lam_s, x_s, y_s, iter_s, res_s_hist] = NewtonM(A_tet, c, xs, ys, eps_tol, Kmax_newton, 'safeguard');
    res_s = res_s_hist(end);
    is_new = true;
    for e = 1:length(all_M_eig)
        if abs(lam_s - all_M_eig(e)) < 1e-4
            is_new = false; break;
        end
    end
    if is_new
        all_M_eig = [all_M_eig, lam_s];
        all_M_it = [all_M_it, iter_s];
        all_M_res = [all_M_res, res_s];
        all_M_x{end+1} = x_s;
        all_M_y{end+1} = y_s;
    end
end
[~, sort_idx] = sort(all_M_eig);
all_M_eig = all_M_eig(sort_idx);
all_M_it = all_M_it(sort_idx);
all_M_res = all_M_res(sort_idx);
fprintf('  All M-eigenvalues (%d):\n', length(all_M_eig));
for e = 1:length(all_M_eig)
    fprintf('    %.4f  iter=%d  res=%.2e\n', all_M_eig(e), all_M_it(e), all_M_res(e));
end
results_M.tetragonal.all_eig = all_M_eig;
results_M.tetragonal.all_iters = all_M_it;
results_M.tetragonal.all_residuals = all_M_res;
results_M.tetragonal.all_x = all_M_x;
results_M.tetragonal.all_y = all_M_y;
results_M.tetragonal.svd_init_lambda = lam_init;
results_M.tetragonal.svd_init_iter = iter_init;
results_M.tetragonal.svd_init_res = res_init(end);

% --- Example 6.7: Random tensor from Wang 2009 ---
fprintf('\n--- M-eigenvalue: Random (Wang 2009) ---\n');
A_rand = make_psym_tensor('random_wang');
m2 = size(A_rand, 1); n2 = size(A_rand, 2);

[x0_rand, y0_rand] = svd_init(A_rand, 'smallest');
tic;
[lam_Nrand, ~, ~, iter_Nrand, res_Nrand] = NewtonM(A_rand, c, x0_rand, y0_rand, eps_tol, Kmax_newton);
t_Nrand = toc;
fprintf('  NewtonM (SVD init): lambda=%.4f, iter=%d, res=%.2e, time=%.4fs\n', lam_Nrand, iter_Nrand, res_Nrand(end), t_Nrand);

tic;
[lam_Srand, ~, ~, iter_Srand] = SIPM_M(A_rand, x0_rand, y0_rand, eps_tol, Kmax_power);
t_Srand = toc;
fprintf('  SIPM (SVD init):    lambda=%.4f, iter=%d, time=%.4fs\n', lam_Srand, iter_Srand, t_Srand);

results_M.random.lambda_NewtonM = lam_Nrand;
results_M.random.iter_NewtonM = iter_Nrand;
results_M.random.res_NewtonM = res_Nrand(end);
results_M.random.time_NewtonM = t_Nrand;
results_M.random.lambda_SIPM = lam_Srand;
results_M.random.iter_SIPM = iter_Srand;
results_M.random.time_SIPM = t_Srand;

% --- Example 6.8: Dolomite ---
fprintf('\n--- M-eigenvalue: Dolomite ---\n');
A_dol = make_psym_tensor('dolomite');
m3 = size(A_dol, 1); n3 = size(A_dol, 2);

[x0_dol, y0_dol] = svd_init(A_dol, 'smallest');
tic;
[lam_Ndol, ~, ~, iter_Ndol, res_Ndol] = NewtonM(A_dol, c, x0_dol, y0_dol, eps_tol, Kmax_newton, 'safeguard');
t_Ndol = toc;
fprintf('  NewtonM (SVD init): lambda=%.4f, iter=%d, res=%.2e, time=%.4fs\n', lam_Ndol, iter_Ndol, res_Ndol(end), t_Ndol);

tic;
[lam_Sdol, ~, ~, iter_Sdol] = SIPM_M(A_dol, x0_dol, y0_dol, eps_tol, Kmax_power);
t_Sdol = toc;
fprintf('  SIPM (SVD init):    lambda=%.4f, iter=%d, time=%.4fs\n', lam_Sdol, iter_Sdol, t_Sdol);

best_lam_dol = inf; best_iter_dol = 0; best_res_dol = 0;
for s = 1:Ns
    xs = randn(m3, 1); xs = xs / norm(xs);
    ys = randn(n3, 1); ys = ys / norm(ys);
    [lam_s, ~, ~, iter_s, res_s_hist] = NewtonM(A_dol, c, xs, ys, eps_tol, Kmax_newton, 'safeguard');
    res_s = res_s_hist(end);
    if res_s <= eps_tol && lam_s < best_lam_dol
        best_lam_dol = lam_s; best_iter_dol = iter_s; best_res_dol = res_s;
    end
end
fprintf('  NewtonM (multi-start smallest): lambda=%.4f, iter=%d, res=%.2e\n', best_lam_dol, best_iter_dol, best_res_dol);

results_M.dolomite.lambda_NewtonM = lam_Ndol;
results_M.dolomite.iter_NewtonM = iter_Ndol;
results_M.dolomite.res_NewtonM = res_Ndol(end);
results_M.dolomite.time_NewtonM = t_Ndol;
results_M.dolomite.lambda_SIPM = lam_Sdol;
results_M.dolomite.iter_SIPM = iter_Sdol;
results_M.dolomite.time_SIPM = t_Sdol;
results_M.dolomite.multistart_lambda = best_lam_dol;
results_M.dolomite.multistart_iter = best_iter_dol;

%% ===== Part 3: High-dimensional random tensors =====
fprintf('\n--- M-eigenvalue: High-dimensional ---\n');
dims = [10 10; 10 15; 10 20; 15 20; 20 20];
results_M.highdim = struct();
rng(42);
for di = 1:size(dims, 1)
    m_d = dims(di, 1); n_d = dims(di, 2);

    A_hd = zeros(m_d, n_d, m_d, n_d);
    for i=1:m_d
        for j=1:n_d
            A_hd(:,:,i,j) = 2*rand(m_d,n_d) - 1;
        end
    end
    A_hd_ps = zeros(m_d, n_d, m_d, n_d);
    for i=1:m_d
        for j=1:n_d
            for k=1:m_d
                for l=1:n_d
                    A_hd_ps(i,j,k,l) = (A_hd(i,j,k,l) + A_hd(k,j,i,l) + A_hd(i,l,k,j) + A_hd(k,l,i,j)) / 4;
                end
            end
        end
    end

    % SCF initialization: alternate smallest-eigenvector iteration to a fixed
    % point (10 starts, keep the one with smallest Rayleigh quotient)
    rng(99);
    best_scf = inf; x_best = []; y_best = [];
    for s = 1:10
        x = randn(m_d,1); x = x/norm(x);
        y = randn(n_d,1); y = y/norm(y);
        for it = 1:1000
            My = zeros(m_d,m_d);
            for i=1:m_d, for kk=1:m_d, for j=1:n_d, for l=1:n_d
                My(i,kk) = My(i,kk) + A_hd_ps(i,j,kk,l)*y(j)*y(l);
            end, end, end, end
            [V, D] = eig(My, 'vector');
            [~, mi2] = min(D);
            x_new = V(:,mi2);
            Nx = zeros(n_d,n_d);
            for j=1:n_d, for l=1:n_d, for i=1:m_d, for kk=1:m_d
                Nx(j,l) = Nx(j,l) + A_hd_ps(i,j,kk,l)*x_new(i)*x_new(kk);
            end, end, end, end
            [W, E] = eig(Nx, 'vector');
            [~, mj2] = min(E);
            y_new = W(:,mj2);
            if (norm(x_new-x)<1e-12 && norm(y_new-y)<1e-12) || (norm(x_new+x)<1e-12 && norm(y_new+y)<1e-12)
                break;
            end
            x = x_new; y = y_new;
        end
        My = zeros(m_d,m_d);
        for i=1:m_d, for kk=1:m_d, for j=1:n_d, for l=1:n_d
            My(i,kk) = My(i,kk) + A_hd_ps(i,j,kk,l)*y(j)*y(l);
        end, end, end, end
        lam_scf = x' * My * x;
        if lam_scf < best_scf
            best_scf = lam_scf; x_best = x; y_best = y;
        end
    end

    tic;
    [lam_N, ~, ~, iter_N, res_N] = NewtonM(A_hd_ps, c, x_best, y_best, eps_tol, Kmax_newton, 'safeguard');
    t_N = toc;

    tic;
    [lam_S, ~, ~, iter_S] = SIPM_M(A_hd_ps, x_best, y_best, eps_tol, Kmax_power);
    t_S = toc;

    fprintf('  (%2d,%2d): NewtonM lam=%8.4f iter=%3d res=%.1e t=%.3fs | SIPM lam=%8.4f iter=%4d t=%.3fs\n', ...
        m_d, n_d, lam_N, iter_N, res_N(end), t_N, lam_S, iter_S, t_S);

    results_M.highdim(di).dims = [m_d, n_d];
    results_M.highdim(di).lambda_N = lam_N;
    results_M.highdim(di).iter_N = iter_N;
    results_M.highdim(di).res_N = res_N(end);
    results_M.highdim(di).time_N = t_N;
    results_M.highdim(di).lambda_S = lam_S;
    results_M.highdim(di).iter_S = iter_S;
    results_M.highdim(di).time_S = t_S;
end

%% ===== Convergence behavior (Tetragonal, random start) =====
fprintf('\n--- Convergence behavior (Tetragonal, random start) ---\n');
rng(2024);
x0_conv = randn(m, 1); x0_conv = x0_conv / norm(x0_conv);
y0_conv = randn(n, 1); y0_conv = y0_conv / norm(y0_conv);
[lam_conv, ~, ~, iter_conv, res_hist_conv] = NewtonM(A_tet, c, x0_conv, y0_conv, eps_tol, Kmax_newton);
fprintf('  lambda=%.4f, iter=%d\n', lam_conv, iter_conv);
for k = 1:length(res_hist_conv)
    fprintf('    iter %d: res = %.2e\n', k, res_hist_conv(k));
end
results_M.convergence = res_hist_conv;
results_M.convergence_lambda = lam_conv;

%% ===== Save =====
save('results_all.mat', 'results_C', 'results_M', 'eps_tol', 'c', 'Ns');
fprintf('\n=== All experiments done. Results saved. ===\n');

%% ===== Summary Tables =====
fprintf('\n=====================================================\n');
fprintf('TABLE 1: C-eigenvalue comparison (eps=1e-8)\n');
fprintf('=====================================================\n');
fprintf('%-12s | %-9s %-9s | %-4s %-5s | %-9s %-9s\n', ...
    'Tensor', 'NC_lam', 'SPM_lam', 'NC_it', 'SPM_it', 'NC_res', 'SPM_res');
fprintf('%s\n', repmat('-', 1, 70));
for mi = 1:n_mat
    r = results_C(mi);
    fprintf('%-12s | %-9.4f %-9.4f | %-4d %-5d | %-9.2e %-9.2e\n', ...
        r.material, r.lambda_NewtonC, r.lambda_SPM, ...
        r.iter_NewtonC, r.iter_SPM, r.res_NewtonC, r.res_SPM);
end

fprintf('\n=====================================================\n');
fprintf('All C-eigenvalues by material (Ns=%d)\n', Ns);
fprintf('=====================================================\n');
for mi = 1:n_mat
    r = results_C(mi);
    fprintf('%s:\n', r.material);
    for e = 1:length(r.all_eigenvalues)
        fprintf('  %.4f  iter=%d  res=%.2e\n', r.all_eigenvalues(e), r.all_iters(e), r.all_residuals(e));
    end
end

fprintf('\n=====================================================\n');
fprintf('M-eigenvalues: Tetragonal (SVD init + multi-start)\n');
fprintf('=====================================================\n');
fprintf('SVD init eigenpair: lam=%.4f iter=%d res=%.1e\n', lam_init, iter_init, res_init(end));
for e = 1:length(results_M.tetragonal.all_eig)
    fprintf('  %.4f  iter=%d  res=%.2e\n', results_M.tetragonal.all_eig(e), ...
        results_M.tetragonal.all_iters(e), results_M.tetragonal.all_residuals(e));
end

fprintf('\n=====================================================\n');
fprintf('M-eigenvalue: Random (Wang 2009) and Dolomite\n');
fprintf('=====================================================\n');
fprintf('Wang:     NewtonM lam=%.4f iter=%d res=%.1e t=%.4f | SIPM lam=%.4f iter=%d t=%.4f\n', ...
    results_M.random.lambda_NewtonM, results_M.random.iter_NewtonM, results_M.random.res_NewtonM, results_M.random.time_NewtonM, ...
    results_M.random.lambda_SIPM, results_M.random.iter_SIPM, results_M.random.time_SIPM);
fprintf('Dolomite: NewtonM lam=%.4f iter=%d res=%.1e t=%.4f | SIPM lam=%.4f iter=%d t=%.4f\n', ...
    results_M.dolomite.lambda_NewtonM, results_M.dolomite.iter_NewtonM, results_M.dolomite.res_NewtonM, results_M.dolomite.time_NewtonM, ...
    results_M.dolomite.lambda_SIPM, results_M.dolomite.iter_SIPM, results_M.dolomite.time_SIPM);

fprintf('\n=====================================================\n');
fprintf('High-dimensional\n');
fprintf('=====================================================\n');
fprintf('%-9s | %-9s %-5s %-9s %-7s | %-9s %-5s %-7s\n', ...
    '(m,n)', 'lam_N', 'iter', 'res_N', 'time_N', 'lam_S', 'iter', 'time_S');
for di = 1:length(results_M.highdim)
    h = results_M.highdim(di);
    fprintf('(%2d,%2d)   | %-9.4f %-5d %-9.1e %-7.3f | %-9.4f %-5d %-7.3f\n', ...
        h.dims(1), h.dims(2), h.lambda_N, h.iter_N, h.res_N, h.time_N, ...
        h.lambda_S, h.iter_S, h.time_S);
end

fprintf('\n=====================================================\n');
fprintf('Convergence history (Tetragonal random start)\n');
fprintf('=====================================================\n');
for k = 1:length(results_M.convergence)
    fprintf('  iter %d: %.2e\n', k, results_M.convergence(k));
end
