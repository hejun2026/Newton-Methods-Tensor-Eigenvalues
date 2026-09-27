% highdim_v4.m — High-dimensional random partially symmetric tensors
% Fix: unified rng(42) for tensor generation, 30 SCF starts (rng(2024)),
% Newton 1-step refine. Report SCF iterations + Newton steps + total time.
% Also run SIPM from SVD init (not SCF) for a fair comparison.

clear; clc;
eps_tol = 1e-8;
Kmax_newton = 200;
Kmax_power = 3000;
c = 1;

dims = [10 10; 10 15; 10 20; 15 20; 20 20];
fprintf('=== High-dimensional random tensors (v4: 30 SCF + Newton refine) ===\n');

for di = 1:size(dims,1)
    m_d = dims(di,1); n_d = dims(di,2);
    
    % Generate tensor with fixed rng(42)
    rng(42);
    A_raw = zeros(m_d, n_d, m_d, n_d);
    for i=1:m_d
        for j=1:n_d
            A_raw(:,:,i,j) = 2*rand(m_d,n_d) - 1;
        end
    end
    A_hd = zeros(m_d, n_d, m_d, n_d);
    for i=1:m_d
        for j=1:n_d
            for k=1:m_d
                for l=1:n_d
                    A_hd(i,j,k,l) = (A_raw(i,j,k,l) + A_raw(k,j,i,l) + A_raw(i,l,k,j) + A_raw(k,l,i,j)) / 4;
                end
            end
        end
    end
    
    % SCF: 30 starts (rng(2024)), keep smallest
    rng(2024);
    best_scf = inf; x_best = []; y_best = []; scf_iters_best = 0;
    for s = 1:30
        x = randn(m_d,1); x = x/norm(x);
        y = randn(n_d,1); y = y/norm(y);
        for it = 1:2000
            My = zeros(m_d,m_d);
            for i=1:m_d, for kk=1:m_d, for j=1:n_d, for l=1:n_d
                My(i,kk) = My(i,kk) + A_hd(i,j,kk,l)*y(j)*y(l);
            end, end, end, end
            [V, D] = eig(My, 'vector');
            [~, mi2] = min(D);
            x_new = V(:,mi2);
            Nx = zeros(n_d,n_d);
            for j=1:n_d, for l=1:n_d, for i=1:m_d, for kk=1:m_d
                Nx(j,l) = Nx(j,l) + A_hd(i,j,kk,l)*x_new(i)*x_new(kk);
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
            My(i,kk) = My(i,kk) + A_hd(i,j,kk,l)*y(j)*y(l);
        end, end, end, end
        lam_scf = x' * My * x;
        if lam_scf < best_scf
            best_scf = lam_scf; x_best = x; y_best = y; scf_iters_best = it;
        end
    end
    
    % Newton refine from SCF fixed point
    tic;
    [lam_N, ~, ~, iter_N, res_N] = NewtonM(A_hd, c, x_best, y_best, eps_tol, Kmax_newton, 'safeguard');
    t_N = toc;
    
    % SIPM from SVD init (same as NewtonM uses SVD init elsewhere)
    [x0_svd, y0_svd] = svd_init(A_hd, 'smallest');
    tic;
    [lam_S, ~, ~, iter_S] = SIPM_M(A_hd, x0_svd, y0_svd, eps_tol, Kmax_power);
    t_S = toc;
    
    fprintf('  (%2d,%2d): SCF_lam=%8.4f SCF_it=%4d | NewtonM lam=%8.4f iter=%2d res=%.1e t=%.3fs | SIPM lam=%8.4f iter=%4d t=%.3fs\n', ...
        m_d, n_d, best_scf, scf_iters_best, lam_N, iter_N, res_N(end), t_N, lam_S, iter_S, t_S);
end
fprintf('=== Done ===\n');
