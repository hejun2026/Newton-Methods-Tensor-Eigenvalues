% enum_bani100.m — Enumerate all C-eigenvalues of BaNiO3 with 100 starts
clear; clc;
eps_tol = 1e-8;
Kmax = 200;
c = 1;
Ns = 100;

A = make_piezo_tensor('BaNiO3');
n = size(A,1);

all_eig = []; all_it = []; all_res = []; all_x = {}; all_y = {};
rng(42);
for s = 1:Ns
    xs = randn(n,1); xs = xs/norm(xs);
    ys = randn(n,1); ys = ys/norm(ys);
    [lam_s, x_s, y_s, iter_s, res_hist_s] = NewtonC(A, c, xs, ys, eps_tol, Kmax);
    res_s = res_hist_s(end);
    
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
        all_x{end+1} = x_s;
        all_y{end+1} = y_s;
    end
end

[~, sort_idx] = sort(abs(all_eig), 'descend');
all_eig = all_eig(sort_idx);
all_it = all_it(sort_idx);
all_res = all_res(sort_idx);

fprintf('=== BaNiO3: %d C-eigenvalues found (Ns=%d) ===\n', length(all_eig), Ns);
for e = 1:length(all_eig)
    fprintf('  lam=%.4f  iter=%d  res=%.2e  x=[%.4f %.4f %.4f]  y=[%.4f %.4f %.4f]\n', ...
        all_eig(e), all_it(e), all_res(e), all_x{sort_idx(e)}, all_y{sort_idx(e)});
end
