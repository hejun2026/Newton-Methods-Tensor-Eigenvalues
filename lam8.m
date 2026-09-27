% lam8.m: enumerate ALL Tetragonal M-eigenpairs by multi-start NewtonM (safeguard)
% Goal: complete the eigenvector table for Example 6.6 (7 eigenvalues).
clear; clc;
c = 1; eps_tol = 1e-8; Kmax = 200; Ns = 400;
A = make_psym_tensor('tetragonal');
m = size(A,1); n = size(A,2);

fprintf('=== Explicit candidate starts ===\n');
% (0,0,1),(0,0,1) should be lambda=3 (a3333=3)
x0 = [0;0;1]; y0 = [0;0;1];
[lam, x, y, itv, rh] = NewtonM(A, c, x0, y0, eps_tol, Kmax, 'safeguard');
fprintf('start e3,e3  : lam=%.6f it=%d res=%.1e x=[%.4f %.4f %.4f] y=[%.4f %.4f %.4f]\n', lam, itv, rh(end), x, y);
% SVD init
[x0s, y0s] = svd_init(A, 'smallest');
[lam, x, y, itv, rh] = NewtonM(A, c, x0s, y0s, eps_tol, Kmax);
fprintf('SVD init     : lam=%.6f it=%d res=%.1e x=[%.4f %.4f %.4f] y=[%.4f %.4f %.4f]\n', lam, itv, rh(end), x, y);

fprintf('\n=== Random multi-start ===\n');
rng(7);
lams = []; Xs = {}; Ys = {}; Itv = []; Res = [];
for s = 1:Ns
    x0 = randn(m,1); x0 = x0/norm(x0);
    y0 = randn(n,1); y0 = y0/norm(y0);
    [lam, x, y, itv, rh] = NewtonM(A, c, x0, y0, eps_tol, Kmax, 'safeguard');
    if rh(end) > eps_tol, continue; end
    is_new = true;
    for e = 1:length(lams)
        if abs(lam - lams(e)) < 1e-4
            is_new = false; break;
        end
    end
    if is_new
        lams = [lams, lam]; Xs{end+1} = x; Ys{end+1} = y; Itv = [Itv, itv]; Res = [Res, rh(end)];
    end
end
[lams, idx] = sort(lams);
fprintf('Found %d distinct M-eigenvalues:\n', length(lams));
for e = 1:length(lams)
    fprintf('  lam=%.6f  it=%d  res=%.1e  x=[%.4f %.4f %.4f]  y=[%.4f %.4f %.4f]\n', ...
        lams(e), Itv(idx(e)), Res(idx(e)), Xs{idx(e)}, Ys{idx(e)});
end