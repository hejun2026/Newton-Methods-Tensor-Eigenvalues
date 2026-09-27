function [lambda_val, x, y, iter, res_hist] = NewtonM(A, c, x0, y0, eps_tol, Kmax, mode)
% NewtonM: Newton method for computing an M-eigenvalue of partially symmetric tensor A
%   A: m×n×m×n partially symmetric tensor
%   c: regularization parameter
%   x0 (m×1), y0 (n×1): initial unit vectors
%   eps_tol: tolerance
%   Kmax: max iterations
%   mode (optional): 'plain' (default, damped steps as in paper) or
%                    'safeguard' (fallback: if plain fails, retry with
%                    Jacobian eigen-regularization + Armijo line search)
% Output: lambda, x, y, iteration count, residual history

    if nargin < 7
        mode = 'plain';
    end
    m = size(A, 1);
    n = size(A, 2);

    [lambda_val, x, y, iter, res_hist] = run_plain(A, c, x0, y0, eps_tol, Kmax);

    if strcmpi(mode, 'safeguard') && res_hist(end) > eps_tol
        % Retry with eigenvalue-regularized Jacobian and Armijo line search
        [lambda_val, x, y, iter, res_hist] = run_safeguard(A, c, x0, y0, eps_tol, Kmax);
    end
end

% ----------------------------------------------------------------------
function [lambda_val, x, y, iter, res_hist] = run_plain(A, c, x0, y0, eps_tol, Kmax)
    m = size(A, 1);
    n = size(A, 2);
    x = x0(:) / norm(x0);
    y = y0(:) / norm(y0);
    res_hist = [];
    mu_new = 0;

    for k = 1:Kmax
        % Compute M(y), N(x), R(x,y)
        M_y = zeros(m, m);
        for i = 1:m
            for kk = 1:m
                for j = 1:n
                    for l = 1:n
                        M_y(i, kk) = M_y(i, kk) + A(i, j, kk, l) * y(j) * y(l);
                    end
                end
            end
        end

        N_x = zeros(n, n);
        for j = 1:n
            for l = 1:n
                for i = 1:m
                    for kk = 1:m
                        N_x(j, l) = N_x(j, l) + A(i, j, kk, l) * x(i) * x(kk);
                    end
                end
            end
        end

        R_xy = zeros(m, n);
        for i = 1:m
            for s = 1:n
                for kk = 1:m
                    for l = 1:n
                        R_xy(i, s) = R_xy(i, s) + A(i, s, kk, l) * x(kk) * y(l);
                    end
                end
            end
        end

        g = M_y * x;
        h = N_x * y;
        mu = x' * g;

        % Assemble Jacobian J ((m+n) × (m+n))
        J11 = M_y - 2 * x * g' - mu * eye(m) + c * x * x' + (c/2) * (x' * x - 1) * eye(m);
        J12 = 2 * R_xy - 2 * x * h';
        J21 = 2 * R_xy' - 2 * y * g';
        J22 = N_x - 2 * y * h' - mu * eye(n) + c * y * y' + (c/2) * (y' * y - 1) * eye(n);
        J = [J11, J12; J21, J22];

        % Assemble residual
        r1 = g - mu * x + (c/2) * (x' * x - 1) * x;
        r2 = h - mu * y + (c/2) * (y' * y - 1) * y;
        r = [r1; r2];

        % Solve
        delta = J \ (-r);
        dx = delta(1:m);
        dy = delta(m+1:m+n);

        % Update with normalization
        x = x + dx;
        y = y + dy;
        x = x / norm(x);
        y = y / norm(y);

        % Compute residual
        M_new = zeros(m, m);
        for i = 1:m
            for kk = 1:m
                for j = 1:n
                    for l = 1:n
                        M_new(i, kk) = M_new(i, kk) + A(i, j, kk, l) * y(j) * y(l);
                    end
                end
            end
        end
        N_new = zeros(n, n);
        for j = 1:n
            for l = 1:n
                for i = 1:m
                    for kk = 1:m
                        N_new(j, l) = N_new(j, l) + A(i, j, kk, l) * x(i) * x(kk);
                    end
                end
            end
        end
        g_new = M_new * x;
        h_new = N_new * y;
        mu_new = x' * g_new;
        res = norm(g_new - mu_new * x) + norm(h_new - mu_new * y);
        res_hist = [res_hist; res];

        if res <= eps_tol
            lambda_val = mu_new;
            iter = k;
            return;
        end
    end

    lambda_val = mu_new;
    iter = Kmax;
end

% ----------------------------------------------------------------------
function [lambda_val, x, y, iter, res_hist] = run_safeguard(A, c, x0, y0, eps_tol, Kmax)
% Safeguarded Newton: eigen-regularized Jacobian + Armijo backtracking on
% the merit function phi = ||r||^2/2.  Used only as a fallback when the
% plain iteration stagnates.
    m = size(A, 1);
    n = size(A, 2);
    x = x0(:) / norm(x0);
    y = y0(:) / norm(y0);
    res_hist = [];
    mu = 0;

    for k = 1:Kmax
        [M_y, N_x, R_xy] = matrices(A, x, y, m, n);
        g = M_y * x; h = N_x * y; mu = x' * g;

        J11 = M_y - 2 * x * g' - mu * eye(m) + c * x * x' + (c/2) * (x' * x - 1) * eye(m);
        J12 = 2 * R_xy - 2 * x * h';
        J21 = 2 * R_xy' - 2 * y * g';
        J22 = N_x - 2 * y * h' - mu * eye(n) + c * y * y' + (c/2) * (y' * y - 1) * eye(n);
        J = [J11, J12; J21, J22];

        r = [g - mu * x + (c/2) * (x' * x - 1) * x; h - mu * y + (c/2) * (y' * y - 1) * y];
        if norm(r) <= eps_tol
            res_hist(end+1, 1) = norm(r);
            lambda_val = mu; iter = k - 1; return;
        end

        % Regularize J by its smallest eigenvalue magnitude
        [JD, JV] = eig((J + J') / 2);
        dmin = min(abs(diag(JD)));
        if dmin < 1e-8
            J = J + (1e-8 - dmin) * eye(m + n);
        end
        delta = J \ (-r);

        % Armijo backtracking line search
        phi0 = 0.5 * norm(r)^2;
        gtd = -(r' * (J * (J \ r)));   % -||J^{-1}r||^2 upper model proxy
        alpha = 1;
        accepted = false;
        for ls = 1:40
            xt = x + alpha * delta(1:m);
            yt = y + alpha * delta(m+1:m+n);
            xt = xt / norm(xt); yt = yt / norm(yt);
            rt = residual_at(A, xt, yt, c, m, n);
            if 0.5 * norm(rt)^2 <= phi0 + 1e-4 * alpha * gtd || norm(rt) < norm(r)
                x = xt; y = yt; accepted = true; res_hist(end+1, 1) = norm(rt);
                break;
            end
            alpha = alpha / 2;
        end
        if ~accepted
            % keep the full step anyway (best effort)
            x = x + delta(1:m); y = y + delta(m+1:m+n);
            x = x / norm(x); y = y / norm(y);
            res_hist(end+1, 1) = norm(residual_at(A, x, y, c, m, n));
        end
        if res_hist(end) <= eps_tol
            [M_y, ~, ~] = matrices(A, x, y, m, n);
            lambda_val = x' * (M_y * x); iter = k; return;
        end
    end
    [M_y, ~, ~] = matrices(A, x, y, m, n);
    lambda_val = x' * (M_y * x);
    iter = Kmax;
end

% ----------------------------------------------------------------------
function [M_y, N_x, R_xy] = matrices(A, x, y, m, n)
    M_y = zeros(m, m);
    for i = 1:m
        for kk = 1:m
            for j = 1:n
                for l = 1:n
                    M_y(i, kk) = M_y(i, kk) + A(i, j, kk, l) * y(j) * y(l);
                end
            end
        end
    end
    N_x = zeros(n, n);
    for j = 1:n
        for l = 1:n
            for i = 1:m
                for kk = 1:m
                    N_x(j, l) = N_x(j, l) + A(i, j, kk, l) * x(i) * x(kk);
                end
            end
        end
    end
    R_xy = zeros(m, n);
    for i = 1:m
        for s = 1:n
            for kk = 1:m
                for l = 1:n
                    R_xy(i, s) = R_xy(i, s) + A(i, s, kk, l) * x(kk) * y(l);
                end
            end
        end
    end
end

% ----------------------------------------------------------------------
function r = residual_at(A, x, y, c, m, n)
    [M_y, N_x, ~] = matrices(A, x, y, m, n);
    g = M_y * x; h = N_x * y; mu = x' * g;
    r = [g - mu * x + (c/2) * (x' * x - 1) * x; h - mu * y + (c/2) * (y' * y - 1) * y];
end