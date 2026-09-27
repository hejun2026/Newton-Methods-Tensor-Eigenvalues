function [lambda_val, x, y, iter, res_hist] = NewtonC(A, c, x0, y0, eps_tol, Kmax)
% NewtonC: Newton method for computing a C-eigenvalue of piezoelectric-type tensor A
%   A: n×n×n tensor
%   c: regularization parameter
%   x0, y0: initial unit vectors
%   eps_tol: tolerance
%   Kmax: max iterations
% Output: lambda, x, y, iteration count, residual history

    n = size(A, 1);
    x = x0(:) / norm(x0);
    y = y0(:) / norm(y0);
    res_hist = [];

    for k = 1:Kmax
        % Compute g = Ayy, h = xAy, mu = x'g
        g = zeros(n, 1);
        for j = 1:n
            for jj = 1:n
                g = g + A(:, j, jj) * y(j) * y(jj);
            end
        end
        h = zeros(n, 1);
        for i = 1:n
            for k = 1:n
                h = h + squeeze(A(i, :, k))' * x(i) * y(k);
            end
        end
        mu = x' * g;

        % Compute P(y), Q(x)
        P = zeros(n, n);
        Q = zeros(n, n);
        for j = 1:n
            for k = 1:n
                P(:, j) = P(:, j) + A(:, j, k) * y(k);
            end
        end
        for i = 1:n
            for k = 1:n
                Q(:, k) = Q(:, k) + A(i, :, k)' * x(i);
            end
        end

        % Assemble Jacobian J (2n × 2n)
        J11 = -mu * eye(n) - x * g' + c * x * x' + (c/2) * (x' * x - 1) * eye(n);
        J12 = 2 * P - 2 * x * h';
        J21 = P' - y * g';
        J22 = Q - 2 * y * h' - mu * eye(n) + c * y * y' + (c/2) * (y' * y - 1) * eye(n);
        J = [J11, J12; J21, J22];

        % Assemble residual r = F(x,y;c)
        r1 = g - mu * x + (c/2) * (x' * x - 1) * x;
        r2 = h - mu * y + (c/2) * (y' * y - 1) * y;
        r = [r1; r2];

        % Solve J * delta = -r
        delta = J \ (-r);
        dx = delta(1:n);
        dy = delta(n+1:2*n);

        % Update with normalization
        x = (x + dx);
        y = (y + dy);
        x = x / norm(x);
        y = y / norm(y);

        % Compute residual
        g_new = zeros(n, 1);
        for j = 1:n
            for jj = 1:n
                g_new = g_new + A(:, j, jj) * y(j) * y(jj);
            end
        end
        h_new = zeros(n, 1);
        for i = 1:n
            for kk = 1:n
                h_new = h_new + squeeze(A(i, :, kk))' * x(i) * y(kk);
            end
        end
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
