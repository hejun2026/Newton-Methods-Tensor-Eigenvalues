function [lambda_val, x, y, iter] = SIPM_M(A, x0, y0, eps_tol, Kmax)
% SIPM_M: Shifted Inverse Power Method for computing the smallest M-eigenvalue
%   of a partially symmetric tensor A (m×n×m×n)
% Following Zhao et al. 2024
%   x0 (m×1), y0 (n×1): initial unit vectors
%   eps_tol: tolerance
%   Kmax: max iterations

    m = size(A, 1);
    n = size(A, 2);
    x = x0(:) / norm(x0);
    y = y0(:) / norm(y0);
    
    % Compute initial M(y), N(x)
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
    g = M_y * x;
    lambda_val = x' * g;
    
    % Shift parameter: tau = norm estimate
    tau = 0;
    for i = 1:m
        for j = 1:n
            for kk = 1:m
                for l = 1:n
                    tau = tau + A(i,j,kk,l)^2;
                end
            end
        end
    end
    tau = sqrt(tau) + 1;
    
    for iter = 1:Kmax
        % Compute M(y), N(x)
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
        
        % Shifted inverse power iteration
        % (M(y) + tau*I) x_new = x,  (N(x) + tau*I) y_new = y
        x_new = (M_y + tau * eye(m)) \ x;
        x_new = x_new / norm(x_new);
        
        y_new = (N_x + tau * eye(n)) \ y;
        y_new = y_new / norm(y_new);
        
        x = x_new;
        y = y_new;
        
        % Compute new eigenvalue and eigenpair residual
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
        g = M_y * x;
        
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
        h = N_x * y;
        lambda_new = x' * g;
        
        % Convergence check: eigenpair residual (same criterion as NewtonM,
        % as stated in the paper: "run until the eigenpair residual drops below eps")
        res = norm(g - lambda_new * x) + norm(h - lambda_new * y);
        lambda_val = lambda_new;
        
        if res <= eps_tol
            return;
        end
    end
end
