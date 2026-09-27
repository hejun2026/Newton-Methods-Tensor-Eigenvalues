function [lambda_val, x, y, iter] = SPM_C(A, x0, y0, eps_tol, Kmax)
% SPM_C: Shifted Power Method for computing the largest C-eigenvalue
%   of a piezoelectric-type tensor A (n×n×n)
% Following Zhao 2023: shifted power method
%   x0, y0: initial unit vectors
%   eps_tol: tolerance
%   Kmax: max iterations

    n = size(A, 1);
    x = x0(:) / norm(x0);
    y = y0(:) / norm(y0);
    
    % Initial Rayleigh quotient
    g = zeros(n,1);
    for j = 1:n
        for k = 1:n
            g = g + A(:,j,k) * y(j) * y(k);
        end
    end
    lambda_val = x' * g;

    for iter = 1:Kmax
        % Compute Ayy
        g = zeros(n,1);
        for j = 1:n
            for k = 1:n
                g = g + A(:,j,k) * y(j) * y(k);
            end
        end
        
        % Compute xAy
        h = zeros(n,1);
        for i = 1:n
            for k = 1:n
                h = h + squeeze(A(i,:,k))' * x(i) * y(k);
            end
        end
        
        % Shift: use alpha = ||A|| estimate + small positive
        % Following Zhao 2023: shift alpha to make shifted operator positive
        alpha = max(abs(eig(squeeze(A(:,:,1)))))*3 + 1;  % crude shift estimate
        for idx = 1:n
            g2 = zeros(n,1);
            for j = 1:n
                for k = 1:n
                    g2 = g2 + A(:,j,k) * y(j) * y(k);
                end
            end
            alpha = max(alpha, norm(g2) + 1);
        end
        
        % Power iteration with shift: x = (alpha*I + P(y))^{-1} * ... 
        % Actually SPM for C-eigenvalue: 
        % Update y: y = (alpha*I + Q(x))^{-1} (alpha*y + Q(x)*y)  normalized
        % Update x: x = (alpha*I + P(y))^{-1} (alpha*x + P(y)*x)  normalized
        
        % Compute P(y), Q(x)
        P = zeros(n,n);
        Q = zeros(n,n);
        for j = 1:n
            for k = 1:n
                P(:,j) = P(:,j) + A(:,j,k) * y(k);
            end
        end
        for i = 1:n
            for k = 1:n
                Q(:,k) = Q(:,k) + A(i,:,k)' * x(i);
            end
        end
        
        % Shifted power iteration
        x_new = (alpha * eye(n) + P) \ ((alpha * x) + P * x);
        x_new = x_new / norm(x_new);
        
        y_new = (alpha * eye(n) + Q) \ ((alpha * y) + Q * y);
        y_new = y_new / norm(y_new);
        
        x = x_new;
        y = y_new;
        
        % Compute new eigenvalue and eigenpair residual
        g = zeros(n,1);
        for j = 1:n
            for k = 1:n
                g = g + A(:,j,k) * y(j) * y(k);
            end
        end
        
        h = zeros(n,1);
        for i = 1:n
            for k = 1:n
                h = h + squeeze(A(i,:,k))' * x(i) * y(k);
            end
        end
        
        lambda_new = x' * g;
        
        % Convergence check: eigenpair residual (same criterion as NewtonC)
        res = norm(g - lambda_new * x) + norm(h - lambda_new * y);
        lambda_val = lambda_new;
        
        if res <= eps_tol
            return;
        end
    end
end
