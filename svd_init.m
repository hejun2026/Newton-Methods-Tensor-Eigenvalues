function [x0, y0, B] = svd_init(A, which)
% svd_init: Initialization strategy for NewtonM (Remark rem:initM in the paper,
%           following Zhao et al. 2024 SIPM).
%   Unfold A + tau*E to the symmetric mn x mn matrix C with
%   C(i + (j-1)*m, k + (l-1)*n) = A(i,j,k,l), tau = ||C||_F + eps,
%   compute the smallest ('smallest', default) or largest ('largest')
%   eigenvector w of C, fold w into the m x n matrix W, and take the leading
%   singular pair (u1, v1) of W as (x0, y0).
%
%   A: m x n x m x n partially symmetric tensor
%   which: 'smallest' or 'largest'
%
%   Output: x0 (m x 1), y0 (n x 1) unit vectors; B = C + tau*I.

    if nargin < 2
        which = 'smallest';
    end
    m = size(A, 1);
    n = size(A, 2);

    % (i,j) x (k,l) unfolding: acts on vec(X) with X = x y^T
    % row index r = i + (j-1)*m (i fast, j slow); col index c = k + (l-1)*m
    C = zeros(m*n, m*n);
    for i = 1:m
        for j = 1:n
            r = i + (j-1)*m;
            for k = 1:m
                for l = 1:n
                    C(r, k + (l-1)*m) = A(i,j,k,l);
                end
            end
        end
    end
    C = (C + C') / 2;

    tau = norm(C, 'fro');
    B = C + tau * eye(m*n);

    [V, D] = eig(B);   % ascending eigenvalues of the shifted matrix
    if strcmpi(which, 'smallest')
        w = V(:, 1);       % eigenvector of C closest to -tau (most negative / smallest)
    else
        w = V(:, end);     % eigenvector of C with largest eigenvalue
    end

    % Fold w (mn x 1) into m x n matrix W
    W = reshape(w, m, n);

    % Leading singular pair of W
    [U, ~, Vt] = svd(W);
    x0 = U(:, 1);
    y0 = Vt(1, :)';

    % Sign convention: make the largest-|entry| component of each vector positive
    [~, im] = max(abs(x0)); if x0(im) < 0, x0 = -x0; end
    [~, im] = max(abs(y0)); if y0(im) < 0, y0 = -y0; end
end
