function A = make_psym_tensor(name)
% make_psym_tensor: Create partially symmetric tensor A (m×n×m×n)
% Partially symmetric: a_ijkl = a_kjil = a_ilkj = a_klij
%
% name = 'tetragonal'  : Example 6.6 (paper Eq. data, NOT Voigt conversion)
% name = 'random_wang' : Example 6.7 random tensor (Wang 2009)
% name = 'dolomite'    : Example 6.8 dolomite (paper a-values)
% name = 'highdim'     : placeholder (no tensor returned)

    switch lower(name)
        case 'tetragonal'
            A = build_tetragonal_paper();
        case 'random_wang'
            % Wang-Qi-Zhang 2009 Example 4.1 tensor (full 9 slices),
            % as reproduced in He-Liu-Xu 2021 (AMC 398:125992) Example 2.
            % A(:,:,k,l) is the slice with 3rd index k, 4th index l.
            A = zeros(3, 3, 3, 3);
            A(:,:,1,1) = [ -0.9727,  0.3169, -0.3437;
                           -0.6332, -0.7866,  0.4257;
                           -0.3350, -0.9896, -0.4323];
            A(:,:,2,1) = [ -0.6332, -0.7866,  0.4257;
                            0.7387,  0.6873, -0.3248;
                           -0.7986, -0.5988, -0.9485];
            A(:,:,3,1) = [ -0.3350, -0.9896, -0.4323;
                           -0.7986, -0.5988, -0.9485;
                            0.5853,  0.5921,  0.6301];
            A(:,:,1,2) = [  0.3169,  0.6158, -0.0184;
                           -0.7866,  0.0160,  0.0085;
                           -0.9896, -0.6663,  0.2559];
            A(:,:,2,2) = [ -0.7866,  0.0160,  0.0085;
                            0.6873,  0.5160, -0.0216;
                           -0.5988,  0.0411,  0.9857];
            A(:,:,3,2) = [ -0.9896, -0.6663,  0.2559;
                           -0.5988,  0.0411,  0.9857;
                            0.5921, -0.2907, -0.3881];
            A(:,:,1,3) = [ -0.3437, -0.0184,  0.5649;
                            0.4257,  0.0085, -0.1439;
                           -0.4323,  0.2559,  0.6162];
            A(:,:,2,3) = [  0.4257,  0.0085, -0.1439;
                           -0.3248, -0.0216, -0.0037;
                           -0.9485,  0.9857, -0.7734];
            A(:,:,3,3) = [ -0.4323,  0.2559,  0.6162;
                           -0.9485,  0.9857, -0.7734;
                            0.6301, -0.3881, -0.8526];
        case 'dolomite'
            A = build_dolomite_paper();
        case 'highdim'
            A = [];
        otherwise
            error('Unknown tensor name: %s', name);
    end
end

% ======================================================================
%  Tetragonal system (paper Example 6.6)
%  Paper lists the partially symmetric tensor entries directly:
%    a1111=a1313=a2222=a2323=a3131=a3232=4,  a1212=a2121=2,
%    a1112=a1211=a1121=a2111=1,
%    a1133=a1331=a3113=a3311=1,
%    a2233=a2332=a3223=a3322=1,
%    a1122=a1221=a2112=a2211=-1,
%    a1222=a2212=a2122=a2221=-1,
%    a3333=3
%  Each group is the full orbit under the partial-symmetry map
%  (i,j,k,l) -> {(i,j,k,l),(k,j,i,l),(i,l,k,j),(k,l,i,j)}.
% ======================================================================
function A = build_tetragonal_paper()
    A = zeros(3,3,3,3);
    seeds = {
        1,1,1,1,  4;    % orbit {1111}
        1,3,1,3,  4;     % {1313}
        2,2,2,2,  4;     % {2222}
        2,3,2,3,  4;     % {2323}
        3,1,3,1,  4;     % {3131}
        3,2,3,2,  4;     % {3232}
        1,2,1,2,  2;     % {1212}
        2,1,2,1,  2;     % {2121}
        1,1,1,2,  1;     % orbit {1112,1211}
        1,1,2,1,  1;     % orbit {1121,2111}
        1,1,3,3,  1;     % orbit {1133,3113,1331,3311}
        2,2,3,3,  1;     % orbit {2233,3223,2332,3322}
        1,1,2,2, -1;     % orbit {1122,2112,1221,2211}
        1,2,2,2, -1;     % orbit {1222,2212}
        2,1,2,2, -1;     % orbit {2122,2221}
        3,3,3,3,  3;     % {3333}
    };
    A = fill_orbits(A, seeds);
end

% ----------------------------------------------------------------------
%  Dolomite (paper Example 6.8)
%  Source: He 2021 "New S-Type Bounds of M-Eigenvalues for Elasticity
%  Tensors" Example 2 (CaMg(CO3)2-dolomite). Nonzero entries:
%    a1111=a2222=196.6, a3311=a2233=83.2, a3333=110,
%    a2323=a3232=a1313=a3131=54.7, a1212=a2121=64.4,
%    a2223=a2232=-31.7 and a1213=a2131=+31.7,
%    a1122=132.2, a2132=a1223=-35.84, a3112=a1321=44.8,
%    a2321=a1232=-25.3 and a1311=a1131=+25.3.
%  Expand each listed entry to its full orbit under the map
%  (i,j,k,l)->{(i,j,k,l),(k,j,i,l),(i,l,k,j),(k,l,i,j)} with the same
%  value.  Overlaps are harmless (values coincide by construction).
% ----------------------------------------------------------------------
function A = build_dolomite_paper()
    A = zeros(3,3,3,3);
    seeds = {
        1,1,1,1, 196.6;
        2,2,2,2, 196.6;
        3,3,3,3, 110.0;
        2,2,3,3, 83.2;     % orbit {2233,3223,2332,3322}
        3,3,1,1, 83.2;     % orbit {3311,1331,3113,1133}
        1,3,1,3, 54.7;     % {1313}
        2,3,2,3, 54.7;     % {2323}
        3,1,3,1, 54.7;     % {3131}
        3,2,3,2, 54.7;     % {3232}
        1,2,1,2, 64.4;     % {1212}
        2,1,2,1, 64.4;     % {2121}
        2,2,2,3, -31.7;    % orbit {2223,2322}
        2,2,3,2, -31.7;    % orbit {2232,3222}
        1,2,1,3,  31.7;    % orbit {1213,1312}
        2,1,3,1,  31.7;    % orbit {2131,3121}
        1,2,3,2, -25.3;    % orbit {1232,3212}
        2,3,2,1, -25.3;    % orbit {2321,2123}
        1,3,1,1,  25.3;    % orbit {1311,1113}
        1,1,3,1,  25.3;    % orbit {1131,3111}
        1,3,2,1, 44.8;     % orbit {1321,2311,1123,2113}
        3,1,1,2, 44.8;     % orbit {3112,1132,3211,1231}
        1,2,2,3, -35.84;   % orbit {1223,2213,1322,2312}
        2,1,3,2, -35.84;   % orbit {2132,3122,2231,3221}
        1,1,2,2, 132.2;    % orbit {1122,2112,1221,2211}
    };
    A = fill_orbits(A, seeds);
end

% ----------------------------------------------------------------------
function A = fill_orbits(A, seeds)
% Expand each seed to its partial-symmetry orbit and return the tensor.
    for s = 1:size(seeds,1)
        i = seeds{s,1}; j = seeds{s,2}; k = seeds{s,3}; l = seeds{s,4}; v = seeds{s,5};
        A(i,j,k,l) = v;
        A(k,j,i,l) = v;
        A(i,l,k,j) = v;
        A(k,l,i,j) = v;
    end
end