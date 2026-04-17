function DtD = build_DtD(T)
%BUILD_DTD  Build the T x T second-difference operator D'D (sparse).
%
% DtD is the matrix form of the second-order finite-difference penalty
% used for temporal smoothness: 0.5 * trace(X * DtD * X').
%
% Input
%   T   : number of time points
%
% Output
%   DtD : T x T sparse matrix

    main = [1; 2*ones(T-2,1); 1];
    off  = -1*ones(T-1,1);
    offL = [off; 0];
    offU = [0; off];
    DtD = spdiags([offL main offU], [-1 0 1], T, T);
end
