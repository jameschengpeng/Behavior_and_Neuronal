function X = soft_thresh_nonneg(X, tau)
%SOFT_THRESH_NONNEG  Combined soft-thresholding and nonnegative projection.
%
% Equivalent to the proximal operator of lambda*||x||_1 with x >= 0
% constraint:  prox(x) = max(0, x - tau).
%
% Inputs
%   X   : array of any size
%   tau : nonnegative threshold scalar (etaA * lambdaA_L1)
%
% Output
%   X   : thresholded, nonnegative array (same size as input)

    X = max(0, X - tau);
end
