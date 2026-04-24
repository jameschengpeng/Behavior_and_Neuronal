function autocorr_map = residual_autocorr_map(X, A, C, B, F, tau, mask)
%RESIDUAL_AUTOCORR_MAP  Per-pixel lag-tau autocorrelation of model residuals.
%
%   autocorr_map = residual_autocorr_map(X, A, C, B, F, tau)
%   autocorr_map = residual_autocorr_map(X, A, C, B, F, tau, mask)
%
% Inputs:
%   X   : P x T data matrix
%   A   : P x K spatial footprints, or Pm x K active-pixel footprints
%   C   : K x T temporal components
%   B   : P x r background spatial components, or Pm x r active-pixel
%         background components (can be empty)
%   F   : r x T background temporal components (can be empty)
%   tau : positive integer lag in frames
%   mask: optional P-vector / HxW logical mask. Pixels outside the mask are
%         assigned NaN in the output. Required if A or B are provided on the
%         active pixels only.
%
% Output:
%   autocorr_map : P x 1 vector, where entry p is the lag-tau Pearson
%                  autocorrelation of the residual trace at pixel p.

    narginchk(6, 7);

    [P, T] = size(X);
    assert(size(C,2) == T, 'C and X must have the same number of columns.');
    assert(isscalar(tau) && tau == floor(tau) && tau >= 1, ...
        'tau must be a positive integer.');
    assert(tau < T, 'tau must be smaller than the number of time points.');

    if nargin < 7 || isempty(mask)
        mask_vec = true(P, 1);
    else
        mask_vec = reshape(mask, [], 1) ~= 0;
        assert(numel(mask_vec) == P, 'mask must have length P or size HxW with H*W = P.');
    end

    if size(A,1) == P
        A_full = A;
    else
        assert(size(A,1) == nnz(mask_vec), ...
            'A must have P rows or number of active-pixel rows nnz(mask).');
        A_full = zeros(P, size(A, 2), 'like', A);
        A_full(mask_vec, :) = A;
    end

    if isempty(B)
        BF = zeros(P, T, 'like', X);
    else
        assert(size(F,2) == T, 'F and X must have the same number of columns.');
        assert(size(B,2) == size(F,1), 'B and F inner dimensions must agree.');
        if size(B,1) == P
            B_full = B;
        else
            assert(size(B,1) == nnz(mask_vec), ...
                'B must have P rows or number of active-pixel rows nnz(mask).');
            B_full = zeros(P, size(B, 2), 'like', B);
            B_full(mask_vec, :) = B;
        end
        BF = B_full * F;
    end

    R = X - A_full * C - BF;
    R1 = double(R(:, 1:(T - tau)));
    R2 = double(R(:, (1 + tau):T));

    R1 = R1 - mean(R1, 2);
    R2 = R2 - mean(R2, 2);

    denom = sqrt(sum(R1.^2, 2) .* sum(R2.^2, 2));
    autocorr_map = sum(R1 .* R2, 2) ./ max(denom, eps);
    autocorr_map(denom <= eps) = NaN;
    autocorr_map(~mask_vec) = NaN;
end