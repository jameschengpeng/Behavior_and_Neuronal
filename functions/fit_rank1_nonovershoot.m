function F = fit_rank1_nonovershoot(B, X, opts_or_eps0)
% FIT_RANK1_NONOVERSHOOT Conservative rank-1 temporal floor estimate.
% Uses a low quantile of X./B and corrects it upward for Gaussian noise.
% B: P x 1 vector (e.g., background spatial component)
% X: P x T matrix (e.g., data or residual), each row is one pixel trace.
% opts_or_eps0: either scalar eps0 or struct with fields:
%   - bg_eps or eps0 (default 1e-12): small constant to avoid division by zero
%   - bg_floor_quantile (default 0.05): quantile for initial floor estimate
%   - bg_floor_noise_sigma (default []): if provided, use this as noise sigma; otherwise estimate from data 

if isstruct(opts_or_eps0)
    if isfield(opts_or_eps0, 'bg_eps')
        eps0 = opts_or_eps0.bg_eps;
    elseif isfield(opts_or_eps0, 'eps0')
        eps0 = opts_or_eps0.eps0;
    else
        eps0 = 1e-12;
    end

    if isfield(opts_or_eps0, 'bg_floor_quantile')
        q = opts_or_eps0.bg_floor_quantile;
    else
        q = 0.05;
    end

    if isfield(opts_or_eps0, 'bg_floor_noise_sigma')
        sigma_ratio = opts_or_eps0.bg_floor_noise_sigma;
    else
        sigma_ratio = [];
    end
else
    eps0 = opts_or_eps0;
    q = 0.05;
    sigma_ratio = [];
end

support = (B > eps0);
if ~any(support)
    F = zeros(1, size(X, 2), 'like', X);
    return;
end

ratio = X(support, :) ./ max(B(support), eps0);
qval = prctile(ratio, 100 * q, 1);

if isempty(sigma_ratio)
    if size(ratio, 2) >= 2
        dratio = diff(ratio, 1, 2);
        med_dratio = median(dratio(:));
        sigma_diff = 1.4826 * median(abs(dratio(:) - med_dratio));
        sigma_ratio = sigma_diff / sqrt(2);
    else
        sigma_ratio = 0;
    end
end

zq = -sqrt(2) * erfcinv(2 * q);
noise_bias = -sigma_ratio * zq;
F = max(0, qval + noise_bias);
end