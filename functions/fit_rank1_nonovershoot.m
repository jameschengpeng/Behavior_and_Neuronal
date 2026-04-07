function F = fit_rank1_nonovershoot(B, X, opts_or_eps0)
% FIT_RANK1_NONOVERSHOOT Conservative rank-1 temporal floor estimate.
% Uses a low quantile of X./B and corrects it upward for Gaussian noise.
% If per-pixel noise variance is provided, lower-noise pixels are weighted
% more strongly when estimating the temporal floor.
% B: P x 1 vector (e.g., background spatial component)
% X: P x T matrix (e.g., data or residual), each row is one pixel trace.
% opts_or_eps0: either scalar eps0 or struct with fields:
%   - bg_eps or eps0 (default 1e-12): small constant to avoid division by zero
%   - bg_floor_quantile (default 0.05): quantile for initial floor estimate
%   - bg_floor_noise_sigma (default []): if provided, use this as noise sigma; otherwise estimate from data
%   - bg_noise_var (default []): per-pixel noise variance for weighting

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

    if isfield(opts_or_eps0, 'bg_noise_var')
        noise_var = opts_or_eps0.bg_noise_var;
    else
        noise_var = [];
    end
else
    eps0 = opts_or_eps0;
    q = 0.05;
    sigma_ratio = [];
    noise_var = [];
end

support = (B > eps0);
if ~any(support)
    F = zeros(1, size(X, 2), 'like', X);
    return;
end

ratio = X(support, :) ./ max(B(support), eps0);
weights = [];
tau2 = [];
if ~isempty(noise_var)
    noise_var = reshape(noise_var, [], 1);
    assert(numel(noise_var) == size(X, 1), ...
        'opts.bg_noise_var must have one entry per row of X.');
    tau2 = noise_var(support) ./ max(B(support).^2, eps0);
    tau2 = max(tau2, eps0);
    weights = 1 ./ tau2;
    weights = cast(weights, 'like', ratio);
end

if isempty(weights)
    qval = prctile(ratio, 100 * q, 1);
else
    qval = weighted_quantile_columns(ratio, weights, q);
end

if isempty(sigma_ratio)
    if ~isempty(weights)
        % Estimate a typical per-pixel noise scale in ratio-space.
        % Using sqrt(1/sum(weights)) treats the lower-tail quantile like a
        % weighted mean standard error and collapses toward zero when many
        % pixels contribute, which can under-correct the quantile bias.
        wnorm = double(weights) / max(sum(double(weights)), eps0);
        sigma_ratio = sqrt(sum(wnorm .* double(tau2)));
    elseif size(ratio, 2) >= 2
        dratio = diff(ratio, 1, 2);
        % approximate the noise sigma from the median absolute deviation (MAD) of the differences
        % MAD is more resilient to outliers than standard deviation, and the factor 1.4826 makes it consistent with std for Gaussian noise
        med_dratio = median(dratio(:));
        sigma_diff = 1.4826 * median(abs(dratio(:) - med_dratio));
        sigma_ratio = sigma_diff / sqrt(2);
    else
        sigma_ratio = 0;
    end
end

zq = -sqrt(2) * erfcinv(2 * q);
noise_bias = -sigma_ratio * zq; % sigma_ratio depends on weights
% F is lower-bounded by the quantile estimate plus the noise bias correction, ensuring a conservative floor that accounts for noise. The max with 0 ensures nonnegativity.
F = max(cast(0, 'like', X), cast(qval + noise_bias, 'like', X));
end

function qval = weighted_quantile_columns(X, weights, q)
[nRows, nCols] = size(X);
weights = max(double(weights(:)), 0);
total_w = sum(weights);

if total_w <= 0
    qval = prctile(X, 100 * q, 1);
    return;
end

% Sort all columns at once (vectorised)
[Xs, order] = sort(X, 1, 'ascend');

% Reorder weights per column and build CDF
ws  = weights(order);                    % nRows x nCols
cdf = cumsum(ws, 1) ./ total_w;          % nRows x nCols

% For each column, find the first row where cdf >= q
[~, k] = max(cdf >= q, [], 1);           % 1 x nCols

lin_idx = sub2ind([nRows, nCols], k, 1:nCols);
qval = cast(Xs(lin_idx), 'like', X);
end
