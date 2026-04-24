function results = test_temporal_heteroscedasticity_fast(X, mask, alpha, sigma_frames, video_lengths)
% Test temporal heteroscedasticity using Gaussian-smoothed level proxy.
%
% For each unmasked pixel p, fits the linear working model
%   Y_p(t) = beta0_p + beta1_p * M_p(t) + eta_p(t),
% where Y_p(t) = (X_p(t+1)-X_p(t))^2 and M_p(t) is a Gaussian-smoothed
% midpoint level proxy. Returns a lean result struct with the fields used
% by current downstream analysis.
%
% X            : [n_pixels x T]
% mask         : [H x W], 1 = unmasked
% alpha        : significance level, default 0.05
% sigma_frames : Gaussian sigma in frames, default 1
% video_lengths: optional lengths of concatenated videos. If omitted, the
%                full trace is treated as one video.

if nargin < 3 || isempty(alpha)
    alpha = 0.05;
end
if nargin < 4 || isempty(sigma_frames)
    sigma_frames = 1;
end

[n_pixels, T] = size(X);
[H, W] = size(mask);

if H * W ~= n_pixels
    error('mask size mismatch: H*W must equal size(X,1).');
end
if T < 5
    error('Need at least 5 time points.');
end

mask = logical(mask(:));
idx = find(mask);
n_use = numel(idx);

X = X(idx, :);

[~, M, Y, valid, video_lengths] = build_segmentwise_pair_features(X, sigma_frames, video_lengths);

n = sum(valid, 2);

M(~valid) = 0;
Y(~valid) = 0;

% Row-wise summary statistics. This is faster than the previous quadratic
% fit because it avoids M^2/M^3/M^4 terms and the 3x3 normal-equation solve.
n_safe = max(double(n), 1);
sumM = sum(double(M), 2);
sumY = sum(double(Y), 2);
sumM2 = sum(double(M .* M), 2);
sumY2 = sum(double(Y .* Y), 2);
sumMY = sum(double(M .* Y), 2);

meanM = sumM ./ n_safe;
meanY = sumY ./ n_safe;

Sxx = sumM2 - (sumM .^ 2) ./ n_safe;
Syy = sumY2 - (sumY .^ 2) ./ n_safe;
Sxy = sumMY - (sumM .* sumY) ./ n_safe;

Sxx = max(Sxx, 0);
Syy = max(Syy, 0);

rho = nan(n_use, 1);
beta0 = nan(n_use, 1);
beta1 = nan(n_use, 1);
se_beta1 = nan(n_use, 1);
sse = nan(n_use, 1);
pval_beta1 = nan(n_use, 1);

ok_corr = (n >= 3) & (Sxx > 0) & (Syy > 0);
ok_reg = (n >= 3) & (Sxx > 0);

rho(ok_corr) = Sxy(ok_corr) ./ sqrt(Sxx(ok_corr) .* Syy(ok_corr));
rho(ok_corr) = max(min(rho(ok_corr), 1), -1);

beta1(ok_reg) = Sxy(ok_reg) ./ Sxx(ok_reg);
beta0(ok_reg) = meanY(ok_reg) - beta1(ok_reg) .* meanM(ok_reg);

% Regression SSE from centered sums
sse(ok_reg) = Syy(ok_reg) - beta1(ok_reg) .* Sxy(ok_reg);
sse(ok_reg) = max(sse(ok_reg), 0);

df = double(n) - 2;
sigma2_hat = nan(n_use, 1);
sigma2_hat(ok_reg & df > 0) = sse(ok_reg & df > 0) ./ df(ok_reg & df > 0);

se_beta1(ok_reg & df > 0) = sqrt(max(sigma2_hat(ok_reg & df > 0) ./ Sxx(ok_reg & df > 0), 0));
t_beta1 = nan(n_use, 1);

ok_t1 = ok_reg & df > 0 & isfinite(se_beta1) & (se_beta1 > 0);

t_beta1(ok_t1) = beta1(ok_t1) ./ se_beta1(ok_t1);

pval_beta1(ok_t1) = 2 * tcdf(-abs(t_beta1(ok_t1)), df(ok_t1));

% BH-FDR
qval_beta1 = nan(n_use, 1);

sig_fdr_beta1 = false(n_use, 1);

valid_p1 = isfinite(pval_beta1);
if any(valid_p1)
    [sig_tmp, q_tmp] = bh_fdr(pval_beta1(valid_p1), alpha);
    qval_beta1(valid_p1) = q_tmp;
    sig_fdr_beta1(valid_p1) = sig_tmp;
end

% Put back to full-size maps
rho_full = nan(n_pixels, 1);
intercept_full = nan(n_pixels, 1);
slope_full = nan(n_pixels, 1);
qval_full = nan(n_pixels, 1);
sig_fdr_full = false(n_pixels, 1);
n_full = zeros(n_pixels, 1);
meanM_full = nan(n_pixels, 1);
meanY_full = nan(n_pixels, 1);
Sxx_full = nan(n_pixels, 1);
Sxy_full = nan(n_pixels, 1);

rho_full(idx) = rho;
intercept_full(idx) = beta0;
slope_full(idx) = beta1;
qval_full(idx) = qval_beta1;
sig_fdr_full(idx) = sig_fdr_beta1;
n_full(idx) = n;
meanM_full(idx) = meanM;
meanY_full(idx) = meanY;
Sxx_full(idx) = Sxx;
Sxy_full(idx) = Sxy;

results = struct();

results.sigma_frames = sigma_frames; % Standard deviation of Gaussian smoothing in frames
results.video_lengths = video_lengths;
results.mask         = reshape(mask, H, W);

results.rho_map       = reshape(rho_full, H, W);
results.intercept_map = reshape(intercept_full, H, W);
results.slope_map     = reshape(slope_full, H, W);
results.qval_map      = reshape(qval_full, H, W);
results.sig_fdr_map   = reshape(sig_fdr_full, H, W);
results.n_valid_map   = reshape(n_full, H, W);
results.mean_level_map = reshape(meanM_full, H, W);
results.mean_response_map = reshape(meanY_full, H, W);
results.level_ss_map = reshape(Sxx_full, H, W);
results.level_response_cross_map = reshape(Sxy_full, H, W);
end

function [sig, qval] = bh_fdr(p, alpha)
p = p(:);
m = numel(p);

[ps, order] = sort(p);
thresh = (1:m)' / m * alpha;

below = ps <= thresh;
sig = false(m, 1);

if any(below)
    k = find(below, 1, 'last');
    sig(order(1:k)) = true;
end

q_sorted = ps .* m ./ (1:m)';
q_sorted = min(q_sorted, 1);
q_sorted = flipud(cummin(flipud(q_sorted)));

qval = nan(m, 1);
qval(order) = q_sorted;
end