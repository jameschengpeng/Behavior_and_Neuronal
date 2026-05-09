function [noise_var_xt, qa] = estimate_noise_var_timecourse(X, mask, video_lengths)
%ESTIMATE_NOISE_VAR_TIMECOURSE Estimate per-pixel noise variance over time.
%
%   [noise_var_xt, qa] = estimate_noise_var_timecourse(X, mask)
%   [noise_var_xt, qa] = estimate_noise_var_timecourse(X, mask, video_lengths)
%
% Inputs
%   X             : n_pixels x T matrix
%   mask          : logical mask with numel(mask) == n_pixels
%   video_lengths : optional lengths of concatenated videos. If omitted,
%                   the full trace is treated as one video.
%
% Output
%   noise_var_xt : n_pixels x T matrix of estimated noise variance.
%                  Masked pixels are set to 0.
%   qa           : compact quality-assurance struct with classification and
%                  fit summaries.
%
% The function uses sigma_frames = 1. Homoscedastic pixels use the
% temporal-differencing MAD estimator. Heteroscedastic pixels are selected
% with a pooled fixed-effect model on log squared differences that estimates
% one shared signal-dependent slope across active pixels and pixel-specific
% intercepts. A pixel uses the time-varying model only when it gives enough
% per-pixel fit improvement and enough predicted noise dynamic range. The
% final fitted squared-difference variance is divided by 2 to convert back
% to noise-variance units.
%
% Boundary handling:
%   - The first and last frames of each video are handled with replicate
%     padding inside that video only.
%   - Temporal differences never cross video boundaries.
%   - If video_lengths is omitted, the full trace is treated as one video.

alpha = NaN; % Legacy QA field; classification now uses practical thresholds.
sigma_frames = 1;
min_model_improvement = 0;
min_noise_dynamic_range = 0.10;

[n_pixels, T] = size(X);
if numel(mask) ~= n_pixels
    error('mask size mismatch: numel(mask) must equal size(X,1).');
end
if T < 5
    error('Need at least 5 time points.');
end
if nargin < 3 || isempty(video_lengths)
    video_lengths = T;
end
video_lengths = validate_video_lengths(T, video_lengths);

mask_size = size(mask);
mask_vec = logical(mask(:));
noise_var_xt = zeros(n_pixels, T, 'like', X);

qa = struct();
qa.classification_method = 'pooled_log_common_slope_with_pixel_intercepts';
qa.alpha = alpha;
qa.sigma_frames = sigma_frames;
qa.min_model_improvement = min_model_improvement;
qa.min_noise_dynamic_range = min_noise_dynamic_range;
qa.mask = reshape(mask_vec, mask_size);
qa.video_lengths = video_lengths;

if ~any(mask_vec)
    qa.homoscedastic_map = false(mask_size);
    qa.heteroscedastic_map = false(mask_size);
    qa.qval_map = nan(mask_size);
    qa.homoscedastic_noise_var_map = zeros(mask_size);
    qa.pixel_intercept_map = nan(mask_size);
    qa.pixel_slope_map = nan(mask_size);
    qa.hetero_intercept_map = nan(mask_size);
    qa.model_improvement_map = nan(mask_size);
    qa.noise_dynamic_range_map = nan(mask_size);
    qa.log_y_floor_map = nan(mask_size);
    qa.common_slope_noisevar = 0;
    qa.common_slope_diffvar = 0;
    qa.common_slope_diffvar_raw = 0;
    qa.common_slope_logdiffvar = 0;
    qa.common_slope_logdiffvar_raw = 0;
    qa.common_vs_pixel_sse_ratio_median = NaN;
    qa.common_vs_pixel_sse_ratio_p95 = NaN;
    qa.n_heteroscedastic = 0;
    qa.n_homoscedastic = 0;
    qa.n_negative_interval_predictions = 0;
    return;
end

active_idx = find(mask_vec);
X_use = X(active_idx, :);
[Xs_use, M_pair, Y_pair, valid_pair, video_lengths] = build_segmentwise_pair_features(X_use, sigma_frames, video_lengths);
noise_var_const = estimate_noise_var_per_pixel_segmentwise(X_use, video_lengths);

Y_floor_source = double(Y_pair);
Y_floor_source(~valid_pair | ~isfinite(Y_floor_source) | (Y_floor_source <= 0)) = NaN;
log_y_floor = prctile(Y_floor_source, 5, 2);
log_y_floor(~isfinite(log_y_floor) | (log_y_floor <= 0)) = eps;

Y_log_input = double(Y_pair);
Y_log_input(~valid_pair | ~isfinite(Y_log_input) | (Y_log_input < 0)) = NaN;
logY_pair = log(max(Y_log_input, log_y_floor));
valid_log_pair = valid_pair & isfinite(M_pair) & isfinite(logY_pair);

M_work = double(M_pair);
logY_work = logY_pair;
M_work(~valid_log_pair) = 0;
logY_work(~valid_log_pair) = 0;

n_valid = sum(valid_log_pair, 2);
n_safe = max(double(n_valid), 1);
sumM = sum(M_work, 2);
sumLogY = sum(logY_work, 2);
sumM2 = sum(M_work .* M_work, 2);
sumLogY2 = sum(logY_work .* logY_work, 2);
sumMLogY = sum(M_work .* logY_work, 2);

meanM = sumM ./ n_safe;
meanLogY = sumLogY ./ n_safe;
Sxx = sumM2 - (sumM .^ 2) ./ n_safe;
Syy = sumLogY2 - (sumLogY .^ 2) ./ n_safe;
Sxy = sumMLogY - (sumM .* sumLogY) ./ n_safe;
Sxx = max(Sxx, 0);
Syy = max(Syy, 0);

valid_stats = (n_valid >= 3) & isfinite(Sxx) & isfinite(Sxy) & (Sxx > 0);
denom = sum(Sxx(valid_stats));
common_slope_logdiffvar_raw = 0;
if denom > 0
    common_slope_logdiffvar_raw = sum(Sxy(valid_stats)) / denom;
end
common_slope_logdiffvar = max(common_slope_logdiffvar_raw, 0);

intercept_log_fit = meanLogY - common_slope_logdiffvar .* meanM;

logYhat_common = intercept_log_fit + common_slope_logdiffvar .* double(M_pair);
logYhat_common(~valid_log_pair) = 0;
logY_res = logY_pair;
logY_res(~valid_log_pair) = 0;
res_common = logY_res - logYhat_common;
res_common(~valid_log_pair) = 0;
sse_common = sum(res_common .^ 2, 2);
sse_const = Syy;
model_improvement = 1 - (sse_common ./ max(sse_const, eps));
model_improvement(~isfinite(model_improvement) | sse_const <= 0) = 0;

M_min = double(M_pair);
M_max = double(M_pair);
M_min(~valid_log_pair) = Inf;
M_max(~valid_log_pair) = -Inf;
minM = min(M_min, [], 2);
maxM = max(M_max, [], 2);
signal_range = maxM - minM;
signal_range(~isfinite(signal_range)) = 0;
noise_dynamic_range = exp(common_slope_logdiffvar .* signal_range) - 1;
noise_dynamic_range(~isfinite(noise_dynamic_range)) = 0;

local_hetero = (common_slope_logdiffvar > 0) & valid_stats & ...
    (model_improvement >= min_model_improvement) & ...
    (noise_dynamic_range >= min_noise_dynamic_range);
local_homo = ~local_hetero;

noise_var_use = zeros(numel(active_idx), T, 'double');
noise_var_use(:,:) = repmat(noise_var_const, 1, T);

hetero_intercept_full = nan(n_pixels, 1);
homo_noise_full = nan(n_pixels, 1);
homo_noise_full(active_idx(local_homo)) = noise_var_const(local_homo);
n_negative_interval = 0;

if any(local_hetero)
    hetero_rows = find(local_hetero);
    valid_hetero = valid_log_pair(hetero_rows, :);
    diffvar_anchor = max(2 .* double(noise_var_const(hetero_rows)), eps);
    a_log = log(diffvar_anchor) - common_slope_logdiffvar .* meanM(hetero_rows);

    interval_logdiffvar = a_log + common_slope_logdiffvar .* M_pair(hetero_rows, :);
    interval_diffvar = exp(interval_logdiffvar);
    interval_noisevar = interval_diffvar / 2;

    n_negative_interval = 0;

    frame_noisevar = interval_to_frame_noisevar(interval_noisevar, Xs_use(hetero_rows, :), a_log, common_slope_logdiffvar, video_lengths, true);

    noise_var_use(hetero_rows, :) = frame_noisevar;
    hetero_intercept_full(active_idx(hetero_rows)) = exp(a_log) / 2;
end

noise_var_xt(active_idx, :) = cast(noise_var_use, 'like', X);

sig_vec = false(n_pixels, 1);
sig_vec(active_idx(local_hetero)) = true;
homo_vec = mask_vec & ~sig_vec;

qval_full = nan(n_pixels, 1);
intercept_full = nan(n_pixels, 1);
slope_full = nan(n_pixels, 1);
model_improvement_full = nan(n_pixels, 1);
noise_dynamic_range_full = nan(n_pixels, 1);
log_y_floor_full = nan(n_pixels, 1);
diffvar_anchor_full = max(2 .* double(noise_var_const), eps);
intercept_full(active_idx) = log(diffvar_anchor_full) - common_slope_logdiffvar .* meanM;
slope_full(active_idx) = common_slope_logdiffvar;
model_improvement_full(active_idx) = model_improvement;
noise_dynamic_range_full(active_idx) = noise_dynamic_range;
log_y_floor_full(active_idx) = log_y_floor;

qa.homoscedastic_map = reshape(homo_vec, mask_size);
qa.heteroscedastic_map = reshape(sig_vec, mask_size);
qa.qval_map = reshape(qval_full, mask_size);
qa.homoscedastic_noise_var_map = reshape(fill_missing_with_zero(homo_noise_full), mask_size);
qa.pixel_intercept_map = reshape(intercept_full, mask_size);
qa.pixel_slope_map = reshape(slope_full, mask_size);
qa.hetero_intercept_map = reshape(hetero_intercept_full, mask_size);
qa.model_improvement_map = reshape(model_improvement_full, mask_size);
qa.noise_dynamic_range_map = reshape(noise_dynamic_range_full, mask_size);
qa.log_y_floor_map = reshape(log_y_floor_full, mask_size);
qa.common_slope_noisevar = NaN;
qa.common_slope_diffvar = NaN;
qa.common_slope_diffvar_raw = NaN;
qa.common_slope_logdiffvar = common_slope_logdiffvar;
qa.common_slope_logdiffvar_raw = common_slope_logdiffvar_raw;
qa.common_vs_pixel_sse_ratio_median = NaN;
qa.common_vs_pixel_sse_ratio_p95 = NaN;
qa.n_heteroscedastic = nnz(sig_vec);
qa.n_homoscedastic = nnz(homo_vec);
qa.n_negative_interval_predictions = n_negative_interval;
end

function noise_var = estimate_noise_var_per_pixel_segmentwise(X, video_lengths)
[n_rows, ~] = size(X);
total_pairs = sum(max(video_lengths - 1, 0));
if total_pairs <= 0
    noise_var = zeros(n_rows, 1);
    return;
end

D = nan(n_rows, total_pairs);
cursor = 1;
start_idx = 1;
for seg = 1:numel(video_lengths)
    seg_len = video_lengths(seg);
    end_idx = start_idx + seg_len - 1;
    if seg_len >= 2
        D_seg = double(diff(X(:, start_idx:end_idx), 1, 2));
        n_cols = size(D_seg, 2);
        D(:, cursor:(cursor + n_cols - 1)) = D_seg;
        cursor = cursor + n_cols;
    end
    start_idx = end_idx + 1;
end

med_D = median(D, 2, 'omitnan');
mad_D = median(abs(D - med_D), 2, 'omitnan');
std_d = 1.4826 * mad_D;
noise_var = (std_d .^ 2) / 2;
noise_var(~isfinite(noise_var)) = 0;
end

function frame_noisevar = interval_to_frame_noisevar(interval_noisevar, Xs_rows, intercept, slope, video_lengths, log_model)
if nargin < 6
    log_model = false;
end

[n_rows, T] = size(Xs_rows);
frame_noisevar = zeros(n_rows, T);

start_idx = 1;
for seg = 1:numel(video_lengths)
    seg_len = video_lengths(seg);
    end_idx = start_idx + seg_len - 1;

    if seg_len == 1
        if log_model
            frame_noisevar(:, start_idx) = exp(intercept + slope .* Xs_rows(:, start_idx)) / 2;
        else
            frame_noisevar(:, start_idx) = max(intercept + slope .* Xs_rows(:, start_idx), 0) / 2;
        end
    else
        pair_cols = start_idx:(end_idx - 1);
        frame_noisevar(:, start_idx) = interval_noisevar(:, pair_cols(1));
        frame_noisevar(:, end_idx) = interval_noisevar(:, pair_cols(end));
        if seg_len > 2
            frame_noisevar(:, (start_idx + 1):(end_idx - 1)) = 0.5 * (interval_noisevar(:, pair_cols(1:end-1)) + interval_noisevar(:, pair_cols(2:end)));
        end
    end

    start_idx = end_idx + 1;
end
end

function x = fill_missing_with_zero(x)
x(~isfinite(x)) = 0;
end
