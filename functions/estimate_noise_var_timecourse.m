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
% The function uses alpha = 0.05 and sigma_frames = 1. Pixels classified as
% homoscedastic use the temporal-differencing MAD estimator. Pixels
% classified as heteroscedastic use outputs from
% test_temporal_heteroscedasticity_fast, then a shared slope with
% pixel-specific intercepts to build a time-varying variance map. The
% fitted squared-difference variance is divided by 2 to convert back to
% noise-variance units.
%
% Boundary handling:
%   - The first and last frames of each video are handled with replicate
%     padding inside that video only.
%   - Temporal differences never cross video boundaries.
%   - If video_lengths is omitted, the full trace is treated as one video.

alpha = 0.05;
sigma_frames = 1;

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
qa.alpha = alpha;
qa.sigma_frames = sigma_frames;
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
    qa.common_slope_noisevar = 0;
    qa.common_slope_diffvar = 0;
    qa.common_vs_pixel_sse_ratio_median = NaN;
    qa.common_vs_pixel_sse_ratio_p95 = NaN;
    qa.n_heteroscedastic = 0;
    qa.n_homoscedastic = 0;
    qa.n_negative_interval_predictions = 0;
    return;
end

active_idx = find(mask_vec);
results = test_temporal_heteroscedasticity_fast(X, reshape(mask_vec, mask_size), alpha, sigma_frames, video_lengths);
X_use = X(active_idx, :);
[Xs_use, M_pair, Y_pair, valid_pair, video_lengths] = build_segmentwise_pair_features(X_use, sigma_frames, video_lengths);
noise_var_const = estimate_noise_var_per_pixel_segmentwise(X_use, video_lengths);

qval_full = results.qval_map(:);
intercept_full = results.intercept_map(:);
slope_full = results.slope_map(:);
n_valid_full = results.n_valid_map(:);
meanM_full = results.mean_level_map(:);
meanY_full = results.mean_response_map(:);
Sxx_full = results.level_ss_map(:);
Sxy_full = results.level_response_cross_map(:);

sig_vec = logical(results.sig_fdr_map(:)) & mask_vec;
homo_vec = mask_vec & ~sig_vec;

local_hetero = sig_vec(active_idx);
local_homo = ~local_hetero;
beta0 = double(intercept_full(active_idx));
beta1 = double(slope_full(active_idx));
n_valid = double(n_valid_full(active_idx));
meanM = double(meanM_full(active_idx));
meanY = double(meanY_full(active_idx));
Sxx = double(Sxx_full(active_idx));
Sxy = double(Sxy_full(active_idx));

noise_var_use = zeros(numel(active_idx), T, 'double');
if any(local_homo)
    noise_var_use(local_homo, :) = repmat(noise_var_const(local_homo), 1, T);
end

hetero_intercept_full = nan(n_pixels, 1);
homo_noise_full = nan(n_pixels, 1);
homo_noise_full(active_idx(local_homo)) = noise_var_const(local_homo);
fitted_hetero = false(numel(active_idx), 1);

common_slope_diffvar = 0;
common_slope_noisevar = 0;
ratio_median = NaN;
ratio_p95 = NaN;
n_negative_interval = 0;

if any(local_hetero)
    hetero_rows = find(local_hetero & (n_valid >= 3));
    if ~isempty(hetero_rows)
        valid_hetero = valid_pair(hetero_rows, :);
        valid_stats = isfinite(Sxx(hetero_rows)) & isfinite(Sxy(hetero_rows));
        denom = sum(Sxx(hetero_rows(valid_stats)));
        if denom > 0
            common_slope_diffvar = sum(Sxy(hetero_rows(valid_stats))) / denom;
        end

        b_pix = beta1;
        a_common = meanY(hetero_rows) - common_slope_diffvar .* meanM(hetero_rows);
        a_pix = beta0(hetero_rows);

        interval_diffvar_raw = a_common + common_slope_diffvar .* M_pair(hetero_rows, :);
        interval_diffvar = interval_diffvar_raw;
        interval_diffvar = max(interval_diffvar, 0);
        interval_noisevar = interval_diffvar / 2;

        n_negative_interval = nnz((interval_diffvar_raw < 0) & valid_hetero);

        frame_noisevar = interval_to_frame_noisevar(interval_noisevar, Xs_use(hetero_rows, :), a_common, common_slope_diffvar, video_lengths);

        noise_var_use(hetero_rows, :) = frame_noisevar;
        fitted_hetero(hetero_rows) = true;
        hetero_intercept_full(active_idx(hetero_rows)) = a_common / 2;

        Yhat_common = a_common + common_slope_diffvar .* M_pair(hetero_rows, :);
        Yhat_pix = a_pix + b_pix(hetero_rows) .* M_pair(hetero_rows, :);
        Yhat_common(~valid_hetero) = 0;
        Yhat_pix(~valid_hetero) = 0;

        Y_res = Y_pair(hetero_rows, :);
        Y_res(~valid_hetero) = 0;

        res_common = Y_res - Yhat_common;
        res_pix = Y_res - Yhat_pix;

        sse_common = sum(res_common .^ 2, 2);
        sse_pix = sum(res_pix .^ 2, 2);
        ratio = sse_common ./ max(sse_pix, eps);
        finite_ratio = ratio(isfinite(ratio));
        if ~isempty(finite_ratio)
            ratio_median = median(finite_ratio);
            ratio_p95 = prctile(finite_ratio, 95);
        end
    end

    remaining_hetero = local_hetero & ~fitted_hetero;
    if any(remaining_hetero)
        noise_var_use(remaining_hetero, :) = repmat(noise_var_const(remaining_hetero), 1, T);
        homo_noise_full(active_idx(remaining_hetero)) = noise_var_const(remaining_hetero);
        hetero_intercept_full(active_idx(remaining_hetero)) = NaN;
        sig_vec(active_idx(remaining_hetero)) = false;
        homo_vec(active_idx(remaining_hetero)) = true;
    end

    common_slope_noisevar = common_slope_diffvar / 2;
end

noise_var_xt(active_idx, :) = cast(noise_var_use, 'like', X);

qa.homoscedastic_map = reshape(homo_vec, mask_size);
qa.heteroscedastic_map = reshape(sig_vec, mask_size);
qa.qval_map = reshape(qval_full, mask_size);
qa.homoscedastic_noise_var_map = reshape(fill_missing_with_zero(homo_noise_full), mask_size);
qa.pixel_intercept_map = reshape(intercept_full, mask_size);
qa.pixel_slope_map = reshape(slope_full, mask_size);
qa.hetero_intercept_map = reshape(hetero_intercept_full, mask_size);
qa.common_slope_noisevar = common_slope_noisevar;
qa.common_slope_diffvar = common_slope_diffvar;
qa.common_vs_pixel_sse_ratio_median = ratio_median;
qa.common_vs_pixel_sse_ratio_p95 = ratio_p95;
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

function frame_noisevar = interval_to_frame_noisevar(interval_noisevar, Xs_rows, a_common, common_slope_diffvar, video_lengths)
[n_rows, T] = size(Xs_rows);
frame_noisevar = zeros(n_rows, T);

start_idx = 1;
for seg = 1:numel(video_lengths)
    seg_len = video_lengths(seg);
    end_idx = start_idx + seg_len - 1;

    if seg_len == 1
        frame_noisevar(:, start_idx) = max(a_common + common_slope_diffvar .* Xs_rows(:, start_idx), 0) / 2;
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