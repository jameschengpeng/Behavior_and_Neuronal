function [Xs, M_pair, Y_pair, valid_pair, video_lengths] = build_segmentwise_pair_features(X, sigma_frames, video_lengths)
%BUILD_SEGMENTWISE_PAIR_FEATURES Build within-video smoothed and differenced features.

[n_rows, T] = size(X);
if nargin < 3 || isempty(video_lengths)
    video_lengths = T;
end
video_lengths = validate_video_lengths(T, video_lengths);

Xs = zeros(n_rows, T, 'double');
M_pair = nan(n_rows, max(T - 1, 0));
Y_pair = nan(n_rows, max(T - 1, 0));
valid_pair = false(n_rows, max(T - 1, 0));

start_idx = 1;
for seg = 1:numel(video_lengths)
    seg_len = video_lengths(seg);
    end_idx = start_idx + seg_len - 1;
    X_seg = X(:, start_idx:end_idx);
    Xs_seg = double(temporal_gaussian_smooth(X_seg, sigma_frames));
    Xs(:, start_idx:end_idx) = Xs_seg;

    if seg_len >= 2
        pair_cols = start_idx:(end_idx - 1);
        Y_pair(:, pair_cols) = double(diff(X_seg, 1, 2)) .^ 2;
        M_pair(:, pair_cols) = 0.5 * (Xs_seg(:, 1:end-1) + Xs_seg(:, 2:end));
        valid_pair(:, pair_cols) = isfinite(M_pair(:, pair_cols)) & isfinite(Y_pair(:, pair_cols));
    end

    start_idx = end_idx + 1;
end
end