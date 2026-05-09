function [noise_var_xt, qa_by_video] = estimate_noise_var_timecourse_by_video(X, mask, video_lengths)
%ESTIMATE_NOISE_VAR_TIMECOURSE_BY_VIDEO Estimate pixel-time noise one video at a time.
% This function relies on estimate_noise_var_timecourse, which deals with
% each video one-by-one

[n_pixels, T] = size(X);
if numel(mask) ~= n_pixels
    error('mask size mismatch: numel(mask) must equal size(X,1).');
end

video_lengths = validate_video_lengths(T, video_lengths);
noise_var_xt = zeros(n_pixels, T, 'like', X);
qa_by_video = cell(numel(video_lengths), 1);

video_end = cumsum(video_lengths);
video_start = [1; video_end(1:end-1) + 1];

for ii = 1:numel(video_lengths)
    frame_idx = video_start(ii):video_end(ii);
    fprintf('Estimating pixel-time noise variance: video %d/%d (%d frames)\n', ii, numel(video_lengths), video_lengths(ii));
    [noise_var_xt(:, frame_idx), qa_by_video{ii}] = estimate_noise_var_timecourse(X(:, frame_idx), mask, video_lengths(ii));
end
end