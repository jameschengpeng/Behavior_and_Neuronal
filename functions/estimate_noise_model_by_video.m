function noise_model = estimate_noise_model_by_video(X, mask, video_lengths)
%ESTIMATE_NOISE_MODEL_BY_VIDEO Estimate and compactly store pixel-time noise.
%
% Homoscedastic pixels are stored as one constant variance per pixel per
% video. Heteroscedastic pixels are stored with their full per-frame
% variance trace for that video. Use restore_noise_var_xt_from_model to
% rebuild the dense n_pixels x T matrix when needed.

[n_pixels, T] = size(X);
if numel(mask) ~= n_pixels
    error('mask size mismatch: numel(mask) must equal size(X,1).');
end

video_lengths = validate_video_lengths(T, video_lengths);
n_videos = numel(video_lengths);
mask_size = size(mask);

noise_model = struct();
noise_model.format_version = 1;
noise_model.storage_type = 'homoscedastic_constants_plus_heteroscedastic_traces';
noise_model.n_pixels = n_pixels;
noise_model.mask_size = mask_size;
noise_model.mask = logical(mask(:));
noise_model.video_lengths = video_lengths;
noise_model.homoscedastic_noise_var = zeros(n_pixels, n_videos, 'like', X);
noise_model.hetero_pixel_idx = cell(n_videos, 1);
noise_model.hetero_noise_var = cell(n_videos, 1);
noise_model.qa_by_video = cell(n_videos, 1);

video_end = cumsum(video_lengths);
video_start = [1; video_end(1:end-1) + 1];

for ii = 1:n_videos
    frame_idx = video_start(ii):video_end(ii);
    fprintf('Estimating compact pixel-time noise model: video %d/%d (%d frames)\n', ii, n_videos, video_lengths(ii));

    [noise_var_video, qa] = estimate_noise_var_timecourse(X(:, frame_idx), mask, video_lengths(ii));
    hetero_vec = logical(qa.heteroscedastic_map(:));
    homo_vec = logical(qa.homoscedastic_map(:));

    homo_noise = zeros(n_pixels, 1, 'like', noise_var_video);
    if any(homo_vec)
        homo_noise(homo_vec) = noise_var_video(homo_vec, 1);
    end

    noise_model.homoscedastic_noise_var(:, ii) = homo_noise;
    noise_model.hetero_pixel_idx{ii} = find(hetero_vec);
    noise_model.hetero_noise_var{ii} = noise_var_video(hetero_vec, :);
    noise_model.qa_by_video{ii} = qa;
end
end
