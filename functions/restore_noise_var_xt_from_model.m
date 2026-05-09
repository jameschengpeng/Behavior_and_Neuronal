function noise_var_xt = restore_noise_var_xt_from_model(noise_model)
%RESTORE_NOISE_VAR_XT_FROM_MODEL Restore dense pixel-time noise variance.
%
% Input is the compact struct produced by estimate_noise_model_by_video.

required_fields = {'n_pixels', 'video_lengths', 'homoscedastic_noise_var', 'hetero_pixel_idx', 'hetero_noise_var'};
for ii = 1:numel(required_fields)
    if ~isfield(noise_model, required_fields{ii})
        error('noise_model is missing required field "%s".', required_fields{ii});
    end
end

n_pixels = noise_model.n_pixels;
video_lengths = validate_video_lengths(sum(noise_model.video_lengths), noise_model.video_lengths);
n_videos = numel(video_lengths);
T = sum(video_lengths);

noise_var_xt = zeros(n_pixels, T, 'like', noise_model.homoscedastic_noise_var);

video_end = cumsum(video_lengths);
video_start = [1; video_end(1:end-1) + 1];

for ii = 1:n_videos
    frame_idx = video_start(ii):video_end(ii);
    noise_var_xt(:, frame_idx) = repmat(noise_model.homoscedastic_noise_var(:, ii), 1, video_lengths(ii));

    hetero_idx = noise_model.hetero_pixel_idx{ii};
    if ~isempty(hetero_idx)
        hetero_noise = noise_model.hetero_noise_var{ii};
        if size(hetero_noise, 1) ~= numel(hetero_idx) || size(hetero_noise, 2) ~= video_lengths(ii)
            error('noise_model heteroscedastic trace size mismatch for video %d.', ii);
        end
        noise_var_xt(hetero_idx, frame_idx) = hetero_noise;
    end
end
end
