function noise_model = concat_noise_models_by_video(noise_models)
%CONCAT_NOISE_MODELS_BY_VIDEO Concatenate compact noise models by video.

noise_models = noise_models(:);
noise_models = noise_models(~cellfun(@isempty, noise_models));
if isempty(noise_models)
    error('noise_models must contain at least one model.');
end

noise_model = noise_models{1};
for ii = 2:numel(noise_models)
    this_model = noise_models{ii};
    if this_model.n_pixels ~= noise_model.n_pixels
        error('All noise models must have the same n_pixels.');
    end
    if ~isequal(this_model.mask_size, noise_model.mask_size) || ~isequal(this_model.mask, noise_model.mask)
        error('All noise models must have the same mask.');
    end

    noise_model.video_lengths = [noise_model.video_lengths; this_model.video_lengths(:)];
    noise_model.homoscedastic_noise_var = [noise_model.homoscedastic_noise_var, this_model.homoscedastic_noise_var];
    noise_model.hetero_pixel_idx = [noise_model.hetero_pixel_idx; this_model.hetero_pixel_idx(:)];
    noise_model.hetero_noise_var = [noise_model.hetero_noise_var; this_model.hetero_noise_var(:)];
    noise_model.qa_by_video = [noise_model.qa_by_video; this_model.qa_by_video(:)];
end
end
