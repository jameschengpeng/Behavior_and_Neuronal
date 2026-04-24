function [X_subset, video_lengths_subset, selected_video_idx, frame_idx] = sample_random_videos_from_concatenated(X, video_lengths, num_videos_to_keep, rng_seed)
%SAMPLE_RANDOM_VIDEOS_FROM_CONCATENATED Randomly keep whole videos from a concatenated movie.

[~, T] = size(X);
video_lengths = validate_video_lengths(T, video_lengths);

n_videos = numel(video_lengths);
if isempty(num_videos_to_keep) || num_videos_to_keep >= n_videos
    selected_video_idx = (1:n_videos).';
else
    if ~isscalar(num_videos_to_keep) || num_videos_to_keep < 1 || num_videos_to_keep ~= round(num_videos_to_keep)
        error('num_videos_to_keep must be a positive integer.');
    end
    if nargin >= 4 && ~isempty(rng_seed)
        rng(rng_seed);
    end
    selected_video_idx = sort(randperm(n_videos, num_videos_to_keep)).';
end

video_end = cumsum(video_lengths);
video_start = [1; video_end(1:end-1) + 1];

frame_idx = cell(numel(selected_video_idx), 1);
for ii = 1:numel(selected_video_idx)
    v = selected_video_idx(ii);
    frame_idx{ii} = (video_start(v):video_end(v)).';
end
frame_idx = cell2mat(frame_idx);

X_subset = X(:, frame_idx);
video_lengths_subset = video_lengths(selected_video_idx);
end