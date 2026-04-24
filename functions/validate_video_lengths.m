function video_lengths = validate_video_lengths(T, video_lengths)
%VALIDATE_VIDEO_LENGTHS Validate concatenated video segment lengths.

video_lengths = double(video_lengths(:));
if isempty(video_lengths)
    error('video_lengths must be non-empty when provided.');
end
if any(~isfinite(video_lengths)) || any(video_lengths < 1) || any(video_lengths ~= round(video_lengths))
    error('video_lengths must contain positive integers.');
end
if sum(video_lengths) ~= T
    error('sum(video_lengths) must equal size(X, 2).');
end
end