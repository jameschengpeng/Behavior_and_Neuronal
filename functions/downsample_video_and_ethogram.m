function [video_ds, ethogram_ds] = downsample_video_and_ethogram(triple_channel_video, ethogram_matrix, rateT, rateH, rateW)
% DOWNSAMPLE_VIDEO_AND_ETHOGRAM downsamples a 4D video and a binary ethogram matrix.
% 
% Inputs:
%   - triple_channel_video: [T, H, W, 3] double or single or uint8
%   - ethogram_matrix: [T, C] binary matrix
%   - rateT: downsampling factor for time (e.g., 4 reduces 40Hz → 10Hz)
%   - rateH: downsampling factor for height (e.g., 2 halves vertical resolution)
%   - rateW: downsampling factor for width (e.g., 2 halves horizontal resolution)
%
% Outputs:
%   - video_ds: [T', H', W', 3] downsampled video
%   - ethogram_ds: [T', C] binary ethogram matrix (logical)

    % Validate dimensions
    [T, H, W, C] = size(triple_channel_video);
    assert(C == 3, 'Expected 3-channel input');
    assert(size(ethogram_matrix, 1) == T, 'Time dimension mismatch');

    % Downsample temporal dimension
    idxT = 1:rateT:T;
    video_ds = triple_channel_video(idxT, :, :, :);
    ethogram_ds = ethogram_matrix(idxT, :);

    % Initialize spatial downsampled video
    T_ds = size(video_ds, 1);
    H_ds = floor(H / rateH);
    W_ds = floor(W / rateW);
    video_temp = zeros(T_ds, H_ds, W_ds, 3, 'like', triple_channel_video);

    for t = 1:T_ds
        for c = 1:3
            % Extract one frame and channel: [H, W]
            frame = squeeze(video_ds(t, :, :, c));
            % Resize using imresize for better interpolation control
            if islogical(frame) || all(frame(:) == 0 | frame(:) == 1)
                % Use nearest neighbor for binary maps
                frame_ds = imresize(frame, [H_ds, W_ds], 'nearest');
            else
                % Use bilinear for continuous channel (dF/F)
                frame_ds = imresize(frame, [H_ds, W_ds], 'bilinear');
            end
            video_temp(t, :, :, c) = frame_ds;
        end
    end
    video_ds = video_temp;

    % Ensure ethogram remains binary (in case of logical conversion loss)
    ethogram_ds = logical(ethogram_ds);
end
