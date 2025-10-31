function save_volume_to_tiff(volume, filename)
% SAVE_VOLUME_TO_TIFF saves a 3D matrix [T, H, W] to a multipage TIFF.
%
% Inputs:
%   - volume: a 3D matrix of size [T, H, W]
%   - filename: output TIFF file path (e.g., 'output.tif')
%
% Notes:
%   - This function normalizes values to 8-bit if needed
%   - Overwrites existing file if it exists

    % Validate input
    if ndims(volume) ~= 3
        error('Input must be a 3D matrix of size [T, H, W]');
    end

    [T, H, W] = size(volume);

    % Normalize and convert to uint8 for TIFF storage
    min_val = min(volume(:));
    max_val = max(volume(:));
    if max_val > min_val
        volume_norm = uint8(255 * (volume - min_val) / (max_val - min_val));
    else
        volume_norm = uint8(zeros(size(volume)));  % Flat image
    end

    % Write TIFF
    for t = 1:T
        frame = squeeze(volume_norm(t, :, :));  % [H, W]
        if t == 1
            imwrite(frame, filename, 'TIFF', 'Compression', 'none');
        else
            imwrite(frame, filename, 'TIFF', 'WriteMode', 'append', 'Compression', 'none');
        end
    end

    fprintf('Saved %d frames to %s\n', T, filename);
end
