function Xs = temporal_gaussian_smooth(X, sigma_frames)
%TEMPORAL_GAUSSIAN_SMOOTH  Apply causal Gaussian smoothing along the time axis.
%
% Each row of X is smoothed independently with a symmetric Gaussian kernel
% of standard deviation sigma_frames.  Boundary frames are padded by
% replication to avoid edge artifacts.
%
% Inputs
%   X             : P x T matrix (pixels x time)
%   sigma_frames  : Gaussian sigma in frames; pass 0 to skip smoothing
%
% Output
%   Xs : P x T smoothed matrix (same size and type as X)

if sigma_frames <= 0
    Xs = X;
    return;
end

half_width = max(1, ceil(3 * sigma_frames));
t = -half_width:half_width;
kernel = exp(-(t.^2) / (2 * sigma_frames^2));
kernel = kernel / sum(kernel);

left_pad  = repmat(X(:, 1),   1, half_width);
right_pad = repmat(X(:, end), 1, half_width);
X_pad = [left_pad, X, right_pad];
Xs = conv2(X_pad, kernel, 'valid');
end
