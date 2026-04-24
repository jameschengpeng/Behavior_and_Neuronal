function [xc, lags, info] = compute_region_crosscorr(data, arg2, arg3, arg4, arg5)
%COMPUTE_REGION_CROSSCORR  Normalized cross-correlation for NMF time courses.
%
% Preferred usage with NMF output C (K x T):
%   [xc, lags, info] = compute_region_crosscorr(C, idx1, idx2)
%   [xc, lags, info] = compute_region_crosscorr(C, idx1, idx2, max_lag)
%   [xc, lags, info] = compute_region_crosscorr(C, idx1, idx2, max_lag, fps)
%
% Also supports direct use with two time-series vectors:
%   [xc, lags, info] = compute_region_crosscorr(sig1, sig2)
%   [xc, lags, info] = compute_region_crosscorr(sig1, sig2, max_lag)
%   [xc, lags, info] = compute_region_crosscorr(sig1, sig2, max_lag, fps)
%
% Outputs:
%   xc   : normalized cross-correlation values
%   lags : lag values in frames
%   info : struct with peak lag/correlation in frames and seconds

    narginchk(2, 5);

    use_component_mode = ndims(data) == 2 && size(data, 1) > 1 && isscalar(arg2) && ...
        nargin >= 3 && isscalar(arg3);

    if use_component_mode
        C = data;
        idx1 = arg2;
        idx2 = arg3;
        assert(idx1 == floor(idx1) && idx1 >= 1 && idx1 <= size(C, 1), ...
            'idx1 must be a valid row index into C.');
        assert(idx2 == floor(idx2) && idx2 >= 1 && idx2 <= size(C, 1), ...
            'idx2 must be a valid row index into C.');
        sig1 = double(C(idx1, :).');
        sig2 = double(C(idx2, :).');
        if nargin < 4 || isempty(arg4)
            max_lag = numel(sig1) - 1;
        else
            max_lag = arg4;
        end
        if nargin < 5 || isempty(arg5)
            fps = 1;
        else
            fps = arg5;
        end
        info.component_idx_1 = idx1;
        info.component_idx_2 = idx2;
    else
        sig1 = double(data(:));
        sig2 = double(arg2(:));
        assert(numel(sig1) == numel(sig2), 'sig1 and sig2 must have the same length.');
        if nargin < 3 || isempty(arg3)
            max_lag = numel(sig1) - 1;
        else
            max_lag = arg3;
        end
        if nargin < 4 || isempty(arg4)
            fps = 1;
        else
            fps = arg4;
        end
        info.component_idx_1 = [];
        info.component_idx_2 = [];
    end

    T = numel(sig1);
    assert(numel(sig2) == T, 'The two signals must have the same length.');
    assert(isscalar(max_lag) && max_lag == floor(max_lag) && max_lag >= 0, ...
        'max_lag must be a nonnegative integer.');
    assert(max_lag < T, 'max_lag must be smaller than the signal length.');
    assert(isscalar(fps) && fps > 0, 'fps must be a positive scalar.');

    sig1 = sig1 - mean(sig1);
    sig2 = sig2 - mean(sig2);

    [xc, lags] = xcorr(sig1, sig2, max_lag, 'coeff');

    [peak_corr, peak_idx] = max(xc);
    peak_lag_frames = lags(peak_idx);

    info.peak_corr = peak_corr;
    info.peak_lag_frames = peak_lag_frames;
    info.peak_lag_seconds = peak_lag_frames / fps;
    info.zero_lag_corr = xc(lags == 0);
end