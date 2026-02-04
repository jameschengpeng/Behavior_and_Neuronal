function Z = nmf_behavior_onset_zscore(C, ethogram_mat, fps, preSec, postSec)
%NMF_BEHAVIOR_ONSET_ZSCORE
%
% Compute onset-triggered Z-scores between NMF temporal components C
% and sparse behavior ethograms.
%
% Inputs:
%   C            : K x T temporal components
%   ethogram_mat : T x 9 binary ethogram matrix
%   fps          : frame rate (Hz)
%   preSec       : seconds BEFORE onset (default = 2)
%   postSec      : seconds AFTER onset  (default = 0)
%
% Output:
%   Z            : K x 9 matrix of Z-scores

% ---------------- defaults ----------------
if nargin < 4 || isempty(preSec)
    preSec = 2.0;
end
if nargin < 5 || isempty(postSec)
    postSec = 0.0;
end

exclusionSec = 2.0;   % exclude +/- around onsets for null sampling
nPerm        = 1000;  % null samples
minOnsets    = 3;     % minimum valid onsets required
% ------------------------------------------

% Dimensions
[K, T] = size(C);

if size(ethogram_mat,1) ~= T
    error('ethogram_mat must be T x 9 with T matching size(C,2).');
end

B = ethogram_mat ~= 0;

% Convert window sizes to frames
Lpre  = round(preSec  * fps);
Lpost = round(postSec * fps);

winLength = Lpre + Lpost + 1;

% Exclusion radius for null sampling
excl = round(exclusionSec * fps);

% Output matrix
Z = nan(K, 9);

% Loop over behaviors
for b = 1:9

    y = B(:, b);

    % ---- find onsets ----
    yPrev = [false; y(1:end-1)];
    onsets_all = find(y & ~yPrev);

    % ---- keep only onsets with valid window ----
    onsets_valid = onsets_all( ...
        onsets_all > Lpre & ...
        onsets_all + Lpost <= T );

    if numel(onsets_valid) < minOnsets
        continue;
    end

    % ---- observed statistic: mean activity in window ----
    obsVals = nan(K, numel(onsets_valid));

    for k = 1:numel(onsets_valid)
        t0 = onsets_valid(k);
        idx = (t0 - Lpre):(t0 + Lpost);
        obsVals(:, k) = mean(C(:, idx), 2);
    end

    obs = mean(obsVals, 2);

    % ---- build null candidates ----
    valid = true(T,1);

    % cannot use times too close to edges
    valid(1:Lpre) = false;
    valid((T-Lpost+1):T) = false;

    % cannot be during behavior
    valid(y) = false;

    % exclude around all onsets
    bad = false(T,1);
    for k = 1:numel(onsets_all)
        lo = max(1, onsets_all(k) - excl);
        hi = min(T, onsets_all(k) + excl);
        bad(lo:hi) = true;
    end
    valid(bad) = false;

    candidates = find(valid);

    if isempty(candidates)
        continue;
    end

    % ---- permutation null distribution ----
    nEvents = numel(onsets_valid);
    nullStats = nan(K, nPerm);

    for p = 1:nPerm
        idx0 = candidates(randperm(numel(candidates), nEvents));

        tmpVals = nan(K, nEvents);
        for k = 1:nEvents
            t0 = idx0(k);
            idx = (t0 - Lpre):(t0 + Lpost);
            tmpVals(:, k) = mean(C(:, idx), 2);
        end

        nullStats(:, p) = mean(tmpVals, 2);
    end

    % ---- compute Z-score ----
    muNull = mean(nullStats, 2);
    sdNull = std(nullStats, 0, 2);

    sdNull(sdNull == 0) = nan;

    Z(:, b) = (obs - muNull) ./ sdNull;
end
end
