%% helper functions to obtain baseline value for every voxel
function voxel_baseline = find_baseline_all_voxel(video, fps, winSec, perc, spatialSigma)
if nargin < 2, fps = 40; end
if nargin < 3, winSec = 6; end
if nargin < 4, perc = 8; end
if nargin < 5, spatialSigma = 2; end

video = single(video);
[rows, cols, T] = size(video);
video2d = reshape(video, [], T);          % (pixels × T)

W  = max(1, round(winSec * fps));
hW = floor(W/2);
videoPad = padarray(video2d, [0 hW], 'replicate', 'both');

% --- choose built-in or fallback ----------------------------------
if exist('movpercentile','file') == 2
    disp('used movpercentile')
    F0 = movpercentile(videoPad, perc, W, 2);
else
    F0 = local_movpercentile_auto(videoPad, perc, W);   % custom fallback
end
F0 = F0(:, hW+1:end-hW);

F0 = reshape(F0, rows, cols, T);
if spatialSigma > 0 && exist('imgaussfilt3','file') == 2
    F0 = imgaussfilt3(F0, [spatialSigma spatialSigma eps]);
end

voxel_baseline = single(F0);
end

% -------- fallback helper (simple, works in any MATLAB) ------------
function out = local_movpercentile_auto(X, perc, W)

if exist('gpuDeviceCount','file') == 2 && gpuDeviceCount > 0
    g = gpuDevice;
    if contains(lower(g.Name), 'rtx')      % fast GPU present
        out = local_movpercentile_gpu(X, perc, W);
        return
    end
end

out = local_movpercentile_fast(X, perc, W); % default CPU path
end

% ---------- CPU, parfor + mink --------------------------------------
function out = local_movpercentile_fast(X, perc, W)
[N, T] = size(X);
k   = max(1, ceil(perc/100 * W));
hW  = floor(W/2);
out = zeros(N, T, 'like', X);

parfor t = 1:T
    lo  = max(1, t-hW);
    hi  = min(T, t+hW);
    seg = double(X(:, lo:hi));         % N × W
    kth = mink(seg, k, 2);             % k smallest along rows
    out(:, t) = kth(:, k);             % k-th smallest = percentile
end
out = cast(out, 'like', X);
end

% ---------- GPU version (RTX) ---------------------------------------
function out = local_movpercentile_gpu(X, perc, W)
[N, T] = size(X);
k   = max(1, ceil(perc/100 * W));
hW  = floor(W/2);

Xg   = gpuArray(double(X));
outg = zeros(N, T, 'like', Xg);

for t = 1:T                          % loop in fast GPU memory
    lo  = max(1, t-hW);
    hi  = min(T, t+hW);
    seg = Xg(:, lo:hi);              % N × W (on GPU)
    kth = mink(seg, k, 2);
    outg(:, t) = kth(:, k);
end
out = gather(cast(outg, 'like', X));
end