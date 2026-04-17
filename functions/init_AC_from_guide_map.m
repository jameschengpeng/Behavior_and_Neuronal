function [Aact, C, info] = init_AC_from_guide_map(Xr, guide_map, mask, K, opts)
%INIT_AC_FROM_GUIDE_MAP Initialize A and C from a 2-D spatial guide map.
%
% Uses marker-controlled watershed on the smoothed guide map to partition
% the field of view into K regions, then initializes each A column from
% guide values within that region and C by nonneg least squares on Xr.
%
% guide_map : H x W nonneg image (e.g. noise-corrected signal variance).

if nargin < 5
    opts = struct();
end
opts = set_default_opts(opts);

[Pm, T] = size(Xr);
[H, W] = size(guide_map);
mask_img = reshape(mask, H, W) ~= 0;
idx = find(mask_img(:));
assert(numel(idx) == Pm, 'mask and Xr size mismatch in init_AC_from_guide_map.');

if opts.AC_init_mode == "svd"
    [Aact, C] = svd_fallback_init(Xr, K, opts.init_evt_ridge);
    info = struct('mode', 'svd', 'centers_rc', []);
    return;
elseif opts.AC_init_mode ~= "event_projection"
    error('opts.AC_init_mode must be "event_projection" or "svd".');
end

guide = double(guide_map);
guide(~mask_img) = 0;

% Smooth for peak detection and watershed partitioning
guide_smooth = guide;
if opts.init_evt_sigma > 0
    guide_smooth = imgaussfilt(guide, opts.init_evt_sigma);
    guide_smooth(~mask_img) = 0;
end

guide_max = max(guide_smooth(:));
if guide_max <= 0 || ~isfinite(guide_max)
    [Aact, C] = svd_fallback_init(Xr, K, opts.init_evt_ridge);
    info = struct('mode', 'svd_fallback', 'centers_rc', []);
    return;
end
guide_norm = guide_smooth / (guide_max + eps);

% ---- Step 1: find K seed centers from regional maxima --------------------
centers_rc = select_seed_centers(guide_norm, mask_img, K, opts);
K_init = size(centers_rc, 1);
if K_init == 0
    [Aact, C] = svd_fallback_init(Xr, K, opts.init_evt_ridge);
    info = struct('mode', 'svd_fallback', 'centers_rc', centers_rc);
    return;
end

% ---- Step 2: threshold to define the active region -----------------------
nz_vals = guide_smooth(mask_img & guide_smooth > 0);
if isempty(nz_vals)
    tau = 0;
else
    tau = prctile(nz_vals, opts.init_active_percentile);
end
active_region = mask_img & (guide_smooth >= tau);

% Ensure every seed pixel is part of the active region
for k = 1:K_init
    active_region(centers_rc(k,1), centers_rc(k,2)) = true;
end

% ---- Step 3: marker-controlled watershed to partition the region ---------
labels = assign_watershed_regions(guide_smooth, active_region, centers_rc);

% ---- Step 4: build A columns from guide values in each region ------------
guide_act = cast(guide_smooth(idx), 'like', Xr);
Aact = zeros(Pm, K, 'like', Xr);
for k = 1:K_init
    support = (labels(idx) == k);
    ak = zeros(Pm, 1, 'like', Xr);
    ak(support) = guide_act(support);
    Aact(:, k) = ak;
end

% ---- Step 5: solve C by NNLS --------------------------------------------
AAt = Aact' * Aact + opts.init_evt_ridge * eye(K, 'like', Xr);
C = AAt \ (Aact' * Xr);
C = max(C, 0);

% Zero dead columns
zero_cols = find(sum(Aact, 1) <= 0 | sum(C, 2)' <= 0);
if ~isempty(zero_cols)
    Aact(:, zero_cols) = 0;
    C(zero_cols, :) = 0;
end

info = struct('mode', 'event_projection', 'centers_rc', centers_rc, ...
    'guide_max', guide_max, 'n_init', K_init, 'zero_cols', zero_cols, ...
    'labels', labels, 'active_tau', tau);
end

%% ---- Helper: select seed centres ----------------------------------------
function centers_rc = select_seed_centers(guide, mask_img, K, opts)
peak_mask = imregionalmax(guide) & mask_img & (guide > 0);

[peak_r, peak_c] = find(peak_mask);
peak_score = guide(peak_mask);

if isempty(peak_score)
    [all_r, all_c] = find(mask_img & guide > 0);
    peak_r = all_r;
    peak_c = all_c;
    peak_score = guide(sub2ind(size(guide), all_r, all_c));
end

[~, order] = sort(peak_score, 'descend');
peak_r = peak_r(order);
peak_c = peak_c(order);

centers_rc = zeros(0, 2);
for ii = 1:numel(peak_r)
    cand = [peak_r(ii), peak_c(ii)];
    if isempty(centers_rc)
        centers_rc = cand;
    else
        d2 = sum((centers_rc - cand).^2, 2);
        if all(d2 >= opts.init_evt_min_peak_dist^2)
            centers_rc(end+1, :) = cand; %#ok<AGROW>
        end
    end

    if size(centers_rc, 1) >= K
        break;
    end
end

if size(centers_rc, 1) > K
    centers_rc = centers_rc(1:K, :);
end
end

%% ---- Helper: marker-controlled watershed --------------------------------
function labels = assign_watershed_regions(guide_smooth, active_region, centers_rc)
% Partition the active region into K territories, one per seed centre.
% Each connected component of the active region is handled separately:
%   - 0 seeds  -> unassigned (label 0)
%   - 1 seed   -> entire component gets that label
%   - 2+ seeds -> marker-controlled watershed splits the component

K_init = size(centers_rc, 1);
[H, W] = size(guide_smooth);
labels = zeros(H, W);

CC = bwconncomp(active_region, 8);

for ci = 1:CC.NumObjects
    pix = CC.PixelIdxList{ci};
    comp_mask = false(H, W);
    comp_mask(pix) = true;

    % Which seeds fall inside this connected component?
    seeds_here = [];
    for k = 1:K_init
        if comp_mask(centers_rc(k,1), centers_rc(k,2))
            seeds_here(end+1) = k; %#ok<AGROW>
        end
    end

    if isempty(seeds_here)
        continue;          % no seed -> unassigned
    end

    if numel(seeds_here) == 1
        labels(pix) = seeds_here(1);
        continue;
    end

    % Multiple seeds -> marker-controlled watershed within this component
    marker_img = false(H, W);
    for k = seeds_here
        marker_img(centers_rc(k,1), centers_rc(k,2)) = true;
    end

    % Invert the guide (peaks -> valleys for watershed)
    gmax = max(guide_smooth(pix));
    guide_inv = gmax - guide_smooth;
    guide_inv(~comp_mask) = gmax + 1;   % barrier outside the component

    guide_marked = imimposemin(guide_inv, marker_img);
    L = watershed(guide_marked);

    % Map watershed basin labels back to seed indices
    for k = seeds_here
        ws_label = L(centers_rc(k,1), centers_rc(k,2));
        if ws_label > 0
            labels(L == ws_label & comp_mask) = k;
        end
    end
end
end

function [Aact, C] = svd_fallback_init(Xr, K, ridge)
try
    [U, S, V] = svds(Xr, K);
catch
    [U, S, V] = svd(Xr, 'econ');
    U = U(:, 1:K);
    S = S(1:K, 1:K);
    V = V(:, 1:K);
end

for k = 1:K
    if sum(U(:, k)) < 0
        U(:, k) = -U(:, k);
        V(:, k) = -V(:, k);
    end
end

Aact = cast(max(0, U * sqrt(S)), 'like', Xr);
C = cast(max(0, sqrt(S) * V'), 'like', Xr);

dead_cols = find(sum(Aact, 1) <= 0 | sum(C, 2)' <= 0);
for k = dead_cols
    Aact(:, k) = max(0, rand(size(Aact, 1), 1));
end

AAt = Aact' * Aact + ridge * eye(K, 'like', Xr);
C = max(0, AAt \ (Aact' * Xr));
end

function opts = set_default_opts(opts)
def.AC_init_mode = "event_projection";
def.init_evt_sigma = 8;
def.init_evt_min_peak_dist = 100;
def.init_active_percentile = 25;
def.init_evt_ridge = 1e-6;

f = fieldnames(def);
for i = 1:numel(f)
    if ~isfield(opts, f{i})
        opts.(f{i}) = def.(f{i});
    end
end

opts.AC_init_mode = string(opts.AC_init_mode);
opts.init_evt_sigma = max(0, opts.init_evt_sigma);
opts.init_evt_min_peak_dist = max(1, round(opts.init_evt_min_peak_dist));
opts.init_active_percentile = min(100, max(0, opts.init_active_percentile));
opts.init_evt_ridge = max(0, opts.init_evt_ridge);
end