function [Aact, C, info] = init_AC_from_event_projection(Xr, evt_domain_projection, mask, K, opts)
%INIT_AC_FROM_EVENT_PROJECTION Initialize A and C from AQuA2 event projection.
% Builds localized spatial seeds from high-probability event regions and
% initializes C by nonnegative least squares on the residual Xr.

if nargin < 5
    opts = struct();
end
opts = set_default_opts(opts);

[Pm, T] = size(Xr);
[H, W] = size(evt_domain_projection);
mask_img = reshape(mask, H, W) ~= 0;
idx = find(mask_img(:));
assert(numel(idx) == Pm, 'mask and Xr size mismatch in init_AC_from_event_projection.');

guide = double(evt_domain_projection);
guide(~mask_img) = 0;

if opts.init_evt_sigma > 0
    guide = imgaussfilt(guide, opts.init_evt_sigma);
    guide(~mask_img) = 0;
end

guide_max = max(guide(:));
if guide_max <= 0 || ~isfinite(guide_max)
    [Aact, C] = svd_fallback_init(Xr, K, opts.init_evt_ridge);
    info = struct('mode', 'svd_fallback', 'centers_rc', []);
    return;
end
guide = guide / (guide_max + eps);

centers_rc = select_seed_centers(guide, mask_img, K, opts);
if size(centers_rc, 1) < K
    [Aact, C] = svd_fallback_init(Xr, K, opts.init_evt_ridge);
    info = struct('mode', 'svd_fallback', 'centers_rc', centers_rc);
    return;
end

guide_act = guide(idx);
Aact = zeros(Pm, K);
target_pixels = ceil(opts.init_evt_min_frac * Pm);
max_pixels = max(target_pixels, floor(opts.init_evt_max_frac * Pm));

for k = 1:K
    support = select_seed_connected_support(guide, mask_img, centers_rc(k, :), target_pixels, max_pixels, opts);
    ak = guide_act;
    ak(~support) = 0;

    if max(ak) <= 0
        ak = double(support);
    end

    Aact(:, k) = ak;
end

dead_cols = find(sum(Aact, 1) <= 0);
if ~isempty(dead_cols)
    [Aact, C] = svd_fallback_init(Xr, K, opts.init_evt_ridge);
    info = struct('mode', 'svd_fallback', 'centers_rc', centers_rc);
    return;
end

AAt = Aact' * Aact + opts.init_evt_ridge * eye(K, 'like', Xr);
C = AAt \ (Aact' * Xr);
C = max(C, 0);

% If a column is temporally dead, fall back to a robust SVD init rather than
% injecting random activity.
if any(sum(C, 2) <= 0)
    [Aact, C] = svd_fallback_init(Xr, K, opts.init_evt_ridge);
    info = struct('mode', 'svd_fallback', 'centers_rc', centers_rc);
    return;
end

info = struct('mode', 'event_projection', 'centers_rc', centers_rc, 'guide_max', guide_max);
end

function centers_rc = select_seed_centers(guide, mask_img, K, opts)
peak_mask = imregionalmax(guide);
peak_mask = peak_mask & mask_img & (guide > 0);

[peak_r, peak_c] = find(peak_mask);
peak_score = guide(peak_mask);

if isempty(peak_score)
    [all_r, all_c] = find(mask_img & guide > 0);
    peak_r = all_r;
    peak_c = all_c;
    peak_score = guide(mask_img & guide > 0);
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

if size(centers_rc, 1) < K
    [all_r, all_c] = find(mask_img);
    all_score = guide(mask_img);
    [~, order_all] = sort(all_score, 'descend');
    all_r = all_r(order_all);
    all_c = all_c(order_all);
    for ii = 1:numel(all_r)
        cand = [all_r(ii), all_c(ii)];
        d2 = sum((centers_rc - cand).^2, 2);
        if isempty(centers_rc) || all(d2 >= opts.init_evt_min_peak_dist^2)
            centers_rc(end+1, :) = cand; %#ok<AGROW>
        end
        if size(centers_rc, 1) >= K
            break;
        end
    end
end

if size(centers_rc, 1) > K
    centers_rc = centers_rc(1:K, :);
end
end

function support = select_seed_connected_support(guide, mask_img, center_rc, target_pixels, max_pixels, opts)
seed_r = center_rc(1);
seed_c = center_rc(2);

guide_vals = guide(mask_img);
guide_vals = guide_vals(guide_vals > 0);
if isempty(guide_vals)
    support = false(nnz(mask_img), 1);
    return;
end

levels = linspace(99, 1, opts.init_evt_threshold_levels);
thresholds = unique([prctile(guide_vals, levels), 0], 'stable');

best_support_img = false(size(mask_img));
best_size = 0;

for ii = 1:numel(thresholds)
    tau = thresholds(ii);
    binary = (guide >= tau) & mask_img;
    if ~binary(seed_r, seed_c)
        continue;
    end

    conn = bwconncomp(binary, 8);
    labels = labelmatrix(conn);
    comp_id = labels(seed_r, seed_c);
    if comp_id == 0
        continue;
    end

    comp_img = (labels == comp_id);
    comp_size = nnz(comp_img);
    if comp_size <= max_pixels && comp_size >= best_size
        best_support_img = comp_img;
        best_size = comp_size;
    end
end

if best_size == 0
    binary = mask_img & (guide > 0);
    if binary(seed_r, seed_c)
        conn = bwconncomp(binary, 8);
        labels = labelmatrix(conn);
        comp_id = labels(seed_r, seed_c);
        if comp_id > 0
            best_support_img = (labels == comp_id);
            best_size = nnz(best_support_img);
        end
    end
end

support = best_support_img(mask_img);
if nnz(support) < target_pixels
    [~, order] = sort(guide_vals, 'descend');
    support = false(nnz(mask_img), 1);
    support(order(1:min(target_pixels, nnz(mask_img)))) = true;
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

Aact = max(0, U * sqrt(S));
C = max(0, sqrt(S) * V');

dead_cols = find(sum(Aact, 1) <= 0 | sum(C, 2)' <= 0);
for k = dead_cols
    Aact(:, k) = max(0, rand(size(Aact, 1), 1));
end

AAt = Aact' * Aact + ridge * eye(K, 'like', Xr);
C = max(0, AAt \ (Aact' * Xr));
end

function opts = set_default_opts(opts)
def.init_evt_sigma = 8;
def.init_evt_min_peak_dist = 200;
def.init_evt_min_frac = 0.10;
def.init_evt_max_frac = 0.30;
def.init_evt_threshold_levels = 40;
def.init_evt_ridge = 1e-6;

f = fieldnames(def);
for i = 1:numel(f)
    if ~isfield(opts, f{i})
        opts.(f{i}) = def.(f{i});
    end
end

opts.init_evt_sigma = max(0, opts.init_evt_sigma);
opts.init_evt_min_peak_dist = max(1, round(opts.init_evt_min_peak_dist));
opts.init_evt_min_frac = min(1, max(0, opts.init_evt_min_frac));
opts.init_evt_max_frac = min(1, max(opts.init_evt_min_frac, opts.init_evt_max_frac));
opts.init_evt_threshold_levels = max(5, round(opts.init_evt_threshold_levels));
opts.init_evt_ridge = max(0, opts.init_evt_ridge);
end