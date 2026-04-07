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

if opts.AC_init_mode == "svd"
    [Aact, C] = svd_fallback_init(Xr, K, opts.init_evt_ridge);
    info = struct('mode', 'svd', 'centers_rc', []);
    return;
elseif opts.AC_init_mode ~= "event_projection"
    error('opts.AC_init_mode must be "event_projection" or "svd".');
end

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

K_init = size(centers_rc, 1);
if K_init == 0
    [Aact, C] = svd_fallback_init(Xr, K, opts.init_evt_ridge);
    info = struct('mode', 'svd_fallback', 'centers_rc', centers_rc);
    return;
end

guide_act = cast(guide(idx), 'like', Xr);
Aact = zeros(Pm, K, 'like', Xr);
target_pixels = ceil(opts.init_evt_min_frac * Pm);
max_pixels = max(target_pixels, floor(opts.init_evt_max_frac * Pm));

for k = 1:K_init
    support = select_seed_connected_support(guide, mask_img, centers_rc(k, :), target_pixels, max_pixels, opts);
    ak = guide_act;
    ak(~support) = 0;

    if max(ak) <= 0
        ak = cast(support, 'like', Xr);
    end

    Aact(:, k) = ak;
end

[Aact, merged_cols] = prune_overlapping_init_columns(Aact, opts);

dead_cols = find(sum(Aact, 1) <= 0);
AAt = Aact' * Aact + opts.init_evt_ridge * eye(K, 'like', Xr);
C = AAt \ (Aact' * Xr);
C = max(C, 0);

% Keep columns with no viable event seed/support at zero instead of forcing
% duplicate or random components.
zero_cols = find(sum(Aact, 1) <= 0 | sum(C, 2)' <= 0);
if ~isempty(zero_cols)
    Aact(:, zero_cols) = 0;
    C(zero_cols, :) = 0;
end

info = struct('mode', 'event_projection', 'centers_rc', centers_rc, 'guide_max', guide_max, 'n_init', K_init, 'zero_cols', zero_cols, 'merged_cols', merged_cols);
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
    grown_support_img = grow_connected_support(mask_img, center_rc, target_pixels);
    support = grown_support_img(mask_img);
end
end

function support_img = grow_connected_support(mask_img, center_rc, target_pixels)
% Grow a seed-centered connected support within the active mask using
% breadth-first expansion, so fallback footprints stay localized.
[H, W] = size(mask_img);
support_img = false(H, W);

seed_r = center_rc(1);
seed_c = center_rc(2);
if ~mask_img(seed_r, seed_c)
    return;
end

visited = false(H, W);
queue = zeros(nnz(mask_img), 2);
head = 1;
tail = 1;
queue(tail, :) = [seed_r, seed_c];
visited(seed_r, seed_c) = true;

neighbors = [
    -1 -1
    -1  0
    -1  1
     0 -1
     0  1
     1 -1
     1  0
     1  1
];

count = 0;
while head <= tail && count < target_pixels
    r = queue(head, 1);
    c = queue(head, 2);
    head = head + 1;

    support_img(r, c) = true;
    count = count + 1;

    for n = 1:size(neighbors, 1)
        rr = r + neighbors(n, 1);
        cc = c + neighbors(n, 2);
        if rr < 1 || rr > H || cc < 1 || cc > W
            continue;
        end
        if visited(rr, cc) || ~mask_img(rr, cc)
            continue;
        end
        tail = tail + 1;
        queue(tail, :) = [rr, cc];
        visited(rr, cc) = true;
    end
end
end

function [Aact, merged_cols] = prune_overlapping_init_columns(Aact, opts)
% Resolve highly overlapping initialized columns.
% Near-duplicates of similar size are merged, but if one support is mostly
% contained inside a larger one, keep the smaller support and subtract it
% from the larger one so the pair becomes core + residual.
K = size(Aact, 2);
merged_cols = zeros(0, 2);

support = Aact > 0;
col_mass = sum(Aact, 1);
support_size = sum(support, 1);

for i = 1:K
    if col_mass(i) <= 0
        continue;
    end
    for j = i+1:K
        if col_mass(j) <= 0
            continue;
        end

        union_ij = nnz(support(:, i) | support(:, j));
        if union_ij == 0
            continue;
        end

        inter_ij = nnz(support(:, i) & support(:, j));
        overlap_iou = inter_ij / union_ij;
        contain_i_in_j = inter_ij / max(1, support_size(i));
        contain_j_in_i = inter_ij / max(1, support_size(j));
        overlap_containment = max(contain_i_in_j, contain_j_in_i);

        % IoU catches near-duplicate supports of similar size.
        % Containment catches the case where one support is mostly nested
        % inside a much larger one, which IoU alone can miss.
        if max(overlap_iou, overlap_containment) >= opts.init_evt_merge_overlap
            similar_size = min(support_size(i), support_size(j)) / max(support_size(i), support_size(j)) >= 0.75;

            if overlap_containment >= opts.init_evt_merge_overlap && ~similar_size
                if support_size(i) <= support_size(j)
                    small_idx = i;
                    large_idx = j;
                else
                    small_idx = j;
                    large_idx = i;
                end

                overlap_mask = support(:, small_idx) & support(:, large_idx);
                Aact(overlap_mask, large_idx) = 0;

                support(:, large_idx) = Aact(:, large_idx) > 0;
                support_size(large_idx) = nnz(support(:, large_idx));
                col_mass(large_idx) = sum(Aact(:, large_idx));

                if support_size(large_idx) == 0 || col_mass(large_idx) <= 0
                    Aact(:, large_idx) = 0;
                    support(:, large_idx) = false;
                    support_size(large_idx) = 0;
                    col_mass(large_idx) = 0;
                end

                merged_cols(end+1, :) = [small_idx, large_idx]; %#ok<AGROW>
            else
                if col_mass(i) >= col_mass(j)
                    Aact(:, j) = 0;
                    col_mass(j) = 0;
                    support(:, j) = false;
                    support_size(j) = 0;
                    merged_cols(end+1, :) = [i, j]; %#ok<AGROW>
                else
                    Aact(:, i) = 0;
                    col_mass(i) = 0;
                    support(:, i) = false;
                    support_size(i) = 0;
                    merged_cols(end+1, :) = [j, i]; %#ok<AGROW>
                    break;
                end
            end
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
def.init_evt_min_frac = 0.10;
def.init_evt_max_frac = 0.20;
def.init_evt_threshold_levels = 40;
def.init_evt_merge_overlap = 0.70;
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
opts.init_evt_min_frac = min(1, max(0, opts.init_evt_min_frac));
opts.init_evt_max_frac = min(1, max(opts.init_evt_min_frac, opts.init_evt_max_frac));
opts.init_evt_threshold_levels = max(5, round(opts.init_evt_threshold_levels));
opts.init_evt_merge_overlap = min(1, max(0, opts.init_evt_merge_overlap));
opts.init_evt_ridge = max(0, opts.init_evt_ridge);
end