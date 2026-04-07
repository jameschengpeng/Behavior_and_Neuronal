function [cfu_maps, info] = extract_cfu_from_event_tensor(evt_tensor, mask, opts)
%EXTRACT_CFU_FROM_EVENT_TENSOR Extract CFUs from a binary spatiotemporal event tensor.
% Implements the CFU extraction idea from AQuA2 using:
%   1) event-wise weighted spatial maps based on per-pixel event duration,
%   2) weighted Jaccard similarity between event maps,
%   3) single-linkage clustering cut at a similarity threshold.
%
% Inputs
%   evt_tensor : H x W x T binary tensor; nonzero voxels belong to events.
%   mask       : H x W logical/binary mask. Pass [] to use all pixels.
%   opts       : optional struct with fields
%       - connectivity        : bwconncomp connectivity in 3D (default 26)
%       - min_similarity      : minimum weighted Jaccard to connect events (default 0.7)
%       - min_events_per_cfu  : minimum number of events in a CFU (default 2)
%       - min_event_pixels    : minimum spatial pixels in an event (default 1)
%       - max_event_pixels    : maximum spatial pixels in an event (default inf)
%       - min_event_duration  : minimum number of active frames in an event (default 1)
%       - max_event_duration  : maximum number of active frames in an event (default inf)
%       - max_centroid_dist   : maximum centroid distance in pixels for two events
%                               to be linked. Use inf to disable (default inf)
%       - direct_pairwise_max_events : switch to direct all-pairs similarity when
%                                      kept event count is at most this value (default 250)
%       - max_candidate_pairs : soft guardrail for pair generation (default 2e7)
%       - verbose             : print diagnostics (default true)
%
% Outputs
%   cfu_maps : (H*W) x N_cfu sparse matrix. Each column is the average weighted
%              spatial footprint of one CFU over the full field of view.
%   info     : struct with diagnostics and intermediate metadata.

if nargin < 2 || isempty(mask)
    mask = [];
end
if nargin < 3
    if isstruct(mask)
        opts = mask;
        mask = [];
    else
        opts = struct();
    end
end
opts = set_default_opts(opts);

[H, W, T] = size(evt_tensor); % scalars
P = H * W; % scalar

if isempty(mask)
    mask_img = true(H, W); % H x W
else
    mask_img = reshape(mask, H, W) ~= 0; % H x W
end
mask_vec = mask_img(:); % P x 1

if ~any(mask_vec)
    cfu_maps = sparse(P, 0);
    info = struct('n_events_raw', 0, 'n_events_kept', 0, 'n_cfu', 0, 'cfu_event_ids', {{}}, 'event_keep_idx', []);
    return;
end

conn = bwconncomp(evt_tensor, opts.connectivity);
n_events_raw = conn.NumObjects;

% Store each event as a sparse weighted spatial footprint, where the weight
% of a pixel is its within-event duration normalized by that event's peak duration.
event_pix = cell(n_events_raw, 1); % n_events_raw x 1 cell, each entry is n_pix_e x 1
event_w = cell(n_events_raw, 1); % n_events_raw x 1 cell, each entry is n_pix_e x 1
event_area = zeros(n_events_raw, 1); % n_events_raw x 1
event_duration = zeros(n_events_raw, 1); % n_events_raw x 1
event_start = zeros(n_events_raw, 1); % n_events_raw x 1
event_mass = zeros(n_events_raw, 1); % n_events_raw x 1
event_centroid_rc = zeros(n_events_raw, 2); % n_events_raw x 2
keep_event = false(n_events_raw, 1); % n_events_raw x 1

for e = 1:n_events_raw
    vox_idx = conn.PixelIdxList{e}; % n_vox_e x 1
    [rr, cc, tt] = ind2sub([H, W, T], vox_idx); % each is n_vox_e x 1
    pix_idx = sub2ind([H, W], rr, cc); % n_vox_e x 1
    [pix_unique, ~, grp] = unique(pix_idx); % pix_unique: n_pix_e x 1, grp: n_vox_e x 1
    counts = accumarray(grp, 1, [], @sum); % n_pix_e x 1

    in_mask = mask_vec(pix_unique); % n_pix_e x 1
    pix_unique = pix_unique(in_mask); % n_pix_in_mask_e x 1
    counts = counts(in_mask); % n_pix_in_mask_e x 1

    if isempty(pix_unique)
        continue;
    end

    area = numel(pix_unique);
    duration = numel(unique(tt));
    if area < opts.min_event_pixels || area > opts.max_event_pixels || ...
            duration < opts.min_event_duration || duration > opts.max_event_duration
        continue;
    end

    weights = single(counts / max(counts)); % n_pix_in_mask_e x 1
    event_pix{e} = uint32(pix_unique); % n_pix_in_mask_e x 1
    event_w{e} = weights; % n_pix_in_mask_e x 1
    event_area(e) = area;
    event_duration(e) = duration;
    event_start(e) = min(tt);
    event_mass(e) = sum(double(weights));
    event_centroid_rc(e, :) = [mean(double(rr(in_mask(grp)))), mean(double(cc(in_mask(grp))))];
    keep_event(e) = true;
end

event_keep_idx = find(keep_event); % n_events_kept x 1
n_events_kept = numel(event_keep_idx); % scalar

if n_events_kept == 0
    cfu_maps = sparse(P, 0);
    info = struct('n_events_raw', n_events_raw, 'n_events_kept', 0, 'n_cfu', 0, 'cfu_event_ids', {{}}, ...
        'event_keep_idx', event_keep_idx, 'event_area', event_area, 'event_duration', event_duration, 'event_start', event_start);
    return;
end

event_pix = event_pix(event_keep_idx); % n_events_kept x 1 cell
event_w = event_w(event_keep_idx); % n_events_kept x 1 cell
event_area = event_area(event_keep_idx); % n_events_kept x 1
event_duration = event_duration(event_keep_idx); % n_events_kept x 1
event_start = event_start(event_keep_idx); % n_events_kept x 1
event_mass = event_mass(event_keep_idx); % n_events_kept x 1
event_centroid_rc = event_centroid_rc(event_keep_idx, :); % n_events_kept x 2

% For small event counts, direct all-pairs weighted Jaccard is simpler and
% avoids overestimating cost when large events overlap on many pixels.
if n_events_kept <= opts.direct_pairwise_max_events
    similarity_mode = "direct_pairwise";
    total_candidate_pairs = n_events_kept * (n_events_kept - 1) / 2; % scalar
    [edge_i, edge_j, sim] = build_similarity_edges_direct(event_pix, event_w, event_mass, event_centroid_rc, opts); % each n_edges x 1
else
    similarity_mode = "sparse_overlap";

% Build one sparse matrix whose columns are event footprints. This is later
% reused for CFU averaging, so we only materialize it once.
support_sizes = cellfun(@numel, event_pix); % n_events_kept x 1
nnz_total = sum(support_sizes); % scalar
pix_all = zeros(nnz_total, 1, 'uint32'); % nnz_total x 1
evt_all = zeros(nnz_total, 1, 'uint32'); % nnz_total x 1
val_all = zeros(nnz_total, 1, 'single'); % nnz_total x 1

cursor = 1;
for e = 1:n_events_kept
    n = support_sizes(e);
    if n == 0
        continue;
    end
    range = cursor:(cursor + n - 1); % 1 x n
    pix_all(range) = event_pix{e};
    evt_all(range) = uint32(e);
    val_all(range) = event_w{e};
    cursor = cursor + n;
end

if cursor <= nnz_total
    pix_all(cursor:end) = [];
    evt_all(cursor:end) = [];
    val_all(cursor:end) = [];
end

W_sparse = sparse(double(pix_all), double(evt_all), double(val_all), P, n_events_kept); % P x n_events_kept

% Sort by pixel so we only compare events that actually overlap somewhere.
% This avoids an O(N^2) all-pairs similarity computation over events.
[pix_sorted, order] = sort(pix_all); % each nnz_total x 1
evt_sorted = evt_all(order); % nnz_total x 1
val_sorted = val_all(order); % nnz_total x 1

group_start = [1; find(diff(pix_sorted) ~= 0) + 1]; % n_unique_pixels_with_events x 1
group_end = [group_start(2:end) - 1; numel(pix_sorted)]; % n_unique_pixels_with_events x 1
group_sizes = group_end - group_start + 1; % n_unique_pixels_with_events x 1
total_candidate_pairs = sum(double(group_sizes) .* double(group_sizes - 1) / 2); % scalar

if total_candidate_pairs > opts.max_candidate_pairs
    error(['CFU extraction would generate too many candidate event pairs (%g). ' ...
        'Tighten event filters or increase opts.max_candidate_pairs if this is expected.'], total_candidate_pairs);
end

pair_i = zeros(total_candidate_pairs, 1, 'uint32'); % total_candidate_pairs x 1
pair_j = zeros(total_candidate_pairs, 1, 'uint32'); % total_candidate_pairs x 1
pair_v = zeros(total_candidate_pairs, 1, 'single'); % total_candidate_pairs x 1
cursor = 1;

for g = 1:numel(group_start)
    s = group_start(g);
    e = group_end(g);
    m = e - s + 1;
    if m < 2
        continue;
    end

    evt_local = evt_sorted(s:e); % m x 1
    w_local = val_sorted(s:e); % m x 1
    % For one shared pixel, accumulate the weighted-intersection term
    % min(w_i(p), w_j(p)) for every event pair touching that pixel.
    for a = 1:(m - 1)
        ea = evt_local(a);
        wa = w_local(a);
        for b = (a + 1):m
            pair_i(cursor) = ea;
            pair_j(cursor) = evt_local(b);
            pair_v(cursor) = min(wa, w_local(b));
            cursor = cursor + 1;
        end
    end
end

if cursor == 1
    overlap_upper = sparse(n_events_kept, n_events_kept); % n_events_kept x n_events_kept
else
    pair_i = pair_i(1:(cursor - 1)); % n_real_pairs x 1
    pair_j = pair_j(1:(cursor - 1)); % n_real_pairs x 1
    pair_v = pair_v(1:(cursor - 1)); % n_real_pairs x 1
    overlap_upper = sparse(double(pair_i), double(pair_j), double(pair_v), n_events_kept, n_events_kept); % n_events_kept x n_events_kept
end

[edge_i, edge_j, overlap_min] = find(overlap_upper); % each n_overlap_edges x 1
if isempty(overlap_min)
    sim = zeros(0, 1); % 0 x 1
    edge_i = zeros(0, 1); % 0 x 1
    edge_j = zeros(0, 1); % 0 x 1
else
    % Weighted Jaccard: sum(min) / sum(max), with
    % sum(max) = mass_i + mass_j - sum(min).
    denom = event_mass(edge_i) + event_mass(edge_j) - overlap_min; % n_overlap_edges x 1
    sim = overlap_min ./ max(denom, eps); % n_overlap_edges x 1

    % Optional centroid gate to prevent single-linkage chaining through
    % moderately overlapping but spatially distant events.
    if isfinite(opts.max_centroid_dist)
        drow = event_centroid_rc(edge_i, 1) - event_centroid_rc(edge_j, 1); % n_overlap_edges x 1
        dcol = event_centroid_rc(edge_i, 2) - event_centroid_rc(edge_j, 2); % n_overlap_edges x 1
        centroid_dist = hypot(drow, dcol); % n_overlap_edges x 1
        sim(centroid_dist > opts.max_centroid_dist) = 0;
    end
end
end

keep_edge = (sim >= opts.min_similarity); % n_edges x 1

% W_sparse is needed later for CFU averaging. Build it here if we took the
% direct pairwise path above.
if ~exist('W_sparse', 'var')
    W_sparse = build_event_weight_matrix(event_pix, event_w, P, n_events_kept); % P x n_events_kept
end

% Thresholding similarity and taking connected components is equivalent to
% cutting a single-linkage hierarchy at the chosen dissimilarity threshold.
adj_keep = sparse(double(edge_i(keep_edge)), double(edge_j(keep_edge)), double(sim(keep_edge)), n_events_kept, n_events_kept); % n_events_kept x n_events_kept
adj_keep = adj_keep + adj_keep.'; % n_events_kept x n_events_kept
G = graph(adj_keep, 'upper'); % graph with n_events_kept nodes
bins = conncomp(G); % 1 x n_events_kept
comp_sizes = accumarray(bins(:), 1); % n_components x 1
keep_comp = find(comp_sizes >= opts.min_events_per_cfu); % n_kept_components x 1

if isempty(keep_comp)
    cfu_maps = sparse(P, 0);
    info = build_info_struct();
    return;
end

[~, sort_order] = sort(comp_sizes(keep_comp), 'descend');
keep_comp = keep_comp(sort_order);

n_cfu = numel(keep_comp);
cfu_maps = sparse(P, n_cfu); % P x n_cfu
cfu_event_ids = cell(n_cfu, 1); % n_cfu x 1 cell
cfu_occurrence = sparse(n_cfu, T); % n_cfu x T
cfu_sizes = zeros(n_cfu, 1); % n_cfu x 1

for k = 1:n_cfu
    comp_id = keep_comp(k);
    event_ids = find(bins == comp_id); % n_events_in_cfu x 1
    % A CFU footprint is the average weighted event footprint of its members.
    cfu_event_ids{k} = event_keep_idx(event_ids);
    cfu_sizes(k) = numel(event_ids);
    cfu_maps(:, k) = sum(W_sparse(:, event_ids), 2) / numel(event_ids);
    cfu_occurrence(k, event_start(event_ids)) = cfu_occurrence(k, event_start(event_ids)) + 1;
end

info = build_info_struct();

if opts.verbose
    fprintf('CFU extraction: raw events=%d, kept events=%d, candidate pairs=%g, CFUs=%d\n', ...
        n_events_raw, n_events_kept, total_candidate_pairs, n_cfu);
end

    function info = build_info_struct()
        if exist('n_cfu', 'var') ~= 1
            n_cfu_local = 0;
            cfu_event_ids_local = {}; % 0 x 0 cell
            cfu_occurrence_local = sparse(0, T); % 0 x T
            cfu_sizes_local = zeros(0, 1); % 0 x 1
        else
            n_cfu_local = n_cfu;
            cfu_event_ids_local = cfu_event_ids;
            cfu_occurrence_local = cfu_occurrence;
            cfu_sizes_local = cfu_sizes;
        end

        info = struct();
        info.n_events_raw = n_events_raw;
        info.n_events_kept = n_events_kept;
        info.n_cfu = n_cfu_local;
        info.event_keep_idx = event_keep_idx;
        info.event_area = event_area;
        info.event_duration = event_duration;
        info.event_start = event_start;
        info.event_mass = event_mass;
        info.event_centroid_rc = event_centroid_rc;
        info.total_candidate_pairs = total_candidate_pairs;
        info.n_similarity_edges = nnz(keep_edge);
        info.similarity_mode = similarity_mode;
        info.cfu_sizes = cfu_sizes_local;
        info.cfu_event_ids = cfu_event_ids_local;
        info.cfu_occurrence = cfu_occurrence_local;
        info.min_similarity = opts.min_similarity;
        info.max_centroid_dist = opts.max_centroid_dist;
        info.min_events_per_cfu = opts.min_events_per_cfu;
    end
end

function opts = set_default_opts(opts)
def.connectivity = 26;
def.min_similarity = 0.7;
def.min_events_per_cfu = 2;
def.min_event_pixels = 1;
def.max_event_pixels = inf;
def.min_event_duration = 1;
def.max_event_duration = inf;
def.max_centroid_dist = inf;
def.direct_pairwise_max_events = 250;
def.max_candidate_pairs = 2e7;
def.verbose = true;

f = fieldnames(def);
for i = 1:numel(f)
    if ~isfield(opts, f{i})
        opts.(f{i}) = def.(f{i});
    end
end

opts.connectivity = round(opts.connectivity);
opts.min_similarity = min(1, max(0, opts.min_similarity));
opts.min_events_per_cfu = max(1, round(opts.min_events_per_cfu));
opts.min_event_pixels = max(1, round(opts.min_event_pixels));
opts.min_event_duration = max(1, round(opts.min_event_duration));
opts.max_event_pixels = max(opts.min_event_pixels, opts.max_event_pixels);
opts.max_event_duration = max(opts.min_event_duration, opts.max_event_duration);
opts.max_centroid_dist = max(0, opts.max_centroid_dist);
opts.direct_pairwise_max_events = max(1, round(opts.direct_pairwise_max_events));
opts.max_candidate_pairs = max(0, opts.max_candidate_pairs);
opts.verbose = logical(opts.verbose);
end

function W_sparse = build_event_weight_matrix(event_pix, event_w, P, n_events_kept)
support_sizes = cellfun(@numel, event_pix); % n_events_kept x 1
nnz_total = sum(support_sizes); % scalar
pix_all = zeros(nnz_total, 1, 'uint32'); % nnz_total x 1
evt_all = zeros(nnz_total, 1, 'uint32'); % nnz_total x 1
val_all = zeros(nnz_total, 1, 'single'); % nnz_total x 1

cursor = 1;
for e = 1:n_events_kept
    n = support_sizes(e);
    if n == 0
        continue;
    end
    range = cursor:(cursor + n - 1); % 1 x n
    pix_all(range) = event_pix{e};
    evt_all(range) = uint32(e);
    val_all(range) = event_w{e};
    cursor = cursor + n;
end

if cursor <= nnz_total
    pix_all(cursor:end) = [];
    evt_all(cursor:end) = [];
    val_all(cursor:end) = [];
end

W_sparse = sparse(double(pix_all), double(evt_all), double(val_all), P, n_events_kept); % P x n_events_kept
end

function [edge_i, edge_j, sim] = build_similarity_edges_direct(event_pix, event_w, event_mass, event_centroid_rc, opts)
n_events_kept = numel(event_pix); % scalar
n_pairs = n_events_kept * (n_events_kept - 1) / 2; % scalar
edge_i = zeros(n_pairs, 1); % n_pairs x 1
edge_j = zeros(n_pairs, 1); % n_pairs x 1
sim = zeros(n_pairs, 1); % n_pairs x 1

cursor = 1;
for i = 1:(n_events_kept - 1)
    pix_i = double(event_pix{i}); % n_pix_i x 1
    w_i = double(event_w{i}); % n_pix_i x 1
    mass_i = event_mass(i); % scalar
    for j = (i + 1):n_events_kept
        if isfinite(opts.max_centroid_dist)
            drow = event_centroid_rc(i, 1) - event_centroid_rc(j, 1); % scalar
            dcol = event_centroid_rc(i, 2) - event_centroid_rc(j, 2); % scalar
            if hypot(drow, dcol) > opts.max_centroid_dist
                edge_i(cursor) = i;
                edge_j(cursor) = j;
                sim(cursor) = 0;
                cursor = cursor + 1;
                continue;
            end
        end

        pix_j = double(event_pix{j}); % n_pix_j x 1
        w_j = double(event_w{j}); % n_pix_j x 1
        overlap_min = weighted_intersection_min_sum(pix_i, w_i, pix_j, w_j); % scalar
        denom = mass_i + event_mass(j) - overlap_min; % scalar
        if denom > 0
            sim(cursor) = overlap_min / denom;
        else
            sim(cursor) = 0;
        end
        edge_i(cursor) = i;
        edge_j(cursor) = j;
        cursor = cursor + 1;
    end
end
end

function overlap_min = weighted_intersection_min_sum(pix_i, w_i, pix_j, w_j)
% Both pixel index vectors are already sorted because they come from unique().
overlap_min = 0;
i = 1;
j = 1;
ni = numel(pix_i);
nj = numel(pix_j);

while i <= ni && j <= nj
    if pix_i(i) == pix_j(j)
        overlap_min = overlap_min + min(w_i(i), w_j(j));
        i = i + 1;
        j = j + 1;
    elseif pix_i(i) < pix_j(j)
        i = i + 1;
    else
        j = j + 1;
    end
end
end