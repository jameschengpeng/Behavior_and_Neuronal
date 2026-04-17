%% 
clear
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));
%%
mouse_num = 2;
day = 28;
recording_type = "GGP"; % can be GGC or GGP
AQuA2_result_path = strcat("F:\Astrocyte_data\", recording_type, "#", num2str(mouse_num), "_d", num2str(day));
preprocessed_storage_path = fullfile(AQuA2_result_path, "preprocessed_data");

stim_side = "L";

% assume the data has already been downsampled and combined
saved_data = load(fullfile(preprocessed_storage_path, stim_side, "data_combined_downsampled.mat"));
adj_dFF_all = single(saved_data.adj_dFF_all); % adj_dFF means F/F0, which is equivalent to 1 + dF/F0
evt_all = single(saved_data.evt_all);
etho_mat_all = saved_data.etho_mat_all;
mask_upper_half = saved_data.mask_upper_half;
mask_lower_half = saved_data.mask_lower_half;
video_lengths = saved_data.video_lengths;
noise_var_ds = saved_data.noise_var_ds;

clear saved_data

%%
[H, W, T] = size(adj_dFF_all);
X_data = reshape(adj_dFF_all, [], size(adj_dFF_all, 3));
clear adj_dFF_all

%% Quality control to make sure there is no outliers
frame_mean = mean(X_data, 1);
figure
plot(frame_mean)
%% if there are outliers, replace with neighboring frames' mean
med_val = median(frame_mean);
mad_val = mad(frame_mean,1);
threshold = med_val - 3*mad_val;

outliers = frame_mean < threshold;

fprintf('Outlier frames: %s\n', mat2str(find(outliers)));

if any(outliers)
    video_end = cumsum(video_lengths);
    video_start = [1; video_end(1:end-1)+1];
    
    for v = 1:length(video_lengths)
    
        idx = video_start(v):video_end(v);      % frames in this video
        good = ~outliers(idx);                  % good frames within video
    
        % interpolate each pixel across time
        X_data(:,idx) = interp1(idx(good), ...
                                X_data(:,idx(good))', ...
                                idx, ...
                                'linear', ...
                                'extrap')';
    end
end
%% show the result after removing outliers
figure
plot(mean(X_data, 1))
ylim([0.95, 1.1])
%% Project the active areas detected by AQuA2 into 2D plane
% for quality control purpose, only select events with medium size
% no need to consider global events, or events that are too minor
evt_domain_projection = zeros(H, W);
evt_weight_tau = 3; % exponential weighting time constant in frames; smaller values emphasize earlier-recruited pixels more strongly
conn = bwconncomp(evt_all, 6);
for ii = 1:conn.NumObjects
    indices = conn.PixelIdxList{ii};
    [xx, yy, zz] = ind2sub(size(evt_all), indices);
    xx_select = xx(zz==mode(zz)); % the largest spatial domain of this evt
    yy_select = yy(zz==mode(zz)); % the largest spatial domain of this evt
    ind_2d = sub2ind([H, W], xx_select, yy_select);
    if numel(ind_2d) < 0.5 * H * W && numel(ind_2d) > 0.01 * H * W
        pix_idx = sub2ind([H, W], xx, yy);
        [pix_unique, ~, grp] = unique(pix_idx);
        first_frame_per_pixel = accumarray(grp, zz, [], @min);
        dt_first = double(first_frame_per_pixel - min(zz));
        pixel_weights = exp(-dt_first / evt_weight_tau);
        [rr_unique, cc_unique] = ind2sub([H, W], pix_unique);
        subs = [rr_unique cc_unique];  % N-by-2, rows are (row,col), one row per pixel in this event
        evt_domain_projection = evt_domain_projection + accumarray( ...
            subs, pixel_weights, size(evt_domain_projection), @sum, 0);
    end
end
evt_domain_projection_for_plot = evt_domain_projection;
evt_domain_projection_for_plot(evt_domain_projection_for_plot == 0) = NaN;
figure
imagesc(evt_domain_projection_for_plot)
cmap = parula(256);
cmap(1,:) = [0 0 0];  % make first color black
colormap(cmap)

clim([min(evt_domain_projection_for_plot(isfinite(evt_domain_projection_for_plot))) max(evt_domain_projection_for_plot(isfinite(evt_domain_projection_for_plot)))])  % ignore NaNs for scaling
colorbar
%% check the synchrony of signals from different locations in the FOV; The FOV is divided into 6 patches by geographical locations
n_select = 2000; % choose number of pixels to average

% reshape pixel index map
pixel_map = reshape(1:(H*W), H, W);

% define patch boundaries
h_mid = floor(H/2);
w_third = floor(W/3);

patches = {
    pixel_map(1:h_mid,        1:w_third)       % north west
    pixel_map(1:h_mid,        w_third+1:2*w_third) % central north
    pixel_map(1:h_mid,        2*w_third+1:W)   % north east
    pixel_map(h_mid+1:H,      1:w_third)       % south west
    pixel_map(h_mid+1:H,      w_third+1:2*w_third) % central south
    pixel_map(h_mid+1:H,      2*w_third+1:W)   % south east
};

mask_combined = mask_upper_half + mask_lower_half;
mask_vec = mask_combined(:);
unmasked_indices = find(mask_combined);
n_samples = size(X_data,2);
patch_ts = zeros(6,n_samples);

for k = 1:6
    pix = patches{k}(:);
    region_pix = sample_connected_region(pix, mask_vec, H, W, n_select);
    if isempty(region_pix)
        continue
    end
    patch_ts(k,:) = mean(X_data(region_pix,:),1);
end

% Plot
figure
labels = {'North West','Central North','North East',...
          'South West','Central South','South East'};
for k = 1:6
    subplot(6,1,k)
    plot(patch_ts(k,:),'k')
    ylabel('Intensity')
    title(labels{k})    
    if k < 6
        set(gca,'XTickLabel',[])
    else
        xlabel('Time')
    end
end

selected_img = zeros(H,W);
for k = 1:6
    pix = patches{k}(:);
    region_pix = sample_connected_region(pix, mask_vec, H, W, n_select);
    selected_img(region_pix) = k;
end

figure
imagesc(selected_img)
axis image
colorbar
title('Connected sampled regions')

corr_matrix = corr(patch_ts');   % transpose so variables = columns
labels = {'NW','CN','NE','SW','CS','SE'};

figure
imagesc(corr_matrix)
axis square
colorbar

set(gca,'XTick',1:6,'XTickLabel',labels)
set(gca,'YTick',1:6,'YTickLabel',labels)

title('Correlation between patch signals')
clim([min(corr_matrix(:))-0.03, 1])
colormap(parula)

%%
mask_all = mask_upper_half + mask_lower_half;
[H, W] = size(mask_all);

% Per-pixel noise variance (precomputed from full-resolution data, then
% divided by temp_down_factor to account for temporal averaging).
noise_map = noise_var_ds;                          % H x W
noise_map(mask_all == 0) = NaN;

% visualize
figure;

imagesc(noise_map);
axis image;
colorbar;
title('Estimated Noise Variance Map (from full-res, scaled to downsampled)');

% Set NaNs (masked regions) to black
colormap("hot");          % or any colormap you like
set(gca, 'Color', 'k');    % background = black (for NaNs)

%% for constrained NMF
k_nmf_comp = 3; % number of components in NMF
opts = struct();

% =====================
% Iteration / stopping
% =====================
opts.maxIter   = 15;
opts.minIter   = 5;
opts.tol       = 0.01;
% =====================
% Spatial penalties
% =====================
% opts.lambdaA_L1   = 80;     % sparcity constraint for uninjured
opts.lambdaA_L1 = 30;
opts.lambdaA_lap  = 2e3;     % smooth contiguous regions
opts.lambdaA_excl = 5e2;        % OFF initially (overlap is allowed)
opts.lambdaA_guide = 0;
opts.lambdaA_compact = 1e2;  % encourage compactness (e.g., for groups of neurons with close proximity)
% =====================
% Temporal penalty
% =====================
opts.lambdaC_smooth = 1e-1;   % smooth calcium dynamics
opts.lambdaF_smooth = 5;
opts.enforce_F_quantile_baseline = false;
opts.F_baseline_anchor_strength = 0.2;
opts.F_baseline_prctile = 10; % baseline quantile used by infer_CF_fixed_AB anchoring
% =====================
% Step sizes / solver
% =====================
opts.use_adaptive_steps = true;
opts.innerA = 1;
opts.innerC = 1;
% =====================
% Normalization
% =====================
opts.doNormalize     = true;
opts.normalizeEvery = 10;     % NOT every iteration (important)
opts.normalize_mode = "l2"; % robust max-like normalization for A columns, change to l2 if you want L2-normalize
opts.normalize_prctile = 99;
% =====================
% Background modeling
% =====================
opts.use_background = true;   % CRITICAL for clean A maps
opts.bg_rank = 1; % trial: one global mode
opts.bg_init_mode = "lowrank"; % use uniform rank-1 global background
opts.update_background = false; % freeze B after initialization
opts.update_F = false; % freeze F after initialization
opts.enforce_background_nonovershoot = true;
opts.bg_floor_quantile = 0.02;
opts.bg_refine_profile_from_F = true;
opts.bg_profile_quantile = 0.15;
opts.bg_profile_min_relF = 0.4;
opts.bg_profile_min_frames = 150;
opts.bg_profile_smooth_sigma = 8;
opts.bg_profile_shrink_uniform = 0.05;
opts.bg_profile_n_alternations = 4;
opts.bg_noise_var = noise_var_ds; % use the precomputed pixel-wise noise variance map
opts.temporally_downsampled = true; % data was binned 10x; diff-based noise estimation invalid
% =====================
% Nonnegativity handling
% =====================
opts.nonneg_mode = "none";    % do NOT shift again inside CNMF
% =====================
% Laplacian neighborhood
% =====================
opts.neighborhood = 4;        % 4-neighbor is safer for thin processes
% =====================
% Safeguards
% =====================
opts.backtracking = true;
opts.maxBacktrack = 15;
% =====================
% Misc
% =====================
opts.seed        = 0;
opts.verbose     = true;
opts.printEvery = 2;

% =====================
% Related to sparsity
% =====================
opts.adapt_lambdaA_L1 = true;
opts.target_A_nnz_frac = 0.04;
opts.target_A_nnz_tol = 0.01;
opts.lambdaA_L1_adapt_rate = 8;
opts.lambdaA_L1_min = 20;
% opts.lambdaA_L1_min = 50; % for uninjured
opts.lambdaA_L1_max = 130;
opts.stop_if_A_all_zero = true;
opts.require_target_A_nnz_for_stop = true;
opts.rollback_on_A_nnz_undershoot = true;

%%
[A_upper, C_upper, info_upper] = custom_cnmf(X_data, H, W, k_nmf_comp, mask_upper_half, evt_domain_projection, opts);
[A_lower, C_lower, info_lower] = custom_cnmf(X_data, H, W, k_nmf_comp, mask_lower_half, evt_domain_projection, opts);
%%
A = [A_upper A_lower];
B = [info_upper.B info_lower.B];
C = [C_upper; C_lower];
plot_nmf_components_with_ethogram(A, C, etho_mat_all, 26.5/10, H, W);

%% Plot the components, their time course, and the onset of behaviors
function plot_nmf_components_with_ethogram(A, C, ethogram_matrix, fps, H, W, time_train, time_test)

% Inputs:
% A: (n_pixels x k)
% C: (k x T)
% ethogram_matrix: (T x n_ethograms)
% fps: frame rate
% H, W: spatial dimensions (H*W = n_pixels)
% time_train: indices of training frames within columns of C
% time_test: indices of testing frames within columns of C

[n_pixels, k] = size(A);
[~, T] = size(C);
[~, n_ethograms] = size(ethogram_matrix);

% if no time_train and time_test, assume all are train
if nargin < 7 || isempty(time_train)
    time_train = 1:T;
end
if nargin < 8 || isempty(time_test)
    time_test = [];
end

% Sanity check
if H*W ~= n_pixels
    error('H*W must equal number of rows in A');
end
if any(time_train < 1) || any(time_train > T) || any(mod(time_train,1) ~= 0)
    error('time_train must contain valid column indices into C.');
end
if any(time_test < 1) || any(time_test > T) || any(mod(time_test,1) ~= 0)
    error('time_test must contain valid column indices into C.');
end

% Time axis (seconds)
t = (0:T-1) / fps;

% ==============================
% Detect behavior onsets (0 -> 1)
% ==============================
behavior_onsets = cell(n_ethograms,1);
for j = 1:n_ethograms
    onsets = find(diff([0; ethogram_matrix(:,j)]) == 1);
    behavior_onsets{j} = onsets;
end

% ==================================
% Shared color scaling (spatial maps)
% ==================================
low = prctile(A(:),1);
high = prctile(A(:),99);


% Behavior colors
behavior_colors = lines(n_ethograms);

% Create figure
figure;
% adjust this to change the gap between spatial map and time trace
n_tiles = 6;
tl = tiledlayout(k,n_tiles,'TileSpacing','compact','Padding','compact');

for i = 1:k
    
    % -------- Spatial Map --------
    nexttile((i-1)*n_tiles + 1, [1 1]);
    
    spatial_map = reshape(A(:,i), H, W);
    imagesc(spatial_map);
    colormap(parula);
    clim([low high]);   % Shared color scale
    axis image off;
    threshold = 0.05 * max(A(:,i));
    percentage = sum(A(:,i) > threshold) / (H*W);
    title(sprintf('Component %d (%.1f%%)', i, percentage*100));
    colorbar;
    
    % -------- Time Course --------
    nexttile((i-1)*n_tiles + 2, [1 n_tiles-1]);
    hold on;

    if ~isempty(time_train)
        c_train = nan(1, T);
        c_train(time_train) = C(i, time_train);
        plot(t, c_train, 'k', 'LineWidth', 1.2);
    end
    if ~isempty(time_test)
        c_test = nan(1, T);
        c_test(time_test) = C(i, time_test);
        plot(t, c_test, 'b', 'LineWidth', 1.2);
    end

    comp_vals = C(i, :);
    comp_ymin = min(comp_vals);
    comp_ymax = max(comp_vals);
    comp_range = comp_ymax - comp_ymin;
    if comp_range <= 0
        comp_range = max(1e-6, abs(comp_ymax));
    end
    comp_margin = 0.05 * comp_range;
    ylim_i = [comp_ymin - comp_margin, comp_ymax + 0.06 * comp_range + comp_margin];
    ylim(ylim_i);

    xlim([t(1), t(end)]);
    
    % Short tick length (small to avoid spike overlap)
    % Short tick length
    tick_length = 0.04 * comp_range;
    y_top = comp_ymax + comp_margin;
    
    % Small vertical offset for label (above tick)
    label_offset = 0.01 * comp_range;
    
    for j = 1:n_ethograms
        onset_frames = behavior_onsets{j};
        onset_times = (onset_frames - 1) / fps;
        
        for m = 1:length(onset_times)
            
            % Draw tick
            line([onset_times(m) onset_times(m)], ...
                 [y_top - tick_length, y_top], ...
                 'Color', behavior_colors(j,:), ...
                 'LineWidth', 1);
            
            % Add behavior number label
            text(onset_times(m), ...
                 y_top + label_offset, ...
                 num2str(j), ...
                 'Color', behavior_colors(j,:), ...
                 'HorizontalAlignment','center', ...
                 'VerticalAlignment','bottom', ...
                 'FontSize',11, ...
                 'FontWeight','bold', ...
                 'Clipping','off');
        end
    end

    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Activity', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Add legend only once
    if i == 1
        legend_strings = arrayfun(@(x) ...
            ['Behavior ', num2str(x)], ...
            1:n_ethograms, ...
            'UniformOutput', false);
        legend(legend_strings, 'Location','northeastoutside');
    end
    
    hold off;
end

end

function region_pixels = sample_connected_region(patch_idx, mask_vec, H, W, n_select)

patch_idx = patch_idx(:);

% keep only unmasked pixels
patch_idx = patch_idx(mask_vec(patch_idx) > 0);

if isempty(patch_idx)
    region_pixels = [];
    return
end

% choose random seed
seed = patch_idx(randi(length(patch_idx)));

visited = false(H*W,1);
queue = seed;
region_pixels = [];

while ~isempty(queue) && length(region_pixels) < n_select
    
    p = queue(1);
    queue(1) = [];
    
    if visited(p)
        continue
    end
    
    visited(p) = true;
    
    if mask_vec(p) && ismember(p, patch_idx)
        region_pixels(end+1) = p;
        
        % convert to row/col
        [r,c] = ind2sub([H W],p);
        
        % 8-connected neighbors
        neighbors = [
            r-1 c-1
            r-1 c
            r-1 c+1
            r   c-1
            r   c+1
            r+1 c-1
            r+1 c
            r+1 c+1
        ];
        
        for i = 1:size(neighbors,1)
            rr = neighbors(i,1);
            cc = neighbors(i,2);
            
            if rr>=1 && rr<=H && cc>=1 && cc<=W
                idx = sub2ind([H W],rr,cc);
                if ~visited(idx)
                    queue(end+1) = idx;
                end
            end
        end
    end
end

end