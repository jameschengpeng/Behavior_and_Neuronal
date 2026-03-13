%% 
clear
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));

%%
% preprocessed_storage_path = "F:\Mouse_behavior_data\D21\preprocessed_data"; further_smoothing = false;
preprocessed_storage_path = "F:\CCK_PilotData_Baseline\preprocessed_data"; further_smoothing = true;

stim_side = "L";

if exist(fullfile(preprocessed_storage_path, stim_side, "data_combined.mat"), 'file')==2
    saved_data = load(fullfile(preprocessed_storage_path, stim_side, "data_combined.mat"));
    adj_dFF_all = single(saved_data.adj_dFF_all); % adj_dFF means F/F0, which is equivalent to 1 + dF/F0
    evt_all = single(saved_data.evt_all);
    etho_mat_all = saved_data.etho_mat_all;
    mask_upper_half = saved_data.mask_upper_half;
    mask_lower_half = saved_data.mask_lower_half;
    video_lengths = saved_data.video_lengths;
    clear saved_data
else
    adj_dFF_all = [];
    evt_all = [];
    etho_mat_all = [];
    matFiles = dir(fullfile(preprocessed_storage_path, stim_side, "*.mat"));
    video_lengths = zeros(numel(matFiles), 1);
    for ii = 1:numel(matFiles)
        filename = matFiles(ii).name;
        filepath = fullfile(preprocessed_storage_path, stim_side, filename);
        saved_data = load(filepath);
        datSmo = saved_data.datSmo;
        F0 = saved_data.F0;
        evt_map = saved_data.evt_map;
        condensed_ethogram_mat = saved_data.condensed_ethogram_mat;
        mask_upper_half = saved_data.mask_upper_half;
        mask_lower_half = saved_data.mask_lower_half;
        
        adj_dFF_all = cat(3, adj_dFF_all, datSmo./F0); % F/F0 = 1+dF/F0, so it is adjusted dFF
        evt_all = cat(3, evt_all, evt_map);
        etho_mat_all = cat(1, etho_mat_all, condensed_ethogram_mat);
        video_lengths(ii) = size(datSmo, 3);
        clear saved_data datSmo F0 evt_map condensed_ethogram_mat
        fprintf(strcat('Processed file ', filename, '\n'));
    end
    savefile = fullfile(preprocessed_storage_path, stim_side, "data_combined.mat");
    save(savefile, "adj_dFF_all", "evt_all", "etho_mat_all", "mask_upper_half", "mask_lower_half", "video_lengths", "-v7.3");
end

%% build the map from ordinal indices (i.e., 1, 2, 3...) to original indices of data (i.e., data09, data11...)
base_path = fileparts(preprocessed_storage_path);   % removes "preprocessed_data"
stim_scoring_filepath = fullfile(base_path, 'StimulusScoring.xlsx');
stim_scoring_table = get_stim_metadata(stim_scoring_filepath);
% Filter rows by stim side
rows = stim_scoring_table.StimLocation == stim_side;
subtbl = stim_scoring_table(rows,:);

% Extract numeric suffix from Data column
data_nums = cellfun(@(x) sscanf(x,'data%d'), cellstr(subtbl.Data));

% Sort suffix numbers
[data_nums_sorted, order] = sort(data_nums);

k = numel(data_nums_sorted);
values = 1:k;

% Build containers.Map
data_map = containers.Map(data_nums_sorted, values);
%%
[H, W, T] = size(adj_dFF_all);
if further_smoothing
    adj_dFF_all = imgaussfilt3(adj_dFF_all, [2, 2, 0.01]);
end
X_data = reshape(adj_dFF_all, [], size(adj_dFF_all, 3));
clear adj_dFF_all


%% Quality control to make sure there is no outliers
frame_mean = mean(X_data, 1);
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
plot(mean(X_data, 1))
%% Project the active areas detected by AQuA2 into 2D plane
% for quality control purpose, only select events with medium size
% no need to consider global events, or events that are too minor
evt_domain_projection = zeros(H, W);
conn = bwconncomp(evt_all, 6);
for ii = 1:conn.NumObjects
    indices = conn.PixelIdxList{ii};
    [xx, yy, zz] = ind2sub(size(evt_all), indices);
    xx_select = xx(zz==mode(zz)); % the largest spatial domain of this evt
    yy_select = yy(zz==mode(zz)); % the largest spatial domain of this evt
    ind_2d = sub2ind([H, W], xx_select, yy_select);
    if numel(ind_2d) < 0.8 * H * W && numel(ind_2d) > 0.01 * H * W
        subs = [xx(:) yy(:)];  % N-by-2, rows are (row,col)
        evt_domain_projection = evt_domain_projection + accumarray( ...
            subs, 1, size(evt_domain_projection), @sum, 0);
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
%%
% clear evt_all subs
%%
k_nmf_comp = 3; % number of components in NMF
mask_combined = mask_upper_half + mask_lower_half;
unmasked_indices = find(mask_combined);

%% Constrained NMF
opts = struct();
opts.bg_quiet_prctile = 10;
% =====================
% Iteration / stopping
% =====================
opts.maxIter   = 20;
opts.minIter   = 10;
opts.tol       = 0.03;

% =====================
% Spatial penalties
% =====================
opts.lambdaA_L1   = 30;     % sparcity constraint
opts.lambdaA_lap  = 2e3;     % smooth contiguous regions
opts.lambdaA_excl = 5e2;        % OFF initially (overlap is allowed)
opts.lambdaA_guide = 0;
opts.lambdaA_compact = 1e2;  % encourage compactness (e.g., for groups of neurons with close proximity)

% =====================
% Temporal penalty
% =====================
opts.lambdaC_smooth = 1e-1;   % smooth calcium dynamics
opts.lambdaF_smooth = 5;
opts.use_lowfreq_F_basis = false; % constrain background to low-frequency subspace
opts.F_basis_type = "dct";
opts.F_basis_count = 12; % moderate flexibility for background without letting it absorb fast C transients
opts.lambdaF_q_l2 = 5e-3;
opts.enforce_F_quantile_baseline = true;
opts.F_baseline_anchor_strength = 0.2;

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
opts.bg_rank = 1; % allow background to be of multi-rank
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
opts.printEvery = 10;
opts.F_baseline_prctile = 10; % baseline quantile used by infer_CF_fixed_AB anchoring
%% train on a small portion of videos, to figure out the A matrix
subset_cutting_points = cumsum(video_lengths);
subset_cutting_points = [0; subset_cutting_points];

videos_train = [25 27 33 35 37 39];
% videos_train = [26 28 34 36 38 40];
for ii = 1:length(videos_train)
    videos_train(ii) = data_map(videos_train(ii));
end
videos_test = setdiff(1:length(video_lengths), videos_train);

% now, videos_train and videos_test are the video indices in this set
% starting from 1
time_train = []; time_test = [];
for ii = 1:length(videos_train)
    v_idx = videos_train(ii);
    start_idx = subset_cutting_points(v_idx)+1;
    end_idx = subset_cutting_points(v_idx+1);
    time_train = [time_train start_idx:end_idx];
end
for ii = 1:length(videos_test)
    v_idx = videos_test(ii);
    start_idx = subset_cutting_points(v_idx)+1;
    end_idx = subset_cutting_points(v_idx+1);
    time_test = [time_test start_idx:end_idx];
end

% 
% time_train = cell2mat(arrayfun(@(i) subset_cutting_points(i)+1 : subset_cutting_points(i+1), videos_train, 'UniformOutput', false));
% time_test  = cell2mat(arrayfun(@(i) subset_cutting_points(i)+1 : subset_cutting_points(i+1), videos_test,  'UniformOutput', false));


[A_upper, C_upper_train, info_upper_train] = custom_cnmf(X_data(:, time_train), H, W, k_nmf_comp, mask_upper_half, evt_domain_projection, opts);
%%
[A_lower, C_lower_train, info_lower_train] = custom_cnmf(X_data(:, time_train), H, W, k_nmf_comp, mask_lower_half, evt_domain_projection, opts);

%% show the training outcome
plot_nmf_components_with_ethogram([A_upper A_lower], [C_upper_train; C_lower_train], etho_mat_all(time_train, :), 40, H, W)

%%
A_upper = single(A_upper);
A_lower = single(A_lower);

B_upper = info_upper_train.B;
B_upper = single(B_upper);
opts.F_quantile_ref = prctile(info_upper_train.F, opts.F_baseline_prctile, 2);

[C_upper_test, F_upper_test, info_upper_test] = infer_CF_fixed_AB(X_data(:, time_test), A_upper, B_upper, H, W, mask_upper_half, opts);

B_lower = info_lower_train.B;
B_lower = single(B_lower);
opts.F_quantile_ref = prctile(info_lower_train.F, opts.F_baseline_prctile, 2);
[C_lower_test, F_lower_test, info_lower_test] = infer_CF_fixed_AB(X_data(:, time_test), A_lower, B_lower, H, W, mask_lower_half, opts);

%%
A = [A_upper A_lower];
B = [B_upper B_lower];
C_train = [C_upper_train; C_lower_train];
C_test =  [C_upper_test; C_lower_test];
C = zeros(2*k_nmf_comp, T);
C(:, time_train) = C_train;
C(:, time_test) = C_test;

F_train = [info_upper_train.F; info_lower_train.F];
F_test = [F_upper_test; F_lower_test];
F = zeros(2*opts.bg_rank, T);
F(:, time_train) = F_train;
F(:, time_test) = F_test;

norm(A(unmasked_indices, :) * C + B(unmasked_indices, :) * F - X_data(unmasked_indices, :), 'fro') / norm(X_data(unmasked_indices, :), 'fro')
%% Shift the test baseline to match the train baseline, to mitigate potential batch effects
opts_shift = struct();
opts_shift.baseline_prctile = 10; % use the same percentile as the one used for background quiet percentile in CNMF
opts_shift.shift_strength = 1;     % full shift
opts_shift.nonneg_clip = true;   % ensure nonnegativity after shifting
[shifted_C, info_shift] = shift_postsplit_baseline(C, time_train, time_test, opts_shift);
C = shifted_C;
[shifted_F, info_shift_F] = shift_postsplit_baseline(F, time_train, time_test, opts_shift);
F = shifted_F;

%% show all
plot_nmf_components_with_ethogram(A, C, etho_mat_all, 40, H, W)
%% keep the significant spatial footprints only and check the reconstruction
selected_comps = [1 4 6];
norm(A(unmasked_indices, selected_comps) * C(selected_comps, :) + B(unmasked_indices, :) * F - X_data(unmasked_indices, :), 'fro') ...
    / norm(X_data(unmasked_indices, :), 'fro')

%% save the result
file_save = fullfile(preprocessed_storage_path, stim_side, "NMF_result.mat");
save(file_save, "A", "B", "C", "F", "etho_mat_all", "H", "W");
%% Plot the components, their time course, and the onset of behaviors
function plot_nmf_components_with_ethogram(A, C, ethogram_matrix, fps, H, W)

% Inputs:
% A: (n_pixels x k)
% C: (k x T)
% ethogram_matrix: (T x n_ethograms)
% fps: frame rate
% H, W: spatial dimensions (H*W = n_pixels)

[n_pixels, k] = size(A);
[~, T] = size(C);
[~, n_ethograms] = size(ethogram_matrix);

% Sanity check
if H*W ~= n_pixels
    error('H*W must equal number of rows in A');
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

% ==============================
% Shared Y-axis scaling (traces)
% ==============================
global_ymin = min(C(:));
global_ymax = max(C(:));
y_margin = 0.05 * (global_ymax - global_ymin);
ylim_all = [global_ymin - y_margin, global_ymax + y_margin];

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
    
    plot(t, C(i,:), 'k', 'LineWidth', 1.2);
    ylim([ylim_all(1), ylim_all(2) + 0.06*(ylim_all(2)-ylim_all(1))]);

    xlim([t(1), t(end)]);
    
    % Short tick length (small to avoid spike overlap)
    % Short tick length
    tick_length = 0.04 * (ylim_all(2) - ylim_all(1));
    y_top = ylim_all(2);
    
    % Small vertical offset for label (above tick)
    label_offset = 0.01 * (ylim_all(2) - ylim_all(1));
    
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



