%% Explore the pixel-wise signal variance map, grouped by stim type
clear
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));

%%
% For injured data
preprocessed_storage_path = "F:\Mouse_behavior_data\D21\preprocessed_data"; further_smoothing = false;
stim_scoring_filepath = "F:\Mouse_behavior_data\D21\StimulusScoring.xlsx";

% % For uninjured data
% preprocessed_storage_path = "F:\CCK_PilotData_Baseline\preprocessed_data"; further_smoothing = true;
% stim_scoring_filepath = "F:\CCK_PilotData_Baseline\StimulusScoring.xlsx";

stim_scoring_table = get_stim_metadata(stim_scoring_filepath);

stim_side = "R";

combined_savefile = fullfile(preprocessed_storage_path, stim_side, "data_combined.mat");

if exist(combined_savefile, 'file')==2
    combined_vars = who('-file', combined_savefile);
    load_vars = {'adj_dFF_all', 'evt_all', 'etho_mat_all', 'mask_upper_half', 'mask_lower_half', 'video_lengths'};
    if any(strcmp(combined_vars, 'noise_model'))
        load_vars{end + 1} = 'noise_model';
    end

    saved_data = load(combined_savefile, load_vars{:});
    adj_dFF_all = single(saved_data.adj_dFF_all); % adj_dFF means F/F0, which is equivalent to 1 + dF/F0
    evt_all = single(saved_data.evt_all);
    etho_mat_all = saved_data.etho_mat_all;
    mask_upper_half = saved_data.mask_upper_half;
    mask_lower_half = saved_data.mask_lower_half;
    video_lengths = saved_data.video_lengths;
    if isfield(saved_data, 'noise_model')
        noise_model = saved_data.noise_model;
    else
        X_noise = reshape(adj_dFF_all, [], size(adj_dFF_all, 3));
        mask_all_noise = logical(mask_upper_half + mask_lower_half);
        noise_model = estimate_noise_model_by_video(X_noise, mask_all_noise, video_lengths);
        save(combined_savefile, 'noise_model', '-append');
        clear X_noise mask_all_noise
    end
    clear saved_data
else
    adj_dFF_all = [];
    evt_all = [];
    etho_mat_all = [];
    noise_model_by_file = {};
    matFiles = dir(fullfile(preprocessed_storage_path, stim_side, "data??.mat"));
    video_lengths = zeros(numel(matFiles), 1);
    for ii = 1:numel(matFiles)
        filename = matFiles(ii).name;
        filepath = fullfile(preprocessed_storage_path, stim_side, filename);
        load_vars = {'datSmo', 'F0', 'evt_map', 'condensed_ethogram_mat', 'mask_upper_half', 'mask_lower_half'};

        saved_data = load(filepath, load_vars{:});
        datSmo = saved_data.datSmo;
        F0 = saved_data.F0;
        evt_map = saved_data.evt_map;
        condensed_ethogram_mat = saved_data.condensed_ethogram_mat;
        mask_upper_half = saved_data.mask_upper_half;
        mask_lower_half = saved_data.mask_lower_half;

        adj_dFF_this = single(datSmo ./ F0);
        mask_all_this = logical(mask_upper_half + mask_lower_half);
        X_this = reshape(adj_dFF_this, [], size(adj_dFF_this, 3));
        noise_model_this = estimate_noise_model_by_video(X_this, mask_all_this, size(adj_dFF_this, 3));
        clear X_this mask_all_this
        
        adj_dFF_all = cat(3, adj_dFF_all, adj_dFF_this); % F/F0 = 1+dF/F0, so it is adjusted dFF
        evt_all = cat(3, evt_all, evt_map);
        etho_mat_all = cat(1, etho_mat_all, condensed_ethogram_mat);
        noise_model_by_file{ii, 1} = noise_model_this;
        video_lengths(ii) = size(datSmo, 3);
        clear saved_data datSmo F0 evt_map condensed_ethogram_mat adj_dFF_this noise_model_this
        fprintf(strcat('Processed file ', filename, '\n'));
    end
    noise_model = concat_noise_models_by_video(noise_model_by_file);
    save(combined_savefile, "adj_dFF_all", "evt_all", "etho_mat_all", "mask_upper_half", "mask_lower_half", "video_lengths", "noise_model", "-v7.3");
    clear noise_model_by_file
end

noise_var_xt = restore_noise_var_xt_from_model(noise_model);


%% video time indices, stimuli onset and offset time indices, grouped by stim_type
stim_by_type_info = stim_times_by_side_and_type(stim_scoring_table, stim_side, video_lengths);
%%
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
mask_all = mask_upper_half + mask_lower_half;
[H, W, T] = size(adj_dFF_all);
if further_smoothing
    adj_dFF_all = imgaussfilt3(adj_dFF_all, [2, 2, 0.01]);
end
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
% also keep only events whose onset is near any behavior onset
evt_domain_projection = zeros(H, W);
evt_weight_tau = 3; % exponential weighting time constant in frames; smaller values emphasize earlier-recruited pixels more strongly
frames_before = 80;
frames_after = 80;

behavior_onset_frames = false(T, 1);
for jj = 1:size(etho_mat_all, 2)
    onset_idx = find(diff([0; etho_mat_all(:, jj)]) == 1);
    for kk = 1:numel(onset_idx)
        t_start = max(1, onset_idx(kk) - frames_before);
        t_end = min(T, onset_idx(kk) + frames_after);
        behavior_onset_frames(t_start:t_end) = true;
    end
end

conn = bwconncomp(evt_all, 6);
for ii = 1:conn.NumObjects
    indices = conn.PixelIdxList{ii};
    [xx, yy, zz] = ind2sub(size(evt_all), indices);
    evt_onset = min(zz);
    xx_select = xx(zz==mode(zz)); % the largest spatial domain of this evt
    yy_select = yy(zz==mode(zz)); % the largest spatial domain of this evt
    ind_2d = sub2ind([H, W], xx_select, yy_select);
    if behavior_onset_frames(evt_onset) && numel(ind_2d) < 0.5 * H * W && numel(ind_2d) > 0.01 * H * W
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
%%
clear evt_all subs grp pix_idx
%%
idx_upper = find(mask_upper_half(:));
idx_lower = find(mask_lower_half(:));
idx_all = find(mask_all(:));

%% Simpler assumption: pixel noise has constant variance; Just have an understanding of noise variance, not exactly true
% Robust per-pixel noise variance via temporal differencing (MAD-based).
noise_var = estimate_noise_var_per_pixel(X_data);   % (pixels x 1)

% Reshape to image and apply mask (set masked pixels to NaN)
noise_map = reshape(noise_var, H, W);
noise_map(mask_all == 0) = NaN;

figure;

imagesc(noise_map);
axis image;

cb = colorbar;
cb.FontSize = 14;   % larger colorbar tick-label font
cb.FontWeight = 'bold';

title('Estimated Noise Variance Map');

% Set NaNs (masked regions) to black
colormap("parula");
set(gca, 'Color', 'k');
clim([quantile(noise_map(idx_all), 0.0), quantile(noise_map(idx_all), 0.999)])

%% Get the guide_map_2d for each stimulus type, assuming constant noise variance for each pixel, just for demonstration
opts = struct();
opts.bg_noise_var = noise_var;
% for brush stimuli
time_brush = cell2mat(stim_by_type_info.B.video_indices);
guide_map_2d_brush = compute_guide_map_2d(X_data(:, time_brush), H, W, mask_all, opts);

% for von Frey 5 stimuli
time_vF5 = cell2mat(stim_by_type_info.x5.video_indices);
guide_map_2d_vF5 = compute_guide_map_2d(X_data(:, time_vF5), H, W, mask_all, opts);

% for von Frey 8 stimuli
time_vF8 = cell2mat(stim_by_type_info.x8.video_indices);
guide_map_2d_vF8 = compute_guide_map_2d(X_data(:, time_vF8), H, W, mask_all, opts);

% for pin prick stimuli
time_pp = cell2mat(stim_by_type_info.P.video_indices);
guide_map_2d_pp = compute_guide_map_2d(X_data(:, time_pp), H, W, mask_all, opts);
%% Visualize the guide_map_2d for each stimulus type
guide_vals_all = [guide_map_2d_brush(idx_all); ...
                  guide_map_2d_vF5(idx_all); ...
                  guide_map_2d_vF8(idx_all); ...
                  guide_map_2d_pp(idx_all)];
guide_vals_all = guide_vals_all(isfinite(guide_vals_all));
guide_clim = quantile(guide_vals_all, [0.01, 0.99]);

figure('Color', 'w');

ax1 = subplot(2,2,1);
imagesc(guide_map_2d_brush)
clim(ax1, guide_clim)
axis(ax1, 'image')
title("Active regions by pixel signal variance (Brush stim only)", 'FontSize', 14, 'FontWeight','bold')

ax2 = subplot(2,2,2);
imagesc(guide_map_2d_vF5)
clim(ax2, guide_clim)
axis(ax2, 'image')
title("Active regions by pixel signal variance (von Frey 5 stim only)", 'FontSize', 14, 'FontWeight','bold')

ax3 = subplot(2,2,3);
imagesc(guide_map_2d_vF8)
clim(ax3, guide_clim)
axis(ax3, 'image')
title("Active regions by pixel signal variance (von Frey 8 stim only)", 'FontSize', 14, 'FontWeight','bold')

ax4 = subplot(2,2,4);
imagesc(guide_map_2d_pp)
clim(ax4, guide_clim)
axis(ax4, 'image')
title("Active regions by pixel signal variance (pin prick stim only)", 'FontSize', 14, 'FontWeight','bold')

cb = colorbar(ax4, 'eastoutside');
cb.FontSize = 14;
cb.FontWeight = 'bold';
cb.Position = [0.92 0.11 0.02 0.815];

%% Another exploration: does each pixel has heteroscedastic noise? i.e., noise variance is not constant given pixel p
% time_selected = [time_brush(1:2000); time_vF8(1:2000); time_vF5(1:2000); time_pp(1:2000)];
% X = X_data(:, time_selected);
X = X_data;
clear X_data
results = test_temporal_heteroscedasticity_fast(X, mask_all, 0.05, 2, video_lengths);

%%
mask2d = logical(results.mask);

rho_map       = results.rho_map;
intercept_map = results.intercept_map;
slope_map     = results.slope_map;
q_map         = results.qval_map;
sig_map       = double(results.sig_fdr_map);

[n_pixels, T] = size(X);
[H, W] = size(mask2d);

if H * W ~= n_pixels
    error('mask size mismatch: H*W must equal size(X,1).');
end

% Use only unmasked pixels
idx = find(mask2d(:));
X_use = X(idx, :);

% Rebuild the same M and D^2 used in the test
X1 = X_use(:, 1:end-1);
X2 = X_use(:, 2:end);
D  = X2 - X1;
Y  = D .^ 2;

% If results came from Gaussian-smoothed version, use the same smoothing
if isfield(results, 'sigma_frames')
    sigma_frames = results.sigma_frames;
else
    sigma_frames = 3; % change if needed
end

halfWidth = max(1, ceil(3 * sigma_frames));
tt = -halfWidth:halfWidth;
g = exp(-(tt .^ 2) / (2 * sigma_frames ^ 2));
g = g / sum(g);

Xs = conv2(X_use, g, 'same');
M = 0.5 * (Xs(:, 1:end-1) + Xs(:, 2:end));

valid = isfinite(M) & isfinite(Y);

% Per-pixel central M range: M95 - M5
M_p5  = prctile(M, 5, 2);
M_p95 = prctile(M, 95, 2);
deltaM = M_p95 - M_p5;

% Mean D^2
meanY = sum(Y .* valid, 2) ./ max(sum(valid, 2), 1);

% Practical effect sizes under linear regression
deltaVar = slope_map(mask2d) .* deltaM;         % fitted change in D^2 across central M range
relEffect = deltaVar ./ max(meanY, eps);        % normalize by mean D^2

% Put back to full maps
deltaVar_map = nan(H * W, 1);
relEffect_map = nan(H * W, 1);
meanY_map = nan(H * W, 1);
deltaM_map = nan(H * W, 1);

deltaVar_map(idx) = deltaVar;
relEffect_map(idx) = relEffect;
meanY_map(idx) = meanY;
deltaM_map(idx) = deltaM;

deltaVar_map = reshape(deltaVar_map, H, W);
relEffect_map = reshape(relEffect_map, H, W);
meanY_map = reshape(meanY_map, H, W);
deltaM_map = reshape(deltaM_map, H, W);

% Masked regions black
rho_map(~mask2d)         = NaN;
intercept_map(~mask2d)   = NaN;
slope_map(~mask2d)       = NaN;
q_map(~mask2d)           = NaN;
sig_map(~mask2d)         = NaN;
deltaVar_map(~mask2d)    = NaN;
relEffect_map(~mask2d)   = NaN;
meanY_map(~mask2d)       = NaN;
deltaM_map(~mask2d)      = NaN;

% Valid values for display scaling
deltaVar_valid = deltaVar_map(mask2d & isfinite(deltaVar_map));
relEffect_valid = relEffect_map(mask2d & isfinite(relEffect_map));
q_valid = q_map(mask2d & isfinite(q_map) & q_map > 0);

if isempty(deltaVar_valid)
    error('No valid deltaVar values found.');
end
if isempty(relEffect_valid)
    error('No valid relEffect values found.');
end

% Robust scales
dv_lo = prctile(deltaVar_valid, 1);
dv_hi = prctile(deltaVar_valid, 99);
deltaVar_lim = max(abs([dv_lo, dv_hi]));
if deltaVar_lim == 0
    deltaVar_lim = 1e-6;
end

re_lo = prctile(relEffect_valid, 1);
re_hi = prctile(relEffect_valid, 99);
relEffect_lim = max(abs([re_lo, re_hi]));
relEffect_lim = max(relEffect_lim, 0.05);

logq_map = nan(size(q_map));
valid_q = isfinite(q_map) & (q_map > 0);
logq_map(valid_q) = -log10(q_map(valid_q));

if isempty(q_valid)
    logq_lim = 2;
else
    logq_valid = -log10(q_valid);
    logq_lim = prctile(logq_valid, 99);
    logq_lim = max(logq_lim, -log10(0.05));
end

% Pick one representative pixel to see whether the linear fit is sensible.
intercept_vec = intercept_map(mask2d);
slope_vec = slope_map(mask2d);
sig_vec = logical(sig_map(mask2d));
if any(sig_vec)
    q_tmp = q_map(mask2d);
    q_tmp(~sig_vec) = inf;
    [~, sample_row] = min(q_tmp);
else
    [~, sample_row] = max(abs(slope_vec));
end

sample_valid = valid(sample_row, :);
sample_M = M(sample_row, sample_valid);
sample_Y = Y(sample_row, sample_valid);
[sample_M_sorted, order] = sort(sample_M);
sample_fit_sorted = intercept_vec(sample_row) + slope_vec(sample_row) .* sample_M_sorted;
sample_Y_sorted = sample_Y(order);

%% Figure
figure('Color', 'w', 'Position', [100 100 1600 900]);

% 1) deltaVar map
ax1 = subplot(2,3,1);
set(ax1, 'Color', 'k');
h1 = imagesc(deltaVar_map);
set(h1, 'AlphaData', isfinite(deltaVar_map));
axis image off;
title('\Delta variance = slope \times (M_{95} - M_{5})');
cb1 = colorbar;
caxis([-deltaVar_lim, deltaVar_lim]);
ylabel(cb1, '\Delta variance');

% 2) relative effect map
ax2 = subplot(2,3,2);
set(ax2, 'Color', 'k');
h2 = imagesc(relEffect_map);
set(h2, 'AlphaData', isfinite(relEffect_map));
axis image off;
title('Relative effect = \Delta variance / mean(D^2)');
cb2 = colorbar;
caxis([-relEffect_lim, relEffect_lim]);
ylabel(cb2, 'Relative effect');

% 3) -log10(q) map
ax3 = subplot(2,3,3);
set(ax3, 'Color', 'k');
h3 = imagesc(logq_map);
set(h3, 'AlphaData', isfinite(logq_map));
axis image off;
title('-log_{10}(p-value) map');
colormap(ax3, hot);
cb3 = colorbar;
caxis([0, logq_lim]);
ylabel(cb3, '-log_{10}(p-value)');

% 4) deltaVar histogram
subplot(2,3,4);
histogram(deltaVar_valid, 100);
xlabel('\Delta variance');
ylabel('Count');
title('\Delta variance distribution');
xlim([-deltaVar_lim, deltaVar_lim]);
grid on;

% 5) relative effect histogram
subplot(2,3,5);
histogram(relEffect_valid, 100);
xlabel('Relative effect');
ylabel('Count');
title('Relative effect distribution');
xlim([-relEffect_lim, relEffect_lim]);
grid on;

% 6) FDR significance map
ax6 = subplot(2,3,6);
set(ax6, 'Color', 'k');
h6 = imagesc(sig_map);
set(h6, 'AlphaData', isfinite(sig_map));
axis image off;
title('FDR significant pixels');
colormap(ax6, gray);
cb6 = colorbar;
caxis([0 1]);
cb6.Ticks = [0 1];
cb6.TickLabels = {'No', 'Yes'};

% Diverging colormap for effect maps
bwr = local_bluewhitered(256);
colormap(ax1, bwr);
colormap(ax2, bwr);

% Summary
fprintf('deltaVar robust display limit: +/- %.4g\n', deltaVar_lim);
fprintf('relEffect robust display limit: +/- %.4g\n', relEffect_lim);
fprintf('-log10(q) upper display limit: %.4f\n', logq_lim);
fprintf('Median deltaVar over unmasked pixels: %.4g\n', median(deltaVar_valid, 'omitnan'));
fprintf('Median relative effect over unmasked pixels: %.4g\n', median(relEffect_valid, 'omitnan'));
fprintf('Fraction FDR significant: %.4f\n', mean(sig_map(mask2d), 'omitnan'));

clear ax1 ax2 ax3 ax6 h1 h2 h3 h6 cb1 cb2 cb3 cb6 bwr
clear deltaVar_map relEffect_map meanY_map deltaM_map
clear deltaVar_valid relEffect_valid q_valid logq_map
clear intercept_vec sig_vec q_tmp sample_valid sample_M sample_Y
clear sample_M_sorted sample_fit_sorted sample_Y_sorted sample_row
clear X_use X1 X2 D rho_map intercept_map q_map sig_map


%% Check whether 1 single slope and pixel-specific intercept sufficies
n = sum(valid, 2);
meanM = sum(M, 2) ./ max(n, 1);
meanY = sum(Y, 2) ./ max(n, 1);

% common slope with pixel-specific intercepts: estimate from within-pixel
% centered data, not globally centered pooled data.
Mc = M - meanM;
Yc = Y - meanY;
Mc(~valid) = 0;
Yc(~valid) = 0;

b_common = sum(Mc(:) .* Yc(:)) / sum(Mc(:) .^ 2);

% pixel-specific slopes
b_pix = slope_map(mask2d);

% row-specific intercepts
a_common = meanY - b_common .* meanM;
a_pix = meanY - b_pix .* meanM;

% fitted values
Yhat_common = a_common + b_common .* M;
Yhat_pix = a_pix + b_pix .* M;

% ignore invalid entries
Yhat_common(~valid) = 0;
Yhat_pix(~valid) = 0;

res_common = Y - Yhat_common;
res_pix = Y - Yhat_pix;

res_common(~valid) = 0;
res_pix(~valid) = 0;

sse_common = sum(res_common.^2, 2);
sse_pix = sum(res_pix.^2, 2);

ratio = sse_common ./ max(sse_pix, eps);

fprintf('Common slope = %.4g\n', b_common);
fprintf('Median SSE ratio (common / pixel-specific) = %.4f\n', median(ratio, 'omitnan'));
fprintf('95th percentile SSE ratio = %.4f\n', prctile(ratio(isfinite(ratio)), 95));

%% Further check on intercept
% Model A: common slope + pixel-specific intercepts
% Model B: common slope + one common intercept

a_one = sum(Y(:) - b_common .* M(:)) / sum(valid(:));   % one pooled intercept

Yhat_one = a_one + b_common .* M;
Yhat_one(~valid) = 0;

res_one = Y - Yhat_one;
res_one(~valid) = 0;

sse_one = sum(res_one.^2, 2);

ratio_AB = sse_one ./ max(sse_common, eps);

fprintf('One-intercept model: a = %.6g\n', a_one);
fprintf('Median SSE ratio (one intercept / pixel-specific intercepts) = %.4f\n', ...
    median(ratio_AB, 'omitnan'));
fprintf('95th percentile SSE ratio = %.4f\n', ...
    prctile(ratio_AB(isfinite(ratio_AB)), 95));
fprintf('Global SSE ratio = %.4f\n', ...
    sum(sse_one, 'omitnan') / max(sum(sse_common, 'omitnan'), eps));

%%



%%
function cmap = local_bluewhitered(n)
if nargin < 1
    n = 256;
end

n = max(2, round(n));
half = floor(n / 2);

if half == 0
    cmap = [1 1 1];
    return;
end

blue_to_white = [linspace(0, 1, half).', linspace(0, 1, half).', ones(half, 1)];
white_to_red = [ones(n - half, 1), linspace(1, 0, n - half).', linspace(1, 0, n - half).'];

cmap = [blue_to_white; white_to_red];
cmap = cmap(1:n, :);
end
