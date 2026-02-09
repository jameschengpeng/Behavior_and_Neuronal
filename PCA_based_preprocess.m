%% 
clear
clc
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));
%%
% AQuA2_file_suffix = "ManualMoco_cropped_AQuA2"; % for cck
AQuA2_file_suffix = "moco_cropped_AQuA2"; % for astrocyte

% % For the CCK neuronal data
% AQuA2_result_path = "D:\Mouse_behavior_data\D21\AQuA2";
% Ethogram_scoring_path = "D:\Mouse_behavior_data\D21\EthogramScoring";

% For the GGP data, various mice, various days
mouse_num = 2;
day = 21;

AQuA2_result_path = strcat("D:\Astrocyte_data\GGP#", num2str(mouse_num), "_d", num2str(day));
Ethogram_scoring_path = strcat("D:\Astrocyte_data\GGP#", num2str(mouse_num), "_d", num2str(day), "\EthogramScoring");

% For all recordings
[dF1, datOrg1, evt_map, subset_cutting_points, early_reaction_regions] = concat_videos(AQuA2_result_path, AQuA2_file_suffix);

% add 0 to the first element of subset_cutting_points
subset_cutting_points = [0; subset_cutting_points];

concat_ethogram_mat = concat_ethograms(Ethogram_scoring_path, subset_cutting_points);

video_lengths = diff(subset_cutting_points);

% load the mask
mask = load(fullfile(AQuA2_result_path, "mask.mat"));
mask = mask.bd0;
upper_unmasked = mask{1}{2};
lower_unmasked = mask{2}{2};
mask = zeros(size(dF1, 1), size(dF1, 2));
mask(upper_unmasked) = 1;
mask(lower_unmasked) = 1;

%% Downsampling
% Downsample in spatial and temporal dimensions
spa_down_factor = 2;
temp_down_factor = 4; 
[height, width, nframes] = size(dF1);

new_height = floor(height / spa_down_factor);
new_width  = floor(width  / spa_down_factor);
new_frames = sum(floor(video_lengths ./ temp_down_factor)); % addition of all down-sampled video lengths. NOT total frames divided by down-sample factor


dF1_downsampled = zeros(new_height, new_width, new_frames);
datOrg1_downsampled = zeros(new_height, new_width, new_frames);
ethogram_mat_downsampled = zeros(new_frames, size(concat_ethogram_mat, 2));
evt_map_downsampled = zeros(new_height, new_width, new_frames);

% Downsample + spatial smoothing each frame. Do it video by video
counter = 0; % record the time index AFTER temp down-sample
for i = 1:numel(video_lengths) % iterate over videos
    start_T = subset_cutting_points(i)+1; % global index BEFORE temp down-sample
    end_T = subset_cutting_points(i+1); % global index BEFORE temp down-sample
    T_down = floor(video_lengths(i) / temp_down_factor); % time length of this video after temp down-sample

    video_dF1 = dF1(:, :, start_T:end_T);
    video_org1 = datOrg1(:, :, start_T:end_T);
    video_ethogram = concat_ethogram_mat(start_T:end_T, :);
    video_evt = evt_map(:, :, start_T:end_T);
    
    % spatially downsample using imresize
    video_dF1_spa_down = imresize(video_dF1, [new_height, new_width], 'bicubic');
    video_org1_spa_down = imresize(video_org1, [new_height, new_width], 'bicubic');
    video_evt_spa_down = imresize(video_evt, [new_height, new_width], 'nearest');

    % temporally downsample by taking average for video and dF1 and also
    % the ethogram. The ethogram becomes a probability
    video_dF1_spa_temp_down = zeros(new_height, new_width, T_down);
    video_org1_spa_temp_down = zeros(new_height, new_width, T_down);
    video_ethogram_temp_down = zeros(T_down, size(concat_ethogram_mat, 2));
    video_evt_spa_temp_down = zeros(new_height, new_width, T_down);
    for j = 1:T_down
        substart_T = (j-1) * temp_down_factor + 1; % local index in a video BEFORE temp down-sample
        subend_T = j * temp_down_factor; % local index in a video BEFORE temp down-sample
        video_dF1_spa_temp_down(:, :, j) = mean(video_dF1_spa_down(:, :, substart_T:subend_T), 3);
        video_org1_spa_temp_down(:, :, j) = mean(video_org1_spa_down(:, :, substart_T:subend_T), 3);
        video_ethogram_temp_down(j, :) = mean(video_ethogram(substart_T:subend_T, :), 1);
        video_evt_spa_temp_down(:, :, j) = mean(video_evt_spa_down(:, :, substart_T:subend_T), 3) > 0.5;
    end
    
    new_start_T = counter + 1;
    new_end_T = counter + T_down;
    dF1_downsampled(:, :, new_start_T:new_end_T) = video_dF1_spa_temp_down;
    datOrg1_downsampled(:, :, new_start_T:new_end_T) = video_org1_spa_temp_down;
    ethogram_mat_downsampled(new_start_T:new_end_T, :) = video_ethogram_temp_down; 
    evt_map_downsampled(:, :, new_start_T:new_end_T) = video_evt_spa_temp_down;
    
    counter = new_end_T;
end
% delete the intermediate variables to save space
clear video_dF1 video_org1 video_ethogram video_dF1_spa_down video_org1_spa_down video_dF1_spa_temp_down video_org1_spa_temp_down video_ethogram_temp_down
clear video_evt_spa_down video_evt_spa_temp_down evt_map

% apply mild smoothing spatially
dF1_downsampled_smoothed = imgaussfilt3(dF1_downsampled, [2, 2, 1e-6]);
datOrg1_downsampled_smoothed = imgaussfilt3(datOrg1_downsampled, [2, 2, 1e-6]);

clear dF1_downsampled datOrg1_downsampled

% correct the subset cutting points after temporal downsampling
new_subset_cutting_points = cumsum(floor(video_lengths ./ temp_down_factor)); % update the subset cutting points
new_subset_cutting_points = [0; new_subset_cutting_points]; % include 0 at beginning

% downsample the mask
mask_downsampled = imresize(mask, [new_height, new_width]);
mask_downsampled(mask_downsampled < 0.5) = 0;
mask_downsampled(mask_downsampled >= 0.5) = 1;

% -------------------------------------------------------------------------
% Reshape: each row = one frame, each column = one pixel
X_dF = reshape(dF1_downsampled_smoothed, [], new_frames);
X_dF = X_dF.';   % size: T x (H*W)
if min(X_dF(:)) < 0
    X_dF = X_dF - min(X_dF(:));
end

%% compute the baseline for dF/F
n_video_select = numel(new_subset_cutting_points)-1; % all
start_video = 1;
select_start = new_subset_cutting_points(start_video); % global, not including starting index of the video, so it can be 0
select_end = new_subset_cutting_points(start_video + n_video_select); % global
voxel_baseline = zeros(size(datOrg1_downsampled_smoothed(:,:,(select_start+1):select_end))); % local 
for ii = 1:n_video_select
    start_ii = new_subset_cutting_points(ii+start_video-1)+1; % global
    end_ii = new_subset_cutting_points(ii+start_video); % global
    single_video_baseline = find_baseline_all_voxel(datOrg1_downsampled_smoothed(:,:,start_ii:end_ii), 40, 8); % global
    voxel_baseline(:,:,(start_ii-select_start):(end_ii-select_start)) = single_video_baseline; % local
end

dFF = (datOrg1_downsampled_smoothed(:,:,(select_start+1):select_end) - voxel_baseline)./voxel_baseline;
X_dFF = reshape(dFF, [], size(dFF, 3));
X_dFF = X_dFF';
if min(X_dFF(:)) < 0
    X_dFF = X_dFF - min(X_dFF(:));  
end

% save preprocessed dF and datOrg
% savefile = "D:\Mouse_behavior_data\D21\downsampled_smoothed_data_all_videos.mat";
% save(savefile, "dF1_downsampled_smoothed", "datOrg1_downsampled_smoothed", ...
%     "ethogram_mat_downsampled", "new_subset_cutting_points", "evt_map_downsampled", ...
%     "dFF", "mask_downsampled", "temp_down_factor", "-v7.3");

savefile = strcat("D:\Astrocyte_data\GGP#", num2str(mouse_num), "_d", num2str(day), "\downsampled_smoothed_data_all_videos.mat");
save(savefile, "dF1_downsampled_smoothed", "datOrg1_downsampled_smoothed", ...
    "ethogram_mat_downsampled", "new_subset_cutting_points", "evt_map_downsampled", ...
    "dFF", "mask_downsampled", "temp_down_factor", "-v7.3");


%% read the saved data, you can start here
savefile = "D:\Mouse_behavior_data\D21\downsampled_smoothed_data_all_videos.mat";
saved_data = load(savefile);
dF1_downsampled_smoothed        = saved_data.dF1_downsampled_smoothed;
datOrg1_downsampled_smoothed    = saved_data.datOrg1_downsampled_smoothed;
ethogram_mat_downsampled        = saved_data.ethogram_mat_downsampled;
new_subset_cutting_points       = saved_data.new_subset_cutting_points;
evt_map_downsampled             = saved_data.evt_map_downsampled;
dFF                             = saved_data.dFF;
mask_downsampled                = saved_data.mask_downsampled;
temp_down_factor                = saved_data.temp_down_factor;

X_dFF = reshape(dFF, [], size(dFF, 3));
X_dFF = X_dFF';
if min(X_dFF(:)) < 0
    X_dFF = X_dFF - min(X_dFF(:));  
end

new_height = size(dF1_downsampled_smoothed, 1);
new_width = size(dF1_downsampled_smoothed, 2);

%%
evt_domain_projection = zeros(size(evt_map_downsampled, 1), size(evt_map_downsampled, 2));
conn = bwconncomp(evt_map_downsampled, 6);
for ii = 1:conn.NumObjects
    indices = conn.PixelIdxList{ii};
    [xx, yy, zz] = ind2sub(size(evt_map_downsampled), indices);
    xx_select = xx(zz==mode(zz));
    yy_select = yy(zz==mode(zz));
    ind_2d = sub2ind([size(evt_map_downsampled, 1), size(evt_map_downsampled, 2)], xx_select, yy_select);
    evt_domain_projection(ind_2d) = evt_domain_projection(ind_2d) + 1;
end
imagesc(evt_domain_projection)
%% 
k_nmf_comp = 8; % number of components in NMF
unmasked_indices = find(mask_downsampled);
unmasked_X_dFF = X_dFF'; % pixels * time
unmasked_X_dFF = unmasked_X_dFF(unmasked_indices, :);

%% vanilla NMF from matlab's built-in function. Set options for NNMF
opt = statset('nnmf');
opt.MaxIter = 1000;
opt.Display = 'final';
opt.TolFun  = 1e-6;
opt.TolX    = 1e-6;

% Run NNMF with multiple replicates (non-convex => helps stability)
% W for spatial components, H for time courses
[W, H, D] = nnmf(unmasked_X_dFF, k_nmf_comp, ...
    'Algorithm', 'als', ...     % try 'mult' if you want
    'Replicates', 5, ...       % increase (15–20) if components vary across runs
    'Options', opt);

% recover to the full picture
W_full = zeros([size(X_dFF,2), k_nmf_comp]);
W_full(unmasked_indices, :) = W;

% Reshape spatial components back to image dimensions (height × width × k_nmf_comp)
spatial_components = reshape(W_full, [new_height, new_width, k_nmf_comp]);

% Temporal traces (k_nmf_comp × T)
temporal_traces = H;

norm(W * H - unmasked_X_dFF) / norm(unmasked_X_dFF)

ethogram_mat = ethogram_mat_downsampled((select_start+1):select_end, :);
plotNMF_withBehaviorOnsets(H, ethogram_mat', 40/temp_down_factor, W_full, [new_height, new_width])

%%
ethogram_mat = ethogram_mat > 0.5;
Z = nmf_behavior_onset_zscore(C, ethogram_mat, 40/temp_down_factor, 2, 2);  % returns K x 9
Z(Z < 2) = -10;
figure
imagesc(Z)

%% constrained NMF
opts = struct();
opts.quiet_prctile = 20;
% =====================
% Iteration / stopping
% =====================
opts.maxIter   = 150;
opts.minIter   = 60;
opts.tol       = 1e-5;

% =====================
% Spatial penalties
% =====================
opts.lambdaA_L1   = 1e-6;     % VERY weak sparsity (astrocytes are diffuse)
opts.lambdaA_lap  = 5e-4;     % smooth contiguous regions
opts.lambdaA_excl = 0;        % OFF initially (overlap is allowed)

% =====================
% Temporal penalty
% =====================
opts.lambdaC_smooth = 1e-3;   % smooth calcium dynamics
opts.lambdaF_smooth = 1e-2;

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

n_video_train = 10;
T_train = new_subset_cutting_points(n_video_train+1);

[A, C_train, info_train] = custom_cnmf(X_dFF(1:T_train, :)', new_height, new_width, k_nmf_comp, mask_downsampled, opts);
%% Use the pre-fitted A and B matrices to figure out the remaining time course
B = info_train.B;
[C_test, F_test, info_test] = infer_CF_fixed_AB(X_dFF(T_train+1:end, :)', A, B, new_height, new_width, mask_downsampled, opts);

C = [C_train C_test];
F = [info_train.F F_test];

norm(A(unmasked_indices, :) * C + B(unmasked_indices, :) * F - unmasked_X_dFF, 'fro') / norm(unmasked_X_dFF, 'fro')
% Visualize component k footprint
% ethogram_mat = ethogram_mat_downsampled((select_start+1):select_end, :);
ethogram_mat = ethogram_mat_downsampled;

%% add the background into C and A
C_augmented = [C; F];
A_augmented = [A B];
plotNMF_withBehaviorOnsets(C_augmented, ethogram_mat', 40/temp_down_factor, A_augmented, [new_height, new_width])

%% obtain the z-score map
% for each behavior and each NMF component, we extract the relevant time
% window, and compute z-score for the signals within time-window compared
% to the baseline signals
ethogram_mat = ethogram_mat > 0.5;
Z = nmf_behavior_onset_zscore(C_augmented, ethogram_mat, 40/temp_down_factor, 3, 2);  % returns K x 9
Z(Z < 2) = NaN;

figure
h = imagesc(Z);
set(gca,'Color','k')                  % black background
set(h,'AlphaData', ~isnan(Z))         % NaNs become transparent -> show black
xlabel("Behavior type", 'FontSize', 12, 'FontWeight','bold');
ylabel("NMF component number", 'FontSize', 12, 'FontWeight','bold');
title("Z-score", 'FontSize', 12, 'FontWeight','bold')
colorbar

%% Train / test evaluation for vanilla NMF (time-split CV)

% -----------------------
% Build active-pixel data matrix: X_active is P_active x T
% -----------------------
X_active = X_dFF';                         % P x T
X_active = X_active(unmasked_indices, :);  % P_active x T

% Optional: ensure nonnegativity for nnmf
% (If your X is already nonnegative, you can skip)
X_active = X_active - min(X_active(:));    % shift to >=0

[P_active, T] = size(X_active);

% -----------------------
% Time split
% -----------------------
trainFrac = 0.8;
Ttrain = floor(trainFrac * T);
train_idx = 1:Ttrain;
test_idx  = (Ttrain+1):T;

X_train = X_active(:, train_idx);   % P_active x Ttrain
X_test  = X_active(:, test_idx);    % P_active x Ttest
Ttest   = numel(test_idx);

% -----------------------
% Fit vanilla NMF on training data: X_train ≈ W_train * H_train
% W_train: P_active x K
% H_train: K x Ttrain
% -----------------------
opt = statset('nnmf');
opt.MaxIter = 1000;
opt.Display = 'final';
opt.TolFun  = 1e-6;
opt.TolX    = 1e-6;

[W_train, H_train, D_train] = nnmf(X_train, k_nmf_comp, ...
    'Algorithm', 'als', ...
    'Replicates', 5, ...
    'Options', opt);

% -----------------------
% Infer test time courses with W_train fixed:
% For each test frame x_t (P_active x 1), solve:
%   h_t = argmin_{h>=0} || x_t - W_train*h ||_2^2
% Collect h_t into H_test (K x Ttest)
% -----------------------
H_test = zeros(k_nmf_comp, Ttest);
for tt = 1:Ttest
    H_test(:, tt) = lsqnonneg(W_train, X_test(:, tt));
end

% Predict X_test
Xhat_test = W_train * H_test;  % P_active x Ttest

% -----------------------
% Metrics (train & test)
% -----------------------
Xhat_train = W_train * H_train;
relRecon_train = norm(X_train - Xhat_train, 'fro')^2 / (norm(X_train, 'fro')^2 + eps);
relRecon_test  = norm(X_test  - Xhat_test,  'fro')^2 / (norm(X_test,  'fro')^2 + eps);

fprintf('Vanilla NMF CV (time split):\n');
fprintf('  relRecon_train = %.6g\n', relRecon_train);
fprintf('  relRecon_test  = %.6g\n', relRecon_test);

% -----------------------
% Recover full-frame spatial maps (optional, for visualization)
% W_full: P x K, with masked pixels = 0
% -----------------------
P = size(X_dFF,2);  % pixels
W_full = zeros(P, k_nmf_comp);
W_full(unmasked_indices, :) = W_train;

spatial_components_train = reshape(W_full, [new_height, new_width, k_nmf_comp]); % H x W x K

% Temporal traces:
temporal_traces_train = H_train;  % K x Ttrain
temporal_traces_test  = H_test;   % K x Ttest

%% Train test split for constrained NMF


% -----------------------
% Build active-pixel data matrix: X_active is P_active x T
% -----------------------


X_full = X_dFF';                            % P x T  (your convention)
X_active = X_full(unmasked_indices, :);     % P_active x T

% Optional: ensure nonnegativity (custom_cnmf assumes nonneg in comments)
X_active = X_active - min(X_active(:));

[P_active, T] = size(X_active);

% -----------------------
% Time split
% -----------------------
trainFrac = 0.8;
Ttrain = floor(trainFrac * T);
train_idx = 1:Ttrain;
test_idx  = (Ttrain+1):T;

X_train = X_active(:, train_idx);   % P_active x Ttrain
X_test  = X_active(:, test_idx);    % P_active x Ttest
Ttest   = numel(test_idx);

% -----------------------
% Train CNMF on active pixels only
% IMPORTANT:
%   custom_cnmf expects X as (H*W) x T and will apply mask internally.
%   Here we already removed masked pixels, so we pass a "mask_all_ones"
%   and set H = P_active, W = 1 (so H*W matches P_active).
%   Also pass a dummy mask of ones so no pixels are removed again.
% -----------------------
H_active = P_active;
W_active = 1;
mask_all_ones = true(P_active, 1);

[A_train_act, C_train, info_train] = custom_cnmf(X_train, H_active, W_active, k_nmf_comp, mask_all_ones, opts);
% A_train_act: P_active x K
% C_train:     K x Ttrain

% -----------------------
% Test: infer C_test with A_train_act fixed using SAME NNLS as vanilla
% -----------------------
C_test = zeros(k_nmf_comp, Ttest);
for tt = 1:Ttest
    C_test(:, tt) = lsqnonneg(A_train_act, X_test(:, tt));
end

Xhat_train = A_train_act * C_train;  % P_active x Ttrain
Xhat_test  = A_train_act * C_test;   % P_active x Ttest

relRecon_train = norm(X_train - Xhat_train, 'fro')^2 / (norm(X_train, 'fro')^2 + eps);
relRecon_test  = norm(X_test  - Xhat_test,  'fro')^2 / (norm(X_test,  'fro')^2 + eps);

fprintf('Custom CNMF CV (time split, unmasked pixels only, NNLS test decoding):\n');
fprintf('  relRecon_train = %.6g\n', relRecon_train);
fprintf('  relRecon_test  = %.6g\n', relRecon_test);

% -----------------------
% Optional: recover full-size A for visualization (masked pixels = 0)
% -----------------------
P = size(X_full,1);               % P = new_height*new_width
A_full = zeros(P, k_nmf_comp);
A_full(unmasked_indices, :) = A_train_act;

A_full_img = reshape(A_full, [new_height, new_width, k_nmf_comp]);  % H x W x K

% Training temporal traces:
C_train_traces = C_train;  % K x Ttrain
C_test_traces  = C_test;   % K x Ttest

%%



%% Step 1: Perform PCA
[coeff, score, latent, tsquared, explained, mu] = pca(X_dFF);

% Let p = H*W, n = T
% coeff: p x (n-1 or p) matrix of principal component coefficients. Each
% column of coeff is a spatial principal component

% score: n x (n-1 or p) matrix of principal component scores. Each column
% of score is the temporal activity of a principal component

% latent: eigenvalues (variances) of principal components
% tsquared: Hotelling's T-squared statistic for each observation
% explained: percentage of variance explained by each component
% mu: mean of each variable (used for centering)

%% Step 2: Visualize the variance explained
figure;
subplot(2,2,1);
plot(explained(1:20), 'bo-', 'LineWidth', 2);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Variance Explained by Each PC');
grid on;

subplot(2,2,2);
plot(cumsum(explained(1:20)), 'ro-', 'LineWidth', 2);
xlabel('Principal Component');
ylabel('Cumulative Variance Explained (%)');
title('Cumulative Variance Explained');
yline(90, '--k', '90%');
yline(95, '--k', '95%');
grid on;

subplot(2,2,3);
pareto(explained(1:20));
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Pareto Chart of Variance');

subplot(2,2,4);
semilogy(latent(1:20), 'go-', 'LineWidth', 2);
xlabel('Principal Component');
ylabel('Eigenvalue (log scale)');
title('Scree Plot (Eigenvalues)');
grid on;

%%
nPC = 6;   % number of PCs to visualize
figure;
for k = 1:nPC
    subplot(2,3,k)
    imagesc(reshape(coeff(:,k), new_height, new_width))
    axis image off
    colormap parula
    title(sprintf('PC %d (%.2f%%)', k, explained(k)))
end


%% select the number of principal components based on how much variance they explain
k = 12;

%% Step 5: Extract reduced data
coeff_reduced = coeff(:, 1:k);      % PC loadings for first k components
score_reduced = score(:, 1:k);      % PC scores for first k components
latent_reduced = latent(1:k);       % Eigenvalues for first k components

%% Step 6: Reconstruct data (optional - to check quality)
X_reconstructed = score_reduced * coeff_reduced' + mu;
reconstruction_error = norm(X_dFF - X_reconstructed, 'fro') / norm(X_dFF, 'fro');
fprintf('Reconstruction error: %.4f\n', reconstruction_error);

%% Show a spatial principal component and its time-lapse coefficient
pc_id = 1;
figure;
subplot(1,2,1)
imagesc(reshape(coeff(:,pc_id), new_height, new_width))
axis image off
colormap gray
title(sprintf('Spatial PC %d', pc_id))

subplot(1,2,2)
plot(score(:,pc_id), 'k')
xlabel('Time')
ylabel('Amplitude')
title('Temporal score')

%% explore whether the neural representation has predictability on ethogram
dFF_PC = score_reduced;
ethogram_mat = ethogram_mat_downsampled((select_start+1):select_end, :);

%%
plotPCwithAllBehaviors(dFF_PC(:,3), ethogram_mat(:,[1,2,3]), 10)

function plotPCwithAllBehaviors(PC_trace, beh_trace, sampling_rate)
% plotPCwithAllBehaviors
% Plot a PC temporal trace and highlight intervals for ALL behaviors (each color-coded)
%
% INPUTS:
%   PC_trace      - T x 1 vector, PC temporal activity
%   beh_trace     - T x K matrix, behavior probabilities in [0,1]
%   sampling_rate - scalar, Hz (frames per second)
%
% EXAMPLE:
%   plotPCwithAllBehaviors(score(:,1), behProbMat, 30)

    % ---------- checks ----------
    PC_trace = PC_trace(:);
    if size(beh_trace,1) ~= numel(PC_trace)
        error('PC_trace length must match number of rows in beh_trace.');
    end
    if ~isscalar(sampling_rate) || sampling_rate <= 0
        error('sampling_rate must be a positive scalar (Hz).');
    end

    [T, K] = size(beh_trace);

    % ---------- parameters (edit if desired) ----------
    thr        = 0.1;   % probability threshold for behavior ON
    minDur_sec = 0.0;   % minimum bout duration (seconds); set e.g. 0.2 to remove tiny blips
    showLegend = true;  % set false if you don't want legend
    FaceAlpha = 0.8;

    % Behavior names (optional; auto if you don't have names)
    behNames = arrayfun(@(k) sprintf('Beh %d', k), 1:K, 'UniformOutput', false);

    % Distinct colors for behaviors
    C = lines(K);

    % ---------- time axis ----------
    t = (0:T-1)' / sampling_rate;

    % ---------- PC y-limits ----------
    yl = [min(PC_trace) max(PC_trace)];
    pad = 0.05 * (yl(2) - yl(1) + eps);
    yl = [yl(1)-pad yl(2)+pad];

    % ---------- plot ----------
    figure; hold on;

    % Plot shaded intervals for each behavior
    legendHandles = gobjects(K,1);

    for k = 1:K
        behProb = beh_trace(:,k);

        % ON/OFF mask
        mask = behProb >= thr;

        % Remove short bouts (optional)
        if minDur_sec > 0
            minSamples = max(1, round(minDur_sec * sampling_rate));

            d = diff([false; mask; false]);
            s = find(d == 1);
            e = find(d == -1) - 1;

            keep = (e - s + 1) >= minSamples;
            mask(:) = false;
            for ii = find(keep)'
                mask(s(ii):e(ii)) = true;
            end
        end

        % Bout boundaries
        d = diff([false; mask; false]);
        starts = find(d == 1);
        ends   = find(d == -1) - 1;

        % Draw patches (semi-transparent)
        for i = 1:numel(starts)
            x1 = t(starts(i));
            x2 = t(ends(i));
            patch([x1 x2 x2 x1], ...
                  [yl(1) yl(1) yl(2) yl(2)], ...
                  C(k,:), ...
                  'FaceAlpha', FaceAlpha, ...
                  'EdgeColor', 'none');
        end

        % Dummy patch for legend (so legend shows color even if no bouts)
        legendHandles(k) = patch(nan, nan, C(k,:), 'FaceAlpha', FaceAlpha, 'EdgeColor', 'none');
    end

    % Plot PC trace on top
    plot(t, PC_trace, 'k', 'LineWidth', 1.2);

    xlabel('Time (s)');
    ylabel('PC amplitude');
    title(sprintf('PC trace with all behaviors highlighted (thr = %.2f)', thr));
    ylim(yl);
    box on;

    if showLegend
        legend(legendHandles, behNames, 'Location', 'eastoutside');
    end

    hold off;
end



%%
function onsetIdx = plotNMF_withBehaviorOnsets(C, E, Fs, A, imgSize)
% plotNMF_withBehaviorOnsets
%
% Inputs:
%   C  : k x T matrix (NMF temporal components)
%   E  : nb x T binary ethogram matrix (nb behaviors)
%   Fs : sampling rate (Hz)
%   A  : (optional) nPixels x k matrix (NMF spatial components; each col = component)
%   imgSize : (optional) [H W] to reshape each spatial component
%
% Output:
%   onsetIdx : 1 x nb cell array, onset indices for each behavior

    [k, T] = size(C);
    [nb, T2] = size(E);
    assert(T == T2, 'C and E must have the same number of time points.');

    % ---- Handle optional spatial components ----
    hasSpatial = (nargin >= 4) && ~isempty(A);
    if hasSpatial
        [nPix, kA] = size(A);
        assert(kA == k, 'A must be nPixels x k so that each component matches each row of C.');

        if nargin < 5 || isempty(imgSize)
            % try to infer square image
            s = sqrt(nPix);
            if abs(s - round(s)) < 1e-10
                imgSize = [round(s), round(s)];
            else
                error('imgSize not provided and nPixels is not a perfect square. Provide imgSize = [H W].');
            end
        end
        assert(prod(imgSize) == nPix, 'imgSize must satisfy prod(imgSize) == size(A,1).');

        % ---- Global CLim across ALL spatial components (shared mapping) ----
        Aval = A(:);
        Aval = Aval(isfinite(Aval)); % ignores NaN/Inf
        low = prctile(Aval, 1);
        high = prctile(Aval, 99.5);
        
        climGlobal = [low high];
        % if isempty(Aval)
        %     climGlobal = [0 1];
        % else
        %     climGlobal = [min(Aval) max(Aval)];
        %     if climGlobal(1) == climGlobal(2)
        %         climGlobal = climGlobal + [-eps eps];
        %     end
        % end
    end

    % Time vector
    t = (0:T-1) / Fs;

    % Behavior names: A, B, C, ...
    behaviorNames = arrayfun(@(i) sprintf('Behavior %c', 'A'+i-1), ...
                             1:nb, 'UniformOutput', false);

    % Short labels: A, B, C, ...
    behaviorShort = arrayfun(@(i) sprintf('%c', 'A'+i-1), ...
                             1:nb, 'UniformOutput', false);

    % Colors for behaviors
    behaviorColors = lines(nb);

    % Compute onset indices (0 -> 1 transitions)
    onsetIdx = cell(1, nb);
    for b = 1:nb
        eb = E(b,:) ~= 0;
        d = diff([0, eb]);          % prepend 0 to catch onset at t=1
        onsetIdx{b} = find(d == 1);
    end

    % ---- Plot ----
    figure('Color','w');

    if hasSpatial
        tl = tiledlayout(k, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        imgAxes = gobjects(k,1);    % store spatial axes handles
    else
        tl = tiledlayout(k, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    end

    traceAxes = gobjects(k,1);

    for i = 1:k

        % ---- Left: spatial component map ----
        if hasSpatial
            axImg = nexttile(tl, (i-1)*2 + 1);
            imgAxes(i) = axImg;

            Ai = reshape(A(:,i), imgSize);
            Ai(Ai < 1e-6) = NaN;          % mask background
            imagesc(axImg, Ai);

            axis(axImg, 'image');
            axis(axImg, 'off');
            title(axImg, sprintf('Spatial feature %d', i), 'FontSize', 9, 'FontWeight', 'bold');

            colormap(axImg, "hot");
            set(axImg, 'Color', 'k');   % NaNs appear black
            clim(axImg, climGlobal);     % <<< shared scaling across ALL spatial plots
        end

        % ---- Right: time trace ----
        if hasSpatial
            axTr = nexttile(tl, (i-1)*2 + 2);
        else
            axTr = nexttile(tl, i);
        end
        traceAxes(i) = axTr;

        plot(axTr, t, C(i,:), 'k', 'LineWidth', 1); hold(axTr, 'on');

        % Plot behavior onset lines WITH labels at top
        for b = 1:nb
            for idx = onsetIdx{b}
                xl = xline(axTr, t(idx), '-', ...
                           'Color', behaviorColors(b,:), ...
                           'LineWidth', 1);

                xl.Label = behaviorShort{b};        % "A"..."I"
                xl.LabelOrientation = 'horizontal';
                xl.LabelVerticalAlignment = 'top';
                xl.LabelHorizontalAlignment = 'center';
                xl.FontSize = 9;
                xl.FontWeight = "bold";
            end
        end

        ylabel(axTr, sprintf('Feature%d ', i), "FontSize", 9, "FontWeight", "bold");

        if i == 1
            title(axTr, 'NMF components with behavior onsets', 'FontSize', 13);
        end

        % if i < k
        %     set(axTr, 'XTickLabel', []);
        % else
        %     xlabel(axTr, 'Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
        % end
        xlabel(axTr, 'Time (s)', 'FontSize', 12, 'FontWeight', 'bold');


        box(axTr, 'off');
    end

    % ---- ONE shared colorbar for all spatial tiles (R2023a; DO NOT use colorbar(tl)) ----
    if hasSpatial
        cb = colorbar(imgAxes(1));   % attach to an axes (valid syntax)
        cb.Layout.Tile = 'west';     % dock to the tiledlayout
        cb.Label.String = 'Spatial weight';
    end

    % ---- Single legend using dummy lines (attach to first trace axis, not image axes) ----
    axForLegend = traceAxes(1);
    hold(axForLegend, 'on');
    dummy = gobjects(1, nb);
    for b = 1:nb
        dummy(b) = plot(axForLegend, nan, nan, '-', ...
                        'Color', behaviorColors(b,:), ...
                        'LineWidth', 2);
    end
    legend(axForLegend, dummy, behaviorNames, 'Location', 'eastoutside');
end


