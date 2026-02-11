%% 
clear
clc
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));
%%
mouse_num = 2;
day1 = 21;
day2 = 28;
savefile1 = strcat("F:\Astrocyte_data\GGP#", num2str(mouse_num), "_d", num2str(day1), "\downsampled_smoothed_data_all_videos.mat");
savefile2 = strcat("F:\Astrocyte_data\GGP#", num2str(mouse_num), "_d", num2str(day2), "\downsampled_smoothed_data_all_videos.mat");
data1 = load(savefile1);
data2 = load(savefile2);
%%
X_dFF1 = reshape(data1.dFF, [], size(data1.dFF, 3));
X_dFF2 = reshape(data2.dFF, [], size(data2.dFF, 3));

k_nmf_comp = 6; % number of components in NMF

combined_mask = zeros(size(data1.mask_downsampled));
combined_mask(data1.mask_downsampled + data2.mask_downsampled == 2) = 1; % take the intersection of the two masks as active area

unmasked_indices = find(combined_mask);
unmasked_X_dFF1 = X_dFF1; % pixels * time
unmasked_X_dFF1 = unmasked_X_dFF1(unmasked_indices, :);

unmasked_X_dFF2 = X_dFF2;
unmasked_X_dFF2 = unmasked_X_dFF2(unmasked_indices, :);

%%
evt_domain_projection = zeros(size(data1.evt_map_downsampled, 1), size(data1.evt_map_downsampled, 2));
conn = bwconncomp(data1.evt_map_downsampled, 6);
for ii = 1:conn.NumObjects
    indices = conn.PixelIdxList{ii};
    [xx, yy, zz] = ind2sub(size(data1.evt_map_downsampled), indices);
    xx_select = xx(zz==mode(zz));
    yy_select = yy(zz==mode(zz));
    ind_2d = sub2ind([size(data1.evt_map_downsampled, 1), size(data1.evt_map_downsampled, 2)], xx_select, yy_select);
    evt_domain_projection(ind_2d) = evt_domain_projection(ind_2d) + 1;
end
figure
imagesc(evt_domain_projection)

%% constrained NMF
opts = struct();

% =====================
% Iteration / stopping
% =====================
opts.maxIter   = 120;
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

new_height = size(data1.datOrg1_downsampled_smoothed, 1);
new_width = size(data1.datOrg1_downsampled_smoothed, 2);
%%
[A, C1, info] = custom_cnmf(X_dFF1, new_height, new_width, k_nmf_comp, combined_mask, opts);

norm(A(unmasked_indices, :) * C1 + info.B(unmasked_indices, :) * info.F - unmasked_X_dFF1, 'fro') / norm(unmasked_X_dFF1, 'fro')
% Visualize component k footprint
ethogram_mat1 = data1.ethogram_mat_downsampled;
C1_augmented = [C1; info.F];
A_augmented = [A info.B];
plotNMF_withBehaviorOnsets(C1_augmented, ethogram_mat1', 40/data1.temp_down_factor, A_augmented, [new_height, new_width])


%% use the A and B to infer C and F
[C2, F] = estimate_CF_from_fixed_AB(X_dFF2, A, info.B);
ethogram_mat2 = data2.ethogram_mat_downsampled;
plotNMF_withBehaviorOnsets(C2, ethogram_mat2', 40/data2.temp_down_factor, A, [new_height, new_width])

%% evaluate the fitting 
Xhat = A(unmasked_indices,:) * C2 + info.B(unmasked_indices,:) * F;
R = Xhat - unmasked_X_dFF2;
R2_global = 1 - (norm(R,'fro')^2 / ...
    norm(unmasked_X_dFF2 - mean(unmasked_X_dFF2(:)),'fro')^2)
X_centered = unmasked_X_dFF2 - mean(unmasked_X_dFF2, 2);

R2_pixelCentered = 1 - (norm(R,'fro')^2 / ...
    norm(X_centered,'fro')^2)
nrmse = norm(R,'fro') / norm(unmasked_X_dFF2,'fro')
relerr = norm(R,'fro') / norm(unmasked_X_dFF2,'fro')
nrmse_centered = norm(R,'fro') / norm(X_centered,'fro')
%%
fs = 4;                 % Hz
win_sec = 10;           % seconds
win = max(3, round(win_sec*fs));
if mod(win,2)==0, win = win+1; end

F_smooth = F;
for k = 1:size(F,1)
    F_smooth(k,:) = smoothdata(F(k,:), 'movmean', win);
end

Xhat_smooth = A(unmasked_indices,:) * C2 + info.B(unmasked_indices,:) * F_smooth;
R_smooth = Xhat_smooth - unmasked_X_dFF2;

R2_pixelCentered_smooth = 1 - norm(R_smooth,'fro')^2 / ...
    (norm(unmasked_X_dFF2 - mean(unmasked_X_dFF2,2),'fro')^2 + eps)


%%
meanAbsResidual = mean(abs(R), 2);
meanAbsResidual_org = zeros(new_width*new_height, 1);
meanAbsResidual_org(unmasked_indices) = meanAbsResidual;

imagesc(reshape(meanAbsResidual_org, new_height, new_width));
axis image; colorbar;
title('Mean |Residual| on Day 28');

%%
SSE_t = sum(R.^2, 1);
SST_t = sum((unmasked_X_dFF2 - mean(unmasked_X_dFF2,1)).^2, 1);

R2_time = 1 - SSE_t ./ (SST_t + eps);

figure;
plot(R2_time);
ylim([0 1]);
title('R^2 per time frame (Day 28)');
xlabel('Frame');
ylabel('R^2');
%%
kshow = randperm(size(C2,1), min(10,size(C2,1)));

figure;
plot(C2(kshow,:)');
title('Sample inferred neuron activity traces C2 (Day 28)');

%%
C = C2;
ethogram_mat = ethogram_mat2;
ethogram_mat = ethogram_mat > 0.5;

fs = 4;

% windows for astrocytes (slow)
pre_sec  = 10;
post_sec = 20;
pre  = round(pre_sec*fs);
post = round(post_sec*fs);
win = -pre:post;

baseline_idx = win < 0;
response_idx = win >= 0 & win <= round(10*fs);  % first 10 s after onset

T = size(C,2);
K = size(C,1);                 % should be 6
B = size(ethogram_mat,2);      % should be 9

% onset_mat from ethogram_mat
onset_mat = zeros(size(ethogram_mat));
for b = 1:B
    onset_mat(:,b) = ([0; diff(ethogram_mat(:,b))] == 1);
end

AUC_all = nan(B,1);
p_perm_all = nan(B,1);
pvals_components = nan(K,B);

for b = 1:B
    onset_frames = find(onset_mat(:,b) == 1);

    % keep only those with full window available
    onset_frames = onset_frames(onset_frames - pre >= 1 & onset_frames + post <= T);
    N = numel(onset_frames);
    if N < 5
        continue
    end

    % event-aligned C
    C_event = zeros(K, numel(win), N);
    for i = 1:N
        t0 = onset_frames(i);
        C_event(:,:,i) = C(:, t0+win);
    end

    % effect size: response - baseline per component per event
    baseline = squeeze(mean(C_event(:,baseline_idx,:), 2));   % K x N
    response = squeeze(mean(C_event(:,response_idx,:), 2));   % K x N
    delta = response - baseline;                               % K x N

    % component-level t-tests
    for k = 1:K
        [~, pvals_components(k,b)] = ttest(delta(k,:));
    end

    % permutation test on a scalar summary effect
    true_effect = mean(delta(:));
    nShuffle = 1000;
    shuf_effect = zeros(nShuffle,1);
    for s = 1:nShuffle
        shuf = randi([pre+1, T-post], size(onset_frames));
        C_shuf = zeros(K, numel(win), N);
        for i = 1:N
            t0 = shuf(i);
            C_shuf(:,:,i) = C(:, t0+win);
        end
        base_s = squeeze(mean(C_shuf(:,baseline_idx,:),2));
        resp_s = squeeze(mean(C_shuf(:,response_idx,:),2));
        shuf_effect(s) = mean((resp_s - base_s), 'all');
    end
    p_perm_all(b) = mean(shuf_effect >= true_effect);

    % decoding (AUC) — use a small positive window to label frames near onset
    y = onset_mat(:,b);                 % T x 1
    X = C';                             % T x 6

    % optional: label +/- 1 s around onset as positive to account for slow timing
    pad_sec = 1;
    pad = round(pad_sec*fs);
    y_pad = y;
    idx = find(y==1);
    for ii = 1:numel(idx)
        y_pad(max(1,idx(ii)-pad):min(T,idx(ii)+pad)) = 1;
    end

    % simple 5-fold CV AUC
    cv = cvpartition(T, 'KFold', 5);
    yhat = zeros(T,1);

    for fold = 1:5
        tr = training(cv, fold);
        te = test(cv, fold);
        mdl = fitglm(X(tr,:), y_pad(tr), 'Distribution','binomial');
        yhat(te) = predict(mdl, X(te,:));
    end

    [~,~,~,AUC_all(b)] = perfcurve(y_pad, yhat, 1);
end

AUC_all
p_perm_all
pvals_components



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

            colormap(axImg, "parula");
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




