%% 
clear
clc
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));
%%
mouse_num1 = 2;
mouse_num2 = 23;
day = 21;
savefile1 = strcat("D:\Astrocyte_data\GGP#", num2str(mouse_num1), "_d", num2str(day), "\downsampled_smoothed_data_all_videos.mat");
savefile2 = strcat("D:\Astrocyte_data\GGP#", num2str(mouse_num2), "_d", num2str(day), "\downsampled_smoothed_data_all_videos.mat");
data1 = load(savefile1);
data2 = load(savefile2);
%%
X_dFF1 = reshape(data1.dFF, [], size(data1.dFF, 3));
X_dFF2 = reshape(data2.dFF, [], size(data2.dFF, 3));

k_nmf_comp = 6; % number of components in NMF

mask1 = data1.mask_downsampled;
mask2 = data2.mask_downsampled;


unmasked_indices1 = find(mask1);
unmasked_X_dFF1 = X_dFF1; % pixels * time
unmasked_X_dFF1 = unmasked_X_dFF1(unmasked_indices1, :);

unmasked_indices2 = find(mask2);
unmasked_X_dFF2 = X_dFF2;
unmasked_X_dFF2 = unmasked_X_dFF2(unmasked_indices2, :);

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
%%
new_height1 = size(data1.datOrg1_downsampled_smoothed, 1);
new_width1 = size(data1.datOrg1_downsampled_smoothed, 2);

[A1, C1, info1] = custom_cnmf(X_dFF1, new_height1, new_width1, k_nmf_comp, mask1, opts);

norm(A1(unmasked_indices1, :) * C1 + info1.B(unmasked_indices1, :) * info1.F - unmasked_X_dFF1, 'fro') / norm(unmasked_X_dFF1, 'fro')
% Visualize component k footprint
ethogram_mat1 = data1.ethogram_mat_downsampled;
C1_augmented = [C1; info1.F];
A1_augmented = [A1 info1.B];
plotNMF_withBehaviorOnsets(C1_augmented, ethogram_mat1', 40/data1.temp_down_factor, A1_augmented, [new_height1, new_width1])

%% obtain the z-score map
% for each behavior and each NMF component, we extract the relevant time
% window, and compute z-score for the signals within time-window compared
% to the baseline signals
ethogram_mat1 = ethogram_mat1 > 0.5;
Z1 = nmf_behavior_onset_zscore(C1_augmented, ethogram_mat1, 40/data1.temp_down_factor, 3, 2);  % returns K x 9
Z1(Z1 < 2) = NaN;

figure
h = imagesc(Z1);
set(gca,'Color','k')                  % black background
set(h,'AlphaData', ~isnan(Z1))         % NaNs become transparent -> show black
xlabel("Behavior type", 'FontSize', 12, 'FontWeight','bold');
ylabel("NMF component number", 'FontSize', 12, 'FontWeight','bold');
title("Z-score", 'FontSize', 12, 'FontWeight','bold')
colorbar



%%
new_height2 = size(data2.datOrg1_downsampled_smoothed, 1);
new_width2 = size(data2.datOrg1_downsampled_smoothed, 2);

% build A2 from A1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Here needs correction! Do not run directly
A2_augmented = mapSpatialFeaturesLinear(A1_augmented, mask1, mask2);

%%
A2 = A2_augmented(:,1:end-1);
B2 = A2_augmented(:, end);
[C2, F2, info2] = infer_CF_fixed_AB(X_dFF2, A2, B2, new_height2, new_width2, mask2, opts);

norm(A2(unmasked_indices2, :) * C2 + B2(unmasked_indices2, :) * F2 - unmasked_X_dFF2, 'fro') / norm(unmasked_X_dFF2, 'fro')
% Visualize component k footprint
ethogram_mat2 = data2.ethogram_mat_downsampled;
C2_augmented = [C2; F2];
plotNMF_withBehaviorOnsets(C2_augmented, ethogram_mat2', 40/data2.temp_down_factor, A2_augmented, [new_height2, new_width2])
%% obtain the z-score map
% for each behavior and each NMF component, we extract the relevant time
% window, and compute z-score for the signals within time-window compared
% to the baseline signals
ethogram_mat2 = ethogram_mat2 > 0.5;
Z2 = nmf_behavior_onset_zscore(C2_augmented, ethogram_mat2, 40/data2.temp_down_factor, 3, 2);  % returns K x 9
Z2(Z2 < 2) = NaN;

figure
h = imagesc(Z2);
set(gca,'Color','k')                  % black background
set(h,'AlphaData', ~isnan(Z2))         % NaNs become transparent -> show black
xlabel("Behavior type", 'FontSize', 12, 'FontWeight','bold');
ylabel("NMF component number", 'FontSize', 12, 'FontWeight','bold');
title("Z-score", 'FontSize', 12, 'FontWeight','bold')
colorbar



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
            title(axImg, sprintf('A %d', i), 'FontSize', 9, 'FontWeight', 'bold');

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

        ylabel(axTr, sprintf('A%d', i), "FontSize", 12, "FontWeight", "bold");

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
