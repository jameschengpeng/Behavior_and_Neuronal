%% 
clear
clc
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));
%%
mouse_num = 2;
day1 = 21;
day2 = 28;
savefile1 = strcat("D:\Astrocyte_data\GGP#", num2str(mouse_num), "_d", num2str(day1), "\downsampled_smoothed_data_all_videos.mat");
savefile2 = strcat("D:\Astrocyte_data\GGP#", num2str(mouse_num), "_d", num2str(day2), "\downsampled_smoothed_data_all_videos.mat");
data1 = load(savefile1);
data2 = load(savefile2);
%%
X_dFF1 = reshape(data1.dFF, [], size(data1.dFF, 3));
X_dFF2 = reshape(data2.dFF, [], size(data2.dFF, 3));

k_nmf_comp = 4; % number of components in NMF

combined_mask = zeros(size(data1.mask_downsampled));
combined_mask(data1.mask_downsampled + data2.mask_downsampled == 2) = 1; % take the intersection of the two masks as active area

unmasked_indices = find(combined_mask);
unmasked_X_dFF1 = X_dFF1; % pixels * time
unmasked_X_dFF1 = unmasked_X_dFF1(unmasked_indices, :);

unmasked_X_dFF2 = X_dFF2;
unmasked_X_dFF2 = unmasked_X_dFF2(unmasked_indices, :);

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

[A, C1, info] = custom_cnmf(X_dFF1, new_height, new_width, k_nmf_comp, combined_mask, opts);

norm(A(unmasked_indices, :) * C1 + info.B(unmasked_indices, :) * info.F - unmasked_X_dFF1) / norm(unmasked_X_dFF1)
% Visualize component k footprint
ethogram_mat = data1.ethogram_mat_downsampled;
plotNMF_withBehaviorOnsets(C1, ethogram_mat', 40/data1.temp_down_factor, A, [new_height, new_width])

figure
imagesc(reshape(info.B(:,1), [new_height, new_width]))

%%


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
    else
        tl = tiledlayout(k, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    end

    traceAxes = gobjects(k,1);

    for i = 1:k

        % ---- Left: spatial component map ----
        if hasSpatial
            axImg = nexttile(tl, (i-1)*2 + 1);
            Ai = reshape(A(:,i), imgSize);
            imagesc(axImg, Ai);
            axis(axImg, 'image');
            axis(axImg, 'off');
            title(axImg, sprintf('A %d', i), 'FontSize', 9, 'FontWeight', 'bold');
            % optional: keep colormap consistent
            colormap(axImg, parula);
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

        if i < k
            set(axTr, 'XTickLabel', []);
        else
            xlabel(axTr, 'Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
        end

        box(axTr, 'off');
    end

    % Single legend using dummy lines (attach to first trace axis, not image axes)
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

