%% 
clear
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));

%%
% preprocessed_storage_path = "F:\Mouse_behavior_data\D21\preprocessed_data";
preprocessed_storage_path = "F:\CCK_PilotData_Baseline\preprocessed_data";
stim_side = "L";
NMF_result = load(fullfile(preprocessed_storage_path, stim_side, "NMF_result.mat"));
A = NMF_result.A; % n_pixels * (2 * n_components); first half for ipsilateral, second half for contralateral
B = NMF_result.B; % n_pixels * 2; first for ipsilateral background, second for contralateral background
C = NMF_result.C; % (2 * n_components) * T
F = NMF_result.F; % 2 * T
etho_mat_all = NMF_result.etho_mat_all; % T * 4
H = NMF_result.H; % Height
W = NMF_result.W; % Width

%% Show the components in A (not including the background)

[~, n_components] = size(A);

figure;

n_cols = ceil(sqrt(n_components));
n_rows = ceil(n_components / n_cols);

t = tiledlayout(n_rows, n_cols);
t.TileSpacing = 'compact';
t.Padding = 'compact';

for i = 1:n_components
    
    nexttile
    
    spatial_map = reshape(A(:, i), H, W);
    
    imagesc(spatial_map);
    axis image off
    colormap(gca,'parula')
    
    title(['Component ', num2str(i)])

end

sgtitle('NMF Spatial Components')

set(gcf,'Color','w')
set(findall(gcf,'-property','FontName'),'FontName','Arial')
set(findall(gcf,'-property','FontSize'),'FontSize',10)

%%
selected_comps = [1 2 3 4 5 6];
plot_nmf_components_with_ethogram(A(:, selected_comps), C(selected_comps, :), etho_mat_all, 40, H, W, selected_comps)

%%
before_onset = 2;
after_onset = 5;
fps = 40;
trace_num = 1;
beh_num = [1];
% beh_num = 1;
aligned_mat = align_trace_and_behavior(C, etho_mat_all, trace_num, beh_num, before_onset, after_onset, fps);

before_frames = round(before_onset * fps);
after_frames  = round(after_onset * fps);

time_vec = (-before_frames:after_frames) / fps;

figure;
imagesc(time_vec, 1:size(aligned_mat, 1), aligned_mat);
axis xy
xlabel('Time (s)', 'FontSize', 15, 'FontWeight', 'bold')
ylabel('Behavior happening', 'FontSize', 15, 'FontWeight', 'bold')
title(['Compoent: ', num2str(trace_num), '; Behavior: ', num2str(beh_num)], 'FontSize', 15, 'FontWeight', 'bold');
colorbar

hold on
xline(0, 'r--', 'LineWidth', 3);
hold off

%%
function aligned_mat = align_trace_and_behavior(C, etho_mat_all, trace_num, beh_num, before_onset, after_onset, fps)

    % Default frame rate
    if nargin < 7
        fps = 40;
    end

    % Extract chosen trace
    trace = C(trace_num, :);

    % ---- Handle behavior grouping ----
    if numel(beh_num) == 1
        behavior = etho_mat_all(:, beh_num);
    else
        % Logical OR across selected behaviors
        behavior = any(etho_mat_all(:, beh_num), 2);
    end

    behavior = double(behavior);  % ensure numeric (0/1)

    % Convert seconds to frames
    before_frames = round(before_onset * fps);
    after_frames  = round(after_onset * fps);

    % Detect onsets (0 -> 1 transitions)
    behavior_shifted = [0; behavior(1:end-1)];
    onset_indices = find(behavior == 1 & behavior_shifted == 0);

    % Window length
    window_length = before_frames + after_frames + 1;

    % Preallocate
    aligned_mat = zeros(length(onset_indices), window_length);
    valid_count = 0;

    for i = 1:length(onset_indices)

        onset = onset_indices(i);

        start_idx = onset - before_frames;
        end_idx   = onset + after_frames;

        % Boundary check
        if start_idx >= 1 && end_idx <= length(trace)

            valid_count = valid_count + 1;
            aligned_mat(valid_count, :) = trace(start_idx:end_idx);

        end
    end

    % Trim unused rows
    aligned_mat = aligned_mat(1:valid_count, :);

end

%% Plot the components, their time course, and the onset of behaviors
function plot_nmf_components_with_ethogram(A, C, ethogram_matrix, fps, H, W, selected_comps)

% Inputs:
% A: (n_pixels x k)
% C: (k x T)
% ethogram_matrix: (T x n_ethograms)
% fps: frame rate
% H, W: spatial dimensions (H*W = n_pixels)
% selected_comps: a list showing what components to show

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
    title(sprintf('Component %d (%.1f%%)', selected_comps(i), percentage*100));
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
