%% 
clear
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));

%%
preprocessed_storage_path = "F:\Mouse_behavior_data\D21\preprocessed_data";
stim_side = "R";
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

% Choose subplot grid layout automatically
n_cols = ceil(sqrt(n_components));
n_rows = ceil(n_components / n_cols);

for i = 1:n_components
    
    subplot(n_rows, n_cols, i)
    
    spatial_map = reshape(A(:, i), H, W);
    
    imagesc(spatial_map);
    axis image off
    colormap(gca, 'parula')
    
    title(['Component ', num2str(i)])
    
end

sgtitle('NMF Spatial Components')

set(gcf, 'Color', 'w')
set(findall(gcf,'-property','FontName'),'FontName','Arial')
set(findall(gcf,'-property','FontSize'),'FontSize',10)

%%
before_onset = 2;
after_onset = 5;
fps = 40;
trace_num = 6;
beh_num = [2 3 4];
aligned_mat = align_trace_and_behavior(C, etho_mat_all, trace_num, beh_num, before_onset, after_onset, fps);

before_frames = round(before_onset * fps);
after_frames  = round(after_onset * fps);

time_vec = (-before_frames:after_frames) / fps;

figure;
imagesc(time_vec, 1:size(aligned_mat,1), aligned_mat);
axis xy
xlabel('Time (s)')
ylabel('Event #')
colorbar

hold on
xline(0, 'r--', 'LineWidth', 1.5);
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