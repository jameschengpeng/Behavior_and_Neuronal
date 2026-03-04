%% 
clear
clc
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\AQuA2_code'));
%%
AQuA2_file_suffix = "ManualMoco_cropped_AQuA2"; % for cck
% AQuA2_file_suffix = "moco_cropped_AQuA2"; % for astrocyte

% % For the CCK neuronal data
AQuA2_result_path = "F:\Mouse_behavior_data\D21\AQuA2";
Ethogram_scoring_path = "F:\Mouse_behavior_data\D21\EthogramScoring";

% For the GGP data, various mice, various days
% mouse_num = 2;
% day = 21;
% AQuA2_result_path = strcat("F:\Astrocyte_data\GGP#", num2str(mouse_num), "_d", num2str(day));
% Ethogram_scoring_path = strcat("F:\Astrocyte_data\GGP#", num2str(mouse_num), "_d", num2str(day), "\EthogramScoring");

% For all recordings
% Note that the dFF is obtained from dF1 in AQuA2's results, where the
% baseline F0 is obtained from AQuA2's function pre.baselineLinearEstimate
% by inputing the smoothed image data
[dFF, datOrg1, evt_map, subset_cutting_points, early_reaction_regions] = concat_videos(AQuA2_result_path, AQuA2_file_suffix);

% add 0 to the first element of subset_cutting_points
subset_cutting_points = [0; subset_cutting_points];

concat_ethogram_mat = concat_ethograms(Ethogram_scoring_path, subset_cutting_points);

video_lengths = diff(subset_cutting_points);

% load the mask
mask = load(fullfile(AQuA2_result_path, "mask.mat"));
mask = mask.bd0;
upper_unmasked = mask{1}{2};
lower_unmasked = mask{2}{2};
mask = zeros(size(dFF, 1), size(dFF, 2));
mask(upper_unmasked) = 1;
mask(lower_unmasked) = 1;

%% Downsampling
% Downsample in spatial and temporal dimensions
spa_down_factor = 2;
temp_down_factor = 1; 
[height, width, nframes] = size(dFF);

new_height = floor(height / spa_down_factor);
new_width  = floor(width  / spa_down_factor);
new_frames = sum(floor(video_lengths ./ temp_down_factor)); % addition of all down-sampled video lengths. NOT total frames divided by down-sample factor


dFF_downsampled = zeros(new_height, new_width, new_frames);
datOrg1_downsampled = zeros(new_height, new_width, new_frames);
ethogram_mat_downsampled = zeros(new_frames, size(concat_ethogram_mat, 2));
evt_map_downsampled = zeros(new_height, new_width, new_frames);

% Downsample + spatial smoothing each frame. Do it video by video
counter = 0; % record the time index AFTER temp down-sample
for i = 1:numel(video_lengths) % iterate over videos
    start_T = subset_cutting_points(i)+1; % global index BEFORE temp down-sample
    end_T = subset_cutting_points(i+1); % global index BEFORE temp down-sample
    T_down = floor(video_lengths(i) / temp_down_factor); % time length of this video after temp down-sample

    video_dFF = dFF(:, :, start_T:end_T);
    video_org1 = datOrg1(:, :, start_T:end_T);
    video_ethogram = concat_ethogram_mat(start_T:end_T, :);
    video_evt = evt_map(:, :, start_T:end_T);
    
    % spatially downsample using imresize
    video_dFF_spa_down = imresize(video_dFF, [new_height, new_width], 'bicubic');
    video_org1_spa_down = imresize(video_org1, [new_height, new_width], 'bicubic');
    video_evt_spa_down = imresize(video_evt, [new_height, new_width], 'nearest');

    % temporally downsample by taking average for video and dFF and also
    % the ethogram. The ethogram becomes a probability
    video_dFF_spa_temp_down = zeros(new_height, new_width, T_down);
    video_org1_spa_temp_down = zeros(new_height, new_width, T_down);
    video_ethogram_temp_down = zeros(T_down, size(concat_ethogram_mat, 2));
    video_evt_spa_temp_down = zeros(new_height, new_width, T_down);
    for j = 1:T_down
        substart_T = (j-1) * temp_down_factor + 1; % local index in a video BEFORE temp down-sample
        subend_T = j * temp_down_factor; % local index in a video BEFORE temp down-sample
        video_dFF_spa_temp_down(:, :, j) = mean(video_dFF_spa_down(:, :, substart_T:subend_T), 3);
        video_org1_spa_temp_down(:, :, j) = mean(video_org1_spa_down(:, :, substart_T:subend_T), 3);
        video_ethogram_temp_down(j, :) = mean(video_ethogram(substart_T:subend_T, :), 1);
        video_evt_spa_temp_down(:, :, j) = mean(video_evt_spa_down(:, :, substart_T:subend_T), 3) > 0.5;
    end
    
    new_start_T = counter + 1;
    new_end_T = counter + T_down;
    dFF_downsampled(:, :, new_start_T:new_end_T) = video_dFF_spa_temp_down;
    datOrg1_downsampled(:, :, new_start_T:new_end_T) = video_org1_spa_temp_down;
    ethogram_mat_downsampled(new_start_T:new_end_T, :) = video_ethogram_temp_down; 
    evt_map_downsampled(:, :, new_start_T:new_end_T) = video_evt_spa_temp_down;
    
    counter = new_end_T;
end
% delete the intermediate variables to save space
clear video_dFF video_org1 video_ethogram video_dFF_spa_down video_org1_spa_down video_dFF_spa_temp_down video_org1_spa_temp_down video_ethogram_temp_down
clear video_evt_spa_down video_evt_spa_temp_down evt_map
clear datOrg1 dFF

% apply mild smoothing spatially
dFF_downsampled_smoothed = imgaussfilt3(dFF_downsampled, [2, 2, 1e-6]);
datOrg1_downsampled_smoothed = imgaussfilt3(datOrg1_downsampled, [2, 2, 1e-6]);

clear dFF_downsampled datOrg1_downsampled

% correct the subset cutting points after temporal downsampling
new_subset_cutting_points = cumsum(floor(video_lengths ./ temp_down_factor)); % update the subset cutting points
new_subset_cutting_points = [0; new_subset_cutting_points]; % include 0 at beginning

% downsample the mask
mask_downsampled = imresize(mask, [new_height, new_width]);
mask_downsampled(mask_downsampled < 0.5) = 0;
mask_downsampled(mask_downsampled >= 0.5) = 1;


%% compute the baseline for dF/F
% save preprocessed dF and datOrg
savefile = "F:\Mouse_behavior_data\D21\AQuA2\downsampled_smoothed_data_all_videos.mat";
save(savefile, "dFF_downsampled_smoothed", "datOrg1_downsampled_smoothed", ...
    "ethogram_mat_downsampled", "new_subset_cutting_points", "evt_map_downsampled", ...
    "mask_downsampled", "temp_down_factor", "-v7.3");
%%

savefile = strcat("F:\Astrocyte_data\GGP#", num2str(mouse_num), "_d", num2str(day), "\downsampled_smoothed_data_all_videos.mat");
save(savefile, "dFF_downsampled_smoothed", "datOrg1_downsampled_smoothed", ...
    "ethogram_mat_downsampled", "new_subset_cutting_points", "evt_map_downsampled", ...
    "mask_downsampled", "temp_down_factor", "-v7.3");


