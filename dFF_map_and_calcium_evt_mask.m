%% 
clear
clc
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));
%%
enhance_factor = 5;
for_TIFF = false;
save_path = fullfile("D:\Mouse_behavior_data", strcat("dual_channel_enhance_factor_", num2str(enhance_factor)));
for video_idx = 1:32
    indexStr = sprintf('%02d', video_idx);
    aqua_result_file = strcat("D:\Mouse_behavior_data\D21\AQuA2\data", indexStr, "_ManualMoCo_cropped_AQuA2.mat");
    aqua_result = load(aqua_result_file);
    fields = fieldnames(aqua_result);
    fieldname = fields{1};
    res_dbg = aqua_result.(fieldname);
    dual_channel_video = video_preprocess(res_dbg, enhance_factor, for_TIFF);
    save_file = fullfile(save_path, strcat('dual_channel_video_', indexStr, '.mat'));
    save(save_file, "dual_channel_video", '-v7.3');
    disp(indexStr)
end

%% the function to preprocess the video data. Make it two channel: enhanced dFF map + mask for calcium events
% the larger the enhance_factor, the stronger enhancement to voxels within
% a calcium event (enhance_factor >= 1)
% if you wish to save it for TIFF, set for_TIFF as true
function dual_channel_video = video_preprocess(res_dbg, enhance_factor, for_TIFF)
video = res_dbg.datOrg1; % still 4-dimensional here, with z=1
video = double(video); % convert to float
video = squeeze(video);
% compute the baseline for each voxel
voxel_baseline = find_baseline_all_voxel(video);
% compute the dF/F for each voxel given the baseline of them
dFF_all_voxel = (video - voxel_baseline) ./ (voxel_baseline + 1e-5);
dFF_all_voxel = max(0, dFF_all_voxel);
% compute the map of enhanced dFF. For non-event voxels, keep its dFF. For
% event-involved voxels, multiply its dFF with enhance_factor
evt_enhanced = ones(size(video));
n_evt = numel(res_dbg.evt1);
evt_mask = zeros(size(evt_enhanced));
for ii = 1:n_evt
    evt_voxels = res_dbg.evt1{ii};
    evt_enhanced(evt_voxels) = enhance_factor;
    evt_mask(evt_voxels) = 1;
end
evt_enhanced = evt_enhanced .* dFF_all_voxel;
if for_TIFF
    % just a scaling in case you need to store to TIF
    evt_enhanced = evt_enhanced .* 100; % the dF/F values are at below 0.1 for most voxels
end
% Permute the dimensions to T x H x W
enhanced_reshaped = permute(evt_enhanced, [3, 1, 2]);
mask_reshaped = permute(evt_mask, [3, 1, 2]);

% Concatenate along the 4th dimension to create dual-channel: T x H x W x 2
dual_channel_video = cat(4, enhanced_reshaped, mask_reshaped);
end
