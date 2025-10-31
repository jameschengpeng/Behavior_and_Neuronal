%% This script computes and stores the dFF map of a calcium image 
clear
clc
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath( 'C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));

%%
save_path = fullfile("D:\Mouse_behavior_data\D21\dFF_and_derivatives");
for video_idx = 2:32
    indexStr = sprintf('%02d', video_idx);
    aqua_result_file = strcat("D:\Mouse_behavior_data\D21\AQuA2\data", indexStr, "_ManualMoCo_cropped_AQuA2.mat");
    aqua_result = load(aqua_result_file);
    fields = fieldnames(aqua_result);
    fieldname = fields{1};
    res_dbg = aqua_result.(fieldname);
    dFF_map = get_dFF_map(res_dbg);
    [~, ~, dFF_deri1] = gradient(dFF_map);
    [~, ~, dFF_deri2] = gradient(dFF_deri1);
    stacked = cat(4, dFF_map, dFF_deri1, dFF_deri2);
    combined = permute(stacked, [3 1 2 4]);
    save(fullfile(save_path, strcat("dFF_", indexStr, ".mat")), "combined", "-v7.3");
    disp(size(combined))
end

%% The function to compute dFF_map
function dFF_map = get_dFF_map(res_dbg)
video = res_dbg.datOrg1;
video = double(video); % convert to float
video = squeeze(video);
% smooth the video spatially and temporally
video = imgaussfilt3(video, [3, 3, 3]);
% compute the baseline for each voxel
voxel_baseline = find_baseline_all_voxel(video);
% compute the dF/F for each voxel given the baseline of them
dFF_map = (video - voxel_baseline) ./ (voxel_baseline + 1e-5);
end
