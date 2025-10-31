%% This script computes and stores the dFF map of a calcium image 
clear
clc
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath( 'C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));

%%
save_path = fullfile("D:\Mouse_behavior_data\D21\dF");
for video_idx = 1:32
    indexStr = sprintf('%02d', video_idx);
    aqua_result_file = strcat("D:\Mouse_behavior_data\D21\AQuA2\data", indexStr, "_ManualMoCo_cropped_AQuA2.mat");
    aqua_result = load(aqua_result_file);
    fields = fieldnames(aqua_result);
    fieldname = fields{1};
    res_dbg = aqua_result.(fieldname);
    dF_map = squeeze(res_dbg.dF1);
    dF_map = permute(dF_map, [3 1 2]); % T*H*W
    save(fullfile(save_path, strcat("dF_", indexStr, ".mat")), "dF_map", "-v7.3");
    disp(size(dF_map))
end