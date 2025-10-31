%%
clear
clc
addpath(genpath("C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal"));
%%
enhance_factor = 5;
save_path = fullfile("D:\Mouse_behavior_data\D21", strcat("dual_channel_enhance_factor_", num2str(enhance_factor)));

filePath = "D:\Mouse_behavior_data\D21\Simulated_ethogram_labeling\StimulusScoring.xlsx";
stim_metadata = get_stim_metadata(filePath);
triple_path = fullfile("D:\Mouse_behavior_data\D21", strcat("triple_channel_enhance_factor_", num2str(enhance_factor)));

for video_idx = 1:32
    indexStr = sprintf('%02d', video_idx);
    dual_channel_video = load(fullfile(save_path, strcat('dual_channel_video_', indexStr, '.mat')));
    dual_channel_video = dual_channel_video.dual_channel_video;
    
    stim_onset = stim_metadata.StimOnset(video_idx);
    stim_offset = stim_metadata.StimOffset(video_idx);
    
    [T, H, W, ~] = size(dual_channel_video);
    
    stim_mask = zeros([T, H, W]);
    stim_mask(stim_onset:stim_offset, :, :) = 1;
    % First ensure stim_mask has a singleton 4th dimension to concatenate
    stim_mask_expanded = reshape(stim_mask, size(stim_mask,1), size(stim_mask,2), size(stim_mask,3), 1);
    
    % Concatenate along the 4th dimension
    triple_channel_video = cat(4, dual_channel_video, stim_mask_expanded);
    
    save_file = fullfile(triple_path, strcat('triple_channel_video_', indexStr, '.mat'));
    save(save_file, "triple_channel_video", '-v7.3');
    disp(indexStr)
end

