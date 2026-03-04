%% if the ethograms are expanded or condensed, use this script to update the ethograms in the saved preprocessed data. It will add the condensed ethogram matrices to the data
%% 
clear
clc
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));
%%
savefile = "F:\Mouse_behavior_data\D21\AQuA2\downsampled_smoothed_data_all_videos.mat";
saved_data = load(savefile);

new_subset_cutting_points       = saved_data.new_subset_cutting_points;
temp_down_factor                = saved_data.temp_down_factor;
%% process the new ethogram
path_to_condensed_ethograms = "F:\Mouse_behavior_data\D21\Condensed_EthogramScoring";
excelFiles = dir(fullfile(path_to_condensed_ethograms, "data*.xlsx"));
video_num = {};

for f = 1:numel(excelFiles)
    excelName = string(excelFiles(f).name);
    tok = regexp(excelName, '^data(\d{2})', 'tokens', 'once');
    video_num = [video_num tok];
end

video_lengths = diff(new_subset_cutting_points);

for ii = 1:numel(video_num)
    indexStr = video_num{ii};
    filename = strcat("data", indexStr, ".xlsx");
    filepath = fullfile(path_to_condensed_ethograms, filename);
    T = readtable(filepath);
    ethogramMatrix = table2array(T(:, 2:end));
    
    if ii == 1
        condensed_ethogram_mat_downsampled = zeros(new_subset_cutting_points(end), size(ethogramMatrix, 2));
    end
    video_len = video_lengths(ii);

    if size(ethogramMatrix, 1) < video_len * temp_down_factor
        mismatch = video_len * temp_down_factor - size(ethogramMatrix, 1);
        ethogramMatrix = [ethogramMatrix; zeros(mismatch, size(ethogramMatrix, 2))];
    elseif size(ethogramMatrix, 1) > video_len * temp_down_factor
        ethogramMatrix = ethogramMatrix(1:video_len * temp_down_factor, :);
    end

    video_ethogram_temp_down = zeros(video_len, size(ethogramMatrix, 2));
    for jj = 1:video_len
        substart_T = (jj-1) * temp_down_factor + 1; % local index in a video BEFORE temp down-sample
        subend_T = jj * temp_down_factor; % local index in a video BEFORE temp down-sample
        video_ethogram_temp_down(jj, :) = mean(ethogramMatrix(substart_T:min(subend_T, size(ethogramMatrix, 1)), :), 1);
    end
    start_T = new_subset_cutting_points(ii)+1;
    end_T = new_subset_cutting_points(ii+1);
    condensed_ethogram_mat_downsampled(start_T:end_T, :) = video_ethogram_temp_down;
    if any(isnan(video_ethogram_temp_down(:)))
        disp(ii)
        disp('isnan')
    end
end
%%
save(savefile, "condensed_ethogram_mat_downsampled", '-append')



