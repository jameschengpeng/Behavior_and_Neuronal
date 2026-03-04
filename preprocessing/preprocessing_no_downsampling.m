%% 
clear
clc
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\AQuA2_code'));
%%
AQuA2_file_suffix = "ManualMoco_cropped_AQuA2"; % for cck

% For the CCK neuronal data
AQuA2_result_path = "F:\Mouse_behavior_data\D21\AQuA2";
condensed_ethogram_scoring_path = "F:\Mouse_behavior_data\D21\Condensed_EthogramScoring";
preprocessed_storage_path = "F:\Mouse_behavior_data\D21\preprocessed_data";
stim_scoring_filepath = "F:\Mouse_behavior_data\D21\StimulusScoring.xlsx";

%% 
stim_scoring_table = get_stim_metadata(stim_scoring_filepath);
matFiles = dir(fullfile(AQuA2_result_path, "data*.mat"));

for f = 1:numel(matFiles)
    matName = string(matFiles(f).name);
    video_idxStr = regexp(matName, '^data(\d{2})', 'tokens', 'once');
    single_video_preprocess_and_save(video_idxStr, stim_scoring_table,...
        AQuA2_file_suffix, AQuA2_result_path, condensed_ethogram_scoring_path, preprocessed_storage_path);
    fprintf('Processed file %s\n', video_idxStr);
end

%% preprocess one video, and store relevant information in single precision
function single_video_preprocess_and_save(video_idxStr, stim_scoring_table,...
    AQuA2_file_suffix, AQuA2_result_path, condensed_ethogram_scoring_path, preprocessed_storage_path)
% indexStr like '01', '02', ...
rowIdx = stim_scoring_table.Data == ("data" + video_idxStr);   % logical 32x1
StimLocation = stim_scoring_table.StimLocation(rowIdx);        % string (or 1x1)

% create the folderPath for StimLocation left or right
folderPath = fullfile(preprocessed_storage_path, StimLocation);
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end

savefile = fullfile(folderPath, strcat("data", video_idxStr, ".mat"));

% read the AQuA2 result file
aqua_result_file = strcat("data", video_idxStr, "_", AQuA2_file_suffix, ".mat");
fullpath = fullfile(AQuA2_result_path, aqua_result_file);
try
    aqua_result = load(fullpath);
catch
    altFolder = replace(AQuA2_result_path, "F:\", "D:\");
    altFullpath = fullfile(altFolder, aqua_result_file);
    aqua_result = load(altFullpath);
end

% use AQuA2's stored information to obtain the baseline and calculate dF/F
res = aqua_result.res;
smoXY = res.opts.smoXY;
T = size(res.datOrg1, 4); % video length
datSmo = res.datOrg1;
for ii = 1:T % smooth the original data as what was done in AQuA2 processing
    datSmo(:,:,:,ii) = imgaussfilt3(datSmo(:,:,:,ii), smoXY);
end
% Use AQuA2's built-in function to compute the baseline
[F0] = pre.baselineLinearEstimate(datSmo, res.opts.cut, res.opts.movAvgWin);
F0 = squeeze(F0);

% if the file exists, append F0 and finish, to save time
if isfile(savefile)    
    F0 = single(F0);
    save(savefile, "F0", "-append");
    return
end

% The noise in dF1 has already been removed
dF1 = squeeze(res.dF1);
% calculate the dFF and convert to single precision to save space
dFF = single(dF1 ./ F0);
% also convert datSmo to single precision
datSmo = squeeze(single(datSmo));

% obtain the event map
evt_map = zeros(size(res.dF1)); % original 4D
for ii = 1:numel(res.evt1)
    evt_voxels = res.evt1{ii};
    evt_map(evt_voxels) = 1;
end
evt_map = single(squeeze(evt_map));

% read the ethogram matrix
ethogram_file = strcat("data", video_idxStr, ".xlsx");
ethogram_table = readtable(fullfile(condensed_ethogram_scoring_path, ethogram_file));
condensed_ethogram_mat = table2array(ethogram_table(:, 2:end));

if size(condensed_ethogram_mat, 1) < T
    mismatch = T - size(condensed_ethogram_mat, 1);
    condensed_ethogram_mat = [condensed_ethogram_mat; zeros(mismatch, size(condensed_ethogram_mat, 2))];
elseif size(condensed_ethogram_mat, 1) > T
    condensed_ethogram_mat = condensed_ethogram_mat(1:T, :);
end

% read the mask
mask = load(fullfile(AQuA2_result_path, "mask.mat"));
mask = mask.bd0;
upper_unmasked = mask{1}{2};
lower_unmasked = mask{2}{2};
mask_upper_half = zeros(size(dFF, 1), size(dFF, 2));
mask_lower_half = zeros(size(dFF, 1), size(dFF, 2));
mask_upper_half(upper_unmasked) = 1;
mask_lower_half(lower_unmasked) = 1;
mask_upper_half = logical(mask_upper_half);
mask_lower_half = logical(mask_lower_half);

% save to mat file
F0 = single(F0);
save(savefile, "dFF", "datSmo", "F0", "evt_map", ...
    "condensed_ethogram_mat", "mask_upper_half", ...
    "mask_lower_half", "-v7.3");
end






