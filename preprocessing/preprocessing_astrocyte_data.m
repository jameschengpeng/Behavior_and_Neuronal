%% 
clear
clc
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\AQuA2_code'));

%% For astrocyte data
AQuA2_file_suffix = "moco_cropped_AQuA2";

mouse_num = 2;
day = 28;
recording_type = "GGP"; % can be GGC or GGP
AQuA2_result_path = strcat("F:\Astrocyte_data\", recording_type, "#", num2str(mouse_num), "_d", num2str(day));
Ethogram_scoring_path = strcat(AQuA2_result_path, "\EthogramScoring");

stim_scoring_filepath = "F:\Astrocyte_data\GGP Behavior Scoring - Uninjected.xlsx";
preprocessed_storage_path = fullfile(AQuA2_result_path, "preprocessed_data");

%% for each video, save the important information. DO NOT DOWNSAMPLE AT THIS STEP
preprocess_single_video_and_save(AQuA2_result_path, AQuA2_file_suffix, stim_scoring_filepath, preprocessed_storage_path, ...
    recording_type, mouse_num, day);

%% Combine the saved data, downsample sharply
temp_down_factor = 10; 
combine_and_downsample(AQuA2_result_path, 'L', temp_down_factor);
combine_and_downsample(AQuA2_result_path, 'R', temp_down_factor);

%% Helper function: combine the video data from each stim side, downsample sharply
function combine_and_downsample(AQuA2_result_path, stim_side, temp_down_factor)
stim_side_path = fullfile(AQuA2_result_path, "preprocessed_data",  stim_side);
saved_data_files = dir(fullfile(stim_side_path, 'data*.mat'));

adj_dFF_all = [];
evt_all = [];
etho_mat_all = [];
video_lengths = zeros(length(saved_data_files), 1);
noise_var_accum = [];   % weighted accumulator for per-pixel noise variance
total_frames = 0;       % total full-res frames used across videos

for i = 1:length(saved_data_files)
    data_filename = saved_data_files(i).name;
    data_filepath = fullfile(stim_side_path, data_filename);
    saved_data = load(data_filepath);
    datSmo = saved_data.datSmo;
    F0 = saved_data.F0;
    evt_map = saved_data.evt_map;
    condensed_ethogram_mat = saved_data.condensed_ethogram_mat;
    mask_upper_half = saved_data.mask_upper_half;
    mask_lower_half = saved_data.mask_lower_half;

    % Downsample temporally by temp_down_factor
    T = size(datSmo, 3);
    T_ds = floor(T / temp_down_factor);
    T_use = T_ds * temp_down_factor;  % trim trailing frames that don't fill a full bin

    % Continuous signals: average every temp_down_factor frames
    datSmo_ds = reshape(datSmo(:,:,1:T_use), size(datSmo,1), size(datSmo,2), temp_down_factor, T_ds);
    datSmo_ds = squeeze(mean(datSmo_ds, 3));

    F0_ds = reshape(F0(:,:,1:T_use), size(F0,1), size(F0,2), temp_down_factor, T_ds);
    F0_ds = squeeze(mean(F0_ds, 3));

    % Binary maps: max (logical OR) over each bin
    evt_ds = reshape(evt_map(:,:,1:T_use), size(evt_map,1), size(evt_map,2), temp_down_factor, T_ds);
    evt_ds = squeeze(max(evt_ds, [], 3));

    etho_ds = reshape(condensed_ethogram_mat(1:T_use,:), temp_down_factor, T_ds, []);
    etho_ds = squeeze(max(etho_ds, [], 1));  % T_ds x num_behaviors

    % Per-pixel noise variance from full-resolution adj_dFF (MAD on diff).
    % At 26.5 Hz the differencing assumption holds: consecutive frames
    % share nearly the same signal, so diff isolates noise.
    adj_dFF_full = single(datSmo(:,:,1:T_use) ./ max(F0(:,:,1:T_use), eps('single')));
    d = diff(adj_dFF_full, 1, 3);                          % H x W x (T_use-1)
    mad_val = median(abs(d - median(d, 3)), 3);            % H x W
    noise_var_i = (1.4826 * mad_val).^2 / 2;               % H x W
    clear adj_dFF_full d mad_val

    % Weighted accumulation across videos (weight by frame count)
    if isempty(noise_var_accum)
        noise_var_accum = double(noise_var_i) * T_use;
    else
        noise_var_accum = noise_var_accum + double(noise_var_i) * T_use;
    end
    total_frames = total_frames + T_use;

    % Compute adjusted dF/F from downsampled signals: F/F0 (guaranteed positive)
    adj_dFF = single(datSmo_ds ./ max(F0_ds, eps('single')));  % H x W x T_ds

    % Append to containers
    adj_dFF_all = cat(3, adj_dFF_all, adj_dFF);
    evt_all = cat(3, evt_all, evt_ds);
    etho_mat_all = cat(1, etho_mat_all, etho_ds);
    video_lengths(i) = T_ds;
end
% Per-pixel noise variance for the downsampled adj_dFF.
% Averaging temp_down_factor iid noise samples reduces variance by that factor.
noise_var_ds = single(noise_var_accum / total_frames / temp_down_factor);  % H x W

% Save combined data for this stim side
savefile = fullfile(stim_side_path, "data_combined_downsampled.mat");
save(savefile, "adj_dFF_all", "evt_all", "etho_mat_all", "mask_upper_half", ...
    "mask_lower_half", "video_lengths", "noise_var_ds", "-v7.3");
end

%% Helper function: preprocess each video and save useful information 
function preprocess_single_video_and_save(AQuA2_result_path, AQuA2_file_suffix, stim_scoring_filepath, preprocessed_storage_path, ...
    recording_type, mouse_num, day)
% Find all AQuA2 result files with the specified suffix
search_pattern = fullfile(AQuA2_result_path, "*" + AQuA2_file_suffix + ".mat");
aqua_files = dir(search_pattern);

for i = 1:length(aqua_files)
    aqua_filename = aqua_files(i).name;
    aqua_filepath = fullfile(AQuA2_result_path, aqua_filename);

    % Extract data_num from filename, e.g.
    % data01_moco_cropped_AQuA2.mat -> '01'
    tokens = regexp(aqua_filename, '^data(\d+)', 'tokens');

    if isempty(tokens)
        warning('Skipping file "%s" because data number could not be extracted.', aqua_filename);
        continue;
    end

    data_num = tokens{1}{1};

    % Get stimulus side using the previously defined function
    stim_side = extract_stim_side( ...
        stim_scoring_filepath, data_num, recording_type, mouse_num, day);

    % Create/get subfolder path under preprocessed_storage_path
    stim_side_folder_path = fullfile(preprocessed_storage_path, stim_side);

    if ~exist(stim_side_folder_path, 'dir')
        mkdir(stim_side_folder_path);
    end
    savefile = fullfile(stim_side_folder_path, strcat("data", data_num, ".mat"));

    aqua_result = load(aqua_filepath);
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
    ethogram_file = strcat("data", data_num, ".xlsx");
    ethogram_table = readtable(fullfile(AQuA2_result_path, "EthogramScoring", ethogram_file));
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
end

%% Helper function: extract the stim side
function stim_side = extract_stim_side(stim_scoring_filepath, data_num, recording_type, mouse_num, day)
%EXTRACT_STIM_SIDE Extract stimulus side ('L' or 'R') from a scoring Excel file.

    % Build sheet name, e.g. "GGP2_day28"
    sheet_name = sprintf('%s%d_day%d', char(recording_type), mouse_num, day);

    % Read raw cell content
    raw = readcell(stim_scoring_filepath, 'Sheet', sheet_name);

    if size(raw, 1) < 2
        error('Sheet "%s" does not contain the expected two header rows.', sheet_name);
    end

    header_row_1 = raw(1, :);
    header_row_2 = raw(2, :);

    % Forward-fill merged headers in row 1
    filled_header_row_1 = cell(size(header_row_1));
    current_header = "";

    for c = 1:numel(header_row_1)
        val_str = string(header_row_1{c});

        if ismissing(val_str) || strlength(strtrim(val_str)) == 0
            filled_header_row_1{c} = current_header;
        else
            current_header = val_str;
            filled_header_row_1{c} = current_header;
        end
    end

    hdr1 = string(filled_header_row_1);
    hdr2 = string(header_row_2);

    % Find Data column from second header row
    data_col = find(strcmp(strtrim(hdr2), "Data"), 1);
    if isempty(data_col)
        error('Could not find the "Data" column in sheet "%s".', sheet_name);
    end

    % Find Stimulus -> Leg column
    stim_leg_col = find(strcmp(strtrim(hdr1), "Stimulus") & ...
                        strcmp(strtrim(hdr2), "Leg"), 1);
    if isempty(stim_leg_col)
        error('Could not find the column under "Stimulus" -> "Leg" in sheet "%s".', sheet_name);
    end

    % Convert data_num like "09" to numeric 9
    target_data_num = str2double(string(data_num));
    if isnan(target_data_num)
        error('data_num must be convertible to a number, but got: %s', string(data_num));
    end

    % Search rows below the two header rows
    data_values = raw(3:end, data_col);
    row_idx_in_data = [];

    for i = 1:numel(data_values)
        this_num = str2double(string(data_values{i}));
        if ~isnan(this_num) && this_num == target_data_num
            row_idx_in_data = i;
            break;
        end
    end

    if isempty(row_idx_in_data)
        error('Could not find a row with Data = %d in sheet "%s".', target_data_num, sheet_name);
    end

    target_row = row_idx_in_data + 2;

    % Extract stimulus side
    stim_value = string(raw{target_row, stim_leg_col});

    if ismissing(stim_value) || strlength(strtrim(stim_value)) == 0
        error('The stimulus side entry is empty for Data = %d in sheet "%s".', ...
              target_data_num, sheet_name);
    end

    stim_side = char(strtrim(stim_value));

    if ~ismember(stim_side, {'L','R'})
        warning('Extracted stim_side is "%s", not the expected ''L'' or ''R''.', stim_side);
    end
end

