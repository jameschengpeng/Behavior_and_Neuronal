%%
clear
clc
addpath(genpath("C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal"));
%%
base_path = "D:\Mouse_behavior_data\D21";
path2dF = "D:\Mouse_behavior_data\D21\dF";
enhance_factor = 5;
path2multi_channel_video = fullfile(base_path, strcat('triple_channel_enhance_factor_', num2str(enhance_factor)));

path2StimScoring = "D:\Mouse_behavior_data\D21\StimulusScoring.xlsx";
stim_metadata = get_stim_metadata(path2StimScoring);
% % Convert StimType to numeric labels starting from 0
% [uniqueTypes, ~, typeIndices] = unique(stim_metadata.StimType);
% stim_metadata.StimType = typeIndices - 1;
% % Convert StimLocation to numeric labels starting from 0
% [uniqueLocations, ~, locationIndices] = unique(stim_metadata.StimLocation);
% stim_metadata.StimLocation = locationIndices - 1;

path2Ethograms = "D:\Mouse_behavior_data\D21\EthogramScoring";
indices = getEthogramIndices(path2Ethograms);

%%
% rate of downsampling
rateT = 4;
rateH = 4;
rateW = 4;

output_h5_file = fullfile(base_path, strcat('input_output_data', '_downsample_', ...
    num2str(rateT), num2str(rateH), num2str(rateW), '.h5'));

% Build a map from 'dataXX' to row in metadata for quick lookup
data_to_row = containers.Map();
for r = 1:height(stim_metadata)
    data_to_row(string(stim_metadata.Data{r})) = r;
end

% Helper: numeric encodings (fixed/stable mapping)
type_to_uint  = containers.Map({'5','8','B','P'}, uint8([0 1 2 3]));
loc_to_uint   = containers.Map({'L','R'},        uint8([0 1]));

%% ====== MAIN LOOP ======================================================
num_videos = length(indices);
for video_idx = 1:num_videos
    indexStr = sprintf('%02d', video_idx);
    dataStr  = "data" + indexStr;

    % ---- Stim row
    if ~isKey(data_to_row, dataStr)
        warning("No metadata row for %s; skipping.", dataStr);
        continue;
    end
    rowIdx = data_to_row(dataStr);

    stim_type_code = string(stim_metadata.StimType(rowIdx));      % "5","8","B","P"
    stim_loc_code  = string(stim_metadata.StimLocation(rowIdx));  % "L","R"
    stim_onset     = double(stim_metadata.StimOnset(rowIdx));
    stim_offset    = double(stim_metadata.StimOffset(rowIdx));

    % Numeric encodings (stable across datasets)
    try
        stim_type_uint = type_to_uint(stim_type_code);
    catch
        error("Unknown StimType code: %s (expected '5','8','B','P')", stim_type_code);
    end
    try
        stim_loc_uint = loc_to_uint(stim_loc_code);
    catch
        error("Unknown StimLocation code: %s (expected 'L','R')", stim_loc_code);
    end

    % Sentence describing stimulation (type + location)
    stim_sentence = buildStimSentence(stim_type_code, stim_loc_code);

    % ---- Load dF map (T×H×W)
    dF_file = fullfile(path2dF, "dF_" + indexStr + ".mat");
    if ~isfile(dF_file)
        warning("Missing dF file: %s; skipping.", dF_file);
        continue;
    end
    S = load(dF_file); % expects variable "dF_map"
    if ~isfield(S, 'dF_map')
        error("File %s does not contain variable 'dF_map'.", dF_file);
    end
    dF_map = S.dF_map; % T×H×W

    % ---- Create pseudo-RGB from dF (t-1, t, t+1) for EfficientNet B0
    % (Ajioka et al. used prev/target/next frames as R/G/B for CNN input.)
    X_rgb = create_pseudo_rgb_from_dF(dF_map);  % T×H×W×3

    % ---- Read ethogram (T×B), pad or trim to match T
    ethogram_file = fullfile(path2Ethograms, "data" + indexStr + ".xlsx");
    eth_table     = readtable(ethogram_file);
    ethogram      = table2array(eth_table(:, 2:end));  % drop first col if it's frame/time
    T = size(X_rgb, 1);
    if size(ethogram, 1) > T
        ethogram = ethogram(1:T, :);
    elseif size(ethogram, 1) < T
        d = T - size(ethogram, 1);
        ethogram = [ethogram; zeros(d, size(ethogram, 2))];
    end

    % ---- Downsample video + ethogram if requested
    [X_rgb_ds, eth_ds] = downsample_video_and_ethogram(X_rgb, ethogram, rateT, rateH, rateW);

    % Downsample stim times (integer frames after downsampling)
    stim_onset_ds  = uint16(round(stim_onset  / rateT));
    stim_offset_ds = uint16(round(stim_offset / rateT));

    % ---- Write to HDF5
    group = "/video_" + indexStr;

    % Video: single precision (T'×H'×W'×3)
    h5create(output_h5_file, group + "/X", size(X_rgb_ds), 'Datatype', 'single');
    h5write(output_h5_file, group + "/X", single(X_rgb_ds));

    % Ethogram: uint8
    h5create(output_h5_file, group + "/y", size(eth_ds), 'Datatype', 'uint8');
    h5write(output_h5_file, group + "/y", uint8(eth_ds));

    % Stim numeric fields (consistent with your prior file structure)
    h5create(output_h5_file, group + "/stim_type",     1, 'Datatype', 'uint8');
    h5create(output_h5_file, group + "/stim_location", 1, 'Datatype', 'uint8');
    h5create(output_h5_file, group + "/stim_onset",    1, 'Datatype', 'uint16');
    h5create(output_h5_file, group + "/stim_offset",   1, 'Datatype', 'uint16');

    h5write(output_h5_file, group + "/stim_type",     uint8(stim_type_uint));
    h5write(output_h5_file, group + "/stim_location", uint8(stim_loc_uint));
    h5write(output_h5_file, group + "/stim_onset",    stim_onset_ds);
    h5write(output_h5_file, group + "/stim_offset",   stim_offset_ds);

    % NEW: human-readable sentence field
    % Try MATLAB string dataset (newer versions). Fallback to uint8 char codes if needed.
    write_string_dataset(output_h5_file, group + "/stim_sentence", stim_sentence);

    fprintf("Written %s\n", group);
end

%% ====== OPTIONAL: quick timing sanity check (fill your file if you have it) ======
sanity_check_h5_timing_consistency(output_h5_file);

%% ======================= LOCAL FUNCTIONS =================================

function X_rgb = create_pseudo_rgb_from_dF(dF_map)
% Input dF_map: T×H×W
% Output X_rgb : T×H×W×3 where channels are [t-1, t, t+1] with edge-repeat.
    [T, H, W] = size(dF_map);

    % Pad first and last frames by replication
    pad_first = dF_map(1, :, :);
    pad_last  = dF_map(end, :, :);
    dF_pad = cat(1, pad_first, dF_map, pad_last);  % (T+2)×H×W

    % Build channels
    chR = dF_pad(1:T,     :, :); % t-1
    chG = dF_pad(2:T+1,   :, :); % t
    chB = dF_pad(3:T+2,   :, :); % t+1

    X_rgb = cat(4, chR, chG, chB); % T×H×W×3
end

function s = buildStimSentence(typeCode, locCode)
% Map codes to natural language (fixed mapping).
    switch char(typeCode)
        case '5', typeStr = 'von Frey 5';
        case '8', typeStr = 'von Frey 8';
        case 'B', typeStr = 'brush';
        case 'P', typeStr = 'pin prick';
        otherwise, error('Unknown stim type code: %s', typeCode);
    end
    switch char(locCode)
        case 'L', locStr = 'ipsilateral hindpaw';
        case 'R', locStr = 'contralateral hindpaw';
        otherwise, error('Unknown stim location code: %s', locCode);
    end
    s = "Stimulation: " + typeStr + " on " + locStr + ".";
end

function write_string_dataset(h5file, dset_path, s)
% Writes a scalar string dataset if supported; otherwise writes uint8 codes.
    try
        % Directly create string dataset
        h5create(h5file, dset_path, 1, 'Datatype', 'string');
        h5write(h5file, dset_path, string(s));
    catch
        % Fallback: fixed-length uint8 char array
        bytes = uint8(char(s));
        dims  = length(bytes);
        h5create(h5file, dset_path, dims, 'Datatype', 'uint8');
        h5write(h5file, dset_path, bytes);
    end
end
