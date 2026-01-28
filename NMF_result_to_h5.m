%% 
clear
clc
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));
%%
% For all recordings
[dF1, datOrg1, evt_map, subset_cutting_points, early_reaction_regions] = concat_videos("D:\Mouse_behavior_data\D21\AQuA2");

% add 0 to the first element of subset_cutting_points
subset_cutting_points = [0; subset_cutting_points];

concat_ethogram_mat = concat_ethograms("D:\Mouse_behavior_data\D21\EthogramScoring", subset_cutting_points);

video_lengths = diff(subset_cutting_points);

% load the mask
mask = load("D:\Mouse_behavior_data\D21\AQuA2\mask.mat");
mask = mask.bd0;
upper_unmasked = mask{1}{2};
lower_unmasked = mask{2}{2};
mask = zeros(size(dF1, 1), size(dF1, 2));
mask(upper_unmasked) = 1;
mask(lower_unmasked) = 1;

%% Downsampling
% Downsample in spatial and temporal dimensions
spa_down_factor = 2;
temp_down_factor = 4;
[height, width, nframes] = size(dF1);

new_height = floor(height / spa_down_factor);
new_width  = floor(width  / spa_down_factor);
new_frames = sum(floor(video_lengths ./ temp_down_factor)); % addition of all down-sampled video lengths. NOT total frames divided by down-sample factor

dF1_downsampled = zeros(new_height, new_width, new_frames);
datOrg1_downsampled = zeros(new_height, new_width, new_frames);
ethogram_mat_downsampled = zeros(new_frames, size(concat_ethogram_mat, 2));

% Downsample + spatial smoothing each frame. Do it video by video
counter = 0; % record the time index AFTER temp down-sample
for i = 1:numel(video_lengths) % iterate over videos
    start_T = subset_cutting_points(i)+1; % global index BEFORE temp down-sample
    end_T = subset_cutting_points(i+1); % global index BEFORE temp down-sample
    T_down = floor(video_lengths(i) / temp_down_factor); % time length of this video after temp down-sample

    video_dF1 = dF1(:, :, start_T:end_T);
    video_org1 = datOrg1(:, :, start_T:end_T);
    video_ethogram = concat_ethogram_mat(start_T:end_T, :);
    
    % spatially downsample using imresize
    video_dF1_spa_down = imresize(video_dF1, [new_height, new_width], 'bicubic');
    video_org1_spa_down = imresize(video_org1, [new_height, new_width], 'bicubic');

    % temporally downsample by taking average for video and dF1 and also
    % the ethogram. The ethogram becomes a probability
    video_dF1_spa_temp_down = zeros(new_height, new_width, T_down);
    video_org1_spa_temp_down = zeros(new_height, new_width, T_down);
    video_ethogram_temp_down = zeros(T_down, size(concat_ethogram_mat, 2));
    for j = 1:T_down
        substart_T = (j-1) * temp_down_factor + 1; % local index in a video BEFORE temp down-sample
        subend_T = j * temp_down_factor; % local index in a video BEFORE temp down-sample
        video_dF1_spa_temp_down(:, :, j) = mean(video_dF1_spa_down(:, :, substart_T:subend_T), 3);
        video_org1_spa_temp_down(:, :, j) = mean(video_org1_spa_down(:, :, substart_T:subend_T), 3);
        video_ethogram_temp_down(j, :) = mean(video_ethogram(substart_T:subend_T, :), 1);
    end
    
    new_start_T = counter + 1;
    new_end_T = counter + T_down;
    dF1_downsampled(:, :, new_start_T:new_end_T) = video_dF1_spa_temp_down;
    datOrg1_downsampled(:, :, new_start_T:new_end_T) = video_org1_spa_temp_down;
    ethogram_mat_downsampled(new_start_T:new_end_T, :) = video_ethogram_temp_down; 
    
    counter = new_end_T;
end
% delete the intermediate variables to save space
clear video_dF1 video_org1 video_ethogram video_dF1_spa_down video_org1_spa_down video_dF1_spa_temp_down video_org1_spa_temp_down video_ethogram_temp_down

% apply mild smoothing spatially
dF1_downsampled_smoothed = imgaussfilt3(dF1_downsampled, [2, 2, 1e-6]);
datOrg1_downsampled_smoothed = imgaussfilt3(datOrg1_downsampled, [2, 2, 1e-6]);
% -------------------------------------------------------------------------
% Reshape: each row = one frame, each column = one pixel
X_dF = reshape(dF1_downsampled_smoothed, [], new_frames);
X_dF = X_dF.';   % size: T x (H*W)
if min(X_dF(:)) < 0
    X_dF = X_dF - min(X_dF(:));
end

% downsample the mask
mask_downsampled = imresize(mask, [new_height, new_width]);
mask_downsampled(mask_downsampled < 0.5) = 0;
mask_downsampled(mask_downsampled >= 0.5) = 1;

%% compute the baseline for dF/F (for part of videos to avoid out-of-memory)
new_subset_cutting_points = cumsum(floor(video_lengths ./ temp_down_factor)); % update the subset cutting points
new_subset_cutting_points = [0; new_subset_cutting_points]; % include 0 at beginning

n_video_select = 10;
start_video = 11;
select_start = new_subset_cutting_points(start_video); % global, not including starting index of the video, so it can be 0
select_end = new_subset_cutting_points(start_video + n_video_select); % global
voxel_baseline = zeros(size(datOrg1_downsampled_smoothed(:,:,(select_start+1):select_end))); % local 
for ii = 1:n_video_select
    start_ii = new_subset_cutting_points(ii+start_video-1)+1; % global
    end_ii = new_subset_cutting_points(ii+start_video); % global
    single_video_baseline = find_baseline_all_voxel(datOrg1_downsampled_smoothed(:,:,start_ii:end_ii), 40, 8); % global
    voxel_baseline(:,:,(start_ii-select_start):(end_ii-select_start)) = single_video_baseline; % local
end

dFF = (datOrg1_downsampled_smoothed(:,:,(select_start+1):select_end) - voxel_baseline)./voxel_baseline;
X_dFF = reshape(dFF, [], size(dFF, 3));
X_dFF = X_dFF';
if min(X_dFF(:)) < 0
    X_dFF = X_dFF - min(X_dFF(:));  
end

%% Constrained NMF
k_nmf_comp = 8; % number of components in NMF
unmasked_indices = find(mask_downsampled);
unmasked_X_dFF = X_dFF'; % pixels * time
unmasked_X_dFF = unmasked_X_dFF(unmasked_indices, :);

opts = struct();
opts.lambdaA_L1     = 3e-4;
opts.lambdaA_lap    = 1;
opts.lambdaA_excl   = 1e-5;
opts.lambdaC_smooth = 1e-1;

opts.maxIter = 300;
opts.tol = 1e-5;

% step sizes often need tuning
opts.etaA = 1e-3;
opts.etaC = 5e-3;

[A_full, C_full, info] = custom_cnmf(X_dFF', new_height, new_width, k_nmf_comp, mask_downsampled, opts);
%% read the stimulus data, store together with A and C into the h5 file
path2StimScoring = "D:\Mouse_behavior_data\D21\StimulusScoring.xlsx";
stim_metadata = get_stim_metadata(path2StimScoring);

%% append to h5 file
h5file = 'D:\Mouse_behavior_data\D21\NMF_preprocessed_videos.h5';

% ------------ Shared A (write once at root) ------------
A = A_full;                 % (n_pixels x k)
k = size(A,2);

if ~h5exists(h5file, '/A')
    h5create(h5file, '/A', size(A), ...
        'Datatype', class(A), ...
        'ChunkSize', chunkFor(size(A)));
end
h5write(h5file, '/A', A);

% Root metadata
h5writeatt(h5file, '/', 'n_pixels', int32(size(A,1)));
h5writeatt(h5file, '/', 'k',        int32(k));

% ------------ Build global encoding for S ------------
% Unique stim types present in your table (e.g., "B","P",...)
stimTypes = unique(string(stim_metadata.StimType), 'stable');
stimLocs  = ["L","R"];  % fixed set
nType = numel(stimTypes);
nLoc  = numel(stimLocs);
nCond = nType * nLoc;   % channels in S

% Store how channels are ordered
% Channel index = (locIdx-1)*nType + typeIdx
h5writeatt(h5file, '/', 'stim_types_csv',     strjoin(stimTypes, ","));
h5writeatt(h5file, '/', 'stim_locations_csv', strjoin(stimLocs, ","));
h5writeatt(h5file, '/', 'stim_channel_order', ...
    'channel = (locIdx-1)*nType + typeIdx; locIdx: L=1,R=2; typeIdx follows stim_types_csv');

% ------------ Loop videos: write C and S per video ------------
prev_time_steps = new_subset_cutting_points(start_video); % constant

for i = 1:n_video_select
    % --- cut C for this video ---
    video_start_sub = new_subset_cutting_points(start_video + i - 1) - prev_time_steps + 1;
    video_end_sub   = new_subset_cutting_points(start_video + i)     - prev_time_steps;

    C = C_full(:, video_start_sub:video_end_sub);  % (k x T_i)
    Ti = size(C,2);

    grp   = sprintf('/video_%02d', i);
    pathC = [grp '/C'];
    pathS = [grp '/S'];

    % --- Write C ---
    if ~h5exists(h5file, pathC)
        h5create(h5file, pathC, size(C), ...
            'Datatype', class(C), ...
            'ChunkSize', chunkFor(size(C)));
    end
    h5write(h5file, pathC, C);

    % --- Build S for this video from stim_metadata ---
    % Match row by Data name like "data01"
    dataName = sprintf("data%02d", i);
    row = find(string(stim_metadata.Data) == dataName, 1, 'first');

    % Initialize S: (nCond x Ti)
    % Use single to save space (you can use logical too, but single is easy for ML pipelines)
    S = zeros(nCond, Ti, 'single');

    if ~isempty(row)
        stimType = string(stim_metadata.StimType(row));
        stimLoc  = string(stim_metadata.StimLocation(row));

        % Downsample onset/offset (and convert to 1-based indices)
        on  = round(double(stim_metadata.StimOnset(row))  / temp_down_factor);
        off = round(double(stim_metadata.StimOffset(row)) / temp_down_factor);

        % Clamp to [1, Ti]
        on  = max(1, min(Ti, on));
        off = max(1, min(Ti, off));

        % Make sure on <= off
        if off < on
            tmp = on; on = off; off = tmp;
        end

        % Map type/location to channel index
        typeIdx = find(stimTypes == stimType, 1, 'first');
        locIdx  = find(stimLocs  == stimLoc,  1, 'first');

        if ~isempty(typeIdx) && ~isempty(locIdx)
            ch = (locIdx - 1) * nType + typeIdx;
            S(ch, on:off) = 1;
        end

        % Store per-video stim metadata as attributes
        h5writeatt(h5file, grp, 'StimType',      char(stimType));
        h5writeatt(h5file, grp, 'StimLocation',  char(stimLoc));
        h5writeatt(h5file, grp, 'StimOnset_ds',  int32(on));
        h5writeatt(h5file, grp, 'StimOffset_ds', int32(off));
    else
        % No stim row found: still store something explicit
        h5writeatt(h5file, grp, 'StimType',      'NA');
        h5writeatt(h5file, grp, 'StimLocation',  'NA');
        h5writeatt(h5file, grp, 'StimOnset_ds',  int32(-1));
        h5writeatt(h5file, grp, 'StimOffset_ds', int32(-1));
    end

    % --- Write S ---
    if ~h5exists(h5file, pathS)
        h5create(h5file, pathS, size(S), ...
            'Datatype', class(S), ...
            'ChunkSize', chunkFor(size(S)));
    end
    h5write(h5file, pathS, S);

    % --- Standard per-video metadata ---
    h5writeatt(h5file, grp, 'T', int32(Ti));
    h5writeatt(h5file, grp, 'k', int32(size(C,1)));
end

disp('Done.');

%% ---------- helpers ----------
function tf = h5exists(filename, datasetOrGroupPath)
    tf = false;
    if ~isfile(filename), return; end
    try
        h5info(filename, datasetOrGroupPath);
        tf = true;
    catch
        tf = false;
    end
end

function cs = chunkFor(sz)
% Safe chunk sizes for 2D datasets; always <= dataset size.
    if numel(sz) < 2, sz(2) = 1; end
    cs = [min(sz(1), 1024), min(sz(2), 256)];
    cs = max(cs, [1 1]);
end


