%% read all AQuA2 files, concatenate all videos and their dF. It is too slow to write the concatenated video and dF into mat file
function [concat_dFF, concat_datOrg, concat_evt_map, subset_cutting_points, early_reaction_regions] = concat_videos(folder, file_suffix)
concat_datOrg = [];
concat_evt_map = [];
concat_dFF = [];

matFiles = dir(fullfile(folder, "data*.mat"));
video_num = cell(1, numel(matFiles));
for f = 1:numel(matFiles)
    matName = string(matFiles(f).name);
    tok = regexp(matName, '^data(\d{2})', 'tokens', 'once');
    video_num{f} = tok;
end

subset_cutting_points = zeros(numel(matFiles), 1); % if you want a subset of data, here are indices on time axis to cut

for video_idx = 1:numel(video_num)
    indexStr = video_num{video_idx};
    aqua_result_file = strcat("data", indexStr, "_", file_suffix, ".mat");
    fullpath = fullfile(folder, aqua_result_file);
    try
        aqua_result = load(fullpath);
    catch
        altFolder = replace(folder, "F:\", "D:\");
        altFullpath = fullfile(altFolder, aqua_result_file);
        aqua_result = load(altFullpath);
    end

    res = aqua_result.res;
    smoXY = res.opts.smoXY;
    T = size(res.datOrg1, 4);
    datSmo = res.datOrg1;
    for ii = 1:T % smooth the original data
        datSmo(:,:,:,ii) = imgaussfilt3(datSmo(:,:,:,ii), smoXY);
    end
    % Use AQuA2's built-in function to compute the baseline
    [F0] = pre.baselineLinearEstimate(datSmo, res.opts.cut, res.opts.movAvgWin);
    F0 = squeeze(F0);
    % The noise in dF1 has already been removed
    dF1 = squeeze(res.dF1);
    concat_dFF = cat(3, concat_dFF, dF1./F0);
    
    if video_idx == 1
        early_reaction_regions = zeros([size(res.datOrg1, 1), size(res.datOrg1, 2)]);
    end
    evt_map = zeros(size(dF1));
    for ii = 1:numel(res.evt1)
        evt_voxels = res.evt1{ii};
        evt_map(evt_voxels) = 1;
        early_region_indices = find_early_region_indices(res, ii, 0.1);
        early_reaction_regions(early_region_indices) = early_reaction_regions(early_region_indices) + 1;
    end
    
    datOrg1 = squeeze(res.datOrg1);
    concat_datOrg = cat(3, concat_datOrg, datOrg1);
    concat_evt_map = cat(3, concat_evt_map, evt_map);
    subset_cutting_points(video_idx) = size(datOrg1, 3);
    fprintf('Processed file %s\n', indexStr);
end
subset_cutting_points = cumsum(subset_cutting_points);
end

%% find the early reaction region
function early_region_indices = find_early_region_indices(res, evt_idx, quantile_num)
dlyMap50 = res.riseLst1{evt_idx}.dlyMap50;
upper_left_h = min(res.riseLst1{evt_idx}.rgh);
upper_left_w = min(res.riseLst1{evt_idx}.rgw);

vals = dlyMap50(:);
vals = vals(~isnan(vals));
q = quantile(vals, quantile_num);
idx = find(dlyMap50 < q);

[early_h, early_w] = ind2sub(size(dlyMap50), idx);
early_h = early_h + upper_left_h - 1;
early_w = early_w + upper_left_w - 1;
early_region_indices = sub2ind([size(res.datOrg1, 1), size(res.datOrg1, 2)], early_h, early_w);
end


