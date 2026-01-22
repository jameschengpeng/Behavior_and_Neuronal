%% read all AQuA2 files, concatenate all videos and their dF. It is too slow to write the concatenated video and dF into mat file
function [concat_dF, concat_datOrg, concat_evt_map, subset_cutting_points, early_reaction_regions] = concat_videos(folder)
concat_dF = [];
concat_datOrg = [];
concat_evt_map = [];
subset_cutting_points = []; % if you want a subset of data, here are indices on time axis to cut

for video_idx = 1:32
    indexStr = sprintf('%02d', video_idx);
    aqua_result_file = strcat("data", indexStr, "_ManualMoCo_cropped_AQuA2.mat");
    aqua_result = load(fullfile(folder, aqua_result_file));
    
    res = aqua_result.res;
    dF1 = squeeze(res.dF1);
    
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
    concat_dF = cat(3, concat_dF, dF1);
    concat_datOrg = cat(3, concat_datOrg, datOrg1);
    concat_evt_map = cat(3, concat_evt_map, evt_map);
    subset_cutting_points = [subset_cutting_points; size(datOrg1, 3)];
    fprintf('Processed file %d\n', video_idx);
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


