function out = stim_times_by_side_and_type(stim_scoring_table, stim_side, video_lengths)
%STIM_TIMES_BY_SIDE_AND_TYPE Group concatenated frame indices by stim side and type.
%
%   out = stim_times_by_side_and_type(stim_scoring_table, stim_side, video_lengths)
%
% Inputs
%   stim_scoring_table : table from get_stim_metadata
%   stim_side          : stimulus side to keep, e.g. "L" or "R"
%   video_lengths      : length of each concatenated video in frames
%
% Output
%   out.(stimTypeField) contains:
%       - video_indices : cell array, one full-frame index vector per video
%       - StimOnset     : concatenated-frame stimulus onset indices
%       - StimOffset    : concatenated-frame stimulus offset indices

stim_side = string(stim_side);
video_lengths = video_lengths(:);

rows = stim_scoring_table.StimLocation == stim_side;
stim_scoring_table_side = stim_scoring_table(rows, :);

if isempty(stim_scoring_table_side)
    error('No rows found in stim_scoring_table for stim side %s.', char(stim_side));
end

data_nums = cellfun(@(x) sscanf(x, 'data%d'), cellstr(stim_scoring_table_side.Data));
if any(cellfun(@isempty, num2cell(data_nums)))
    error('Could not parse numeric suffix from stim_scoring_table_side.Data.');
end
[~, order] = sort(data_nums);
stim_scoring_table_side = stim_scoring_table_side(order, :);

if height(stim_scoring_table_side) ~= numel(video_lengths)
    error('Number of rows for stim side %s does not match numel(video_lengths).', char(stim_side));
end

video_end_idx = cumsum(video_lengths);
video_start_idx = [1; video_end_idx(1:end-1) + 1];

stim_types = unique(stim_scoring_table_side.StimType, 'stable');
out = struct();

for i = 1:numel(stim_types)
    this_type = stim_types(i);
    idx = find(stim_scoring_table_side.StimType == this_type);

    field_name = matlab.lang.makeValidName(char(this_type));

    video_indices = cell(numel(idx), 1);
    for j = 1:numel(idx)
        k = idx(j);
        video_indices{j} = (video_start_idx(k) : video_end_idx(k)).';
    end

    out.(field_name).video_indices = video_indices;
    out.(field_name).StimOnset = video_start_idx(idx) + stim_scoring_table_side.StimOnset(idx) - 1;
    out.(field_name).StimOffset = video_start_idx(idx) + stim_scoring_table_side.StimOffset(idx) - 1;
end
end