%% concatenate the ethogram matrices
function concat_etho_mat = concat_ethograms(path, subset_cutting_points)
concat_etho_mat = [];
video_lengths = diff(subset_cutting_points);

excelFiles = dir(fullfile(path, "data*.xlsx"));
video_num = {};
for f = 1:numel(excelFiles)
    excelName = string(excelFiles(f).name);
    tok = regexp(excelName, '^data(\d{2})', 'tokens', 'once');
    video_num = [video_num tok];
end

for video_idx = 1:numel(video_num)
    indexStr = video_num{video_idx};
    filename = strcat("data", indexStr, ".xlsx");
    filepath = fullfile(path, filename);
    T = readtable(filepath);
    ethogramMatrix = table2array(T(:, 2:end));
    
    video_len = video_lengths(video_idx);
    if video_len > size(ethogramMatrix, 1)
        d = video_len - size(ethogramMatrix, 1);
        ethogramMatrix = [ethogramMatrix; zeros(d, size(ethogramMatrix, 2))];
    elseif video_len < size(ethogramMatrix, 1)
        ethogramMatrix = ethogramMatrix(1:video_len, :);
    end

    concat_etho_mat = [concat_etho_mat; ethogramMatrix];
end
end