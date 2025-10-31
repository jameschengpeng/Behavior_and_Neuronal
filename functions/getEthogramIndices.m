%% given the path of ethogram files, return the indices of trials available in the folder
function indices = getEthogramIndices(path2Ethograms)
    % Get list of .xlsx files in the directory
    files = dir(fullfile(path2Ethograms, 'data*.xlsx'));
    
    % Initialize array to store indices
    indices = [];
    
    for i = 1:length(files)
        fname = files(i).name;
        
        % Use regular expression to extract number from 'dataXX.xlsx'
        tokens = regexp(fname, '^data(\d+)\.xlsx$', 'tokens');
        
        if ~isempty(tokens)
            index = str2double(tokens{1}{1});
            indices(end+1) = index;
        end
    end
    
    % Sort indices just in case
    indices = sort(indices);
end
