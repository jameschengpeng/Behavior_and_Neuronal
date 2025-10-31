%% obtain the metadata from stimulus scoring 
function T = get_stim_metadata(filePath)
% Read only columns A to E
opts = detectImportOptions(filePath);
opts.SelectedVariableNames = {'Data', 'StimType', 'StimLocation', 'StimOnset', 'StimOffset'};

% Force A, B, C to string, and D, E to double
opts = setvartype(opts, {'Data', 'StimType', 'StimLocation'}, 'string');
opts = setvartype(opts, {'StimOnset', 'StimOffset'}, 'double');

% Read the table
T = readtable(filePath, opts);
% Step 4: Round D and E to nearest integer
T.StimOnset = round(T.StimOnset);
T.StimOffset = round(T.StimOffset);
end