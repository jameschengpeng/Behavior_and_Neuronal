%% read the behavior scoring, transforming to standard form
clear;

filename = "D:\Astrocyte_data\GGP Behavior Scoring - Uninjected.xlsx";
mouse_num = 2;
day = 21;
sheetname = strcat("GGP", num2str(mouse_num), "_day", num2str(day));
AQuA2_folder = strcat("D:\Astrocyte_data\GGP#", num2str(mouse_num), "_d", num2str(day));
ethogram_scoring_folder = strcat("D:\Astrocyte_data\GGP#", num2str(mouse_num), "_d", num2str(day), "\EthogramScoring");
fps_video = 40;
fps_behavior_scoring = 26.5;
behavior_scoring_processing(filename, sheetname, AQuA2_folder, ethogram_scoring_folder, fps_video, fps_behavior_scoring);

%%
function behavior_scoring_processing(filename, sheetname, AQuA2_folder, ethogram_scoring_folder, fps_video, fps_behavior_scoring)
%BEHAVIOR_SCORING_PROCESSING Convert behavior scoring sheet to per-frame ethogram XLSX files.
%
% Inputs
%   filename               - full path to the behavior scoring .xlsx
%   sheetname              - sheet name (e.g., "GGP2_day28")
%   AQuA2_folder           - folder containing AQuA2 mat files named like dataXX*.mat (suffix allowed)
%   ethogram_scoring_folder- output folder for generated dataXX.xlsx
%   fps_video              - fps of original video (output step size is 1/fps_video)
%   fps_behavior_scoring   - fps used for behavior scoring steps (input onset/offset units)
%
% For each MAT file dataXX*.mat:
%   - XX matches the "Data" column in the scoring sheet
%   - output length is the true video length: size(res.datOrg1,4)
%   - parses triplets starting at column "Onset_1" and moving by 3 columns:
%       (Onset_k, Offset_k, Ethogram_k)
%     WITHOUT relying on the specific names of the Ethogram columns.
%   - marks frames onset..offset (after fps conversion) as 1 in Ethogram_A..I
%   - writes ethogram_scoring_folder\dataXX.xlsx
%   - highlights the 1's using Excel conditional formatting (Windows-only)

    arguments
        filename (1,1) string
        sheetname (1,1) string
        AQuA2_folder (1,1) string
        ethogram_scoring_folder (1,1) string
        fps_video (1,1) double {mustBePositive}
        fps_behavior_scoring (1,1) double {mustBePositive}
    end

    if ~isfolder(AQuA2_folder)
        error("AQuA2_folder does not exist: %s", AQuA2_folder);
    end
    if ~isfolder(ethogram_scoring_folder)
        mkdir(ethogram_scoring_folder);
    end

    % ---- Read scoring sheet (robust mixed-type import)
    opts = detectImportOptions(filename, "Sheet", sheetname, "TextType", "string");
    opts.VariableNamingRule = "preserve";

    % Force Stimulus to string if present (prevents NaN coercion)
    stimIdx = find(strcmp(opts.VariableNames, "Stimulus"), 1);
    if ~isempty(stimIdx)
        opts = setvartype(opts, stimIdx, "string");
    end

    T = readtable(filename, opts);
    vnames = string(T.Properties.VariableNames);

    % ---- Locate required columns
    dataCol = find(vnames == "Data", 1);
    if isempty(dataCol)
        error('Could not find a column named "Data" in the sheet.');
    end

    onsetStart = find(vnames == "Onset_1", 1);
    if isempty(onsetStart)
        error('Could not find "Onset_1". Make sure you are reading the correct sheet.');
    end

    % ---- Find mat files dataXX*.mat (suffix allowed)
    matFiles = dir(fullfile(AQuA2_folder, "data*.mat"));
    if isempty(matFiles)
        warning("No mat files matching data*.mat found in: %s", AQuA2_folder);
        return;
    end

    % Output labels Ethogram_A..Ethogram_I (correct)
    ethLabels = compose("Ethogram_%c", 'A':'I');  % 1x9 string array

    % Precompute Data column numeric values for matching
    dataVals = T{:, dataCol};
    if isstring(dataVals) || iscellstr(dataVals)
        dataValsNum = str2double(string(dataVals));
    else
        dataValsNum = double(dataVals);
    end

    for f = 1:numel(matFiles)
        matName = string(matFiles(f).name);

        % Extract the 2-digit number right after 'data'
        tok = regexp(matName, '^data(\d{2})', 'tokens', 'once');
        if isempty(tok)
            continue;
        end
        dataNum = str2double(tok{1});

        % ---- Load MAT and get TRUE number of frames at fps_video
        matPath = fullfile(AQuA2_folder, matFiles(f).name);
        S = load(matPath);

        if ~isfield(S, "res") || ~isfield(S.res, "datOrg1")
            warning("Skipping %s: missing res.datOrg1", matName);
            continue;
        end

        sz = size(S.res.datOrg1);
        if numel(sz) < 4
            warning("Skipping %s: res.datOrg1 is not 4D", matName);
            continue;
        end
        nFramesVideo = sz(4);
        if isempty(nFramesVideo) || nFramesVideo < 1
            warning("Skipping %s: invalid video length from res.datOrg1", matName);
            continue;
        end

        % ---- Match row by Data column
        rowIdx = find(dataValsNum == dataNum, 1);
        if isempty(rowIdx)
            warning("No matching row for data%02d (file %s) in sheet %s.", dataNum, matName, sheetname);
            continue;
        end

        % ---- Initialize output matrix: [Index, Ethogram_A..I]
        outMat = zeros(nFramesVideo, 1 + numel(ethLabels));
        outMat(:,1) = (1:nFramesVideo)';

        % ---- Parse triplets starting at Onset_1 and stepping by 3 columns.
        %      DO NOT rely on the column names beyond finding Onset_1.
        lastTripletStart = numel(vnames) - 2;
        for j = onsetStart:3:lastTripletStart
            onsetBeh  = T{rowIdx, j};
            offsetBeh = T{rowIdx, j+1};
            ethType   = T{rowIdx, j+2};

            % Skip missing triplets
            if any(isnan([onsetBeh, offsetBeh, ethType]))
                continue;
            end

            % Ethogram type should be numeric 1..9
            ethType = double(ethType);
            if ethType < 1 || ethType > 9
                continue;
            end

            % Convert behavior-step indices -> seconds -> video-frame indices
            % Onset/Offset are 1-based step indices at fps_behavior_scoring
            onsetSec  = (double(onsetBeh)  - 1) / fps_behavior_scoring;
            offsetSec = (double(offsetBeh) - 1) / fps_behavior_scoring;

            onsetVid  = floor(onsetSec  * fps_video) + 1;
            offsetVid = ceil(offsetSec * fps_video) + 1;

            % Clamp to true video length and ensure valid order
            onsetVid  = max(1, min(nFramesVideo, onsetVid));
            offsetVid = max(1, min(nFramesVideo, offsetVid));
            if offsetVid < onsetVid
                tmp = onsetVid; onsetVid = offsetVid; offsetVid = tmp;
            end

            % Fill 1's in corresponding ethogram column
            col = 1 + ethType; % 2..10 (since column 1 is Index)
            outMat(onsetVid:offsetVid, col) = 1;
        end

        % ---- Write to Excel
        outT = array2table(outMat, "VariableNames", ["Index", ethLabels]);
        outFile = fullfile(ethogram_scoring_folder, sprintf("data%02d.xlsx", dataNum));
        writetable(outT, outFile, "FileType", "spreadsheet");

        % ---- Highlight 1's (Windows + Excel required)
        try
            highlightOnesInExcel(outFile);
        catch ME
            warning("Wrote %s but could not apply highlighting (Excel automation failed): %s", outFile, ME.message);
        end

        fprintf("Wrote %s (nFrames=%d) from %s\n", outFile, nFramesVideo, matName);
    end
end


function highlightOnesInExcel(xlsxFile)
%HIGHLIGHTONESINEXCEL Apply conditional formatting to highlight cells == 1.
% Windows-only (requires Excel installed). Uses conditional formatting on used range
% excluding header row and excluding Index column.

    Excel = actxserver('Excel.Application');
    Excel.Visible = false;

    % Ensure we always close Excel
    cleaner = onCleanup(@() safeQuitExcel(Excel));

    WB = Excel.Workbooks.Open(xlsxFile);
    WS = WB.Sheets.Item(1);

    usedRange = WS.UsedRange;
    nRows = usedRange.Rows.Count;
    nCols = usedRange.Columns.Count;

    % Data region: row 2..nRows, col 2..nCols (skip headers + Index)
    if nRows < 2 || nCols < 2
        WB.Save();
        WB.Close(false);
        return;
    end

    startCell = WS.Cells(2, 2);
    endCell   = WS.Cells(nRows, nCols);
    rangeObj  = WS.Range(startCell, endCell);

    % Clear existing conditional formats (optional but avoids stacking rules)
    rangeObj.FormatConditions.Delete();

    % Add conditional formatting: cell value == 1
    % xlCellValue = 1, xlEqual = 3  (constants)
    fc = rangeObj.FormatConditions.Add(1, 3, "1");

    % Fill color (yellow)
    fc.Interior.Color = 65535;

    WB.Save();
    WB.Close(false);
end


function safeQuitExcel(Excel)
%SAFEQUITEXCEL Best-effort cleanup for Excel COM server.
    try
        Excel.Quit();
    catch
    end
    try
        delete(Excel);
    catch
    end
end

