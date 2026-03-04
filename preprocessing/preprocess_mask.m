%% the function that processes the mask file (in original resolution)
function [mask_upper_half, mask_lower_half] = preprocess_mask(mask_path)
mask = load(mask_path);
bd0 = mask.bd0;

if ~isfield(mask, 'FOV_size')
    FOV_size = struct();
    folder = fileparts(mask_path);
    files = dir(fullfile(folder, '*AQuA2.mat'));
    if isempty(files)
        error('No *AQuA2.mat file found in %s', folder);
    end
    ref_AQuA2_result = fullfile(folder, files(1).name);
    m = load(ref_AQuA2_result);
    % if the variable inside the mat is called "res"
    FOV_size.Height = size(m.res.datOrg1, 1);
    FOV_size.Width = size(m.res.datOrg1, 2);
    save(mask_path, "bd0", "FOV_size"); % save it, next time you can read the size directly from mask.mat
    H = FOV_size.Height;
    W = FOV_size.Width;
else
    H = mask.FOV_size.Height;
    W = mask.FOV_size.Width;
end

mask_upper_half = zeros(H, W);
mask_lower_half = zeros(H, W);

upper_unmasked = bd0{1}{2};
lower_unmasked = bd0{2}{2};

mask_upper_half(upper_unmasked) = 1;
mask_lower_half(lower_unmasked) = 1;
end