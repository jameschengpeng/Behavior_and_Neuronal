function guide_map_2d = compute_guide_map_2d(X_data, H, W, mask, opts)
%COMPUTE_GUIDE_MAP_2D  Build the noise-corrected signal variance guide map.
%
%   guide_map_2d = compute_guide_map_2d(X_data, H, W, mask)
%   guide_map_2d = compute_guide_map_2d(X_data, H, W, mask, opts)

    if nargin < 5
        opts = struct();
    end
    opts = set_default_opts_local(opts);

    [P, ~] = size(X_data);
    assert(P == H * W, 'X_data must have P=H*W rows.');

    mask = reshape(mask, [], 1) ~= 0;
    assert(numel(mask) == P, 'mask must be HxW or P-vector.');

    idx = find(mask);
    Pm = numel(idx);
    X = X_data(idx, :);

    bg_noise_var_act = [];
    if isfield(opts, 'bg_noise_var') && ~isempty(opts.bg_noise_var)
        bg_noise_var_full = reshape(opts.bg_noise_var, [], 1);
        if numel(bg_noise_var_full) == P
            bg_noise_var_act = bg_noise_var_full(idx);
        elseif numel(bg_noise_var_full) == Pm
            bg_noise_var_act = bg_noise_var_full;
        else
            error('opts.bg_noise_var must have length P or number of active pixels Pm.');
        end
    end

    if opts.nonneg_mode == "shift"
        X = X - min(X(:));
    elseif opts.nonneg_mode == "clip"
        X = max(X, 0);
    elseif opts.nonneg_mode ~= "none"
        error('opts.nonneg_mode must be "none"|"shift"|"clip".');
    end

    rng(opts.seed);

    if opts.use_background
        bgopts = struct();
        bgopts.bg_rank = opts.bg_rank;
        bgopts.n_refine = 1;
        bgopts.nonneg_mode = "clip";
        bgopts.eps0 = opts.bg_eps;
        bgopts.bg_noise_var = bg_noise_var_act;
        bgopts.bg_floor_quantile = opts.bg_floor_quantile;
        bgopts.bg_floor_noise_sigma = opts.bg_floor_noise_sigma;
        [B, F] = init_background_lowrank(X, bgopts);
    else
        B = zeros(Pm, 0, 'like', X);
        F = zeros(0, size(X, 2), 'like', X);
    end

    Xr = X - B * F;
    if opts.temporally_downsampled && ~isempty(bg_noise_var_act)
        noise_var_xr = double(bg_noise_var_act);
    else
        noise_var_xr = double(estimate_noise_var_per_pixel(Xr));
    end

    var_xr = var(double(Xr), 0, 2);
    signal_var_act = max(var_xr - noise_var_xr, 0);

    signal_var_full = zeros(H * W, 1);
    signal_var_full(idx) = signal_var_act;
    guide_map_2d = reshape(signal_var_full, H, W);
end

function opts = set_default_opts_local(opts)
    def.use_background = true;
    def.bg_rank = 1;
    def.bg_eps = 1e-12;
    def.bg_floor_quantile = 0.02;
    def.bg_floor_noise_sigma = [];
    def.bg_noise_var = [];
    def.temporally_downsampled = false;
    def.nonneg_mode = "none";
    def.seed = 0;

    f = fieldnames(def);
    for i = 1:numel(f)
        if ~isfield(opts, f{i})
            opts.(f{i}) = def.(f{i});
        end
    end

    if opts.bg_floor_quantile > 1
        opts.bg_floor_quantile = opts.bg_floor_quantile / 100;
    end
    opts.bg_floor_quantile = min(0.49, max(0.001, opts.bg_floor_quantile));
end