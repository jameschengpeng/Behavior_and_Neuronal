function [B, F] = refine_rank1_background_profile(X, B, F, H, W, mask, opts)
%REFINE_RANK1_BACKGROUND_PROFILE  Refine a rank-1 background spatial profile.
%
% Estimates a per-pixel lower-quantile scaling of X relative to F, then
% smooths the result spatially to keep B broad and field-wide.  Intended
% to run once after init_background_lowrank when opts.bg_refine_profile_from_F
% is true.
%
% Inputs
%   X    : Pm x T active-pixel data matrix
%   B    : Pm x 1 current spatial background profile
%   F    : 1  x T current temporal background trace
%   H, W : image dimensions
%   mask : H*W logical mask (or H x W logical image)
%   opts : struct with background-profile fields:
%          bg_eps, bg_profile_n_alternations, bg_profile_min_relF,
%          bg_profile_min_frames, bg_profile_quantile,
%          bg_profile_smooth_sigma, bg_profile_shrink_uniform,
%          bg_noise_var (optional, Pm x 1): per-pixel noise variance used to
%          correct the downward bias of the lower-quantile B estimate.
%          If absent or wrong size, estimated from X via temporal differencing.
%
% Outputs
%   B : Pm x 1 refined spatial profile (unit norm)
%   F : 1  x T refined temporal trace

if isfield(opts, 'bg_eps')
    eps_bg = opts.bg_eps;
elseif isfield(opts, 'eps0')
    eps_bg = opts.eps0;
else
    eps_bg = 1e-12;
end

if size(B, 2) ~= 1 || isempty(F)
    return;
end

mask = reshape(mask, [], 1) ~= 0;
idx = find(mask);
B = max(B, 0);

% Extract per-pixel noise variance (Pm x 1) for bias correction.
Pm = size(X, 1);
if isfield(opts, 'bg_noise_var') && ~isempty(opts.bg_noise_var)
    noise_var_use = double(reshape(opts.bg_noise_var, [], 1));
    if numel(noise_var_use) ~= Pm
        warning('refine_rank1_background_profile: bg_noise_var length mismatch; estimating from data.');
        noise_var_use = double(estimate_noise_var_per_pixel(X));
    end
else
    noise_var_use = double(estimate_noise_var_per_pixel(X));
end

for it = 1:opts.bg_profile_n_alternations
    f = double(F(1, :));
    fmax = max(f);
    if ~(isfinite(fmax) && fmax > eps_bg)
        break;
    end

    informative = (f >= opts.bg_profile_min_relF * fmax);
    if nnz(informative) < opts.bg_profile_min_frames
        [~, order] = sort(f, 'descend');
        keep_n = min(numel(order), max(opts.bg_profile_min_frames, ceil(0.1 * numel(order))));
        informative = false(size(f));
        informative(order(1:keep_n)) = true;
    end

    f_use = max(f(informative), eps_bg);
    ratio = double(X(:, informative)) ./ reshape(f_use, 1, []);
    b_raw = prctile(ratio, 100 * opts.bg_profile_quantile, 2);
    b_raw = max(b_raw, 0);

    % Noise-bias correction: the q-quantile of (b_p*f_t + eps_p)/f_t undershoots
    % b_p by |Phi^{-1}(q)| * sigma_p / f_bar.  Correct upward per pixel.
    % This mirrors the temporal correction in fit_rank1_nonovershoot.
    if ~isempty(noise_var_use)
        f_bar = mean(f_use) + eps_bg;
        zq = -sqrt(2) * erfcinv(2 * opts.bg_profile_quantile);  % Phi^{-1}(q) < 0
        sigma_p = sqrt(max(noise_var_use, 0));                   % Pm x 1
        b_raw = b_raw + (-zq) * sigma_p / f_bar;                % positive upward shift
    end

    if opts.bg_profile_smooth_sigma > 0
        b_full = zeros(H * W, 1);
        b_full(idx) = b_raw;
        b_img = reshape(b_full, H, W);

        mask_img = reshape(double(mask), H, W);
        num = imgaussfilt(b_img .* mask_img, opts.bg_profile_smooth_sigma);
        den = imgaussfilt(mask_img, opts.bg_profile_smooth_sigma);
        b_img = num ./ max(den, eps_bg);
        b_raw = b_img(mask);
    end

    if opts.bg_profile_shrink_uniform > 0
        b_mean = mean(b_raw);
        b_raw = (1 - opts.bg_profile_shrink_uniform) * b_raw + opts.bg_profile_shrink_uniform * b_mean;
    end

    B = cast(max(b_raw, 0), 'like', X);
    % Normalize by mean so B elements stay ~1 and F remains in data units.
    B = B / (mean(B) + eps_bg);
    F = fit_rank1_nonovershoot(B, X, opts);
end
end
