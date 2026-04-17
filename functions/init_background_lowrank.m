function [B, F, info] = init_background_lowrank(X, opts)
%INIT_BACKGROUND_LOWRANK  Initialize low-rank nonnegative background B,F for CNMF/NMF.
%
% Model: X ≈ B*F,  with B>=0, F>=0
%
% For rank-1 (default): B is a spatially uniform (constant) profile, and F
% is estimated as the conservative lower-envelope of X across pixels at
% each time point via fit_rank1_nonovershoot.  This represents the globally
% shared signal (e.g. neuropil) rather than a slowly drifting baseline.
%
% For rank > 1: B columns are initialised via truncated SVD on all frames.
%
% Inputs
%   X    : (Pm x T) double, active-pixel matrix (can be dF/F)
%   opts : struct with fields (optional)
%          - bg_rank   (default 1)    : background rank r
%          - n_refine  (default 1)    : alternating-refinement steps
%          - nonneg_mode (default "clip") : "clip" | "shift" | "none"
%          - eps0      (default 1e-12): small ridge constant
%
% Outputs
%   B    : (Pm x r) background spatial modes
%   F    : (r x T) background temporal modes
%   info : struct with fields:
%          - Xpos      (Pm x T) nonnegative version used for init
%          - recon_err (scalar) ||Xpos - B*F||_F / ||Xpos||_F after init

if nargin < 2, opts = struct(); end
opts = set_defaults_for_background(opts);

[Pm, T] = size(X);
r = opts.bg_rank;

% Nonnegativity handling for background init
Xpos = X;
switch string(opts.nonneg_mode)
    case "clip"
        Xpos = max(Xpos, 0);
    case "shift"
        Xpos = Xpos - min(Xpos(:));
    case "none"
        % leave as-is (not recommended for this helper; B,F are nonneg)
        % but we still clip negatives for stability in init
        Xpos = max(Xpos, 0);
    otherwise
        error('opts.nonneg_mode must be "clip", "shift", or "none".');
end

% Rank-1: uniform spatial profile + conservative lower-envelope F.
% This represents the globally shared signal across all pixels
% (e.g. neuropil synchrony), not a slowly drifting baseline.
if r == 1
    % B elements = 1 (uniform participation), so F is in the same units as X
    % (for dF/F data, F will be ~1; for raw fluorescence F will be ~baseline level).
    B = ones(Pm, 1, 'like', Xpos);
    F = fit_rank1_nonovershoot(B, Xpos, opts); % 1 x T
else
    % SVD-based init on all frames for multi-rank backgrounds.
    % equivalent to initialization by principal components, except we enforce nonnegativity by clipping
    try
        [U,S,~] = svds(Xpos, r);
    catch
        [U,S,~] = svd(Xpos, 'econ');
        U = U(:, 1:r);
        S = S(1:r, 1:r);
    end

    B = max(0, U * sqrt(S));      % Pm x r
    BtB = (B' * B) + opts.eps0 * eye(r, 'like', B);
    F = max(0, (BtB \ (B' * Xpos))); % r x T
end

% Optional alternating refinement for rank > 1 (no-op for rank-1 since B is
% fixed as uniform and fit_rank1_nonovershoot is already called at init).
if r > 1
    for k = 1:opts.n_refine
        BtB = (B' * B) + opts.eps0 * eye(r, 'like', B);
        Ftmp = max(0, (BtB \ (B' * Xpos)));  % r x T
        FFt = (Ftmp * Ftmp') + opts.eps0 * eye(r, 'like', Ftmp);
        B = max(0, (Xpos * Ftmp') / FFt);    % Pm x r
        BtB = (B' * B) + opts.eps0 * eye(r, 'like', B);
        F = max(0, (BtB \ (B' * Xpos)));     % r x T
    end
end

% Normalize columns of B: for rank-1 use column mean so B elements stay ~1
% and F is interpretable in data units; for rank > 1 use L2-norm.
if r == 1
    colnorm = mean(B, 1) + opts.eps0;
else
    colnorm = sqrt(sum(B.^2, 1)) + opts.eps0;
end
B = B ./ colnorm;
F = F .* colnorm';

% Diagnostics (only when caller requests the third output)
if nargout >= 3
    den = norm(Xpos, 'fro') + opts.eps0;
    info = struct();
    info.Xpos = Xpos;
    info.recon_err = norm(Xpos - B*F, 'fro') / den;
else
    info = struct();
end
end

%% -----------------------
% local helper
% -----------------------
function opts = set_defaults_for_background(opts)
def.bg_rank = 1;
def.n_refine = 1;
def.nonneg_mode = "none"; % safest for dF/F init
def.eps0 = 1e-12;
def.bg_floor_quantile = 0.03;
def.bg_floor_noise_sigma = [];
def.bg_refine_profile_from_F = false;
def.bg_profile_quantile = 0.1;
def.bg_profile_min_relF = 0.2;
def.bg_profile_min_frames = 50;
def.bg_profile_smooth_sigma = 20;
def.bg_profile_shrink_uniform = 0.5;
def.bg_profile_n_alternations = 1;

f = fieldnames(def);
for i = 1:numel(f)
    if ~isfield(opts, f{i})
        opts.(f{i}) = def.(f{i});
    end
end

% Guardrails
opts.bg_rank = max(1, round(opts.bg_rank));
opts.n_refine = max(0, round(opts.n_refine));
if opts.bg_floor_quantile > 1
    opts.bg_floor_quantile = opts.bg_floor_quantile / 100;
end
opts.bg_floor_quantile = min(0.49, max(0.001, opts.bg_floor_quantile));
end
