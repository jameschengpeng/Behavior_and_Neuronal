function [C, F, info] = infer_CF_fixed_AB(X_dFF, A_full, B_full, H, W, mask, opts)
%INFER_CF_FIXED_AB  Infer temporal components C and F given fixed spatial components A and B.
%
% Model (active pixels only):
%   X ~= A*C + B*F
%
% A: Pm x K  (spatial footprints, FIXED/given, nonnegative)
% C: K  x T  (time courses, nonnegative, TO BE INFERRED)
% B: Pm x r  (background spatial profile, FIXED/given, nonnegative)
% F: r  x T  (background time course, nonnegative, TO BE INFERRED)
%
% Penalties:
%   - Temporal smoothness on C: 0.5*lambdaC_smooth * tr(C DtD C')
%   - Temporal smoothness on F: 0.5*lambdaF_smooth * tr(F DtD F')
%
% Inputs:
%   X_dFF : P x T matrix (can be reshaped time-lapse data)
%   A_full: P x K matrix (pre-fitted spatial components, full image size)
%   B_full: P x r matrix (pre-fitted background spatial, full image size)
%           - If empty [] or not provided, no background is used
%   H, W  : image dimensions (P = H*W)
%   mask  : H x W or P-vector, logical/binary mask for active pixels
%   opts  : struct with options (see set_default_opts)
%
% Outputs:
%   C     : K x T temporal components
%   F     : r x T background temporal components (empty if no background)
%   info  : struct with diagnostics (obj, relchg, relRecon)

if nargin < 7, opts = struct(); end
if nargin < 3 || isempty(B_full)
    B_full = [];
end

opts = set_default_opts(opts);
use_background = ~isempty(B_full);
if isfield(opts, 'use_background')
    use_background = logical(opts.use_background) && ~isempty(B_full);
end

[P, T] = size(X_dFF);
assert(P == H*W, 'X_dFF must have P=H*W rows.');
assert(size(A_full, 1) == P, 'A_full must have P rows.');
if use_background
    assert(size(B_full, 1) == P, 'B_full must have P rows.');
end

mask = reshape(mask, [], 1) ~= 0;
assert(numel(mask) == P, 'mask must be HxW or P-vector.');

K = size(A_full, 2);

% Active pixels only (tissue)
idx = find(mask);
Pm  = numel(idx);
Xraw = X_dFF(idx, :);

% Nonnegativity handling for X
X = Xraw;
if opts.nonneg_mode == "shift"
    X = X - min(X(:));
elseif opts.nonneg_mode == "clip"
    X = max(X, 0);
elseif opts.nonneg_mode == "none"
    % do nothing
else
    error('opts.nonneg_mode must be "none"|"shift"|"clip".');
end

% Extract active pixels from A
Aact = A_full(idx, :);  % Pm x K
assert(all(Aact(:) >= 0), 'A_full must be nonnegative.');

% Extract active pixels from B (if provided)
if use_background
    Bact = B_full(idx, :);  % Pm x r
    assert(all(Bact(:) >= 0), 'B_full must be nonnegative.');
    r = size(Bact, 2);
else
    Bact = zeros(Pm, 0);
    r = 0;
end

% Optional baseline anchoring for F using a low quantile from training phase.
% If enabled, after each F update we shift each row of F so that its chosen
% quantile matches the provided reference quantile baseline.
if use_background && opts.enforce_F_quantile_baseline
    assert(~isempty(opts.F_quantile_ref), ...
        'opts.F_quantile_ref must be provided when enforce_F_quantile_baseline=true.');
    assert(numel(opts.F_quantile_ref) == r, ...
        'numel(opts.F_quantile_ref) must equal number of background rows r.');
end

% Temporal smoothness operator D'D (T x T)
DtD = build_DtD(T);
if isa(X, 'single')
    % Some MATLAB versions do not allow single() cast on sparse matrices.
    % Keep DtD as-is if casting is unsupported.
    try
        DtD = single(DtD);
    catch
        % fallback: leave DtD as sparse double
    end
end

% ---------------------------
% Initialization
% ---------------------------
rng(opts.seed);

% Initialize C using NNLS with small ridge for stability
if use_background
    % Solve: min ||X - A*C - B*F||^2 for C, F jointly (alternating init)
    % Start with F = 0
    F = zeros(r, T);
    AAt = Aact' * Aact + opts.bg_eps * eye(K, 'like', Aact);
    AtX = Aact' * X;
    C = AAt \ AtX;  % K x T
    C = max(C, 0);
    
    % Then update F given C
    BtB = Bact' * Bact + opts.bg_eps * eye(r, 'like', Bact);
    BtR = Bact' * (X - Aact*C);
    F = BtB \ BtR;  % r x T
    F = max(F, 0);
else
    % No background: simple NNLS
    AAt = Aact' * Aact + opts.bg_eps * eye(K, 'like', Aact);
    AtX = Aact' * X;
    C = AAt \ AtX;  % K x T
    C = max(C, 0);
    F = zeros(0, T);
end

% For fixed-A inference, a dead component can be valid (absent in this split).
% Do not randomly reinitialize zero rows of C, to avoid injecting false activity.

info.obj    = zeros(opts.maxIter,1);
info.relchg = zeros(opts.maxIter,1);
info.relRecon = zeros(opts.maxIter,1);

% ---------------------------
% Main loop (optimize C and F only, A and B are FIXED)
% ---------------------------
prevObj = inf;

for it = 1:opts.maxIter

    % ===== Step sizes (with safer spectral-norm estimates)
    if opts.use_adaptive_steps
        % LipC ~ ||A'A||_2 + lambdaC*||DtD||_2 (||DtD||_2 <= 4)
        AAt  = Aact' * Aact;                 % K x K
        LipC = norm(AAt, 2) + opts.lambdaC_smooth * 4;
        etaC = 0.9 / (LipC + eps);
        
        if use_background
            % LipF ~ ||B'B||_2 + lambdaF*||DtD||_2
            BtB = Bact' * Bact;              % r x r
            LipF = norm(BtB, 2) + opts.lambdaF_smooth * 4;
            etaF = 0.9 / (LipF + eps);
        end
    else
        etaC = opts.etaC;
        etaF = opts.etaF;
    end

    % =========================
    % (1) Update C (projected gradient + smoothness)
    % =========================
    for s = 1:opts.innerC
        % grad wrt C: A'*(A*C + B*F - X) + lambdaC*(C*DtD)
        R = (Aact*C + Bact*F) - X;
        gradC = (Aact' * R) + opts.lambdaC_smooth * apply_DtD_right(C, DtD);
        
        Cnew = max(0, C - etaC * gradC);

        if opts.backtracking
            % backtrack if objective increases
            [objOld] = objective_val(X, Aact, C, Bact, F, DtD, opts);
            [objNew] = objective_val(X, Aact, Cnew, Bact, F, DtD, opts);
            bt = 0;
            while objNew > objOld && bt < opts.maxBacktrack
                etaC = etaC * 0.5;
                Cnew = max(0, C - etaC * gradC);
                objNew = objective_val(X, Aact, Cnew, Bact, F, DtD, opts);
                bt = bt + 1;
            end
        end

        C = Cnew;
    end

    % =========================
    % (2) Update F (background temporal, projected gradient + smoothness)
    % =========================
    if use_background
        for s = 1:opts.innerF
            % grad wrt F: B'*(A*C + B*F - X) + lambdaF*(F*DtD)
            R = (Aact*C + Bact*F) - X;
            gradF = (Bact' * R) + opts.lambdaF_smooth * apply_DtD_right(F, DtD);
            
            Fnew = max(0, F - etaF * gradF);

            if opts.backtracking
                [objOld] = objective_val(X, Aact, C, Bact, F, DtD, opts);
                [objNew] = objective_val(X, Aact, C, Bact, Fnew, DtD, opts);
                bt = 0;
                while objNew > objOld && bt < opts.maxBacktrack
                    etaF = etaF * 0.5;
                    Fnew = max(0, F - etaF * gradF);
                    objNew = objective_val(X, Aact, C, Bact, Fnew, DtD, opts);
                    bt = bt + 1;
                end
            end

            F = Fnew;
        end

        % Optional quantile-baseline anchoring (row-wise)
        if opts.enforce_F_quantile_baseline
            q_cur = prctile(F, opts.F_baseline_prctile, 2);      % r x 1
            q_ref = reshape(opts.F_quantile_ref, [], 1);         % r x 1
            delta = cast(q_ref - q_cur, 'like', F);              % r x 1
            F = max(0, F + opts.F_baseline_anchor_strength * (delta * ones(1, T, 'like', F)));
        end
    end

    % ---- Diagnostics
    obj = objective_val(X, Aact, C, Bact, F, DtD, opts);
    info.obj(it) = obj;

    relRecon = norm(X - (Aact*C + Bact*F), 'fro')^2 / (norm(X,'fro')^2 + eps);
    info.relRecon(it) = relRecon;

    if it > 1
        info.relchg(it) = abs(obj - prevObj) / (abs(prevObj) + eps);
    end
    prevObj = obj;

    if opts.verbose && (mod(it, opts.printEvery) == 0 || it == 1)
        fprintf('Iter %4d | obj %.4e | relRecon %.4g | ||C|| %.3e | ||F|| %.3e\n', ...
            it, obj, relRecon, norm(C,'fro'), norm(F,'fro'));
    end

    % stopping
    if it >= opts.minIter && it > 1 && info.relchg(it) < opts.tol
        info.obj = info.obj(1:it);
        info.relchg = info.relchg(1:it);
        info.relRecon = info.relRecon(1:it);
        break;
    end
end

end

%% ======================================================================
% Helpers
% ======================================================================

function opts = set_default_opts(opts)
    def.maxIter = 200;
    def.minIter = 50;
    def.tol = 1e-5;

    def.lambdaC_smooth = 1e-4;
    def.lambdaF_smooth = 1e-2; % background should be very smooth

    % Optional baseline anchoring for F)
    def.enforce_F_quantile_baseline = true; % whether to shift F after each update to keep a chosen quantile anchored
    def.F_baseline_prctile = 5;      % e.g., 5 or 10
    def.F_quantile_ref = [];          % r x 1 reference from training F
    def.F_baseline_anchor_strength = 1; % 0..1 (1 = full correction each iter)

    def.etaC = 1e-3;
    def.etaF = 1e-3;

    def.use_adaptive_steps = true;

    def.innerC = 1;
    def.innerF = 1;  % similar to innerC

    def.seed = 0;
    def.bg_eps = 1e-12;

    % Backtracking safeguard
    def.backtracking = true;
    def.maxBacktrack = 15;

    def.nonneg_mode = "none";  % "none"|"shift"|"clip"

    def.verbose = true;
    def.printEvery = 10;

    f = fieldnames(def);
    for i = 1:numel(f)
        if ~isfield(opts, f{i})
            opts.(f{i}) = def.(f{i});
        end
    end

    % Accept either fraction (0-1) or percentile (0-100)
    if opts.F_baseline_prctile <= 1
        opts.F_baseline_prctile = 100 * opts.F_baseline_prctile;
    end
    opts.F_baseline_prctile = min(100, max(0, opts.F_baseline_prctile));
    opts.F_baseline_anchor_strength = min(1, max(0, opts.F_baseline_anchor_strength));
end

function obj = objective_val(X, A, C, B, F, DtD, opts)
    R = X - (A*C + B*F);
    fitTerm = 0.5 * (norm(R,'fro')^2);

    smoothCTerm = 0;
    if opts.lambdaC_smooth > 0
        CDtD = apply_DtD_right(C, DtD);
        smoothCTerm = 0.5 * opts.lambdaC_smooth * sum(C(:) .* CDtD(:));
    end

    smoothFTerm = 0;
    if opts.lambdaF_smooth > 0 && ~isempty(F)
        FDtD = apply_DtD_right(F, DtD);
        smoothFTerm = 0.5 * opts.lambdaF_smooth * sum(F(:) .* FDtD(:));
    end

    obj = fitTerm + smoothCTerm + smoothFTerm;
end

function DtD = build_DtD(T)
    main = [1; 2*ones(T-2,1); 1];
    off  = -1*ones(T-1,1);
    offL = [off; 0];
    offU = [0; off];
    DtD = spdiags([offL main offU], [-1 0 1], T, T);
end

function Y = apply_DtD_right(X, DtD)
% Compute X * DtD with compatibility for sparse-double DtD and single X.
% Uses explicit 1D second-difference form when sparse-single mtimes is unsupported.

if issparse(DtD) && isa(X, 'single')
    [R, T] = size(X);
    Y = zeros(R, T, 'like', X);

    if T == 1
        Y(:,1) = X(:,1);
        return;
    end

    Y(:,1)   = X(:,1) - X(:,2);
    Y(:,2:T-1) = 2*X(:,2:T-1) - X(:,1:T-2) - X(:,3:T);
    Y(:,T)   = X(:,T) - X(:,T-1);
else
    Y = X * DtD;
end
end