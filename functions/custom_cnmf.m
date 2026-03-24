function [A, C, info] = custom_cnmf(X_data, H, W, K, mask, evt_domain_projection, opts)
%CUSTOM_CNMF  CNMF-like factorization with OPTIONAL rank-1 background.
%
% Model (active pixels only):
%   X ~= A*C + B*F
%
% A: Pm x K  (spatial footprints, nonnegative)
% C: K  x T  (time courses, nonnegative)
% B: Pm x r  (background/global spatial profile, nonnegative)
% F: r  x T  (background/global time course, nonnegative)
%
% Penalties (optional):
%   - L1(A) sparsity (prox)
%   - Laplacian smoothness on A: 0.5*lambdaA_lap * tr(A' L A)
%   - compactness on A (discourage scattered support)
%   - exclusivity on A (pixelwise overlap)
%   - temporal smoothness on C: 0.5*lambdaC_smooth * tr(C DtD C')
%
% Inputs unchanged from your original implementation.

if nargin < 7, opts = struct(); end
opts = set_default_opts(opts);

[P, T] = size(X_data);
assert(P == H*W, 'X_data must have P=H*W rows.');
mask = reshape(mask, [], 1) ~= 0;
assert(numel(mask) == P, 'mask must be HxW or P-vector.');

% Active pixels only (tissue)
idx = find(mask);
Pm  = numel(idx);
Xraw = double(X_data(idx, :));

% use the AQuA2 detected events' projection as a guide to CNMF
guide_full = reshape(evt_domain_projection, [], 1);
guide_act  = double(guide_full(idx));
guide_act  = guide_act / (max(guide_act) + eps);  % normalize to [0,1]

% Nonnegativity handling for X (IMPORTANT: avoid double-shifting)
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

% Build Laplacian on ACTIVE pixel graph
L = build_laplacian_active(H, W, mask, opts.neighborhood);

% Active-pixel coordinates for compactness penalty on A
[row_act, col_act] = ind2sub([H, W], idx);
row_act = double(row_act);
col_act = double(col_act);
if opts.compact_norm_coords
    row_act = (row_act - 1) / max(1, H - 1);
    col_act = (col_act - 1) / max(1, W - 1);
end

% Temporal smoothness operator D'D (T x T)
DtD = build_DtD(T);

% ---------------------------
% Initialization
% ---------------------------
rng(opts.seed);

% initialize background first (if use_background)
r = opts.bg_rank;

if opts.use_background
    if opts.bg_init_mode == "pc1_global"
        [B, F] = init_background_pc1_global(X, r, opts);
    else
        bgopts = struct();
        bgopts.bg_rank = r;
        bgopts.use_quiet_init = true;
        bgopts.quiet_prctile = opts.bg_quiet_prctile;
        bgopts.n_refine = 1;
        bgopts.nonneg_mode = "clip";
        bgopts.eps0 = opts.bg_eps;
        [B, F] = init_background_lowrank(X, bgopts);   % B: Pm x r, F: r x T
    end
else
    B = zeros(Pm, 0);
    F = zeros(0, T);
end

% Optional baseline anchoring for F during training.
% If no reference is provided, use the initialized F quantile as reference.
if opts.use_background && opts.enforce_F_quantile_baseline
    if isempty(opts.F_quantile_ref)
        F_quantile_ref_train = prctile(F, opts.F_baseline_prctile, 2);
    else
        assert(numel(opts.F_quantile_ref) == r, ...
            'numel(opts.F_quantile_ref) must equal background rank r.');
        F_quantile_ref_train = reshape(opts.F_quantile_ref, [], 1);
    end
else
    F_quantile_ref_train = [];
end


% Residual for component init
Xr = X - B*F;
Xr = max(Xr, 0);  % keep nonnegativity
[Aact, C, init_info] = init_AC_from_event_projection(Xr, evt_domain_projection, reshape(mask, H, W), K, opts);

% this block is for checking the initialization of spatial footprints
A_init = zeros(size(mask, 1), K);
A_init(idx, :) = Aact;
A_init_comb = reshape(sum(A_init, 2), [H, W]);

% Reinit any dead components
for k = 1:K
    if all(Aact(:,k) == 0) || all(C(k,:) == 0)
        Aact(:,k) = max(0, rand(Pm,1));
        C(k,:)    = max(0, rand(1,T));
    end
end

% Optional normalization (do NOT do every iter by default)
if opts.doNormalize
    [Aact, C] = normalize_factors(Aact, C, opts);
end

info.obj    = zeros(opts.maxIter,1);
info.relchg = zeros(opts.maxIter,1);
info.relRecon = zeros(opts.maxIter,1);
info.init = init_info;

% ---------------------------
% Main loop
% ---------------------------
prevObj = inf;

for it = 1:opts.maxIter

    % ---- Compute residual with current background
    Xhat = Aact*C + B*F;
    R = Xhat - X;

    % ===== Step sizes (with safer spectral-norm estimates)
    if opts.use_adaptive_steps
        % LipC ~ ||A'A||_2 + lambdaC*||DtD||_2 (||DtD||_2 <= 4)
        AAt  = Aact' * Aact;                 % K x K
        LipC = norm(AAt, 2) + opts.lambdaC_smooth * 4;
        etaC = 0.9 / (LipC + eps);

        % LipA ~ ||CC'||_2 + lambdaA*||L||_2 + excl
        CCt  = C * C';
        LipA = norm(CCt, 2);
        if opts.lambdaA_lap > 0
            dmax = opts.neighborhood;     % 4 or 8
            LipL = 2 * dmax;              % safe bound for ||L||_2
            LipA = LipA + opts.lambdaA_lap * LipL;
        end
        if opts.lambdaA_excl > 0
            LipA = LipA + opts.lambdaA_excl * K;
        end
        if opts.lambdaA_compact > 0
            % conservative bound: add compactness contribution scale
            LipA = LipA + opts.lambdaA_compact * 2;
        end
        etaA = 0.9 / (LipA + eps);
    else
        etaA = opts.etaA;
        etaC = opts.etaC;
    end

    % =========================
    % (1) Update C (projected gradient + smoothness)
    % =========================
    for s = 1:opts.innerC
        % grad wrt C: A'*(A*C + B*F - X) + lambdaC*(C*DtD)
        R = (Aact*C + B*F) - X;
        gradC = (Aact' * R) + opts.lambdaC_smooth * (C * DtD);

        Cnew = max(0, C - etaC * gradC);

        if opts.backtracking
            % backtrack if objective increases
            [objOld] = objective_val(X, Aact, C, B, F, L, DtD, opts, guide_act, row_act, col_act);
            [objNew] = objective_val(X, Aact, Cnew, B, F, L, DtD, opts, guide_act, row_act, col_act);
            bt = 0;
            while objNew > objOld && bt < opts.maxBacktrack
                etaC = etaC * 0.5;
                Cnew = max(0, C - etaC * gradC);
                objNew = objective_val(X, Aact, Cnew, B, F, L, DtD, opts, guide_act, row_act, col_act);
                bt = bt + 1;
            end
        end

        C = Cnew;
    end

    % =========================
    % (2) Update background/global term
    % =========================
    if opts.use_background
        R0 = X - Aact*C;
        R0pos = max(R0, 0);

        if opts.update_background
            % ---- Optional B update on ALL frames using current F
            G   = (F * F');
            RHS = (R0pos * F');

            % Lipschitz bound for B: ||F*F'||_2 + lambdaB*||L||_2
            LipL = 2 * opts.neighborhood;
            LipB = norm(G, 2) + opts.lambdaB_lap * LipL;
            etaB = 0.9 / (LipB + eps);

            for sB = 1:opts.innerB
                gradB = (B * G - RHS) + opts.lambdaB_lap * (L * B);
                B = max(0, B - etaB * gradB);
            end
        end

        if opts.update_F
            % ---- Update F on ALL frames using full residual (with B fixed)
            BtB = (B' * B) + opts.bg_eps * eye(r);
            BR  = (B' * R0pos);
            use_nonovershoot_F = opts.enforce_background_nonovershoot && r == 1;
            if use_nonovershoot_F
                F_cap = fit_rank1_nonovershoot(B, R0, opts);
            end

            if opts.lambdaF_smooth > 0 && ~use_nonovershoot_F
                % Solve: min_F 0.5||R0pos - B*F||_F^2 + 0.5*lambdaF*tr(F*DtD*F')
                % via projected-gradient (nonnegative F)
                LipF = norm(BtB, 2) + opts.lambdaF_smooth * 4;
                etaF = 0.9 / (LipF + eps);

                for sF = 1:opts.innerF
                    gradF = (BtB * F - BR) + opts.lambdaF_smooth * (F * DtD);
                    F = max(0, F - etaF * gradF);
                end
            elseif use_nonovershoot_F
                % In conservative rank-1 mode, use a noise-corrected lower-tail
                % estimate of the shared floor instead of an average-error fit.
                F = F_cap;
            else
                % Closed-form NNLS proxy when no temporal smoothing is requested
                Lb = chol(BtB, 'lower');
                F = Lb'\(Lb\BR);
                F = max(F, 0);
            end
        else
            use_nonovershoot_F = false;
        end
    
        % Optional stabilization: normalize columns of B, rescale F
        if opts.update_background
            coln = sqrt(sum(B.^2,1)) + opts.bg_eps;
            B = B ./ coln;
            F = F .* coln';
        end

        % Optional quantile-baseline anchoring (row-wise) to prevent F drift.
        if opts.update_F && opts.enforce_F_quantile_baseline && ~isempty(F_quantile_ref_train)
            q_cur = prctile(F, opts.F_baseline_prctile, 2);      % r x 1
            delta = cast(F_quantile_ref_train - q_cur, 'like', F);
            F = max(0, F + opts.F_baseline_anchor_strength * (delta * ones(1, T, 'like', F)));
        end

        if opts.update_F && use_nonovershoot_F && ~isempty(F)
            F = min(F, F_cap);
        end
    end

    % =========================
    % (3) Update A (prox-grad: fit + lap + excl, then L1+nonneg prox)
    % =========================
    for s = 1:opts.innerA
        R = (Aact*C + B*F) - X;          % Pm x T
        gradA = (R * C');                % Pm x K

        if opts.lambdaA_lap > 0
            gradA = gradA + opts.lambdaA_lap * (L * Aact);
        end

        if opts.lambdaA_excl > 0
            sumA = sum(Aact, 2);
            gradExcl = (sumA * ones(1,K)) - Aact;
            gradA = gradA + opts.lambdaA_excl * gradExcl;
        end

        % ---- Compactness penalty (discourage scattered A support)
        if opts.lambdaA_compact > 0
            [dist2_compact, ~] = compactness_dist2_and_term(Aact, row_act, col_act, opts.compact_eps);
            gradA = gradA + opts.lambdaA_compact * dist2_compact;
        end

        % ---- Guide penalty (match sum of A to AQuA2 projection)
        if opts.lambdaA_guide > 0
            Aact_sum = sum(Aact, 2);                 % Pm x 1
            g = guide_act;                    % Pm x 1 (already masked)
            
            if opts.use_guide_scale
                beta = (g' * Aact_sum) / (g' * g + opts.guide_eps);
            else
                beta = 1;
            end
            
            gradGuide = opts.lambdaA_guide * ((Aact_sum - beta * g) * ones(1,K));
            
            gradA = gradA + gradGuide;
        end


        Anew = Aact - etaA * gradA;

        if opts.lambdaA_L1 > 0
            Anew = soft_thresh_nonneg(Anew, etaA * opts.lambdaA_L1);
        else
            Anew = max(Anew, 0);
        end

        if opts.backtracking
            objOld = objective_val(X, Aact, C, B, F, L, DtD, opts, guide_act, row_act, col_act);
            objNew = objective_val(X, Anew, C, B, F, L, DtD, opts, guide_act, row_act, col_act);
            bt = 0;
            while objNew > objOld && bt < opts.maxBacktrack
                etaA = etaA * 0.5;
                Anew = Aact - etaA * gradA;
                if opts.lambdaA_L1 > 0
                    Anew = soft_thresh_nonneg(Anew, etaA * opts.lambdaA_L1);
                else
                    Anew = max(Anew, 0);
                end
                objNew = objective_val(X, Anew, C, B, F, L, DtD, opts, guide_act, row_act, col_act);
                bt = bt + 1;
            end
        end

        Aact = Anew;
    end

    % Optional normalization (do rarely)
    if opts.doNormalize && mod(it, opts.normalizeEvery) == 0
        [Aact, C] = normalize_factors(Aact, C, opts);
    end

    % ---- Diagnostics
    obj = objective_val(X, Aact, C, B, F, L, DtD, opts, guide_act, row_act, col_act);
    info.obj(it) = obj;

    relRecon = norm(X - (Aact*C + B*F), 'fro')^2 / (norm(X,'fro')^2 + eps);
    info.relRecon(it) = relRecon;

    if it > 1
        info.relchg(it) = abs(obj - prevObj) / (abs(prevObj) + eps);
    end
    prevObj = obj;

    % Print the loss, the norms of A and C, and the contributions of
    % different terms to the gradient
    if opts.verbose && (mod(it, opts.printEvery) == 0 || it == 1)
    
        % =========================
        % ---- A diagnostics ----
        % =========================
        Rtmp = (Aact*C + B*F) - X;
    
        % Fit gradient
        gradA_fit = (Rtmp * C');
    
        % Laplacian gradient
        gradA_lap = zeros(size(Aact));
        if opts.lambdaA_lap > 0
            gradA_lap = opts.lambdaA_lap * (L * Aact);
        end
    
        % Exclusivity gradient
        gradA_excl = zeros(size(Aact));
        if opts.lambdaA_excl > 0
            sumA = sum(Aact, 2);
            gradA_excl = opts.lambdaA_excl * ((sumA * ones(1,size(Aact,2))) - Aact);
        end

        gradA_compact = zeros(size(Aact));
        if opts.lambdaA_compact > 0
            [dist2_compact, compact_obj] = compactness_dist2_and_term(Aact, row_act, col_act, opts.compact_eps);
            gradA_compact = opts.lambdaA_compact * dist2_compact;
        else
            compact_obj = 0;
        end
        compact_grad_ratio = norm(gradA_compact,'fro') / (norm(gradA_fit,'fro') + eps);
    
        % L1 diagnostics
        tau = etaA * opts.lambdaA_L1;
        l1_obj = opts.lambdaA_L1 * sum(Aact(:));

        % Sparsity diagnostics per component (column of A)
        nnzA_col = sum(Aact > 0, 1);                              % 1 x K
        pctA_col = 100 * (nnzA_col / max(1, size(Aact,1)));       % 1 x K
        nnzA_col_str = strtrim(sprintf('%d ', nnzA_col));
        pctA_col_str = strtrim(sprintf('%.2f%% ', pctA_col));
    
        Atemp = Aact - etaA * (gradA_fit + gradA_lap + gradA_excl + gradA_compact);
        if opts.lambdaA_L1 > 0
            Aprox = soft_thresh_nonneg(Atemp, tau);
        else
            Aprox = max(Atemp, 0);
        end
        l1_shrink = norm(Atemp - Aprox, 'fro');
    
        % =========================
        % ---- C diagnostics ----
        % =========================
        gradC_fit = Aact' * Rtmp;
    
        gradC_smooth = zeros(size(C));
        if opts.lambdaC_smooth > 0
            gradC_smooth = opts.lambdaC_smooth * (C * DtD);
        end
    
        % =========================
        % ---- F diagnostics ----
        % =========================
        gradF_fit = zeros(size(F));
        gradF_smooth = zeros(size(F));
        if ~isempty(F)
            gradF_fit = B' * Rtmp;
            if opts.lambdaF_smooth > 0
                gradF_smooth = opts.lambdaF_smooth * (F * DtD);
            end
        end
    
        % AQuA2 guide diagnostics
        grad_guide = zeros(size(Aact));
        guide_obj = 0;
        
        if opts.lambdaA_guide > 0
            Aact_sum = sum(Aact,2);
            g = guide_act;
            if opts.use_guide_scale
                beta = (g' * Aact_sum) / (g' * g + opts.guide_eps);
            else
                beta = 1;
            end
            remainder = Aact_sum - beta*g;
            grad_guide = opts.lambdaA_guide * (remainder * ones(1,K));
            guide_obj = 0.5 * opts.lambdaA_guide * (remainder' * remainder);
        end

        % =========================
        % ---- Print ----
        % =========================
        fprintf(['Iter %4d | obj %.4e | relRecon %.4g | ||A|| %.3e nnzA %d | ||C|| %.3e | bg %d\n' ...
             '   A-grad:  fit %.3e | lap %.3e | excl %.3e | compact %.3e (ratio %.3f) | L1obj %.3e | prox %.3e | tau %.3e\n' ...
                 '   A-nnz:   count/col [%s]\n' ...
                 '   A-nnz%%:  pct/col   [%s]\n' ...
                 '   C-grad:  fit %.3e | smooth %.3e\n' ...
                 '   F-grad:  fit %.3e | smooth %.3e\n' ...
                 '   Guide:   %.3e | CompactObj(raw) %.3e\n'], ...
                it, obj, relRecon, norm(Aact,'fro'), nnz(Aact), norm(C,'fro'), opts.use_background, ...
            norm(gradA_fit,'fro'), norm(gradA_lap,'fro'), norm(gradA_excl,'fro'), norm(gradA_compact,'fro'), ...
                compact_grad_ratio, l1_obj, l1_shrink, tau, ...
                nnzA_col_str, pctA_col_str, ...
                norm(gradC_fit,'fro'), norm(gradC_smooth,'fro'), ...
                norm(gradF_fit,'fro'), norm(gradF_smooth,'fro'), ...
                norm(grad_guide,'fro'), compact_obj);

        disp('-----------')
    end

    % stopping
    if it >= opts.minIter && it > 1 && info.relchg(it) < opts.tol
        info.obj = info.obj(1:it);
        info.relchg = info.relchg(1:it);
        info.relRecon = info.relRecon(1:it);
        break;
    end
end

% Put back into full P x K with masked pixels = 0
A = zeros(P, K);
A(idx, :) = Aact;

% Save background in info (doesn't change function signature)
info.B = zeros(P,size(B,2));
info.B(idx, :) = B;
info.F = F;

end

%% ======================================================================
% Helpers
% ======================================================================
function [B, F, info] = init_background_lowrank(X, opts)
%INIT_BACKGROUND_LOWRANK  Initialize low-rank nonnegative background B,F for CNMF/NMF.
%
% Model: X ≈ B*F,  with B>=0, F>=0
%
% Inputs
%   X    : (Pm x T) double, active-pixel matrix (can be dF/F)
%   opts : struct with fields (optional)
%          - bg_rank        (default 3)   : background rank r
%          - use_quiet_init (default true): use "quiet frames" for init SVD
%          - quiet_prctile  (default 20)  : percentile for quiet frames (bottom p%)
%          - n_refine       (default 1)   : number of alternating refinement steps
%          - nonneg_mode    (default "clip") : "clip" or "shift" or "none"
%          - eps0           (default 1e-12)
%
% Outputs
%   B    : (Pm x r) background spatial modes
%   F    : (r x T) background temporal modes
%   info : struct with fields:
%          - quiet_idx (1 x T logical)
%          - Xpos      (Pm x T) nonnegative version used for init
%          - recon_err (scalar) ||Xpos - B*F||_F / ||Xpos||_F after init

if nargin < 2, opts = struct(); end
opts = set_defaults_for_background(opts);

[Pm, T] = size(X);
r = opts.bg_rank;

% Nonnegativity handling for background init
Xpos = double(X);
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

% Quiet frames selection
quiet_idx = true(1, T);
if opts.use_quiet_init
    g = mean(Xpos, 1); % 1 x T
    thr = prctile(g, opts.quiet_prctile);
    quiet_idx = (g <= thr);
    % Ensure we have enough frames
    if nnz(quiet_idx) < max(10, 2*r)
        quiet_idx = true(1, T);
    end
end

Xq = Xpos(:, quiet_idx);

% For rank-1 background, use a uniform spatial profile to represent a
% biologically shared field-wide response rather than a data-adaptive mode.
if r == 1
    B = ones(Pm, 1);
    B = B / (norm(B) + opts.eps0);
    F = fit_rank1_nonovershoot(B, Xpos, opts); % 1 x T
else
    % SVD-based init on quiet frames for multi-rank backgrounds.
    try
        [U,S,~] = svds(Xq, r);
    catch
        [U,S,~] = svd(Xq, 'econ');
        U = U(:, 1:r);
        S = S(1:r, 1:r);
    end

    B = max(0, U * sqrt(S));      % Pm x r
    BtB = (B' * B) + opts.eps0 * eye(r);
    F = max(0, (BtB \ (B' * Xpos))); % r x T
end

% Optional refinement:
% keep B tied to quiet frames, then refit F on all frames.
for k = 1:opts.n_refine
    if r == 1
        % Keep the rank-1 global profile uniform; only refresh F.
        F = fit_rank1_nonovershoot(B, Xpos, opts);  % 1 x T
    else
        Fq = F(:, quiet_idx);
        FFt = (Fq * Fq') + opts.eps0 * eye(r);
        B = max(0, (Xq * Fq') / FFt);      % Pm x r, quiet-frame only

        BtB = (B' * B) + opts.eps0 * eye(r);
        F = max(0, (BtB \ (B' * Xpos)));  % r x T
    end
end

% Normalize columns of B to reduce scale ambiguity
colnorm = sqrt(sum(B.^2, 1)) + opts.eps0;
B = B ./ colnorm;
F = F .* colnorm';

% Diagnostics
den = norm(Xpos, 'fro') + opts.eps0;
info = struct();
info.quiet_idx = quiet_idx;
info.Xpos = Xpos;
info.recon_err = norm(Xpos - B*F, 'fro') / den;
end

function [B, F] = init_background_pc1_global(X, r, opts)
% Initialize a global background mode from the first principal component.
% B is inferred from the centered PC1 spatial loading and constrained
% nonnegative. F is then fit conservatively so B*F does not exceed X.

if r ~= 1
    error('bg_init_mode="pc1_global" currently requires bg_rank = 1.');
end

X_for_bg = temporal_gaussian_smooth(X, opts.bg_pc1_temporal_sigma_frames);

Xc = X_for_bg;
if opts.bg_pc1_center
    Xc = Xc - mean(Xc, 2);
end

try
    [u, s, ~] = svds(Xc, 1);
catch
    [u, s, ~] = svd(Xc, 'econ');
    u = u(:,1);
    s = s(1,1);
end

if sum(u) < 0
    u = -u;
end

b0 = u * sqrt(s);
switch opts.bg_pc1_nonneg_mode
    case "clip"
        B = max(b0, 0);
    case "shift"
        B = b0 - min(b0);
    otherwise
        error('opts.bg_pc1_nonneg_mode must be "clip" or "shift".');
end

if all(B == 0)
    B = abs(b0);
end

colnorm = sqrt(sum(B.^2, 1)) + opts.bg_eps;
B = B ./ colnorm;

% Fit a conservative temporal strength from the lower tail rather than a
% projection fit, reducing immediate capture of localized signal.
F = fit_rank1_nonovershoot(B, X, opts);
end

%% -----------------------
% local helper
% -----------------------
function opts = set_defaults_for_background(opts)
def.bg_rank = 3;
def.use_quiet_init = true;
def.quiet_prctile = 20;
def.n_refine = 1;
def.nonneg_mode = "none"; % safest for dF/F init
def.eps0 = 1e-12;
def.bg_floor_quantile = 0.05;
def.bg_floor_noise_sigma = [];

f = fieldnames(def);
for i = 1:numel(f)
    if ~isfield(opts, f{i})
        opts.(f{i}) = def.(f{i});
    end
end

% Guardrails
opts.bg_rank = max(1, round(opts.bg_rank));
opts.quiet_prctile = min(50, max(1, opts.quiet_prctile));
opts.n_refine = max(0, round(opts.n_refine));
if opts.bg_floor_quantile > 1
    opts.bg_floor_quantile = opts.bg_floor_quantile / 100;
end
opts.bg_floor_quantile = min(0.49, max(0.001, opts.bg_floor_quantile));
end



%%
function opts = set_default_opts(opts)
    % Backward compatibility:
    % if old field `quiet_prctile` is provided, map it to bg_quiet_prctile
    if isfield(opts, 'quiet_prctile') && ~isfield(opts, 'bg_quiet_prctile')
        opts.bg_quiet_prctile = opts.quiet_prctile;
    end

    def.maxIter = 200;
    def.minIter = 50;
    def.tol = 1e-5;

    def.lambdaA_L1 = 1e-6;
    def.lambdaA_lap = 1e-4;
    def.lambdaA_excl = 0;          % start OFF
    def.lambdaA_compact = 0;       % OFF by default (discourage scattered A)
    def.compact_eps = 1e-12;
    def.compact_norm_coords = true;
    def.lambdaC_smooth = 1e-4;
    def.lambdaF_smooth = 1e-2; % global/background temporal smoothness
    def.enforce_background_nonovershoot = false;
    def.bg_floor_quantile = 0.05;
    def.bg_floor_noise_sigma = [];
    def.enforce_F_quantile_baseline = false;
    def.F_baseline_prctile = 5;
    def.F_quantile_ref = [];
    def.F_baseline_anchor_strength = 0.5;
    def.lambdaB_lap = 5e-3;   % start ~ 5–20x lambdaA_lap
    def.innerB = 3;           % how many gradient steps for B
    def.innerF = 3;           % how many projected-gradient steps for F
    def.update_F = true;

    % the guide from AQuA2 events' projections
    def.lambdaA_guide = 0;        % guide strength (OFF by default)
    def.use_guide_scale = true;   % compute beta each iter
    def.guide_eps = 1e-12;
    def.init_evt_sigma = 8;
    def.init_evt_min_peak_dist = 200;
    def.init_evt_min_frac = 0.10;
    def.init_evt_max_frac = 0.30;
    def.init_evt_threshold_levels = 40;
    def.init_evt_ridge = 1e-6;

    def.etaA = 1e-3;
    def.etaC = 1e-3;

    def.use_adaptive_steps = true;

    def.innerA = 1;
    def.innerC = 1;

    def.doNormalize = true;
    def.normalizeEvery = 10;       % NEW: normalize infrequently
    def.normalize_mode = "l2";    % "l2" or "p99"
    def.normalize_prctile = 99;    % used when normalize_mode="p99"
    def.neighborhood = 4;
    def.seed = 0;

    % NEW: background term
    def.use_background = true;
    def.bg_rank = 3;
    def.bg_eps = 1e-12;          % small ridge for SPD solves
    def.bg_init_mode = "lowrank"; % "lowrank" or "pc1_global"
    def.update_background = true;
    def.bg_pc1_center = true;
    def.bg_pc1_nonneg_mode = "clip";
    def.bg_pc1_temporal_sigma_frames = 10;

    % NEW: backtracking safeguard
    def.backtracking = true;
    def.maxBacktrack = 15;

    % IMPORTANT: default to "none" so you don't double-shift
    def.nonneg_mode = "none";  % "none"|"shift"|"clip"

    def.verbose = true;
    def.printEvery = 10;

    f = fieldnames(def);
    for i = 1:numel(f)
        if ~isfield(opts, f{i})
            opts.(f{i}) = def.(f{i});
        end
    end

    % Guardrail for quiet frame percentile used by background updates
    opts.normalize_prctile = min(100, max(50, opts.normalize_prctile));
    if opts.bg_floor_quantile > 1
        opts.bg_floor_quantile = opts.bg_floor_quantile / 100;
    end
    opts.bg_floor_quantile = min(0.49, max(0.001, opts.bg_floor_quantile));
    if opts.F_baseline_prctile <= 1
        opts.F_baseline_prctile = 100 * opts.F_baseline_prctile;
    end
    opts.F_baseline_prctile = min(100, max(0, opts.F_baseline_prctile));
    opts.F_baseline_anchor_strength = min(1, max(0, opts.F_baseline_anchor_strength));
end
%%
function obj = objective_val(X, A, C, B, F, L, DtD, opts, guide_act, row_act, col_act)
    R = X - (A*C + B*F);
    fitTerm = 0.5 * (norm(R,'fro')^2);

    lapTerm = 0;
    if opts.lambdaA_lap > 0
        lapTerm = 0.5 * opts.lambdaA_lap * trace(A' * (L * A));
    end

    l1Term = opts.lambdaA_L1 * sum(A(:));

    exclTerm = 0;
    if opts.lambdaA_excl > 0
        sumA = sum(A,2);
        exclTerm = opts.lambdaA_excl * 0.5 * sum( sumA.^2 - sum(A.^2,2) );
    end

    compactTerm = 0;
    if opts.lambdaA_compact > 0
        [~, compact_raw] = compactness_dist2_and_term(A, row_act, col_act, opts.compact_eps);
        compactTerm = opts.lambdaA_compact * compact_raw;
    end

    smoothCTerm = 0;
    if opts.lambdaC_smooth > 0
        smoothCTerm = 0.5 * opts.lambdaC_smooth * trace(C * (DtD * C'));
    end

    smoothFTerm = 0;
    if opts.lambdaF_smooth > 0 && ~isempty(F)
        smoothFTerm = 0.5 * opts.lambdaF_smooth * trace(F * (DtD * F'));
    end

    guideTerm = 0;
    if opts.lambdaA_guide > 0
        s = sum(A,2);
        g = guide_act;   % must be accessible (see note below)
        
        if opts.use_guide_scale
            beta = (g' * s) / (g' * g + opts.guide_eps);
        else
            beta = 1;
        end
        
        r = s - beta * g;
        guideTerm = 0.5 * opts.lambdaA_guide * (r' * r);
    end

    obj = fitTerm + lapTerm + l1Term + exclTerm + compactTerm + smoothCTerm + smoothFTerm + guideTerm;
end

function [dist2, compact_raw] = compactness_dist2_and_term(A, row_act, col_act, eps0)
% Distances to per-component centroid for compactness penalty on A.
% dist2(p,k) = ||x_p - mu_k||^2, where mu_k is weighted by A(:,k).

[Pm, K] = size(A);
rr = row_act(:);  % Pm x 1
cc = col_act(:);  % Pm x 1

mass = sum(A, 1) + eps0;                     % 1 x K
mu_r = (rr' * A) ./ mass;                    % 1 x K
mu_c = (cc' * A) ./ mass;                    % 1 x K

dist2 = (rr - mu_r).^2 + (cc - mu_c).^2;     % Pm x K (implicit expansion)
compact_raw = sum(sum(A .* dist2));
end
%%
function DtD = build_DtD(T)
    main = [1; 2*ones(T-2,1); 1];
    off  = -1*ones(T-1,1);
    offL = [off; 0];
    offU = [0; off];
    DtD = spdiags([offL main offU], [-1 0 1], T, T);
end
%%
function L = build_laplacian_active(H, W, maskVec, neighborhood)
    maskImg = reshape(maskVec, H, W);
    idxFull = find(maskVec);
    Pm = numel(idxFull);

    map = zeros(H*W,1);
    map(idxFull) = 1:Pm;

    Wmat = spalloc(Pm, Pm, Pm*4);
    lin = @(i,j) sub2ind([H W], i, j);

    for i = 1:H
        for j = 1:W
            if ~maskImg(i,j), continue; end
            p_full = lin(i,j);
            p = map(p_full);

            neigh = [];
            if i > 1, neigh(end+1,:) = [i-1 j]; end %#ok<AGROW>
            if i < H, neigh(end+1,:) = [i+1 j]; end %#ok<AGROW>
            if j > 1, neigh(end+1,:) = [i j-1]; end %#ok<AGROW>
            if j < W, neigh(end+1,:) = [i j+1]; end %#ok<AGROW>

            if neighborhood == 8
                if i>1 && j>1, neigh(end+1,:)=[i-1 j-1]; end %#ok<AGROW>
                if i>1 && j<W, neigh(end+1,:)=[i-1 j+1]; end %#ok<AGROW>
                if i<H && j>1, neigh(end+1,:)=[i+1 j-1]; end %#ok<AGROW>
                if i<H && j<W, neigh(end+1,:)=[i+1 j+1]; end %#ok<AGROW>
            end

            for t = 1:size(neigh,1)
                ii = neigh(t,1); jj = neigh(t,2);
                if ~maskImg(ii,jj), continue; end
                q_full = lin(ii,jj);
                q = map(q_full);

                Wmat(p,q) = 1;
                Wmat(q,p) = 1;
            end
        end
    end

    deg = sum(Wmat, 2);
    D = spdiags(deg, 0, Pm, Pm);
    L = D - Wmat;
end
%%
function X = soft_thresh_nonneg(X, tau)
    X = max(0, X - tau);
end

%% L2-normalization on the columns of A
function [A, C] = normalize_factors(A, C, opts)
    mode = string(opts.normalize_mode);
    switch mode
        case "l2"
            scale = sqrt(sum(A.^2, 1)) + eps;
        case "p99"
            K = size(A,2);
            scale = zeros(1, K);
            for k = 1:K
                scale(k) = prctile(A(:,k), opts.normalize_prctile);
            end
            scale = max(scale, eps);
        otherwise
            error('opts.normalize_mode must be "l2" or "p99".');
    end

    A = A ./ scale;
    C = C .* scale';
end

function Xs = temporal_gaussian_smooth(X, sigma_frames)
if sigma_frames <= 0
    Xs = X;
    return;
end

half_width = max(1, ceil(3 * sigma_frames));
t = -half_width:half_width;
kernel = exp(-(t.^2) / (2 * sigma_frames^2));
kernel = kernel / sum(kernel);

left_pad = repmat(X(:,1), 1, half_width);
right_pad = repmat(X(:,end), 1, half_width);
X_pad = [left_pad, X, right_pad];
Xs = conv2(X_pad, kernel, 'valid');
end
