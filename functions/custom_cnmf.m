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
tic
if nargin < 7, opts = struct(); end
opts = set_default_opts(opts);

[P, T] = size(X_data);
assert(P == H*W, 'X_data must have P=H*W rows.');
mask = reshape(mask, [], 1) ~= 0;
assert(numel(mask) == P, 'mask must be HxW or P-vector.');

% Active pixels only (tissue)
idx = find(mask);
Pm  = numel(idx);
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

opts_bg = opts;
opts_bg.bg_noise_var = bg_noise_var_act;

% use the AQuA2 detected events' projection as a guide to CNMF
evt_domain_projection_2d = reshape(evt_domain_projection, H, W);
evt_domain_projection_vec = reshape(evt_domain_projection_2d, [], 1);
guide_act  = cast(double(evt_domain_projection_vec(idx)), 'like', X);
guide_act  = guide_act / (max(guide_act) + eps);  % normalize to [0,1]

% Nonnegativity handling for X (IMPORTANT: avoid double-shifting)
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

% (DtD is no longer materialized; apply_DtD applies the second-difference
% operator in O(K*T) via the tridiagonal structure.)

% ---------------------------
% Initialization
% ---------------------------
rng(opts.seed);

% initialize background first (if use_background)
r = opts.bg_rank;

if opts.use_background
    bgopts = struct();
    bgopts.bg_rank = r;
    bgopts.n_refine = 1;
    bgopts.nonneg_mode = "clip";
    bgopts.eps0 = opts.bg_eps;
    bgopts.bg_noise_var = bg_noise_var_act;
    bgopts.bg_floor_quantile = opts.bg_floor_quantile;
    bgopts.bg_floor_noise_sigma = opts.bg_floor_noise_sigma;
    [B, F] = init_background_lowrank(X, bgopts);   % B: Pm x r, F: r x T

    if r == 1 && opts.bg_refine_profile_from_F
        [B, F] = refine_rank1_background_profile(X, B, F, H, W, mask, opts_bg);
    end
else
    B = zeros(Pm, 0, 'like', X);
    F = zeros(0, T, 'like', X);
end

% Residual for component init
Xr = X - B*F;

% Compute noise-corrected signal variance map from unclipped Xr.
% Var(Xr_p) = Var(S_p) + sigma_p^2  (exact, no clipping approximation needed).
% Subtracting the per-pixel noise variance recovers the pure signal variance.
if opts.temporally_downsampled && ~isempty(bg_noise_var_act)
    % Data was temporally downsampled: diff-based MAD would overestimate
    % noise because the signal changes between consecutive frames.
    % Use the precomputed noise variance (estimated at full frame rate,
    % then divided by the downsample factor) instead.
    noise_var_xr = double(bg_noise_var_act);                    % Pm x 1
else
    noise_var_xr = double(estimate_noise_var_per_pixel(Xr));    % Pm x 1
end
var_xr = var(double(Xr), 0, 2);                            % Pm x 1
signal_var_act = max(var_xr - noise_var_xr, 0);            % Pm x 1

% Embed active-pixel signal variance back into the full H x W image.
signal_var_full = zeros(H * W, 1);
signal_var_full(idx) = signal_var_act;
guide_map_2d = reshape(signal_var_full, H, W);   % H x W, noise-corrected signal variance

Xr = max(Xr, 0);  % clipping to keep nonnegativity for NMF algorithm
if opts.AC_init_method == "svd"
    [Aact, C, init_info] = init_AC_svd(Xr, K);
elseif opts.AC_init_method == "random"
    [Aact, C, init_info] = init_AC_random(Xr, K);
else
    [Aact, C, init_info] = init_AC_from_guide_map(Xr, guide_map_2d, reshape(mask, H, W), K, opts);
end
clear Xr signal_var_full signal_var_act var_xr

% this block is for checking the initialization of spatial footprints
A_init = zeros(size(mask, 1), K, 'like', Aact);
A_init(idx, :) = Aact;
A_init_comb = reshape(sum(A_init, 2), [H, W]);
clear mask

% Reinit any dead components
for k = 1:K
    if all(Aact(:,k) == 0) || all(C(k,:) == 0)
        if opts.AC_init_mode == "event_projection"
            Aact(:,k) = 0;
            C(k,:) = 0;
        else
            Aact(:,k) = max(0, rand(Pm,1));
            C(k,:)    = max(0, rand(1,T));
        end
    end
end

% Optional normalization (do NOT do every iter by default)
if opts.doNormalize
    [Aact, C] = normalize_factors(Aact, C, opts);
end

info.obj    = zeros(opts.maxIter,1);
info.relchg = zeros(opts.maxIter,1);
info.relRecon = zeros(opts.maxIter,1);
info.lambdaA_L1 = zeros(opts.maxIter,1);
info.A_nnz_frac = zeros(opts.maxIter,1);
info.relRecon_support = zeros(opts.maxIter,1);
info.stop_reason = "maxIter";
info.init = init_info;
clear init_info

pre_processing_elapsed = toc;
fprintf('Preprocessing elapsed time: %.6f seconds\n', pre_processing_elapsed)

% ---------------------------
% Main loop
% ---------------------------
prevObj = inf;

% Cache B*F outside the loop (B and F are frozen after init)
BF = B * F;   % Pm x T (empty 0×T if no background)

% Per-pixel inverse-variance weights for weighted least squares.
% Normalized to mean 1 so the objective scale and Lipschitz constants
% remain comparable to the unweighted case.
w2 = 1 ./ max(double(noise_var_xr), eps);          % Pm x 1
w2 = cast(w2 / mean(w2), 'like', X);               % normalize, match data type
sqrt_w2 = sqrt(w2);                                % for Lipschitz computation
clear noise_var_xr

% Pre-compute ||Xr||_F^2 for relRecon (constant across iterations).
% Using Xr = X - BF as denominator so relRecon tracks how well AC explains
% the signal after background removal, not the BF-dominated baseline.
normXr2 = norm(X - BF, 'fro')^2 + eps;

for it = 1:opts.maxIter
    info.lambdaA_L1(it) = opts.lambdaA_L1;

    if opts.adapt_lambdaA_L1 && opts.rollback_on_A_nnz_undershoot && it > 1
        % Keep the full previous factor state so an overshoot below the
        % lower sparsity bound can revert to the last acceptable iterate.
        A_prev_iter = Aact;
        C_prev_iter = C;
    end

    % ===== Step sizes (with safer spectral-norm estimates)
    if opts.use_adaptive_steps
        % LipC ~ ||A' diag(w2) A||_2 + lambdaC*||DtD||_2 (||DtD||_2 <= 4)
        Aw   = sqrt_w2 .* Aact;              % Pm x K, weighted
        AAt  = Aw' * Aw;                     % K x K, = A' * diag(w2) * A
        LipC = norm(AAt, 2) + opts.lambdaC_smooth * 4;
        etaC = 0.9 / (LipC + eps);

        % LipA ~ max(w2)*||CC'||_2 + lambdaA*||L||_2 + excl
        CCt  = C * C';
        LipA = max(w2) * norm(CCt, 2);
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
        R = Aact*C + BF - X;                                     % Pm x T
        gradC = (Aact' * (w2 .* R)) + opts.lambdaC_smooth * apply_DtD(C); % K x T

        Cnew = max(0, C - etaC * gradC);

        if opts.backtracking
            objOld = objective_from_residual(R, Aact, C, L, opts, guide_act, row_act, col_act, w2);
            Rnew = R + Aact*(Cnew - C);                           % cheap rank-K update
            objNew = objective_from_residual(Rnew, Aact, Cnew, L, opts, guide_act, row_act, col_act, w2);
            bt = 0;
            while objNew > objOld && bt < opts.maxBacktrack
                etaC = etaC * 0.5;
                Cnew = max(0, C - etaC * gradC);
                Rnew = R + Aact*(Cnew - C);
                objNew = objective_from_residual(Rnew, Aact, Cnew, L, opts, guide_act, row_act, col_act, w2);
                bt = bt + 1;
            end
            R = Rnew;  % keep the accepted residual
        end

        C = Cnew;
    end

    % =========================
    % (2) Update A (prox-grad: fit + lap + excl, then L1+nonneg prox)
    % =========================
    for s = 1:opts.innerA
        R = Aact*C + BF - X;                 % Pm x T
        gradA = ((w2 .* R) * C');            % Pm x K

        if opts.lambdaA_lap > 0
            gradA = gradA + opts.lambdaA_lap * (L * Aact);
        end

        if opts.lambdaA_excl > 0
            sumA = sum(Aact, 2);
            gradExcl = sumA - Aact;
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
            
            gradGuide = opts.lambdaA_guide * (Aact_sum - beta * g);
            
            gradA = gradA + gradGuide;
        end

        Anew = Aact - etaA * gradA;

        if opts.lambdaA_L1 > 0
            Anew = soft_thresh_nonneg(Anew, etaA * opts.lambdaA_L1);
        else
            Anew = max(Anew, 0);
        end

        if opts.backtracking
            objOld = objective_from_residual(R, Aact, C, L, opts, guide_act, row_act, col_act, w2);
            Rnew = R + (Anew - Aact)*C;                           % cheap rank-K update
            objNew = objective_from_residual(Rnew, Anew, C, L, opts, guide_act, row_act, col_act, w2);
            bt = 0;
            while objNew > objOld && bt < opts.maxBacktrack
                etaA = etaA * 0.5;
                Anew = Aact - etaA * gradA;
                if opts.lambdaA_L1 > 0
                    Anew = soft_thresh_nonneg(Anew, etaA * opts.lambdaA_L1);
                else
                    Anew = max(Anew, 0);
                end
                Rnew = R + (Anew - Aact)*C;
                objNew = objective_from_residual(Rnew, Anew, C, L, opts, guide_act, row_act, col_act, w2);
                bt = bt + 1;
            end
            R = Rnew;  % keep the accepted residual
        end

        Aact = Anew;
    end

    % Optional normalization (do rarely)
    if opts.doNormalize && mod(it, opts.normalizeEvery) == 0
        [Aact, C] = normalize_factors(Aact, C, opts);
    end

    % ---- Diagnostics (reuse cached R or recompute once after normalization)
    R = Aact*C + BF - X;
    obj = objective_from_residual(R, Aact, C, L, opts, guide_act, row_act, col_act, w2);
    info.obj(it) = obj;

    % R = AC + BF - X, so AC - Xr = R where Xr = X - BF.
    % relRecon = ||AC - Xr||^2 / ||Xr||^2  (all active pixels)
    relRecon = sum(R(:).^2) / normXr2;
    info.relRecon(it) = relRecon;

    % relRecon_support: same metric restricted to the footprint of A.
    % Shows how well AC explains Xr where the model claims neurons exist.
    support_mask = any(Aact > 0, 2);    % Pm x 1 logical
    R_sup = R(support_mask, :);
    Xr_sup = X(support_mask, :) - BF(support_mask, :);
    relRecon_support = sum(R_sup(:).^2) / (sum(Xr_sup(:).^2) + eps);
    info.relRecon_support(it) = relRecon_support;

    info.A_nnz_frac(it) = nnz(Aact) / max(1, numel(Aact));

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
        % R is already (Aact*C + BF - X) from diagnostics above
    
        % Fit gradient
        gradA_fit = ((w2 .* R) * C');
    
        % Laplacian gradient
        gradA_lap = zeros(size(Aact), 'like', Aact);
        if opts.lambdaA_lap > 0
            gradA_lap = opts.lambdaA_lap * (L * Aact);
        end
    
        % Exclusivity gradient
        gradA_excl = zeros(size(Aact), 'like', Aact);
        if opts.lambdaA_excl > 0
            sumA = sum(Aact, 2);
            gradA_excl = opts.lambdaA_excl * (sumA - Aact);
        end

        gradA_compact = zeros(size(Aact), 'like', Aact);
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
        gradC_fit = Aact' * (w2 .* R);
    
        gradC_smooth = zeros(size(C), 'like', C);
        if opts.lambdaC_smooth > 0
            gradC_smooth = opts.lambdaC_smooth * apply_DtD(C);
        end
    
        % =========================
        % ---- F diagnostics ----
        % =========================
        gradF_fit = zeros(size(F), 'like', F);
        if ~isempty(F)
            gradF_fit = B' * (w2 .* R);
        end
    
        % AQuA2 guide diagnostics
        grad_guide = zeros(size(Aact), 'like', Aact);
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
            grad_guide = opts.lambdaA_guide * remainder;
            guide_obj = 0.5 * opts.lambdaA_guide * (remainder' * remainder);
        end

        % =========================
        % ---- Print ----
        % =========================
        fprintf(['Iter %4d | obj %.4e | ||A|| %.3e nnzA %d | ||C|| %.3e | bg %d\n' ...
                 '   relRecon(Xr, all px) %.4g | relRecon(Xr, footprint) %.4g\n' ...
             '   A-grad:  fit %.3e | lap %.3e | excl %.3e | compact %.3e (ratio %.3f) | L1obj %.3e | prox %.3e | tau %.3e\n' ...
                 '   A-nnz:   count/col [%s]\n' ...
                 '   A-nnz%%:  pct/col   [%s]\n' ...
                 '   C-grad:  fit %.3e | smooth %.3e\n' ...
                 '   F-grad:  fit %.3e\n' ...
                 '   Guide:   %.3e | CompactObj(raw) %.3e\n'], ...
                it, obj, norm(Aact,'fro'), nnz(Aact), norm(C,'fro'), opts.use_background, ...
                relRecon, relRecon_support, ...
            norm(gradA_fit,'fro'), norm(gradA_lap,'fro'), norm(gradA_excl,'fro'), norm(gradA_compact,'fro'), ...
                compact_grad_ratio, l1_obj, l1_shrink, tau, ...
                nnzA_col_str, pctA_col_str, ...
                norm(gradC_fit,'fro'), norm(gradC_smooth,'fro'), ...
                norm(gradF_fit,'fro'), ...
                norm(grad_guide,'fro'), compact_obj);

        disp('-----------')
    end

    % If A has fully collapsed, stop immediately rather than continuing
    % with a zero support that will not be useful for factorization.
    if opts.stop_if_A_all_zero && nnz(Aact) == 0
        info.obj = info.obj(1:it);
        info.relchg = info.relchg(1:it);
        info.relRecon = info.relRecon(1:it);
        info.relRecon_support = info.relRecon_support(1:it);
        info.lambdaA_L1 = info.lambdaA_L1(1:it);
        info.A_nnz_frac = info.A_nnz_frac(1:it);
        info.stop_reason = "A_all_zero";
        if opts.verbose
            fprintf('Stopping early at iter %d because Aact became all zero.\n', it);
        end
        break;
    end

    if opts.adapt_lambdaA_L1 && opts.rollback_on_A_nnz_undershoot && it > 1
        target_lo = max(0, opts.target_A_nnz_frac - opts.target_A_nnz_tol);
        if info.A_nnz_frac(it) < target_lo
            nnz_frac_undershoot = info.A_nnz_frac(it);
            nnz_frac_reverted = info.A_nnz_frac(it - 1);

            Aact = A_prev_iter;
            C = C_prev_iter;
            prevObj = info.obj(it - 1);

            info.obj = info.obj(1:it-1);
            info.relchg = info.relchg(1:it-1);
            info.relRecon = info.relRecon(1:it-1);
            info.relRecon_support = info.relRecon_support(1:it-1);
            info.lambdaA_L1 = info.lambdaA_L1(1:it-1);
            info.A_nnz_frac = info.A_nnz_frac(1:it-1);
            info.stop_reason = "A_nnz_undershoot_rollback";

            if opts.verbose
                fprintf(['Stopping early at iter %d because A support undershot the lower target bound ' ...
                         '(%.4f < %.4f); reverting to iter %d with support fraction %.4f.\n'], ...
                        it, nnz_frac_undershoot, target_lo, it - 1, nnz_frac_reverted);
            end
            break;
        end
    end

    % Adapt lambdaA_L1 toward the desired support fraction for the next iteration.
    if opts.adapt_lambdaA_L1
        nnz_frac_cur = info.A_nnz_frac(it);
        target_hi = min(1, opts.target_A_nnz_frac + opts.target_A_nnz_tol);
        target_lo = max(0, opts.target_A_nnz_frac - opts.target_A_nnz_tol);

        if nnz_frac_cur > target_hi
            % If the footprint is too large, shrink the size by multiplying
            % an exponential function
            step = exp(opts.lambdaA_L1_adapt_rate * (nnz_frac_cur - opts.target_A_nnz_frac));
            opts.lambdaA_L1 = min(opts.lambdaA_L1_max, opts.lambdaA_L1 * step);
        elseif nnz_frac_cur < target_lo
            % If the footprint is too small, enlarge the size by dividing
            % an exponential function
            step = exp(opts.lambdaA_L1_adapt_rate * (opts.target_A_nnz_frac - nnz_frac_cur));
            opts.lambdaA_L1 = max(opts.lambdaA_L1_min, opts.lambdaA_L1 / step);
        end
    end

    % stopping
    stop_on_relchg = (it >= opts.minIter && it > 1 && info.relchg(it) < opts.tol);
    if opts.adapt_lambdaA_L1 && opts.require_target_A_nnz_for_stop
        target_hi = min(1, opts.target_A_nnz_frac + opts.target_A_nnz_tol);
        target_lo = max(0, opts.target_A_nnz_frac - opts.target_A_nnz_tol);
        stop_on_relchg = stop_on_relchg && info.A_nnz_frac(it) >= target_lo && info.A_nnz_frac(it) <= target_hi;
    end

    if stop_on_relchg
        info.obj = info.obj(1:it);
        info.relchg = info.relchg(1:it);
        info.relRecon = info.relRecon(1:it);
        info.relRecon_support = info.relRecon_support(1:it);
        info.lambdaA_L1 = info.lambdaA_L1(1:it);
        info.A_nnz_frac = info.A_nnz_frac(1:it);
        info.stop_reason = "relchg_tol";
        break;
    end
end

% Put back into full P x K with masked pixels = 0
A = zeros(P, K, 'like', Aact);
A(idx, :) = Aact;

% Save background in info (doesn't change function signature)
info.B = zeros(P, size(B,2), 'like', B);
info.B(idx, :) = B;
info.F = F;

end

%% ======================================================================
% Helpers (init_background_lowrank, refine_rank1_background_profile,
%  build_DtD, build_laplacian_active, soft_thresh_nonneg,
%  temporal_gaussian_smooth are in separate .m files in this folder)
% ======================================================================

%%
function opts = set_default_opts(opts)
    def.maxIter = 200;
    def.minIter = 50;
    def.tol = 1e-5;

    def.lambdaA_L1 = 1e-6;
    def.adapt_lambdaA_L1 = false;
    def.target_A_nnz_frac = 0.10;
    def.target_A_nnz_tol = 0.02;
    def.lambdaA_L1_adapt_rate = 8;
    def.lambdaA_L1_min = 1e-10;
    def.lambdaA_L1_max = 1e6;
    def.stop_if_A_all_zero = true;
    def.require_target_A_nnz_for_stop = true;
    % Default ON: once A undershoots below the lower support target, later
    % regrowth often returns as scattered pixels rather than the original
    % compact event-guided support.
    def.rollback_on_A_nnz_undershoot = true;
    def.lambdaA_lap = 1e-4;
    def.lambdaA_excl = 0;          % start OFF
    def.lambdaA_compact = 0;       % OFF by default (discourage scattered A)
    def.compact_eps = 1e-12;
    def.compact_norm_coords = true;
    def.lambdaC_smooth = 1e-4;
    def.bg_floor_quantile = 0.05;
    def.bg_floor_noise_sigma = [];
    def.bg_noise_var = [];
    def.temporally_downsampled = false; % set true when data was temporally binned

    % the guide from AQuA2 events' projections
    def.lambdaA_guide = 0;        % guide strength (OFF by default)
    def.use_guide_scale = true;   % compute beta each iter
    def.guide_eps = 1e-12;
    def.init_evt_sigma = 8;
    def.init_evt_min_peak_dist = 200;
    def.init_active_percentile = 25;
    def.init_evt_ridge = 1e-6;
    def.AC_init_mode = "event_projection";
    def.AC_init_method = "guide_map";  % "guide_map", "svd", or "random"

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
    def.bg_init_mode = "lowrank";
    def.bg_refine_profile_from_F = false;
    def.bg_profile_quantile = 0.1;
    def.bg_profile_min_relF = 0.2;
    def.bg_profile_min_frames = 50;
    def.bg_profile_smooth_sigma = 20;
    def.bg_profile_shrink_uniform = 0.5;
    def.bg_profile_n_alternations = 1;

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

    opts.normalize_prctile = min(100, max(50, opts.normalize_prctile));
    opts.target_A_nnz_frac = min(1, max(0, opts.target_A_nnz_frac));
    opts.target_A_nnz_tol = min(1, max(0, opts.target_A_nnz_tol));
    opts.lambdaA_L1_adapt_rate = max(0, opts.lambdaA_L1_adapt_rate);
    opts.lambdaA_L1_min = max(0, opts.lambdaA_L1_min);
    opts.lambdaA_L1_max = max(opts.lambdaA_L1_min, opts.lambdaA_L1_max);
    opts.adapt_lambdaA_L1 = logical(opts.adapt_lambdaA_L1);
    opts.stop_if_A_all_zero = logical(opts.stop_if_A_all_zero);
    opts.require_target_A_nnz_for_stop = logical(opts.require_target_A_nnz_for_stop);
    opts.rollback_on_A_nnz_undershoot = logical(opts.rollback_on_A_nnz_undershoot);
    opts.AC_init_mode = string(opts.AC_init_mode);
    if opts.bg_floor_quantile > 1
        opts.bg_floor_quantile = opts.bg_floor_quantile / 100;
    end
    opts.bg_floor_quantile = min(0.49, max(0.001, opts.bg_floor_quantile));
    if opts.bg_profile_quantile > 1
        opts.bg_profile_quantile = opts.bg_profile_quantile / 100;
    end
    opts.bg_profile_quantile = min(0.49, max(0.001, opts.bg_profile_quantile));
    opts.bg_profile_min_relF = min(1, max(0, opts.bg_profile_min_relF));
    opts.bg_profile_min_frames = max(5, round(opts.bg_profile_min_frames));
    opts.bg_profile_smooth_sigma = max(0, opts.bg_profile_smooth_sigma);
    opts.bg_profile_shrink_uniform = min(1, max(0, opts.bg_profile_shrink_uniform));
    opts.bg_profile_n_alternations = max(0, round(opts.bg_profile_n_alternations));
end
%%
function obj = objective_from_residual(R, A, C, L, opts, guide_act, row_act, col_act, w2)
%OBJECTIVE_FROM_RESIDUAL  Compute objective from pre-computed residual R = AC+BF-X.
% Avoids rebuilding the Pm x T residual; all terms use R, A, C directly.
% w2 (Pm x 1): per-pixel inverse-variance weights (mean-normalised to 1).
    fitTerm = 0.5 * (w2' * sum(R.^2, 2)); % weighted fit: sum_p w2_p * ||R_p||^2

    lapTerm = 0;
    if opts.lambdaA_lap > 0
        LA = L * A;                         % sparse Pm x K
        lapTerm = 0.5 * opts.lambdaA_lap * sum(A(:) .* LA(:));  % = trace(A'*L*A)
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

    % Temporal smoothness: 0.5*lambda*||diff(C)||^2 = 0.5*lambda*trace(C*DtD*C')
    smoothCTerm = 0;
    if opts.lambdaC_smooth > 0
        dC = diff(C, 1, 2);                 % K x (T-1), O(K*T)
        smoothCTerm = 0.5 * opts.lambdaC_smooth * sum(dC(:).^2);
    end

    guideTerm = 0;
    if opts.lambdaA_guide > 0
        s = sum(A,2);
        g = guide_act;
        
        if opts.use_guide_scale
            beta = (g' * s) / (g' * g + opts.guide_eps);
        else
            beta = 1;
        end
        
        r = s - beta * g;
        guideTerm = 0.5 * opts.lambdaA_guide * (r' * r);
    end

    obj = fitTerm + lapTerm + l1Term + exclTerm + compactTerm + smoothCTerm + guideTerm;
end

function Y = apply_DtD(C)
%APPLY_DTD  Compute C * DtD without materializing the T x T matrix.
% DtD is the tridiagonal second-difference operator D'D where D is (T-1)xT.
% C*DtD is equivalent to the 1-D convolution [1 -2 1] along each row of C,
% with boundary handling that matches the sparse DtD exactly.
% Cost: O(K*T) instead of O(K*T^2).
    [K, T] = size(C);
    if T <= 1
        Y = zeros(K, T, 'like', C);
        return;
    end
    Y = zeros(K, T, 'like', C);
    % Interior: y(:,t) = -c(:,t-1) + 2*c(:,t) - c(:,t+1)
    Y(:, 2:T-1) = -C(:, 1:T-2) + 2*C(:, 2:T-1) - C(:, 3:T);
    % Boundaries: y(:,1) = c(:,1) - c(:,2), y(:,T) = -c(:,T-1) + c(:,T)
    Y(:, 1) = C(:, 1) - C(:, 2);
    Y(:, T) = -C(:, T-1) + C(:, T);
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
function [A, C, info] = init_AC_svd(Xr, K)
%INIT_AC_SVD  Initialize A, C via truncated SVD of the (clipped) residual.
%  Xr: Pm x T (nonneg-clipped background residual)
%  Returns A (Pm x K), C (K x T), both nonneg-clipped.
    [U, S, V] = svds(double(Xr), K);
    sqrtS = sqrt(S);
    A = max(U * sqrtS, 0);     % Pm x K
    C = max(sqrtS * V', 0);    % K x T
    A = cast(A, 'like', Xr);
    C = cast(C, 'like', Xr);
    info.method = "svd";
    info.singular_values = diag(S);
end

%%
function [A, C, info] = init_AC_random(Xr, K)
%INIT_AC_RANDOM  Initialize A, C with uniform random values scaled to data.
%  Xr: Pm x T (nonneg-clipped background residual)
%  Returns A (Pm x K), C (K x T), both nonneg.
    [Pm, T] = size(Xr);
    scale = sqrt(mean(Xr(:).^2) / K + eps);   % match energy scale
    A = scale * rand(Pm, K, 'like', Xr);
    C = scale * rand(K, T, 'like', Xr);
    info.method = "random";
    info.scale = scale;
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


