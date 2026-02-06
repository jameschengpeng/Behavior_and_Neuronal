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
    use_background = false;
else
    use_background = true;
end

opts = set_default_opts(opts);

[P, T] = size(X_dFF);
assert(P == H*W, 'X_dFF must have P=H*W rows.');
assert(size(A_full, 1) == P, 'A_full must have P rows.');

mask = reshape(mask, [], 1) ~= 0;
assert(numel(mask) == P, 'mask must be HxW or P-vector.');

K = size(A_full, 2);

% Active pixels only (tissue)
idx = find(mask);
Pm  = numel(idx);
Xraw = double(X_dFF(idx, :));

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

% Temporal smoothness operator D'D (T x T)
DtD = build_DtD(T);

% ---------------------------
% Initialization
% ---------------------------
rng(opts.seed);

% Initialize C using NNLS with small ridge for stability
if use_background
    % Solve: min ||X - A*C - B*F||^2 for C, F jointly (alternating init)
    % Start with F = 0
    F = zeros(r, T);
    AAt = Aact' * Aact + opts.bg_eps * eye(K);
    AtX = Aact' * X;
    C = AAt \ AtX;  % K x T
    C = max(C, 0);
    
    % Then update F given C
    BtB = Bact' * Bact + opts.bg_eps * eye(r);
    BtR = Bact' * (X - Aact*C);
    F = BtB \ BtR;  % r x T
    F = max(F, 0);
else
    % No background: simple NNLS
    AAt = Aact' * Aact + opts.bg_eps * eye(K);
    AtX = Aact' * X;
    C = AAt \ AtX;  % K x T
    C = max(C, 0);
    F = zeros(0, T);
end

% Reinit any dead components
for k = 1:K
    if all(C(k,:) == 0)
        C(k,:) = max(0, rand(1,T));
    end
end

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
        gradC = (Aact' * R) + opts.lambdaC_smooth * (C * DtD);
        
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
            gradF = (Bact' * R) + opts.lambdaF_smooth * (F * DtD);
            
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
end

function obj = objective_val(X, A, C, B, F, DtD, opts)
    R = X - (A*C + B*F);
    fitTerm = 0.5 * (norm(R,'fro')^2);

    smoothCTerm = 0;
    if opts.lambdaC_smooth > 0
        smoothCTerm = 0.5 * opts.lambdaC_smooth * trace(C * (DtD * C'));
    end

    smoothFTerm = 0;
    if opts.lambdaF_smooth > 0 && ~isempty(F)
        smoothFTerm = 0.5 * opts.lambdaF_smooth * trace(F * (DtD * F'));
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