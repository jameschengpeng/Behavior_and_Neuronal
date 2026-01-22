%%
function [A, C, info] = custom_cnmf(X_dFF, H, W, K, mask, opts)
%CUSTOM_CNMF  Constrained NMF for calcium imaging (CNMF-like).
%
% Constraints:
%   - A >= 0, C >= 0
%   - masked pixels fixed to 0 in A
%   - L1(A) sparsity (prox)
%   - Laplacian spatial smoothness on A (quadratic)
%   - mutual exclusivity penalty on A (pixelwise overlap)
%   - temporal smoothness on C via first-difference quadratic
%
% X_dFF:  P x T, P=H*W
% mask:   HxW or P-vector, 1=tissue, 0=masked out

if nargin < 6, opts = struct(); end
opts = set_default_opts(opts);

[P, T] = size(X_dFF);
assert(P == H*W, 'X_dFF must have P=H*W rows.');
mask = reshape(mask, [], 1) ~= 0;
assert(numel(mask) == P, 'mask must be HxW or P-vector.');

% Active pixels only (tissue)
idx = find(mask);
Pm  = numel(idx);
Xraw = X_dFF(idx, :);

% Ensure nonnegativity of X for NMF
X = Xraw;
if opts.nonneg_mode == "shift"
    X = X - min(X(:));
elseif opts.nonneg_mode == "clip"
    X = max(X, 0);
end
X = double(X);

s = norm(X, 'fro');


% Build Laplacian on ACTIVE pixel graph
L = build_laplacian_active(H, W, mask, opts.neighborhood);

% Temporal smoothness operator D'D (T x T)
DtD = build_DtD(T);

% ---------------------------
% Initialization (SVD-based)
% ---------------------------
rng(opts.seed);

% Use a low-rank SVD of X (on active pixels) for a good starting point.
% X is Pm x T. We want Aact (Pm x K) and C (K x T).
%
% This is a simple NNDSVD-like init:
%   [U,S,V] = svds(X, K);  A0 = max(0, U*sqrt(S)), C0 = max(0, sqrt(S)*V')
%
% Works best when X is nonnegative (we enforced above).
try
    [U,S,V] = svds(X, K);
catch
    % fallback if svds fails for any reason
    [U,S,V] = svd(X, 'econ');
    U = U(:,1:K); S = S(1:K,1:K); V = V(:,1:K);
end

Aact = max(0, U * sqrt(S));
C    = max(0, (sqrt(S) * V'));   % K x T

% If any component is completely zero (can happen), reinit that component
for k = 1:K
    if all(Aact(:,k) == 0) || all(C(k,:) == 0)
        Aact(:,k) = max(0, rand(Pm,1));
        C(k,:)    = max(0, rand(1,T));
    end
end

% Normalize columns to stabilize scaling
if opts.doNormalize
    [Aact, C] = normalize_factors(Aact, C);
end

info.obj = zeros(opts.maxIter,1);
info.relchg = zeros(opts.maxIter,1);

Xnorm = norm(X, 'fro') + eps;

% ---------------------------
% Main loop
% ---------------------------
for it = 1:opts.maxIter

    % ===== Adaptive step sizes (important for stability)
    if opts.use_adaptive_steps
        % For C: gradC Lipschitz approx ~ ||A'A|| + lambdaC*||DtD||
        AAt = Aact' * Aact;                 % K x K
        LipC = norm(AAt, 'fro');
        if opts.lambdaC_smooth > 0
            % DtD spectral norm for first-diff is <= 4 (for large T)
            LipC = LipC + opts.lambdaC_smooth * 4;
        end
        etaC = 0.9 / (LipC + eps);

        % For A: gradA Lipschitz approx ~ ||CC'|| + lambdaA*||L|| + excl term
        CCt = C * C';                       % K x K
        LipA = norm(CCt, 'fro');
        if opts.lambdaA_lap > 0
            % Graph Laplacian spectral norm bounded by 2*dmax
            % dmax ~ 4 (or 8) on grid; use neighborhood to be safe
            dmax = (opts.neighborhood == 8) * 8 + (opts.neighborhood == 4) * 4;
            LipA = LipA + opts.lambdaA_lap * (2*dmax);
        end
        if opts.lambdaA_excl > 0
            LipA = LipA + opts.lambdaA_excl * K;
        end
        etaA = 0.9 / (LipA + eps);
    else
        etaA = opts.etaA;
        etaC = opts.etaC;
    end

    % ---- Update C (projected gradient + temporal smoothness)
    for s = 1:opts.innerC
        R = Aact*C - X;  % Pm x T
        gradC = (Aact' * R) + opts.lambdaC_smooth * (C * DtD); % K x T
        C = C - etaC * gradC;
        C = max(C, 0);
    end

    % ---- Update A (proximal gradient: fit + lap + excl, then L1+nonneg prox)
    for s = 1:opts.innerA
        R = Aact*C - X;    % Pm x T
        gradA = (R * C');  % Pm x K

        if opts.lambdaA_lap > 0
            gradA = gradA + opts.lambdaA_lap * (L * Aact);
        end

        if opts.lambdaA_excl > 0
            sumA = sum(Aact, 2);                   % Pm x 1
            gradExcl = (sumA * ones(1,K)) - Aact;  % Pm x K
            gradA = gradA + opts.lambdaA_excl * gradExcl;
        end

        Aact = Aact - etaA * gradA;

        if opts.lambdaA_L1 > 0
            Aact = soft_thresh_nonneg(Aact, etaA * opts.lambdaA_L1);
        else
            Aact = max(Aact, 0);
        end
    end

    if opts.doNormalize
        [Aact, C] = normalize_factors(Aact, C);
    end

    % ---- Diagnostics
    R = X - Aact*C;
    fitTerm = 0.5 * (norm(R,'fro')^2);

    lapTerm = 0;
    if opts.lambdaA_lap > 0
        lapTerm = 0.5 * opts.lambdaA_lap * trace(Aact' * (L * Aact));
    end

    l1Term = opts.lambdaA_L1 * sum(Aact(:));

    exclTerm = 0;
    if opts.lambdaA_excl > 0
        sumA = sum(Aact,2);
        exclTerm = opts.lambdaA_excl * 0.5 * sum( sumA.^2 - sum(Aact.^2,2) );
    end

    smoothCTerm = 0;
    if opts.lambdaC_smooth > 0
        smoothCTerm = 0.5 * opts.lambdaC_smooth * trace(C * (DtD * C'));
    end

    obj = fitTerm + lapTerm + l1Term + exclTerm + smoothCTerm;
    info.obj(it) = obj;


    if opts.verbose && (mod(it, opts.printEvery) == 0 || it == 1)
        fprintf('Iter %4d | obj %.4e | fit %.3e/%.3e | ||A|| %.3e nnzA %d | ||C|| %.3e\n', ...
            it, obj, norm(R,'fro'), Xnorm, norm(Aact,'fro'), nnz(Aact), norm(C,'fro'));
    end

    % ---- Stopping: prevent early stop before minIter
    if it > 1
        relchg = abs(info.obj(it) - info.obj(it-1)) / (abs(info.obj(it-1)) + eps);
        info.relchg(it) = relchg;
        if it >= opts.minIter && relchg < opts.tol
            info.obj = info.obj(1:it);
            info.relchg = info.relchg(1:it);
            break;
        end
    end
end

% Put back into full P x K with masked pixels = 0
A = zeros(P, K);
A(idx, :) = Aact;

relRecon = norm(X - Aact*C,'fro')^2 / (norm(X,'fro')^2 + eps);

rawLap = trace(Aact'*(L*Aact));
rawL1  = sum(Aact(:));
sumA   = sum(Aact,2);
rawExcl = 0.5 * sum(sumA.^2 - sum(Aact.^2,2));
rawSmoothC = trace(C*(DtD*C'));

fprintf('relRecon=%.4g\n', relRecon);
fprintf('rawLap=%.4g, rawL1=%.4g, rawExcl=%.4g, rawSmoothC=%.4g\n', ...
    rawLap, rawL1, rawExcl, rawSmoothC);


end

%% ======================================================================
% Helpers
% ======================================================================

function opts = set_default_opts(opts)
    def.maxIter = 200;
    def.minIter = 50;                 % NEW: prevent stopping at iter 2
    def.tol = 1e-5;

    def.lambdaA_L1 = 1e-6;            % safer default than 1e-3
    def.lambdaA_lap = 1e-4;
    def.lambdaA_excl = 0;             % start OFF; enable later
    def.lambdaC_smooth = 1e-4;

    def.etaA = 1e-3;                  % used only if adaptive steps OFF
    def.etaC = 1e-3;

    def.use_adaptive_steps = true;    % NEW: strongly recommended

    def.innerA = 1;
    def.innerC = 1;

    def.doNormalize = true;
    def.neighborhood = 4;
    def.seed = 0;

    % How to enforce nonnegativity on X inside the function:
    %   "none"  -> assume already nonnegative
    %   "shift" -> subtract min(X(:)) over active pixels
    %   "clip"  -> max(X,0)
    def.nonneg_mode = "shift";

    def.verbose = true;
    def.printEvery = 10;

    f = fieldnames(def);
    for i = 1:numel(f)
        if ~isfield(opts, f{i})
            opts.(f{i}) = def.(f{i});
        end
    end
end

function DtD = build_DtD(T)
main = [1; 2*ones(T-2,1); 1];
off  = -1*ones(T-1,1);
offL = [off; 0];
offU = [0; off];
DtD = spdiags([offL main offU], [-1 0 1], T, T);
end

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

function X = soft_thresh_nonneg(X, tau)
X = max(0, X - tau);
end

function [A, C] = normalize_factors(A, C)
colNorm = sqrt(sum(A.^2, 1)) + eps;
A = A ./ colNorm;
C = C .* colNorm';
end
