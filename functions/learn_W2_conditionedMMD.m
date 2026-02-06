function [W2, info] = learn_W2_conditionedMMD(C1, C2, ethogram_mat1, ethogram_mat2, opts)
% learn_W2_conditionedMMD
% Learn nonnegative (optionally sparse) W2 to align mouse2 latent C2 into mouse1 latent C1
% by matching state-conditioned distributions p(C | behavior i) using MMD.
%
% This method does NOT require time alignment across mice.
% It uses all frames where a behavior is active, plus optionally a "rest" class.

% -------------------- defaults --------------------
if nargin < 8, opts = struct(); end 
opts = setDefault(opts, 'minFrameCount', 50);
opts = setDefault(opts, 'includeRest', true);
opts = setDefault(opts, 'maxSamplesPerClass', 2000);
opts = setDefault(opts, 'sigmas', [0.5 1 2 4]);
opts = setDefault(opts, 'lambda1', 1e-3);
opts = setDefault(opts, 'lambda2', 1e-3);
opts = setDefault(opts, 'maxIters', 500);
opts = setDefault(opts, 'lr', 1e-2);
opts = setDefault(opts, 'normType', 'madzscore'); % 'none'|'zscore'|'madzscore'
opts = setDefault(opts, 'eps', 1e-8);
opts = setDefault(opts, 'verbose', true);
opts = setDefault(opts, 'printEvery', 25);
opts = setDefault(opts, 'init', 'identity'); % 'identity' or 'zerosU'

[K, T1] = size(C1);
[K2, T2] = size(C2);
assert(K2 == K, 'C1 and C2 must have same K.');
assert(size(ethogram_mat1,1) == T1, 'ethogram_mat1 rows must match T1.');
assert(size(ethogram_mat2,1) == T2, 'ethogram_mat2 rows must match T2.');
B = size(ethogram_mat1,2);
assert(size(ethogram_mat2,2) == B, 'ethogram matrices must have same number of behaviors.');

% preSec/postSec/fps are unused here (kept for interface compatibility)
% If you want, we can add an optional onset-window term later.

% -------------------- preprocess --------------------
C1n = normalizeRows(C1, opts.normType, opts.eps);
C2n = normalizeRows(C2, opts.normType, opts.eps);

% -------------------- define classes (behaviors + optional rest) --------------------
% For each behavior i, indices are frames where ethogram(:,i)==1.
idx1 = cell(0,1);
idx2 = cell(0,1);
classNames = {};

keptBeh = false(B,1);
for i = 1:B
    id1 = find(ethogram_mat1(:,i) ~= 0);
    id2 = find(ethogram_mat2(:,i) ~= 0);
    if numel(id1) >= opts.minFrameCount && numel(id2) >= opts.minFrameCount
        keptBeh(i) = true;
        idx1{end+1,1} = id1; %#ok<AGROW>
        idx2{end+1,1} = id2; %#ok<AGROW>
        classNames{end+1,1} = sprintf('behavior_%d', i); %#ok<AGROW>
    end
end

% Add "rest" class: frames where no behavior is active
if opts.includeRest
    rest1 = find(sum(ethogram_mat1 ~= 0, 2) == 0);
    rest2 = find(sum(ethogram_mat2 ~= 0, 2) == 0);
    if numel(rest1) >= opts.minFrameCount && numel(rest2) >= opts.minFrameCount
        idx1{end+1,1} = rest1; %#ok<AGROW>
        idx2{end+1,1} = rest2; %#ok<AGROW>
        classNames{end+1,1} = 'rest';
    end
end

if isempty(idx1)
    error('No classes meet minFrameCount=%d in both mice (including rest=%d).', opts.minFrameCount, opts.includeRest);
end

C = numel(idx1); % number of classes used

% -------------------- subsample indices per class (for speed & stability) --------------------
rng(0); % deterministic; change if you like
for c = 1:C
    idx1{c} = subsampleIdx(idx1{c}, opts.maxSamplesPerClass);
    idx2{c} = subsampleIdx(idx2{c}, opts.maxSamplesPerClass);
end

% -------------------- initialize U so W2 ~= I (no transform) --------------------
switch lower(opts.init)
    case 'identity'
        U = initU_identity(K, 1e-6); % W2 ~ I (diag~1, offdiag~eps)
    case 'zerosu'
        U = zeros(K,K);              % W2 ~ log(2) everywhere (not recommended)
    otherwise
        error('Unknown opts.init: %s', opts.init);
end

lossHist = zeros(opts.maxIters,1);

% -------------------- optimize --------------------
for it = 1:opts.maxIters
    W2 = softplus(U);      % nonnegative
    dW_dU = sigmoid(U);    % derivative

    totalLoss = 0;
    gradW = zeros(K,K);

    % class-conditioned MMD terms
    for c = 1:C
        I1 = idx1{c};
        I2 = idx2{c};
        X = C1n(:, I1);          % K x n
        Z = C2n(:, I2);          % K x m
        Y = W2 * Z;              % K x m

        [mmd2, dL_dY] = mmd2_and_gradY(X, Y, opts.sigmas);
        totalLoss = totalLoss + mmd2;

        % backprop: Y = W2 * Z  => dL/dW2 = (dL/dY) * Z'
        gradW = gradW + dL_dY * Z';
    end

    % regularization: smooth L1 + L2 on W2
    regL1 = sum(sum( sqrt(W2.^2 + opts.eps) ));
    regL2 = sum(sum( W2.^2 ));
    totalLoss = totalLoss + opts.lambda1 * regL1 + opts.lambda2 * regL2;

    gradW = gradW + opts.lambda1 * (W2 ./ sqrt(W2.^2 + opts.eps)) + opts.lambda2 * (2*W2);

    % chain rule through softplus
    gradU = gradW .* dW_dU;

    % step
    U = U - opts.lr * gradU;

    lossHist(it) = totalLoss;

    if opts.verbose && (mod(it, opts.printEvery)==0 || it==1 || it==opts.maxIters)
        fprintf('[%4d/%4d] loss=%.6g | classes=%d\n', it, opts.maxIters, totalLoss, C);
    end
end

W2 = softplus(U);

info.lossHist = lossHist;
info.keptBehaviors = find(keptBeh);
info.classNames = classNames;
info.opts = opts;

end

% ==================== helpers ====================

function opts = setDefault(opts, field, val)
if ~isfield(opts, field) || isempty(opts.(field))
    opts.(field) = val;
end
end

function Xn = normalizeRows(X, normType, epsv)
Xn = X;
switch lower(normType)
    case 'none'
        return
    case 'zscore'
        mu = mean(Xn, 2);
        sd = std(Xn, 0, 2) + epsv;
        Xn = (Xn - mu) ./ sd;
    case 'madzscore'
        mu = median(Xn, 2);
        madv = median(abs(Xn - mu), 2) + epsv; % MAD
        Xn = (Xn - mu) ./ madv;
    otherwise
        error('Unknown normType: %s', normType);
end
end

function idx = subsampleIdx(idx, maxN)
idx = idx(:);
if numel(idx) > maxN
    p = randperm(numel(idx), maxN);
    idx = idx(p);
end
end

function U = initU_identity(K, eps0)
% Initialize U so that softplus(U) ~ I (diag~1, offdiag~eps0)
U = log(exp(eps0) - 1) * ones(K,K);  % off-diagonal ~ eps0
diagVal = log(exp(1) - 1);           % softplus(diagVal) = 1
for k = 1:K
    U(k,k) = diagVal;
end
end

function y = softplus(x)
y = log1p(exp(-abs(x))) + max(x,0);
end

function s = sigmoid(x)
s = 1 ./ (1 + exp(-x));
end

function [mmd2, dL_dY] = mmd2_and_gradY(X, Y, sigmas)
% Unbiased MMD^2 between samples X (d x n) and Y (d x m), with multi-RBF kernel.
% Returns gradient wrt Y.

[d, n] = size(X);
[~, m] = size(Y);
J = numel(sigmas);

% Kxx
Kxx = 0;
for a = 1:n
    for b = 1:n
        if a == b, continue; end
        Kxx = Kxx + k_rbf_multi(X(:,a), X(:,b), sigmas, J);
    end
end
Kxx = Kxx / (n*(n-1));

% Kyy + grad
Kyy = 0;
dL_dY = zeros(d, m);
for a = 1:m
    for b = 1:m
        if a == b, continue; end
        [kval, dk_dya] = k_rbf_multi_grad_second(Y(:,b), Y(:,a), sigmas, J);
        Kyy = Kyy + kval;
        dL_dY(:,a) = dL_dY(:,a) + dk_dya;
    end
end
Kyy = Kyy / (m*(m-1));
dL_dY = dL_dY * (1/(m*(m-1)));

% Kxy + grad
Kxy = 0;
gradCross = zeros(d,m);
for a = 1:n
    for b = 1:m
        [kval, dk_dyb] = k_rbf_multi_grad_second(X(:,a), Y(:,b), sigmas, J);
        Kxy = Kxy + kval;
        gradCross(:,b) = gradCross(:,b) + dk_dyb;
    end
end
Kxy = Kxy / (n*m);

mmd2 = Kxx + Kyy - 2*Kxy;

% gradient: d/dY [Kyy] - 2 d/dY[Kxy]
dL_dY = dL_dY - 2 * (gradCross / (n*m));
end

function k = k_rbf_multi(u, v, sigmas, J)
diff = u - v;
dist2 = sum(diff.^2);
k = 0;
for j = 1:J
    s = sigmas(j);
    k = k + exp(-dist2 / (2*s*s));
end
k = k / J;
end

function [k, grad_v] = k_rbf_multi_grad_second(u, v, sigmas, J)
% k(u,v) and grad wrt v: exp(..) * (u-v)/s^2
diff = u - v;
dist2 = sum(diff.^2);
k = 0;
grad_v = zeros(size(v));
for j = 1:J
    s = sigmas(j);
    kj = exp(-dist2 / (2*s*s));
    k = k + kj;
    grad_v = grad_v + kj * (diff / (s*s));
end
k = k / J;
grad_v = grad_v / J;
end
