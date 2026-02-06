function A2_refined = refineSpatialA(X_dFF2, A2, mask2)
%REFINESPATIALA2_MASKED Refines initial A2 using only valid pixels.
%
% Inputs:
%   X_dFF2  - (n_pixels2 x T) calcium imaging data for mouse2
%   A2      - (n_pixels2 x K) initial mapped spatial footprints
%   mask2   - (height2 x width2) binary mask of valid pixels
%
% Output:
%   A2_refined - (n_pixels2 x K) refined spatial footprints
%
% A2_refined is non-negative and each column has norm=1.

[n_pixels2, K] = size(A2);
assert(size(X_dFF2,1) == n_pixels2, ...
    'X and A2 must have matching pixel dimension.');

% Flatten mask to vector
mask_vec = mask2(:) ~= 0;

% Pre-allocate
A2_refined = zeros(n_pixels2, K);

% Zero out masked X rows
X_masked = zeros(size(X_dFF2));
X_masked(mask_vec,:) = X_dFF2(mask_vec,:);

% Loop over components
for k = 1:K
    % current basis vector
    a_old = A2(:,k);
    
    % enforce non-negativity
    a_old(a_old < 0) = 0;
    
    % projection onto data to get an approximate timecourse
    s_est = max(0, a_old(mask_vec)' * X_masked(mask_vec,:)); % 1×T
    
    if all(s_est == 0)
        % if projection gives zeros (rare), skip refinement
        A2_refined(:,k) = a_old / (norm(a_old)+eps);
        continue
    end
    
    % build nonnegative least squares target for spatial update
    % we solve (valid pixels only):
    %   min_a >= 0 || X_valid - a * s_est ||_F^2
    X_valid = X_masked(mask_vec,:);   % n_valid×T
    b = X_valid * s_est';             % n_valid×1
    
    % solve NNLS: min ||a * (s_est*s_est') - b||_2^2 subject to a>=0
    % since (s_est*s_est') is scalar, we use lsqnonneg
    w = (s_est * s_est');             % scalar
    if w > 0
        a_valid_new = lsqnonneg(w*eye(sum(mask_vec)), b);
    else
        a_valid_new = zeros(sum(mask_vec), 1);
    end
    
    % put back into full vector
    a_new = zeros(n_pixels2,1);
    a_new(mask_vec) = a_valid_new;
    
    % enforce non-negativity
    a_new(a_new < 0) = 0;
    
    % normalize to unit norm
    if norm(a_new) > 0
        a_new = a_new / norm(a_new);
    end
    
    A2_refined(:,k) = a_new;
end

end
