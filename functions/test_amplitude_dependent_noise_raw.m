function stats = test_amplitude_dependent_noise_raw(X, n_bins, mask, noise_map)
%TEST_AMPLITUDE_DEPENDENT_NOISE_RAW  Raw-data test for amplitude-dependent noise.
%
%   stats = test_amplitude_dependent_noise_raw(X)
%   stats = test_amplitude_dependent_noise_raw(X, n_bins)
%   stats = test_amplitude_dependent_noise_raw(X, n_bins, mask)
%   stats = test_amplitude_dependent_noise_raw(X, n_bins, mask, noise_map)
%
% For each pixel p, this function:
%   1) divides the raw trace X_p(t) into n_bins quantile intervals,
%   2) finds adjacent frame pairs (t, t+1) for which both samples fall into
%      the same quantile interval,
%   3) estimates the within-bin noise variance using first differences,
%      sigma^2 \approx (1.4826 * MAD(diff))^2 / 2.
%
% The implementation is designed for large single-precision movies with many
% active pixels and thousands of frames. All masked-in pixels are processed
% together in a vectorized pass, which is appropriate on high-memory systems.
% It is still an approximation: if the latent signal changes rapidly even
% within a quantile bin, the estimate can be inflated by signal dynamics.
%
% Inputs:
%   X      : P x T data matrix
%   n_bins : number of per-pixel quantile bins (default 10)
%   mask   : optional P-vector / HxW logical mask. Pixels outside the mask
%            are excluded and assigned NaN in pixelwise outputs.
%   noise_map : optional P-vector / HxW map of baseline noise variance. If
%               provided, per-pixel binwise variance estimates are divided by
%               this map before pooling, so the normalized pooled curve
%               reflects relative variance change after removing spatial
%               heteroscedasticity.
%
% Output:
%   stats is a struct with fields:
%       noise_var_by_bin        - P x n_bins estimated noise variance
%       noise_var_by_bin_norm   - P x n_bins variance normalized by noise_map
%       sample_count_by_bin     - P x n_bins number of adjacent pairs used
%       signal_level_by_bin     - P x n_bins mean raw signal in each bin
%       global_signal_level     - n_bins x 1 weighted mean signal level
%       global_noise_var        - n_bins x 1 weighted mean noise variance
%       global_noise_var_norm   - n_bins x 1 weighted mean normalized variance
%       global_sample_count     - n_bins x 1 total adjacent-pair counts
%       quantile_levels         - 1 x (n_bins+1) quantile levels in [0,1]
%
% A rising global_noise_var curve suggests amplitude-dependent noise.

    narginchk(1, 4);
    if nargin < 2 || isempty(n_bins)
        n_bins = 10;
    end

    [P, T] = size(X);
    assert(isscalar(n_bins) && n_bins == floor(n_bins) && n_bins >= 2, ...
        'n_bins must be an integer >= 2.');
    assert(T >= 2, 'X must have at least two time points.');

    if nargin < 3 || isempty(mask)
        mask_vec = true(P, 1);
    else
        mask_vec = reshape(mask, [], 1) ~= 0;
        assert(numel(mask_vec) == P, 'mask must have length P or size HxW with H*W = P.');
    end

    if nargin < 4 || isempty(noise_map)
        noise_map_vec = [];
    else
        noise_map_vec = reshape(noise_map, [], 1);
        assert(numel(noise_map_vec) == P, 'noise_map must have length P or size HxW with H*W = P.');
        noise_map_vec = double(noise_map_vec);
        noise_map_vec(~mask_vec) = NaN;
        noise_map_vec(noise_map_vec <= 0) = NaN;
    end

    noise_var_by_bin = NaN(P, n_bins, 'single');
    noise_var_by_bin_norm = NaN(P, n_bins, 'single');
    sample_count_by_bin = zeros(P, n_bins, 'uint32');
    signal_level_by_bin = NaN(P, n_bins, 'single');
    quantile_levels = linspace(0, 1, n_bins + 1);

    active_rows = find(mask_vec);
    X_act = X(active_rows, :);
    D_act = diff(X_act, 1, 2);

    % MATLAB versions differ in how prctile lays out the quantile dimension.
    % Convert explicitly to [num_active_pixels x (n_bins+1)].
    edges = prctile(X_act, quantile_levels * 100, 2);
    if size(edges, 1) ~= numel(active_rows) && size(edges, 2) == numel(active_rows)
        edges = edges';
    end
    assert(size(edges, 1) == numel(active_rows) && size(edges, 2) == (n_bins + 1), ...
        'Unexpected size returned by prctile for per-pixel quantile edges.');

    for b = 1:n_bins
        lower = edges(:, b);
        upper = edges(:, b + 1);
        if b < n_bins
            in_bin = bsxfun(@ge, X_act, lower) & bsxfun(@lt, X_act, upper);
        else
            in_bin = bsxfun(@ge, X_act, lower) & bsxfun(@le, X_act, upper);
        end

        in_bin_count = sum(in_bin, 2);
        signal_sum = sum(X_act .* cast(in_bin, 'like', X_act), 2);
        signal_mean = signal_sum ./ max(cast(in_bin_count, 'like', signal_sum), 1);
        signal_mean(in_bin_count == 0) = NaN;
        signal_level_by_bin(active_rows, b) = single(signal_mean);

        pair_idx = in_bin(:, 1:(T - 1)) & in_bin(:, 2:T);
        pair_count = sum(pair_idx, 2);
        sample_count_by_bin(active_rows, b) = uint32(pair_count);
        if ~any(pair_count >= 2)
            continue;
        end

        D_bin = D_act;
        D_bin(~pair_idx) = NaN;
        med_d = median(D_bin, 2, 'omitnan');
        mad_d = median(abs(bsxfun(@minus, D_bin, med_d)), 2, 'omitnan');
        std_d = 1.4826 * mad_d;
        noise_var = (std_d .^ 2) / 2;
        noise_var(pair_count < 2) = NaN;
        noise_var_by_bin(active_rows, b) = single(noise_var);
        if ~isempty(noise_map_vec)
            noise_var_by_bin_norm(active_rows, b) = single(noise_var ./ noise_map_vec(active_rows));
        end
    end

    global_signal_level = NaN(n_bins, 1);
    global_noise_var = NaN(n_bins, 1);
    global_noise_var_norm = NaN(n_bins, 1);
    global_sample_count = double(sum(sample_count_by_bin, 1))';
    for b = 1:n_bins
        valid = mask_vec & ~isnan(noise_var_by_bin(:, b)) & sample_count_by_bin(:, b) > 0;
        if ~any(valid)
            continue;
        end

        weights = double(sample_count_by_bin(valid, b));
        signal_vals = double(signal_level_by_bin(valid, b));
        noise_vals = double(noise_var_by_bin(valid, b));
        global_signal_level(b) = sum(signal_vals .* weights) / sum(weights);
        global_noise_var(b) = sum(noise_vals .* weights) / sum(weights);

        if ~isempty(noise_map_vec)
            valid_norm = valid & ~isnan(noise_var_by_bin_norm(:, b));
            if any(valid_norm)
                weights_norm = double(sample_count_by_bin(valid_norm, b));
                noise_vals_norm = double(noise_var_by_bin_norm(valid_norm, b));
                global_noise_var_norm(b) = sum(noise_vals_norm .* weights_norm) / sum(weights_norm);
            end
        end
    end

    stats = struct();
    stats.noise_var_by_bin = noise_var_by_bin;
    stats.noise_var_by_bin_norm = noise_var_by_bin_norm;
    stats.sample_count_by_bin = sample_count_by_bin;
    stats.signal_level_by_bin = signal_level_by_bin;
    stats.global_signal_level = global_signal_level;
    stats.global_noise_var = global_noise_var;
    stats.global_noise_var_norm = global_noise_var_norm;
    stats.global_sample_count = global_sample_count;
    stats.quantile_levels = quantile_levels;
    stats.noise_map = noise_map_vec;
end