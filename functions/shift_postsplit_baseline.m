function [shifted_comp_time_courses, info] = shift_postsplit_baseline(comp_time_courses, time_train, time_test, opts)
%SHIFT_POSTSPLIT_BASELINE Shift test baseline to match train baseline (row-wise).
%
%   [shifted_comp_time_courses, info] = shift_postsplit_baseline(comp_time_courses, time_train, time_test, opts)
%
% Inputs
%   comp_time_courses : R x T matrix (e.g., C or F), each row is one component trace.
%   time_train        : vector of training frame indices.
%   time_test         : vector of testing/inference frame indices.
%   opts              : optional struct
%       - baseline_prctile (default 10)
%       - shift_strength   (default 1)
%       - nonneg_clip      (default true)

if nargin < 4, opts = struct(); end
if ~isfield(opts, 'baseline_prctile'), opts.baseline_prctile = 10; end
if ~isfield(opts, 'shift_strength'),   opts.shift_strength = 1; end
if ~isfield(opts, 'nonneg_clip'),      opts.nonneg_clip = true; end

assert(ismatrix(comp_time_courses), 'comp_time_courses must be a 2D matrix.');
[R, T] = size(comp_time_courses);

assert(~isempty(time_train), 'time_train cannot be empty.');
assert(~isempty(time_test),  'time_test cannot be empty.');
assert(all(time_train >= 1 & time_train <= T), 'time_train has out-of-range indices.');
assert(all(time_test  >= 1 & time_test  <= T), 'time_test has out-of-range indices.');

% accept fraction or percentile
if opts.baseline_prctile <= 1
    opts.baseline_prctile = 100 * opts.baseline_prctile;
end
opts.baseline_prctile = min(100, max(0, opts.baseline_prctile));
opts.shift_strength   = min(1, max(0, opts.shift_strength));

shifted_comp_time_courses = comp_time_courses;

baseline_train = prctile(comp_time_courses(:, time_train), opts.baseline_prctile, 2);
baseline_test  = prctile(comp_time_courses(:, time_test),  opts.baseline_prctile, 2);

baseline_train = cast(baseline_train, 'like', comp_time_courses);
baseline_test  = cast(baseline_test,  'like', comp_time_courses);

delta = opts.shift_strength * (baseline_train - baseline_test); % R x 1
shifted_comp_time_courses(:, time_test) = shifted_comp_time_courses(:, time_test) + ...
    delta * ones(1, numel(time_test), 'like', comp_time_courses);

if opts.nonneg_clip
    shifted_comp_time_courses = max(shifted_comp_time_courses, cast(0, 'like', comp_time_courses));
end

info = struct();
info.baseline_train = baseline_train;
info.baseline_test = baseline_test;
info.delta = delta;
info.time_train = time_train;
info.time_test = time_test;
info.baseline_prctile = opts.baseline_prctile;
info.shift_strength = opts.shift_strength;
info.n_components = R;
end
