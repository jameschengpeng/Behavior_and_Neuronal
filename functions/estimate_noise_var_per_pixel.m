function noise_var = estimate_noise_var_per_pixel(X)
%ESTIMATE_NOISE_VAR_PER_PIXEL  Robust per-pixel noise variance via temporal differencing.
%
% For each pixel (row of X), first-order temporal differences are computed:
%
%   d_p(t) = X_p(t+1) - X_p(t)
%
% Under the model X_p(t) = b_p*f(t) + S_p(t) + eps_p(t), with eps_p(t) iid
% N(0, sigma_p^2), the differences are:
%
%   d_p(t) = b_p*(f(t+1)-f(t)) + (S_p(t+1)-S_p(t)) + (eps_p(t+1)-eps_p(t))
%
% If the background and calcium signal are slowly varying relative to the
% frame rate, the dominant term is eps_p(t+1)-eps_p(t) ~ N(0, 2*sigma_p^2).
% The noise standard deviation of d_p is therefore sqrt(2)*sigma_p, so:
%
%   sigma_p^2 = (1.4826 * MAD(d_p))^2 / 2
%
% MAD-based estimation is used throughout so that the sparse large differences
% caused by calcium event onsets/offsets do not inflate the estimate.
%
% Input
%   X         : Pm x T matrix (active pixels x frames), single or double
%
% Output
%   noise_var : Pm x 1 vector of per-pixel noise variance estimates

if size(X, 2) < 2
    noise_var = zeros(size(X, 1), 1, 'like', X);
    return;
end

D = double(diff(X, 1, 2));   % Pm x (T-1): first-order temporal differences

% Robust std of d_p via MAD: std_d_p = 1.4826 * MAD(d_p)
% Then sigma_p^2 = std_d_p^2 / 2
med_D   = median(D, 2);                         % Pm x 1
mad_D   = median(abs(D - med_D), 2);            % Pm x 1
std_d   = 1.4826 * mad_D;                       % Pm x 1, estimates sqrt(2)*sigma_p
noise_var = (std_d .^ 2) / 2;                   % Pm x 1, estimates sigma_p^2

noise_var = cast(noise_var, 'like', X);
end
