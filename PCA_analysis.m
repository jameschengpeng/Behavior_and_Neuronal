%% 
clear
clc
addpath(genpath('C:\Users\james\CBIL\Astrocyte\scalable_calcium_model_prev'));
addpath(genpath('C:\Users\james\CBIL\Astrocyte\Behavior_and_Neuronal'));
%% read the saved data, you can start here
savefile = "F:\Mouse_behavior_data\D21\downsampled_smoothed_data_all_videos.mat";
saved_data = load(savefile);
dF1_downsampled_smoothed        = saved_data.dF1_downsampled_smoothed;
datOrg1_downsampled_smoothed    = saved_data.datOrg1_downsampled_smoothed;
ethogram_mat_downsampled        = saved_data.ethogram_mat_downsampled;
new_subset_cutting_points       = saved_data.new_subset_cutting_points;
evt_map_downsampled             = saved_data.evt_map_downsampled;
dFF                             = saved_data.dFF;
mask_downsampled                = saved_data.mask_downsampled;
temp_down_factor                = saved_data.temp_down_factor;

X_dFF = reshape(dFF, [], size(dFF, 3)); % 
X_dFF = X_dFF'; % take transpose so X_dFF is of shape T * n_pixels
if min(X_dFF(:)) < 0
    X_dFF = X_dFF - min(X_dFF(:));  
end

new_height = size(dF1_downsampled_smoothed, 1);
new_width = size(dF1_downsampled_smoothed, 2);

%% Step 1: Perform PCA
unmasked_indices = find(mask_downsampled);
unmasked_X_dFF = X_dFF; % T * n_pixels
unmasked_X_dFF = unmasked_X_dFF(:, unmasked_indices);

[coeff, score, latent, tsquared, explained, mu] = pca(unmasked_X_dFF);

% Let p = H*W, n = T
% coeff: p x (n-1 or p) matrix of principal component coefficients. Each
% column of coeff is a spatial principal component

% score: n x (n-1 or p) matrix of principal component scores. Each column
% of score is the temporal activity of a principal component

% latent: eigenvalues (variances) of principal components
% tsquared: Hotelling's T-squared statistic for each observation
% explained: percentage of variance explained by each component
% mu: mean of each variable (used for centering)

%% Step 2: Visualize the variance explained
figure;
subplot(2,2,1);
plot(explained(1:20), 'bo-', 'LineWidth', 2);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Variance Explained by Each PC');
grid on;

subplot(2,2,2);
plot(cumsum(explained(1:20)), 'ro-', 'LineWidth', 2);
xlabel('Principal Component');
ylabel('Cumulative Variance Explained (%)');
title('Cumulative Variance Explained');
yline(90, '--k', '90%');
yline(95, '--k', '95%');
grid on;

subplot(2,2,3);
pareto(explained(1:20));
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Pareto Chart of Variance');

subplot(2,2,4);
semilogy(latent(1:20), 'go-', 'LineWidth', 2);
xlabel('Principal Component');
ylabel('Eigenvalue (log scale)');
title('Scree Plot (Eigenvalues)');
grid on;

%%
component_idx = 3;
[spa_comp, time_course] = extract_spa_comp_time_course(coeff, score, mask_downsampled, component_idx);
fps = 10;
time_axis = (0:(length(time_course)-1)) ./ fps;

figure;
% Top: spatial component
subplot(2,1,1);
imagesc(spa_comp);
axis tight;
colorbar;
title(strcat('Principal Component', num2str(component_idx)));
% Bottom: time course
subplot(2,1,2);
plot(time_axis, time_course, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Time Course');

%%
function [spa_comp, time_course] = extract_spa_comp_time_course(coeff, score, mask, idx)
% Each column of coeff is a spatial principal component
% Each column of score is the temporal activity of a principal component
spa_comp = zeros(size(mask));
unmasked_indices = find(mask);
spa_comp(unmasked_indices) = coeff(:, idx);
time_course = score(:, idx);
end





