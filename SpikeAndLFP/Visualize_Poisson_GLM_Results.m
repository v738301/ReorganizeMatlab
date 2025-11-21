%% ========================================================================
%  VISUALIZE POISSON GLM RESULTS
%  ========================================================================
%
%  Comprehensive visualization of nested Poisson GLM models
%
%  This script creates detailed visualizations of the fitted GLM models:
%    - Model comparison (deviance explained across 5 nested models)
%    - Temporal kernels for reward events (IR1ON, IR2ON, WP1ON, WP2ON)
%    - Temporal kernels for aversive events
%    - Spike history effects (autoregressive dynamics)
%    - Spatial tuning (2D XY position + 1D Z height)
%    - Speed tuning (X, Y, Z velocities)
%    - Breathing modulation (8Hz and 1.5Hz bands)
%    - Statistical significance of predictors
%    - Population-level summaries
%
%  Input: Unit_GLM_Nested_Results.mat
%  Output: Figures saved to GLM_Figures/ directory
%% ========================================================================

clear all
close all

fprintf('\n=== VISUALIZING POISSON GLM RESULTS ===\n\n');

%% Load results
fprintf('Loading results...\n');
if ~exist('Unit_GLM_Nested_Results.mat', 'file')
    error('Unit_GLM_Nested_Results.mat not found! Run Unit_Poisson_GLM_Analysis.m first.');
end

load('Unit_GLM_Nested_Results.mat', 'all_results', 'config');
n_units = length(all_results);
fprintf('✓ Loaded %d units\n\n', n_units);

% Create output directory
output_dir = 'GLM_Figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Extract predictor structure information
fprintf('Extracting predictor structure...\n');
predictor_info = all_results(1).predictor_info;
predictor_names = predictor_info.predictor_names;

% Parse predictor indices
pred_idx = parsePredictor Indices(predictor_info, config);
fprintf('✓ Predictor structure extracted\n\n');

%% ========================================================================
%  1. POPULATION-LEVEL MODEL COMPARISON
%% ========================================================================
fprintf('Creating model comparison figure...\n');

fig1 = figure('Position', [100, 100, 1400, 500]);

% Extract deviance explained for all units
dev_exp = zeros(n_units, 5);
for i = 1:n_units
    dev_exp(i, 1) = all_results(i).model1.deviance_explained;
    dev_exp(i, 2) = all_results(i).model2.deviance_explained;
    dev_exp(i, 3) = all_results(i).model3.deviance_explained;
    dev_exp(i, 4) = all_results(i).model4.deviance_explained;
    dev_exp(i, 5) = all_results(i).model5.deviance_explained;
end

% Panel A: Average deviance explained
subplot(1, 3, 1);
mean_dev = mean(dev_exp, 1);
sem_dev = std(dev_exp, 0, 1) / sqrt(n_units);
b = bar(1:5, mean_dev, 'FaceColor', [0.3 0.5 0.8]);
hold on;
errorbar(1:5, mean_dev, sem_dev, 'k.', 'LineWidth', 1.5);
ylabel('Deviance Explained (%)');
xlabel('Model');
title(sprintf('Model Comparison (n=%d units)', n_units));
set(gca, 'XTickLabel', {'Hist', '+Events', '+Spatial', '+Speed', '+Breath'});
xtickangle(45);
grid on;
ylim([0, max(mean_dev) * 1.2]);

% Panel B: Incremental contributions
subplot(1, 3, 2);
incremental = [mean_dev(1), diff(mean_dev)];
colors = [0.7 0.7 0.7; 0.8 0.4 0.4; 0.4 0.7 0.4; 0.4 0.4 0.8; 0.8 0.6 0.3];
b = bar(1:5, incremental);
b.FaceColor = 'flat';
for k = 1:5
    b.CData(k,:) = colors(k,:);
end
ylabel('Incremental Deviance Explained (%)');
xlabel('Model Component');
title('Contribution of Each Component');
set(gca, 'XTickLabel', {'History', 'Events', 'Spatial', 'Speed', 'Breathing'});
xtickangle(45);
grid on;

% Panel C: Distribution of deviance explained (full model)
subplot(1, 3, 3);
histogram(dev_exp(:, 5), 20, 'FaceColor', [0.3 0.5 0.8]);
xlabel('Deviance Explained (%)');
ylabel('Number of Units');
title('Distribution (Full Model)');
hold on;
xline(mean(dev_exp(:, 5)), 'r--', 'LineWidth', 2);
text(mean(dev_exp(:, 5)), max(ylim)*0.9, sprintf('Mean: %.1f%%', mean(dev_exp(:, 5))), ...
    'Color', 'r', 'FontWeight', 'bold', 'FontSize', 10);
grid on;

saveas(fig1, fullfile(output_dir, 'Fig1_Model_Comparison.png'));
fprintf('✓ Saved: Fig1_Model_Comparison.png\n\n');

%% ========================================================================
%  2. POPULATION AVERAGE: EVENT KERNELS
%% ========================================================================
fprintf('Creating event kernel figure...\n');

fig2 = figure('Position', [100, 100, 1600, 800]);

% Extract coefficients for reward events (Gaussian kernels)
reward_event_names = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON'};
n_reward_kernels = config.n_reward_kernels;

% Time axis for reward kernels (symmetric -2 to +2 sec)
reward_time = linspace(-config.reward_window_pre, config.reward_window_post, n_reward_kernels);

for ev = 1:4
    subplot(2, 4, ev);
    idx_start = pred_idx.reward_events(ev, 1);
    idx_end = pred_idx.reward_events(ev, 2);

    % Extract coefficients across all units (Model 5)
    coefs = zeros(n_units, n_reward_kernels);
    for i = 1:n_units
        coefs(i, :) = all_results(i).model5.coefficients(idx_start:idx_end);
    end

    % Plot mean ± SEM
    mean_coef = mean(coefs, 1);
    sem_coef = std(coefs, 0, 1) / sqrt(n_units);

    shadedErrorBar(reward_time, mean_coef, sem_coef, 'lineprops', {'Color', [0.8 0.3 0.3], 'LineWidth', 2});
    hold on;
    xline(0, 'k--', 'LineWidth', 1.5);
    yline(0, 'k:', 'LineWidth', 1);
    xlabel('Time from event (s)');
    ylabel('GLM Coefficient');
    title(reward_event_names{ev});
    grid on;
    xlim([-config.reward_window_pre, config.reward_window_post]);
end

% Aversive event (raised cosine basis, causal only)
subplot(2, 4, 5);
n_aversive_basis = config.n_basis_funcs;
aversive_time = linspace(0, config.aversive_window_post, n_aversive_basis);

idx_start = pred_idx.aversive(1);
idx_end = pred_idx.aversive(2);

coefs_aversive = zeros(n_units, n_aversive_basis);
for i = 1:n_units
    coefs_aversive(i, :) = all_results(i).model5.coefficients(idx_start:idx_end);
end

mean_coef = mean(coefs_aversive, 1);
sem_coef = std(coefs_aversive, 0, 1) / sqrt(n_units);

shadedErrorBar(aversive_time, mean_coef, sem_coef, 'lineprops', {'Color', [0.3 0.3 0.8], 'LineWidth', 2});
hold on;
xline(0, 'k--', 'LineWidth', 1.5);
yline(0, 'k:', 'LineWidth', 1);
xlabel('Time from event (s)');
ylabel('GLM Coefficient');
title('Aversive Sound');
grid on;
xlim([0, config.aversive_window_post]);

% Spike history
subplot(2, 4, 6);
n_lags = config.history_lags;
lag_times = (1:n_lags) * config.bin_size * 1000;  % in ms

coefs_history = zeros(n_units, n_lags);
for i = 1:n_units
    coefs_history(i, :) = all_results(i).model5.coefficients(pred_idx.spike_history(1):pred_idx.spike_history(2));
end

mean_coef = mean(coefs_history, 1);
sem_coef = std(coefs_history, 0, 1) / sqrt(n_units);

bar(lag_times, mean_coef, 'FaceColor', [0.5 0.5 0.5]);
hold on;
errorbar(lag_times, mean_coef, sem_coef, 'k.', 'LineWidth', 1.5);
xlabel('Lag (ms)');
ylabel('GLM Coefficient');
title('Spike History');
grid on;
yline(0, 'k:', 'LineWidth', 1);

% Fraction of units with significant effects
subplot(2, 4, 7:8);
sig_threshold = 0.01;
sig_counts = zeros(1, 5);
labels = {'History', 'Events', 'Spatial', 'Speed', 'Breathing'};

for i = 1:n_units
    % History: any lag significant
    if any(all_results(i).model1.p_values(2:end) < sig_threshold)
        sig_counts(1) = sig_counts(1) + 1;
    end

    % Events: any event kernel significant
    event_pvals = all_results(i).model5.p_values(pred_idx.all_events(1):pred_idx.all_events(2));
    if any(event_pvals < sig_threshold)
        sig_counts(2) = sig_counts(2) + 1;
    end

    % Spatial: any spatial bin significant
    spatial_pvals = all_results(i).model5.p_values(pred_idx.spatial(1):pred_idx.spatial(2));
    if any(spatial_pvals < sig_threshold)
        sig_counts(3) = sig_counts(3) + 1;
    end

    % Speed: any speed kernel significant
    speed_pvals = all_results(i).model5.p_values(pred_idx.speeds(1):pred_idx.speeds(2));
    if any(speed_pvals < sig_threshold)
        sig_counts(4) = sig_counts(4) + 1;
    end

    % Breathing: any breathing kernel significant
    breathing_pvals = all_results(i).model5.p_values(pred_idx.breathing(1):pred_idx.breathing(2));
    if any(breathing_pvals < sig_threshold)
        sig_counts(5) = sig_counts(5) + 1;
    end
end

sig_fraction = (sig_counts / n_units) * 100;
b = bar(1:5, sig_fraction);
b.FaceColor = 'flat';
colors = [0.7 0.7 0.7; 0.8 0.4 0.4; 0.4 0.7 0.4; 0.4 0.4 0.8; 0.8 0.6 0.3];
for k = 1:5
    b.CData(k,:) = colors(k,:);
end
ylabel('% Units with Significant Effect');
xlabel('Predictor Type');
title(sprintf('Significant Effects (p < %.2f)', sig_threshold));
set(gca, 'XTickLabel', labels);
xtickangle(45);
grid on;
ylim([0, 100]);

saveas(fig2, fullfile(output_dir, 'Fig2_Event_Kernels_Population.png'));
fprintf('✓ Saved: Fig2_Event_Kernels_Population.png\n\n');

%% ========================================================================
%  3. EXAMPLE UNIT: DETAILED VIEW
%% ========================================================================
fprintf('Creating example unit figure...\n');

% Find a good example unit (high deviance explained in full model)
[~, example_idx] = max(dev_exp(:, 5));
example_unit = all_results(example_idx);

fig3 = figure('Position', [100, 100, 1600, 1200]);
sgtitle(sprintf('Example Unit: %s, Unit %d (Dev.Exp=%.1f%%)', ...
    example_unit.session_name, example_unit.unit_idx, example_unit.model5.deviance_explained), ...
    'Interpreter', 'none', 'FontSize', 14, 'FontWeight', 'bold');

% Panel 1: Model comparison for this unit
subplot(3, 4, 1);
unit_dev = [example_unit.model1.deviance_explained, example_unit.model2.deviance_explained, ...
            example_unit.model3.deviance_explained, example_unit.model4.deviance_explained, ...
            example_unit.model5.deviance_explained];
bar(1:5, unit_dev, 'FaceColor', [0.3 0.5 0.8]);
ylabel('Deviance Explained (%)');
xlabel('Model');
title('Nested Model Performance');
set(gca, 'XTickLabel', {'Hist', '+Evt', '+Sp', '+Spd', '+Br'});
xtickangle(45);
grid on;

% Panel 2-5: Reward event kernels
for ev = 1:4
    subplot(3, 4, ev+1);
    idx_start = pred_idx.reward_events(ev, 1);
    idx_end = pred_idx.reward_events(ev, 2);

    coefs = example_unit.model5.coefficients(idx_start:idx_end);
    plot(reward_time, coefs, 'Color', [0.8 0.3 0.3], 'LineWidth', 2);
    hold on;
    xline(0, 'k--', 'LineWidth', 1.5);
    yline(0, 'k:', 'LineWidth', 1);
    xlabel('Time (s)');
    ylabel('Coefficient');
    title(reward_event_names{ev});
    grid on;
    xlim([-config.reward_window_pre, config.reward_window_post]);
end

% Panel 6: Aversive kernel
subplot(3, 4, 6);
idx_start = pred_idx.aversive(1);
idx_end = pred_idx.aversive(2);
coefs = example_unit.model5.coefficients(idx_start:idx_end);
plot(aversive_time, coefs, 'Color', [0.3 0.3 0.8], 'LineWidth', 2);
hold on;
xline(0, 'k--', 'LineWidth', 1.5);
yline(0, 'k:', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Coefficient');
title('Aversive Sound');
grid on;

% Panel 7: Spike history
subplot(3, 4, 7);
coefs_hist = example_unit.model5.coefficients(pred_idx.spike_history(1):pred_idx.spike_history(2));
bar(lag_times, coefs_hist, 'FaceColor', [0.5 0.5 0.5]);
xlabel('Lag (ms)');
ylabel('Coefficient');
title('Spike History');
grid on;
yline(0, 'k:', 'LineWidth', 1);

% Panel 8: 2D XY Spatial tuning
subplot(3, 4, 8);
n_xy_bins = predictor_info.n_spatial_bins_xy;
xy_coefs = example_unit.model5.coefficients(pred_idx.spatial_xy(1):pred_idx.spatial_xy(2));
xy_map = reshape(xy_coefs, [n_xy_bins, n_xy_bins]);
imagesc(xy_map);
colorbar;
colormap(gca, 'jet');
axis square;
title('2D Spatial Tuning (XY)');
xlabel('X bin');
ylabel('Y bin');

% Panel 9: 1D Z Spatial tuning
subplot(3, 4, 9);
z_coefs = example_unit.model5.coefficients(pred_idx.spatial_z(1):pred_idx.spatial_z(2));
bar(1:length(z_coefs), z_coefs, 'FaceColor', [0.4 0.7 0.4]);
xlabel('Z bin (height)');
ylabel('Coefficient');
title('1D Spatial Tuning (Z)');
grid on;
yline(0, 'k:', 'LineWidth', 1);

% Panel 10: Speed tuning (average across kernels)
subplot(3, 4, 10);
speed_names = {'X', 'Y', 'Z'};
n_kernels = config.n_continuous_kernels;
speed_coefs_avg = zeros(1, 3);
for dim = 1:3
    idx_start = pred_idx.speeds_individual{dim}(1);
    idx_end = pred_idx.speeds_individual{dim}(2);
    speed_coefs_avg(dim) = mean(example_unit.model5.coefficients(idx_start:idx_end));
end
bar(1:3, speed_coefs_avg, 'FaceColor', [0.4 0.4 0.8]);
set(gca, 'XTickLabel', speed_names);
ylabel('Mean Coefficient');
title('Speed Tuning (avg across kernels)');
grid on;
yline(0, 'k:', 'LineWidth', 1);

% Panel 11: Breathing 8Hz
subplot(3, 4, 11);
breath_8Hz_coefs = example_unit.model5.coefficients(pred_idx.breathing_8Hz(1):pred_idx.breathing_8Hz(2));
breath_time = linspace(-config.continuous_window, config.continuous_window, n_kernels);
plot(breath_time, breath_8Hz_coefs, 'Color', [0.8 0.6 0.3], 'LineWidth', 2, 'Marker', 'o');
xlabel('Time lag (s)');
ylabel('Coefficient');
title('Breathing 8Hz Modulation');
grid on;
yline(0, 'k:', 'LineWidth', 1);
xline(0, 'k--', 'LineWidth', 1);

% Panel 12: Breathing 1.5Hz
subplot(3, 4, 12);
breath_1p5Hz_coefs = example_unit.model5.coefficients(pred_idx.breathing_1p5Hz(1):pred_idx.breathing_1p5Hz(2));
plot(breath_time, breath_1p5Hz_coefs, 'Color', [0.6 0.4 0.6], 'LineWidth', 2, 'Marker', 'o');
xlabel('Time lag (s)');
ylabel('Coefficient');
title('Breathing 1.5Hz Modulation');
grid on;
yline(0, 'k:', 'LineWidth', 1);
xline(0, 'k--', 'LineWidth', 1);

saveas(fig3, fullfile(output_dir, sprintf('Fig3_Example_Unit_%d.png', example_idx)));
fprintf('✓ Saved: Fig3_Example_Unit_%d.png\n\n', example_idx);

%% ========================================================================
%  4. POPULATION SPATIAL TUNING
%% ========================================================================
fprintf('Creating population spatial tuning figure...\n');

fig4 = figure('Position', [100, 100, 1200, 500]);

% Average XY spatial maps (only units with positive deviance)
subplot(1, 2, 1);
n_xy_bins = predictor_info.n_spatial_bins_xy;
avg_xy_map = zeros(n_xy_bins, n_xy_bins);
count = 0;

for i = 1:n_units
    if dev_exp(i, 5) > 0
        xy_coefs = all_results(i).model5.coefficients(pred_idx.spatial_xy(1):pred_idx.spatial_xy(2));
        xy_map = reshape(xy_coefs, [n_xy_bins, n_xy_bins]);
        avg_xy_map = avg_xy_map + xy_map;
        count = count + 1;
    end
end
avg_xy_map = avg_xy_map / count;

imagesc(avg_xy_map);
colorbar;
colormap(gca, 'jet');
axis square;
title(sprintf('Population Average 2D Spatial Tuning (n=%d)', count));
xlabel('X bin');
ylabel('Y bin');

% Average Z spatial tuning
subplot(1, 2, 2);
n_z_bins = predictor_info.n_spatial_bins_z;
avg_z = zeros(n_z_bins, 1);
sem_z = zeros(n_z_bins, 1);
all_z_coefs = zeros(n_units, n_z_bins);

for i = 1:n_units
    all_z_coefs(i, :) = all_results(i).model5.coefficients(pred_idx.spatial_z(1):pred_idx.spatial_z(2));
end

avg_z = mean(all_z_coefs, 1);
sem_z = std(all_z_coefs, 0, 1) / sqrt(n_units);

bar(1:n_z_bins, avg_z, 'FaceColor', [0.4 0.7 0.4]);
hold on;
errorbar(1:n_z_bins, avg_z, sem_z, 'k.', 'LineWidth', 1.5);
xlabel('Z bin (height, 1=bottom, 5=top)');
ylabel('Mean GLM Coefficient');
title(sprintf('Population Average Z Tuning (n=%d)', n_units));
grid on;
yline(0, 'k:', 'LineWidth', 1);

saveas(fig4, fullfile(output_dir, 'Fig4_Population_Spatial_Tuning.png'));
fprintf('✓ Saved: Fig4_Population_Spatial_Tuning.png\n\n');

%% ========================================================================
%  5. MODEL SELECTION SUMMARY
%% ========================================================================
fprintf('Creating model selection summary...\n');

fig5 = figure('Position', [100, 100, 1400, 500]);

% Extract AIC and BIC for all units
AIC_all = zeros(n_units, 5);
BIC_all = zeros(n_units, 5);

for i = 1:n_units
    AIC_all(i, :) = [all_results(i).model1.AIC, all_results(i).model2.AIC, ...
                     all_results(i).model3.AIC, all_results(i).model4.AIC, ...
                     all_results(i).model5.AIC];
    BIC_all(i, :) = [all_results(i).model1.BIC, all_results(i).model2.BIC, ...
                     all_results(i).model3.BIC, all_results(i).model4.BIC, ...
                     all_results(i).model5.BIC];
end

% Find best model for each unit
[~, best_AIC] = min(AIC_all, [], 2);
[~, best_BIC] = min(BIC_all, [], 2);

% Panel A: AIC-based model selection
subplot(1, 3, 1);
histogram(best_AIC, 0.5:5.5, 'FaceColor', [0.3 0.5 0.8]);
xlabel('Best Model (by AIC)');
ylabel('Number of Units');
title('Model Selection (AIC)');
set(gca, 'XTick', 1:5);
set(gca, 'XTickLabel', {'1:Hist', '2:+Evt', '3:+Sp', '4:+Spd', '5:+Br'});
xtickangle(45);
grid on;

% Panel B: BIC-based model selection
subplot(1, 3, 2);
histogram(best_BIC, 0.5:5.5, 'FaceColor', [0.8 0.4 0.4]);
xlabel('Best Model (by BIC)');
ylabel('Number of Units');
title('Model Selection (BIC, more conservative)');
set(gca, 'XTick', 1:5);
set(gca, 'XTickLabel', {'1:Hist', '2:+Evt', '3:+Sp', '4:+Spd', '5:+Br'});
xtickangle(45);
grid on;

% Panel C: Likelihood ratio test significance
subplot(1, 3, 3);
LRT_sig = zeros(4, 1);  % Models 2-5 compared to previous
for model_idx = 2:5
    sig_count = 0;
    for i = 1:n_units
        switch model_idx
            case 2
                pval = all_results(i).model2.LRT_p_value;
            case 3
                pval = all_results(i).model3.LRT_p_value;
            case 4
                pval = all_results(i).model4.LRT_p_value;
            case 5
                pval = all_results(i).model5.LRT_p_value;
        end
        if pval < 0.05
            sig_count = sig_count + 1;
        end
    end
    LRT_sig(model_idx - 1) = (sig_count / n_units) * 100;
end

bar(2:5, LRT_sig, 'FaceColor', [0.4 0.7 0.4]);
xlabel('Model');
ylabel('% Units Significantly Better');
title('Likelihood Ratio Test (p < 0.05)');
set(gca, 'XTick', 2:5);
set(gca, 'XTickLabel', {'2:+Evt', '3:+Sp', '4:+Spd', '5:+Br'});
xtickangle(45);
grid on;
ylim([0, 100]);
yline(50, 'r--', 'LineWidth', 1.5);

saveas(fig5, fullfile(output_dir, 'Fig5_Model_Selection_Summary.png'));
fprintf('✓ Saved: Fig5_Model_Selection_Summary.png\n\n');

%% ========================================================================
%  SUMMARY STATISTICS
%% ========================================================================
fprintf('\n=== VISUALIZATION COMPLETE ===\n');
fprintf('All figures saved to: %s/\n', output_dir);
fprintf('\nKey Statistics:\n');
fprintf('  Total units analyzed: %d\n', n_units);
fprintf('  Mean deviance explained (full model): %.2f%% ± %.2f%%\n', ...
    mean(dev_exp(:, 5)), std(dev_exp(:, 5)));
fprintf('  Best model by AIC: Model %d (mode)\n', mode(best_AIC));
fprintf('  Best model by BIC: Model %d (mode)\n', mode(best_BIC));
fprintf('  Units with >20%% deviance explained: %d (%.1f%%)\n', ...
    sum(dev_exp(:, 5) > 20), sum(dev_exp(:, 5) > 20) / n_units * 100);
fprintf('\n');

%% ========================================================================
%  HELPER FUNCTIONS
%% ========================================================================

function pred_idx = parsePredictorIndices(predictor_info, config)
% Parse the predictor indices from predictor_info structure
%
% Returns a struct with index ranges for each predictor type

    pred_idx = struct();

    % Intercept
    pred_idx.intercept = 1;

    % Spike history (lags 1 to n_lags)
    n_lags = config.history_lags;
    pred_idx.spike_history = [2, 1 + n_lags];

    % Current index
    idx = 1 + n_lags + 1;  % After intercept + spike history

    % Reward events (4 events × n_reward_kernels)
    n_reward_kernels = config.n_reward_kernels;
    pred_idx.reward_events = zeros(4, 2);
    reward_names = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON'};
    for ev = 1:4
        pred_idx.reward_events(ev, :) = [idx, idx + n_reward_kernels - 1];
        idx = idx + n_reward_kernels;
    end

    % Aversive event (raised cosine basis)
    n_aversive_basis = config.n_basis_funcs;
    pred_idx.aversive = [idx, idx + n_aversive_basis - 1];
    idx = idx + n_aversive_basis;

    % All events combined
    pred_idx.all_events = [pred_idx.reward_events(1, 1), pred_idx.aversive(2)];

    % Coordinates (X, Y, Z each with n_continuous_kernels)
    n_kernels = config.n_continuous_kernels;
    pred_idx.coordinates = [idx, idx + 3*n_kernels - 1];
    pred_idx.x_coord = [idx, idx + n_kernels - 1];
    idx = idx + n_kernels;
    pred_idx.y_coord = [idx, idx + n_kernels - 1];
    idx = idx + n_kernels;
    pred_idx.z_coord = [idx, idx + n_kernels - 1];
    idx = idx + n_kernels;

    % Spatial (XY: 10×10 + Z: 5)
    n_xy_bins = predictor_info.n_spatial_bins_xy;
    n_z_bins = predictor_info.n_spatial_bins_z;
    pred_idx.spatial_xy = [idx, idx + n_xy_bins*n_xy_bins - 1];
    idx = idx + n_xy_bins*n_xy_bins;
    pred_idx.spatial_z = [idx, idx + n_z_bins - 1];
    idx = idx + n_z_bins;
    pred_idx.spatial = [pred_idx.spatial_xy(1), pred_idx.spatial_z(2)];

    % Speeds (X, Y, Z each with n_continuous_kernels)
    pred_idx.speeds = [idx, idx + 3*n_kernels - 1];
    pred_idx.speeds_individual = cell(3, 1);
    pred_idx.speeds_individual{1} = [idx, idx + n_kernels - 1];  % X speed
    idx = idx + n_kernels;
    pred_idx.speeds_individual{2} = [idx, idx + n_kernels - 1];  % Y speed
    idx = idx + n_kernels;
    pred_idx.speeds_individual{3} = [idx, idx + n_kernels - 1];  % Z speed
    idx = idx + n_kernels;

    % Breathing (8Hz and 1.5Hz, each with n_continuous_kernels)
    pred_idx.breathing_8Hz = [idx, idx + n_kernels - 1];
    idx = idx + n_kernels;
    pred_idx.breathing_1p5Hz = [idx, idx + n_kernels - 1];
    idx = idx + n_kernels;
    pred_idx.breathing = [pred_idx.breathing_8Hz(1), pred_idx.breathing_1p5Hz(2)];
end
