%% ========================================================================
%  VISUALIZE GLM RESULTS
%  ========================================================================
%
%  This script visualizes the Poisson GLM results including:
%    1. Coefficient heatmaps across units
%    2. Feature importance analysis
%    3. Temporal filters for event predictors
%    4. Cluster-specific summaries (if clustering available)
%    5. Model performance metrics
%    6. Example units with predictions vs actual
%
%  Input: Unit_GLM_Results.mat (from Unit_Poisson_GLM_Analysis.m)
%  Optional: unit_cluster_assignments.mat (from clustering analysis)
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: LOAD RESULTS
%% ========================================================================

fprintf('=== VISUALIZING GLM RESULTS ===\n\n');

% Load GLM results
if ~exist('Unit_GLM_Results.mat', 'file')
    error('Unit_GLM_Results.mat not found. Please run Unit_Poisson_GLM_Analysis.m first.');
end

fprintf('Loading GLM results...\n');
load('Unit_GLM_Results.mat', 'results');

glm_results = results.glm_results;
config = results.config;
predictor_names = results.predictor_names;
n_units = length(glm_results);

fprintf('  Loaded results for %d units\n', n_units);

% Try to load clustering assignments
has_clusters = false;
if exist('unit_cluster_assignments.mat', 'file')
    fprintf('Loading cluster assignments...\n');
    cluster_data = load('unit_cluster_assignments.mat');
    has_clusters = true;
    fprintf('  Loaded cluster assignments\n');
else
    fprintf('  No cluster assignments found (optional)\n');
end

fprintf('\n');

%% ========================================================================
%  SECTION 2: PREPARE DATA MATRICES
%% ========================================================================

fprintf('Preparing data matrices...\n');

% Extract coefficient matrix
n_predictors = length(predictor_names);
coef_matrix = zeros(n_units, n_predictors);
deviance_explained = zeros(n_units, 1);
mean_fr = zeros(n_units, 1);

for i = 1:n_units
    coef_matrix(i, :) = glm_results(i).coefficients';
    deviance_explained(i) = glm_results(i).deviance_explained;
    mean_fr(i) = glm_results(i).mean_firing_rate;
end

% Extract feature importance
feature_names = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON', 'Aversive', 'Speed', 'Breathing'};
importance_matrix = zeros(n_units, length(feature_names));

for i = 1:n_units
    fi = glm_results(i).feature_importance;
    for f = 1:length(feature_names)
        fname = feature_names{f};
        if isfield(fi, fname)
            importance_matrix(i, f) = fi.(fname).percent_deviance;
        end
    end
end

fprintf('  Coefficient matrix: %d units × %d predictors\n', size(coef_matrix));
fprintf('  Mean deviance explained: %.2f%%\n', mean(deviance_explained) * 100);
fprintf('\n');

%% ========================================================================
%  SECTION 3: CREATE OUTPUT DIRECTORY
%% ========================================================================

output_dir = 'GLM_Figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% ========================================================================
%  SECTION 4: FIGURE 1 - COEFFICIENT HEATMAP
%% ========================================================================

fprintf('Creating Figure 1: Coefficient Heatmap...\n');

figure('Position', [100, 100, 1400, 800]);

% Sort units by deviance explained
[~, sort_idx] = sort(deviance_explained, 'descend');

% Plot heatmap (exclude bias term)
coef_to_plot = coef_matrix(sort_idx, 2:end);  % Exclude bias

% Z-score across units for visualization
coef_zscore = zscore(coef_to_plot, 0, 1);

imagesc(coef_zscore');
colormap(redblue);
colorbar;
caxis([-3, 3]);

xlabel('Units (sorted by deviance explained)', 'FontSize', 12);
ylabel('Predictors', 'FontSize', 12);
title('GLM Coefficients Across Units (z-scored)', 'FontSize', 14, 'FontWeight', 'bold');

% Set y-tick labels
yticks(1:length(predictor_names)-1);
yticklabels(predictor_names(2:end));

% Add grid
set(gca, 'GridLineStyle', '-', 'GridColor', 'k', 'GridAlpha', 0.1);
grid on;

saveas(gcf, fullfile(output_dir, 'Fig1_Coefficient_Heatmap.png'));
fprintf('  ✓ Saved Fig1_Coefficient_Heatmap.png\n');

%% ========================================================================
%  SECTION 5: FIGURE 2 - FEATURE IMPORTANCE
%% ========================================================================

fprintf('Creating Figure 2: Feature Importance...\n');

figure('Position', [100, 100, 1200, 600]);

% Sort units by deviance explained
importance_sorted = importance_matrix(sort_idx, :);

imagesc(importance_sorted');
colormap(hot);
colorbar;
ylabel('Feature', 'FontSize', 12);

xlabel('Units (sorted by deviance explained)', 'FontSize', 12);
ylabel('Predictor Group', 'FontSize', 12);
title('Feature Importance: % Deviance Contribution', 'FontSize', 14, 'FontWeight', 'bold');

yticks(1:length(feature_names));
yticklabels(feature_names);

saveas(gcf, fullfile(output_dir, 'Fig2_Feature_Importance.png'));
fprintf('  ✓ Saved Fig2_Feature_Importance.png\n');

%% ========================================================================
%  SECTION 6: FIGURE 3 - FEATURE IMPORTANCE SUMMARY
%% ========================================================================

fprintf('Creating Figure 3: Feature Importance Summary...\n');

figure('Position', [100, 100, 1000, 600]);

% Average importance across units
mean_importance = mean(importance_matrix, 1);
std_importance = std(importance_matrix, 0, 1);

bar(mean_importance);
hold on;
errorbar(1:length(feature_names), mean_importance, std_importance, 'k.', 'LineWidth', 1.5);

xticks(1:length(feature_names));
xticklabels(feature_names);
xtickangle(45);

ylabel('% Deviance Contribution', 'FontSize', 12);
title('Average Feature Importance Across All Units', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

saveas(gcf, fullfile(output_dir, 'Fig3_Feature_Importance_Summary.png'));
fprintf('  ✓ Saved Fig3_Feature_Importance_Summary.png\n');

%% ========================================================================
%  SECTION 7: FIGURE 4 - MODEL PERFORMANCE
%% ========================================================================

fprintf('Creating Figure 4: Model Performance...\n');

figure('Position', [100, 100, 1400, 500]);

% Panel 1: Deviance explained distribution
subplot(1, 3, 1);
histogram(deviance_explained * 100, 20, 'FaceColor', [0.3 0.6 0.9]);
xlabel('Deviance Explained (%)', 'FontSize', 11);
ylabel('Number of Units', 'FontSize', 11);
title(sprintf('Model Performance\nMean: %.1f%%', mean(deviance_explained) * 100), ...
    'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Panel 2: Deviance vs firing rate
subplot(1, 3, 2);
scatter(mean_fr, deviance_explained * 100, 30, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Mean Firing Rate (spikes/s)', 'FontSize', 11);
ylabel('Deviance Explained (%)', 'FontSize', 11);
title('Performance vs Firing Rate', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Panel 3: Units with positive vs negative deviance
subplot(1, 3, 3);
good_units = sum(deviance_explained > 0);
bad_units = sum(deviance_explained <= 0);
bar([good_units, bad_units]);
xticks([1, 2]);
xticklabels({'Positive', 'Negative/Zero'});
ylabel('Number of Units', 'FontSize', 11);
title('Model Fit Quality', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

saveas(gcf, fullfile(output_dir, 'Fig4_Model_Performance.png'));
fprintf('  ✓ Saved Fig4_Model_Performance.png\n');

%% ========================================================================
%  SECTION 8: FIGURE 5 - TEMPORAL FILTERS FOR EVENTS
%% ========================================================================

fprintf('Creating Figure 5: Temporal Filters...\n');

n_basis = config.n_basis_funcs;
event_names = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON', 'Aversive'};

% Time axis for basis functions (now spans [-1, +2] seconds)
n_bins_pre = round(config.event_window_pre / config.bin_size);
n_bins_post = round(config.event_window_post / config.bin_size);
event_duration_bins = n_bins_pre + n_bins_post;
time_axis = ((-n_bins_pre):(n_bins_post-1)) * config.bin_size * 1000;  % -1000ms to +2000ms

figure('Position', [100, 100, 1600, 900]);

for ev = 1:length(event_names)
    event_name = event_names{ev};

    % Extract coefficients for this event (average across units)
    start_idx = 2 + (ev - 1) * n_basis;  % +2 for bias
    end_idx = start_idx + n_basis - 1;

    event_coefs = coef_matrix(:, start_idx:end_idx);
    mean_coefs = mean(event_coefs, 1);
    std_coefs = std(event_coefs, 0, 1);

    % Reconstruct basis functions
    basis_funcs = createRaisedCosineBasis(n_basis, event_duration_bins);

    % Weighted sum to get temporal filter
    temporal_filter = basis_funcs * mean_coefs';
    temporal_filter_std = basis_funcs * std_coefs';

    % Plot
    subplot(2, 3, ev);
    plot(time_axis, temporal_filter, 'LineWidth', 2, 'Color', [0.2 0.4 0.8]);
    hold on;
    fill([time_axis, fliplr(time_axis)], ...
         [temporal_filter' + temporal_filter_std', fliplr(temporal_filter' - temporal_filter_std')], ...
         [0.2 0.4 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

    % Add vertical line at event onset (t=0)
    plot([0 0], ylim, 'k--', 'LineWidth', 1.5);

    xlabel('Time from event (ms)', 'FontSize', 10);
    ylabel('Weight', 'FontSize', 10);
    title(sprintf('%s Temporal Filter', event_name), 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    xlim([min(time_axis), max(time_axis)]);

    % Add shaded region for pre-event period
    yl = ylim;
    fill([min(time_axis) 0 0 min(time_axis)], [yl(1) yl(1) yl(2) yl(2)], ...
         [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    uistack(gca().Children(1), 'bottom');  % Send shading to back
end

saveas(gcf, fullfile(output_dir, 'Fig5_Temporal_Filters.png'));
fprintf('  ✓ Saved Fig5_Temporal_Filters.png\n');

%% ========================================================================
%  SECTION 9: FIGURE 6 - CLUSTER-SPECIFIC ANALYSIS (if available)
%% ========================================================================

if has_clusters
    fprintf('Creating Figure 6: Cluster-Specific Analysis...\n');

    % Match units to clusters
    cluster_assignments = matchUnitsToClusterswithGLM(glm_results, cluster_data);

    if ~isempty(cluster_assignments)
        unique_clusters = unique(cluster_assignments);
        n_clusters = length(unique_clusters);

        figure('Position', [100, 100, 1400, 900]);

        for c = 1:n_clusters
            cluster_id = unique_clusters(c);
            cluster_units = find(cluster_assignments == cluster_id);

            if isempty(cluster_units), continue; end

            % Average importance for this cluster
            cluster_importance = mean(importance_matrix(cluster_units, :), 1);

            subplot(ceil(n_clusters/2), 2, c);
            bar(cluster_importance);
            xticks(1:length(feature_names));
            xticklabels(feature_names);
            xtickangle(45);
            ylabel('% Deviance', 'FontSize', 10);
            title(sprintf('Cluster %d (n=%d)', cluster_id, length(cluster_units)), ...
                'FontSize', 11, 'FontWeight', 'bold');
            grid on;
            ylim([0, max(importance_matrix(:)) * 1.1]);
        end

        saveas(gcf, fullfile(output_dir, 'Fig6_Cluster_Specific_Importance.png'));
        fprintf('  ✓ Saved Fig6_Cluster_Specific_Importance.png\n');
    else
        fprintf('  WARNING: Could not match units to clusters\n');
    end
else
    fprintf('Skipping cluster analysis (no cluster data)\n');
end

%% ========================================================================
%  SECTION 10: SUMMARY STATISTICS
%% ========================================================================

fprintf('\n=== SUMMARY STATISTICS ===\n');
fprintf('Total units analyzed: %d\n', n_units);
fprintf('Mean deviance explained: %.2f%% (±%.2f%%)\n', ...
    mean(deviance_explained) * 100, std(deviance_explained) * 100);
fprintf('Units with positive deviance: %d (%.1f%%)\n', ...
    sum(deviance_explained > 0), sum(deviance_explained > 0) / n_units * 100);

fprintf('\nAverage feature importance:\n');
for f = 1:length(feature_names)
    fprintf('  %12s: %.2f%% (±%.2f%%)\n', ...
        feature_names{f}, mean_importance(f), std_importance(f));
end

% Save summary to text file
summary_file = fullfile(output_dir, 'GLM_Summary_Statistics.txt');
fid = fopen(summary_file, 'w');
fprintf(fid, '=== GLM ANALYSIS SUMMARY ===\n\n');
fprintf(fid, 'Total units analyzed: %d\n', n_units);
fprintf(fid, 'Mean deviance explained: %.2f%% (±%.2f%%)\n', ...
    mean(deviance_explained) * 100, std(deviance_explained) * 100);
fprintf(fid, 'Units with positive deviance: %d (%.1f%%)\n\n', ...
    sum(deviance_explained > 0), sum(deviance_explained > 0) / n_units * 100);

fprintf(fid, 'Average Feature Importance (percent deviance):\n');
for f = 1:length(feature_names)
    fprintf(fid, '  %12s: %.2f%% (±%.2f%%)\n', ...
        feature_names{f}, mean_importance(f), std_importance(f));
end
fclose(fid);

fprintf('\n✓ Summary saved to: %s\n', summary_file);

fprintf('\n=== VISUALIZATION COMPLETE ===\n');
fprintf('All figures saved to: %s/\n', output_dir);


%% ========================================================================
%  HELPER FUNCTIONS
%% ========================================================================

function cmap = redblue(m)
% Red-Blue colormap
    if nargin < 1
        m = 256;
    end

    r = [(0:m/2-1)'/max(m/2-1,1); ones(m/2,1)];
    g = [(0:m/2-1)'/max(m/2-1,1); (m/2-1:-1:0)'/max(m/2-1,1)];
    b = [ones(m/2,1); (m/2-1:-1:0)'/max(m/2-1,1)];

    cmap = [r g b];
end


function basis = createRaisedCosineBasis(n_basis, n_bins)
% Create raised cosine basis functions (same as in main script)
% Fix: Ensure first peak is at or near t=1 (event onset) to capture immediate response

    peaks = logspace(log10(0.5), log10(n_bins), n_basis);  % Start from 0.5 for better coverage at onset
    width = (log10(n_bins) - log10(0.5)) / (n_basis - 1) * 2;

    basis = zeros(n_bins, n_basis);

    for i = 1:n_basis
        for t = 1:n_bins
            arg = (log10(max(t, 1)) - log10(peaks(i))) / width;
            if abs(arg) < 0.5
                basis(t, i) = (cos(arg * pi) + 1) / 2;
            end
        end
        if sum(basis(:, i)) > 0
            basis(:, i) = basis(:, i) / sum(basis(:, i));
        end
    end
end


function cluster_assignments = matchUnitsToClusterswithGLM(glm_results, cluster_data)
% Match GLM results to cluster assignments
%
% This function tries to match units in glm_results to their cluster assignments
% by matching session names and unit IDs

    n_units = length(glm_results);
    cluster_assignments = nan(n_units, 1);

    % Check if cluster_data has the right fields
    if ~isfield(cluster_data, 'cluster_assignments')
        warning('cluster_data does not have cluster_assignments field');
        return;
    end

    cluster_info = cluster_data.cluster_assignments;

    % Try to match by session name and unit ID
    for i = 1:n_units
        glm_session = glm_results(i).session_name;
        glm_unit_id = glm_results(i).unit_id;

        % Search in cluster data
        for j = 1:length(cluster_info)
            if isfield(cluster_info(j), 'session_name') && isfield(cluster_info(j), 'unit_id')
                if strcmp(cluster_info(j).session_name, glm_session) && ...
                   cluster_info(j).unit_id == glm_unit_id
                    cluster_assignments(i) = cluster_info(j).cluster;
                    break;
                end
            end
        end
    end

    % Report matching success
    matched = sum(~isnan(cluster_assignments));
    fprintf('  Matched %d / %d units to clusters\n', matched, n_units);
end
