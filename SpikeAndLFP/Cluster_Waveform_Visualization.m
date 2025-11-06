%% ========================================================================
%  CLUSTER-WAVEFORM-PROPERTY VISUALIZATION
%  Comprehensive visualization of relationships between:
%  - Waveform properties (cell type)
%  - Firing rate & CV (temporal dynamics)
%  - Functional cluster assignments
%  ========================================================================

clear all;
close all;

fprintf('========================================\n');
fprintf('CLUSTER-WAVEFORM VISUALIZATION\n');
fprintf('========================================\n\n');

%% ========================================================================
%  SECTION 1: LOAD ANALYSIS RESULTS
%  ========================================================================

fprintf('Loading analysis results...\n');

% Prompt user to select analysis results
[filename, pathname] = uigetfile('cluster_waveform_analysis_*.mat', ...
                                  'Select Cluster-Waveform Analysis Results');
if isequal(filename, 0)
    error('No file selected. Exiting.');
end

data = load(fullfile(pathname, filename));
results = data.analysis_results;
integrated = results.integrated_data;
clustering = results.clustering_results;

fprintf('✓ Loaded results: %s\n', filename);
fprintf('  Units: %d\n', length(integrated));
fprintf('  Clusters: %d\n', clustering.clustering.n_clusters);
fprintf('  Session type: %s\n\n', clustering.metadata.session_type);

%% ========================================================================
%  FIGURE 1: WAVEFORM PROPERTIES BY CLUSTER
%  ========================================================================

fprintf('Creating Figure 1: Waveform Properties by Cluster\n');

fig1 = figure('Position', [100, 100, 1600, 900], ...
              'Name', 'Waveform Properties by Cluster');

n_clusters = clustering.clustering.n_clusters;

% Extract waveform features
trough_to_peak = [integrated.wf_trough_to_peak]';
fwhm = [integrated.wf_fwhm]';
asymmetry = [integrated.wf_asymmetry]';
cluster_ids = [integrated.cluster_id]';
cell_types = [integrated.cell_type_code]';

valid_wf = ~isnan(trough_to_peak);

% Panel A: Trough-to-peak by cluster
subplot(2, 3, 1);
hold on;
for c = 1:n_clusters
    cluster_mask = (cluster_ids == c) & valid_wf;
    vals = trough_to_peak(cluster_mask);

    if ~isempty(vals)
        x_pos = c + randn(size(vals)) * 0.1;
        scatter(x_pos, vals, 30, 'filled', 'MarkerFaceAlpha', 0.5);

        % Plot mean
        plot(c, mean(vals), 'r_', 'LineWidth', 3, 'MarkerSize', 20);
    end
end
plot([0, n_clusters+1], [0.4, 0.4], 'k--', 'LineWidth', 2);  % Cell type threshold
xlabel('Cluster ID', 'FontWeight', 'bold');
ylabel('Trough-to-Peak (ms)', 'FontWeight', 'bold');
title('Waveform Trough-to-Peak Time', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0.5, n_clusters+0.5]);
set(gca, 'XTick', 1:n_clusters);
grid on;
hold off;

% Panel B: FWHM by cluster
subplot(2, 3, 2);
hold on;
for c = 1:n_clusters
    cluster_mask = (cluster_ids == c) & valid_wf;
    vals = fwhm(cluster_mask);

    if ~isempty(vals)
        x_pos = c + randn(size(vals)) * 0.1;
        scatter(x_pos, vals, 30, 'filled', 'MarkerFaceAlpha', 0.5);
        plot(c, mean(vals), 'r_', 'LineWidth', 3, 'MarkerSize', 20);
    end
end
xlabel('Cluster ID', 'FontWeight', 'bold');
ylabel('FWHM (ms)', 'FontWeight', 'bold');
title('Waveform Width (FWHM)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0.5, n_clusters+0.5]);
set(gca, 'XTick', 1:n_clusters);
grid on;
hold off;

% Panel C: Cell type composition
subplot(2, 3, 3);
cell_type_matrix = zeros(n_clusters, 2);  % [Interneuron, Pyramidal]
for c = 1:n_clusters
    cluster_mask = (cluster_ids == c) & valid_wf;
    cell_type_matrix(c, 1) = sum(cell_types(cluster_mask) == 1);  % Interneuron
    cell_type_matrix(c, 2) = sum(cell_types(cluster_mask) == 2);  % Pyramidal
end

bar(1:n_clusters, cell_type_matrix, 'stacked');
xlabel('Cluster ID', 'FontWeight', 'bold');
ylabel('Number of Units', 'FontWeight', 'bold');
title('Cell Type Composition', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Interneuron', 'Pyramidal'}, 'Location', 'best');
grid on;

% Panel D: Trough-to-peak vs FWHM scatter
subplot(2, 3, 4);
hold on;
colors = lines(n_clusters);
for c = 1:n_clusters
    cluster_mask = (cluster_ids == c) & valid_wf;
    scatter(trough_to_peak(cluster_mask), fwhm(cluster_mask), 50, colors(c, :), ...
            'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName', sprintf('Cluster %d', c));
end
plot([0.4, 0.4], ylim, 'k--', 'LineWidth', 2, 'HandleVisibility', 'off');
xlabel('Trough-to-Peak (ms)', 'FontWeight', 'bold');
ylabel('FWHM (ms)', 'FontWeight', 'bold');
title('Waveform Feature Space', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
hold off;

% Panel E: Example waveforms by cluster (if available)
subplot(2, 3, 5);
if ~isempty(results.waveform_features)
    % Plot mean waveform for each cluster
    hold on;
    for c = 1:min(n_clusters, 5)  % Limit to 5 clusters for clarity
        % Get first unit from this cluster with waveform data
        cluster_unit_ids = [integrated(cluster_ids == c).global_unit_id];
        if ~isempty(cluster_unit_ids)
            wf_idx = find([results.waveform_features.global_unit_id] == cluster_unit_ids(1), 1);
            if ~isempty(wf_idx)
                wf = results.waveform_features(wf_idx);
                time_ms = (0:length(wf.waveform)-1) / wf.sampling_rate * 1000;
                plot(time_ms, wf.waveform, 'LineWidth', 2, 'Color', colors(c, :), ...
                     'DisplayName', sprintf('Cluster %d', c));
            end
        end
    end
    xlabel('Time (ms)', 'FontWeight', 'bold');
    ylabel('Amplitude (normalized)', 'FontWeight', 'bold');
    title('Example Waveforms', 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    hold off;
else
    text(0.5, 0.5, 'No waveform data available', ...
         'HorizontalAlignment', 'center', 'FontSize', 12);
    axis off;
end

% Panel F: Asymmetry by cluster
subplot(2, 3, 6);
hold on;
for c = 1:n_clusters
    cluster_mask = (cluster_ids == c) & valid_wf;
    vals = asymmetry(cluster_mask);

    if ~isempty(vals)
        x_pos = c + randn(size(vals)) * 0.1;
        scatter(x_pos, vals, 30, 'filled', 'MarkerFaceAlpha', 0.5);
        plot(c, mean(vals), 'r_', 'LineWidth', 3, 'MarkerSize', 20);
    end
end
xlabel('Cluster ID', 'FontWeight', 'bold');
ylabel('Asymmetry Index', 'FontWeight', 'bold');
title('Waveform Asymmetry', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0.5, n_clusters+0.5]);
set(gca, 'XTick', 1:n_clusters);
grid on;
hold off;

sgtitle(sprintf('Waveform Properties Across %d Clusters', n_clusters), ...
        'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  FIGURE 2: FIRING RATE & CV BY CLUSTER
%  ========================================================================

fprintf('Creating Figure 2: Firing Rate & CV by Cluster\n');

fig2 = figure('Position', [150, 150, 1600, 900], ...
              'Name', 'Firing Properties by Cluster');

% Extract FR/CV features
mean_fr = [integrated.mean_fr]';
mean_cv = [integrated.mean_cv]';
fr_std = [integrated.fr_std]';

valid_fr = ~isnan(mean_fr);
valid_cv = ~isnan(mean_cv);

% Panel A: Mean firing rate by cluster
subplot(2, 3, 1);
hold on;
for c = 1:n_clusters
    cluster_mask = (cluster_ids == c) & valid_fr;
    vals = mean_fr(cluster_mask);

    if ~isempty(vals)
        x_pos = c + randn(size(vals)) * 0.1;
        scatter(x_pos, vals, 30, 'filled', 'MarkerFaceAlpha', 0.5);
        plot(c, mean(vals), 'r_', 'LineWidth', 3, 'MarkerSize', 20);
    end
end
xlabel('Cluster ID', 'FontWeight', 'bold');
ylabel('Mean Firing Rate (Hz)', 'FontWeight', 'bold');
title('Firing Rate by Cluster', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0.5, n_clusters+0.5]);
set(gca, 'XTick', 1:n_clusters);
grid on;
hold off;

% Panel B: CV by cluster
subplot(2, 3, 2);
hold on;
for c = 1:n_clusters
    cluster_mask = (cluster_ids == c) & valid_cv;
    vals = mean_cv(cluster_mask);

    if ~isempty(vals)
        x_pos = c + randn(size(vals)) * 0.1;
        scatter(x_pos, vals, 30, 'filled', 'MarkerFaceAlpha', 0.5);
        plot(c, mean(vals), 'r_', 'LineWidth', 3, 'MarkerSize', 20);
    end
end
xlabel('Cluster ID', 'FontWeight', 'bold');
ylabel('Coefficient of Variation', 'FontWeight', 'bold');
title('CV by Cluster', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0.5, n_clusters+0.5]);
set(gca, 'XTick', 1:n_clusters);
grid on;
hold off;

% Panel C: FR vs CV scatter
subplot(2, 3, 3);
hold on;
for c = 1:n_clusters
    cluster_mask = (cluster_ids == c) & valid_fr & valid_cv;
    scatter(mean_fr(cluster_mask), mean_cv(cluster_mask), 50, colors(c, :), ...
            'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName', sprintf('Cluster %d', c));
end
xlabel('Mean Firing Rate (Hz)', 'FontWeight', 'bold');
ylabel('Coefficient of Variation', 'FontWeight', 'bold');
title('FR vs CV', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
hold off;

% Panel D: FR variability by cluster
subplot(2, 3, 4);
hold on;
for c = 1:n_clusters
    cluster_mask = (cluster_ids == c) & ~isnan(fr_std);
    vals = fr_std(cluster_mask);

    if ~isempty(vals)
        x_pos = c + randn(size(vals)) * 0.1;
        scatter(x_pos, vals, 30, 'filled', 'MarkerFaceAlpha', 0.5);
        plot(c, mean(vals), 'r_', 'LineWidth', 3, 'MarkerSize', 20);
    end
end
xlabel('Cluster ID', 'FontWeight', 'bold');
ylabel('FR Std Dev (Hz)', 'FontWeight', 'bold');
title('Firing Rate Variability', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0.5, n_clusters+0.5]);
set(gca, 'XTick', 1:n_clusters);
grid on;
hold off;

% Panel E: FR by cell type
subplot(2, 3, 5);
cell_type_fr = [];
cell_type_labels = {};
for ct = 1:2
    ct_mask = (cell_types == ct) & valid_fr;
    if sum(ct_mask) > 0
        cell_type_fr = [cell_type_fr; mean_fr(ct_mask)];
        if ct == 1
            cell_type_labels = [cell_type_labels; repmat({'Interneuron'}, sum(ct_mask), 1)];
        else
            cell_type_labels = [cell_type_labels; repmat({'Pyramidal'}, sum(ct_mask), 1)];
        end
    end
end

if ~isempty(cell_type_fr)
    boxplot(cell_type_fr, cell_type_labels);
    ylabel('Mean Firing Rate (Hz)', 'FontWeight', 'bold');
    title('FR by Putative Cell Type', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
end

% Panel F: CV by cell type
subplot(2, 3, 6);
cell_type_cv = [];
cell_type_labels = {};
for ct = 1:2
    ct_mask = (cell_types == ct) & valid_cv;
    if sum(ct_mask) > 0
        cell_type_cv = [cell_type_cv; mean_cv(ct_mask)];
        if ct == 1
            cell_type_labels = [cell_type_labels; repmat({'Interneuron'}, sum(ct_mask), 1)];
        else
            cell_type_labels = [cell_type_labels; repmat({'Pyramidal'}, sum(ct_mask), 1)];
        end
    end
end

if ~isempty(cell_type_cv)
    boxplot(cell_type_cv, cell_type_labels);
    ylabel('Coefficient of Variation', 'FontWeight', 'bold');
    title('CV by Putative Cell Type', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
end

sgtitle('Firing Properties Across Clusters and Cell Types', ...
        'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  FIGURE 3: WAVEFORM-FUNCTIONAL FEATURE RELATIONSHIPS
%  ========================================================================

fprintf('Creating Figure 3: Waveform-Functional Feature Relationships\n');

fig3 = figure('Position', [200, 200, 1600, 900], ...
              'Name', 'Waveform-Functional Feature Relationships');

% Get functional features
feature_names = clustering.features.names;
n_features = length(feature_names);

% Compute correlations
correlations = nan(n_features, 4);  % [trough-to-peak, FWHM, FR, CV]
p_values = nan(n_features, 4);

for f = 1:n_features
    feat_vals = arrayfun(@(x) x.functional_features(f), integrated);

    % Trough-to-peak
    valid = valid_wf & ~isnan(feat_vals');
    if sum(valid) > 10
        [correlations(f, 1), p_values(f, 1)] = corr(trough_to_peak(valid), ...
                                                     feat_vals(valid)', ...
                                                     'Type', 'Spearman', 'Rows', 'complete');
    end

    % FWHM
    valid = ~isnan(fwhm) & ~isnan(feat_vals');
    if sum(valid) > 10
        [correlations(f, 2), p_values(f, 2)] = corr(fwhm(valid), ...
                                                     feat_vals(valid)', ...
                                                     'Type', 'Spearman', 'Rows', 'complete');
    end

    % FR
    valid = valid_fr & ~isnan(feat_vals');
    if sum(valid) > 10
        [correlations(f, 3), p_values(f, 3)] = corr(mean_fr(valid), ...
                                                     feat_vals(valid)', ...
                                                     'Type', 'Spearman', 'Rows', 'complete');
    end

    % CV
    valid = valid_cv & ~isnan(feat_vals');
    if sum(valid) > 10
        [correlations(f, 4), p_values(f, 4)] = corr(mean_cv(valid), ...
                                                     feat_vals(valid)', ...
                                                     'Type', 'Spearman', 'Rows', 'complete');
    end
end

% Panel A: Correlation heatmap
subplot(2, 2, [1, 2]);
imagesc(correlations');
colormap(bluewhitered(256));
caxis([-1, 1]);
cb = colorbar;
ylabel(cb, 'Spearman r', 'FontWeight', 'bold');

set(gca, 'XTick', 1:n_features);
set(gca, 'XTickLabel', feature_names);
set(gca, 'XTickLabelRotation', 45);
set(gca, 'YTick', 1:4);
set(gca, 'YTickLabel', {'Trough-Peak', 'FWHM', 'Mean FR', 'Mean CV'});
title('Correlations: Waveform/FR/CV vs Functional Features', ...
      'FontSize', 12, 'FontWeight', 'bold');

% Add significance stars
for f = 1:n_features
    for p = 1:4
        if p_values(f, p) < 0.001
            text(f, p, '***', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        elseif p_values(f, p) < 0.01
            text(f, p, '**', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        elseif p_values(f, p) < 0.05
            text(f, p, '*', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        end
    end
end

% Panel B: Top correlations scatter plots
subplot(2, 2, 3);
% Find strongest correlation
[max_r, max_idx] = max(abs(correlations(:)));
[feat_idx, prop_idx] = ind2sub(size(correlations), max_idx);

feat_vals = arrayfun(@(x) x.functional_features(feat_idx), integrated);

switch prop_idx
    case 1
        x_vals = trough_to_peak;
        x_label = 'Trough-to-Peak (ms)';
        valid = valid_wf;
    case 2
        x_vals = fwhm;
        x_label = 'FWHM (ms)';
        valid = ~isnan(fwhm);
    case 3
        x_vals = mean_fr;
        x_label = 'Mean FR (Hz)';
        valid = valid_fr;
    case 4
        x_vals = mean_cv;
        x_label = 'Mean CV';
        valid = valid_cv;
end

valid = valid & ~isnan(feat_vals');

hold on;
for c = 1:n_clusters
    cluster_mask = (cluster_ids == c) & valid;
    scatter(x_vals(cluster_mask), feat_vals(cluster_mask), 50, colors(c, :), ...
            'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName', sprintf('Cluster %d', c));
end
xlabel(x_label, 'FontWeight', 'bold');
ylabel(feature_names{feat_idx}, 'FontWeight', 'bold');
title(sprintf('Strongest Correlation (r=%.3f, p=%.4f)', ...
              correlations(max_idx), p_values(max_idx)), ...
      'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
hold off;

% Panel C: Significant correlations summary
subplot(2, 2, 4);
sig_correlations = abs(correlations);
sig_correlations(p_values >= 0.05) = 0;

bar(max(sig_correlations, [], 2));
set(gca, 'XTick', 1:n_features);
set(gca, 'XTickLabel', feature_names);
set(gca, 'XTickLabelRotation', 45);
ylabel('Max |r| (p < 0.05)', 'FontWeight', 'bold');
title('Significant Correlations per Feature', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

%% ========================================================================
%  FIGURE 4: TEMPORAL FR/CV DYNAMICS BY CLUSTER
%  ========================================================================

fprintf('Creating Figure 4: Temporal FR/CV Dynamics\n');

if ~isempty(results.temporal_fr_cv) && isfield(integrated(1), 'fr_timecourse')

    fig4 = figure('Position', [250, 250, 1600, 1000], ...
                  'Name', 'Temporal Dynamics by Cluster');

    % Panel A-C: Mean FR timecourse per cluster
    for c = 1:min(n_clusters, 6)  % Limit to 6 clusters
        subplot(3, 3, c);

        cluster_units = integrated(cluster_ids == c);

        if isempty(cluster_units)
            continue;
        end

        % Get FR timecourses
        n_units_with_tc = sum(arrayfun(@(x) isfield(x, 'fr_timecourse') && ...
                                        ~isempty(x.fr_timecourse), cluster_units));

        if n_units_with_tc > 0
            % Get max length
            max_len = max(arrayfun(@(x) length(x.fr_timecourse), cluster_units));

            % Align all timecourses
            fr_matrix = nan(length(cluster_units), max_len);
            for u = 1:length(cluster_units)
                if isfield(cluster_units(u), 'fr_timecourse')
                    tc = cluster_units(u).fr_timecourse;
                    fr_matrix(u, 1:length(tc)) = tc;
                end
            end

            % Plot mean ± SEM
            mean_fr_tc = nanmean(fr_matrix, 1);
            sem_fr_tc = nanstd(fr_matrix, 0, 1) / sqrt(size(fr_matrix, 1));

            time_vec = (1:max_len) * results.config.fr_bin_size / 60;  % Convert to minutes

            hold on;
            plot(time_vec, mean_fr_tc, 'LineWidth', 2, 'Color', colors(c, :));
            fill([time_vec, fliplr(time_vec)], ...
                 [mean_fr_tc + sem_fr_tc, fliplr(mean_fr_tc - sem_fr_tc)], ...
                 colors(c, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            xlabel('Time (min)', 'FontWeight', 'bold');
            ylabel('Firing Rate (Hz)', 'FontWeight', 'bold');
            title(sprintf('Cluster %d (n=%d)', c, length(cluster_units)), ...
                  'FontWeight', 'bold');
            grid on;
            hold off;
        end
    end

    % Panel: CV timecourse comparison
    subplot(3, 3, [7, 8, 9]);
    hold on;
    for c = 1:min(n_clusters, 6)
        cluster_units = integrated(cluster_ids == c);

        if ~isempty(cluster_units)
            max_len = max(arrayfun(@(x) length(x.cv_timecourse), cluster_units));
            cv_matrix = nan(length(cluster_units), max_len);

            for u = 1:length(cluster_units)
                if isfield(cluster_units(u), 'cv_timecourse')
                    tc = cluster_units(u).cv_timecourse;
                    cv_matrix(u, 1:length(tc)) = tc;
                end
            end

            mean_cv_tc = nanmean(cv_matrix, 1);
            time_vec = (1:max_len) * results.config.fr_bin_size / 60;

            plot(time_vec, mean_cv_tc, 'LineWidth', 2, 'Color', colors(c, :), ...
                 'DisplayName', sprintf('Cluster %d', c));
        end
    end
    xlabel('Time (min)', 'FontWeight', 'bold');
    ylabel('Coefficient of Variation', 'FontWeight', 'bold');
    title('CV Timecourse Across Clusters', 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    hold off;

    sgtitle('Temporal Dynamics of Firing Properties', ...
            'FontSize', 14, 'FontWeight', 'bold');
end

%% ========================================================================
%  SAVE FIGURES
%  ========================================================================

fprintf('\n=== SAVING FIGURES ===\n');

timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
fig_folder = 'Cluster_Waveform_Figures';
if ~exist(fig_folder, 'dir')
    mkdir(fig_folder);
end

saveas(fig1, fullfile(fig_folder, sprintf('Fig1_Waveform_Properties_%s.png', timestamp)));
saveas(fig1, fullfile(fig_folder, sprintf('Fig1_Waveform_Properties_%s.fig', timestamp)));

saveas(fig2, fullfile(fig_folder, sprintf('Fig2_Firing_Properties_%s.png', timestamp)));
saveas(fig2, fullfile(fig_folder, sprintf('Fig2_Firing_Properties_%s.fig', timestamp)));

saveas(fig3, fullfile(fig_folder, sprintf('Fig3_Feature_Correlations_%s.png', timestamp)));
saveas(fig3, fullfile(fig_folder, sprintf('Fig3_Feature_Correlations_%s.fig', timestamp)));

if exist('fig4', 'var')
    saveas(fig4, fullfile(fig_folder, sprintf('Fig4_Temporal_Dynamics_%s.png', timestamp)));
    saveas(fig4, fullfile(fig_folder, sprintf('Fig4_Temporal_Dynamics_%s.fig', timestamp)));
end

fprintf('✓ Figures saved to: %s\n', fig_folder);

fprintf('\n========================================\n');
fprintf('VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
