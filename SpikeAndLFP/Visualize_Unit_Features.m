%% ========================================================================
%  VISUALIZE UNIT FEATURES: Period × SessionType Analysis
%  Comprehensive visualization of 22 spike train metrics
%  ========================================================================
%
%  Visualizes results from Unit_Features_Analysis.m
%
%  Plots organized by metric categories:
%  1. Basic Firing Metrics (FR, CV)
%  2. ISI Variability (ISI_FanoFactor, LV, CV2, LVR)
%  3. ISI Auto-Correlation (peak, lag, decay)
%  4. Burst Metrics (index, rate, mean length)
%  5. ISI Distribution Shape (skewness, kurtosis, mode)
%  6. Spike Count Metrics (Fano Factor & ACF for 3 bin sizes)
%  7. Quality Metrics (refractory violations)
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: LOAD DATA
%  ========================================================================

fprintf('=== VISUALIZING UNIT FEATURES ===\n\n');

% Find the most recent results file
files = dir('Unit_features_analysis_*.mat');
if isempty(files)
    error('No Unit_features_analysis_*.mat file found in current directory');
end

% Get most recent file
[~, idx] = max([files.datenum]);
data_file = files(idx).name;

fprintf('Loading data from: %s\n', data_file);
load(data_file);

tbl = results.tbl_data;
config = results.config;

fprintf('✓ Data loaded\n');
fprintf('  Total data points: %d\n', height(tbl));
fprintf('  Aversive sessions: %d (7 periods each)\n', results.n_aversive_sessions);
fprintf('  Reward sessions: %d (4 periods each)\n', results.n_reward_sessions);
fprintf('  Total units: %d\n\n', length(unique(tbl.Unit)));

%% ========================================================================
%  SECTION 2: SETUP VISUALIZATION PARAMETERS
%  ========================================================================

% Colors
color_aversive = [0.8 0.2 0.2];  % Red
color_reward = [0.2 0.4 0.8];    % Blue

% Create output directory for figures
output_dir = 'Unit_Features_Figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('Figures will be saved to: %s/\n\n', output_dir);

%% ========================================================================
%  SECTION 3: BASIC FIRING METRICS
%  ========================================================================

fprintf('Plotting basic firing metrics...\n');

figure('Position', [100, 100, 1400, 600]);
sgtitle('Basic Firing Metrics: Period × SessionType', 'FontSize', 14, 'FontWeight', 'bold');

% FR - Firing Rate
subplot(1, 2, 1);
plotMetricByPeriod(tbl, 'FR', 'Firing Rate (Hz)', color_aversive, color_reward);

% CV - Coefficient of Variation
subplot(1, 2, 2);
plotMetricByPeriod(tbl, 'CV', 'Coefficient of Variation', color_aversive, color_reward);

saveas(gcf, fullfile(output_dir, 'Fig1_Basic_Firing_Metrics.png'));
fprintf('  ✓ Saved: Fig1_Basic_Firing_Metrics.png\n');

%% ========================================================================
%  SECTION 4: ISI VARIABILITY METRICS
%  ========================================================================

fprintf('Plotting ISI variability metrics...\n');

figure('Position', [100, 100, 1400, 900]);
sgtitle('ISI Variability Metrics: Period × SessionType', 'FontSize', 14, 'FontWeight', 'bold');

% ISI Fano Factor
subplot(2, 2, 1);
plotMetricByPeriod(tbl, 'ISI_FanoFactor', 'ISI Fano Factor', color_aversive, color_reward);

% LV - Local Variation
subplot(2, 2, 2);
plotMetricByPeriod(tbl, 'LV', 'Local Variation (LV)', color_aversive, color_reward);

% CV2 - Local CV
subplot(2, 2, 3);
plotMetricByPeriod(tbl, 'CV2', 'Local CV (CV2)', color_aversive, color_reward);

% LVR - Revised Local Variation
subplot(2, 2, 4);
plotMetricByPeriod(tbl, 'LVR', 'Revised Local Variation (LVR)', color_aversive, color_reward);

saveas(gcf, fullfile(output_dir, 'Fig2_ISI_Variability_Metrics.png'));
fprintf('  ✓ Saved: Fig2_ISI_Variability_Metrics.png\n');

%% ========================================================================
%  SECTION 5: ISI AUTO-CORRELATION METRICS
%  ========================================================================

fprintf('Plotting ISI auto-correlation metrics...\n');

figure('Position', [100, 100, 1400, 500]);
sgtitle('ISI Auto-Correlation Metrics: Period × SessionType', 'FontSize', 14, 'FontWeight', 'bold');

% ACF Peak
subplot(1, 3, 1);
plotMetricByPeriod(tbl, 'ISI_ACF_peak', 'ISI ACF Peak (exclude lag 0)', color_aversive, color_reward);

% ACF Lag
subplot(1, 3, 2);
plotMetricByPeriod(tbl, 'ISI_ACF_lag', 'ISI ACF Lag at Peak (s)', color_aversive, color_reward);

% ACF Decay
subplot(1, 3, 3);
plotMetricByPeriod(tbl, 'ISI_ACF_decay', 'ISI ACF Decay Time (s)', color_aversive, color_reward);

saveas(gcf, fullfile(output_dir, 'Fig3_ISI_ACF_Metrics.png'));
fprintf('  ✓ Saved: Fig3_ISI_ACF_Metrics.png\n');

%% ========================================================================
%  SECTION 6: BURST METRICS
%  ========================================================================

fprintf('Plotting burst metrics...\n');

figure('Position', [100, 100, 1400, 500]);
sgtitle('Burst Metrics: Period × SessionType', 'FontSize', 14, 'FontWeight', 'bold');

% Burst Index
subplot(1, 3, 1);
plotMetricByPeriod(tbl, 'BurstIndex', 'Burst Index (fraction < 10ms)', color_aversive, color_reward);

% Burst Rate
subplot(1, 3, 2);
plotMetricByPeriod(tbl, 'BurstRate', 'Burst Rate (bursts/s)', color_aversive, color_reward);

% Mean Burst Length
subplot(1, 3, 3);
plotMetricByPeriod(tbl, 'MeanBurstLength', 'Mean Burst Length (spikes)', color_aversive, color_reward);

saveas(gcf, fullfile(output_dir, 'Fig4_Burst_Metrics.png'));
fprintf('  ✓ Saved: Fig4_Burst_Metrics.png\n');

%% ========================================================================
%  SECTION 7: ISI DISTRIBUTION SHAPE
%  ========================================================================

fprintf('Plotting ISI distribution shape metrics...\n');

figure('Position', [100, 100, 1400, 500]);
sgtitle('ISI Distribution Shape: Period × SessionType', 'FontSize', 14, 'FontWeight', 'bold');

% Skewness
subplot(1, 3, 1);
plotMetricByPeriod(tbl, 'ISI_Skewness', 'ISI Skewness', color_aversive, color_reward);

% Kurtosis
subplot(1, 3, 2);
plotMetricByPeriod(tbl, 'ISI_Kurtosis', 'ISI Kurtosis', color_aversive, color_reward);

% Mode
subplot(1, 3, 3);
plotMetricByPeriod(tbl, 'ISI_Mode', 'ISI Mode (s)', color_aversive, color_reward);

saveas(gcf, fullfile(output_dir, 'Fig5_ISI_Distribution_Shape.png'));
fprintf('  ✓ Saved: Fig5_ISI_Distribution_Shape.png\n');

%% ========================================================================
%  SECTION 8: SPIKE COUNT FANO FACTOR
%  ========================================================================

fprintf('Plotting spike count Fano Factor metrics...\n');

figure('Position', [100, 100, 1400, 500]);
sgtitle('Spike Count Fano Factor (3 bin sizes): Period × SessionType', 'FontSize', 14, 'FontWeight', 'bold');

% 1ms bins
subplot(1, 3, 1);
plotMetricByPeriod(tbl, 'CountFanoFactor_1ms', 'Count Fano Factor (1ms bins)', color_aversive, color_reward);

% 25ms bins
subplot(1, 3, 2);
plotMetricByPeriod(tbl, 'CountFanoFactor_25ms', 'Count Fano Factor (25ms bins)', color_aversive, color_reward);

% 50ms bins
subplot(1, 3, 3);
plotMetricByPeriod(tbl, 'CountFanoFactor_50ms', 'Count Fano Factor (50ms bins)', color_aversive, color_reward);

saveas(gcf, fullfile(output_dir, 'Fig6_Count_FanoFactor.png'));
fprintf('  ✓ Saved: Fig6_Count_FanoFactor.png\n');

%% ========================================================================
%  SECTION 9: SPIKE COUNT AUTO-CORRELATION
%  ========================================================================

fprintf('Plotting spike count ACF metrics...\n');

figure('Position', [100, 100, 1400, 500]);
sgtitle('Spike Count ACF Peak (3 bin sizes, exclude lag 0): Period × SessionType', 'FontSize', 14, 'FontWeight', 'bold');

% 1ms bins
subplot(1, 3, 1);
plotMetricByPeriod(tbl, 'Count_ACF_1ms_peak', 'Count ACF Peak (1ms bins)', color_aversive, color_reward);

% 25ms bins
subplot(1, 3, 2);
plotMetricByPeriod(tbl, 'Count_ACF_25ms_peak', 'Count ACF Peak (25ms bins)', color_aversive, color_reward);

% 50ms bins
subplot(1, 3, 3);
plotMetricByPeriod(tbl, 'Count_ACF_50ms_peak', 'Count ACF Peak (50ms bins)', color_aversive, color_reward);

saveas(gcf, fullfile(output_dir, 'Fig7_Count_ACF.png'));
fprintf('  ✓ Saved: Fig7_Count_ACF.png\n');

%% ========================================================================
%  SECTION 10: QUALITY METRICS
%  ========================================================================

fprintf('Plotting quality metrics...\n');

figure('Position', [100, 100, 700, 500]);
sgtitle('Quality Metrics: Period × SessionType', 'FontSize', 14, 'FontWeight', 'bold');

plotMetricByPeriod(tbl, 'RefracViolations', 'Refractory Violations (%)', color_aversive, color_reward);

saveas(gcf, fullfile(output_dir, 'Fig8_Quality_Metrics.png'));
fprintf('  ✓ Saved: Fig8_Quality_Metrics.png\n');

%% ========================================================================
%  SECTION 11: COMPREHENSIVE HEATMAP
%  ========================================================================

fprintf('Creating comprehensive heatmap...\n');

% List all 22 metrics
all_metrics = {'FR', 'CV', 'ISI_FanoFactor', 'ISI_ACF_peak', 'ISI_ACF_lag', 'ISI_ACF_decay', ...
               'Count_ACF_1ms_peak', 'Count_ACF_25ms_peak', 'Count_ACF_50ms_peak', ...
               'LV', 'CV2', 'LVR', 'BurstIndex', 'BurstRate', 'MeanBurstLength', ...
               'ISI_Skewness', 'ISI_Kurtosis', 'ISI_Mode', ...
               'CountFanoFactor_1ms', 'CountFanoFactor_25ms', 'CountFanoFactor_50ms', ...
               'RefracViolations'};

metric_labels = {'FR', 'CV', 'ISI Fano', 'ISI ACF peak', 'ISI ACF lag', 'ISI ACF decay', ...
                 'Count ACF 1ms', 'Count ACF 25ms', 'Count ACF 50ms', ...
                 'LV', 'CV2', 'LVR', 'Burst Index', 'Burst Rate', 'Burst Length', ...
                 'ISI Skew', 'ISI Kurt', 'ISI Mode', ...
                 'Count FF 1ms', 'Count FF 25ms', 'Count FF 50ms', ...
                 'Refrac Viol'};

% Create heatmap for Aversive (7 periods)
figure('Position', [100, 100, 1000, 800]);
subplot(1, 2, 1);
aversive_tbl = tbl(tbl.SessionType == 'Aversive', :);
heatmap_data_aversive = createHeatmapData(aversive_tbl, all_metrics, 7);
imagesc(heatmap_data_aversive');
colorbar;
xlabel('Period', 'FontSize', 12);
ylabel('Metric', 'FontSize', 12);
title('Aversive Sessions (7 periods)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', 1:7, 'YTick', 1:22, 'YTickLabel', metric_labels);
colormap(jet);

% Create heatmap for Reward (4 periods)
subplot(1, 2, 2);
reward_tbl = tbl(tbl.SessionType == 'Reward', :);
heatmap_data_reward = createHeatmapData(reward_tbl, all_metrics, 4);
imagesc(heatmap_data_reward');
colorbar;
xlabel('Period', 'FontSize', 12);
ylabel('Metric', 'FontSize', 12);
title('Reward Sessions (4 periods)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', 1:4, 'YTick', 1:22, 'YTickLabel', metric_labels);
colormap(jet);

sgtitle('All Metrics Heatmap (Z-scored): Period × SessionType', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(output_dir, 'Fig9_Comprehensive_Heatmap.png'));
fprintf('  ✓ Saved: Fig9_Comprehensive_Heatmap.png\n');

%% ========================================================================
%  SECTION 12: SUMMARY STATISTICS
%  ========================================================================

fprintf('\nGenerating summary statistics...\n');

summary_file = fullfile(output_dir, 'Summary_Statistics.txt');
fid = fopen(summary_file, 'w');

fprintf(fid, '========================================\n');
fprintf(fid, 'UNIT FEATURES ANALYSIS - SUMMARY STATISTICS\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, 'Data file: %s\n', data_file);
fprintf(fid, 'Generated: %s\n\n', datestr(now));

fprintf(fid, 'Dataset Overview:\n');
fprintf(fid, '  Total data points: %d\n', height(tbl));
fprintf(fid, '  Aversive sessions: %d (7 periods each)\n', results.n_aversive_sessions);
fprintf(fid, '  Reward sessions: %d (4 periods each)\n', results.n_reward_sessions);
fprintf(fid, '  Total units analyzed: %d\n\n', length(unique(tbl.Unit)));

fprintf(fid, 'Metrics Summary:\n');
fprintf(fid, '  Total metrics: 22\n\n');

% Statistics by SessionType
for session_type = categorical({'Aversive', 'Reward'})
    fprintf(fid, '--- %s Sessions ---\n', char(session_type));
    subset = tbl(tbl.SessionType == session_type, :);

    n_periods = length(unique(subset.Period));
    fprintf(fid, 'Periods: %d\n', n_periods);
    fprintf(fid, 'Data points: %d\n\n', height(subset));

    % Key metrics statistics
    fprintf(fid, 'Key Metrics (mean ± std):\n');
    fprintf(fid, '  FR: %.2f ± %.2f Hz\n', mean(subset.FR, 'omitnan'), std(subset.FR, 'omitnan'));
    fprintf(fid, '  CV: %.2f ± %.2f\n', mean(subset.CV, 'omitnan'), std(subset.CV, 'omitnan'));
    fprintf(fid, '  ISI_FanoFactor: %.2f ± %.2f\n', mean(subset.ISI_FanoFactor, 'omitnan'), std(subset.ISI_FanoFactor, 'omitnan'));
    fprintf(fid, '  BurstIndex: %.3f ± %.3f\n', mean(subset.BurstIndex, 'omitnan'), std(subset.BurstIndex, 'omitnan'));
    fprintf(fid, '  RefracViolations: %.2f ± %.2f %%\n\n', mean(subset.RefracViolations, 'omitnan'), std(subset.RefracViolations, 'omitnan'));
end

fclose(fid);
fprintf('  ✓ Saved: Summary_Statistics.txt\n');

%% ========================================================================
%  COMPLETION MESSAGE
%  ========================================================================

fprintf('\n========================================\n');
fprintf('VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('All figures saved to: %s/\n', output_dir);
fprintf('Total figures generated: 9\n');
fprintf('Summary statistics: %s\n', summary_file);
fprintf('========================================\n');

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function plotMetricByPeriod(tbl, metric_name, ylabel_text, color_aversive, color_reward)
% Plot metric as function of period for both session types
% Shows mean ± SEM

    % Aversive data (7 periods)
    aversive_tbl = tbl(tbl.SessionType == 'Aversive', :);
    periods_aversive = 1:7;
    mean_aversive = zeros(1, 7);
    sem_aversive = zeros(1, 7);

    for p = 1:7
        data = aversive_tbl.(metric_name)(aversive_tbl.Period == categorical(p));
        data = data(~isnan(data));
        if ~isempty(data)
            mean_aversive(p) = mean(data);
            sem_aversive(p) = std(data) / sqrt(length(data));
        else
            mean_aversive(p) = NaN;
            sem_aversive(p) = NaN;
        end
    end

    % Reward data (4 periods)
    reward_tbl = tbl(tbl.SessionType == 'Reward', :);
    periods_reward = 1:4;
    mean_reward = zeros(1, 4);
    sem_reward = zeros(1, 4);

    for p = 1:4
        data = reward_tbl.(metric_name)(reward_tbl.Period == categorical(p));
        data = data(~isnan(data));
        if ~isempty(data)
            mean_reward(p) = mean(data);
            sem_reward(p) = std(data) / sqrt(length(data));
        else
            mean_reward(p) = NaN;
            sem_reward(p) = NaN;
        end
    end

    % Plot
    hold on;

    % Aversive
    errorbar(periods_aversive, mean_aversive, sem_aversive, '-o', ...
             'Color', color_aversive, 'LineWidth', 2, 'MarkerSize', 8, ...
             'MarkerFaceColor', color_aversive, 'DisplayName', 'Aversive');

    % Reward
    errorbar(periods_reward, mean_reward, sem_reward, '-s', ...
             'Color', color_reward, 'LineWidth', 2, 'MarkerSize', 8, ...
             'MarkerFaceColor', color_reward, 'DisplayName', 'Reward');

    xlabel('Period', 'FontSize', 11);
    ylabel(ylabel_text, 'FontSize', 11);
    legend('Location', 'best');
    grid on;
    box on;

    % Set x-axis to show all periods
    xlim([0.5, 7.5]);
    set(gca, 'XTick', 1:7);

    hold off;
end

function heatmap_data = createHeatmapData(tbl, metrics, n_periods)
% Create heatmap matrix: periods × metrics (Z-scored)

    heatmap_data = nan(n_periods, length(metrics));

    for p = 1:n_periods
        period_data = tbl(tbl.Period == categorical(p), :);

        for m = 1:length(metrics)
            metric = metrics{m};
            values = period_data.(metric);
            values = values(~isnan(values));

            if ~isempty(values)
                heatmap_data(p, m) = mean(values);
            end
        end
    end

    % Z-score each metric across periods
    for m = 1:length(metrics)
        metric_values = heatmap_data(:, m);
        if sum(~isnan(metric_values)) > 1
            heatmap_data(:, m) = (metric_values - nanmean(metric_values)) / nanstd(metric_values);
        end
    end
end
