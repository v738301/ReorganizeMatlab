%% ========================================================================
%  VISUALIZE UNIT FEATURES: Period × SessionType Analysis
%  Comprehensive visualization of 22 spike train metrics
%  ========================================================================
%
%  Visualizes results from Unit_Features_Analysis.m
%
%  Creates 3 comprehensive figures:
%  Figure 1: Core Spike Metrics (4x4 grid, 16 subplots)
%  Figure 2: Temporal Structure Metrics (3x3 grid, 9 subplots)
%  Figure 3: Comprehensive Heatmap (all 22 metrics)
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
%  FIGURE 1: CORE SPIKE METRICS (4x4 = 16 subplots)
%  ========================================================================

fprintf('Creating Figure 1: Core Spike Metrics...\n');

fig1 = figure('Position', [50, 50, 1800, 1400]);
sgtitle('Core Spike Metrics: Period × SessionType', 'FontSize', 16, 'FontWeight', 'bold');

% Row 1: Basic Firing Metrics
subplot(4, 4, 1);
plotMetricByPeriod(tbl, 'FR', 'Firing Rate (Hz)', color_aversive, color_reward);

subplot(4, 4, 2);
plotMetricByPeriod(tbl, 'CV', 'Coefficient of Variation', color_aversive, color_reward);

subplot(4, 4, 3);
plotMetricByPeriod(tbl, 'ISI_FanoFactor', 'ISI Fano Factor', color_aversive, color_reward);

subplot(4, 4, 4);
plotMetricByPeriod(tbl, 'RefracViolations', 'Refractory Violations (%)', color_aversive, color_reward);

% Row 2: ISI Variability
subplot(4, 4, 5);
plotMetricByPeriod(tbl, 'LV', 'Local Variation (LV)', color_aversive, color_reward);

subplot(4, 4, 6);
plotMetricByPeriod(tbl, 'CV2', 'Local CV (CV2)', color_aversive, color_reward);

subplot(4, 4, 7);
plotMetricByPeriod(tbl, 'LVR', 'Revised Local Var (LVR)', color_aversive, color_reward);

subplot(4, 4, 8);
plotMetricByPeriod(tbl, 'ISI_Mode', 'ISI Mode (s)', color_aversive, color_reward);

% Row 3: Burst Metrics
subplot(4, 4, 9);
plotMetricByPeriod(tbl, 'BurstIndex', 'Burst Index', color_aversive, color_reward);

subplot(4, 4, 10);
plotMetricByPeriod(tbl, 'BurstRate', 'Burst Rate (bursts/s)', color_aversive, color_reward);

subplot(4, 4, 11);
plotMetricByPeriod(tbl, 'MeanBurstLength', 'Mean Burst Length (spikes)', color_aversive, color_reward);

subplot(4, 4, 12);
plotMetricByPeriod(tbl, 'ISI_Skewness', 'ISI Skewness', color_aversive, color_reward);

% Row 4: Distribution Shape
subplot(4, 4, 13);
plotMetricByPeriod(tbl, 'ISI_Kurtosis', 'ISI Kurtosis', color_aversive, color_reward);

subplot(4, 4, 14);
plotMetricByPeriod(tbl, 'ISI_ACF_peak', 'ISI ACF Peak', color_aversive, color_reward);

subplot(4, 4, 15);
plotMetricByPeriod(tbl, 'ISI_ACF_lag', 'ISI ACF Lag (s)', color_aversive, color_reward);

subplot(4, 4, 16);
plotMetricByPeriod(tbl, 'ISI_ACF_decay', 'ISI ACF Decay (s)', color_aversive, color_reward);

saveas(fig1, fullfile(output_dir, 'Figure1_Core_Spike_Metrics.png'));
fprintf('  ✓ Saved: Figure1_Core_Spike_Metrics.png\n');

%% ========================================================================
%  FIGURE 2: TEMPORAL STRUCTURE METRICS (3x3 = 9 subplots)
%  ========================================================================

fprintf('Creating Figure 2: Temporal Structure Metrics...\n');

fig2 = figure('Position', [100, 100, 1600, 1200]);
sgtitle('Temporal Structure Metrics: Period × SessionType', 'FontSize', 16, 'FontWeight', 'bold');

% Row 1: Count ACF (3 bin sizes)
subplot(3, 3, 1);
plotMetricByPeriod(tbl, 'Count_ACF_1ms_peak', 'Count ACF Peak (1ms)', color_aversive, color_reward);

subplot(3, 3, 2);
plotMetricByPeriod(tbl, 'Count_ACF_25ms_peak', 'Count ACF Peak (25ms)', color_aversive, color_reward);

subplot(3, 3, 3);
plotMetricByPeriod(tbl, 'Count_ACF_50ms_peak', 'Count ACF Peak (50ms)', color_aversive, color_reward);

% Row 2: Count Fano Factor (3 bin sizes)
subplot(3, 3, 4);
plotMetricByPeriod(tbl, 'CountFanoFactor_1ms', 'Count Fano Factor (1ms)', color_aversive, color_reward);

subplot(3, 3, 5);
plotMetricByPeriod(tbl, 'CountFanoFactor_25ms', 'Count Fano Factor (25ms)', color_aversive, color_reward);

subplot(3, 3, 6);
plotMetricByPeriod(tbl, 'CountFanoFactor_50ms', 'Count Fano Factor (50ms)', color_aversive, color_reward);

% Row 3: Summary comparison plots
subplot(3, 3, 7);
% Compare ACF across bin sizes for Aversive
plotACFComparison(tbl, 'Aversive', color_aversive);

subplot(3, 3, 8);
% Compare ACF across bin sizes for Reward
plotACFComparison(tbl, 'Reward', color_reward);

subplot(3, 3, 9);
% Compare Fano Factor across bin sizes
plotFanoComparison(tbl, color_aversive, color_reward);

saveas(fig2, fullfile(output_dir, 'Figure2_Temporal_Structure_Metrics.png'));
fprintf('  ✓ Saved: Figure2_Temporal_Structure_Metrics.png\n');

%% ========================================================================
%  FIGURE 3: COMPREHENSIVE HEATMAP
%  ========================================================================

fprintf('Creating Figure 3: Comprehensive Heatmap...\n');

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

fig3 = figure('Position', [150, 150, 1400, 900]);
sgtitle('All Metrics Heatmap (Z-scored): Period × SessionType', 'FontSize', 16, 'FontWeight', 'bold');

% Create heatmap for Aversive (7 periods)
subplot(1, 2, 1);
aversive_tbl = tbl(tbl.SessionType == 'Aversive', :);
heatmap_data_aversive = createHeatmapData(aversive_tbl, all_metrics, 7);
imagesc(heatmap_data_aversive');
colorbar;
caxis([-2, 2]);  % Set consistent color scale
xlabel('Period', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Metric', 'FontSize', 12, 'FontWeight', 'bold');
title('Aversive Sessions (7 periods)', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'XTick', 1:7, 'YTick', 1:22, 'YTickLabel', metric_labels, 'FontSize', 10);
colormap(jet);

% Create heatmap for Reward (4 periods)
subplot(1, 2, 2);
reward_tbl = tbl(tbl.SessionType == 'Reward', :);
heatmap_data_reward = createHeatmapData(reward_tbl, all_metrics, 4);
imagesc(heatmap_data_reward');
colorbar;
caxis([-2, 2]);  % Set consistent color scale
xlabel('Period', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Metric', 'FontSize', 12, 'FontWeight', 'bold');
title('Reward Sessions (4 periods)', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'XTick', 1:4, 'YTick', 1:22, 'YTickLabel', metric_labels, 'FontSize', 10);
colormap(jet);

saveas(fig3, fullfile(output_dir, 'Figure3_Comprehensive_Heatmap.png'));
fprintf('  ✓ Saved: Figure3_Comprehensive_Heatmap.png\n');

%% ========================================================================
%  SECTION 3: SUMMARY STATISTICS
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
fprintf('Total figures generated: 3\n');
fprintf('  - Figure 1: Core Spike Metrics (16 subplots)\n');
fprintf('  - Figure 2: Temporal Structure Metrics (9 subplots)\n');
fprintf('  - Figure 3: Comprehensive Heatmap\n');
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
             'Color', color_aversive, 'LineWidth', 1.5, 'MarkerSize', 6, ...
             'MarkerFaceColor', color_aversive, 'DisplayName', 'Aversive');

    % Reward
    errorbar(periods_reward, mean_reward, sem_reward, '-s', ...
             'Color', color_reward, 'LineWidth', 1.5, 'MarkerSize', 6, ...
             'MarkerFaceColor', color_reward, 'DisplayName', 'Reward');

    xlabel('Period', 'FontSize', 10);
    ylabel(ylabel_text, 'FontSize', 10);
    legend('Location', 'best', 'FontSize', 8);
    grid on;
    box on;

    % Set x-axis to show all periods
    xlim([0.5, 7.5]);
    set(gca, 'XTick', 1:7);

    hold off;
end

function plotACFComparison(tbl, session_type, color)
% Compare ACF peaks across different bin sizes for one session type

    subset = tbl(tbl.SessionType == session_type, :);
    periods = unique(subset.Period);
    n_periods = length(periods);

    mean_1ms = zeros(1, n_periods);
    mean_25ms = zeros(1, n_periods);
    mean_50ms = zeros(1, n_periods);

    for i = 1:n_periods
        p = periods(i);
        period_data = subset(subset.Period == p, :);

        mean_1ms(i) = mean(period_data.Count_ACF_1ms_peak, 'omitnan');
        mean_25ms(i) = mean(period_data.Count_ACF_25ms_peak, 'omitnan');
        mean_50ms(i) = mean(period_data.Count_ACF_50ms_peak, 'omitnan');
    end

    hold on;
    plot(1:n_periods, mean_1ms, '-o', 'Color', color, 'LineWidth', 1.5, 'DisplayName', '1ms');
    plot(1:n_periods, mean_25ms, '-s', 'Color', color*0.7, 'LineWidth', 1.5, 'DisplayName', '25ms');
    plot(1:n_periods, mean_50ms, '-d', 'Color', color*0.4, 'LineWidth', 1.5, 'DisplayName', '50ms');

    xlabel('Period', 'FontSize', 10);
    ylabel('ACF Peak', 'FontSize', 10);
    title(sprintf('%s: ACF by Bin Size', session_type), 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 8);
    grid on;
    box on;
    xlim([0.5, n_periods + 0.5]);
    hold off;
end

function plotFanoComparison(tbl, color_aversive, color_reward)
% Compare Fano Factor across bin sizes for both session types

    % Aversive
    aversive_tbl = tbl(tbl.SessionType == 'Aversive', :);
    periods_aversive = 1:7;

    mean_1ms_av = zeros(1, 7);
    mean_25ms_av = zeros(1, 7);
    mean_50ms_av = zeros(1, 7);

    for p = 1:7
        period_data = aversive_tbl(aversive_tbl.Period == categorical(p), :);
        mean_1ms_av(p) = mean(period_data.CountFanoFactor_1ms, 'omitnan');
        mean_25ms_av(p) = mean(period_data.CountFanoFactor_25ms, 'omitnan');
        mean_50ms_av(p) = mean(period_data.CountFanoFactor_50ms, 'omitnan');
    end

    % Reward
    reward_tbl = tbl(tbl.SessionType == 'Reward', :);
    periods_reward = 1:4;

    mean_25ms_rw = zeros(1, 4);

    for p = 1:4
        period_data = reward_tbl(reward_tbl.Period == categorical(p), :);
        mean_25ms_rw(p) = mean(period_data.CountFanoFactor_25ms, 'omitnan');
    end

    hold on;
    plot(periods_aversive, mean_25ms_av, '-o', 'Color', color_aversive, ...
         'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Aversive 25ms');
    plot(periods_reward, mean_25ms_rw, '-s', 'Color', color_reward, ...
         'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Reward 25ms');

    xlabel('Period', 'FontSize', 10);
    ylabel('Fano Factor', 'FontSize', 10);
    title('Fano Factor Comparison (25ms)', 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 8);
    grid on;
    box on;
    xlim([0.5, 7.5]);
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
