%% ========================================================================
%  VISUALIZE LFP-BREATHING POWER CORRELATION & COHERENCE
%  Multi-panel visualization of time-resolved power-coherence relationships
%  ========================================================================
%
%  This script creates comprehensive visualizations addressing:
%  1. Do LFP and breathing power correlate at specific frequencies?
%  2. Is this power correlation constant across the session?
%  3. Can coherence be explained by concurrent power correlation?
%
%  Visualizations:
%  - Time-frequency heatmaps of power correlation
%  - Time series of power correlation for key frequencies
%  - Scatter plots: coherence vs power correlation
%  - Period-wise comparisons
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  CONFIGURATION
%  ========================================================================

fprintf('=== VISUALIZE LFP-BREATHING POWER CORRELATION & COHERENCE ===\n\n');

% Data paths
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_LFP_Breathing_PowerCoh');
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_LFP_Breathing_PowerCoh');

% Figure output path
FigurePath = fullfile(DataSetsPath, 'Figures_LFP_Breathing_PowerCoh');
if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end

% Visualization parameters
viz_config = struct();
viz_config.freq_bands_of_interest = [0.5, 1.5, 3.5, 8.5, 15.5];  % Center frequencies to highlight
viz_config.significance_threshold = 0.05;  % p-value threshold
viz_config.colormap_corr = 'RdBu';  % Diverging colormap for correlation
viz_config.colormap_coh = 'hot';    % Sequential colormap for coherence

%% ========================================================================
%  LOAD DATA
%  ========================================================================

fprintf('Loading data files...\n');

% Load aversive sessions
aversive_files = dir(fullfile(RewardAversivePath, '*_lfp_breathing_powercoh.mat'));
fprintf('  Found %d aversive sessions\n', length(aversive_files));

% Load reward sessions
reward_files = dir(fullfile(RewardSeekingPath, '*_lfp_breathing_powercoh.mat'));
fprintf('  Found %d reward sessions\n', length(reward_files));

if isempty(aversive_files) && isempty(reward_files)
    error('No data files found. Please run LFP_Breathing_Power_Coherence_Analysis.m first.');
end

%% ========================================================================
%  VISUALIZE AVERSIVE SESSIONS
%  ========================================================================

if ~isempty(aversive_files)
    fprintf('\n==== VISUALIZING AVERSIVE SESSIONS ====\n');

    for file_idx = 1:length(aversive_files)
        fprintf('\n[%d/%d] Processing: %s\n', file_idx, length(aversive_files), aversive_files(file_idx).name);

        % Load session data
        load(fullfile(RewardAversivePath, aversive_files(file_idx).name), 'session_results', 'config');
        data = session_results.data;

        % Create comprehensive visualization
        fig = figure('Position', [50, 50, 1800, 1200]);
        sgtitle(sprintf('LFP-Breathing Power-Coherence Analysis: %s', ...
                       strrep(session_results.filename, '_', '\_')), ...
               'FontSize', 14, 'FontWeight', 'bold');

        % === PANEL 1: Time-Frequency Heatmap of Power Correlation ===
        subplot(3, 3, [1, 2, 4, 5]);
        plot_power_correlation_heatmap(data, 'Aversive');
        title('1. Power Correlation: Time × Frequency', 'FontSize', 12, 'FontWeight', 'bold');

        % === PANEL 2: Time-Frequency Heatmap of Coherence ===
        subplot(3, 3, [3, 6]);
        plot_coherence_heatmap(data, 'Aversive');
        title('2. Coherence: Time × Frequency', 'FontSize', 12, 'FontWeight', 'bold');

        % === PANEL 3: Time Series of Power Correlation at Key Frequencies ===
        subplot(3, 3, 7);
        plot_power_correlation_timeseries(data, viz_config.freq_bands_of_interest);
        title('3. Power Correlation Over Time', 'FontSize', 11, 'FontWeight', 'bold');
        xlabel('Time (sec)');
        ylabel('Power Correlation (r)');

        % === PANEL 4: Coherence vs Power Correlation Scatter ===
        subplot(3, 3, 8);
        plot_coherence_vs_power_correlation(data);
        title('4. Coherence vs Power Correlation', 'FontSize', 11, 'FontWeight', 'bold');
        xlabel('Power Correlation (r)');
        ylabel('Coherence');

        % === PANEL 5: Period-wise Power Correlation ===
        subplot(3, 3, 9);
        plot_period_comparison(data);
        title('5. Power Correlation by Period', 'FontSize', 11, 'FontWeight', 'bold');
        xlabel('Period');
        ylabel('Mean Power Correlation (r)');

        % Save figure
        [~, base_name, ~] = fileparts(aversive_files(file_idx).name);
        saveas(fig, fullfile(FigurePath, sprintf('%s_visualization.png', base_name)));
        saveas(fig, fullfile(FigurePath, sprintf('%s_visualization.fig', base_name)));
        close(fig);

        fprintf('  ✓ Saved visualization\n');
    end
end

%% ========================================================================
%  VISUALIZE REWARD SESSIONS
%  ========================================================================

if ~isempty(reward_files)
    fprintf('\n==== VISUALIZING REWARD SESSIONS ====\n');

    for file_idx = 1:length(reward_files)
        fprintf('\n[%d/%d] Processing: %s\n', file_idx, length(reward_files), reward_files(file_idx).name);

        % Load session data
        load(fullfile(RewardSeekingPath, reward_files(file_idx).name), 'session_results', 'config');
        data = session_results.data;

        % Create comprehensive visualization
        fig = figure('Position', [50, 50, 1800, 1200]);
        sgtitle(sprintf('LFP-Breathing Power-Coherence Analysis: %s', ...
                       strrep(session_results.filename, '_', '\_')), ...
               'FontSize', 14, 'FontWeight', 'bold');

        % === PANEL 1: Time-Frequency Heatmap of Power Correlation ===
        subplot(3, 3, [1, 2, 4, 5]);
        plot_power_correlation_heatmap(data, 'Reward');
        title('1. Power Correlation: Time × Frequency', 'FontSize', 12, 'FontWeight', 'bold');

        % === PANEL 2: Time-Frequency Heatmap of Coherence ===
        subplot(3, 3, [3, 6]);
        plot_coherence_heatmap(data, 'Reward');
        title('2. Coherence: Time × Frequency', 'FontSize', 12, 'FontWeight', 'bold');

        % === PANEL 3: Time Series of Power Correlation at Key Frequencies ===
        subplot(3, 3, 7);
        plot_power_correlation_timeseries(data, viz_config.freq_bands_of_interest);
        title('3. Power Correlation Over Time', 'FontSize', 11, 'FontWeight', 'bold');
        xlabel('Time (sec)');
        ylabel('Power Correlation (r)');

        % === PANEL 4: Coherence vs Power Correlation Scatter ===
        subplot(3, 3, 8);
        plot_coherence_vs_power_correlation(data);
        title('4. Coherence vs Power Correlation', 'FontSize', 11, 'FontWeight', 'bold');
        xlabel('Power Correlation (r)');
        ylabel('Coherence');

        % === PANEL 5: Period-wise Power Correlation ===
        subplot(3, 3, 9);
        plot_period_comparison(data);
        title('5. Power Correlation by Period', 'FontSize', 11, 'FontWeight', 'bold');
        xlabel('Period');
        ylabel('Mean Power Correlation (r)');

        % Save figure
        [~, base_name, ~] = fileparts(reward_files(file_idx).name);
        saveas(fig, fullfile(FigurePath, sprintf('%s_visualization.png', base_name)));
        saveas(fig, fullfile(FigurePath, sprintf('%s_visualization.fig', base_name)));
        close(fig);

        fprintf('  ✓ Saved visualization\n');
    end
end

%% ========================================================================
%  COMPLETION
%  ========================================================================

fprintf('\n========================================\n');
fprintf('VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Figures saved to: %s\n', FigurePath);
fprintf('========================================\n');


%% ========================================================================
%  PLOTTING FUNCTIONS
%  ========================================================================

function plot_power_correlation_heatmap(data, session_type)
% Plot time-frequency heatmap of power correlation
%
% Addresses Question 1 & 2:
% - Do signals power-correlate at specific frequencies?
% - Is this correlation constant across session?

    % Get unique frequencies and times
    unique_freqs = unique(data.Freq_Low_Hz);
    time_centers = data.Window_Center_Time;

    % Create grid for heatmap
    n_freqs = length(unique_freqs);
    time_min = min(time_centers);
    time_max = max(time_centers);

    % Bin time into regular intervals for cleaner visualization
    n_time_bins = 50;
    time_edges = linspace(time_min, time_max, n_time_bins + 1);
    time_bin_centers = (time_edges(1:end-1) + time_edges(2:end)) / 2;

    % Initialize correlation matrix
    corr_matrix = nan(n_freqs, n_time_bins);

    % Fill matrix with correlation values
    for freq_idx = 1:n_freqs
        freq = unique_freqs(freq_idx);
        freq_mask = data.Freq_Low_Hz == freq;

        for time_idx = 1:n_time_bins
            time_mask = time_centers >= time_edges(time_idx) & time_centers < time_edges(time_idx + 1);
            combined_mask = freq_mask & time_mask;

            if sum(combined_mask) > 0
                % Average correlation in this bin
                corr_matrix(freq_idx, time_idx) = mean(data.Power_Corr_R(combined_mask), 'omitnan');
            end
        end
    end

    % Plot heatmap
    imagesc(time_bin_centers, unique_freqs, corr_matrix);
    set(gca, 'YDir', 'normal');
    colormap(gca, brewermap(256, 'RdBu'));
    caxis([-1, 1]);
    cb = colorbar;
    cb.Label.String = 'Power Correlation (r)';
    xlabel('Time (sec)');
    ylabel('Frequency (Hz)');

    % Mark period boundaries (if available in data)
    hold on;
    periods = unique(data.Period);
    for p = 1:length(periods)-1
        period_mask = data.Period == periods(p);
        if sum(period_mask) > 0
            period_end = max(data.Window_End_Time(period_mask));
            plot([period_end, period_end], [min(unique_freqs), max(unique_freqs)], ...
                 'k--', 'LineWidth', 1.5);
        end
    end
    hold off;

    grid on;
end

function plot_coherence_heatmap(data, session_type)
% Plot time-frequency heatmap of coherence

    % Get unique frequencies and times
    unique_freqs = unique(data.Freq_Low_Hz);
    time_centers = data.Window_Center_Time;

    % Create grid
    n_freqs = length(unique_freqs);
    time_min = min(time_centers);
    time_max = max(time_centers);

    % Bin time
    n_time_bins = 50;
    time_edges = linspace(time_min, time_max, n_time_bins + 1);
    time_bin_centers = (time_edges(1:end-1) + time_edges(2:end)) / 2;

    % Initialize coherence matrix
    coh_matrix = nan(n_freqs, n_time_bins);

    % Fill matrix
    for freq_idx = 1:n_freqs
        freq = unique_freqs(freq_idx);
        freq_mask = data.Freq_Low_Hz == freq;

        for time_idx = 1:n_time_bins
            time_mask = time_centers >= time_edges(time_idx) & time_centers < time_edges(time_idx + 1);
            combined_mask = freq_mask & time_mask;

            if sum(combined_mask) > 0
                coh_matrix(freq_idx, time_idx) = mean(data.Coherence_Mean(combined_mask), 'omitnan');
            end
        end
    end

    % Plot heatmap
    imagesc(time_bin_centers, unique_freqs, coh_matrix);
    set(gca, 'YDir', 'normal');
    colormap(gca, hot);
    caxis([0, 1]);
    cb = colorbar;
    cb.Label.String = 'Coherence';
    xlabel('Time (sec)');
    ylabel('Frequency (Hz)');

    % Mark period boundaries
    hold on;
    periods = unique(data.Period);
    for p = 1:length(periods)-1
        period_mask = data.Period == periods(p);
        if sum(period_mask) > 0
            period_end = max(data.Window_End_Time(period_mask));
            plot([period_end, period_end], [min(unique_freqs), max(unique_freqs)], ...
                 'w--', 'LineWidth', 1.5);
        end
    end
    hold off;

    grid on;
end

function plot_power_correlation_timeseries(data, freq_bands_of_interest)
% Plot time series of power correlation at specific frequencies
%
% Addresses Question 2: Is power correlation constant across session?

    hold on;

    % Define colors for different frequencies
    colors = lines(length(freq_bands_of_interest));

    for freq_idx = 1:length(freq_bands_of_interest)
        target_freq = freq_bands_of_interest(freq_idx);

        % Find closest frequency band
        [~, closest_idx] = min(abs(unique(data.Freq_Low_Hz) - target_freq));
        actual_freq = unique(data.Freq_Low_Hz);
        actual_freq = actual_freq(closest_idx);

        % Get data for this frequency
        freq_mask = data.Freq_Low_Hz == actual_freq;
        times = data.Window_Center_Time(freq_mask);
        corrs = data.Power_Corr_R(freq_mask);

        % Sort by time
        [times_sorted, sort_idx] = sort(times);
        corrs_sorted = corrs(sort_idx);

        % Plot
        plot(times_sorted, corrs_sorted, 'o-', 'Color', colors(freq_idx, :), ...
             'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', sprintf('%.1f Hz', actual_freq));
    end

    hold off;
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    ylim([-1, 1]);
    yline(0, 'k--', 'LineWidth', 1);
end

function plot_coherence_vs_power_correlation(data)
% Scatter plot: coherence vs power correlation
%
% Addresses Question 3: Can coherence be explained by power correlation?
% If high coherence is driven by concurrent high power, we expect positive correlation

    % Remove NaN values
    valid_mask = ~isnan(data.Power_Corr_R) & ~isnan(data.Coherence_Mean);
    corr_vals = data.Power_Corr_R(valid_mask);
    coh_vals = data.Coherence_Mean(valid_mask);
    freqs = data.Freq_Low_Hz(valid_mask);

    % Color points by frequency
    scatter(corr_vals, coh_vals, 30, freqs, 'filled', 'MarkerFaceAlpha', 0.5);
    colormap(gca, jet);
    cb = colorbar;
    cb.Label.String = 'Frequency (Hz)';

    % Add regression line
    hold on;
    p = polyfit(corr_vals, coh_vals, 1);
    x_fit = linspace(min(corr_vals), max(corr_vals), 100);
    y_fit = polyval(p, x_fit);
    plot(x_fit, y_fit, 'r--', 'LineWidth', 2);

    % Compute correlation
    [r, pval] = corr(corr_vals, coh_vals);
    text(0.05, 0.95, sprintf('r = %.3f, p = %.4f', r, pval), ...
         'Units', 'normalized', 'FontSize', 10, 'BackgroundColor', 'white');

    hold off;
    grid on;
    xlim([-1, 1]);
    ylim([0, 1]);
end

function plot_period_comparison(data)
% Box plot comparing power correlation across periods
%
% Addresses Question 2: Is power correlation constant across session?

    periods = unique(data.Period);
    n_periods = length(periods);

    % Prepare data for boxplot
    period_data = cell(n_periods, 1);
    for p = 1:n_periods
        period_mask = data.Period == periods(p);
        period_data{p} = data.Power_Corr_R(period_mask);
    end

    % Create boxplot
    boxplot([period_data{:}], 'Labels', cellstr(num2str(periods')));
    grid on;
    ylim([-1, 1]);
    yline(0, 'k--', 'LineWidth', 1);

    % Add mean line
    hold on;
    period_means = cellfun(@(x) mean(x, 'omitnan'), period_data);
    plot(1:n_periods, period_means, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
    hold off;
end

% Helper function to get brewermap colormap (simplified version)
function cmap = brewermap(n, scheme)
% Simplified RdBu colormap generator
    if strcmp(scheme, 'RdBu')
        % Red-Blue diverging colormap
        colors = [
            178, 24, 43;      % Dark red
            214, 96, 77;      % Red
            244, 165, 130;    % Light red
            253, 219, 199;    % Very light red
            247, 247, 247;    % White (center)
            209, 229, 240;    % Very light blue
            146, 197, 222;    % Light blue
            67, 147, 195;     % Blue
            33, 102, 172      % Dark blue
        ] / 255;

        % Interpolate to n colors
        x = linspace(1, size(colors, 1), n);
        cmap = interp1(1:size(colors, 1), colors, x);
    else
        cmap = hot(n);
    end
end
