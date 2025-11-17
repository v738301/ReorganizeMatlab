%% ========================================================================
%  VISUALIZE AMPLITUDE CORRELATION ANALYSIS WITH BREATHING SIGNAL
%  Pearson Correlation & Mutual Information: Spike rate vs Breathing amplitude
%  ========================================================================
%
%  Creates visualizations to understand spike-breathing amplitude coupling
%
%  Figure 1: Pearson Correlation Spectrum (Linear coupling)
%  Figure 2: Mutual Information Spectrum (Nonlinear/context-dependent)
%  Figure 3: MI vs Pearson Scatter (Identify nonlinear relationships)
%
%  BREATHING SIGNAL: Uses channel 32 from ephy data
%
%  Interpretation:
%  - High MI + Low Pearson → Context-dependent nonlinear coupling
%  - High MI + Low PPC → Amplitude-modulated, not phase-locked
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: LOAD DATA
%  ========================================================================

fprintf('=== VISUALIZING AMPLITUDE CORRELATION (BREATHING) ===\n');
fprintf('Breathing Signal (Channel 32)\n\n');

DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_AmpCorr_Breathing');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_AmpCorr_Breathing');

% Load files
fprintf('Loading amplitude correlation data...\n');
aversive_files = dir(fullfile(RewardAversivePath, '*_ampcorr_breathing.mat'));
reward_files = dir(fullfile(RewardSeekingPath, '*_ampcorr_breathing.mat'));

fprintf('  Aversive: %d files\n', length(aversive_files));
fprintf('  Reward: %d files\n', length(reward_files));

% Combine data
all_data_aversive = [];
all_data_reward = [];

for i = 1:length(aversive_files)
    load(fullfile(RewardAversivePath, aversive_files(i).name), 'session_results');
    if ~isempty(session_results.data)
        session_results.data.SessionType = repmat({'Aversive'}, height(session_results.data), 1);
        all_data_aversive = [all_data_aversive; session_results.data];
    end
end

for i = 1:length(reward_files)
    load(fullfile(RewardSeekingPath, reward_files(i).name), 'session_results');
    if ~isempty(session_results.data)
        session_results.data.SessionType = repmat({'Reward'}, height(session_results.data), 1);
        all_data_reward = [all_data_reward; session_results.data];
    end
end

tbl = [all_data_aversive; all_data_reward];
tbl.SessionType = categorical(tbl.SessionType);

fprintf('✓ Data loaded: %d total entries\n\n', height(tbl));

%% ========================================================================
%  SECTION 2: SETUP
%  ========================================================================

output_dir = 'AmpCorr_Breathing_Figures';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

color_aversive = [0.8 0.2 0.2];
color_reward = [0.2 0.4 0.8];
color_periods = lines(7);  % For period-specific plots

%% ========================================================================
%  FIGURE 1: PEARSON CORRELATION SPECTRUM
%  ========================================================================

fprintf('Creating Figure 1: Pearson Correlation Spectrum (Breathing)...\n');

fig1 = figure('Position', [50, 50, 1600, 900]);
sgtitle('Spike Rate vs Breathing Amplitude: Pearson Correlation (Linear Coupling, Ch 32)', 'FontSize', 16, 'FontWeight', 'bold');

% Get frequency centers
freq_centers = (tbl.Freq_Low_Hz + tbl.Freq_High_Hz) / 2;
unique_freqs = unique(freq_centers);

% Aversive (7 periods)
for p = 1:7
    subplot(2, 4, p);
    aversive_data = tbl(tbl.SessionType == 'Aversive' & tbl.Period == p, :);

    if ~isempty(aversive_data)
        plot_correlation_spectrum(aversive_data, unique_freqs, color_aversive);
        title(sprintf('Aversive P%d', p), 'FontSize', 12, 'FontWeight', 'bold');
        if p == 1, ylabel('Pearson R', 'FontSize', 11); end
        if p >= 5, xlabel('Frequency (Hz)', 'FontSize', 11); end
    end
end

% Reward (4 periods)
subplot(2, 4, 8);
for p = 1:4
    reward_data = tbl(tbl.SessionType == 'Reward' & tbl.Period == p, :);

    if ~isempty(reward_data)
        [mean_r, ~, ~] = compute_frequency_average(reward_data, unique_freqs, 'Pearson_R');
        hold on;
        plot(unique_freqs, mean_r, 'o-', 'Color', color_periods(p, :), ...
             'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', color_periods(p, :), ...
             'DisplayName', sprintf('P%d', p));
    end
end

% Highlight breathing frequencies
plot([2 2], ylim, 'r--', 'LineWidth', 2, 'HandleVisibility', 'off');
text(2, max(ylim)*0.9, ' 2 Hz', 'FontSize', 10, 'Color', 'r', 'FontWeight', 'bold');
plot([8 8], ylim, 'b--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
text(8, max(ylim)*0.8, ' 8 Hz', 'FontSize', 10, 'Color', 'b', 'FontWeight', 'bold');
xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Pearson R', 'FontSize', 11);
title('Reward (All Periods)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on; box on;
xlim([0 20]);

saveas(fig1, fullfile(output_dir, 'Figure1_Pearson_Correlation_Spectrum_Breathing.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  FIGURE 2: MUTUAL INFORMATION SPECTRUM
%  ========================================================================

fprintf('Creating Figure 2: Mutual Information Spectrum (Breathing)...\n');

fig2 = figure('Position', [100, 100, 1600, 900]);
sgtitle('Spike Rate vs Breathing Amplitude: Mutual Information (Nonlinear, Ch 32)', 'FontSize', 16, 'FontWeight', 'bold');

% Aversive (7 periods)
for p = 1:7
    subplot(2, 4, p);
    aversive_data = tbl(tbl.SessionType == 'Aversive' & tbl.Period == p, :);

    if ~isempty(aversive_data)
        plot_mi_spectrum(aversive_data, unique_freqs, color_aversive);
        title(sprintf('Aversive P%d', p), 'FontSize', 12, 'FontWeight', 'bold');
        if p == 1, ylabel('Mutual Info (bits)', 'FontSize', 11); end
        if p >= 5, xlabel('Frequency (Hz)', 'FontSize', 11); end
    end
end

% Reward (4 periods)
subplot(2, 4, 8);
for p = 1:4
    reward_data = tbl(tbl.SessionType == 'Reward' & tbl.Period == p, :);

    if ~isempty(reward_data)
        [mean_mi, ~, ~] = compute_frequency_average(reward_data, unique_freqs, 'Mutual_Info');
        hold on;
        plot(unique_freqs, mean_mi, 'o-', 'Color', color_periods(p, :), ...
             'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', color_periods(p, :), ...
             'DisplayName', sprintf('P%d', p));
    end
end

plot([2 2], ylim, 'r--', 'LineWidth', 2, 'HandleVisibility', 'off');
text(2, max(ylim)*0.9, ' 2 Hz', 'FontSize', 10, 'Color', 'r', 'FontWeight', 'bold');
plot([8 8], ylim, 'b--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
text(8, max(ylim)*0.8, ' 8 Hz', 'FontSize', 10, 'Color', 'b', 'FontWeight', 'bold');
xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Mutual Info (bits)', 'FontSize', 11);
title('Reward (All Periods)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on; box on;
xlim([0 20]);

saveas(fig2, fullfile(output_dir, 'Figure2_Mutual_Information_Spectrum_Breathing.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  FIGURE 3: MI vs PEARSON SCATTER (Identify Nonlinear Coupling)
%  ========================================================================

fprintf('Creating Figure 3: MI vs Pearson Scatter (Breathing)...\n');

fig3 = figure('Position', [150, 150, 1600, 800]);
sgtitle('Mutual Information vs Pearson Correlation (Breathing, Ch 32)', 'FontSize', 16, 'FontWeight', 'bold');

% Aversive
subplot(1, 2, 1);
aversive_data = tbl(tbl.SessionType == 'Aversive', :);
plot_mi_vs_pearson_scatter(aversive_data, 7, color_periods);
title('Aversive Sessions', 'FontSize', 14, 'FontWeight', 'bold');

% Reward
subplot(1, 2, 2);
reward_data = tbl(tbl.SessionType == 'Reward', :);
plot_mi_vs_pearson_scatter(reward_data, 4, color_periods);
title('Reward Sessions', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig3, fullfile(output_dir, 'Figure3_MI_vs_Pearson_Scatter_Breathing.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  FIGURE 4: BREATHING FREQUENCY BANDS DETAILED COMPARISON
%  ========================================================================

fprintf('Creating Figure 4: Breathing Frequency Bands Analysis...\n');

fig4 = figure('Position', [200, 200, 1400, 600]);
sgtitle('Breathing Frequency Bands: Amplitude Coupling Analysis (Ch 32)', 'FontSize', 16, 'FontWeight', 'bold');

% Filter data for breathing-relevant frequencies (1-4 Hz)
data_breathing = tbl(tbl.Freq_Low_Hz >= 1 & tbl.Freq_High_Hz <= 4, :);

% Panel 1: Pearson correlation
subplot(1, 3, 1);
plot_breathing_comparison(data_breathing, 'Pearson_R', 'Pearson R', color_aversive, color_reward);

% Panel 2: Mutual information
subplot(1, 3, 2);
plot_breathing_comparison(data_breathing, 'Mutual_Info', 'Mutual Info (bits)', color_aversive, color_reward);

% Panel 3: Scatter for breathing band only
subplot(1, 3, 3);
hold on;

% Aversive
aversive_breathing = data_breathing(data_breathing.SessionType == 'Aversive', :);
scatter(aversive_breathing.Pearson_R, aversive_breathing.Mutual_Info, 50, color_aversive, 'filled', 'MarkerFaceAlpha', 0.5);

% Reward
reward_breathing = data_breathing(data_breathing.SessionType == 'Reward', :);
scatter(reward_breathing.Pearson_R, reward_breathing.Mutual_Info, 50, color_reward, 'filled', 'MarkerFaceAlpha', 0.5);

% Reference lines
plot([0 0], ylim, 'k--', 'LineWidth', 1);
plot(xlim, [0 0], 'k--', 'LineWidth', 1);

xlabel('Pearson R (Linear)', 'FontSize', 12);
ylabel('Mutual Info (Nonlinear)', 'FontSize', 12);
title('Breathing Band (1-4 Hz): MI vs Pearson', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'best', 'FontSize', 10);
grid on; box on;
axis equal tight;

saveas(fig4, fullfile(output_dir, 'Figure4_Breathing_Bands_Analysis.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  COMPLETION
%  ========================================================================

fprintf('\n========================================\n');
fprintf('AMPLITUDE CORRELATION VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Figures saved to: %s/\n', output_dir);
fprintf('\nInterpretation (Breathing Signal - Channel 32):\n');
fprintf('  Figure 1: Pearson correlation shows LINEAR spike-breathing amplitude coupling\n');
fprintf('  Figure 2: Mutual information shows ALL coupling (linear + nonlinear)\n');
fprintf('  Figure 3: High MI + Low Pearson → Nonlinear/context-dependent breathing coupling\n');
fprintf('  Figure 4: Breathing band (1-4 Hz) detail\n');
fprintf('\nKey Insights:\n');
fprintf('  High MI at breathing frequencies → Neurons ARE amplitude-modulated by breathing\n');
fprintf('  Compare with PPC results → Amplitude vs phase coupling\n');
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function plot_correlation_spectrum(data, unique_freqs, color)
% Plot Pearson correlation spectrum with error bars

    [mean_r, sem_r, freq_centers] = compute_frequency_average(data, unique_freqs, 'Pearson_R');

    hold on;
    errorbar(freq_centers, mean_r, sem_r, 'o-', 'Color', color, ...
             'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', color, 'CapSize', 4);

    % Highlight breathing frequencies
    plot([2 2], ylim, 'r--', 'LineWidth', 2);
    text(2, max(ylim)*0.9, ' 2 Hz', 'FontSize', 9, 'Color', 'r', 'FontWeight', 'bold');

    % Zero reference
    plot(xlim, [0 0], 'k--', 'LineWidth', 0.5);

    xlabel('Frequency (Hz)', 'FontSize', 11);
    ylabel('Pearson R', 'FontSize', 11);
    grid on; box on;
    xlim([0 20]);
    ylim([-0.3 0.6]);
end

function plot_mi_spectrum(data, unique_freqs, color)
% Plot mutual information spectrum with error bars

    [mean_mi, sem_mi, freq_centers] = compute_frequency_average(data, unique_freqs, 'Mutual_Info');

    hold on;
    errorbar(freq_centers, mean_mi, sem_mi, 'o-', 'Color', color, ...
             'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', color, 'CapSize', 4);

    % Highlight breathing frequencies
    plot([2 2], ylim, 'r--', 'LineWidth', 2);
    text(2, max(ylim)*0.9, ' 2 Hz', 'FontSize', 9, 'Color', 'r', 'FontWeight', 'bold');

    xlabel('Frequency (Hz)', 'FontSize', 11);
    ylabel('Mutual Info (bits)', 'FontSize', 11);
    grid on; box on;
    xlim([0 20]);
    ylim([0 max(mean_mi)*1.2]);
end

function [mean_vals, sem_vals, freq_centers] = compute_frequency_average(data, unique_freqs, variable_name)
% Compute average and SEM for a variable across frequencies

    n_freqs = length(unique_freqs);
    mean_vals = zeros(1, n_freqs);
    sem_vals = zeros(1, n_freqs);
    freq_centers = unique_freqs;

    for f = 1:n_freqs
        freq = unique_freqs(f);
        freq_mask = abs((data.Freq_Low_Hz + data.Freq_High_Hz)/2 - freq) < 0.1;
        values = data.(variable_name)(freq_mask);

        if ~isempty(values)
            mean_vals(f) = mean(values, 'omitnan');
            sem_vals(f) = std(values, 'omitnan') / sqrt(sum(~isnan(values)));
        end
    end
end

function plot_mi_vs_pearson_scatter(data, n_periods, color_periods)
% Scatter plot: MI vs Pearson, colored by period

    hold on;

    for p = 1:n_periods
        period_data = data(data.Period == p, :);

        if ~isempty(period_data)
            scatter(period_data.Pearson_R, period_data.Mutual_Info, 50, ...
                    color_periods(p, :), 'filled', 'MarkerFaceAlpha', 0.4, ...
                    'DisplayName', sprintf('P%d', p));
        end
    end

    % Reference lines
    plot([0 0], ylim, 'k--', 'LineWidth', 1);
    plot(xlim, [0 0], 'k--', 'LineWidth', 1);

    % Unity line (if MI ≈ Pearson, coupling is linear)
    max_val = max([xlim, ylim]);
    plot([0 max_val], [0 max_val], 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    text(max_val*0.7, max_val*0.75, 'Linear', 'FontSize', 10, 'Rotation', 45, 'Color', [0.5 0.5 0.5]);

    xlabel('Pearson R (Linear Coupling)', 'FontSize', 12);
    ylabel('Mutual Information (Total Coupling)', 'FontSize', 12);
    legend('Location', 'northwest', 'FontSize', 9);
    grid on; box on;
    xlim([-0.3 0.6]);
    ylim([0 max(data.Mutual_Info)*1.1]);
end

function plot_breathing_comparison(data_breathing, variable_name, ylabel_text, color_aversive, color_reward)
% Bar plot comparing aversive vs reward for breathing band

    % Compute averages
    aversive_vals = data_breathing.(variable_name)(data_breathing.SessionType == 'Aversive');
    reward_vals = data_breathing.(variable_name)(data_breathing.SessionType == 'Reward');

    mean_aversive = mean(aversive_vals, 'omitnan');
    mean_reward = mean(reward_vals, 'omitnan');

    sem_aversive = std(aversive_vals, 'omitnan') / sqrt(sum(~isnan(aversive_vals)));
    sem_reward = std(reward_vals, 'omitnan') / sqrt(sum(~isnan(reward_vals)));

    % Bar plot
    hold on;
    bar(1, mean_aversive, 'FaceColor', color_aversive, 'EdgeColor', 'k', 'LineWidth', 1.5);
    bar(2, mean_reward, 'FaceColor', color_reward, 'EdgeColor', 'k', 'LineWidth', 1.5);

    errorbar([1 2], [mean_aversive mean_reward], [sem_aversive sem_reward], ...
             'k', 'LineStyle', 'none', 'LineWidth', 2, 'CapSize', 10);

    % Zero reference
    plot(xlim, [0 0], 'k--', 'LineWidth', 1);

    set(gca, 'XTick', [1 2], 'XTickLabel', {'Aversive', 'Reward'});
    ylabel(ylabel_text, 'FontSize', 12);
    title(sprintf('Breathing (1-4 Hz): %s', ylabel_text), 'FontSize', 12, 'FontWeight', 'bold');
    grid on; box on;
    xlim([0.5 2.5]);
end
