%% ========================================================================
%  VISUALIZE FIELD-TRIGGERED AVERAGE (BREATHING)
%  Spike rate as function of Breathing Phase
%  ========================================================================
%
%  Creates visualizations to test phase-locking to breathing
%
%  Figure 1: FTA Curves (Cartesian) - spike rate vs phase
%  Figure 2: FTA Polar Plots - circular representation
%  Figure 3: Mean Vector Length Spectrum - phase-locking strength
%  Figure 4: 2 Hz Detailed Analysis (breathing frequency)
%  Figure 5: 8 Hz Detailed Analysis (sniffing frequency)
%
%  BREATHING SIGNAL: Uses channel 32 from ephy data
%
%  Interpretation:
%  - Peaked FTA → Strong phase-locking (high MVL)
%  - Flat FTA → No phase preference (low MVL)
%  - MVL ≈ PPC (both measure phase-locking)
%  - 2 Hz coupling → Normal breathing rhythm
%  - 8 Hz coupling → Sniffing/exploratory breathing
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: LOAD DATA
%  ========================================================================

fprintf('=== VISUALIZING FIELD-TRIGGERED AVERAGE (BREATHING) ===\n');
fprintf('Breathing Signal (Channel 32)\n\n');

DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_FTA_Breathing');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_FTA_Breathing');

% Load files
fprintf('Loading FTA breathing data...\n');
aversive_files = dir(fullfile(RewardAversivePath, '*_fta_breathing.mat'));
reward_files = dir(fullfile(RewardSeekingPath, '*_fta_breathing.mat'));

fprintf('  Aversive: %d files\n', length(aversive_files));
fprintf('  Reward: %d files\n', length(reward_files));

% Combine data
all_data_aversive = [];
all_data_reward = [];

for i = 1:length(aversive_files)
    load(fullfile(RewardAversivePath, aversive_files(i).name), 'session_results', 'config');
    if ~isempty(session_results.data)
        session_results.data.SessionType = repmat({'Aversive'}, height(session_results.data), 1);
        session_results.data.SessionID = repmat(i, height(session_results.data), 1);
        session_results.data.SessionName = repmat({aversive_files(i).name}, height(session_results.data), 1);
        % Create unique unit ID: SessionID * 1000 + Unit (ensures uniqueness across sessions)
        session_results.data.UniqueUnitID = i * 1000 + double(session_results.data.Unit);
        all_data_aversive = [all_data_aversive; session_results.data];
    end
end

for i = 1:length(reward_files)
    load(fullfile(RewardSeekingPath, reward_files(i).name), 'session_results', 'config');
    if ~isempty(session_results.data)
        session_results.data.SessionType = repmat({'Reward'}, height(session_results.data), 1);
        session_results.data.SessionID = repmat(i + 10000, height(session_results.data), 1);
        session_results.data.SessionName = repmat({reward_files(i).name}, height(session_results.data), 1);
        % Create unique unit ID: SessionID * 1000 + Unit (10000 offset for reward sessions)
        session_results.data.UniqueUnitID = (i + 10000) * 1000 + double(session_results.data.Unit);
        all_data_reward = [all_data_reward; session_results.data];
    end
end

tbl = [all_data_aversive; all_data_reward];
tbl.SessionType = categorical(tbl.SessionType);

fprintf('✓ Data loaded: %d total entries\n\n', height(tbl));

% Get phase centers from first entry
phase_centers = tbl.Phase_Centers{1};
n_phase_bins = length(phase_centers);

%% ========================================================================
%  SECTION 2: SETUP
%  ========================================================================

output_dir = 'FTA_Breathing_Figures';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

color_aversive = [0.8 0.2 0.2];
color_reward = [0.2 0.4 0.8];
color_periods = lines(7);

%% ========================================================================
%  FIGURE 1: FTA CURVES (CARTESIAN) - 2 Hz BREATHING
%  ========================================================================

fprintf('Creating Figure 1: FTA Curves (Cartesian) at 2 Hz...\n');

fig1 = figure('Position', [50, 50, 1800, 900]);
sgtitle('Field-Triggered Average: Spike Rate vs Breathing Phase (2 Hz)', 'FontSize', 16, 'FontWeight', 'bold');

% Plot first 4 periods for both session types, at 2 Hz
data_2hz = tbl(abs((tbl.Freq_Low_Hz + tbl.Freq_High_Hz)/2 - 2) < 1, :);

for p = 1:4
    % Aversive
    subplot(2, 4, p);
    aversive_data = data_2hz(data_2hz.SessionType == 'Aversive' & data_2hz.Period == p, :);
    if ~isempty(aversive_data)
        plot_fta_cartesian(aversive_data, phase_centers, color_aversive);
        title(sprintf('Aversive P%d (2 Hz, n=%d)', p, height(aversive_data)), 'FontSize', 11, 'FontWeight', 'bold');
        if p == 1, ylabel('Normalized Spike Rate', 'FontSize', 10); end
        xlabel('Phase (rad)', 'FontSize', 10);
    end

    % Reward
    subplot(2, 4, 4 + p);
    reward_data = data_2hz(data_2hz.SessionType == 'Reward' & data_2hz.Period == p, :);
    if ~isempty(reward_data)
        plot_fta_cartesian(reward_data, phase_centers, color_reward);
        title(sprintf('Reward P%d (2 Hz, n=%d)', p, height(reward_data)), 'FontSize', 11, 'FontWeight', 'bold');
        if p == 1, ylabel('Normalized Spike Rate', 'FontSize', 10); end
        xlabel('Phase (rad)', 'FontSize', 10);
    end
end

saveas(fig1, fullfile(output_dir, 'Figure1_FTA_Curves_Cartesian_2Hz_Breathing.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  FIGURE 2: FTA POLAR PLOTS - 2 Hz BREATHING
%  ========================================================================

fprintf('Creating Figure 2: FTA Polar Plots at 2 Hz...\n');

fig2 = figure('Position', [100, 100, 1800, 900]);
sgtitle('Field-Triggered Average: Polar Representation (2 Hz Breathing)', 'FontSize', 16, 'FontWeight', 'bold');

for p = 1:4
    % Aversive
    subplot(2, 4, p, polaraxes);
    aversive_data = data_2hz(data_2hz.SessionType == 'Aversive' & data_2hz.Period == p, :);
    if ~isempty(aversive_data)
        plot_fta_polar(aversive_data, phase_centers, color_aversive);
        title(sprintf('Aversive P%d', p), 'FontSize', 11, 'FontWeight', 'bold');
    end

    % Reward
    subplot(2, 4, 4 + p, polaraxes);
    reward_data = data_2hz(data_2hz.SessionType == 'Reward' & data_2hz.Period == p, :);
    if ~isempty(reward_data)
        plot_fta_polar(reward_data, phase_centers, color_reward);
        title(sprintf('Reward P%d', p), 'FontSize', 11, 'FontWeight', 'bold');
    end
end

saveas(fig2, fullfile(output_dir, 'Figure2_FTA_Polar_Plots_2Hz_Breathing.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  FIGURE 3: MEAN VECTOR LENGTH SPECTRUM
%  ========================================================================

fprintf('Creating Figure 3: Mean Vector Length Spectrum...\n');

fig3 = figure('Position', [150, 150, 1600, 900]);
sgtitle('Mean Vector Length vs Frequency (Phase-Locking Strength to Breathing)', 'FontSize', 16, 'FontWeight', 'bold');

% Get frequency centers
freq_centers = (tbl.Freq_Low_Hz + tbl.Freq_High_Hz) / 2;
unique_freqs = unique(freq_centers);

% Aversive (7 periods)
for p = 1:7
    subplot(2, 4, p);
    aversive_data = tbl(tbl.SessionType == 'Aversive' & tbl.Period == p, :);

    if ~isempty(aversive_data)
        plot_mvl_spectrum(aversive_data, unique_freqs, color_aversive);
        title(sprintf('Aversive P%d', p), 'FontSize', 12, 'FontWeight', 'bold');
        if p == 1, ylabel('Mean Vector Length', 'FontSize', 11); end
        if p >= 5, xlabel('Frequency (Hz)', 'FontSize', 11); end
    end
end

% Reward (4 periods)
subplot(2, 4, 8);
for p = 1:4
    reward_data = tbl(tbl.SessionType == 'Reward' & tbl.Period == p, :);

    if ~isempty(reward_data)
        [mean_mvl, ~, ~] = compute_frequency_average(reward_data, unique_freqs, 'Mean_Vector_Length');
        hold on;
        plot(unique_freqs, mean_mvl, 'o-', 'Color', color_periods(p, :), ...
             'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', color_periods(p, :), ...
             'DisplayName', sprintf('P%d', p));
    end
end

% Highlight breathing-relevant frequencies
plot([2 2], ylim, 'r--', 'LineWidth', 2, 'HandleVisibility', 'off');
text(2, max(ylim)*0.95, ' 2 Hz (breathing)', 'FontSize', 10, 'Color', 'r', 'FontWeight', 'bold');
plot([8 8], ylim, 'b--', 'LineWidth', 2, 'HandleVisibility', 'off');
text(8, max(ylim)*0.85, ' 8 Hz (sniffing)', 'FontSize', 10, 'Color', 'b', 'FontWeight', 'bold');

xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Mean Vector Length', 'FontSize', 11);
title('Reward (All Periods)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on; box on;
xlim([0 20]);

saveas(fig3, fullfile(output_dir, 'Figure3_Mean_Vector_Length_Spectrum_Breathing.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  FIGURE 4: 2 Hz DETAILED ANALYSIS (BREATHING)
%  ========================================================================

fprintf('Creating Figure 4: 2 Hz Detailed Analysis (Breathing)...\n');

fig4 = figure('Position', [200, 200, 1800, 1000]);
sgtitle('2 Hz Band (Breathing): Detailed Phase-Locking Analysis', 'FontSize', 16, 'FontWeight', 'bold');

% Panel 1: Average FTA curves
subplot(2, 3, 1);
plot_frequency_average_fta(data_2hz, phase_centers, color_aversive, color_reward, '2 Hz');

% Panel 2: Mean vector length comparison
subplot(2, 3, 2);
plot_frequency_mvl_comparison(data_2hz, color_aversive, color_reward, '2 Hz');

% Panel 3: Preferred phase distribution (aversive)
subplot(2, 3, 3);
plot_preferred_phase_distribution(data_2hz(data_2hz.SessionType == 'Aversive', :), color_aversive, 'Aversive (2 Hz)');

% Panel 4: Preferred phase distribution (reward)
subplot(2, 3, 4);
plot_preferred_phase_distribution(data_2hz(data_2hz.SessionType == 'Reward', :), color_reward, 'Reward (2 Hz)');

% Panel 5: MVL histogram (aversive)
subplot(2, 3, 5);
histogram(data_2hz.Mean_Vector_Length(data_2hz.SessionType == 'Aversive'), 20, ...
          'FaceColor', color_aversive, 'EdgeColor', 'k', 'FaceAlpha', 0.7);
xlabel('Mean Vector Length', 'FontSize', 11);
ylabel('Count', 'FontSize', 11);
title('Aversive: MVL Distribution', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;

% Panel 6: MVL histogram (reward)
subplot(2, 3, 6);
histogram(data_2hz.Mean_Vector_Length(data_2hz.SessionType == 'Reward'), 20, ...
          'FaceColor', color_reward, 'EdgeColor', 'k', 'FaceAlpha', 0.7);
xlabel('Mean Vector Length', 'FontSize', 11);
ylabel('Count', 'FontSize', 11);
title('Reward: MVL Distribution', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;

saveas(fig4, fullfile(output_dir, 'Figure4_2Hz_Breathing_Detailed_Analysis.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  FIGURE 5: 8 Hz DETAILED ANALYSIS (SNIFFING)
%  ========================================================================

fprintf('Creating Figure 5: 8 Hz Detailed Analysis (Sniffing)...\n');

fig5 = figure('Position', [250, 250, 1800, 1000]);
sgtitle('8 Hz Band (Sniffing): Detailed Phase-Locking Analysis', 'FontSize', 16, 'FontWeight', 'bold');

% Get 8 Hz data
data_8hz = tbl(abs((tbl.Freq_Low_Hz + tbl.Freq_High_Hz)/2 - 8) < 1, :);

% Panel 1: Average FTA curves
subplot(2, 3, 1);
plot_frequency_average_fta(data_8hz, phase_centers, color_aversive, color_reward, '8 Hz');

% Panel 2: Mean vector length comparison
subplot(2, 3, 2);
plot_frequency_mvl_comparison(data_8hz, color_aversive, color_reward, '8 Hz');

% Panel 3: Preferred phase distribution (aversive)
subplot(2, 3, 3);
plot_preferred_phase_distribution(data_8hz(data_8hz.SessionType == 'Aversive', :), color_aversive, 'Aversive (8 Hz)');

% Panel 4: Preferred phase distribution (reward)
subplot(2, 3, 4);
plot_preferred_phase_distribution(data_8hz(data_8hz.SessionType == 'Reward', :), color_reward, 'Reward (8 Hz)');

% Panel 5: MVL histogram (aversive)
subplot(2, 3, 5);
histogram(data_8hz.Mean_Vector_Length(data_8hz.SessionType == 'Aversive'), 20, ...
          'FaceColor', color_aversive, 'EdgeColor', 'k', 'FaceAlpha', 0.7);
xlabel('Mean Vector Length', 'FontSize', 11);
ylabel('Count', 'FontSize', 11);
title('Aversive: MVL Distribution', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;

% Panel 6: MVL histogram (reward)
subplot(2, 3, 6);
histogram(data_8hz.Mean_Vector_Length(data_8hz.SessionType == 'Reward'), 20, ...
          'FaceColor', color_reward, 'EdgeColor', 'k', 'FaceAlpha', 0.7);
xlabel('Mean Vector Length', 'FontSize', 11);
ylabel('Count', 'FontSize', 11);
title('Reward: MVL Distribution', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;

saveas(fig5, fullfile(output_dir, 'Figure5_8Hz_Sniffing_Detailed_Analysis.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  FIGURE 5: UNIT × FREQUENCY HEATMAP (All Units, All Frequencies)
%  ========================================================================

fprintf('Creating Figure 5: Unit × Frequency Heatmap...\n');

% CONFIGURABLE: Select which periods to include in heatmap (default: 1-4)
periods_to_include = 1:4;

fig5 = figure('Position', [250, 250, 1800, 900]);
if length(periods_to_include) == 1
    sgtitle(sprintf('Mean Vector Length: Units × Frequencies (Period %d)', periods_to_include), 'FontSize', 16, 'FontWeight', 'bold');
else
    sgtitle(sprintf('Mean Vector Length: Units × Frequencies (Periods %d-%d avg)', min(periods_to_include), max(periods_to_include)), 'FontSize', 16, 'FontWeight', 'bold');
end

% Aversive (average across selected periods)
subplot(1, 2, 1);
aversive_selected = tbl(tbl.SessionType == 'Aversive' & ismember(tbl.Period, periods_to_include), :);
if ~isempty(aversive_selected)
    heatmap_data_av = createFTAHeatmap(aversive_selected, unique_freqs);
    imagesc(unique_freqs, 1:size(heatmap_data_av, 1), heatmap_data_av, [0,0.1]);
    set(gca, 'YDir', 'normal');
    colorbar;
    colormap(jet);
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Unit', 'FontSize', 12, 'FontWeight', 'bold');
    if length(periods_to_include) == 1
        title(sprintf('Aversive P%d (n=%d units)', periods_to_include, size(heatmap_data_av, 1)), 'FontSize', 13, 'FontWeight', 'bold');
    else
        title(sprintf('Aversive P%d-%d avg (n=%d units)', min(periods_to_include), max(periods_to_include), size(heatmap_data_av, 1)), 'FontSize', 13, 'FontWeight', 'bold');
    end
    xlim([0 20]);
    hold on;
    plot([8 8], ylim, 'k--', 'LineWidth', 2);
    hold off;
end

% Reward (average across selected periods)
subplot(1, 2, 2);
reward_selected = tbl(tbl.SessionType == 'Reward' & ismember(tbl.Period, periods_to_include), :);
if ~isempty(reward_selected)
    heatmap_data_rw = createFTAHeatmap(reward_selected, unique_freqs);
    imagesc(unique_freqs, 1:size(heatmap_data_rw, 1), heatmap_data_rw, [0,0.1]);
    set(gca, 'YDir', 'normal');
    colorbar;
    colormap(jet);
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Unit', 'FontSize', 12, 'FontWeight', 'bold');
    if length(periods_to_include) == 1
        title(sprintf('Reward P%d (n=%d units)', periods_to_include, size(heatmap_data_rw, 1)), 'FontSize', 13, 'FontWeight', 'bold');
    else
        title(sprintf('Reward P%d-%d avg (n=%d units)', min(periods_to_include), max(periods_to_include), size(heatmap_data_rw, 1)), 'FontSize', 13, 'FontWeight', 'bold');
    end
    xlim([0 20]);
    hold on;
    plot([8 8], ylim, 'k--', 'LineWidth', 2);
    hold off;
end

saveas(fig5, fullfile(output_dir, 'Figure5_Unit_Frequency_MVL_Heatmap.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  COMPLETION
%  ========================================================================

fprintf('\n========================================\n');
fprintf('FIELD-TRIGGERED AVERAGE (BREATHING) VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Figures saved to: %s/\n', output_dir);
fprintf('\nInterpretation (Breathing Signal - Channel 32):\n');
fprintf('  - Peaked FTA curve → Strong phase-locking (high MVL)\n');
fprintf('  - Flat FTA curve → No phase preference (low MVL)\n');
fprintf('  - MVL spectrum similar to PPC (both measure phase-locking)\n');
fprintf('  - 2 Hz MVL peak → Coupling to normal breathing rhythm\n');
fprintf('  - 8 Hz MVL peak → Coupling to sniffing/exploratory breathing\n');
fprintf('  - Compare with LFP-FTA to identify breathing-specific phase-locking\n');
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function plot_fta_cartesian(data, phase_centers, color)
% Plot FTA curve in Cartesian coordinates

    % Average FTA curves across all entries
    all_curves = cell2mat(data.FTA_Curve);
    mean_fta = mean(all_curves, 1);
    sem_fta = std(all_curves, 0, 1) / sqrt(size(all_curves, 1));

    hold on;
    fill([phase_centers, fliplr(phase_centers)], ...
         [mean_fta - sem_fta, fliplr(mean_fta + sem_fta)], ...
         color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    plot(phase_centers, mean_fta, 'Color', color, 'LineWidth', 2);

    % Uniform expectation
    uniform_level = 1 / length(phase_centers);
    plot(xlim, [uniform_level uniform_level], 'k--', 'LineWidth', 1);

    xlabel('Phase (rad)', 'FontSize', 11);
    ylabel('Normalized Spike Rate', 'FontSize', 11);
    grid on; box on;
    xlim([-pi pi]);
    set(gca, 'XTick', [-pi -pi/2 0 pi/2 pi], 'XTickLabel', {'-π', '-π/2', '0', 'π/2', 'π'});
end

function plot_fta_polar(data, phase_centers, color)
% Plot FTA curve in polar coordinates

    % Average FTA curves
    all_curves = cell2mat(data.FTA_Curve);
    mean_fta = mean(all_curves, 1);

    % Plot in polar coordinates
    hold on;
    polarplot([phase_centers, phase_centers(1)], [mean_fta, mean_fta(1)], ...
              'Color', color, 'LineWidth', 2);

    % Uniform circle
    uniform_level = 1 / length(phase_centers);
    theta_circle = linspace(0, 2*pi, 100);
    polarplot(theta_circle, uniform_level * ones(size(theta_circle)), 'k--', 'LineWidth', 1);

    thetalim([-180 180]);
end

function plot_mvl_spectrum(data, unique_freqs, color)
% Plot mean vector length spectrum

    [mean_mvl, sem_mvl, freq_centers] = compute_frequency_average(data, unique_freqs, 'Mean_Vector_Length');

    hold on;
    errorbar(freq_centers, mean_mvl, sem_mvl, 'o-', 'Color', color, ...
             'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', color, 'CapSize', 4);

    % Highlight breathing frequencies
    plot([2 2], ylim, 'r--', 'LineWidth', 2);
    text(2, max(ylim)*0.95, ' 2 Hz', 'FontSize', 9, 'Color', 'r', 'FontWeight', 'bold');
    plot([8 8], ylim, 'b--', 'LineWidth', 1.5);
    text(8, max(ylim)*0.85, ' 8 Hz', 'FontSize', 9, 'Color', 'b', 'FontWeight', 'bold');

    xlabel('Frequency (Hz)', 'FontSize', 11);
    ylabel('Mean Vector Length', 'FontSize', 11);
    grid on; box on;
    xlim([0 20]);
    ylim([0 max(mean_mvl)*1.3]);
end

function [mean_vals, sem_vals, freq_centers] = compute_frequency_average(data, unique_freqs, variable_name)
% Compute average and SEM across frequencies

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

function plot_frequency_average_fta(data_freq, phase_centers, color_aversive, color_reward, freq_label)
% Plot average FTA curves for aversive vs reward at specified frequency

    hold on;

    % Aversive
    aversive_data = data_freq(data_freq.SessionType == 'Aversive', :);
    if ~isempty(aversive_data)
        all_curves = cell2mat(aversive_data.FTA_Curve);
        mean_fta = mean(all_curves, 1);
        sem_fta = std(all_curves, 0, 1) / sqrt(size(all_curves, 1));

        fill([phase_centers, fliplr(phase_centers)], ...
             [mean_fta - sem_fta, fliplr(mean_fta + sem_fta)], ...
             color_aversive, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(phase_centers, mean_fta, 'Color', color_aversive, 'LineWidth', 2, 'DisplayName', 'Aversive');
    end

    % Reward
    reward_data = data_freq(data_freq.SessionType == 'Reward', :);
    if ~isempty(reward_data)
        all_curves = cell2mat(reward_data.FTA_Curve);
        mean_fta = mean(all_curves, 1);
        sem_fta = std(all_curves, 0, 1) / sqrt(size(all_curves, 1));

        fill([phase_centers, fliplr(phase_centers)], ...
             [mean_fta - sem_fta, fliplr(mean_fta + sem_fta)], ...
             color_reward, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(phase_centers, mean_fta, 'Color', color_reward, 'LineWidth', 2, 'DisplayName', 'Reward');
    end

    % Uniform expectation
    uniform_level = 1 / length(phase_centers);
    plot(xlim, [uniform_level uniform_level], 'k--', 'LineWidth', 1, 'DisplayName', 'Uniform');

    xlabel('Phase (rad)', 'FontSize', 12);
    ylabel('Normalized Spike Rate', 'FontSize', 12);
    title(sprintf('Average FTA at %s', freq_label), 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    grid on; box on;
    xlim([-pi pi]);
    set(gca, 'XTick', [-pi -pi/2 0 pi/2 pi], 'XTickLabel', {'-π', '-π/2', '0', 'π/2', 'π'});
end

function plot_frequency_mvl_comparison(data_freq, color_aversive, color_reward, freq_label)
% Bar plot comparing MVL at specified frequency

    % Compute averages
    aversive_mvl = data_freq.Mean_Vector_Length(data_freq.SessionType == 'Aversive');
    reward_mvl = data_freq.Mean_Vector_Length(data_freq.SessionType == 'Reward');

    mean_aversive = mean(aversive_mvl, 'omitnan');
    mean_reward = mean(reward_mvl, 'omitnan');

    sem_aversive = std(aversive_mvl, 'omitnan') / sqrt(sum(~isnan(aversive_mvl)));
    sem_reward = std(reward_mvl, 'omitnan') / sqrt(sum(~isnan(reward_mvl)));

    % Bar plot
    hold on;
    bar(1, mean_aversive, 'FaceColor', color_aversive, 'EdgeColor', 'k', 'LineWidth', 1.5);
    bar(2, mean_reward, 'FaceColor', color_reward, 'EdgeColor', 'k', 'LineWidth', 1.5);

    errorbar([1 2], [mean_aversive mean_reward], [sem_aversive sem_reward], ...
             'k', 'LineStyle', 'none', 'LineWidth', 2, 'CapSize', 10);

    set(gca, 'XTick', [1 2], 'XTickLabel', {'Aversive', 'Reward'});
    ylabel('Mean Vector Length', 'FontSize', 12);
    title(sprintf('%s: Phase-Locking Strength', freq_label), 'FontSize', 12, 'FontWeight', 'bold');
    grid on; box on;
    xlim([0.5 2.5]);
    ylim([0 max([mean_aversive, mean_reward]) * 1.3]);
end

function plot_preferred_phase_distribution(data, color, session_label)
% Circular histogram of preferred phases

    preferred_phases = data.Preferred_Phase;

    % Create circular histogram
    polarhistogram(preferred_phases, 18, 'FaceColor', color, 'EdgeColor', 'k', 'FaceAlpha', 0.7);

    title(sprintf('%s: Preferred Phase', session_label), 'FontSize', 12, 'FontWeight', 'bold');
    thetalim([-180 180]);
end


function heatmap_data = createFTAHeatmap(data, unique_freqs)
% Create heatmap matrix: units × frequencies (Mean Vector Length)
%
% INPUTS:
%   data         - Table with FTA data for one period
%   unique_freqs - Vector of unique frequency values
%
% OUTPUTS:
%   heatmap_data - Matrix (n_units × n_freqs) of MVL values

    % FIX: Use UniqueUnitID to prevent collision across sessions
    unique_units = unique(data.UniqueUnitID);
    n_units = length(unique_units);
    n_freqs = length(unique_freqs);

    heatmap_data = nan(n_units, n_freqs);

    for u = 1:n_units
        % FIX: Filter by UniqueUnitID instead of Unit
        unit_data = data(data.UniqueUnitID == unique_units(u), :);

        for f = 1:n_freqs
            freq = unique_freqs(f);
            freq_center = (unit_data.Freq_Low_Hz + unit_data.Freq_High_Hz) / 2;
            freq_data = unit_data(abs(freq_center - freq) < 0.1, :);

            if ~isempty(freq_data)
                heatmap_data(u, f) = mean(freq_data.Mean_Vector_Length, 'omitnan');
            end
        end
    end
end