%% ========================================================================
%  VISUALIZE UNIT PPC ANALYSIS
%  Comprehensive visualization of PPC spike-LFP coupling across frequencies
%  ========================================================================
%
%  Visualizes results from Unit_PPC_Analysis.m
%
%  Creates visualizations:
%  Figure 1: PPC Spectrograms (Frequency × Period)
%  Figure 2: Frequency Band Summaries
%  Figure 3: Preferred Phase Analysis
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: LOAD DATA
%  ========================================================================

fprintf('=== VISUALIZING UNIT PPC ANALYSIS ===\n\n');

% Data paths
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_UnitPPC');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_UnitPPC');

% Load aversive session files
fprintf('Loading aversive session files...\n');
aversive_files = dir(fullfile(RewardAversivePath, '*_unit_ppc.mat'));
n_aversive = length(aversive_files);
fprintf('  Found %d aversive session files\n', n_aversive);

% Load reward session files
fprintf('Loading reward session files...\n');
reward_files = dir(fullfile(RewardSeekingPath, '*_unit_ppc.mat'));
n_reward = length(reward_files);
fprintf('  Found %d reward session files\n', n_reward);

% Combine all data
fprintf('\nCombining data from all sessions...\n');
all_data_aversive = [];
all_data_reward = [];

for i = 1:n_aversive
    load(fullfile(RewardAversivePath, aversive_files(i).name), 'session_results');
    if ~isempty(session_results.data)
        session_results.data.SessionType = repmat({'Aversive'}, height(session_results.data), 1);
        all_data_aversive = [all_data_aversive; session_results.data];
    end
end

for i = 1:n_reward
    load(fullfile(RewardSeekingPath, reward_files(i).name), 'session_results', 'config');
    if ~isempty(session_results.data)
        session_results.data.SessionType = repmat({'Reward'}, height(session_results.data), 1);
        all_data_reward = [all_data_reward; session_results.data];
    end
end

% Combine
tbl = [all_data_aversive; all_data_reward];
tbl.SessionType = categorical(tbl.SessionType);

fprintf('✓ Data combined\n');
fprintf('  Total data points: %d\n', height(tbl));
fprintf('  Aversive data points: %d\n', height(all_data_aversive));
fprintf('  Reward data points: %d\n\n', height(all_data_reward));

%% ========================================================================
%  SECTION 2: SETUP VISUALIZATION PARAMETERS
%  ========================================================================

% Colors
color_aversive = [0.8 0.2 0.2];  % Red
color_reward = [0.2 0.4 0.8];    % Blue

% Create output directory
output_dir = 'Unit_PPC_Figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('Figures will be saved to: %s/\n\n', output_dir);

% Get unique frequency bands
unique_freqs = unique(tbl.Freq_Low_Hz);
n_freqs = length(unique_freqs);

fprintf('Frequency resolution:\n');
fprintf('  Number of frequency bands: %d\n', n_freqs);
fprintf('  Frequency range: %.1f - %.1f Hz\n\n', min(tbl.Freq_Low_Hz), max(tbl.Freq_High_Hz));

%% ========================================================================
%  FIGURE 1: PPC SPECTROGRAMS (Frequency × Period)
%  ========================================================================

fprintf('Creating Figure 1: PPC Spectrograms...\n');

fig1 = figure('Position', [50, 50, 1600, 800]);
sgtitle('PPC Spectrograms: Frequency × Period', 'FontSize', 16, 'FontWeight', 'bold');

% Aversive spectrogram (7 periods)
subplot(1, 2, 1);
aversive_data = tbl(tbl.SessionType == 'Aversive', :);
spectrogram_aversive = createPPCSpectrogram(aversive_data, unique_freqs, 7);
imagesc(1:7, unique_freqs, spectrogram_aversive);
set(gca, 'YDir', 'normal');
colorbar;
caxis([0, 0.3]);  % PPC typically ranges 0-0.3 for significant coupling
xlabel('Period', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
title('Aversive Sessions (7 periods)', 'FontSize', 13, 'FontWeight', 'bold');
colormap(jet);
set(gca, 'XTick', 1:7);

% Reward spectrogram (4 periods)
subplot(1, 2, 2);
reward_data = tbl(tbl.SessionType == 'Reward', :);
spectrogram_reward = createPPCSpectrogram(reward_data, unique_freqs, 4);
imagesc(1:4, unique_freqs, spectrogram_reward);
set(gca, 'YDir', 'normal');
colorbar;
caxis([0, 0.3]);
xlabel('Period', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
title('Reward Sessions (4 periods)', 'FontSize', 13, 'FontWeight', 'bold');
colormap(jet);
set(gca, 'XTick', 1:4);

saveas(fig1, fullfile(output_dir, 'Figure1_PPC_Spectrograms.png'));
fprintf('  ✓ Saved: Figure1_PPC_Spectrograms.png\n');

%% ========================================================================
%  FIGURE 2: FREQUENCY BAND SUMMARIES
%  ========================================================================

fprintf('Creating Figure 2: Frequency Band Summaries...\n');

fig2 = figure('Position', [100, 100, 1600, 1000]);
sgtitle('Frequency Band PPC Summary', 'FontSize', 16, 'FontWeight', 'bold');

% Define frequency bands of interest
freq_bands = {
    'Delta', [1, 4];
    'Theta', [5, 12];
    'Beta', [15, 30];
    'Low Gamma', [30, 60];
    'High Gamma', [80, 100]
};

n_bands = size(freq_bands, 1);

for b = 1:n_bands
    subplot(2, 3, b);

    band_name = freq_bands{b, 1};
    band_range = freq_bands{b, 2};

    % Filter data for this frequency band
    band_mask = tbl.Freq_Low_Hz >= band_range(1) & tbl.Freq_High_Hz <= band_range(2);
    band_data = tbl(band_mask, :);

    if isempty(band_data)
        continue;
    end

    % Plot Aversive
    aversive_band = band_data(band_data.SessionType == 'Aversive', :);
    periods_av = 1:7;
    mean_ppc_av = zeros(1, 7);
    sem_ppc_av = zeros(1, 7);

    for p = 1:7
        period_data = aversive_band(aversive_band.Period == p, :);
        if ~isempty(period_data)
            mean_ppc_av(p) = mean(period_data.PPC, 'omitnan');
            sem_ppc_av(p) = std(period_data.PPC, 'omitnan') / sqrt(sum(~isnan(period_data.PPC)));
        end
    end

    % Plot Reward
    reward_band = band_data(band_data.SessionType == 'Reward', :);
    periods_rw = 1:4;
    mean_ppc_rw = zeros(1, 4);
    sem_ppc_rw = zeros(1, 4);

    for p = 1:4
        period_data = reward_band(reward_band.Period == p, :);
        if ~isempty(period_data)
            mean_ppc_rw(p) = mean(period_data.PPC, 'omitnan');
            sem_ppc_rw(p) = std(period_data.PPC, 'omitnan') / sqrt(sum(~isnan(period_data.PPC)));
        end
    end

    hold on;
    errorbar(periods_av, mean_ppc_av, sem_ppc_av, '-o', ...
             'Color', color_aversive, 'LineWidth', 2, 'MarkerSize', 8, ...
             'MarkerFaceColor', color_aversive, 'DisplayName', 'Aversive');
    errorbar(periods_rw, mean_ppc_rw, sem_ppc_rw, '-s', ...
             'Color', color_reward, 'LineWidth', 2, 'MarkerSize', 8, ...
             'MarkerFaceColor', color_reward, 'DisplayName', 'Reward');

    xlabel('Period', 'FontSize', 11);
    ylabel('Mean PPC', 'FontSize', 11);
    title(sprintf('%s (%.0f-%.0f Hz)', band_name, band_range(1), band_range(2)), ...
          'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    box on;
    xlim([0.5, 7.5]);
    set(gca, 'XTick', 1:7);
    hold off;
end

% Overall summary (all frequencies)
subplot(2, 3, 6);
plotOverallPPCSummary(tbl, color_aversive, color_reward);

saveas(fig2, fullfile(output_dir, 'Figure2_Frequency_Band_Summaries.png'));
fprintf('  ✓ Saved: Figure2_Frequency_Band_Summaries.png\n');

%% ========================================================================
%  FIGURE 3: PREFERRED PHASE ANALYSIS
%  ========================================================================

fprintf('Creating Figure 3: Preferred Phase Analysis...\n');

fig3 = figure('Position', [150, 150, 1600, 900]);
sgtitle('Preferred Phase Analysis (Theta Band 5-12 Hz)', 'FontSize', 16, 'FontWeight', 'bold');

% Focus on theta band for phase analysis
theta_mask = tbl.Freq_Low_Hz >= 5 & tbl.Freq_High_Hz <= 12;
theta_data = tbl(theta_mask, :);

% Aversive phase distributions
for p = 1:4  % Show first 4 periods for comparison
    subplot(2, 4, p);
    aversive_theta = theta_data(theta_data.SessionType == 'Aversive' & theta_data.Period == p, :);
    if ~isempty(aversive_theta)
        phases = aversive_theta.Preferred_Phase_rad(~isnan(aversive_theta.Preferred_Phase_rad));
        polarhistogram(phases, 18, 'FaceColor', color_aversive, 'FaceAlpha', 0.7);
        title(sprintf('Aversive P%d', p), 'FontWeight', 'bold');
    end
end

% Reward phase distributions
for p = 1:4
    subplot(2, 4, 4 + p);
    reward_theta = theta_data(theta_data.SessionType == 'Reward' & theta_data.Period == p, :);
    if ~isempty(reward_theta)
        phases = reward_theta.Preferred_Phase_rad(~isnan(reward_theta.Preferred_Phase_rad));
        polarhistogram(phases, 18, 'FaceColor', color_reward, 'FaceAlpha', 0.7);
        title(sprintf('Reward P%d', p), 'FontWeight', 'bold');
    end
end

saveas(fig3, fullfile(output_dir, 'Figure3_Preferred_Phase_Analysis.png'));
fprintf('  ✓ Saved: Figure3_Preferred_Phase_Analysis.png\n');

%% ========================================================================
%  SECTION 3: SUMMARY STATISTICS
%  ========================================================================

fprintf('\nGenerating summary statistics...\n');

summary_file = fullfile(output_dir, 'PPC_Summary_Statistics.txt');
fid = fopen(summary_file, 'w');

fprintf(fid, '========================================\n');
fprintf(fid, 'UNIT PPC ANALYSIS - SUMMARY STATISTICS\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, 'Generated: %s\n\n', datestr(now));

fprintf(fid, 'Dataset Overview:\n');
fprintf(fid, '  Total data points: %d\n', height(tbl));
fprintf(fid, '  Aversive sessions: %d\n', n_aversive);
fprintf(fid, '  Reward sessions: %d\n', n_reward);
fprintf(fid, '  Frequency bands: %d (1 Hz resolution)\n', n_freqs);
fprintf(fid, '  Frequency range: %.1f - %.1f Hz\n\n', min(tbl.Freq_Low_Hz), max(tbl.Freq_High_Hz));

% PPC statistics by SessionType
for session_type = categorical({'Aversive', 'Reward'})
    fprintf(fid, '--- %s Sessions ---\n', char(session_type));
    subset = tbl(tbl.SessionType == session_type, :);

    fprintf(fid, 'Data points: %d\n', height(subset));
    fprintf(fid, 'Mean PPC: %.4f ± %.4f\n', mean(subset.PPC, 'omitnan'), std(subset.PPC, 'omitnan'));
    fprintf(fid, 'Median PPC: %.4f\n', median(subset.PPC, 'omitnan'));
    fprintf(fid, 'Max PPC: %.4f\n', max(subset.PPC));

    % PPC by frequency band
    fprintf(fid, '\nPPC by Frequency Band:\n');
    for b = 1:size(freq_bands, 1)
        band_name = freq_bands{b, 1};
        band_range = freq_bands{b, 2};
        band_mask = subset.Freq_Low_Hz >= band_range(1) & subset.Freq_High_Hz <= band_range(2);
        band_subset = subset(band_mask, :);

        if ~isempty(band_subset)
            fprintf(fid, '  %s (%.0f-%.0f Hz): %.4f ± %.4f\n', ...
                    band_name, band_range(1), band_range(2), ...
                    mean(band_subset.PPC, 'omitnan'), std(band_subset.PPC, 'omitnan'));
        end
    end
    fprintf(fid, '\n');
end

fclose(fid);
fprintf('  ✓ Saved: PPC_Summary_Statistics.txt\n');

%% ========================================================================
%  COMPLETION MESSAGE
%  ========================================================================

fprintf('\n========================================\n');
fprintf('VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('All figures saved to: %s/\n', output_dir);
fprintf('Total figures generated: 3\n');
fprintf('  - Figure 1: PPC Spectrograms\n');
fprintf('  - Figure 2: Frequency Band Summaries\n');
fprintf('  - Figure 3: Preferred Phase Analysis\n');
fprintf('Summary statistics: %s\n', summary_file);
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function spectrogram = createPPCSpectrogram(data, unique_freqs, n_periods)
% Create PPC spectrogram matrix: frequency × period
%
% INPUTS:
%   data         - Table with PPC data
%   unique_freqs - Vector of unique frequency values
%   n_periods    - Number of periods (7 for aversive, 4 for reward)
%
% OUTPUTS:
%   spectrogram  - Matrix (n_freqs × n_periods) of mean PPC values

    n_freqs = length(unique_freqs);
    spectrogram = nan(n_freqs, n_periods);

    for f = 1:n_freqs
        freq = unique_freqs(f);

        for p = 1:n_periods
            % Find data for this frequency and period
            mask = (data.Freq_Low_Hz == freq) & (data.Period == p);
            period_freq_data = data(mask, :);

            if ~isempty(period_freq_data)
                % Average PPC across all units for this freq-period combination
                spectrogram(f, p) = mean(period_freq_data.PPC, 'omitnan');
            end
        end
    end
end

function plotOverallPPCSummary(tbl, color_aversive, color_reward)
% Plot overall PPC summary across all frequencies
%
% INPUTS:
%   tbl              - Combined data table
%   color_aversive   - Color for aversive
%   color_reward     - Color for reward

    % Aversive
    aversive_data = tbl(tbl.SessionType == 'Aversive', :);
    periods_av = 1:7;
    mean_ppc_av = zeros(1, 7);
    sem_ppc_av = zeros(1, 7);

    for p = 1:7
        period_data = aversive_data(aversive_data.Period == p, :);
        if ~isempty(period_data)
            mean_ppc_av(p) = mean(period_data.PPC, 'omitnan');
            sem_ppc_av(p) = std(period_data.PPC, 'omitnan') / sqrt(sum(~isnan(period_data.PPC)));
        end
    end

    % Reward
    reward_data = tbl(tbl.SessionType == 'Reward', :);
    periods_rw = 1:4;
    mean_ppc_rw = zeros(1, 4);
    sem_ppc_rw = zeros(1, 4);

    for p = 1:4
        period_data = reward_data(reward_data.Period == p, :);
        if ~isempty(period_data)
            mean_ppc_rw(p) = mean(period_data.PPC, 'omitnan');
            sem_ppc_rw(p) = std(period_data.PPC, 'omitnan') / sqrt(sum(~isnan(period_data.PPC)));
        end
    end

    hold on;
    errorbar(periods_av, mean_ppc_av, sem_ppc_av, '-o', ...
             'Color', color_aversive, 'LineWidth', 2, 'MarkerSize', 8, ...
             'MarkerFaceColor', color_aversive, 'DisplayName', 'Aversive');
    errorbar(periods_rw, mean_ppc_rw, sem_ppc_rw, '-s', ...
             'Color', color_reward, 'LineWidth', 2, 'MarkerSize', 8, ...
             'MarkerFaceColor', color_reward, 'DisplayName', 'Reward');

    xlabel('Period', 'FontSize', 11);
    ylabel('Mean PPC', 'FontSize', 11);
    title('Overall (All Frequencies)', 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    box on;
    xlim([0.5, 7.5]);
    set(gca, 'XTick', 1:7);
    hold off;
end
