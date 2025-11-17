%% ========================================================================
%  COMPREHENSIVE SUMMARY: SPIKE-BREATHING COHERENCE
%  Aggregates across all sessions for population-level analysis
%  ========================================================================
%
%  This script loads all session results and creates comprehensive summaries:
%  1. Population-level coherence spectra (averaged across all units)
%  2. Band-specific coherence comparison: Aversive vs Reward
%  3. Unit-level coherence distributions
%  4. Spike count vs coherence relationship
%  5. Breathing-specific frequency analysis (1-4 Hz)
%
%% ========================================================================

clear all;
% close all;

%% ========================================================================
%  SECTION 1: LOAD ALL SESSION DATA
%  ========================================================================

fprintf('=== COMPREHENSIVE SUMMARY: SPIKE-BREATHING COHERENCE ===\n\n');

% Define paths
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_SpikeBreathingCoherence_Overall');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_SpikeBreathingCoherence_Overall');

fprintf('Loading all session data...\n');

% Load aversive sessions
aversive_files = dir(fullfile(RewardAversivePath, '*_spike_breathing_coherence_overall.mat'));
fprintf('  Found %d aversive sessions\n', length(aversive_files));

aversive_sessions = cell(length(aversive_files), 1);
for i = 1:length(aversive_files)
    data = load(fullfile(RewardAversivePath, aversive_files(i).name));
    aversive_sessions{i} = data.session_results;
    aversive_sessions{i}.config = data.config;
end

% Load reward sessions
reward_files = dir(fullfile(RewardSeekingPath, '*_spike_breathing_coherence_overall.mat'));
fprintf('  Found %d reward sessions\n', length(reward_files));

reward_sessions = cell(length(reward_files), 1);
for i = 1:length(reward_files)
    data = load(fullfile(RewardSeekingPath, reward_files(i).name));
    reward_sessions{i} = data.session_results;
    reward_sessions{i}.config = data.config;
end

if isempty(aversive_sessions) && isempty(reward_sessions)
    error('No session data found!');
end

fprintf('\n✓ Loaded %d aversive + %d reward sessions\n\n', ...
    length(aversive_sessions), length(reward_sessions));

%% ========================================================================
%  SECTION 2: AGGREGATE DATA
%  ========================================================================

fprintf('Aggregating data across all sessions...\n');

band_names = {'Delta', 'Theta', 'Beta', 'Low_Gamma', 'High_Gamma', 'Ultra_Gamma'};
n_bands = length(band_names);

% Storage for all units
all_data = struct();
all_data.session_type = {};
all_data.session_id = [];
all_data.unit_id = [];
all_data.coherence_spectrum = {};
all_data.freq = {};
all_data.n_spikes = [];
all_data.mean_coherence = [];
all_data.band_coherence = [];  % n_units × n_bands

% Process aversive sessions
for sess_idx = 1:length(aversive_sessions)
    session = aversive_sessions{sess_idx};

    for unit_idx = 1:length(session.unit_coherence_results)
        unit_result = session.unit_coherence_results{unit_idx};

        if isempty(unit_result) || unit_result.skipped
            continue;
        end

        all_data.session_type{end+1} = 'Aversive';
        all_data.session_id(end+1) = sess_idx;
        all_data.unit_id(end+1) = unit_idx;
        all_data.coherence_spectrum{end+1} = unit_result.coherence;
        all_data.freq{end+1} = unit_result.freq;
        all_data.n_spikes(end+1) = unit_result.n_spikes;
        all_data.mean_coherence(end+1) = nanmean(unit_result.coherence);

        % Extract band-specific coherence
        band_vals = zeros(1, n_bands);
        for b = 1:n_bands
            band_vals(b) = unit_result.band_mean_coherence.(band_names{b});
        end
        all_data.band_coherence(end+1, :) = band_vals;
    end
end

% Process reward sessions
for sess_idx = 1:length(reward_sessions)
    session = reward_sessions{sess_idx};

    for unit_idx = 1:length(session.unit_coherence_results)
        unit_result = session.unit_coherence_results{unit_idx};

        if isempty(unit_result) || unit_result.skipped
            continue;
        end

        all_data.session_type{end+1} = 'Reward';
        all_data.session_id(end+1) = sess_idx;
        all_data.unit_id(end+1) = unit_idx;
        all_data.coherence_spectrum{end+1} = unit_result.coherence;
        all_data.freq{end+1} = unit_result.freq;
        all_data.n_spikes(end+1) = unit_result.n_spikes;
        all_data.mean_coherence(end+1) = nanmean(unit_result.coherence);

        band_vals = zeros(1, n_bands);
        for b = 1:n_bands
            band_vals(b) = unit_result.band_mean_coherence.(band_names{b});
        end
        all_data.band_coherence(end+1, :) = band_vals;
    end
end

n_total_units = length(all_data.mean_coherence);
fprintf('✓ Aggregated %d units total\n\n', n_total_units);

%% ========================================================================
%  SECTION 3: COMPUTE POPULATION STATISTICS
%  ========================================================================

fprintf('Computing population statistics...\n');

% Separate by session type
aver_mask = strcmp(all_data.session_type, 'Aversive');
rew_mask = strcmp(all_data.session_type, 'Reward');

% Get frequency vector (from first unit)
freq = all_data.freq{1};
n_freq = length(freq);

% Compute mean coherence spectra
coherence_matrix_aver = nan(sum(aver_mask), n_freq);
coherence_matrix_rew = nan(sum(rew_mask), n_freq);

aver_idx = 0;
rew_idx = 0;
for i = 1:n_total_units
    if aver_mask(i)
        aver_idx = aver_idx + 1;
        coherence_matrix_aver(aver_idx, :) = all_data.coherence_spectrum{i}';
    else
        rew_idx = rew_idx + 1;
        coherence_matrix_rew(rew_idx, :) = all_data.coherence_spectrum{i}';
    end
end

mean_coherence_aver = nanmean(coherence_matrix_aver, 1);
sem_coherence_aver = nanstd(coherence_matrix_aver, 0, 1) / sqrt(sum(aver_mask));

mean_coherence_rew = nanmean(coherence_matrix_rew, 1);
sem_coherence_rew = nanstd(coherence_matrix_rew, 0, 1) / sqrt(sum(rew_mask));

% Band-specific statistics
band_coherence_aver = all_data.band_coherence(aver_mask, :);
band_coherence_rew = all_data.band_coherence(rew_mask, :);

mean_band_aver = nanmean(band_coherence_aver, 1);
sem_band_aver = nanstd(band_coherence_aver, 0, 1) / sqrt(sum(aver_mask));

mean_band_rew = nanmean(band_coherence_rew, 1);
sem_band_rew = nanstd(band_coherence_rew, 0, 1) / sqrt(sum(rew_mask));

fprintf('✓ Statistics computed\n\n');

%% ========================================================================
%  SECTION 4: VISUALIZATION 1 - POPULATION COHERENCE SPECTRA
%  ========================================================================

fprintf('Creating population coherence spectra...\n');

figure('Position', [50, 50, 1400, 600], 'Name', 'Population Breathing Coherence Spectra');

% Plot with shaded error bars
subplot(1, 2, 1);
hold on;

% Aversive
fill([freq; flipud(freq)], ...
     [mean_coherence_aver' - sem_coherence_aver'; flipud(mean_coherence_aver' + sem_coherence_aver')], ...
     [1, 0.8, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(freq, mean_coherence_aver, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('Aversive (n=%d)', sum(aver_mask)));

% Reward
fill([freq; flipud(freq)], ...
     [mean_coherence_rew' - sem_coherence_rew'; flipud(mean_coherence_rew' + sem_coherence_rew')], ...
     [0.8, 1, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(freq, mean_coherence_rew, 'g-', 'LineWidth', 2, 'DisplayName', sprintf('Reward (n=%d)', sum(rew_mask)));

% Highlight breathing frequencies
plot([2 2], ylim, 'k--', 'LineWidth', 2);
text(2.5, max(ylim)*0.9, '2 Hz (breathing)', 'FontSize', 10, 'FontWeight', 'bold');

hold off;

xlim([0, 15]);
xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Mean Coherence', 'FontSize', 11);
title('Population-Level Breathing Coherence Spectra', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

% Plot difference
subplot(1, 2, 2);
coherence_diff = mean_coherence_aver - mean_coherence_rew;
plot(freq, coherence_diff, 'k-', 'LineWidth', 2);
hold on;
yline(0, 'k--', 'LineWidth', 1);
plot([2 2], ylim, 'r--', 'LineWidth', 2);
hold off;

xlim([0, 15]);
xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Coherence Difference (Aversive - Reward)', 'FontSize', 11);
title('Difference in Breathing Coherence Spectra', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 10);

sgtitle('Spike-Breathing Coherence: Population-Level Analysis', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Population coherence spectra created\n');

%% ========================================================================
%  SECTION 5: VISUALIZATION 2 - BAND-SPECIFIC COMPARISON
%  ========================================================================

fprintf('Creating band-specific comparison...\n');

figure('Position', [100, 100, 1400, 600], 'Name', 'Band-Specific Breathing Coherence');

% Grouped bar plot
subplot(1, 2, 1);
bar_data = [mean_band_aver; mean_band_rew]';
b = bar(bar_data);
b(1).FaceColor = [1, 0.6, 0.6];
b(1).EdgeColor = [0.8, 0, 0];
b(2).FaceColor = [0.6, 1, 0.6];
b(2).EdgeColor = [0, 0.6, 0];

hold on;
% Add error bars
x_aver = b(1).XEndPoints;
x_rew = b(2).XEndPoints;
errorbar(x_aver, mean_band_aver, sem_band_aver, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
errorbar(x_rew, mean_band_rew, sem_band_rew, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
hold off;

xticks(1:n_bands);
xticklabels(band_names);
xtickangle(45);
ylabel('Mean Coherence', 'FontSize', 11);
title('Mean Breathing Coherence by Frequency Band', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'northwest', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

% Breathing-specific analysis
subplot(1, 2, 2);
breathing_coherence_aver = mean_band_aver(1);  % Delta band (1-4 Hz)
sniffing_coherence_aver = mean_band_aver(2);   % Theta band (5-12 Hz)
breathing_coherence_rew = mean_band_rew(1);
sniffing_coherence_rew = mean_band_rew(2);

bar_data_breathing = [breathing_coherence_aver, breathing_coherence_rew; ...
                      sniffing_coherence_aver, sniffing_coherence_rew]';
b2 = bar(bar_data_breathing);
b2(1).FaceColor = [0.8, 0.9, 1];
b2(2).FaceColor = [0.9, 1, 0.8];

xticks([1 2]);
xticklabels({'Aversive', 'Reward'});
ylabel('Mean Coherence', 'FontSize', 11);
title('Breathing-Specific Coherence', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Breathing (1-4 Hz)', 'Sniffing (5-12 Hz)'}, 'Location', 'northwest', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

sgtitle('Breathing Frequency Band Analysis: Aversive vs Reward', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Band-specific comparison created\n');

%% ========================================================================
%  SECTION 6: VISUALIZATION 3 - UNIT × FREQUENCY HEATMAP
%  ========================================================================

fprintf('Creating unit × frequency heatmap...\n');

figure('Position', [150, 150, 1800, 900], 'Name', 'Unit-Level Breathing Coherence Heatmaps');

% Get unique sessions for aversive
aver_indices = find(aver_mask);
unique_sessions_aver = unique(all_data.session_id(aver_indices));
n_sessions_aver = length(unique_sessions_aver);

% Get unique sessions for reward
rew_indices = find(rew_mask);
unique_sessions_rew = unique(all_data.session_id(rew_indices));
n_sessions_rew = length(unique_sessions_rew);

% Create coherence matrix for aversive (units × frequencies)
coherence_matrix_aver_all = nan(sum(aver_mask), n_freq);
session_labels_aver = zeros(sum(aver_mask), 1);
for i = 1:length(aver_indices)
    idx = aver_indices(i);
    coherence_matrix_aver_all(i, :) = all_data.coherence_spectrum{idx}';
    session_labels_aver(i) = all_data.session_id(idx);
end

% Create coherence matrix for reward
coherence_matrix_rew_all = nan(sum(rew_mask), n_freq);
session_labels_rew = zeros(sum(rew_mask), 1);
for i = 1:length(rew_indices)
    idx = rew_indices(i);
    coherence_matrix_rew_all(i, :) = all_data.coherence_spectrum{idx}';
    session_labels_rew(i) = all_data.session_id(idx);
end

% Sort by session
[session_labels_aver_sorted, sort_idx_aver] = sort(session_labels_aver);
coherence_matrix_aver_sorted = coherence_matrix_aver_all(sort_idx_aver, :);

[session_labels_rew_sorted, sort_idx_rew] = sort(session_labels_rew);
coherence_matrix_rew_sorted = coherence_matrix_rew_all(sort_idx_rew, :);

% Subplot 1: Aversive sessions
subplot(2, 1, 1);
imagesc(freq, 1:sum(aver_mask), coherence_matrix_aver_sorted);
colormap(jet);
colorbar;
caxis([0, 0.1]);
xlim([0, 15]);

% Add breathing frequency markers
hold on;
plot([2 2], ylim, 'w--', 'LineWidth', 2);
text(2.5, sum(aver_mask)*0.95, '2 Hz', 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');
hold off;

xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Unit (grouped by session)', 'FontSize', 11);
title(sprintf('Aversive Sessions - Individual Units (n=%d units, %d sessions)', ...
    sum(aver_mask), n_sessions_aver), 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 10);
yticks([]);

% Subplot 2: Reward sessions
subplot(2, 1, 2);
imagesc(freq, 1:sum(rew_mask), coherence_matrix_rew_sorted);
colormap(jet);
colorbar;
caxis([0, 0.1]);
xlim([0, 15]);

hold on;
plot([2 2], ylim, 'w--', 'LineWidth', 2);
text(2.5, sum(rew_mask)*0.95, '2 Hz', 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');
hold off;

xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Unit (grouped by session)', 'FontSize', 11);
title(sprintf('Reward Sessions - Individual Units (n=%d units, %d sessions)', ...
    sum(rew_mask), n_sessions_rew), 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 10);
yticks([]);

sgtitle('Unit-by-Unit Breathing Coherence Spectra (grouped by session)', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Unit × frequency heatmap created\n');

%% ========================================================================
%  SECTION 7: SUMMARY STATISTICS
%  ========================================================================

fprintf('\n=== SUMMARY STATISTICS ===\n');

fprintf('\nAversive Sessions (%d sessions, %d units):\n', length(aversive_sessions), sum(aver_mask));
fprintf('  Mean coherence: %.3f ± %.3f\n', nanmean(all_data.mean_coherence(aver_mask)), ...
    nanstd(all_data.mean_coherence(aver_mask)));
fprintf('  Median coherence: %.3f\n', nanmedian(all_data.mean_coherence(aver_mask)));
fprintf('  Mean spike count: %.1f\n', nanmean(all_data.n_spikes(aver_mask)));

fprintf('\nReward Sessions (%d sessions, %d units):\n', length(reward_sessions), sum(rew_mask));
fprintf('  Mean coherence: %.3f ± %.3f\n', nanmean(all_data.mean_coherence(rew_mask)), ...
    nanstd(all_data.mean_coherence(rew_mask)));
fprintf('  Median coherence: %.3f\n', nanmedian(all_data.mean_coherence(rew_mask)));
fprintf('  Mean spike count: %.1f\n', nanmean(all_data.n_spikes(rew_mask)));

fprintf('\nBand-specific mean coherence:\n');
fprintf('%-15s %10s %10s\n', 'Band', 'Aversive', 'Reward');
fprintf('%-15s %10s %10s\n', '---------------', '----------', '----------');
for b = 1:n_bands
    fprintf('%-15s %10.3f %10.3f\n', band_names{b}, mean_band_aver(b), mean_band_rew(b));
end

fprintf('\nBreathing-specific coherence:\n');
fprintf('  Breathing (1-4 Hz):  Aversive=%.3f, Reward=%.3f\n', breathing_coherence_aver, breathing_coherence_rew);
fprintf('  Sniffing (5-12 Hz):  Aversive=%.3f, Reward=%.3f\n', sniffing_coherence_aver, sniffing_coherence_rew);

fprintf('\n========================================\n');
fprintf('COMPREHENSIVE BREATHING COHERENCE SUMMARY COMPLETE!\n');
fprintf('========================================\n');
fprintf('Total sessions: %d\n', length(aversive_sessions) + length(reward_sessions));
fprintf('Total units: %d\n', n_total_units);
fprintf('Figures created:\n');
fprintf('  1. Population breathing coherence spectra\n');
fprintf('  2. Band-specific breathing coherence comparison\n');
fprintf('  3. Unit-by-unit breathing coherence heatmap\n');
fprintf('\nInterpretation:\n');
fprintf('  - High coherence at 1-4 Hz → Coupling to breathing rhythm\n');
fprintf('  - High coherence at 5-12 Hz → Coupling to sniffing/exploration\n');
fprintf('  - Compare with LFP coherence to separate breathing-specific from general rhythmic coupling\n');
fprintf('========================================\n');
