%% ========================================================================
%  COMPREHENSIVE SUMMARY: SPIKE-LFP COHERENCE (OVERALL)
%  Aggregates across all sessions for population-level analysis
%  ========================================================================
%
%  This script loads all session results and creates comprehensive summaries:
%  1. Population-level coherence spectra (averaged across all units)
%  2. Band-specific coherence comparison: Aversive vs Reward
%  3. Unit-level coherence distributions
%  4. Spike count vs coherence relationship
%
%% ========================================================================

clear all;
% close all;

%% ========================================================================
%  SECTION 1: LOAD ALL SESSION DATA
%  ========================================================================

fprintf('=== COMPREHENSIVE SUMMARY: SPIKE-LFP COHERENCE ===\n\n');

% Define paths
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_SpikeLFPCoherence_Overall');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_SpikeLFPCoherence_Overall');

fprintf('Loading all session data...\n');

% Load aversive sessions
aversive_files = dir(fullfile(RewardAversivePath, '*_spike_lfp_coherence_overall.mat'));
fprintf('  Found %d aversive sessions\n', length(aversive_files));

aversive_sessions = cell(length(aversive_files), 1);
for i = 1:length(aversive_files)
    data = load(fullfile(RewardAversivePath, aversive_files(i).name));
    aversive_sessions{i} = data.session_results;
    aversive_sessions{i}.config = data.config;
end

% Load reward sessions
reward_files = dir(fullfile(RewardSeekingPath, '*_spike_lfp_coherence_overall.mat'));
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

figure('Position', [50, 50, 1400, 600], 'Name', 'Population Coherence Spectra');

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

hold off;

xlim([0, 150]);
xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Mean Coherence', 'FontSize', 11);
title('Population-Level Coherence Spectra', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

% Plot difference
subplot(1, 2, 2);
coherence_diff = mean_coherence_aver - mean_coherence_rew;
plot(freq, coherence_diff, 'k-', 'LineWidth', 2);
hold on;
yline(0, 'k--', 'LineWidth', 1);
hold off;

xlim([0, 150]);
xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Coherence Difference (Aversive - Reward)', 'FontSize', 11);
title('Difference in Coherence Spectra', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 10);

sgtitle('Spike-LFP Coherence: Population-Level Analysis', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Population coherence spectra created\n');

%% ========================================================================
%  SECTION 5: VISUALIZATION 2 - BAND-SPECIFIC COMPARISON
%  ========================================================================

fprintf('Creating band-specific comparison...\n');

figure('Position', [100, 100, 1400, 600], 'Name', 'Band-Specific Coherence');

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
title('Mean Coherence by Frequency Band', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'northwest', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

% Box plots
subplot(1, 2, 2);
band_data_combined = [band_coherence_aver; band_coherence_rew];
group_labels = [repmat({'Aversive'}, sum(aver_mask), 1); repmat({'Reward'}, sum(rew_mask), 1)];

% Create grouped box plot for selected bands (to avoid overcrowding)
selected_bands = [2, 3, 4];  % Theta, Beta, Low Gamma
band_positions = [];
band_tick_labels = {};
for b_idx = 1:length(selected_bands)
    b = selected_bands(b_idx);
    band_positions = [band_positions, (b_idx-1)*3 + [1, 2]];
    band_tick_labels{end+1} = band_names{b};
end

hold on;
colors = [1, 0.6, 0.6; 0.6, 1, 0.6];
for b_idx = 1:length(selected_bands)
    b = selected_bands(b_idx);
    pos_base = (b_idx-1)*3;

    % Aversive
    bp1 = boxplot(band_coherence_aver(:, b), 'Positions', pos_base + 1, ...
        'Widths', 0.8, 'Colors', colors(1,:));
    set(bp1, 'LineWidth', 1.5);

    % Reward
    bp2 = boxplot(band_coherence_rew(:, b), 'Positions', pos_base + 2, ...
        'Widths', 0.8, 'Colors', colors(2,:));
    set(bp2, 'LineWidth', 1.5);
end
hold off;

xticks((0:length(selected_bands)-1)*3 + 1.5);
xticklabels(band_tick_labels);
ylabel('Coherence', 'FontSize', 11);
title('Coherence Distribution (Selected Bands)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 10);

sgtitle('Frequency Band Analysis: Aversive vs Reward', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Band-specific comparison created\n');

%% ========================================================================
%  SECTION 6: VISUALIZATION 3 - SPIKE COUNT VS COHERENCE
%  ========================================================================

fprintf('Creating spike count vs coherence plot...\n');

figure('Position', [150, 150, 1200, 500], 'Name', 'Spike Count vs Coherence');

subplot(1, 2, 1);
scatter(all_data.n_spikes(aver_mask), all_data.mean_coherence(aver_mask), ...
    50, [1, 0.6, 0.6], 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
scatter(all_data.n_spikes(rew_mask), all_data.mean_coherence(rew_mask), ...
    50, [0.6, 1, 0.6], 'filled', 'MarkerFaceAlpha', 0.5);
hold off;

set(gca, 'XScale', 'log');
xlabel('Spike Count (log scale)', 'FontSize', 11);
ylabel('Mean Coherence', 'FontSize', 11);
title('Spike Count vs Mean Coherence', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'southeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

% Coherence distributions
subplot(1, 2, 2);
edges = 0:0.02:0.5;
histogram(all_data.mean_coherence(aver_mask), edges, ...
    'FaceColor', [1, 0.6, 0.6], 'EdgeColor', [0.8, 0, 0], ...
    'FaceAlpha', 0.6, 'Normalization', 'probability');
hold on;
histogram(all_data.mean_coherence(rew_mask), edges, ...
    'FaceColor', [0.6, 1, 0.6], 'EdgeColor', [0, 0.6, 0], ...
    'FaceAlpha', 0.6, 'Normalization', 'probability');

xline(nanmedian(all_data.mean_coherence(aver_mask)), 'r--', 'LineWidth', 2);
xline(nanmedian(all_data.mean_coherence(rew_mask)), 'g--', 'LineWidth', 2);
hold off;

xlabel('Mean Coherence', 'FontSize', 11);
ylabel('Probability', 'FontSize', 11);
title('Mean Coherence Distributions', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'northeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

sgtitle('Spike Count and Coherence Distributions', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Spike count vs coherence plot created\n');

%% ========================================================================
%  SECTION 7: NARROW-BAND COHERENCE ANALYSIS (5-7 Hz and 8-10 Hz)
%  ========================================================================

fprintf('Creating narrow-band coherence analysis (5-7 Hz and 8-10 Hz)...\n');

% Define narrow frequency bands
narrow_bands = struct();
narrow_bands.band1_name = '5-7 Hz';
narrow_bands.band1_range = [5, 7];
narrow_bands.band2_name = '8-10 Hz';
narrow_bands.band2_range = [8, 10];

% Compute mean coherence in each narrow band for each unit
band1_coherence = zeros(n_total_units, 1);
band2_coherence = zeros(n_total_units, 1);

for i = 1:n_total_units
    freq_vec = all_data.freq{i};
    coh_vec = all_data.coherence_spectrum{i};

    % Band 1: 5-7 Hz
    band1_mask = freq_vec >= narrow_bands.band1_range(1) & freq_vec <= narrow_bands.band1_range(2);
    band1_coherence(i) = nanmean(coh_vec(band1_mask));

    % Band 2: 8-10 Hz
    band2_mask = freq_vec >= narrow_bands.band2_range(1) & freq_vec <= narrow_bands.band2_range(2);
    band2_coherence(i) = nanmean(coh_vec(band2_mask));
end

% Get unique sessions and assign colors
unique_sessions_aver = unique(all_data.session_id(aver_mask));
unique_sessions_rew = unique(all_data.session_id(rew_mask));
n_sessions_aver = length(unique_sessions_aver);
n_sessions_rew = length(unique_sessions_rew);

% Create colormap for sessions
cmap_aver = lines(n_sessions_aver);  % Different color for each aversive session
cmap_rew = lines(n_sessions_rew);    % Different color for each reward session

% Create figure
figure('Position', [200, 200, 1600, 1000], 'Name', 'Narrow-Band Coherence by Session');

%% Subplot 1: 5-7 Hz - Aversive
subplot(2, 2, 1);
hold on;

aver_indices = find(aver_mask);
for sess_idx = 1:n_sessions_aver
    session_id = unique_sessions_aver(sess_idx);
    unit_mask = all_data.session_id(aver_indices) == session_id;
    units_in_session = aver_indices(unit_mask);

    % Scatter plot with same color for units from same session
    scatter(units_in_session, band1_coherence(units_in_session), 60, ...
        cmap_aver(sess_idx, :), 'filled', 'MarkerFaceAlpha', 0.7, ...
        'DisplayName', sprintf('Session %d', session_id));
end

hold off;
xlabel('Unit Index', 'FontSize', 11);
ylabel('Mean Coherence (5-7 Hz)', 'FontSize', 11);
title('Aversive Sessions - 5-7 Hz', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'eastoutside', 'FontSize', 8);
grid on;
set(gca, 'FontSize', 10);

%% Subplot 2: 5-7 Hz - Reward
subplot(2, 2, 2);
hold on;

rew_indices = find(rew_mask);
for sess_idx = 1:n_sessions_rew
    session_id = unique_sessions_rew(sess_idx);
    unit_mask = all_data.session_id(rew_indices) == session_id;
    units_in_session = rew_indices(unit_mask);

    scatter(units_in_session, band1_coherence(units_in_session), 60, ...
        cmap_rew(sess_idx, :), 'filled', 'MarkerFaceAlpha', 0.7, ...
        'DisplayName', sprintf('Session %d', session_id));
end

hold off;
xlabel('Unit Index', 'FontSize', 11);
ylabel('Mean Coherence (5-7 Hz)', 'FontSize', 11);
title('Reward Sessions - 5-7 Hz', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'eastoutside', 'FontSize', 8);
grid on;
set(gca, 'FontSize', 10);

%% Subplot 3: 8-10 Hz - Aversive
subplot(2, 2, 3);
hold on;

for sess_idx = 1:n_sessions_aver
    session_id = unique_sessions_aver(sess_idx);
    unit_mask = all_data.session_id(aver_indices) == session_id;
    units_in_session = aver_indices(unit_mask);

    scatter(units_in_session, band2_coherence(units_in_session), 60, ...
        cmap_aver(sess_idx, :), 'filled', 'MarkerFaceAlpha', 0.7, ...
        'DisplayName', sprintf('Session %d', session_id));
end

hold off;
xlabel('Unit Index', 'FontSize', 11);
ylabel('Mean Coherence (8-10 Hz)', 'FontSize', 11);
title('Aversive Sessions - 8-10 Hz', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'eastoutside', 'FontSize', 8);
grid on;
set(gca, 'FontSize', 10);

%% Subplot 4: 8-10 Hz - Reward
subplot(2, 2, 4);
hold on;

for sess_idx = 1:n_sessions_rew
    session_id = unique_sessions_rew(sess_idx);
    unit_mask = all_data.session_id(rew_indices) == session_id;
    units_in_session = rew_indices(unit_mask);

    scatter(units_in_session, band2_coherence(units_in_session), 60, ...
        cmap_rew(sess_idx, :), 'filled', 'MarkerFaceAlpha', 0.7, ...
        'DisplayName', sprintf('Session %d', session_id));
end

hold off;
xlabel('Unit Index', 'FontSize', 11);
ylabel('Mean Coherence (8-10 Hz)', 'FontSize', 11);
title('Reward Sessions - 8-10 Hz', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'eastoutside', 'FontSize', 8);
grid on;
set(gca, 'FontSize', 10);

sgtitle('Narrow-Band Coherence: Units Colored by Session', 'FontSize', 13, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 8: INDIVIDUAL UNIT COHERENCE SPECTRA (HEATMAP)
%  ========================================================================

fprintf('Creating individual unit coherence heatmaps...\n');

%% Prepare data organized by session
aver_indices = find(aver_mask);
rew_indices = find(rew_mask);

% Get unique sessions for aversive
unique_sessions_aver = unique(all_data.session_id(aver_indices));
n_sessions_aver = length(unique_sessions_aver);

% Get unique sessions for reward
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

%% Figure 1: Heatmap with session divisions
%% Figure 1: Heatmap with session divisions
figure('Position', [250, 250, 1600, 900], 'Name', 'Individual Unit Coherence Heatmaps');
ax = [];

% Subplot 1: Aversive sessions
ax(end+1) = subplot(2, 1, 1);
imagesc(freq, 1:sum(aver_mask), coherence_matrix_aver_sorted);
colormap(jet);
colorbar;
caxis([0, 0.1]);
xlim([0,20])

% Add session dividers and calculate session centers
hold on;
session_boundaries_aver = [0];  % Start with 0
session_centers_aver = [];
for sess_idx = 1:n_sessions_aver
    sess_id = unique_sessions_aver(sess_idx);
    sess_end = find(session_labels_aver_sorted == sess_id, 1, 'last');
    if sess_idx < n_sessions_aver
        plot([0, 150], [sess_end + 0.5, sess_end + 0.5], 'w-', 'LineWidth', 2);
    end
    session_boundaries_aver(end+1) = sess_end;
    % Calculate center of each session for label placement
    session_centers_aver(sess_idx) = (session_boundaries_aver(sess_idx) + sess_end) / 2;
end

% Add vertical lines for narrow bands
plot([5, 5], ylim, 'w--', 'LineWidth', 2);
plot([7, 7], ylim, 'w--', 'LineWidth', 2);
plot([8, 8], ylim, 'w--', 'LineWidth', 2);
plot([10, 10], ylim, 'w--', 'LineWidth', 2);

% Add session number labels (using data coordinates)
for sess_idx = 1:n_sessions_aver
    text(0.5, session_centers_aver(sess_idx), sprintf('S%d', sess_idx), ...
        'Color', 'white', 'FontSize', 9, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'BackgroundColor', 'black');
end
hold off;

xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Unit (grouped by session)', 'FontSize', 11);
title(sprintf('Aversive Sessions - Individual Units (n=%d units, %d sessions)', ...
    sum(aver_mask), n_sessions_aver), 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 10);
yticks([]);

% Subplot 2: Reward sessions
ax(end+1) = subplot(2, 1, 2);
imagesc(freq, 1:sum(rew_mask), coherence_matrix_rew_sorted);
colormap(jet);
colorbar;
caxis([0, 0.1]);
xlim([0,20])

% Add session dividers and calculate session centers
hold on;
session_boundaries_rew = [0];  % Start with 0
session_centers_rew = [];
for sess_idx = 1:n_sessions_rew
    sess_id = unique_sessions_rew(sess_idx);
    sess_end = find(session_labels_rew_sorted == sess_id, 1, 'last');
    if sess_idx < n_sessions_rew
        plot([0, 150], [sess_end + 0.5, sess_end + 0.5], 'w-', 'LineWidth', 2);
    end
    session_boundaries_rew(end+1) = sess_end;
    % Calculate center of each session for label placement
    session_centers_rew(sess_idx) = (session_boundaries_rew(sess_idx) + sess_end) / 2;
end

% Add vertical lines for narrow bands
plot([5, 5], ylim, 'w--', 'LineWidth', 2);
plot([7, 7], ylim, 'w--', 'LineWidth', 2);
plot([8, 8], ylim, 'w--', 'LineWidth', 2);
plot([10, 10], ylim, 'w--', 'LineWidth', 2);

% Add session number labels (using data coordinates)
for sess_idx = 1:n_sessions_rew
    text(0.5, session_centers_rew(sess_idx), sprintf('S%d', sess_idx), ...
        'Color', 'white', 'FontSize', 9, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'BackgroundColor', 'black');
end
hold off;

xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Unit (grouped by session)', 'FontSize', 11);
title(sprintf('Reward Sessions - Individual Units (n=%d units, %d sessions)', ...
    sum(rew_mask), n_sessions_rew), 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 10);
yticks([]);

sgtitle('Unit-by-Unit Coherence Spectra (grouped by session)', 'FontSize', 13, 'FontWeight', 'bold');
fprintf('✓ Coherence heatmaps created\n');
linkaxes([ax],'x')
xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Unit (grouped by session)', 'FontSize', 11);
title(sprintf('Reward Sessions - Individual Units (n=%d units, %d sessions)', ...
    sum(rew_mask), n_sessions_rew), 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 10);
yticks([]);

sgtitle('Unit-by-Unit Coherence Spectra (grouped by session)', 'FontSize', 13, 'FontWeight', 'bold');
fprintf('✓ Coherence heatmaps created\n');
linkaxes([ax],'x')

%% Figure 2: Session-averaged coherence with individual session traces
figure('Position', [300, 300, 1600, 700], 'Name', 'Session-Level Coherence');
ax = [];
% Subplot 1: Aversive sessions
ax(end+1) = subplot(1, 2, 1);
hold on;

% Plot individual session means
cmap_aver = colorcube(n_sessions_aver);
for sess_idx = 1:n_sessions_aver
    sess_id = unique_sessions_aver(sess_idx);
    sess_mask = session_labels_aver == sess_id;
    sess_coherence = nanmean(coherence_matrix_aver_all(sess_mask, :), 1);

    plot(freq, sess_coherence, 'Color', [cmap_aver(sess_idx, :), 0.5], 'LineWidth', 2, ...
        'DisplayName', sprintf('Session %d (n=%d)', sess_id, sum(sess_mask)));
end

% Overlay population mean
plot(freq, mean_coherence_aver, 'k-', 'LineWidth', 3, 'DisplayName', 'Population Mean');

% Add shaded regions for narrow bands
ylims = ylim;
patch([5, 7, 7, 5], [ylims(1), ylims(1), ylims(2), ylims(2)], ...
    [1, 1, 0.8], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([8, 10, 10, 8], [ylims(1), ylims(1), ylims(2), ylims(2)], ...
    [1, 0.8, 1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

hold off;

xlim([0, 150]);
ylim([0, 0.4]);
xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Mean Coherence', 'FontSize', 11);
title('Aversive Sessions - Session Averages', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'eastoutside', 'FontSize', 8);
grid on;
set(gca, 'FontSize', 10);

% Subplot 2: Reward sessions
ax(end+1) = subplot(1, 2, 2);
hold on;

% Plot individual session means
cmap_rew = colorcube(n_sessions_rew);
for sess_idx = 1:n_sessions_rew
    sess_id = unique_sessions_rew(sess_idx);
    sess_mask = session_labels_rew == sess_id;
    sess_coherence = nanmean(coherence_matrix_rew_all(sess_mask, :), 1);

    plot(freq, sess_coherence, 'Color', [cmap_rew(sess_idx, :), 0.5], 'LineWidth', 2, ...
        'DisplayName', sprintf('Session %d (n=%d)', sess_id, sum(sess_mask)));
end

% Overlay population mean
plot(freq, mean_coherence_rew, 'k-', 'LineWidth', 3, 'DisplayName', 'Population Mean');

% Add shaded regions for narrow bands
ylims = ylim;
patch([5, 7, 7, 5], [ylims(1), ylims(1), ylims(2), ylims(2)], ...
    [1, 1, 0.8], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([8, 10, 10, 8], [ylims(1), ylims(1), ylims(2), ylims(2)], ...
    [1, 0.8, 1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

hold off;

xlim([0, 150]);
ylim([0, 0.4]);
xlabel('Frequency (Hz)', 'FontSize', 11);
ylabel('Mean Coherence', 'FontSize', 11);
title('Reward Sessions - Session Averages', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'eastoutside', 'FontSize', 8);
grid on;
set(gca, 'FontSize', 10);

sgtitle('Session-Level Coherence Spectra', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Session-level coherence plot created\n');
linkaxes([ax],'xy')
xlim([0,15])

%% Figure 3: High coherence units identification
figure('Position', [350, 350, 1600, 900], 'Name', 'High Coherence Units');
ax = [];
% Define high coherence thresholds for narrow bands
threshold_5_7 = prctile(band1_coherence(~isnan(band1_coherence)), 75);  % Top 25%
threshold_8_10 = prctile(band2_coherence(~isnan(band2_coherence)), 75); % Top 25%

% Find high coherence units
high_coh_5_7_aver = aver_mask(:) & band1_coherence > threshold_5_7;
high_coh_8_10_aver = aver_mask(:) & band2_coherence > threshold_8_10;
high_coh_5_7_rew = rew_mask(:) & band1_coherence > threshold_5_7;
high_coh_8_10_rew = rew_mask(:) & band2_coherence > threshold_8_10;

% Subplot 1: 5-7 Hz high coherence units - Aversive
ax(end+1) = subplot(2, 2, 1);
hold on;
high_indices = find(high_coh_5_7_aver);
for i = 1:length(high_indices)
    idx = high_indices(i);
    sess_id = all_data.session_id(idx);
    plot(all_data.freq{idx}, all_data.coherence_spectrum{idx}, ...
        'Color', [cmap_aver(sess_id, :), 0.5], 'LineWidth', 1.5);
end
plot(freq, mean_coherence_aver, 'k--', 'LineWidth', 2, 'DisplayName', 'Population Mean');
hold off;
xlim([0, 150]);
ylim([0, 0.6]);
xlabel('Frequency (Hz)', 'FontSize', 10);
ylabel('Coherence', 'FontSize', 10);
title(sprintf('Aversive: High 5-7 Hz Coherence (n=%d, top 25%%)', sum(high_coh_5_7_aver)), ...
    'FontSize', 11, 'FontWeight', 'bold');
grid on;
patch([5, 7, 7, 5], [0, 0, 0.6, 0.6], [1, 1, 0.8], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Subplot 2: 5-7 Hz high coherence units - Reward
ax(end+1) = subplot(2, 2, 2);
hold on;
high_indices = find(high_coh_5_7_rew);
for i = 1:length(high_indices)
    idx = high_indices(i);
    sess_id = all_data.session_id(idx);
    plot(all_data.freq{idx}, all_data.coherence_spectrum{idx}, ...
        'Color', [cmap_rew(sess_id, :), 0.5], 'LineWidth', 1.5);
end
plot(freq, mean_coherence_rew, 'k--', 'LineWidth', 2, 'DisplayName', 'Population Mean');
hold off;
xlim([0, 150]);
ylim([0, 0.6]);
xlabel('Frequency (Hz)', 'FontSize', 10);
ylabel('Coherence', 'FontSize', 10);
title(sprintf('Reward: High 5-7 Hz Coherence (n=%d, top 25%%)', sum(high_coh_5_7_rew)), ...
    'FontSize', 11, 'FontWeight', 'bold');
grid on;
patch([5, 7, 7, 5], [0, 0, 0.6, 0.6], [1, 1, 0.8], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Subplot 3: 8-10 Hz high coherence units - Aversive
ax(end+1) = subplot(2, 2, 3);
hold on;
high_indices = find(high_coh_8_10_aver);
for i = 1:length(high_indices)
    idx = high_indices(i);
    sess_id = all_data.session_id(idx);
    plot(all_data.freq{idx}, all_data.coherence_spectrum{idx}, ...
        'Color', [cmap_aver(sess_id, :), 0.5], 'LineWidth', 1.5);
end
plot(freq, mean_coherence_aver, 'k--', 'LineWidth', 2, 'DisplayName', 'Population Mean');
hold off;
xlim([0, 150]);
ylim([0, 0.6]);
xlabel('Frequency (Hz)', 'FontSize', 10);
ylabel('Coherence', 'FontSize', 10);
title(sprintf('Aversive: High 8-10 Hz Coherence (n=%d, top 25%%)', sum(high_coh_8_10_aver)), ...
    'FontSize', 11, 'FontWeight', 'bold');
grid on;
patch([8, 10, 10, 8], [0, 0, 0.6, 0.6], [1, 0.8, 1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Subplot 4: 8-10 Hz high coherence units - Reward
ax(end+1) = subplot(2, 2, 4);
hold on;
high_indices = find(high_coh_8_10_rew);
for i = 1:length(high_indices)
    idx = high_indices(i);
    sess_id = all_data.session_id(idx);
    plot(all_data.freq{idx}, all_data.coherence_spectrum{idx}, ...
        'Color', [cmap_rew(sess_id, :), 0.5], 'LineWidth', 1.5);
end
plot(freq, mean_coherence_rew, 'k--', 'LineWidth', 2, 'DisplayName', 'Population Mean');
hold off;
xlim([0, 150]);
ylim([0, 0.6]);
xlabel('Frequency (Hz)', 'FontSize', 10);
ylabel('Coherence', 'FontSize', 10);
title(sprintf('Reward: High 8-10 Hz Coherence (n=%d, top 25%%)', sum(high_coh_8_10_rew)), ...
    'FontSize', 11, 'FontWeight', 'bold');
grid on;
patch([8, 10, 10, 8], [0, 0, 0.6, 0.6], [1, 0.8, 1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

sgtitle('High Coherence Units by Narrow Band (Top 25%)', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ High coherence unit identification created\n');
linkaxes([ax],'xy')
xlim([0,20])

%% Print session distribution of high coherence units
fprintf('\n=== HIGH COHERENCE UNITS BY SESSION ===\n\n');

fprintf('5-7 Hz HIGH COHERENCE (top 25%%, threshold=%.3f):\n', threshold_5_7);
fprintf('Aversive sessions:\n');
for sess_idx = 1:n_sessions_aver
    sess_id = unique_sessions_aver(sess_idx);
    n_high = sum(high_coh_5_7_aver & all_data.session_id == sess_id);
    n_total = sum(aver_mask & all_data.session_id == sess_id);
    fprintf('  Session %d: %d/%d units (%.1f%%)\n', sess_id, n_high, n_total, (n_high/n_total)*100);
end
fprintf('Reward sessions:\n');
for sess_idx = 1:n_sessions_rew
    sess_id = unique_sessions_rew(sess_idx);
    n_high = sum(high_coh_5_7_rew & all_data.session_id == sess_id);
    n_total = sum(rew_mask & all_data.session_id == sess_id);
    fprintf('  Session %d: %d/%d units (%.1f%%)\n', sess_id, n_high, n_total, (n_high/n_total)*100);
end

fprintf('\n8-10 Hz HIGH COHERENCE (top 25%%, threshold=%.3f):\n', threshold_8_10);
fprintf('Aversive sessions:\n');
for sess_idx = 1:n_sessions_aver
    sess_id = unique_sessions_aver(sess_idx);
    n_high = sum(high_coh_8_10_aver & all_data.session_id == sess_id);
    n_total = sum(aver_mask & all_data.session_id == sess_id);
    fprintf('  Session %d: %d/%d units (%.1f%%)\n', sess_id, n_high, n_total, (n_high/n_total)*100);
end
fprintf('Reward sessions:\n');
for sess_idx = 1:n_sessions_rew
    sess_id = unique_sessions_rew(sess_idx);
    n_high = sum(high_coh_8_10_rew & all_data.session_id == sess_id);
    n_total = sum(rew_mask & all_data.session_id == sess_id);
    fprintf('  Session %d: %d/%d units (%.1f%%)\n', sess_id, n_high, n_total, (n_high/n_total)*100);
end
fprintf('\n');

%% ========================================================================
%  SECTION 9: SUMMARY STATISTICS
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

fprintf('\nNarrow-band coherence statistics (5-7 Hz and 8-10 Hz):\n');
fprintf('  5-7 Hz:  Aversive=%.3f, Reward=%.3f\n', nanmean(band1_coherence(aver_mask)), nanmean(band1_coherence(rew_mask)));
fprintf('  8-10 Hz: Aversive=%.3f, Reward=%.3f\n', nanmean(band2_coherence(aver_mask)), nanmean(band2_coherence(rew_mask)));

fprintf('\n========================================\n');
fprintf('COMPREHENSIVE SUMMARY COMPLETE!\n');
fprintf('========================================\n');
fprintf('Total sessions: %d\n', length(aversive_sessions) + length(reward_sessions));
fprintf('Total units: %d\n', n_total_units);
fprintf('Figures created:\n');
fprintf('  1. Population coherence spectra\n');
fprintf('  2. Band-specific comparison\n');
fprintf('  3. Spike count vs coherence\n');
fprintf('  4. Narrow-band coherence by session (5-7 Hz and 8-10 Hz)\n');
fprintf('  5. Session-grouped box plots (5-7 Hz and 8-10 Hz)\n');
fprintf('  6. Unit-by-unit coherence heatmap (grouped by session)\n');
fprintf('  7. Session-level coherence spectra\n');
fprintf('  8. High coherence units (top 25%% in 5-7 and 8-10 Hz)\n');
fprintf('========================================\n');
