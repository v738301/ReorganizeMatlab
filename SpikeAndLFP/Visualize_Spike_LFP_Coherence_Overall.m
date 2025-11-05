%% ========================================================================
%  VISUALIZE SPIKE-LFP COHERENCE (OVERALL)
%  Individual session viewer for overall coherence results
%  ========================================================================
%
%  This script visualizes spike-LFP coherence for a single session.
%  Shows coherence spectra for all units in that session.
%
%% ========================================================================

clear all;
close all;

%% ========================================================================
%  SECTION 1: LOAD DATA
%  ========================================================================

fprintf('=== VISUALIZING SPIKE-LFP COHERENCE (OVERALL) ===\n\n');

% Define paths
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_SpikeLFPCoherence_Overall');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_SpikeLFPCoherence_Overall');

% Select session type
fprintf('Select session type to visualize:\n');
fprintf('  1. Reward-seeking sessions\n');
fprintf('  2. Reward-aversive sessions\n');
session_type_choice = input('Enter choice (1 or 2): ');

if session_type_choice == 1
    data_path = RewardSeekingPath;
    session_type_name = 'RewardSeeking';
elseif session_type_choice == 2
    data_path = RewardAversivePath;
    session_type_name = 'RewardAversive';
else
    error('Invalid choice. Please enter 1 or 2.');
end

% Get all result files
result_files = dir(fullfile(data_path, '*_spike_lfp_coherence_overall.mat'));

if isempty(result_files)
    error('No result files found in: %s', data_path);
end

fprintf('Found %d sessions in %s\n', length(result_files), data_path);

% List sessions
for i = 1:length(result_files)
    fprintf('  %d. %s\n', i, result_files(i).name);
end

% Select session
session_idx = input(sprintf('Select session to visualize (1-%d): ', length(result_files)));

if session_idx < 1 || session_idx > length(result_files)
    error('Invalid session index');
end

% Load selected session
fprintf('\nLoading: %s\n', result_files(session_idx).name);
load(fullfile(data_path, result_files(session_idx).name));

fprintf('✓ Loaded session: %s\n', session_results.filename);
fprintf('  Duration: %.1f min\n', session_results.session_duration_min);
fprintf('  Units: %d\n', session_results.n_units);
fprintf('  LFP channel: %d\n\n', session_results.best_channel);

%% ========================================================================
%  SECTION 2: EXTRACT COHERENCE SPECTRA
%  ========================================================================

fprintf('Extracting coherence spectra...\n');

n_units = session_results.n_units;
unit_coherence_spectra = cell(n_units, 1);
unit_spike_counts = zeros(n_units, 1);
unit_mean_coherence = zeros(n_units, 1);

for unit_idx = 1:n_units
    unit_result = session_results.unit_coherence_results{unit_idx};

    if ~isempty(unit_result) && ~unit_result.skipped
        unit_coherence_spectra{unit_idx} = unit_result.coherence;
        unit_spike_counts(unit_idx) = unit_result.n_spikes;
        unit_mean_coherence(unit_idx) = nanmean(unit_result.coherence);
    else
        unit_coherence_spectra{unit_idx} = [];
        unit_spike_counts(unit_idx) = 0;
        unit_mean_coherence(unit_idx) = NaN;
    end
end

n_valid_units = sum(~cellfun(@isempty, unit_coherence_spectra));
fprintf('✓ Extracted coherence for %d/%d units\n\n', n_valid_units, n_units);

%% ========================================================================
%  SECTION 3: VISUALIZATION 1 - COHERENCE SPECTRA HEATMAP
%  ========================================================================

fprintf('Creating coherence spectra heatmap...\n');

% Get frequency vector from first valid unit
freq = [];
for unit_idx = 1:n_units
    if ~isempty(unit_coherence_spectra{unit_idx})
        unit_result = session_results.unit_coherence_results{unit_idx};
        freq = unit_result.freq;
        break;
    end
end

if ~isempty(freq)
    % Create matrix: units × frequencies
    coherence_matrix = nan(n_units, length(freq));

    for unit_idx = 1:n_units
        if ~isempty(unit_coherence_spectra{unit_idx})
            coherence_matrix(unit_idx, :) = unit_coherence_spectra{unit_idx}';
        end
    end

    % Plot heatmap
    figure('Position', [50, 50, 1400, 800], 'Name', 'Coherence Spectra Heatmap');

    imagesc(freq, 1:n_units, coherence_matrix);
    colormap(jet);
    colorbar;
    caxis([0, 0.5]);

    xlabel('Frequency (Hz)', 'FontSize', 12);
    ylabel('Unit', 'FontSize', 12);
    title(sprintf('Spike-LFP Coherence Spectra - %s', session_results.filename), ...
        'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'none');

    % Add vertical lines for frequency bands
    hold on;
    band_edges = [1, 4, 5, 12, 15, 30, 30, 60, 80, 100, 100, 150];
    for i = 1:2:length(band_edges)
        xline(band_edges(i), 'w--', 'LineWidth', 1, 'Alpha', 0.3);
        xline(band_edges(i+1), 'w--', 'LineWidth', 1, 'Alpha', 0.3);
    end
    hold off;

    set(gca, 'FontSize', 10);
end

fprintf('✓ Coherence heatmap created\n');

%% ========================================================================
%  SECTION 4: VISUALIZATION 2 - INDIVIDUAL UNIT SPECTRA
%  ========================================================================

fprintf('Creating individual unit coherence spectra...\n');

% Find top 16 units by mean coherence
[~, sorted_idx] = sort(unit_mean_coherence, 'descend', 'MissingPlacement', 'last');
top_units = sorted_idx(1:min(16, n_valid_units));

figure('Position', [100, 100, 1600, 1000], 'Name', 'Individual Unit Coherence Spectra');

for plot_idx = 1:length(top_units)
    unit_idx = top_units(plot_idx);

    if isempty(unit_coherence_spectra{unit_idx})
        continue;
    end

    subplot(4, 4, plot_idx);

    unit_result = session_results.unit_coherence_results{unit_idx};

    % Plot coherence spectrum
    plot(unit_result.freq, unit_result.coherence, 'b-', 'LineWidth', 1.5);
    hold on;

    % Add shaded regions for frequency bands
    band_colors = [0.9, 0.9, 1; 0.9, 1, 0.9; 1, 0.9, 0.9; 1, 1, 0.9; 1, 0.9, 1; 0.9, 1, 1];
    band_ranges = [1, 4; 5, 12; 15, 30; 30, 60; 80, 100; 100, 150];
    ylims = ylim;

    for b = 1:size(band_ranges, 1)
        patch([band_ranges(b,1), band_ranges(b,2), band_ranges(b,2), band_ranges(b,1)], ...
              [ylims(1), ylims(1), ylims(2), ylims(2)], ...
              band_colors(b, :), 'EdgeColor', 'none', 'FaceAlpha', 0.2);
    end

    % Replot coherence on top
    plot(unit_result.freq, unit_result.coherence, 'b-', 'LineWidth', 1.5);
    hold off;

    xlim([0, 150]);
    ylim([0, max(0.5, max(unit_result.coherence))]);
    xlabel('Frequency (Hz)', 'FontSize', 9);
    ylabel('Coherence', 'FontSize', 9);
    title(sprintf('Unit %d (n=%d spikes, mean=%.3f)', unit_idx, unit_result.n_spikes, nanmean(unit_result.coherence)), ...
        'FontSize', 10);
    grid on;
    set(gca, 'FontSize', 8);
end

sgtitle(sprintf('Top 16 Units by Mean Coherence - %s', session_type_name), ...
    'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Individual unit spectra created\n');

%% ========================================================================
%  SECTION 5: VISUALIZATION 3 - BAND-SPECIFIC COHERENCE
%  ========================================================================

fprintf('Creating band-specific coherence plot...\n');

% Extract band-specific mean coherence
band_names = {'Delta', 'Theta', 'Beta', 'Low_Gamma', 'High_Gamma', 'Ultra_Gamma'};
n_bands = length(band_names);
band_coherence = nan(n_units, n_bands);

for unit_idx = 1:n_units
    unit_result = session_results.unit_coherence_results{unit_idx};

    if ~isempty(unit_result) && ~unit_result.skipped
        for b = 1:n_bands
            band_coherence(unit_idx, b) = unit_result.band_mean_coherence.(band_names{b});
        end
    end
end

% Plot
figure('Position', [150, 150, 1200, 600], 'Name', 'Band-Specific Coherence');

% Box plot
subplot(1, 2, 1);
boxplot(band_coherence, 'Labels', band_names);
ylabel('Mean Coherence', 'FontSize', 11);
title('Mean Coherence by Frequency Band', 'FontSize', 12, 'FontWeight', 'bold');
xtickangle(45);
grid on;
set(gca, 'FontSize', 10);

% Bar plot with error bars
subplot(1, 2, 2);
mean_band_coherence = nanmean(band_coherence, 1);
sem_band_coherence = nanstd(band_coherence, 0, 1) / sqrt(sum(~isnan(band_coherence(:,1))));

bar(1:n_bands, mean_band_coherence, 'FaceColor', [0.3, 0.6, 0.9], 'EdgeColor', [0, 0.3, 0.6], 'LineWidth', 1.5);
hold on;
errorbar(1:n_bands, mean_band_coherence, sem_band_coherence, 'k.', 'LineWidth', 1.5, 'CapSize', 10);
hold off;

xticks(1:n_bands);
xticklabels(band_names);
xtickangle(45);
ylabel('Mean Coherence', 'FontSize', 11);
title('Population Mean Coherence by Band', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 10);

sgtitle(sprintf('Frequency Band Analysis - %s', session_results.filename), ...
    'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'none');

fprintf('✓ Band-specific coherence plot created\n');

%% ========================================================================
%  SECTION 6: SUMMARY STATISTICS
%  ========================================================================

fprintf('\n=== SUMMARY STATISTICS ===\n');
fprintf('Session: %s\n', session_results.filename);
fprintf('Session type: %s\n', session_type_name);
fprintf('Duration: %.1f min\n', session_results.session_duration_min);
fprintf('Total units: %d\n', n_units);
fprintf('Valid units: %d\n', n_valid_units);
fprintf('\nCoherence statistics (valid units):\n');
fprintf('  Mean coherence: %.3f ± %.3f\n', nanmean(unit_mean_coherence), nanstd(unit_mean_coherence));
fprintf('  Median coherence: %.3f\n', nanmedian(unit_mean_coherence));
fprintf('  Range: [%.3f, %.3f]\n', min(unit_mean_coherence(~isnan(unit_mean_coherence))), ...
    max(unit_mean_coherence(~isnan(unit_mean_coherence))));

fprintf('\nBand-specific mean coherence:\n');
for b = 1:n_bands
    fprintf('  %s: %.3f ± %.3f\n', band_names{b}, mean_band_coherence(b), sem_band_coherence(b));
end

fprintf('\n========================================\n');
fprintf('VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
