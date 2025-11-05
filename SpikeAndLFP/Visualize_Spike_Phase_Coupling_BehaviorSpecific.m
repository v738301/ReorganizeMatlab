%% ========================================================================
%  VISUALIZE BEHAVIOR-SPECIFIC SPIKE-PHASE COUPLING RESULTS
%  ========================================================================
%
%  This script visualizes results from Spike_Phase_Coupling_BehaviorSpecific.m
%
%  Creates:
%  1. Heatmaps: MRL values across Units × Behaviors for each frequency band
%  2. Reliability overlays: Shows statistical confidence for each measurement
%  3. Polar plots: Phase distributions with confidence cones for selected examples
%  4. Error bar plots: MRL with bootstrap confidence intervals
%  5. Summary statistics: Overall reliability distributions
%
%% ========================================================================

clear all;
close all;

%% ========================================================================
%  SECTION 1: LOAD DATA
%  ========================================================================

fprintf('=== VISUALIZING BEHAVIOR-SPECIFIC SPIKE-PHASE COUPLING ===\n\n');

% Define paths
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_SpikePhase_BehaviorSpecific');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_SpikePhase_BehaviorSpecific');

% Select session type to visualize
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
result_files = dir(fullfile(data_path, '*_spike_phase_coupling_by_behavior.mat'));

if isempty(result_files)
    error('No result files found in: %s', data_path);
end

fprintf('Found %d sessions in %s\n', length(result_files), data_path);

% List sessions
for i = 1:length(result_files)
    fprintf('  %d. %s\n', i, result_files(i).name);
end

% Select session to visualize
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
fprintf('  Frequency bands: %d\n', size(config.frequency_bands, 1));
fprintf('  Behaviors: %d\n\n', length(config.behavior_names));

%% ========================================================================
%  SECTION 2: EXTRACT DATA INTO MATRICES
%  ========================================================================

fprintf('Extracting data into matrices...\n');

n_units = session_results.n_units;
n_bands = size(config.frequency_bands, 1);
n_behaviors = length(config.behavior_names);

% Initialize matrices
MRL_matrix = cell(n_bands, 1);
phase_matrix = cell(n_bands, 1);
pvalue_matrix = cell(n_bands, 1);
spike_count_matrix = cell(n_bands, 1);
reliability_score_matrix = cell(n_bands, 1);
MRL_CI_lower_matrix = cell(n_bands, 1);
MRL_CI_upper_matrix = cell(n_bands, 1);

for band_idx = 1:n_bands
    MRL_matrix{band_idx} = nan(n_units, n_behaviors);
    phase_matrix{band_idx} = nan(n_units, n_behaviors);
    pvalue_matrix{band_idx} = nan(n_units, n_behaviors);
    spike_count_matrix{band_idx} = zeros(n_units, n_behaviors);
    reliability_score_matrix{band_idx} = zeros(n_units, n_behaviors);
    MRL_CI_lower_matrix{band_idx} = nan(n_units, n_behaviors);
    MRL_CI_upper_matrix{band_idx} = nan(n_units, n_behaviors);
end

% Extract data
for unit_idx = 1:n_units
    unit_result = session_results.unit_results{unit_idx};

    if isempty(unit_result)
        continue;
    end

    for band_idx = 1:n_bands
        band_result = unit_result.band_results{band_idx};

        for beh_idx = 1:n_behaviors
            beh_result = band_result.behavior_results{beh_idx};

            MRL_matrix{band_idx}(unit_idx, beh_idx) = beh_result.MRL;
            phase_matrix{band_idx}(unit_idx, beh_idx) = beh_result.preferred_phase;
            pvalue_matrix{band_idx}(unit_idx, beh_idx) = beh_result.rayleigh_p;
            spike_count_matrix{band_idx}(unit_idx, beh_idx) = beh_result.n_spikes;
            reliability_score_matrix{band_idx}(unit_idx, beh_idx) = beh_result.reliability_score;
            MRL_CI_lower_matrix{band_idx}(unit_idx, beh_idx) = beh_result.MRL_CI_lower;
            MRL_CI_upper_matrix{band_idx}(unit_idx, beh_idx) = beh_result.MRL_CI_upper;
        end
    end
end

fprintf('✓ Data extraction complete\n\n');

%% ========================================================================
%  SECTION 3: CREATE MRL HEATMAPS WITH RELIABILITY OVERLAYS
%  ========================================================================

fprintf('Creating MRL heatmaps...\n');

figure('Position', [100, 100, 1600, 1000], 'Name', 'MRL Heatmaps with Reliability');

for band_idx = 1:n_bands
    subplot(2, 3, band_idx);

    % Get data for this band
    MRL_data = MRL_matrix{band_idx};
    reliability_scores = reliability_score_matrix{band_idx};
    pvalues = pvalue_matrix{band_idx};

    % Create heatmap
    imagesc(MRL_data);
    colormap(jet);
    colorbar;
    caxis([0, 1]);

    % Add reliability markers (circle size indicates reliability)
    hold on;
    for unit_idx = 1:n_units
        for beh_idx = 1:n_behaviors
            rel_score = reliability_scores(unit_idx, beh_idx);
            p_val = pvalues(unit_idx, beh_idx);

            % Skip if no data
            if rel_score == 0 || isnan(p_val)
                continue;
            end

            % Marker size based on reliability
            marker_size = rel_score * 3;  % 3-15 points

            % Marker style based on significance
            if p_val < config.alpha
                % Significant: filled circle
                plot(beh_idx, unit_idx, 'ko', 'MarkerSize', marker_size, ...
                    'MarkerFaceColor', 'k', 'LineWidth', 1);
            else
                % Non-significant: open circle
                plot(beh_idx, unit_idx, 'ko', 'MarkerSize', marker_size, ...
                    'LineWidth', 1);
            end
        end
    end
    hold off;

    % Format
    band_name = config.frequency_bands{band_idx, 1};
    band_range = config.frequency_bands{band_idx, 2};
    title(sprintf('%s (%d-%d Hz)', band_name, band_range(1), band_range(2)), ...
        'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Behavior Class', 'FontSize', 10);
    ylabel('Unit', 'FontSize', 10);
    xticks(1:n_behaviors);
    xticklabels(config.behavior_names);
    xtickangle(45);
    yticks(1:n_units);
    set(gca, 'FontSize', 9);
end

sgtitle(sprintf('Mean Resultant Length (MRL) - %s - %s', session_results.filename, session_type_name), ...
    'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

% Add legend
annotation('textbox', [0.02, 0.02, 0.2, 0.05], 'String', ...
    'Circle size = reliability | Filled = significant (p < 0.05)', ...
    'EdgeColor', 'none', 'FontSize', 9);

fprintf('✓ MRL heatmaps created\n');

%% ========================================================================
%  SECTION 4: SPIKE COUNT HEATMAPS
%  ========================================================================

fprintf('Creating spike count heatmaps...\n');

figure('Position', [150, 150, 1600, 1000], 'Name', 'Spike Count Heatmaps');

for band_idx = 1:n_bands
    subplot(2, 3, band_idx);

    % Get data
    spike_counts = spike_count_matrix{band_idx};

    % Log scale for better visualization
    log_spike_counts = log10(spike_counts + 1);  % +1 to handle zeros

    imagesc(log_spike_counts);
    colormap(hot);
    colorbar;

    % Format
    band_name = config.frequency_bands{band_idx, 1};
    band_range = config.frequency_bands{band_idx, 2};
    title(sprintf('%s (%d-%d Hz)', band_name, band_range(1), band_range(2)), ...
        'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Behavior Class', 'FontSize', 10);
    ylabel('Unit', 'FontSize', 10);
    xticks(1:n_behaviors);
    xticklabels(config.behavior_names);
    xtickangle(45);
    yticks(1:n_units);
    set(gca, 'FontSize', 9);
end

sgtitle(sprintf('Spike Counts (log10 scale) - %s - %s', session_results.filename, session_type_name), ...
    'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

fprintf('✓ Spike count heatmaps created\n');

%% ========================================================================
%  SECTION 5: POLAR PLOTS FOR SELECTED EXAMPLES
%  ========================================================================

fprintf('Creating polar plots for high-reliability examples...\n');

% Find examples with high MRL, high reliability, and significance
example_units = [];
example_bands = [];
example_behaviors = [];
example_MRLs = [];

for band_idx = 1:n_bands
    for unit_idx = 1:n_units
        for beh_idx = 1:n_behaviors
            MRL = MRL_matrix{band_idx}(unit_idx, beh_idx);
            rel_score = reliability_score_matrix{band_idx}(unit_idx, beh_idx);
            p_val = pvalue_matrix{band_idx}(unit_idx, beh_idx);
            n_spikes = spike_count_matrix{band_idx}(unit_idx, beh_idx);

            % Select high-quality examples
            if ~isnan(MRL) && MRL > 0.3 && rel_score >= 3 && p_val < config.alpha && n_spikes >= 50
                example_units = [example_units; unit_idx];
                example_bands = [example_bands; band_idx];
                example_behaviors = [example_behaviors; beh_idx];
                example_MRLs = [example_MRLs; MRL];
            end
        end
    end
end

% Sort by MRL and take top examples
[~, sort_idx] = sort(example_MRLs, 'descend');
n_examples = min(6, length(example_units));

if n_examples > 0
    figure('Position', [200, 200, 1400, 900], 'Name', 'Polar Plots - Top Examples');

    for ex_idx = 1:n_examples
        subplot(2, 3, ex_idx);

        unit_idx = example_units(sort_idx(ex_idx));
        band_idx = example_bands(sort_idx(ex_idx));
        beh_idx = example_behaviors(sort_idx(ex_idx));

        % Get spike phases
        unit_result = session_results.unit_results{unit_idx};
        band_result = unit_result.band_results{band_idx};
        beh_result = band_result.behavior_results{beh_idx};

        spike_phases = beh_result.spike_phases;

        % Create polar histogram
        polarhistogram(spike_phases, config.n_phase_bins, 'Normalization', 'probability');
        hold on;

        % Add preferred phase arrow
        preferred_phase = beh_result.preferred_phase;
        MRL = beh_result.MRL;
        polarplot([preferred_phase, preferred_phase], [0, MRL], 'r-', 'LineWidth', 3);

        % Add confidence cone
        phase_CI_lower = beh_result.phase_CI_lower;
        phase_CI_upper = beh_result.phase_CI_upper;

        if ~isnan(phase_CI_lower) && ~isnan(phase_CI_upper)
            % Draw confidence cone
            theta_cone = linspace(phase_CI_lower, phase_CI_upper, 50);
            r_cone = repmat(MRL, 1, length(theta_cone));
            polarplot(theta_cone, r_cone, 'r--', 'LineWidth', 1.5);
        end

        hold off;

        % Format
        band_name = config.frequency_bands{band_idx, 1};
        beh_name = config.behavior_names{beh_idx};
        title(sprintf('Unit %d | %s | %s\nMRL=%.2f, p=%.1e, n=%d', ...
            unit_idx, band_name, beh_name, MRL, beh_result.rayleigh_p, beh_result.n_spikes), ...
            'FontSize', 10);
    end

    sgtitle(sprintf('Phase Distribution Examples - %s', session_type_name), ...
        'FontSize', 14, 'FontWeight', 'bold');

    fprintf('✓ Created %d polar plots\n', n_examples);
else
    fprintf('  No high-quality examples found (MRL > 0.3, reliability ≥ 3, p < 0.05, n ≥ 50)\n');
end

%% ========================================================================
%  SECTION 6: MRL WITH ERROR BARS (BOOTSTRAP CI)
%  ========================================================================

fprintf('Creating MRL error bar plots...\n');

% Select a frequency band for detailed visualization
fprintf('Select frequency band for detailed MRL error bars:\n');
for i = 1:n_bands
    fprintf('  %d. %s\n', i, config.frequency_bands{i, 1});
end
band_choice = input(sprintf('Enter choice (1-%d): ', n_bands));

if band_choice < 1 || band_choice > n_bands
    error('Invalid band choice');
end

figure('Position', [250, 250, 1400, 800], 'Name', 'MRL with Bootstrap CI');

for beh_idx = 1:n_behaviors
    subplot(2, 4, beh_idx);

    % Get data for this behavior
    MRL_vals = MRL_matrix{band_choice}(:, beh_idx);
    MRL_CI_lower = MRL_CI_lower_matrix{band_choice}(:, beh_idx);
    MRL_CI_upper = MRL_CI_upper_matrix{band_choice}(:, beh_idx);
    pvalues = pvalue_matrix{band_choice}(:, beh_idx);
    reliability_scores = reliability_score_matrix{band_choice}(:, beh_idx);

    % Filter out NaN values
    valid_idx = ~isnan(MRL_vals);
    unit_ids = find(valid_idx);
    MRL_vals = MRL_vals(valid_idx);
    MRL_CI_lower = MRL_CI_lower(valid_idx);
    MRL_CI_upper = MRL_CI_upper(valid_idx);
    pvalues = pvalues(valid_idx);
    reliability_scores = reliability_scores(valid_idx);

    if isempty(unit_ids)
        title(config.behavior_names{beh_idx}, 'FontSize', 11);
        text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
        continue;
    end

    % Calculate error bars
    err_lower = MRL_vals - MRL_CI_lower;
    err_upper = MRL_CI_upper - MRL_vals;

    % Color by significance
    colors = zeros(length(pvalues), 3);
    for i = 1:length(pvalues)
        if pvalues(i) < config.alpha
            colors(i, :) = [0.8, 0.2, 0.2];  % Red for significant
        else
            colors(i, :) = [0.5, 0.5, 0.5];  % Gray for non-significant
        end
    end

    % Plot error bars
    hold on;
    for i = 1:length(unit_ids)
        errorbar(i, MRL_vals(i), err_lower(i), err_upper(i), ...
            'o', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), ...
            'MarkerSize', 5 + reliability_scores(i), 'LineWidth', 1.5);
    end
    hold off;

    % Format
    ylim([0, 1]);
    xlim([0, length(unit_ids) + 1]);
    xlabel('Unit', 'FontSize', 9);
    ylabel('MRL', 'FontSize', 9);
    title(config.behavior_names{beh_idx}, 'FontSize', 11);
    grid on;
end

band_name = config.frequency_bands{band_choice, 1};
sgtitle(sprintf('MRL with Bootstrap CI - %s - %s - %s', band_name, session_results.filename, session_type_name), ...
    'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'none');

fprintf('✓ MRL error bar plots created\n');

%% ========================================================================
%  SECTION 7: RELIABILITY SUMMARY STATISTICS
%  ========================================================================

fprintf('\n=== RELIABILITY SUMMARY ===\n');

for band_idx = 1:n_bands
    band_name = config.frequency_bands{band_idx, 1};
    fprintf('\n%s:\n', band_name);

    spike_counts = spike_count_matrix{band_idx}(:);
    reliability_scores = reliability_score_matrix{band_idx}(:);
    pvalues = pvalue_matrix{band_idx}(:);

    % Remove zero entries (no data)
    valid_mask = spike_counts > 0;
    spike_counts = spike_counts(valid_mask);
    reliability_scores = reliability_scores(valid_mask);
    pvalues = pvalues(valid_mask);

    if isempty(spike_counts)
        fprintf('  No data\n');
        continue;
    end

    % Count by reliability class
    n_very_low = sum(reliability_scores == 1);
    n_low = sum(reliability_scores == 2);
    n_moderate = sum(reliability_scores == 3);
    n_good = sum(reliability_scores == 4);
    n_excellent = sum(reliability_scores == 5);
    n_total = length(reliability_scores);

    fprintf('  Total measurements: %d\n', n_total);
    fprintf('  Reliability distribution:\n');
    fprintf('    Very low (1):  %3d (%.1f%%)\n', n_very_low, 100*n_very_low/n_total);
    fprintf('    Low (2):       %3d (%.1f%%)\n', n_low, 100*n_low/n_total);
    fprintf('    Moderate (3):  %3d (%.1f%%)\n', n_moderate, 100*n_moderate/n_total);
    fprintf('    Good (4):      %3d (%.1f%%)\n', n_good, 100*n_good/n_total);
    fprintf('    Excellent (5): %3d (%.1f%%)\n', n_excellent, 100*n_excellent/n_total);

    % Significance statistics
    n_significant = sum(pvalues < config.alpha);
    fprintf('  Significant coupling (p < %.2f): %d (%.1f%%)\n', ...
        config.alpha, n_significant, 100*n_significant/n_total);

    % Spike count statistics
    fprintf('  Spike counts: median=%d, mean=%.1f, range=[%d, %d]\n', ...
        median(spike_counts), mean(spike_counts), min(spike_counts), max(spike_counts));
end

%% ========================================================================
%  SECTION 8: SUMMARY
%  ========================================================================

fprintf('\n========================================\n');
fprintf('VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Session: %s\n', session_results.filename);
fprintf('Session type: %s\n', session_type_name);
fprintf('Figures created:\n');
fprintf('  1. MRL heatmaps with reliability overlays\n');
fprintf('  2. Spike count heatmaps\n');
if n_examples > 0
    fprintf('  3. Polar plots (%d examples)\n', n_examples);
end
fprintf('  4. MRL error bar plots with bootstrap CI\n');
fprintf('========================================\n');
