%% Visualize Spike-Phase Coupling Results
% Loads and visualizes spike-phase coupling analysis results
% Compares reward-seeking vs reward-aversive sessions across frequency bands

clear all;
% close all;

%% Load all session results
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_SpikePhase');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_SpikePhase');

fprintf('Loading spike-phase coupling results...\n');

% Load reward-seeking sessions
reward_seeking_files = dir(fullfile(RewardSeekingPath, '*_spike_phase_coupling.mat'));
n_reward = length(reward_seeking_files);
reward_seeking_results = cell(n_reward, 1);

fprintf('  Loading %d reward-seeking sessions...\n', n_reward);
for i = 1:n_reward
    temp = load(fullfile(RewardSeekingPath, reward_seeking_files(i).name));
    reward_seeking_results{i} = temp.session_results;
    config = temp.config;  % Get config from last file
end

% Load reward-aversive sessions
reward_aversive_files = dir(fullfile(RewardAversivePath, '*_spike_phase_coupling.mat'));
n_aversive = length(reward_aversive_files);
reward_aversive_results = cell(n_aversive, 1);

fprintf('  Loading %d reward-aversive sessions...\n', n_aversive);
for i = 1:n_aversive
    temp = load(fullfile(RewardAversivePath, reward_aversive_files(i).name));
    reward_aversive_results{i} = temp.session_results;
end

fprintf('Loading complete!\n\n');

% Get band information
n_bands = size(config.frequency_bands, 1);
band_names = config.frequency_bands(:, 1);
band_ranges = cell2mat(config.frequency_bands(:, 2));

%% Aggregate data across all sessions
fprintf('Aggregating data across sessions...\n');

% Extract coupling metrics for all units across all sessions
[reward_data] = aggregate_session_data(reward_seeking_results, n_bands);
[aversive_data] = aggregate_session_data(reward_aversive_results, n_bands);

fprintf('  Reward-seeking: %d total units\n', reward_data.n_total_units);
fprintf('  Reward-aversive: %d total units\n', aversive_data.n_total_units);

%% Create visualizations
fprintf('\nGenerating visualizations...\n');

%% Figure 1: Overview - MRL distributions per band
create_MRL_distribution_plot(reward_data, aversive_data, band_names, band_ranges);

%% Figure 2: Percentage of significantly modulated units
create_significance_summary_plot(reward_data, aversive_data, band_names, band_ranges);

%% Figure 3: Heatmaps of coupling strength
create_coupling_heatmaps(reward_data, aversive_data, band_names);

%% Figure 4: Preferred phase distributions
create_preferred_phase_plots(reward_data, aversive_data, band_names, band_ranges);

%% Figure 5: Example units with strong phase-locking
create_example_units_plot(reward_seeking_results, reward_aversive_results, band_names, config);

%% Figure 6: Simple Units × Bands Heatmap Overview
fprintf('Creating Figure 6: Simple Units × Bands Heatmap...\n');

fig = figure('Position', [300, 300, 1600, 900]);
set(fig, 'Color', 'white');

% Aversive heatmap
subplot(1, 2, 1);
[~, sort_idx_av] = sort(max(aversive_data.MRL, [], 2, 'omitnan'), 'descend');
MRL_sorted_av = aversive_data.MRL(sort_idx_av, :);
imagesc(1:n_bands, 1:size(MRL_sorted_av, 1), MRL_sorted_av);
colormap(hot);
colorbar;
xlabel('Frequency Band', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Unit (sorted by max MRL)', 'FontSize', 12, 'FontWeight', 'bold');
title('Aversive: Units × Bands (MRL)', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'XTick', 1:n_bands, 'XTickLabel', band_names, 'XTickLabelRotation', 45);
set(gca, 'YTick', []);

% Reward heatmap
subplot(1, 2, 2);
[~, sort_idx_rw] = sort(max(reward_data.MRL, [], 2, 'omitnan'), 'descend');
MRL_sorted_rw = reward_data.MRL(sort_idx_rw, :);
imagesc(1:n_bands, 1:size(MRL_sorted_rw, 1), MRL_sorted_rw);
colormap(hot);
colorbar;
xlabel('Frequency Band', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Unit (sorted by max MRL)', 'FontSize', 12, 'FontWeight', 'bold');
title('Reward: Units × Bands (MRL)', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'XTick', 1:n_bands, 'XTickLabel', band_names, 'XTickLabelRotation', 45);
set(gca, 'YTick', []);

sgtitle('Phase-Locking Overview: All Units × Frequency Bands', ...
    'FontSize', 16, 'FontWeight', 'bold');

fprintf('  ✓ Saved: Figure 6: Simple Units × Bands Heatmap\n');

fprintf('\nVisualization complete!\n');

%% =========================
%% Helper Functions
%% =========================

function [data] = aggregate_session_data(session_results, n_bands)
% Aggregate coupling data across all sessions

data = struct();
data.MRL = [];          % [n_units x n_bands]
data.p_values = [];     % [n_units x n_bands]
data.is_significant = []; % [n_units x n_bands]
data.preferred_phase = []; % [n_units x n_bands]
data.n_spikes = [];     % [n_units x 1]
data.n_total_units = 0;

for sess_idx = 1:length(session_results)
    session = session_results{sess_idx};
    n_units = session.n_units;

    for unit_idx = 1:n_units
        unit = session.units{unit_idx};

        % Extract metrics for all bands
        unit_MRL = zeros(1, n_bands);
        unit_p = zeros(1, n_bands);
        unit_sig = false(1, n_bands);
        unit_phase = zeros(1, n_bands);

        for band_idx = 1:n_bands
            unit_MRL(band_idx) = unit.band_coupling(band_idx).MRL;
            unit_p(band_idx) = unit.band_coupling(band_idx).rayleigh_p;
            unit_sig(band_idx) = unit.band_coupling(band_idx).is_significant;
            unit_phase(band_idx) = unit.band_coupling(band_idx).preferred_phase;
        end

        % Append to aggregate data
        data.MRL = [data.MRL; unit_MRL];
        data.p_values = [data.p_values; unit_p];
        data.is_significant = [data.is_significant; unit_sig];
        data.preferred_phase = [data.preferred_phase; unit_phase];
        data.n_spikes = [data.n_spikes; unit.n_spikes];
        data.n_total_units = data.n_total_units + 1;
    end
end
end


function create_MRL_distribution_plot(reward_data, aversive_data, band_names, band_ranges)
% Create violin/box plots of MRL distributions per band

fig = figure('Position', [100, 100, 1800, 600]);
set(fig, 'Color', 'white');

n_bands = length(band_names);
colors_reward = [0.3 0.7 0.3];
colors_aversive = [0.8 0.3 0.2];

for band_idx = 1:n_bands
    subplot(1, n_bands, band_idx);

    % Get MRL values for this band (remove NaNs)
    reward_MRL = reward_data.MRL(:, band_idx);
    aversive_MRL = aversive_data.MRL(:, band_idx);

    reward_MRL = reward_MRL(~isnan(reward_MRL));
    aversive_MRL = aversive_MRL(~isnan(aversive_MRL));

    % Create grouped box plot
    all_MRL = [reward_MRL; aversive_MRL];
    groups = [ones(length(reward_MRL), 1); 2*ones(length(aversive_MRL), 1)];

    boxplot(all_MRL, groups, 'Labels', {'Reward', 'Aversive'}, ...
        'Colors', [colors_reward; colors_aversive], 'Symbol', '');
    hold on;

    % Add individual points with jitter
    jitter_reward = 1 + (rand(length(reward_MRL), 1) - 0.5) * 0.2;
    jitter_aversive = 2 + (rand(length(aversive_MRL), 1) - 0.5) * 0.2;

    scatter(jitter_reward, reward_MRL, 20, colors_reward, 'filled', 'MarkerFaceAlpha', 0.3);
    scatter(jitter_aversive, aversive_MRL, 20, colors_aversive, 'filled', 'MarkerFaceAlpha', 0.3);

    % Statistical test
    if ~isempty(reward_MRL) && ~isempty(aversive_MRL)
        [p_val, ~] = ranksum(reward_MRL, aversive_MRL);
        if p_val < 0.001
            sig_text = '***';
        elseif p_val < 0.01
            sig_text = '**';
        elseif p_val < 0.05
            sig_text = '*';
        else
            sig_text = 'ns';
        end

        y_max = max([reward_MRL; aversive_MRL]);
        text(1.5, y_max*1.1, sig_text, 'HorizontalAlignment', 'center', 'FontSize', 14);
    end

    title(sprintf('%s\n%.1f-%.1f Hz', band_names{band_idx}, band_ranges(band_idx, 1), band_ranges(band_idx, 2)), ...
        'FontWeight', 'bold');
    ylabel('Mean Resultant Length (MRL)');
    ylim([0 1]);
    box off;
end

sgtitle('Distribution of Phase-Locking Strength Across Frequency Bands', ...
    'FontSize', 16, 'FontWeight', 'bold');
end


function create_significance_summary_plot(reward_data, aversive_data, band_names, band_ranges)
% Create bar plot showing percentage of significantly modulated units

fig = figure('Position', [150, 150, 1200, 500]);
set(fig, 'Color', 'white');

n_bands = length(band_names);

% Calculate percentage of significant units per band
reward_pct = mean(reward_data.is_significant, 1) * 100;
aversive_pct = mean(aversive_data.is_significant, 1) * 100;

% Create grouped bar plot
x_pos = 1:n_bands;
width = 0.35;

bar(x_pos - width/2, reward_pct, width, 'FaceColor', [0.3 0.7 0.3], 'EdgeColor', 'none');
hold on;
bar(x_pos + width/2, aversive_pct, width, 'FaceColor', [0.8 0.3 0.2], 'EdgeColor', 'none');

% Add value labels
for i = 1:n_bands
    text(i - width/2, reward_pct(i) + 2, sprintf('%.1f%%', reward_pct(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
    text(i + width/2, aversive_pct(i) + 2, sprintf('%.1f%%', aversive_pct(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

% Formatting
set(gca, 'XTick', x_pos, 'XTickLabel', band_names);
ylabel('Percentage of Significantly Modulated Units (%)');
xlabel('Frequency Band');
legend({'Reward-Seeking', 'Reward-Aversive'}, 'Location', 'best');
title('Percentage of Units Showing Significant Phase-Locking', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0 max([reward_pct, aversive_pct]) * 1.2]);
box off;
grid on;
grid minor;
end


function create_coupling_heatmaps(reward_data, aversive_data, band_names)
% Create heatmaps showing coupling strength across units and bands

fig = figure('Position', [200, 200, 1400, 600]);
set(fig, 'Color', 'white');

% Sort units by maximum MRL
[~, sort_idx_reward] = sort(max(reward_data.MRL, [], 2, 'omitnan'), 'descend');
[~, sort_idx_aversive] = sort(max(aversive_data.MRL, [], 2, 'omitnan'), 'descend');

% Plot reward-seeking heatmap
subplot(1, 2, 1);
MRL_sorted = reward_data.MRL(sort_idx_reward, :);
imagesc(MRL_sorted);
colormap(hot);
colorbar;
% caxis([0 0.5]);
set(gca, 'XTick', 1:length(band_names), 'XTickLabel', band_names);
ylabel('Units (sorted by max MRL)');
xlabel('Frequency Band');
title('Reward-Seeking: Phase-Locking Strength', 'FontWeight', 'bold');

% Plot reward-aversive heatmap
subplot(1, 2, 2);
MRL_sorted = aversive_data.MRL(sort_idx_aversive, :);
imagesc(MRL_sorted);
colormap(hot);
colorbar;
% caxis([0 0.5]);
set(gca, 'XTick', 1:length(band_names), 'XTickLabel', band_names);
ylabel('Units (sorted by max MRL)');
xlabel('Frequency Band');
title('Reward-Aversive: Phase-Locking Strength', 'FontWeight', 'bold');

sgtitle('Heatmap of Phase-Locking Strength (MRL) Across Units and Bands', ...
    'FontSize', 16, 'FontWeight', 'bold');
end


function create_preferred_phase_plots(reward_data, aversive_data, band_names, band_ranges)
% Create circular plots showing preferred phase distributions

fig = figure('Position', [250, 250, 1800, 600]);
set(fig, 'Color', 'white');

n_bands = length(band_names);

for band_idx = 1:n_bands
    subplot(1, n_bands, band_idx);

    % Get significantly modulated units only
    reward_sig_mask = reward_data.is_significant(:, band_idx);
    aversive_sig_mask = aversive_data.is_significant(:, band_idx);

    reward_phases = reward_data.preferred_phase(reward_sig_mask==1, band_idx);
    aversive_phases = aversive_data.preferred_phase(aversive_sig_mask==1, band_idx);

    % Create polar histogram
    n_bins = 18;
    edges = linspace(-pi, pi, n_bins + 1);

    reward_hist = histcounts(reward_phases, edges);
    aversive_hist = histcounts(aversive_phases, edges);

    bin_centers = edges(1:end-1) + diff(edges(1:2))/2;

    % Convert to polar coordinates
    polarplot(bin_centers, reward_hist, '-o', 'LineWidth', 2, 'Color', [0.3 0.7 0.3], 'MarkerFaceColor', [0.3 0.7 0.3]);
    hold on;
    polarplot(bin_centers, aversive_hist, '-s', 'LineWidth', 2, 'Color', [0.8 0.3 0.2], 'MarkerFaceColor', [0.8 0.3 0.2]);

    title(sprintf('%s\n%.1f-%.1f Hz', band_names{band_idx}, band_ranges(band_idx, 1), band_ranges(band_idx, 2)), ...
        'FontWeight', 'bold');

    if band_idx == 1
        legend({'Reward', 'Aversive'}, 'Location', 'best');
    end
end

sgtitle('Preferred Phase Distributions (Significantly Modulated Units Only)', ...
    'FontSize', 16, 'FontWeight', 'bold');
end


function create_example_units_plot(reward_seeking_results, reward_aversive_results, band_names, config)
% Plot example units with strongest phase-locking

fig = figure('Position', [300, 300, 1800, 1000]);
set(fig, 'Color', 'white');

n_bands = length(band_names);
n_examples = 2;  % Show top 2 units per condition

% Find top units from reward-seeking sessions
[top_reward_units, top_reward_bands] = find_top_units(reward_seeking_results, n_examples);

% Find top units from reward-aversive sessions
[top_aversive_units, top_aversive_bands] = find_top_units(reward_aversive_results, n_examples);

% Plot reward-seeking examples
for i = 1:n_examples
    if ~isempty(top_reward_units{i})
        subplot(2, n_examples, i);
        plot_phase_histogram(top_reward_units{i}, top_reward_bands(i), band_names, config);
        if i == 1
            ylabel('Reward-Seeking', 'FontSize', 12, 'FontWeight', 'bold');
        end
    end
end

% Plot reward-aversive examples
for i = 1:n_examples
    if ~isempty(top_aversive_units{i})
        subplot(2, n_examples, n_examples + i);
        plot_phase_histogram(top_aversive_units{i}, top_aversive_bands(i), band_names, config);
        if i == 1
            ylabel('Reward-Aversive', 'FontSize', 12, 'FontWeight', 'bold');
        end
    end
end

sgtitle('Example Units with Strongest Phase-Locking', 'FontSize', 16, 'FontWeight', 'bold');
end


function [top_units, top_bands] = find_top_units(session_results, n_examples)
% Find units with strongest phase-locking

max_MRL = 0;
top_units = cell(n_examples, 1);
top_bands = zeros(n_examples, 1);
top_MRL_values = zeros(n_examples, 1);

for sess_idx = 1:length(session_results)
    session = session_results{sess_idx};

    for unit_idx = 1:session.n_units
        unit = session.units{unit_idx};

        for band_idx = 1:length(unit.band_coupling)
            MRL = unit.band_coupling(band_idx).MRL;

            if ~isnan(MRL) && unit.band_coupling(band_idx).is_significant
                % Check if this is in top N
                [min_val, min_idx] = min(top_MRL_values);
                if MRL > min_val
                    top_units{min_idx} = unit;
                    top_bands(min_idx) = band_idx;
                    top_MRL_values(min_idx) = MRL;
                end
            end
        end
    end
end
end


function plot_phase_histogram(unit, band_idx, band_names, config)
% Plot phase histogram for a single unit

coupling = unit.band_coupling(band_idx);

% Create phase histogram
phase_edges = linspace(-pi, pi, config.n_phase_bins + 1);
phase_centers = phase_edges(1:end-1) + diff(phase_edges(1:2))/2;

bar(phase_centers, coupling.phase_hist, 'FaceColor', [0.5 0.5 0.8], 'EdgeColor', 'k');
hold on;

% Add preferred phase line
line([coupling.preferred_phase, coupling.preferred_phase], ylim, ...
    'Color', 'r', 'LineWidth', 2, 'LineStyle', '--');

% Format
xlabel('Phase (radians)');
ylabel('Spike Count');
title(sprintf('%s: MRL=%.3f, p=%.4f\n%d spikes', ...
    coupling.band_name, coupling.MRL, coupling.rayleigh_p, unit.n_spikes), ...
    'FontWeight', 'bold', 'FontSize', 10);
xlim([-pi, pi]);
set(gca, 'XTick', [-pi, -pi/2, 0, pi/2, pi], 'XTickLabel', {'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});
box off;
end
