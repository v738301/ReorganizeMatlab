%% ========================================================================
%  VISUALIZE SPIKE-PHASE COUPLING (NARROW BANDS)
%  Visualizes behavior-specific coupling in 1-3 Hz, 5-7 Hz, 8-10 Hz
%  ========================================================================
%
%  This script creates comprehensive visualizations for narrow-band
%  spike-phase coupling analysis across behaviors and session types.
%
%  Visualizations:
%  1. MRL heatmap: Behaviors × Narrow Bands × Session Type
%  2. Band-specific comparison: Aversive vs Reward
%  3. Preferred phase distributions (circular plots)
%  4. Session-grouped box plots with statistical tests
%
%% ========================================================================

clear all;
% close all;

%% ========================================================================
%  SECTION 1: LOAD DATA
%  ========================================================================

fprintf('=== VISUALIZING NARROW-BAND SPIKE-PHASE COUPLING ===\n\n');

% Define paths
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_SpikePhase_BehaviorSpecific_NarrowBands');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_SpikePhase_BehaviorSpecific_NarrowBands');

fprintf('Loading data...\n');

% Load aversive sessions
aversive_files = dir(fullfile(RewardAversivePath, '*_spike_phase_coupling_narrowbands.mat'));
fprintf('  Found %d aversive sessions\n', length(aversive_files));

aversive_sessions = cell(length(aversive_files), 1);
for i = 1:length(aversive_files)
    data = load(fullfile(RewardAversivePath, aversive_files(i).name));
    aversive_sessions{i} = data.session_results;
    aversive_sessions{i}.config = data.config;
end

% Load reward sessions
reward_files = dir(fullfile(RewardSeekingPath, '*_spike_phase_coupling_narrowbands.mat'));
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

% Get configuration from first session
config = aversive_sessions{1}.config;
n_bands = size(config.frequency_bands, 1);
n_behaviors = length(config.behavior_names);

fprintf('Frequency bands:\n');
for i = 1:n_bands
    fprintf('  %d. %s: %.1f-%.1f Hz\n', i, config.frequency_bands{i,1}, ...
        config.frequency_bands{i,2}(1), config.frequency_bands{i,2}(2));
end
fprintf('\n');

%% ========================================================================
%  SECTION 2: AGGREGATE DATA
%  ========================================================================

fprintf('Aggregating data...\n');

% Initialize storage
all_data = struct();
all_data.session_type = {};
all_data.session_id = [];
all_data.unit_id = [];
all_data.band_idx = [];
all_data.behavior_idx = [];
all_data.MRL = [];
all_data.preferred_phase = [];
all_data.is_significant = [];
all_data.n_spikes = [];
all_data.reliability_score = [];

unit_counter = 0;

% Process aversive sessions
for sess_idx = 1:length(aversive_sessions)
    session = aversive_sessions{sess_idx};

    for unit_idx = 1:length(session.unit_results)
        unit = session.unit_results{unit_idx};

        if isempty(unit)
            continue;
        end

        unit_counter = unit_counter + 1;

        for band_idx = 1:n_bands
            band_result = unit.band_results{band_idx};

            for beh_idx = 1:n_behaviors
                beh_result = band_result.behavior_results{beh_idx};

                % Store data
                all_data.session_type{end+1} = 'Aversive';
                all_data.session_id(end+1) = sess_idx;
                all_data.unit_id(end+1) = unit_counter;
                all_data.band_idx(end+1) = band_idx;
                all_data.behavior_idx(end+1) = beh_idx;
                all_data.MRL(end+1) = beh_result.MRL;
                all_data.preferred_phase(end+1) = beh_result.preferred_phase;
                all_data.is_significant(end+1) = beh_result.is_significant;
                all_data.n_spikes(end+1) = beh_result.n_spikes;
                all_data.reliability_score(end+1) = beh_result.reliability_score;
            end
        end
    end
end

% Process reward sessions
for sess_idx = 1:length(reward_sessions)
    session = reward_sessions{sess_idx};

    for unit_idx = 1:length(session.unit_results)
        unit = session.unit_results{unit_idx};

        if isempty(unit)
            continue;
        end

        unit_counter = unit_counter + 1;

        for band_idx = 1:n_bands
            band_result = unit.band_results{band_idx};

            for beh_idx = 1:n_behaviors
                beh_result = band_result.behavior_results{beh_idx};

                % Store data
                all_data.session_type{end+1} = 'Reward';
                all_data.session_id(end+1) = sess_idx;
                all_data.unit_id(end+1) = unit_counter;
                all_data.band_idx(end+1) = band_idx;
                all_data.behavior_idx(end+1) = beh_idx;
                all_data.MRL(end+1) = beh_result.MRL;
                all_data.preferred_phase(end+1) = beh_result.preferred_phase;
                all_data.is_significant(end+1) = beh_result.is_significant;
                all_data.n_spikes(end+1) = beh_result.n_spikes;
                all_data.reliability_score(end+1) = beh_result.reliability_score;
            end
        end
    end
end

fprintf('✓ Aggregated data from %d units\n\n', unit_counter);

%% ========================================================================
%  SECTION 3: VISUALIZATION 1 - MRL HEATMAPS
%  ========================================================================

fprintf('Creating MRL heatmaps...\n');

% Separate by session type
aver_mask = strcmp(all_data.session_type, 'Aversive');
rew_mask = strcmp(all_data.session_type, 'Reward');

% Create MRL matrices: Behaviors × Bands
MRL_aver = nan(n_behaviors, n_bands);
MRL_rew = nan(n_behaviors, n_bands);

for beh_idx = 1:n_behaviors
    for band_idx = 1:n_bands
        % Aversive
        mask = aver_mask & all_data.behavior_idx == beh_idx & all_data.band_idx == band_idx;
        MRL_aver(beh_idx, band_idx) = nanmean(all_data.MRL(mask));

        % Reward
        mask = rew_mask & all_data.behavior_idx == beh_idx & all_data.band_idx == band_idx;
        MRL_rew(beh_idx, band_idx) = nanmean(all_data.MRL(mask));
    end
end

% Plot heatmaps
figure('Position', [50, 50, 1400, 600], 'Name', 'MRL Heatmaps');

subplot(1, 3, 1);
imagesc(MRL_aver);
colormap(jet);
colorbar;
caxis([0, 0.5]);
xticks(1:n_bands);
xticklabels({config.frequency_bands{:,1}});
xtickangle(45);
yticks(1:n_behaviors);
yticklabels(config.behavior_names);
title('Aversive Sessions - Mean MRL', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Frequency Band', 'FontSize', 11);
ylabel('Behavior', 'FontSize', 11);
set(gca, 'FontSize', 10);

subplot(1, 3, 2);
imagesc(MRL_rew);
colormap(jet);
colorbar;
caxis([0, 0.5]);
xticks(1:n_bands);
xticklabels({config.frequency_bands{:,1}});
xtickangle(45);
yticks(1:n_behaviors);
yticklabels(config.behavior_names);
title('Reward Sessions - Mean MRL', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Frequency Band', 'FontSize', 11);
ylabel('Behavior', 'FontSize', 11);
set(gca, 'FontSize', 10);

subplot(1, 3, 3);
imagesc(MRL_aver - MRL_rew);
colormap(redblue);
colorbar;
caxis([-0.2, 0.2]);
xticks(1:n_bands);
xticklabels({config.frequency_bands{:,1}});
xtickangle(45);
yticks(1:n_behaviors);
yticklabels(config.behavior_names);
title('Difference (Aversive - Reward)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Frequency Band', 'FontSize', 11);
ylabel('Behavior', 'FontSize', 11);
set(gca, 'FontSize', 10);

sgtitle('Spike-Phase Coupling Strength (MRL) - Narrow Bands', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ MRL heatmaps created\n');

%% ========================================================================
%  SECTION 4: VISUALIZATION 2 - BAND-SPECIFIC COMPARISON
%  ========================================================================

fprintf('Creating band-specific comparisons...\n');

figure('Position', [100, 100, 1600, 500], 'Name', 'Band-Specific Comparison');

for band_idx = 1:n_bands
    subplot(1, 3, band_idx);
    hold on;

    % Collect MRL values for this band
    MRL_aver_band = [];
    MRL_rew_band = [];
    behavior_labels_aver = [];
    behavior_labels_rew = [];

    for beh_idx = 1:n_behaviors
        % Aversive
        mask = aver_mask & all_data.band_idx == band_idx & all_data.behavior_idx == beh_idx;
        MRL_vals = all_data.MRL(mask);
        MRL_vals = MRL_vals(~isnan(MRL_vals));
        MRL_aver_band = [MRL_aver_band; MRL_vals(:)];
        behavior_labels_aver = [behavior_labels_aver; repmat(beh_idx, length(MRL_vals), 1)];

        % Reward
        mask = rew_mask & all_data.band_idx == band_idx & all_data.behavior_idx == beh_idx;
        MRL_vals = all_data.MRL(mask);
        MRL_vals = MRL_vals(~isnan(MRL_vals));
        MRL_rew_band = [MRL_rew_band; MRL_vals(:)];
        behavior_labels_rew = [behavior_labels_rew; repmat(beh_idx, length(MRL_vals), 1)];
    end

    % Grouped box plot
    x_positions = [];
    group_colors = [];

    for beh_idx = 1:n_behaviors
        pos_base = (beh_idx - 1) * 3;

        % Aversive
        aver_vals = MRL_aver_band(behavior_labels_aver == beh_idx);
        if ~isempty(aver_vals)
            h1 = boxplot(aver_vals, 'Positions', pos_base + 1, 'Width', 0.6, ...
                'Colors', [1, 0.6, 0.6], 'Symbol', '');
            set(h1, 'LineWidth', 1.5);
        end

        % Reward
        rew_vals = MRL_rew_band(behavior_labels_rew == beh_idx);
        if ~isempty(rew_vals)
            h2 = boxplot(rew_vals, 'Positions', pos_base + 2, 'Width', 0.6, ...
                'Colors', [0.6, 1, 0.6], 'Symbol', '');
            set(h2, 'LineWidth', 1.5);
        end
    end

    hold off;

    xticks((0:n_behaviors-1)*3 + 1.5);
    xticklabels(config.behavior_names);
    xtickangle(45);
    ylabel('Mean Resultant Length (MRL)', 'FontSize', 11);
    title(sprintf('%s: %.1f-%.1f Hz', config.frequency_bands{band_idx,1}, ...
        config.frequency_bands{band_idx,2}(1), config.frequency_bands{band_idx,2}(2)), ...
        'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 9);
    ylim([0, 0.6]);
end

sgtitle('Spike-Phase Coupling by Behavior and Band', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Band-specific comparisons created\n');

%% ========================================================================
%  SECTION 5: VISUALIZATION 3 - SIGNIFICANT COUPLING PERCENTAGE
%  ========================================================================

fprintf('Creating significance analysis...\n');

figure('Position', [150, 150, 1400, 600], 'Name', 'Significant Coupling');

% Calculate percentage of significant units for each band × behavior × session type
sig_pct_aver = nan(n_behaviors, n_bands);
sig_pct_rew = nan(n_behaviors, n_bands);

for beh_idx = 1:n_behaviors
    for band_idx = 1:n_bands
        % Aversive
        mask = aver_mask & all_data.band_idx == band_idx & all_data.behavior_idx == beh_idx;
        mask = mask & all_data.n_spikes > 0;  % Only count units with spikes
        if sum(mask) > 0
            sig_pct_aver(beh_idx, band_idx) = sum(all_data.is_significant(mask)) / sum(mask) * 100;
        end

        % Reward
        mask = rew_mask & all_data.band_idx == band_idx & all_data.behavior_idx == beh_idx;
        mask = mask & all_data.n_spikes > 0;
        if sum(mask) > 0
            sig_pct_rew(beh_idx, band_idx) = sum(all_data.is_significant(mask)) / sum(mask) * 100;
        end
    end
end

% Plot
for band_idx = 1:n_bands
    subplot(1, 3, band_idx);

    x = 1:n_behaviors;
    bar_data = [sig_pct_aver(:, band_idx), sig_pct_rew(:, band_idx)];

    b = bar(x, bar_data);
    b(1).FaceColor = [1, 0.6, 0.6];
    b(1).EdgeColor = [0.8, 0, 0];
    b(2).FaceColor = [0.6, 1, 0.6];
    b(2).EdgeColor = [0, 0.6, 0];

    xticks(1:n_behaviors);
    xticklabels(config.behavior_names);
    xtickangle(45);
    ylabel('% Significant Units', 'FontSize', 11);
    title(sprintf('%s: %.1f-%.1f Hz', config.frequency_bands{band_idx,1}, ...
        config.frequency_bands{band_idx,2}(1), config.frequency_bands{band_idx,2}(2)), ...
        'FontSize', 12, 'FontWeight', 'bold');
    legend({'Aversive', 'Reward'}, 'Location', 'northwest', 'FontSize', 9);
    grid on;
    set(gca, 'FontSize', 9);
    ylim([0, 100]);
end

sgtitle('Percentage of Units with Significant Phase Coupling', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Significance analysis created\n');

%% ========================================================================
%  SECTION 6: VISUALIZATION 4 - CIRCULAR PHASE PLOTS
%  ========================================================================

fprintf('Creating circular phase plots for 5-7 Hz and 8-10 Hz...\n');

% Focus on bands 2 and 3 (5-7 Hz and 8-10 Hz)
bands_to_plot = [2, 3];  % Indices of 5-7 Hz and 8-10 Hz
behaviors_to_plot = [1, 2, 3];  % Reward, Walking, Rearing

figure('Position', [200, 200, 1400, 900], 'Name', 'Circular Phase Distributions');

plot_idx = 1;
for band_plot_idx = 1:length(bands_to_plot)
    band_idx = bands_to_plot(band_plot_idx);

    for beh_plot_idx = 1:length(behaviors_to_plot)
        beh_idx = behaviors_to_plot(beh_plot_idx);

        % Subplot for Aversive
        subplot(length(bands_to_plot), length(behaviors_to_plot)*2, plot_idx);

        % Get significant phases for aversive
        mask = aver_mask & all_data.band_idx == band_idx & all_data.behavior_idx == beh_idx;
        mask = mask & all_data.is_significant & ~isnan(all_data.preferred_phase);
        phases_aver = all_data.preferred_phase(mask);

        if ~isempty(phases_aver)
            polarhistogram(phases_aver, 18, 'FaceColor', [1, 0.6, 0.6], 'EdgeColor', [0.8, 0, 0]);
        end
        title(sprintf('%s - %s\nAversive (n=%d)', config.frequency_bands{band_idx,1}, ...
            config.behavior_names{beh_idx}, length(phases_aver)), 'FontSize', 9);
        set(gca, 'FontSize', 8);

        plot_idx = plot_idx + 1;

        % Subplot for Reward
        subplot(length(bands_to_plot), length(behaviors_to_plot)*2, plot_idx);

        % Get significant phases for reward
        mask = rew_mask & all_data.band_idx == band_idx & all_data.behavior_idx == beh_idx;
        mask = mask & all_data.is_significant & ~isnan(all_data.preferred_phase);
        phases_rew = all_data.preferred_phase(mask);

        if ~isempty(phases_rew)
            polarhistogram(phases_rew, 18, 'FaceColor', [0.6, 1, 0.6], 'EdgeColor', [0, 0.6, 0]);
        end
        title(sprintf('%s - %s\nReward (n=%d)', config.frequency_bands{band_idx,1}, ...
            config.behavior_names{beh_idx}, length(phases_rew)), 'FontSize', 9);
        set(gca, 'FontSize', 8);

        plot_idx = plot_idx + 1;
    end
end

sgtitle('Preferred Phase Distributions (Significant Units Only)', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Circular phase plots created\n');

%% ========================================================================
%  SECTION 7: STATISTICAL SUMMARY
%  ========================================================================

fprintf('\n=== STATISTICAL SUMMARY ===\n\n');

for band_idx = 1:n_bands
    fprintf('%s (%.1f-%.1f Hz):\n', config.frequency_bands{band_idx,1}, ...
        config.frequency_bands{band_idx,2}(1), config.frequency_bands{band_idx,2}(2));
    fprintf('%-25s  Aver MRL  Rew MRL   Diff    p-value\n', 'Behavior');
    fprintf('---------------------------------------------------------------\n');

    for beh_idx = 1:n_behaviors
        % Aversive
        mask = aver_mask & all_data.band_idx == band_idx & all_data.behavior_idx == beh_idx;
        MRL_aver_vals = all_data.MRL(mask & ~isnan(all_data.MRL));

        % Reward
        mask = rew_mask & all_data.band_idx == band_idx & all_data.behavior_idx == beh_idx;
        MRL_rew_vals = all_data.MRL(mask & ~isnan(all_data.MRL));

        if ~isempty(MRL_aver_vals) && ~isempty(MRL_rew_vals)
            mean_aver = mean(MRL_aver_vals);
            mean_rew = mean(MRL_rew_vals);
            diff = mean_aver - mean_rew;

            % Wilcoxon rank-sum test
            [p, h] = ranksum(MRL_aver_vals, MRL_rew_vals);

            sig_str = '';
            if p < 0.001
                sig_str = '***';
            elseif p < 0.01
                sig_str = '**';
            elseif p < 0.05
                sig_str = '*';
            end

            fprintf('%-25s  %.3f     %.3f    %+.3f   %.4f%s\n', ...
                config.behavior_names{beh_idx}, mean_aver, mean_rew, diff, p, sig_str);
        else
            fprintf('%-25s  N/A       N/A      N/A     N/A\n', config.behavior_names{beh_idx});
        end
    end
    fprintf('\n');
end

fprintf('========================================\n');
fprintf('VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Figures created:\n');
fprintf('  1. MRL heatmaps (behaviors × bands)\n');
fprintf('  2. Band-specific comparisons (box plots)\n');
fprintf('  3. Significant coupling percentage\n');
fprintf('  4. Circular phase distributions (5-7 & 8-10 Hz)\n');
fprintf('========================================\n');

%% Helper function for red-blue colormap
function cmap = redblue(n)
    if nargin < 1
        n = 256;
    end

    % Create red-white-blue colormap
    top = [1, 0, 0];
    middle = [1, 1, 1];
    bottom = [0, 0, 1];

    half = ceil(n/2);

    r = [linspace(bottom(1), middle(1), half), linspace(middle(1), top(1), n-half)]';
    g = [linspace(bottom(2), middle(2), half), linspace(middle(2), top(2), n-half)]';
    b = [linspace(bottom(3), middle(3), half), linspace(middle(3), top(3), n-half)]';

    cmap = [r, g, b];
end
