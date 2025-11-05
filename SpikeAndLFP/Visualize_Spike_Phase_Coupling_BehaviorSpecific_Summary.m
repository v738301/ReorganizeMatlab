%% ========================================================================
%  COMPREHENSIVE SUMMARY: BEHAVIOR-SPECIFIC SPIKE-PHASE COUPLING
%  Aggregates across all sessions for population-level analysis
%  ========================================================================
%
%  This script loads all session results and creates comprehensive summaries:
%  1. Population-level MRL heatmaps (averaged across sessions)
%  2. Session-level statistics with error bars
%  3. Proportion of significant coupling by behavior × band
%  4. MRL distributions
%  5. Reliability statistics
%  6. Comparison: Aversive vs Reward session types
%
%% ========================================================================

clear all;
% close all;

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== COMPREHENSIVE SUMMARY: SPIKE-PHASE COUPLING BY BEHAVIOR ===\n\n');

% Define paths
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_SpikePhase_BehaviorSpecific');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_SpikePhase_BehaviorSpecific');

%% ========================================================================
%  SECTION 2: LOAD ALL SESSION DATA
%  ========================================================================

fprintf('Loading all session data...\n');

% Load aversive sessions
aversive_files = dir(fullfile(RewardAversivePath, '*_spike_phase_coupling_by_behavior.mat'));
fprintf('  Found %d aversive sessions\n', length(aversive_files));

aversive_sessions = cell(length(aversive_files), 1);
for i = 1:length(aversive_files)
    data = load(fullfile(RewardAversivePath, aversive_files(i).name));
    aversive_sessions{i} = data.session_results;
    aversive_sessions{i}.config = data.config;
end

% Load reward sessions
reward_files = dir(fullfile(RewardSeekingPath, '*_spike_phase_coupling_by_behavior.mat'));
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

% Get configuration from first available session
if ~isempty(aversive_sessions)
    config = aversive_sessions{1}.config;
elseif ~isempty(reward_sessions)
    config = reward_sessions{1}.config;
end

n_bands = size(config.frequency_bands, 1);
n_behaviors = length(config.behavior_names);

fprintf('\n✓ Loaded %d aversive + %d reward sessions\n', ...
    length(aversive_sessions), length(reward_sessions));
fprintf('  Frequency bands: %d\n', n_bands);
fprintf('  Behaviors: %d\n\n', n_behaviors);

%% ========================================================================
%  SECTION 3: AGGREGATE DATA ACROSS ALL SESSIONS
%  ========================================================================

fprintf('Aggregating data across all sessions...\n');

% Initialize storage for all measurements
all_data = struct();
all_data.session_type = {};  % 'Aversive' or 'Reward'
all_data.session_id = [];
all_data.unit_id = [];
all_data.band = [];
all_data.behavior = [];
all_data.MRL = [];
all_data.preferred_phase = [];
all_data.rayleigh_p = [];
all_data.is_significant = [];
all_data.n_spikes = [];
all_data.reliability_score = [];
all_data.MRL_CI_lower = [];
all_data.MRL_CI_upper = [];

% Process aversive sessions
for sess_idx = 1:length(aversive_sessions)
    session = aversive_sessions{sess_idx};

    for unit_idx = 1:length(session.unit_results)
        unit_result = session.unit_results{unit_idx};

        if isempty(unit_result)
            continue;
        end

        for band_idx = 1:n_bands
            band_result = unit_result.band_results{band_idx};

            for beh_idx = 1:n_behaviors
                beh_result = band_result.behavior_results{beh_idx};

                % Store data point
                all_data.session_type{end+1} = 'Aversive';
                all_data.session_id(end+1) = sess_idx;
                all_data.unit_id(end+1) = unit_idx;
                all_data.band(end+1) = band_idx;
                all_data.behavior(end+1) = beh_idx;
                all_data.MRL(end+1) = beh_result.MRL;
                all_data.preferred_phase(end+1) = beh_result.preferred_phase;
                all_data.rayleigh_p(end+1) = beh_result.rayleigh_p;
                all_data.is_significant(end+1) = beh_result.is_significant;
                all_data.n_spikes(end+1) = beh_result.n_spikes;
                all_data.reliability_score(end+1) = beh_result.reliability_score;
                all_data.MRL_CI_lower(end+1) = beh_result.MRL_CI_lower;
                all_data.MRL_CI_upper(end+1) = beh_result.MRL_CI_upper;
            end
        end
    end
end

% Process reward sessions
for sess_idx = 1:length(reward_sessions)
    session = reward_sessions{sess_idx};

    for unit_idx = 1:length(session.unit_results)
        unit_result = session.unit_results{unit_idx};

        if isempty(unit_result)
            continue;
        end

        for band_idx = 1:n_bands
            band_result = unit_result.band_results{band_idx};

            for beh_idx = 1:n_behaviors
                beh_result = band_result.behavior_results{beh_idx};

                all_data.session_type{end+1} = 'Reward';
                all_data.session_id(end+1) = sess_idx;
                all_data.unit_id(end+1) = unit_idx;
                all_data.band(end+1) = band_idx;
                all_data.behavior(end+1) = beh_idx;
                all_data.MRL(end+1) = beh_result.MRL;
                all_data.preferred_phase(end+1) = beh_result.preferred_phase;
                all_data.rayleigh_p(end+1) = beh_result.rayleigh_p;
                all_data.is_significant(end+1) = beh_result.is_significant;
                all_data.n_spikes(end+1) = beh_result.n_spikes;
                all_data.reliability_score(end+1) = beh_result.reliability_score;
                all_data.MRL_CI_lower(end+1) = beh_result.MRL_CI_lower;
                all_data.MRL_CI_upper(end+1) = beh_result.MRL_CI_upper;
            end
        end
    end
end

fprintf('✓ Aggregated %d measurements\n\n', length(all_data.MRL));

%% ========================================================================
%  SECTION 4: COMPUTE POPULATION-LEVEL STATISTICS
%  ========================================================================

fprintf('Computing population-level statistics...\n');

% Compute mean MRL for each Band × Behavior × SessionType
mean_MRL_aversive = nan(n_bands, n_behaviors);
mean_MRL_reward = nan(n_bands, n_behaviors);
sem_MRL_aversive = nan(n_bands, n_behaviors);
sem_MRL_reward = nan(n_bands, n_behaviors);
prop_sig_aversive = nan(n_bands, n_behaviors);
prop_sig_reward = nan(n_bands, n_behaviors);
n_units_aversive = zeros(n_bands, n_behaviors);
n_units_reward = zeros(n_bands, n_behaviors);

for band_idx = 1:n_bands
    for beh_idx = 1:n_behaviors
        % Aversive
        mask_aver = strcmp(all_data.session_type, 'Aversive') & ...
                    all_data.band == band_idx & ...
                    all_data.behavior == beh_idx & ...
                    all_data.n_spikes > 0;  % Exclude no-data entries

        MRL_vals_aver = all_data.MRL(mask_aver);
        sig_vals_aver = all_data.is_significant(mask_aver);

        if ~isempty(MRL_vals_aver)
            mean_MRL_aversive(band_idx, beh_idx) = nanmean(MRL_vals_aver);
            sem_MRL_aversive(band_idx, beh_idx) = nanstd(MRL_vals_aver) / sqrt(sum(~isnan(MRL_vals_aver)));
            prop_sig_aversive(band_idx, beh_idx) = sum(sig_vals_aver) / length(sig_vals_aver);
            n_units_aversive(band_idx, beh_idx) = length(MRL_vals_aver);
        end

        % Reward
        mask_rew = strcmp(all_data.session_type, 'Reward') & ...
                   all_data.band == band_idx & ...
                   all_data.behavior == beh_idx & ...
                   all_data.n_spikes > 0;

        MRL_vals_rew = all_data.MRL(mask_rew);
        sig_vals_rew = all_data.is_significant(mask_rew);

        if ~isempty(MRL_vals_rew)
            mean_MRL_reward(band_idx, beh_idx) = nanmean(MRL_vals_rew);
            sem_MRL_reward(band_idx, beh_idx) = nanstd(MRL_vals_rew) / sqrt(sum(~isnan(MRL_vals_rew)));
            prop_sig_reward(band_idx, beh_idx) = sum(sig_vals_rew) / length(sig_vals_rew);
            n_units_reward(band_idx, beh_idx) = length(MRL_vals_rew);
        end
    end
end

fprintf('✓ Statistics computed\n\n');

%% ========================================================================
%  SECTION 5: VISUALIZATION 1 - POPULATION MRL HEATMAPS
%  ========================================================================

fprintf('Creating population-level MRL heatmaps...\n');

figure('Position', [50, 50, 1600, 900], 'Name', 'Population MRL: Aversive vs Reward');

% Aversive sessions
for band_idx = 1:n_bands
    subplot(2, n_bands, band_idx);

    % Create bar plot for this band across behaviors
    MRL_vals = mean_MRL_aversive(band_idx, :);
    SEM_vals = sem_MRL_aversive(band_idx, :);

    bar(1:n_behaviors, MRL_vals, 'FaceColor', [1, 0.6, 0.6], 'EdgeColor', [0.8, 0, 0], 'LineWidth', 1.5);
    hold on;
    errorbar(1:n_behaviors, MRL_vals, SEM_vals, 'k.', 'LineWidth', 1.5, 'CapSize', 10);

    % Mark significant proportions
    for beh_idx = 1:n_behaviors
        if prop_sig_aversive(band_idx, beh_idx) > 0.5
            text(beh_idx, MRL_vals(beh_idx) + SEM_vals(beh_idx) + 0.03, '*', ...
                'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'r');
        end
    end
    hold off;

    ylim([0, 0.5]);
    band_name = config.frequency_bands{band_idx, 1};
    title(sprintf('Aversive - %s', band_name), 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.8, 0, 0]);
    ylabel('Mean MRL', 'FontSize', 10);
    xticks(1:n_behaviors);
    xticklabels(config.behavior_names);
    xtickangle(45);
    grid on;
    set(gca, 'FontSize', 9);
end

% Reward sessions
for band_idx = 1:n_bands
    subplot(2, n_bands, n_bands + band_idx);

    MRL_vals = mean_MRL_reward(band_idx, :);
    SEM_vals = sem_MRL_reward(band_idx, :);

    bar(1:n_behaviors, MRL_vals, 'FaceColor', [0.6, 1, 0.6], 'EdgeColor', [0, 0.6, 0], 'LineWidth', 1.5);
    hold on;
    errorbar(1:n_behaviors, MRL_vals, SEM_vals, 'k.', 'LineWidth', 1.5, 'CapSize', 10);

    for beh_idx = 1:n_behaviors
        if prop_sig_reward(band_idx, beh_idx) > 0.5
            text(beh_idx, MRL_vals(beh_idx) + SEM_vals(beh_idx) + 0.03, '*', ...
                'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0, 0.6, 0]);
        end
    end
    hold off;

    ylim([0, 0.5]);
    band_name = config.frequency_bands{band_idx, 1};
    title(sprintf('Reward - %s', band_name), 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0, 0.6, 0]);
    ylabel('Mean MRL', 'FontSize', 10);
    xticks(1:n_behaviors);
    xticklabels(config.behavior_names);
    xtickangle(45);
    grid on;
    set(gca, 'FontSize', 9);
end

sgtitle('Population-Level Mean Resultant Length (MRL) by Behavior and Frequency Band', ...
    'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Population MRL heatmaps created\n');

%% ========================================================================
%  SECTION 6: VISUALIZATION 2 - PROPORTION OF SIGNIFICANT COUPLING
%  ========================================================================

fprintf('Creating significant coupling proportion heatmaps...\n');

figure('Position', [100, 100, 1400, 600], 'Name', 'Proportion Significant Coupling');

% Aversive
subplot(1, 2, 1);
imagesc(prop_sig_aversive');
colormap(hot);
colorbar;
caxis([0, 1]);

xlabel('Frequency Band', 'FontSize', 11);
ylabel('Behavior', 'FontSize', 11);
title('Aversive Sessions - Proportion Significant (p < 0.05)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.8, 0, 0]);
xticks(1:n_bands);
xticklabels(cellfun(@(x) x, {config.frequency_bands{:,1}}, 'UniformOutput', false));
xtickangle(45);
yticks(1:n_behaviors);
yticklabels(config.behavior_names);
set(gca, 'FontSize', 10);

% Add text annotations
for band_idx = 1:n_bands
    for beh_idx = 1:n_behaviors
        prop_val = prop_sig_aversive(band_idx, beh_idx);
        if ~isnan(prop_val)
            text(band_idx, beh_idx, sprintf('%.2f', prop_val), ...
                'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'w', 'FontWeight', 'bold');
        end
    end
end

% Reward
subplot(1, 2, 2);
imagesc(prop_sig_reward');
colormap(hot);
colorbar;
caxis([0, 1]);

xlabel('Frequency Band', 'FontSize', 11);
ylabel('Behavior', 'FontSize', 11);
title('Reward Sessions - Proportion Significant (p < 0.05)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', [0, 0.6, 0]);
xticks(1:n_bands);
xticklabels(cellfun(@(x) x,  {config.frequency_bands{:,1}}, 'UniformOutput', false));
xtickangle(45);
yticks(1:n_behaviors);
yticklabels(config.behavior_names);
set(gca, 'FontSize', 10);

% Add text annotations
for band_idx = 1:n_bands
    for beh_idx = 1:n_behaviors
        prop_val = prop_sig_reward(band_idx, beh_idx);
        if ~isnan(prop_val)
            text(band_idx, beh_idx, sprintf('%.2f', prop_val), ...
                'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'w', 'FontWeight', 'bold');
        end
    end
end

sgtitle('Proportion of Units with Significant Spike-Phase Coupling', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Significant coupling heatmaps created\n');

%% ========================================================================
%  SECTION 7: VISUALIZATION 3 - MRL DISTRIBUTIONS
%  ========================================================================

fprintf('Creating MRL distribution plots...\n');

figure('Position', [150, 150, 1600, 800], 'Name', 'MRL Distributions');

for band_idx = 1:n_bands
    subplot(2, 3, band_idx);

    % Get MRL values for this band (all behaviors combined, with spikes > 0)
    mask_aver = strcmp(all_data.session_type, 'Aversive') & ...
                all_data.band == band_idx & ...
                all_data.n_spikes > 0;
    mask_rew = strcmp(all_data.session_type, 'Reward') & ...
               all_data.band == band_idx & ...
               all_data.n_spikes > 0;

    MRL_aver = all_data.MRL(mask_aver);
    MRL_rew = all_data.MRL(mask_rew);

    % Remove NaN
    MRL_aver = MRL_aver(~isnan(MRL_aver));
    MRL_rew = MRL_rew(~isnan(MRL_rew));

    % Plot histograms
    edges = 0:0.05:1;
    histogram(MRL_aver, edges, 'FaceColor', [1, 0.6, 0.6], 'EdgeColor', [0.8, 0, 0], ...
        'FaceAlpha', 0.6, 'Normalization', 'probability');
    hold on;
    histogram(MRL_rew, edges, 'FaceColor', [0.6, 1, 0.6], 'EdgeColor', [0, 0.6, 0], ...
        'FaceAlpha', 0.6, 'Normalization', 'probability');

    % Add vertical lines for medians
    if ~isempty(MRL_aver)
        xline(median(MRL_aver), 'r--', 'LineWidth', 2);
    end
    if ~isempty(MRL_rew)
        xline(median(MRL_rew), 'g--', 'LineWidth', 2);
    end
    hold off;

    band_name = config.frequency_bands{band_idx, 1};
    title(band_name, 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('MRL', 'FontSize', 10);
    ylabel('Probability', 'FontSize', 10);
    xlim([0, 1]);
    legend({'Aversive', 'Reward'}, 'Location', 'northeast', 'FontSize', 9);
    grid on;
    set(gca, 'FontSize', 9);

    % Add median values as text
    text(0.98, 0.95, sprintf('Med(Aver)=%.3f\nMed(Rew)=%.3f', median(MRL_aver), median(MRL_rew)), ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
        'FontSize', 8, 'BackgroundColor', 'w');
end

sgtitle('MRL Distributions by Frequency Band (All Behaviors Combined)', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ MRL distribution plots created\n');

%% ========================================================================
%  SECTION 8: VISUALIZATION 4 - RELIABILITY STATISTICS
%  ========================================================================

fprintf('Creating reliability statistics...\n');

figure('Position', [200, 200, 1200, 500], 'Name', 'Reliability Statistics');

% Count by reliability score
reliability_counts_aver = zeros(5, 1);
reliability_counts_rew = zeros(5, 1);

for rel_score = 1:5
    mask_aver = strcmp(all_data.session_type, 'Aversive') & ...
                all_data.reliability_score == rel_score & ...
                all_data.n_spikes > 0;
    mask_rew = strcmp(all_data.session_type, 'Reward') & ...
               all_data.reliability_score == rel_score & ...
               all_data.n_spikes > 0;

    reliability_counts_aver(rel_score) = sum(mask_aver);
    reliability_counts_rew(rel_score) = sum(mask_rew);
end

% Convert to percentages
total_aver = sum(reliability_counts_aver);
total_rew = sum(reliability_counts_rew);
reliability_pct_aver = reliability_counts_aver / total_aver * 100;
reliability_pct_rew = reliability_counts_rew / total_rew * 100;

% Plot
subplot(1, 2, 1);
bar([reliability_pct_aver, reliability_pct_rew]);
xlabel('Reliability Score', 'FontSize', 11);
ylabel('Percentage (%)', 'FontSize', 11);
title('Reliability Score Distribution', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'northwest');
xticks(1:5);
xticklabels({'Very Low (1)', 'Low (2)', 'Moderate (3)', 'Good (4)', 'Excellent (5)'});
xtickangle(45);
grid on;
set(gca, 'FontSize', 10);

% Spike count distributions
subplot(1, 2, 2);
mask_aver = strcmp(all_data.session_type, 'Aversive') & all_data.n_spikes > 0;
mask_rew = strcmp(all_data.session_type, 'Reward') & all_data.n_spikes > 0;

spikes_aver = all_data.n_spikes(mask_aver);
spikes_rew = all_data.n_spikes(mask_rew);

edges = logspace(0, 4, 30);  % Log scale from 1 to 10,000
histogram(spikes_aver, edges, 'FaceColor', [1, 0.6, 0.6], 'EdgeColor', [0.8, 0, 0], ...
    'FaceAlpha', 0.6, 'Normalization', 'probability');
hold on;
histogram(spikes_rew, edges, 'FaceColor', [0.6, 1, 0.6], 'EdgeColor', [0, 0.6, 0], ...
    'FaceAlpha', 0.6, 'Normalization', 'probability');
hold off;

set(gca, 'XScale', 'log');
xlabel('Number of Spikes (log scale)', 'FontSize', 11);
ylabel('Probability', 'FontSize', 11);
title('Spike Count Distribution', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'northeast');
grid on;
set(gca, 'FontSize', 10);

sgtitle('Reliability and Spike Count Statistics', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Reliability statistics created\n');

%% ========================================================================
%  SECTION 9: SUMMARY STATISTICS
%  ========================================================================

fprintf('\n==== SUMMARY STATISTICS ====\n');

fprintf('\nAversive Sessions (%d sessions):\n', length(aversive_sessions));
mask_aver = strcmp(all_data.session_type, 'Aversive') & all_data.n_spikes > 0;
fprintf('  Total measurements: %d\n', sum(mask_aver));
fprintf('  Mean MRL: %.3f ± %.3f\n', nanmean(all_data.MRL(mask_aver)), nanstd(all_data.MRL(mask_aver)));
fprintf('  Median MRL: %.3f\n', nanmedian(all_data.MRL(mask_aver)));
fprintf('  Proportion significant: %.1f%%\n', 100 * sum(all_data.is_significant(mask_aver)) / sum(mask_aver));
fprintf('  Median spike count: %d\n', median(all_data.n_spikes(mask_aver)));

fprintf('\nReward Sessions (%d sessions):\n', length(reward_sessions));
mask_rew = strcmp(all_data.session_type, 'Reward') & all_data.n_spikes > 0;
fprintf('  Total measurements: %d\n', sum(mask_rew));
fprintf('  Mean MRL: %.3f ± %.3f\n', nanmean(all_data.MRL(mask_rew)), nanstd(all_data.MRL(mask_rew)));
fprintf('  Median MRL: %.3f\n', nanmedian(all_data.MRL(mask_rew)));
fprintf('  Proportion significant: %.1f%%\n', 100 * sum(all_data.is_significant(mask_rew)) / sum(mask_rew));
fprintf('  Median spike count: %d\n', median(all_data.n_spikes(mask_rew)));

fprintf('\n========================================\n');
fprintf('COMPREHENSIVE SUMMARY COMPLETE!\n');
fprintf('========================================\n');
fprintf('Total sessions: %d\n', length(aversive_sessions) + length(reward_sessions));
fprintf('Total measurements: %d\n', length(all_data.MRL));
fprintf('Figures created:\n');
fprintf('  1. Population-level MRL by behavior and band\n');
fprintf('  2. Proportion of significant coupling\n');
fprintf('  3. MRL distributions\n');
fprintf('  4. Reliability statistics\n');
fprintf('========================================\n');
