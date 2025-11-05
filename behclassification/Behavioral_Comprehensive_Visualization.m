%% ========================================================================
%  BEHAVIORAL COMPREHENSIVE VISUALIZATION
%  Creates comprehensive visualizations from aggregated behavioral data
%  ========================================================================
%
%  This script loads pre-aggregated behavioral data and creates:
%  - Figure 1: Time Budget Analysis
%  - Figure 2: Behavioral Matrix State Occupancy
%  - Figure 3: Session Clustering (PCA)
%  - Figure 4: Session Profiles (Radar Plots)
%  - Figure 5: Breathing Rate by Behavior - Individual Sessions
%  - Figure 6: Breathing Rate Heatmap
%  - Figure 7: Session-Level Breathing Distributions
%
%  Prerequisites: Run Behavioral_Data_Aggregation.m first
%
%% ========================================================================

clear all;
close all;

%% ========================================================================
%  SECTION 1: LOAD AGGREGATED DATA
%  ========================================================================

fprintf('=== BEHAVIORAL COMPREHENSIVE VISUALIZATION ===\n\n');

fprintf('Loading aggregated data...\n');

try
    data = load('behavioral_data_summary.mat');
    behavioral_summary = data.behavioral_summary;
    fprintf('✓ Loaded behavioral_data_summary.mat\n\n');
catch ME
    error('Failed to load data. Run Behavioral_Data_Aggregation.m first!\nError: %s', ME.message);
end

% Extract data
aversive_sessions = behavioral_summary.aversive_sessions;
reward_sessions = behavioral_summary.reward_sessions;
config = behavioral_summary.config;
behavior_names = config.behavior_names;
matrix_feature_names = config.matrix_feature_names;
n_behaviors = config.n_behaviors;
n_matrix_features = config.n_matrix_features;

n_aversive = length(aversive_sessions);
n_reward = length(reward_sessions);
n_total = n_aversive + n_reward;

fprintf('Sessions loaded:\n');
fprintf('  Aversive: %d\n', n_aversive);
fprintf('  Reward: %d\n', n_reward);
fprintf('  Total: %d\n\n', n_total);

%% ========================================================================
%  SECTION 2: FIGURE 1 - TIME BUDGET ANALYSIS
%  ========================================================================

fprintf('Creating Figure 1: Time Budget Analysis...\n');

figure('Position', [50, 50, 1600, 800], 'Name', 'Time Budget Analysis');

% Extract behavior time percentages
behavior_time_aver = nan(n_aversive, n_behaviors);
behavior_time_rew = nan(n_reward, n_behaviors);

for i = 1:n_aversive
    behavior_time_aver(i, :) = aversive_sessions(i).behavior_time_percent;
end

for i = 1:n_reward
    behavior_time_rew(i, :) = reward_sessions(i).behavior_time_percent;
end

% Subplot 1: Heatmap for Aversive
subplot(2, 2, 1);
imagesc(behavior_time_aver');
colormap(hot);
colorbar;
caxis([0, 100]);
yticks(1:n_behaviors);
yticklabels(behavior_names);
xlabel('Session', 'FontSize', 11);
ylabel('Behavior', 'FontSize', 11);
title('Aversive Sessions - Time Budget (%)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 10);

% Subplot 2: Heatmap for Reward
subplot(2, 2, 2);
imagesc(behavior_time_rew');
colormap(hot);
colorbar;
caxis([0, 100]);
yticks(1:n_behaviors);
yticklabels(behavior_names);
xlabel('Session', 'FontSize', 11);
ylabel('Behavior', 'FontSize', 11);
title('Reward Sessions - Time Budget (%)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 10);

% Subplot 3: Mean comparison
subplot(2, 2, 3);
mean_aver = nanmean(behavior_time_aver, 1);
sem_aver = nanstd(behavior_time_aver, 0, 1) / sqrt(n_aversive);
mean_rew = nanmean(behavior_time_rew, 1);
sem_rew = nanstd(behavior_time_rew, 0, 1) / sqrt(n_reward);

x = 1:n_behaviors;
b = bar(x, [mean_aver; mean_rew]');
b(1).FaceColor = [1, 0.6, 0.6];
b(1).EdgeColor = [0.8, 0, 0];
b(2).FaceColor = [0.6, 1, 0.6];
b(2).EdgeColor = [0, 0.6, 0];

hold on;
errorbar(b(1).XEndPoints, mean_aver, sem_aver, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
errorbar(b(2).XEndPoints, mean_rew, sem_rew, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
hold off;

xticks(1:n_behaviors);
xticklabels(behavior_names);
xtickangle(45);
ylabel('Time Budget (%)', 'FontSize', 11);
title('Mean Time Budget Comparison', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'northwest', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

% Subplot 4: Variability comparison
subplot(2, 2, 4);
std_aver = nanstd(behavior_time_aver, 0, 1);
std_rew = nanstd(behavior_time_rew, 0, 1);

b2 = bar(x, [std_aver; std_rew]');
b2(1).FaceColor = [1, 0.6, 0.6];
b2(1).EdgeColor = [0.8, 0, 0];
b2(2).FaceColor = [0.6, 1, 0.6];
b2(2).EdgeColor = [0, 0.6, 0];

xticks(1:n_behaviors);
xticklabels(behavior_names);
xtickangle(45);
ylabel('Standard Deviation (%)', 'FontSize', 11);
title('Inter-Session Variability', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'northwest', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

sgtitle('LSTM Behavior Time Budget Analysis', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Figure 1 created\n');

%% ========================================================================
%  SECTION 3: FIGURE 2 - BEHAVIORAL MATRIX STATE OCCUPANCY
%  ========================================================================

fprintf('Creating Figure 2: Behavioral Matrix State Occupancy...\n');

figure('Position', [100, 100, 1600, 800], 'Name', 'Behavioral Matrix State Occupancy');

% Extract matrix feature percentages
matrix_percent_aver = nan(n_aversive, n_matrix_features);
matrix_percent_rew = nan(n_reward, n_matrix_features);

for i = 1:n_aversive
    matrix_percent_aver(i, :) = aversive_sessions(i).matrix_feature_percent;
end

for i = 1:n_reward
    matrix_percent_rew(i, :) = reward_sessions(i).matrix_feature_percent;
end

% Subplot 1: Heatmap for Aversive
subplot(2, 2, 1);
imagesc(matrix_percent_aver');
colormap(hot);
colorbar;
caxis([0, 100]);
yticks(1:n_matrix_features);
yticklabels(matrix_feature_names);
xlabel('Session', 'FontSize', 11);
ylabel('Feature', 'FontSize', 11);
title('Aversive Sessions - State Occupancy (%)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 9);

% Subplot 2: Heatmap for Reward
subplot(2, 2, 2);
imagesc(matrix_percent_rew');
colormap(hot);
colorbar;
caxis([0, 100]);
yticks(1:n_matrix_features);
yticklabels(matrix_feature_names);
xlabel('Session', 'FontSize', 11);
ylabel('Feature', 'FontSize', 11);
title('Reward Sessions - State Occupancy (%)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 9);

% Subplot 3: Mean comparison
subplot(2, 2, 3);
mean_matrix_aver = nanmean(matrix_percent_aver, 1);
sem_matrix_aver = nanstd(matrix_percent_aver, 0, 1) / sqrt(n_aversive);
mean_matrix_rew = nanmean(matrix_percent_rew, 1);
sem_matrix_rew = nanstd(matrix_percent_rew, 0, 1) / sqrt(n_reward);

x = 1:n_matrix_features;
b = bar(x, [mean_matrix_aver; mean_matrix_rew]');
b(1).FaceColor = [1, 0.6, 0.6];
b(1).EdgeColor = [0.8, 0, 0];
b(2).FaceColor = [0.6, 1, 0.6];
b(2).EdgeColor = [0, 0.6, 0];

hold on;
errorbar(b(1).XEndPoints, mean_matrix_aver, sem_matrix_aver, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
errorbar(b(2).XEndPoints, mean_matrix_rew, sem_matrix_rew, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
hold off;

xticks(1:n_matrix_features);
xticklabels(matrix_feature_names);
xtickangle(45);
ylabel('State Occupancy (%)', 'FontSize', 11);
title('Mean State Occupancy Comparison', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'northwest', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 9);

% Subplot 4: Breathing rate distribution
subplot(2, 2, 4);
breathing_aver = [aversive_sessions.breathing_overall_mean];
breathing_rew = [reward_sessions.breathing_overall_mean];

hold on;
histogram(breathing_aver, 15, 'FaceColor', [1, 0.6, 0.6], 'EdgeColor', [0.8, 0, 0], ...
    'FaceAlpha', 0.6, 'Normalization', 'probability');
histogram(breathing_rew, 15, 'FaceColor', [0.6, 1, 0.6], 'EdgeColor', [0, 0.6, 0], ...
    'FaceAlpha', 0.6, 'Normalization', 'probability');

xline(nanmean(breathing_aver), 'r--', 'LineWidth', 2);
xline(nanmean(breathing_rew), 'g--', 'LineWidth', 2);
hold off;

xlabel('Mean Breathing Rate (Hz)', 'FontSize', 11);
ylabel('Probability', 'FontSize', 11);
title('Overall Breathing Rate Distribution', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward', 'Aver mean', 'Rew mean'}, 'Location', 'northeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

sgtitle('Behavioral Matrix Feature Occupancy', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Figure 2 created\n');

%% ========================================================================
%  SECTION 4: FIGURE 3 - SESSION CLUSTERING (PCA)
%  ========================================================================

fprintf('Creating Figure 3: Session Clustering (PCA)...\n');

% Combine all features for PCA
all_features = [behavior_time_aver, matrix_percent_aver(:, 1:7);  % Don't include breathing freq
                behavior_time_rew, matrix_percent_rew(:, 1:7)];

% Remove rows with NaN
valid_rows = ~any(isnan(all_features), 2);
features_clean = all_features(valid_rows, :);
n_aver_valid = sum(valid_rows(1:n_aversive));
n_rew_valid = sum(valid_rows(n_aversive+1:end));

if sum(valid_rows) < 3
    fprintf('  WARNING: Not enough valid sessions for PCA - skipping Figure 3\n');
else
    figure('Position', [150, 150, 1400, 600], 'Name', 'Session Clustering');

    % Perform PCA
    [coeff, score, ~, ~, explained] = pca(features_clean);

    % Subplot 1: PC1 vs PC2
    subplot(1, 2, 1);
    hold on;

    % Plot aversive
    scatter(score(1:n_aver_valid, 1), score(1:n_aver_valid, 2), 100, ...
        [1, 0.6, 0.6], 'filled', 'MarkerEdgeColor', [0.8, 0, 0], 'LineWidth', 1.5);

    % Plot reward
    scatter(score(n_aver_valid+1:end, 1), score(n_aver_valid+1:end, 2), 100, ...
        [0.6, 1, 0.6], 'filled', 'MarkerEdgeColor', [0, 0.6, 0], 'LineWidth', 1.5);

    hold off;

    xlabel(sprintf('PC1 (%.1f%% variance)', explained(1)), 'FontSize', 11);
    ylabel(sprintf('PC2 (%.1f%% variance)', explained(2)), 'FontSize', 11);
    title('Session Clustering (PCA)', 'FontSize', 12, 'FontWeight', 'bold');
    legend({'Aversive', 'Reward'}, 'Location', 'best', 'FontSize', 10);
    grid on;
    set(gca, 'FontSize', 10);

    % Subplot 2: Variance explained
    subplot(1, 2, 2);
    bar(explained(1:min(10, length(explained))), 'FaceColor', [0.3, 0.6, 0.9]);
    xlabel('Principal Component', 'FontSize', 11);
    ylabel('Variance Explained (%)', 'FontSize', 11);
    title('PCA Variance Explained', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 10);

    sgtitle('Session Clustering Based on Behavioral Features', 'FontSize', 13, 'FontWeight', 'bold');

    fprintf('✓ Figure 3 created\n');
end

%% ========================================================================
%  SECTION 5: FIGURE 4 - SESSION PROFILES (RADAR PLOTS)
%  ========================================================================

fprintf('Creating Figure 4: Session Profiles (Radar Plots)...\n');

% Plot up to 6 sessions per type (12 total)
n_plot_aver = min(6, n_aversive);
n_plot_rew = min(6, n_reward);

figure('Position', [200, 200, 1600, 900], 'Name', 'Session Profiles');

for i = 1:n_plot_aver
    subplot(2, 6, i);

    % Combine behavior time budget and key matrix features
    profile_data = [aversive_sessions(i).behavior_time_percent; ...
                    aversive_sessions(i).matrix_feature_percent(4); ... % AtRewardPort
                    aversive_sessions(i).matrix_feature_percent(7)];    % GoalDirected

    % Normalize to 0-1 for radar plot
    profile_norm = profile_data / 100;

    % Create radar plot
    theta = linspace(0, 2*pi, length(profile_norm)+1);
    r = [profile_norm; profile_norm(1)];

    polarplot(theta, r, 'r-', 'LineWidth', 2);
    hold on;
    polarplot(theta, r, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
    hold off;

    title(sprintf('Aver Sess %d', i), 'FontSize', 10, 'FontWeight', 'bold');
    rlim([0, 1]);
end

for i = 1:n_plot_rew
    subplot(2, 6, 6 + i);

    % Combine behavior time budget and key matrix features
    profile_data = [reward_sessions(i).behavior_time_percent; ...
                    reward_sessions(i).matrix_feature_percent(4); ...
                    reward_sessions(i).matrix_feature_percent(7)];

    profile_norm = profile_data / 100;

    theta = linspace(0, 2*pi, length(profile_norm)+1);
    r = [profile_norm; profile_norm(1)];

    polarplot(theta, r, 'g-', 'LineWidth', 2);
    hold on;
    polarplot(theta, r, 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 6);
    hold off;

    title(sprintf('Rew Sess %d', i), 'FontSize', 10, 'FontWeight', 'bold');
    rlim([0, 1]);
end

sgtitle('Session Behavioral Profiles (Radar Plots)', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Figure 4 created\n');

%% ========================================================================
%  SECTION 6: FIGURE 5 - BREATHING RATE BY BEHAVIOR (INDIVIDUAL SESSIONS)
%  ========================================================================

fprintf('Creating Figure 5: Breathing Rate by Behavior - Individual Sessions...\n');

figure('Position', [250, 250, 1600, 900], 'Name', 'Breathing Rate by Behavior');

% Extract breathing by behavior
breathing_by_beh_aver = nan(n_aversive, n_behaviors);
breathing_by_beh_rew = nan(n_reward, n_behaviors);

for i = 1:n_aversive
    breathing_by_beh_aver(i, :) = aversive_sessions(i).breathing_by_behavior;
end

for i = 1:n_reward
    breathing_by_beh_rew(i, :) = reward_sessions(i).breathing_by_behavior;
end

% Create one subplot per behavior
for beh = 1:n_behaviors
    subplot(3, 3, beh);
    hold on;

    % Plot aversive sessions
    plot(1:n_aversive, breathing_by_beh_aver(:, beh), 'ro-', 'LineWidth', 1.5, ...
        'MarkerFaceColor', [1, 0.6, 0.6], 'MarkerSize', 8);

    % Plot reward sessions (offset on x-axis)
    plot(n_aversive + (1:n_reward), breathing_by_beh_rew(:, beh), 'go-', 'LineWidth', 1.5, ...
        'MarkerFaceColor', [0.6, 1, 0.6], 'MarkerSize', 8);

    % Add vertical separator
    plot([n_aversive + 0.5, n_aversive + 0.5], ylim, 'k--', 'LineWidth', 2);

    hold off;

    xlabel('Session', 'FontSize', 10);
    ylabel('Breathing Rate (Hz)', 'FontSize', 10);
    title(behavior_names{beh}, 'FontSize', 11, 'FontWeight', 'bold');
    xlim([0, n_total + 1]);
    ylim([0, 15]);
    grid on;
    set(gca, 'FontSize', 9);

    if beh == 1
        legend({'Aversive', 'Reward'}, 'Location', 'northeast', 'FontSize', 8);
    end
end

sgtitle('Breathing Rate During Each Behavior (Individual Sessions)', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Figure 5 created\n');

%% ========================================================================
%  SECTION 7: FIGURE 6 - BREATHING RATE HEATMAP
%  ========================================================================

fprintf('Creating Figure 6: Breathing Rate Heatmap...\n');

figure('Position', [300, 300, 1400, 800], 'Name', 'Breathing Rate Heatmap');

% Subplot 1: Aversive heatmap
subplot(2, 1, 1);
imagesc(breathing_by_beh_aver');
colormap(jet);
colorbar;
caxis([0, 12]);
yticks(1:n_behaviors);
yticklabels(behavior_names);
xlabel('Session', 'FontSize', 11);
ylabel('Behavior', 'FontSize', 11);
title(sprintf('Aversive Sessions - Breathing Rate by Behavior (n=%d)', n_aversive), ...
    'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 10);

% Subplot 2: Reward heatmap
subplot(2, 1, 2);
imagesc(breathing_by_beh_rew');
colormap(jet);
colorbar;
caxis([0, 12]);
yticks(1:n_behaviors);
yticklabels(behavior_names);
xlabel('Session', 'FontSize', 11);
ylabel('Behavior', 'FontSize', 11);
title(sprintf('Reward Sessions - Breathing Rate by Behavior (n=%d)', n_reward), ...
    'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 10);

sgtitle('Breathing Rate Heatmap: Sessions × Behaviors', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Figure 6 created\n');

%% ========================================================================
%  SECTION 8: FIGURE 7 - SESSION-LEVEL BREATHING DISTRIBUTIONS
%  ========================================================================

fprintf('Creating Figure 7: Session-Level Breathing Distributions...\n');

figure('Position', [350, 350, 1400, 600], 'Name', 'Session-Level Breathing');

% Subplot 1: Overall mean breathing per session
subplot(1, 2, 1);
hold on;

x_aver = 1:n_aversive;
x_rew = n_aversive + (1:n_reward);

scatter(x_aver, breathing_aver, 100, [1, 0.6, 0.6], 'filled', ...
    'MarkerEdgeColor', [0.8, 0, 0], 'LineWidth', 1.5);
scatter(x_rew, breathing_rew, 100, [0.6, 1, 0.6], 'filled', ...
    'MarkerEdgeColor', [0, 0.6, 0], 'LineWidth', 1.5);

plot([n_aversive + 0.5, n_aversive + 0.5], ylim, 'k--', 'LineWidth', 2);

% Add mean lines
yline(nanmean(breathing_aver), 'r--', 'LineWidth', 2, 'DisplayName', 'Aver mean');
yline(nanmean(breathing_rew), 'g--', 'LineWidth', 2, 'DisplayName', 'Rew mean');

hold off;

xlabel('Session', 'FontSize', 11);
ylabel('Mean Breathing Rate (Hz)', 'FontSize', 11);
title('Overall Breathing Rate per Session', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'northeast', 'FontSize', 10);
xlim([0, n_total + 1]);
grid on;
set(gca, 'FontSize', 10);

% Subplot 2: Distribution of breathing across behaviors per session
subplot(1, 2, 2);

% Calculate variability (std across behaviors) per session
std_aver = nanstd(breathing_by_beh_aver, 0, 2);
std_rew = nanstd(breathing_by_beh_rew, 0, 2);

hold on;
scatter(x_aver, std_aver, 100, [1, 0.6, 0.6], 'filled', ...
    'MarkerEdgeColor', [0.8, 0, 0], 'LineWidth', 1.5);
scatter(x_rew, std_rew, 100, [0.6, 1, 0.6], 'filled', ...
    'MarkerEdgeColor', [0, 0.6, 0], 'LineWidth', 1.5);

plot([n_aversive + 0.5, n_aversive + 0.5], ylim, 'k--', 'LineWidth', 2);
hold off;

xlabel('Session', 'FontSize', 11);
ylabel('Breathing Rate Std Dev Across Behaviors (Hz)', 'FontSize', 11);
title('Breathing Variability Across Behaviors', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'northeast', 'FontSize', 10);
xlim([0, n_total + 1]);
grid on;
set(gca, 'FontSize', 10);

sgtitle('Session-Level Breathing Rate Analysis', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Figure 7 created\n');

%% ========================================================================
%  SECTION 9: SUMMARY
%  ========================================================================

fprintf('\n========================================\n');
fprintf('VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Figures created:\n');
fprintf('  1. Time Budget Analysis\n');
fprintf('  2. Behavioral Matrix State Occupancy\n');
fprintf('  3. Session Clustering (PCA)\n');
fprintf('  4. Session Profiles (Radar Plots)\n');
fprintf('  5. Breathing Rate by Behavior - Individual Sessions\n');
fprintf('  6. Breathing Rate Heatmap\n');
fprintf('  7. Session-Level Breathing Distributions\n');
fprintf('========================================\n');
fprintf('Key findings to look for:\n');
fprintf('  - Inter-session variability in behavior budgets\n');
fprintf('  - Breathing rate differences across behaviors\n');
fprintf('  - Session clustering patterns\n');
fprintf('  - Outlier sessions with unusual patterns\n');
fprintf('========================================\n');
