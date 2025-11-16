%% ========================================================================
%  SESSION CLUSTERING: Behavioral Profiles Across Time (Aversive + Reward)
%  ========================================================================
%
%  Goal: Identify distinct behavioral response profiles across session types
%  Method: Cluster sessions based on 28-dimensional behavioral percentage features
%          (7 behaviors × 4 periods)
%
%  Analysis reveals whether different animals show distinct behavioral
%  strategies and whether session type (aversive vs reward) clusters separately
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== SESSION CLUSTERING ANALYSIS: AVERSIVE + REWARD ===\n');
fprintf('Behavioral Profiles: 7 Behaviors × 4 Periods\n\n');

config = struct();
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
config.behavior_names_short = {'Rew', 'Wlk', 'Rer', 'Scn', 'Snf', 'Grm', 'Std'};
config.n_behaviors = 7;
config.n_periods = 4;
config.confidence_threshold = 0.3;
config.n_features = config.n_behaviors * config.n_periods;  % 28 features

%% ========================================================================
%  SECTION 2: LOAD AVERSIVE AND REWARD DATA
%  ========================================================================

fprintf('Loading session data...\n');

% Configuration
prediction_folder = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/BehaviorPrediction';
spike_folder = '/Volumes/My980Pro/Data/Struct_spike';

% Load sorting parameters
[T_sorted] = loadSortingParameters();

% Load aversive sessions
try
    [allfiles_aversive, ~, ~, ~] = selectFilesWithAnimalIDFiltering(spike_folder, 999, '2025*RewardAversive*.mat');
    sessions_aversive = loadSessionMetricsFromSpikeFiles(allfiles_aversive, T_sorted);
    prediction_sessions_aversive = loadBehaviorPredictionsFromSpikeFiles(allfiles_aversive, prediction_folder);
    fprintf('✓ Loaded aversive data: %d sessions\n', length(sessions_aversive));
catch ME
    fprintf('❌ Failed to load aversive data: %s\n', ME.message);
    return;
end

% Load reward sessions
try
    [allfiles_reward, ~, ~, ~] = selectFilesWithAnimalIDFiltering(spike_folder, 999, '2025*RewardSeeking*.mat');
    sessions_reward = loadSessionMetricsFromSpikeFiles(allfiles_reward, T_sorted);
    prediction_sessions_reward = loadBehaviorPredictionsFromSpikeFiles(allfiles_reward, prediction_folder);
    fprintf('✓ Loaded reward data: %d sessions\n\n', length(sessions_reward));
catch ME
    fprintf('❌ Failed to load reward data: %s\n', ME.message);
    return;
end

%% ========================================================================
%  SECTION 3: EXTRACT BEHAVIORAL PERCENTAGES (P1-P4)
%  ========================================================================

fprintf('Extracting behavioral percentage features...\n');

% Initialize storage
feature_matrix = [];  % Will be N_sessions × 28
session_info = struct('session_id', {}, 'session_type', {}, 'filename', {}, 'animal_id', {}, 'valid', {});

n_valid_sessions = 0;

% Process AVERSIVE sessions
fprintf('  Processing aversive sessions...\n');
for sess_idx = 1:length(sessions_aversive)
    session = sessions_aversive{sess_idx};
    Animal_id = regexp(session.filename, 'R[A-Z]\d{2}', 'match');

    % Check required fields
    if ~isfield(session, 'all_aversive_time') || ...
       ~isfield(session, 'NeuralTime') || ...
       ~isfield(session, 'TriggerMid') || ...
       sess_idx > length(prediction_sessions_aversive)
        continue;
    end

    aversive_times = session.all_aversive_time;
    if length(aversive_times) < 6
        continue;
    end

    n_valid_sessions = n_valid_sessions + 1;

    neural_time = session.NeuralTime;
    prediction_scores = prediction_sessions_aversive(sess_idx).prediction_scores;
    prediction_ind = 1:20:length(session.TriggerMid);
    prediction_ind = prediction_ind + 10;
    prediction_time = session.TriggerMid(prediction_ind);

    % Define period boundaries (P1-P4) based on aversive noise
    period_boundaries = [session.TriggerMid(1), ...
                         aversive_times(1:3)' + session.TriggerMid(1), ...
                         aversive_times(4) + session.TriggerMid(1)];

    % Initialize feature vector for this session
    session_features = zeros(1, config.n_features);

    % Process each period
    for period = 1:config.n_periods
        period_start = period_boundaries(period);
        period_end = period_boundaries(period + 1);

        % Find prediction windows in this period
        prediction_idx = prediction_time >= period_start & prediction_time < period_end;

        if sum(prediction_idx) < 10
            continue;
        end

        % Get predictions for this period
        predictions_in_period = prediction_scores(prediction_idx, :);

        % Find dominant behavior for each window
        [max_confidence, dominant_beh] = max(predictions_in_period, [], 2);

        % Filter by confidence threshold
        valid_mask = max_confidence > config.confidence_threshold;
        valid_dominant = dominant_beh(valid_mask);

        total_valid = sum(valid_mask);

        if total_valid > 0
            % Calculate percentage for each behavior
            for beh = 1:config.n_behaviors
                count = sum(valid_dominant == beh);
                percentage = (count / total_valid) * 100;

                % Feature index: (period-1)*7 + beh
                feature_idx = (period - 1) * config.n_behaviors + beh;
                session_features(feature_idx) = percentage;
            end
        end
    end

    % Store features for this session
    feature_matrix(end+1, :) = session_features;

    % Store session metadata
    session_info(n_valid_sessions).session_id = sess_idx;
    session_info(n_valid_sessions).session_type = 'Aversive';
    session_info(n_valid_sessions).filename = session.filename;
    session_info(n_valid_sessions).animal_id = Animal_id{1};
    session_info(n_valid_sessions).valid = true;
end

fprintf('    ✓ Processed %d aversive sessions\n', n_valid_sessions);

% Process REWARD sessions
fprintf('  Processing reward sessions...\n');
n_reward_start = n_valid_sessions;
for sess_idx = 1:length(sessions_reward)
    session = sessions_reward{sess_idx};
    Animal_id = regexp(session.filename, 'R[A-Z]\d{2}', 'match');

    % Check required fields
    if ~isfield(session, 'NeuralTime') || ...
       ~isfield(session, 'TriggerMid') || ...
       sess_idx > length(prediction_sessions_reward)
        continue;
    end

    n_valid_sessions = n_valid_sessions + 1;

    neural_time = session.NeuralTime;
    prediction_scores = prediction_sessions_reward(sess_idx).prediction_scores;
    prediction_ind = 1:20:length(session.TriggerMid);
    prediction_ind = prediction_ind + 10;
    prediction_time = session.TriggerMid(prediction_ind);

    % Define period boundaries (P1-P4) based on time
    time_boundaries = [0, 8*60, 16*60, 24*60, 30*60];  % seconds
    period_boundaries = [session.TriggerMid(1), ...
                         time_boundaries(2:end) + session.TriggerMid(1)];

    % Initialize feature vector for this session
    session_features = zeros(1, config.n_features);

    % Process each period
    for period = 1:config.n_periods
        period_start = period_boundaries(period);
        period_end = period_boundaries(period + 1);

        % Find prediction windows in this period
        prediction_idx = prediction_time >= period_start & prediction_time < period_end;

        if sum(prediction_idx) < 10
            continue;
        end

        % Get predictions for this period
        predictions_in_period = prediction_scores(prediction_idx, :);

        % Find dominant behavior for each window
        [max_confidence, dominant_beh] = max(predictions_in_period, [], 2);

        % Filter by confidence threshold
        valid_mask = max_confidence > config.confidence_threshold;
        valid_dominant = dominant_beh(valid_mask);

        total_valid = sum(valid_mask);

        if total_valid > 0
            % Calculate percentage for each behavior
            for beh = 1:config.n_behaviors
                count = sum(valid_dominant == beh);
                percentage = (count / total_valid) * 100;

                % Feature index: (period-1)*7 + beh
                feature_idx = (period - 1) * config.n_behaviors + beh;
                session_features(feature_idx) = percentage;
            end
        end
    end

    % Store features for this session
    feature_matrix(end+1, :) = session_features;

    % Store session metadata
    session_info(n_valid_sessions).session_id = sess_idx;
    session_info(n_valid_sessions).session_type = 'Reward';
    session_info(n_valid_sessions).filename = session.filename;
    session_info(n_valid_sessions).animal_id = Animal_id;
    session_info(n_valid_sessions).valid = true;
end

n_reward_sessions = n_valid_sessions - n_reward_start;
fprintf('    ✓ Processed %d reward sessions\n\n', n_reward_sessions);

fprintf('✓ Feature matrix: %d total sessions × %d features\n', size(feature_matrix, 1), size(feature_matrix, 2));
fprintf('  Aversive sessions: %d\n', n_reward_start);
fprintf('  Reward sessions: %d\n', n_reward_sessions);
fprintf('  Feature structure: [Beh1-7 P1, Beh1-7 P2, Beh1-7 P3, Beh1-7 P4]\n\n');

n_sessions = size(feature_matrix, 1);

%% ========================================================================
%  SECTION 4: PREPROCESSING
%  ========================================================================

fprintf('Preprocessing features...\n');

% Z-score normalization (standardize each feature across sessions)
feature_matrix_z = zscore(feature_matrix);

% PCA for dimensionality reduction and visualization
fprintf('Performing PCA...\n');
[coeff, score, latent, tsquared, explained] = pca(feature_matrix_z);

fprintf('✓ PCA complete\n');
fprintf('  First 3 PCs explain %.2f%% of variance\n', sum(explained(1:3)));
fprintf('  First 5 PCs explain %.2f%% of variance\n\n', sum(explained(1:5)));

%% ========================================================================
%  SECTION 5: DETERMINE OPTIMAL NUMBER OF CLUSTERS
%  ========================================================================

fprintf('Determining optimal number of clusters...\n');

max_k = min(8, floor(n_sessions / 2));  % Don't exceed n_sessions/2
wcss = zeros(max_k, 1);
silhouette_scores = zeros(max_k, 1);

% Elbow method (Within-cluster sum of squares)
fprintf('  Computing elbow curve...\n');
for k = 1:max_k
    if k == 1
        wcss(k) = sum(sum((feature_matrix_z - mean(feature_matrix_z, 1)).^2));
        continue;
    end
    [idx, C, sumd] = kmeans(feature_matrix_z, k, 'Replicates', 100, 'Display', 'off');
    wcss(k) = sum(sumd);
end

% Silhouette analysis
fprintf('  Computing silhouette scores...\n');
for k = 2:max_k
    [idx, ~] = kmeans(feature_matrix_z, k, 'Replicates', 100, 'Display', 'off');
    silhouette_vals = silhouette(feature_matrix_z, idx, 'sqeuclidean');
    silhouette_scores(k) = mean(silhouette_vals);
end

% Gap statistic
fprintf('  Computing gap statistic...\n');
try
    eva = evalclusters(feature_matrix_z, 'kmeans', 'gap', 'KList', 2:max_k);
    optimal_k_gap = eva.OptimalK;
    fprintf('✓ Gap statistic suggests k = %d\n', optimal_k_gap);
catch
    optimal_k_gap = 3;
    fprintf('  Gap statistic failed, defaulting to k = 3\n');
end

% Find elbow point (maximum second derivative)
if max_k >= 3
    wcss_diff2 = diff(diff(wcss));
    [~, elbow_k] = max(wcss_diff2);
    elbow_k = elbow_k + 1;  % Adjust for diff offset
    fprintf('  Elbow method suggests k = %d\n', elbow_k);
end

% Find best silhouette
[~, optimal_k_sil] = max(silhouette_scores);
fprintf('  Silhouette method suggests k = %d (score = %.3f)\n\n', optimal_k_sil, silhouette_scores(optimal_k_sil));

% Plot cluster evaluation metrics
fig_eval = figure('Position', [100, 100, 1400, 400]);

subplot(1, 3, 1);
plot(1:max_k, wcss, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
xlabel('Number of Clusters (k)', 'FontSize', 11);
ylabel('Within-Cluster Sum of Squares', 'FontSize', 11);
title('Elbow Method', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
if exist('elbow_k', 'var')
    hold on;
    plot(elbow_k, wcss(elbow_k), 'ro', 'MarkerSize', 12, 'LineWidth', 2);
    text(elbow_k, wcss(elbow_k), sprintf('  k=%d', elbow_k), 'FontSize', 10, 'Color', 'r');
end

subplot(1, 3, 2);
plot(2:max_k, silhouette_scores(2:max_k), 's-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'g');
xlabel('Number of Clusters (k)', 'FontSize', 11);
ylabel('Mean Silhouette Score', 'FontSize', 11);
title('Silhouette Analysis', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
hold on;
plot(optimal_k_sil, silhouette_scores(optimal_k_sil), 'ro', 'MarkerSize', 12, 'LineWidth', 2);
text(optimal_k_sil, silhouette_scores(optimal_k_sil), sprintf('  k=%d', optimal_k_sil), 'FontSize', 10, 'Color', 'r');

subplot(1, 3, 3);
try
    plot(eva);
    title('Gap Statistic', 'FontSize', 12, 'FontWeight', 'bold');
catch
    text(0.5, 0.5, 'Gap statistic not available', 'HorizontalAlignment', 'center');
    axis off;
end

sgtitle('Cluster Optimization Metrics', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 6: PERFORM CLUSTERING
%  ========================================================================

% Use optimal k (can be adjusted based on visual inspection)
optimal_k = optimal_k_gap;  % or use optimal_k_sil, elbow_k, or manually set

fprintf('=== PERFORMING K-MEANS CLUSTERING ===\n');
fprintf('Using k = %d clusters\n\n', optimal_k);

% K-means clustering with multiple replicates
rng(42);  % For reproducibility
[cluster_idx, centroids, sumd] = kmeans(feature_matrix_z, optimal_k, ...
                                         'Replicates', 1000, ...
                                         'Distance', 'sqeuclidean', ...
                                         'Display', 'final');

% Compute silhouette scores for chosen k
silhouette_vals = silhouette(feature_matrix_z, cluster_idx, 'sqeuclidean');
fprintf('Mean silhouette score: %.3f\n', mean(silhouette_vals));
fprintf('Silhouette range: [%.3f, %.3f]\n\n', min(silhouette_vals), max(silhouette_vals));

%% ========================================================================
%  SECTION 7: CHARACTERIZE CLUSTERS
%  ========================================================================

fprintf('=== CLUSTER CHARACTERISTICS ===\n\n');

cluster_profiles = struct();
cluster_colors = lines(optimal_k);

for cluster = 1:optimal_k

    cluster_mask = cluster_idx == cluster;
    n_sessions_in_cluster = sum(cluster_mask);

    fprintf('CLUSTER %d: %d sessions (%.1f%%)\n', cluster, n_sessions_in_cluster, ...
            100 * n_sessions_in_cluster / n_sessions);

    % Get session IDs in this cluster
    session_ids_in_cluster = find(cluster_mask);
    animal_ids_in_cluster = [session_info(session_ids_in_cluster).animal_id];
    fprintf('  Sessions: %s\n', mat2str(session_ids_in_cluster'));
    fprintf('  Animals: %s\n', strjoin(unique(animal_ids_in_cluster(:)), ', '));

    % Get feature matrix for this cluster (original scale, not z-scored)
    cluster_features = feature_matrix(cluster_mask, :);

    % Compute median and std for each behavior × period
    cluster_median = median(cluster_features, 1);  % 1×28
    cluster_std = std(cluster_features, 0, 1);

    % Reshape to [7 behaviors × 4 periods]
    cluster_median_matrix = reshape(cluster_median, config.n_behaviors, config.n_periods);
    cluster_std_matrix = reshape(cluster_std, config.n_behaviors, config.n_periods);

    % Store profiles
    cluster_profiles(cluster).median = cluster_median_matrix;
    cluster_profiles(cluster).std = cluster_std_matrix;
    cluster_profiles(cluster).n_sessions = n_sessions_in_cluster;
    cluster_profiles(cluster).session_ids = session_ids_in_cluster;
    cluster_profiles(cluster).animal_ids = animal_ids_in_cluster;
    cluster_profiles(cluster).color = cluster_colors(cluster, :);

    % Print top 3 behaviors by percentage in each period
    fprintf('  Behavioral profile:\n');
    for period = 1:config.n_periods
        [sorted_pct, sorted_idx] = sort(cluster_median_matrix(:, period), 'descend');
        fprintf('    P%d: ', period);
        for i = 1:3
            fprintf('%s (%.1f%%), ', config.behavior_names{sorted_idx(i)}, sorted_pct(i));
        end
        fprintf('\n');
    end
    fprintf('\n');
end

%% ========================================================================
%  SECTION 8: VISUALIZATION - PCA with Clusters
%  ========================================================================

fprintf('Creating visualizations...\n');

% FIGURE 2: PCA scatter plot with cluster colors
fig_pca = figure('Position', [100, 100, 1400, 500]);

% 2D PCA plot
subplot(1, 2, 1);
hold on;
for cluster = 1:optimal_k
    cluster_mask = cluster_idx == cluster;
    scatter(score(cluster_mask, 1), score(cluster_mask, 2), 100, ...
            cluster_profiles(cluster).color, 'filled', 'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.5, 'DisplayName', sprintf('Cluster %d (n=%d)', cluster, sum(cluster_mask)));
end
xlabel(sprintf('PC1 (%.1f%%)', explained(1)), 'FontSize', 11);
ylabel(sprintf('PC2 (%.1f%%)', explained(2)), 'FontSize', 11);
title('PCA: Sessions Colored by Cluster', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
axis equal;

% 3D PCA plot
subplot(1, 2, 2);
hold on;
for cluster = 1:optimal_k
    cluster_mask = cluster_idx == cluster;
    scatter3(score(cluster_mask, 1), score(cluster_mask, 2), score(cluster_mask, 3), 100, ...
             cluster_profiles(cluster).color, 'filled', 'MarkerEdgeColor', 'k', ...
             'LineWidth', 1.5, 'DisplayName', sprintf('Cluster %d', cluster));
end
xlabel(sprintf('PC1 (%.1f%%)', explained(1)), 'FontSize', 11);
ylabel(sprintf('PC2 (%.1f%%)', explained(2)), 'FontSize', 11);
zlabel(sprintf('PC3 (%.1f%%)', explained(3)), 'FontSize', 11);
title('PCA 3D: Sessions Colored by Cluster', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
view(45, 30);

sgtitle(sprintf('Aversive Session Clustering (k=%d)', optimal_k), 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 9: VISUALIZATION - Behavioral Trajectories by Cluster
%  ========================================================================

% FIGURE 3: Cluster profiles - behavioral trajectories
fig_profiles = figure('Position', [50, 50, 1800, 1000]);
ax = [];
for beh = 1:config.n_behaviors
    ax(end+1) = subplot(3, 3, beh);
    hold on;

    % Plot each cluster's trajectory for this behavior
    for cluster = 1:optimal_k
        profile = cluster_profiles(cluster).median(beh, :);  % 1×4
        plot(1:4, profile, 'o-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
             'Color', cluster_profiles(cluster).color, ...
             'MarkerFaceColor', cluster_profiles(cluster).color, ...
             'DisplayName', sprintf('Cluster %d (n=%d)', cluster, cluster_profiles(cluster).n_sessions));
    end

    title(config.behavior_names{beh}, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Period', 'FontSize', 11);
    ylabel('Percentage (%)', 'FontSize', 11);
    xticks(1:4);
    xticklabels({'P1', 'P2', 'P3', 'P4'});
    grid on;

    if beh == 1
        legend('Location', 'best', 'FontSize', 9);
    end
    hold off;
end
linkaxes([ax],'xy')
sgtitle(sprintf('Behavioral Profiles by Cluster (k=%d)', optimal_k), 'FontSize', 15, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 10: VISUALIZATION - Heatmaps for Each Cluster
%  ========================================================================

% FIGURE 4: Heatmaps
% fig_heatmaps = figure('Position', [100, 100, 400*optimal_k, 500]);
% 
% for cluster = 1:optimal_k
%     subplot(1, optimal_k, cluster);
% 
%     % Transpose to show periods as rows, behaviors as columns
%     imagesc(cluster_profiles(cluster).median');
%     colorbar;
%     caxis([0, 50]);  % Adjust based on your data range
% 
%     title(sprintf('Cluster %d (n=%d)', cluster, cluster_profiles(cluster).n_sessions), ...
%           'FontSize', 12, 'FontWeight', 'bold');
%     xlabel('Behavior', 'FontSize', 11);
%     ylabel('Period', 'FontSize', 11);
%     xticks(1:config.n_behaviors);
%     xticklabels(config.behavior_names_short);
%     xtickangle(45);
%     yticks(1:4);
%     yticklabels({'P1', 'P2', 'P3', 'P4'});
%     set(gca, 'FontSize', 10);
% end
% 
% sgtitle('Behavioral Percentage Heatmaps by Cluster', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 11: VISUALIZATION - Silhouette Plot
%  ========================================================================

% FIGURE 5: Silhouette plot
fig_silhouette = figure('Position', [100, 100, 800, 600]); hold on
[silh_vals, h] = silhouette(feature_matrix_z, cluster_idx, 'sqeuclidean');
xline(mean(silh_vals))

title(sprintf('Silhouette Plot (k=%d, mean=%.3f)', optimal_k, mean(silh_vals)), ...
      'FontSize', 13, 'FontWeight', 'bold');
xlabel('Silhouette Value', 'FontSize', 11);
ylabel('Cluster', 'FontSize', 11);

%% ========================================================================
%  SECTION 12: STATISTICAL COMPARISON BETWEEN CLUSTERS
%  ========================================================================

fprintf('\n=== STATISTICAL COMPARISON BETWEEN CLUSTERS ===\n\n');

p_values = nan(config.n_behaviors, config.n_periods);

for beh = 1:config.n_behaviors
    for period = 1:config.n_periods

        feature_idx = (period - 1) * config.n_behaviors + beh;
        feature_values = feature_matrix(:, feature_idx);

        % Kruskal-Wallis test (non-parametric ANOVA for multiple groups)
        [p_val, tbl, stats] = kruskalwallis(feature_values, cluster_idx, 'off');
        p_values(beh, period) = p_val;

        if p_val < 0.05
            fprintf('%s P%d: Significant difference between clusters (p=%.4f) ***\n', ...
                   config.behavior_names{beh}, period, p_val);

            % Post-hoc pairwise comparisons
            c = multcompare(stats, 'Display', 'off', 'CType', 'dunn-sidak');

            % Display significant pairwise comparisons
            for i = 1:size(c, 1)
                if c(i, 6) < 0.05  % Column 6 is p-value
                    fprintf('  Cluster %d vs %d: p=%.4f\n', c(i, 1), c(i, 2), c(i, 6));
                end
            end
        end
    end
end

if ~any(p_values(:) < 0.05)
    fprintf('No significant differences found between clusters.\n');
end

fprintf('\n');

%% ========================================================================
%  SECTION 13: CLUSTER COMPOSITION ANALYSIS
%  ========================================================================

fprintf('=== CLUSTER COMPOSITION BY ANIMAL ===\n\n');

% Create contingency table: Animals × Clusters
animal_ids = [session_info.animal_id];
unique_animals = unique(animal_ids);
n_animals = length(unique_animals);

contingency_table = zeros(n_animals, optimal_k);

for i = 1:n_animals
    animal_id = unique_animals(i);
    animal_mask = strcmp(animal_ids, animal_id);

    for cluster = 1:optimal_k
        contingency_table(i, cluster) = sum(animal_mask & cluster_idx' == cluster);
    end
end

fprintf('Contingency Table (Animals × Clusters):\n');
fprintf('Animal\t');
for cluster = 1:optimal_k
    fprintf('C%d\t', cluster);
end
fprintf('\n');

for i = 1:n_animals
    fprintf('%s\t', unique_animals{i});
    for cluster = 1:optimal_k
        fprintf('%d\t', contingency_table(i, cluster));
    end
    fprintf('\n');
end

fprintf('\n');

%% ========================================================================
%  SECTION 14: SESSION TYPE DISTRIBUTION ACROSS CLUSTERS
%  ========================================================================

fprintf('=== SESSION TYPE DISTRIBUTION ACROSS CLUSTERS ===\n\n');

% Count session types per cluster
session_types = {session_info.session_type};
cluster_session_type_counts = zeros(optimal_k, 2);  % [Aversive, Reward]

for cluster = 1:optimal_k
    cluster_mask = cluster_idx == cluster;
    cluster_session_types = session_types(cluster_mask);

    n_aversive = sum(strcmp(cluster_session_types, 'Aversive'));
    n_reward = sum(strcmp(cluster_session_types, 'Reward'));

    cluster_session_type_counts(cluster, 1) = n_aversive;
    cluster_session_type_counts(cluster, 2) = n_reward;

    fprintf('CLUSTER %d: %d Aversive, %d Reward\n', cluster, n_aversive, n_reward);
end

fprintf('\n');

% Visualize session type distribution
fig_session_types = figure('Position', [100, 100, 800, 500]);

subplot(1, 2, 1);
bar(cluster_session_type_counts, 'grouped');
xlabel('Cluster', 'FontSize', 11);
ylabel('Number of Sessions', 'FontSize', 11);
title('Session Type Distribution Across Clusters', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

subplot(1, 2, 2);
bar(cluster_session_type_counts, 'stacked');
xlabel('Cluster', 'FontSize', 11);
ylabel('Number of Sessions', 'FontSize', 11);
title('Session Type Composition (Stacked)', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

sgtitle('Aversive vs Reward Session Distribution in Clusters', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 15: SAVE RESULTS
%  ========================================================================

fprintf('Saving clustering results...\n');

results = struct();
results.config = config;
results.cluster_idx = cluster_idx;
results.cluster_profiles = cluster_profiles;
results.session_info = session_info;
results.feature_matrix = feature_matrix;
results.feature_matrix_z = feature_matrix_z;
results.pca_score = score;
results.pca_explained = explained;
results.optimal_k = optimal_k;
results.silhouette_vals = silhouette_vals;
results.cluster_session_type_counts = cluster_session_type_counts;

save('session_clustering_aversive_reward_results.mat', 'results');

fprintf('✓ Results saved to: session_clustering_aversive_reward_results.mat\n');
fprintf('\n=== ANALYSIS COMPLETE ===\n');
