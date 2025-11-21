%% ========================================================================
%  PCA CLUSTERING AND VISUALIZATION OF PSTH TIME SERIES
%  Hierarchical clustering on PCA-reduced PSTH time series features.
%% ========================================================================
%
%  This script clusters units based on their PSTH time series responses.
%  1. Loads data from Unit_Feature_Extraction_PSTH_TimeSeries.m
%  2. Handles NaN values resulting from missing events (e.g., AversiveOnset
%     in Reward sessions) by imputing them with the feature mean.
%  3. Performs PCA on the high-dimensional PSTH feature space.
%  4. Performs hierarchical clustering on the first N principal components.
%  5. Saves the cluster assignments.
%  6. Visualizes the results by plotting the mean PSTH trace for each
%     cluster, for each event type.
%
%  Input: unit_features_for_clustering_PSTH_TimeSeries.mat
%  Output: Cluster assignments and summary figures of mean PSTH traces.
%
%% ========================================================================

clear all;
close all;

%% ========================================================================
%  SECTION 1: CONFIGURATION
%% ========================================================================

fprintf('=== CLUSTERING & VISUALIZATION OF PSTH TIME SERIES ===\n\n');

% ========== MAIN CONFIGURATION ==========
config = struct();

% Clustering configuration
config.separate_by_session = true;   % Analyze Aversive/Reward separately
config.normalize_features = true;    % Z-score normalize each feature (already z-scored, but good practice)
config.n_principal_components = 4;  % Number of PCs to use for clustering

% Session-specific cluster thresholds (distance threshold for cluster assignment)
% These may need tuning for the PCA-reduced space.
config.cluster_threshold_aversive = 115; % Lowered for PCA space
config.cluster_threshold_reward = 115;   % Lowered for PCA space

% Visualization configuration
config.colormap_name = 'bluewhitered';
config.show_dendrogram = true;

fprintf('Configuration:\n');
fprintf('  Separate by session: %s\n', mat2str(config.separate_by_session));
fprintf('  Normalize features: %s\n', mat2str(config.normalize_features));
fprintf('  Num Principal Components: %d\n', config.n_principal_components);
fprintf('  Cluster threshold - Aversive: %.1f, Reward: %.1f\n', ...
    config.cluster_threshold_aversive, config.cluster_threshold_reward);
fprintf('\n');

%% ========================================================================
%  SECTION 2: LOAD DATA
%% ========================================================================

fprintf('Loading PSTH time series feature data...\n');

try
    loaded = load('unit_features_for_clustering_PSTH_TimeSeries.mat');
    results = loaded.results;
    master_features = results.master_features;
    feature_names = results.feature_names;
    fprintf('✓ Loaded: %d units × %d features\n\n', results.n_units, results.n_features);
catch ME
    fprintf('❌ Failed to load: %s\n', ME.message);
    fprintf('Please run Unit_Feature_Extraction_PSTH_TimeSeries.m first\n');
    return;
end

%% ========================================================================
%  SECTION 3: PREPARE FEATURE MATRIX
%% ========================================================================

fprintf('Preparing feature matrix...\n');

% Extract metadata columns
n_units = height(master_features);
session_ids = master_features.Session;
unique_unit_ids = master_features.UniqueUnitID;
session_types = master_features.SessionType;
is_aversive = contains(string(session_types), 'Aversive');

% Extract feature columns
feature_col_start = 5;
feature_matrix = table2array(master_features(:, feature_col_start:end));

% Categorize features (all are PSTH in this version)
feature_categories = repmat({'PSTH'}, 1, length(feature_names));

fprintf('✓ Feature matrix prepared\n');
fprintf('  Units: %d, Features: %d\n', n_units, length(feature_names));
fprintf('  Aversive units: %d, Reward units: %d\n\n', sum(is_aversive), sum(~is_aversive));

%% ========================================================================
%  SECTION 4: NORMALIZE FEATURES (Optional but Recommended)
%% ========================================================================

if config.normalize_features
    fprintf('Normalizing features (z-score)...\n');
    % Although input is z-scored PSTH, normalizing across units can be beneficial
    feature_matrix_norm = nan(size(feature_matrix));

    for f = 1:size(feature_matrix, 2)
        feature_col = feature_matrix(:, f);
        valid_data = feature_col(~isnan(feature_col));

        if ~isempty(valid_data) && std(valid_data) > 0
            feature_matrix_norm(:, f) = (feature_col - mean(valid_data)) / std(valid_data);
        else
            feature_matrix_norm(:, f) = feature_col; % Keep as is if no variance
        end
    end

    feature_matrix = feature_matrix_norm;
    fprintf('✓ Features normalized\n\n');
end

%% ========================================================================
%  SECTION 5: PCA-BASED CLUSTERING (MAIN ANALYSIS)
%% ========================================================================

fprintf('=== PCA-BASED CLUSTERING ON PSTH TIME SERIES ===\n\n');

if config.separate_by_session
    session_types_to_analyze = {'Aversive', 'Reward'};
else
    session_types_to_analyze = {'Combined'};
end

% Store clustering results for the final visualization step
clustering_results = struct();

for sess_type_idx = 1:length(session_types_to_analyze)
    sess_name = session_types_to_analyze{sess_type_idx};

    fprintf('--- Processing %s sessions (PCA) ---\n', sess_name);

    % Select units for the current session type
    if strcmp(sess_name, 'Aversive')
        unit_mask = is_aversive;
    elseif strcmp(sess_name, 'Reward')
        unit_mask = ~is_aversive;
    else
        unit_mask = true(n_units, 1);
    end

    sess_feature_matrix = feature_matrix(unit_mask, :);
    n_sess_units = sum(unit_mask);
    fprintf('  Units: %d\n', n_sess_units);

    % --- HANDLE NaN VALUES ---
    % This is critical for PCA. We will replace NaNs in each feature column
    % with the mean of that column (imputation), with a fallback to zero
    % if the entire column is NaN.
    fprintf('  Imputing NaN values for PCA...\n');
    n_features_imputed_mean = 0;
    n_features_imputed_zero = 0;
    for f = 1:size(sess_feature_matrix, 2)
        col = sess_feature_matrix(:, f);
        if any(isnan(col))
            mean_val = nanmean(col);
            if isnan(mean_val) % If mean_val is NaN, the entire column was NaN
                col(isnan(col)) = 0; % Impute with zero (assumed baseline for non-existent event)
                n_features_imputed_zero = n_features_imputed_zero + 1;
            else
                col(isnan(col)) = mean_val; % Impute with column mean
                n_features_imputed_mean = n_features_imputed_mean + 1;
            end
            sess_feature_matrix(:, f) = col;
        end
    end
    fprintf('  ✓ Imputed NaNs: %d features with mean, %d features with zero.\n', n_features_imputed_mean, n_features_imputed_zero);

    % Perform PCA on the imputed feature matrix
    fprintf('  Performing PCA...\n');
    n_pcs = min(config.n_principal_components, n_sess_units - 1); % Cannot have more PCs than units
    [coeff, score, latent, ~, explained] = pca(sess_feature_matrix, 'NumComponents', n_pcs);
    fprintf('  Variance explained by first %d PCs: %.1f%%\n', n_pcs, sum(explained(1:n_pcs)));

    % Use principal components for clustering
    pca_features = score(:, 1:n_pcs);

    % Select session-specific cluster threshold
    if strcmp(sess_name, 'Aversive')
        cluster_threshold = config.cluster_threshold_aversive;
    elseif strcmp(sess_name, 'Reward')
        cluster_threshold = config.cluster_threshold_reward;
    else
        cluster_threshold = config.cluster_threshold_aversive; % Default
    end

    % Perform hierarchical clustering on PCA features
    [unit_sort_idx, linkage_tree, cluster_assignments] = perform_clustering(...
        pca_features, cluster_threshold);

    n_clusters = max(cluster_assignments(~isnan(cluster_assignments)));
    if isempty(n_clusters), n_clusters = 0; end
    fprintf('  Found %d clusters with threshold %.1f\n', n_clusters, cluster_threshold);

    % --- Store results for this session type ---
    clustering_results.(sess_name).cluster_assignments = cluster_assignments;
    clustering_results.(sess_name).n_clusters = n_clusters;
    clustering_results.(sess_name).unit_mask = unit_mask;
    clustering_results.(sess_name).unique_unit_ids = unique_unit_ids(unit_mask);
    clustering_results.(sess_name).session_names = session_ids(unit_mask);
    clustering_results.(sess_name).unit_numbers = master_features.Unit(unit_mask);

    % Create PCA-based heatmap
    create_pca_heatmap(pca_features, unit_sort_idx, linkage_tree, ...
        cluster_assignments, is_aversive(unit_mask), sess_name, explained(1:n_pcs), ...
        config, cluster_threshold);
    
    % Analyze and visualize the composition of clusters by recording session
    sess_session_ids = session_ids(unit_mask);
    analyze_cluster_composition(cluster_assignments, sess_session_ids, ...
        [sess_name '_PSTH_TimeSeries'], linkage_tree, config, cluster_threshold);

    % The loadings heatmap is too large to be easily interpretable with thousands of features,
    % but can be enabled for diagnostics.
    % create_pca_loadings_heatmap(coeff, feature_names, explained(1:n_pcs), ...
    %     [sess_name '_PSTH_TimeSeries'], feature_categories);

    fprintf('✓ PCA-based %s clustering complete\n\n', sess_name);
end

% Save cluster assignments
fprintf('Saving cluster assignments...\n');
cluster_assignments_output = struct();
cluster_assignments_output.config = config;
cluster_assignments_output.timestamp = datetime('now');

for sess_type_idx = 1:length(session_types_to_analyze)
    sess_name = session_types_to_analyze{sess_type_idx};
    if ~isfield(clustering_results, sess_name), continue; end

    unit_cluster_table = table(...
        clustering_results.(sess_name).unique_unit_ids, ...
        clustering_results.(sess_name).session_names, ...
        clustering_results.(sess_name).unit_numbers, ...
        clustering_results.(sess_name).cluster_assignments, ...
        'VariableNames', {'UniqueUnitID', 'Session', 'Unit', 'ClusterID'});
    
    cluster_assignments_output.(sess_name) = unit_cluster_table;
end

save('unit_cluster_assignments_PSTH_TimeSeries.mat', 'cluster_assignments_output');
fprintf('✓ Saved to: unit_cluster_assignments_PSTH_TimeSeries.mat\n\n');


%% ========================================================================
%  SECTION 6: VISUALIZE CLUSTER PSTH TRACES
%  ========================================================================
% fprintf('=== VISUALIZING MEAN PSTH FOR EACH CLUSTER ===\n\n');
% 
% % Load original PSTH data required for plotting full traces
% fprintf('Loading original PSTH survey data for visualization...\n');
% if ~exist('PSTH_Survey_Results.mat', 'file')
%     error('PSTH_Survey_Results.mat not found! Cannot generate visualizations.');
% end
% psth_survey_loaded = load('PSTH_Survey_Results.mat', 'results');
% psth_survey_data = psth_survey_loaded.results;
% fprintf('✓ PSTH survey data loaded.\n\n');
% 
% for sess_type_idx = 1:length(session_types_to_analyze)
%     sess_name = session_types_to_analyze{sess_type_idx};
%     fprintf('--- Creating summary plot for %s sessions ---\n', sess_name);
% 
%     if ~isfield(clustering_results, sess_name) || clustering_results.(sess_name).n_clusters == 0
%         fprintf('  No clusters found for %s sessions, skipping plot.\n\n', sess_name);
%         continue;
%     end
%     
%     create_cluster_psth_summary_plot(cluster_assignments_output.(sess_name), ...
%         psth_survey_data, sess_name);
% 
%     fprintf('✓ Summary plot created for %s sessions.\n\n', sess_name);
% end


%% ========================================================================
%  SECTION 7: SUMMARY
%% ========================================================================

fprintf('\n========================================\n');
fprintf('PSTH CLUSTERING & VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Total units analyzed: %d\n', n_units);
fprintf('Total features used for PCA: %d\n', length(feature_names));
fprintf('\nFiles saved:\n');
fprintf('  - unit_cluster_assignments_PSTH_TimeSeries.mat\n');
fprintf('\nFigures created:\n');
fprintf('  - PCA heatmaps for each session type\n');
fprintf('  - Cluster PSTH summary plots for each session type\n');
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS
%% ========================================================================

function feature_categories = categorize_features(feature_names)
% In this script, all features are PSTH time points.
    feature_categories = cell(size(feature_names));
    for i = 1:length(feature_names)
        feature_categories{i} = 'PSTH';
    end
end

function [unit_sort_idx, linkage_tree, cluster_assignments] = perform_clustering(feature_matrix, cluster_threshold)
% Perform hierarchical clustering on feature matrix

    n_units = size(feature_matrix, 1);

    % Data should already be imputed, but as a safeguard:
    valid_units = sum(~isnan(feature_matrix), 2) > size(feature_matrix, 2) * 0.5;
    if sum(valid_units) < 3
        fprintf('    WARNING: Not enough valid units for clustering, using original order.\n');
        unit_sort_idx = 1:n_units;
        linkage_tree = [];
        cluster_assignments = nan(n_units, 1);
        return;
    end
    feature_matrix_clean = feature_matrix(valid_units, :);

    % Compute distance and linkage
    distances = pdist(feature_matrix_clean, 'euclidean');
    linkage_tree = linkage(distances, 'ward');

    % Get dendrogram order for plotting WITHOUT showing a plot
    if n_units > 1
        % Create a temporary invisible figure to prevent plot interference
        temp_fig = figure('Visible', 'off');
        try
            [~, ~, sort_idx_clean] = dendrogram(linkage_tree, 0);
        catch ME
            close(temp_fig);
            rethrow(ME);
        end
        close(temp_fig); % Close the invisible figure immediately
    else
        sort_idx_clean = 1;
    end
    
    % Assign clusters based on distance cutoff
    if ~isempty(linkage_tree)
        cluster_assignments_clean = cluster(linkage_tree, 'cutoff', cluster_threshold, 'criterion', 'distance');
    else
        cluster_assignments_clean = ones(n_units,1);
    end

    % Map back to original indices
    valid_idx = find(valid_units);
    unit_sort_idx = valid_idx(sort_idx_clean);
    cluster_assignments = nan(n_units, 1);
    cluster_assignments(valid_idx) = cluster_assignments_clean;
end

function create_pca_heatmap(pca_features, unit_sort_idx, linkage_tree, ...
    cluster_assignments, is_aversive, session_name, explained_variance, ...
    config, cluster_threshold)

    n_pcs = size(pca_features, 2);
    pca_features_sorted = pca_features(unit_sort_idx, :);
    is_aversive_sorted = is_aversive(unit_sort_idx);
    cluster_assignments_sorted = cluster_assignments(unit_sort_idx);

    n_clusters = max(cluster_assignments(~isnan(cluster_assignments)));
    if isempty(n_clusters), n_clusters = 0; end

    fig = figure('Position', [100 100 1200 900], 'Name', sprintf('%s - PSTH PCA Clustering', session_name));
    
    heatmap_left = 0.08;
    if ~isempty(linkage_tree) && config.show_dendrogram
        ax_dendro = axes('Position', [0.05 0.15 0.10 0.75]); % Explicitly create axes
        dendrogram(ax_dendro, linkage_tree, 0, 'Orientation', 'left', 'ColorThreshold', cluster_threshold); % Pass axes handle
        set(ax_dendro, 'YDir', 'reverse', 'XTickLabel', []);
        ylabel(ax_dendro, 'Units'); % Pass axes handle to label functions
        title(ax_dendro, 'Clustering'); % Pass axes handle to title function
        heatmap_left = 0.18;
    end

    if n_clusters > 0
        subplot('Position', [heatmap_left - 0.02 0.15 0.015 0.75]);
        cluster_colormap = lines(n_clusters);
        cluster_colors = nan(length(unit_sort_idx), 1, 3);
        for i = 1:length(unit_sort_idx)
            cluster_id = cluster_assignments_sorted(i);
            if ~isnan(cluster_id)
                cluster_colors(i, 1, :) = cluster_colormap(cluster_id, :);
            end
        end
        image(cluster_colors);
        set(gca, 'XTick', [], 'YTick', []);
    end

    heatmap_width = 0.50;
    subplot('Position', [heatmap_left 0.15 heatmap_width 0.75]);
    imagesc(pca_features_sorted);
    colormap(gca, bluewhitered(256));
    data_range = prctile(pca_features_sorted(:), [1 99]);
    clim_val = max(abs(data_range));
    caxis([-clim_val clim_val]);
    axis tight;
    cb = colorbar; ylabel(cb, 'PC Score');
    ylabel('Units'); xlabel('Principal Components');

    pc_labels = cell(n_pcs, 1);
    for i = 1:n_pcs, pc_labels{i} = sprintf('PC%%d (%%.%df%%)', i, explained_variance(i)); end
    set(gca, 'XTick', 1:n_pcs, 'XTickLabel', pc_labels, 'YTick', [], 'XTickLabelRotation', 45);

    title_str = sprintf('%s: PCA-Based Clustering on PSTH Time Series\n%d units | %d clusters', session_name, size(pca_features_sorted, 1), n_clusters);
    sgtitle(title_str, 'FontSize', 14, 'FontWeight', 'bold');

    fig_filename = sprintf('PCA_Heatmap_%s_PSTH_TimeSeries.png', session_name);
    saveas(fig, fig_filename);
    fprintf('  Saved PCA heatmap: %s\n', fig_filename);
end

function create_cluster_psth_summary_plot(cluster_table, psth_survey_data, session_name)
% Creates a grid plot showing the mean PSTH trace for each cluster for each event type.

    % --- 1. Match unit data ---
    unit_data = psth_survey_data.unit_data;
    time_centers = psth_survey_data.time_centers;
    
    all_fields = fieldnames(unit_data);
    zscore_fields = all_fields(contains(all_fields, '_zscore'));
    event_types = cellfun(@(x) strrep(x, '_zscore', ''), zscore_fields, 'UniformOutput', false);

    matched_units = [];
    for i = 1:height(cluster_table)
        unique_id = cluster_table.UniqueUnitID{i};
        cluster_id = cluster_table.ClusterID(i);
        
        for u = 1:length(unit_data)
            [~, session_base, ext] = fileparts(unit_data(u).session_name);
            psth_unique_id = sprintf('%s%s_Unit%d', session_base, ext, unit_data(u).unit_id);
            
            if strcmp(psth_unique_id, unique_id)
                matched_unit.psth_index = u;
                matched_unit.cluster_id = cluster_id;
                matched_units = [matched_units, matched_unit];
                break;
            end
        end
    end

    % --- 2. Organize by cluster ---
    cluster_ids = unique([matched_units.cluster_id]);
    cluster_ids = cluster_ids(~isnan(cluster_ids));
    n_clusters = length(cluster_ids);
    if n_clusters == 0, return; end

    cluster_psth_analysis = struct();
    for c_idx = 1:n_clusters
        cluster_id = cluster_ids(c_idx);
        psth_indices = [matched_units([matched_units.cluster_id] == cluster_id).psth_index];
        
        cluster_psth_analysis.(['C' num2str(cluster_id)]).n_units = length(psth_indices);
        
        for e_idx = 1:length(event_types)
            event_type = event_types{e_idx};
            zscore_matrix = [];
            for p_idx = psth_indices
                trace = unit_data(p_idx).([event_type '_zscore']);
                if ~isempty(trace)
                    zscore_matrix = [zscore_matrix; trace(:)'];
                end
            end
            
            if ~isempty(zscore_matrix)
                cluster_psth_analysis.(['C' num2str(cluster_id)]).([event_type '_mean']) = mean(zscore_matrix, 1, 'omitnan');
                cluster_psth_analysis.(['C' num2str(cluster_id)]).([event_type '_sem']) = std(zscore_matrix, 0, 1, 'omitnan') / sqrt(size(zscore_matrix, 1));
            else
                cluster_psth_analysis.(['C' num2str(cluster_id)]).([event_type '_mean']) = nan(size(time_centers));
                cluster_psth_analysis.(['C' ' num2str(cluster_id)']).([event_type '_sem']) = nan(size(time_centers));
            end
        end
    end

    % --- 3. Create plot ---
    n_events = length(event_types);
    fig = figure('Position', [50 50 250*n_events 200*n_clusters]);
    sgtitle(sprintf('%s Sessions: Mean PSTH by Cluster', session_name), 'FontSize', 16, 'FontWeight', 'bold');
    
    cluster_colors = lines(n_clusters);
    plot_idx = 1;
    
    for c_idx = 1:n_clusters
        cluster_id = cluster_ids(c_idx);
        cluster_results = cluster_psth_analysis.(['C' num2str(cluster_id)]);

        for e_idx = 1:n_events
            event_type = event_types{e_idx};
            mean_trace = cluster_results.([event_type '_mean']);
            sem_trace = cluster_results.([event_type '_sem']);
            
            subplot(n_clusters, n_events, plot_idx);
            hold on;
            
            if ~all(isnan(mean_trace))
                fill([time_centers, fliplr(time_centers)], [mean_trace - sem_trace, fliplr(mean_trace + sem_trace)], ...
                     cluster_colors(c_idx, :), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
                plot(time_centers, mean_trace, 'Color', cluster_colors(c_idx, :), 'LineWidth', 2);
            end
            
            xline(0, 'k--', 'LineWidth', 1);
            yline(0, 'k:', 'LineWidth', 0.5);
            grid on;
            xlim([-2, 2]);
            ylim([-1, 3]); % Fixed Y-lim for comparison

            if e_idx == 1
                ylabel(sprintf('Cluster %d (n=%d)', cluster_id, cluster_results.n_units), 'FontWeight', 'bold', 'Color', cluster_colors(c_idx, :));
            end
            if c_idx == 1
                title(strrep(event_type, '_', ' '), 'Interpreter', 'none');
            end
            if c_idx == n_clusters
                xlabel('Time (s)');
            end
            
            plot_idx = plot_idx + 1;
        end
    end

    fig_filename = sprintf('PSTH_TimeSeries_Cluster_Summary_%s.png', session_name);
    saveas(fig, fig_filename);
    fprintf('  Saved summary plot: %s\n', fig_filename);
end

function cmap = bluewhitered(n)
% Create blue-white-red colormap
    if nargin < 1, n = 256; end
    half = ceil(n/2);
    r1 = linspace(0, 1, half)'; g1 = linspace(0, 1, half)'; b1 = ones(half, 1);
    r2 = ones(half, 1); g2 = linspace(1, 0, half)'; b2 = linspace(1, 0, half)';
    cmap = [r1 g1 b1; r2 g2 b2];
    cmap = cmap(1:n, :);
end

function analyze_cluster_composition(cluster_assignments, session_ids, session_name, linkage_tree, config, cluster_threshold)
% Analyze and visualize cluster composition by session

    valid_clusters = ~isnan(cluster_assignments);
    cluster_assignments = cluster_assignments(valid_clusters);
    session_ids = session_ids(valid_clusters);

    n_clusters = max(cluster_assignments);
    if isempty(n_clusters) || n_clusters == 0
        fprintf('  No valid clusters to analyze for composition.\n');
        return;
    end
    
    unique_sessions = unique(session_ids);
    n_sessions = length(unique_sessions);

    % Build session × cluster matrix
    session_cluster_matrix = zeros(n_sessions, n_clusters);

    for s = 1:n_sessions
        for c = 1:n_clusters
            session_mask = ismember(session_ids,unique_sessions(s));
            cluster_mask = cluster_assignments == c;
            session_cluster_matrix(s, c) = sum(session_mask & cluster_mask);
        end
    end

    % Create figure
    fig = figure('Position', [100 100 1600 800], ...
        'Name', sprintf('%s - Cluster Composition', session_name));

    % Cluster color bar (left of heatmap)
    subplot('Position', [0.12 0.15 0.015 0.70]);
    cluster_colormap = lines(n_clusters);
    cluster_img = zeros(n_clusters, 1, 3);
    for c = 1:n_clusters
        cluster_img(c, 1, :) = cluster_colormap(c, :);
    end
    image(cluster_img);
    set(gca, 'XTick', [], 'YTick', 1:n_clusters);
    set(gca, 'YDir', 'normal');
    ylabel('Cluster ID', 'FontSize', 10, 'FontWeight', 'bold');

    % Heatmap
    subplot('Position', [0.15 0.15 0.60 0.70]);
    imagesc(session_cluster_matrix');
    colormap(hot);
    cb = colorbar;
    ylabel(cb, 'Unit Count', 'FontSize', 10);

    xlabel('Session ID', 'FontSize', 11, 'FontWeight', 'bold');
    title('Units per Session per Cluster', 'FontSize', 12, 'FontWeight', 'bold');

    set(gca, 'XTick', 1:n_sessions);
    set(gca, 'XTickLabel', cellstr(string(unique_sessions)));
    set(gca, 'YTick', 1:n_clusters);
    set(gca, 'YDir', 'normal');
    axis tight;

    % Add count annotations
    for s = 1:n_sessions
        for c = 1:n_clusters
            count = session_cluster_matrix(s, c);
            if count > 0
                if count > max(session_cluster_matrix(:)) * 0.5
                    text_color = 'white';
                else
                    text_color = 'black';
                end
                text(s, c, num2str(count), 'HorizontalAlignment', 'center', ...
                    'Color', text_color, 'FontSize', 9, 'FontWeight', 'bold');
            end
        end
    end

    % Bar chart
    subplot('Position', [0.80 0.15 0.15 0.70]);
    units_per_session = sum(session_cluster_matrix, 2);
    barh(1:n_sessions, units_per_session, 'FaceColor', [0.5 0.5 0.8]);
    xlabel('Total Units', 'FontSize', 11);
    ylabel('Session ID', 'FontSize', 11);
    set(gca, 'YTick', 1:n_sessions);
    set(gca, 'YTickLabel', unique_sessions);
    set(gca, 'YDir', 'normal');
    grid on;

    % Overall title
    sgtitle(sprintf('%s: %d clusters, %d sessions', session_name, n_clusters, n_sessions), ...
        'FontSize', 14, 'FontWeight', 'bold');

    % Save figure
    fig_filename = sprintf('Cluster_Composition_%s.png', session_name);
    saveas(fig, fig_filename);
    fprintf('  Saved: %s\n', fig_filename);
end