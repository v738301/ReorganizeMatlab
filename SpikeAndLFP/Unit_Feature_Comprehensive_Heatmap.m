%% ========================================================================
%  COMPREHENSIVE UNIT CLUSTERING AND VISUALIZATION
%  Hierarchical clustering and heatmap visualization of all unit features
%% ========================================================================
%
%  This script creates comprehensive visualizations of unit features for clustering:
%  - All 23 spike train metrics
%  - PPC, Coherence, Amplitude Correlation (LFP + Breathing) at 2 freq bands
%  - PSTH responses to various events
%
%  Clustering options:
%  1. Hierarchical clustering (Ward's method)
%  2. Separate analysis for Aversive vs Reward sessions
%  3. Simplified clustering with key features only
%
%  Input: unit_features_for_clustering.mat (from Unit_Feature_Extraction.m)
%  Output: Comprehensive heatmap figures with dendrograms
%
%% ========================================================================

clear all;
close all;

%% ========================================================================
%  SECTION 1: CONFIGURATION
%% ========================================================================

fprintf('=== COMPREHENSIVE UNIT CLUSTERING AND VISUALIZATION ===\n\n');

% ========== MAIN CONFIGURATION ==========
config = struct();

% Clustering configuration
config.separate_by_session = true;   % Analyze Aversive/Reward separately
config.normalize_features = true;    % Z-score normalize each feature

% Session-specific cluster thresholds (distance threshold for cluster assignment)
config.cluster_threshold_aversive = 55;  % Aversive sessions (lower = more clusters)
config.cluster_threshold_reward = 55;     % Reward sessions (lower = more clusters)

% Visualization configuration
config.colormap_name = 'bluewhitered';  % 'bluewhitered', 'jet', 'parula'
config.show_dendrogram = true;       % Show hierarchical clustering dendrogram

% Feature selection configuration
% config.exclude_categories = {'Spike_Basic','Spike_Temporal','PPC','MutualInfo','Coherence'};      % e.g., {'PSTH'} to exclude PSTH features
config.exclude_categories = [];

% ========== SIMPLIFIED CLUSTERING CONFIGURATION ==========
% Controls which features are used for simplified clustering (Section 8)
config.use_data_driven_features = true;  % true = use top features from importance analysis
                                          % false = use manually specified features
config.feature_selection_method = 'RFE'; % Feature selection method:
                                          % 'Univariate' - ANOVA + effect size (fast)
                                          % 'RandomForest' - RF importance (captures interactions)
                                          % 'RFE' - Recursive elimination with validation (most robust)
config.n_top_features = 10;               % Number of top features to use (Univariate/RF only)
                                          % Options: 5, 10, 15, or any number
                                          % RFE determines optimal number automatically
config.rfe_min_features = 10;              % Minimum features for RFE
config.rfe_quality_threshold = 0.05;      % Max quality drop for RFE (0.05 = 5% degradation allowed)

fprintf('Configuration:\n');
fprintf('  Separate by session: %s\n', mat2str(config.separate_by_session));
fprintf('  Normalize features: %s\n', mat2str(config.normalize_features));
fprintf('  Cluster threshold - Aversive: %.1f, Reward: %.1f\n', ...
    config.cluster_threshold_aversive, config.cluster_threshold_reward);
fprintf('  Colormap: %s\n', config.colormap_name);
if ~isempty(config.exclude_categories)
    fprintf('  Exclude categories: %s\n', strjoin(config.exclude_categories, ', '));
end
if config.use_data_driven_features
    fprintf('  Simplified clustering: Data-driven (top %d features)\n', config.n_top_features);
else
    fprintf('  Simplified clustering: Manual selection\n');
end
fprintf('\n');

%% ========================================================================
%  SECTION 2: LOAD DATA
%% ========================================================================

fprintf('Loading feature data...\n');

try
    loaded = load('unit_features_for_clustering.mat');
    results = loaded.results;
    master_features = results.master_features;
    feature_names = results.feature_names;
    fprintf('✓ Loaded: %d units × %d features\n\n', results.n_units, results.n_features);
catch ME
    fprintf('❌ Failed to load: %s\n', ME.message);
    fprintf('Please run Unit_Feature_Extraction.m first\n');
    return;
end

%% ========================================================================
%  SECTION 3: PREPARE FEATURE MATRIX
%% ========================================================================

fprintf('Preparing feature matrix...\n');

% Extract metadata columns
n_units = height(master_features);
session_ids = master_features.Session;
unit_ids = master_features.Unit;
session_types = master_features.SessionType;
is_aversive = contains(string(session_types), 'Aversive');

% Extract feature columns (skip metadata: GlobalUnitID, Session, Unit, SessionType)
feature_col_start = 5;  % Features start at column 5
feature_matrix = table2array(master_features(:, feature_col_start:end));

% Categorize features for color coding
feature_categories = categorize_features(feature_names);
unique_categories = unique(feature_categories, 'stable');

fprintf('✓ Feature matrix prepared\n');
fprintf('  Units: %d\n', n_units);
fprintf('  Features: %d\n', length(feature_names));
fprintf('  Categories: %s\n', strjoin(unique_categories, ', '));
fprintf('  Aversive units: %d\n', sum(is_aversive));
fprintf('  Reward units: %d\n\n', sum(~is_aversive));

%% ========================================================================
%  SECTION 4: FILTER FEATURE CATEGORIES (OPTIONAL)
%% ========================================================================

if ~isempty(config.exclude_categories)
    fprintf('Filtering feature categories...\n');

    features_to_keep = true(size(feature_categories));
    for i = 1:length(feature_categories)
        if any(strcmp(feature_categories{i}, config.exclude_categories))
            features_to_keep(i) = false;
        end
    end

    n_excluded = sum(~features_to_keep);
    fprintf('  Removing %d features from excluded categories\n', n_excluded);

    feature_matrix = feature_matrix(:, features_to_keep);
    feature_names = feature_names(features_to_keep);
    feature_categories = feature_categories(features_to_keep);

    fprintf('✓ Filtered to %d features\n\n', size(feature_matrix, 2));
end

%% ========================================================================
%  SECTION 5: NORMALIZE FEATURES
%% ========================================================================

if config.normalize_features
    fprintf('Normalizing features (z-score)...\n');
    feature_matrix_norm = nan(size(feature_matrix));

    for f = 1:size(feature_matrix, 2)
        feature_col = feature_matrix(:, f);
        valid_data = feature_col(~isnan(feature_col));

        if ~isempty(valid_data) && std(valid_data) > 0
            feature_matrix_norm(:, f) = (feature_col - mean(valid_data)) / std(valid_data);
        else
            feature_matrix_norm(:, f) = feature_col;
        end
    end

    feature_matrix = feature_matrix_norm;
    fprintf('✓ Features normalized\n\n');
end

%% ========================================================================
%  SECTION 6: COMPREHENSIVE CLUSTERING AND VISUALIZATION
%% ========================================================================

fprintf('=== COMPREHENSIVE CLUSTERING ===\n\n');

if config.separate_by_session
    session_types_to_analyze = {'Aversive', 'Reward'};
else
    session_types_to_analyze = {'Combined'};
end

% Store clustering results for feature importance analysis
clustering_results = struct();

for sess_type_idx = 1:length(session_types_to_analyze)
    sess_name = session_types_to_analyze{sess_type_idx};

    fprintf('--- Processing %s sessions ---\n', sess_name);

    % Select units
    if strcmp(sess_name, 'Aversive')
        unit_mask = is_aversive;
    elseif strcmp(sess_name, 'Reward')
        unit_mask = ~is_aversive;
    else
        unit_mask = true(n_units, 1);
    end

    sess_feature_matrix = feature_matrix(unit_mask, :);
    sess_session_ids = session_ids(unit_mask);
    sess_is_aversive = is_aversive(unit_mask);
    n_sess_units = sum(unit_mask);

    fprintf('  Units: %d\n', n_sess_units);

    % Select session-specific cluster threshold
    if strcmp(sess_name, 'Aversive')
        cluster_threshold = config.cluster_threshold_aversive;
    elseif strcmp(sess_name, 'Reward')
        cluster_threshold = config.cluster_threshold_reward;
    else
        % For 'Combined', use Aversive threshold as default
        cluster_threshold = config.cluster_threshold_aversive;
    end

    % Perform hierarchical clustering
    [unit_sort_idx, linkage_tree, cluster_assignments] = perform_clustering(...
        sess_feature_matrix, cluster_threshold);

    n_clusters = max(cluster_assignments(~isnan(cluster_assignments)));
    fprintf('  Clusters: %d\n', n_clusters);

    % Get unique unit IDs for these units
    sess_unique_unit_ids = master_features.UniqueUnitID(unit_mask);
    sess_session_names = master_features.Session(unit_mask);
    sess_unit_numbers = master_features.Unit(unit_mask);

    % Store results for feature importance analysis
    clustering_results.(sess_name).feature_matrix = sess_feature_matrix;
    clustering_results.(sess_name).cluster_assignments = cluster_assignments;
    clustering_results.(sess_name).n_clusters = n_clusters;
    clustering_results.(sess_name).unit_mask = unit_mask;
    clustering_results.(sess_name).unique_unit_ids = sess_unique_unit_ids;
    clustering_results.(sess_name).session_names = sess_session_names;
    clustering_results.(sess_name).unit_numbers = sess_unit_numbers;

    % Create comprehensive heatmap
    create_comprehensive_heatmap(sess_feature_matrix, feature_names, feature_categories, ...
        unit_sort_idx, linkage_tree, sess_is_aversive, sess_name, config, cluster_threshold);

    % Analyze cluster composition
    analyze_cluster_composition(cluster_assignments, sess_session_ids, sess_name, linkage_tree, config, cluster_threshold);

    fprintf('✓ %s sessions complete\n\n', sess_name);
end

% Save cluster assignments with unique unit IDs
fprintf('Saving cluster assignments...\n');
cluster_assignments_output = struct();
cluster_assignments_output.config = config;
cluster_assignments_output.timestamp = datetime('now');

for sess_type_idx = 1:length(session_types_to_analyze)
    sess_name = session_types_to_analyze{sess_type_idx};

    if ~isfield(clustering_results, sess_name)
        continue;
    end

    % Create table with unit IDs and cluster assignments
    unit_cluster_table = table();
    unit_cluster_table.UniqueUnitID = clustering_results.(sess_name).unique_unit_ids;
    unit_cluster_table.Session = clustering_results.(sess_name).session_names;
    unit_cluster_table.Unit = clustering_results.(sess_name).unit_numbers;
    unit_cluster_table.ClusterID = clustering_results.(sess_name).cluster_assignments;
    unit_cluster_table.SessionType = repmat({sess_name}, height(unit_cluster_table), 1);

    % Store in output structure
    cluster_assignments_output.(sess_name) = unit_cluster_table;

    % Print summary
    valid_assignments = ~isnan(unit_cluster_table.ClusterID);
    fprintf('  %s: %d units assigned to %d clusters\n', ...
        sess_name, sum(valid_assignments), clustering_results.(sess_name).n_clusters);
end

% Save to file
save('unit_cluster_assignments.mat', 'cluster_assignments_output');
fprintf('✓ Saved to: unit_cluster_assignments.mat\n\n');

%% ========================================================================
%  SECTION 6b: PCA-BASED COMPREHENSIVE CLUSTERING
%% ========================================================================

fprintf('=== PCA-BASED COMPREHENSIVE CLUSTERING ===\n\n');
fprintf('Performing clustering using first 5 principal components...\n\n');

for sess_type_idx = 1:length(session_types_to_analyze)
    sess_name = session_types_to_analyze{sess_type_idx};

    fprintf('--- Processing %s sessions (PCA) ---\n', sess_name);

    % Select units
    if strcmp(sess_name, 'Aversive')
        unit_mask = is_aversive;
    elseif strcmp(sess_name, 'Reward')
        unit_mask = ~is_aversive;
    else
        unit_mask = true(n_units, 1);
    end

    sess_feature_matrix = feature_matrix(unit_mask, :);
    sess_session_ids = session_ids(unit_mask);
    sess_is_aversive = is_aversive(unit_mask);
    n_sess_units = sum(unit_mask);

    fprintf('  Units: %d\n', n_sess_units);

    % Perform PCA on the feature matrix
    fprintf('  Performing PCA...\n');
    [coeff, score, latent, ~, explained] = pca(sess_feature_matrix, 'NumComponents', 5);

    fprintf('  Variance explained by first 5 PCs: %.1f%%\n', sum(explained(1:5)));
    fprintf('    PC1: %.1f%%, PC2: %.1f%%, PC3: %.1f%%, PC4: %.1f%%, PC5: %.1f%%\n', ...
        explained(1), explained(2), explained(3), explained(4), explained(5));

    % Use first 5 PCs for clustering
    pca_features = score(:, 1:5);

    % Select session-specific cluster threshold
    if strcmp(sess_name, 'Aversive')
        cluster_threshold = config.cluster_threshold_aversive;
    elseif strcmp(sess_name, 'Reward')
        cluster_threshold = config.cluster_threshold_reward;
    else
        cluster_threshold = config.cluster_threshold_aversive;
    end

    %% Perform hierarchical clustering on PCA features
    [unit_sort_idx, linkage_tree, cluster_assignments] = perform_clustering(...
        pca_features, cluster_threshold);

    n_clusters = max(cluster_assignments(~isnan(cluster_assignments)));
    fprintf('  Clusters: %d\n', n_clusters);

    % Create PCA-based heatmap
    create_pca_heatmap(pca_features, unit_sort_idx, linkage_tree, ...
        cluster_assignments, sess_is_aversive, sess_name, explained(1:5), ...
        config, cluster_threshold);

    % Create PC loadings heatmap
    create_pca_loadings_heatmap(coeff(:, 1:5), feature_names, explained(1:5), ...
        sess_name, feature_categories);

    % Analyze cluster composition
    analyze_cluster_composition(cluster_assignments, sess_session_ids, ...
        [sess_name '_PCA'], linkage_tree, config, cluster_threshold);

    fprintf('✓ PCA-based %s clustering complete\n\n', sess_name);
end

%% ========================================================================
%  SECTION 7: FEATURE IMPORTANCE ANALYSIS
%% ========================================================================

fprintf('=== FEATURE IMPORTANCE ANALYSIS ===\n\n');
fprintf('Analyzing which features best discriminate between clusters...\n\n');

% Store feature importance results
feature_importance_results = struct();

for sess_type_idx = 1%:length(session_types_to_analyze)
    sess_name = session_types_to_analyze{sess_type_idx};

    fprintf('--- %s sessions ---\n', sess_name);

    % Get clustering results
    sess_data = clustering_results.(sess_name);
    n_clusters = sess_data.n_clusters;

    % Check if we have valid clusters
    if isnan(n_clusters) || n_clusters < 2
        fprintf('  Skipping: insufficient clusters (n=%d)\n\n', n_clusters);
        continue;
    end

    fprintf('  Analyzing %d clusters...\n', n_clusters);

    % Get valid units (those with cluster assignments)
    valid_units = ~isnan(sess_data.cluster_assignments);
    feature_matrix_valid = sess_data.feature_matrix(valid_units, :);
    cluster_labels = sess_data.cluster_assignments(valid_units);

    % 1. Global feature importance (ANOVA + effect sizes)
    fprintf('  Computing global feature importance...\n');
    [feature_pvalues, feature_effect_sizes, feature_rankings] = ...
        compute_global_feature_importance(feature_matrix_valid, cluster_labels, feature_names);

    % 2. Pairwise cluster discrimination
    fprintf('  Computing pairwise cluster discrimination...\n');
    [pairwise_discrimination, top_features_per_pair] = ...
        compute_pairwise_discrimination(feature_matrix_valid, cluster_labels, feature_names, n_clusters);

    % 3. Random Forest feature importance
    fprintf('  Computing Random Forest feature importance...\n');
    [rf_importance, rf_rankings] = ...
        compute_random_forest_importance(feature_matrix_valid, cluster_labels, feature_names);

    % 4. Recursive Feature Elimination with cluster validation
    fprintf('  Performing Recursive Feature Elimination...\n');
    [rfe_selected_features, rfe_quality_curve, rfe_rankings] = ...
        perform_rfe_with_validation(feature_matrix_valid, cluster_labels, feature_names, ...
        config.rfe_min_features, config.rfe_quality_threshold);

    % 5. Create visualizations
    fprintf('  Creating visualizations...\n');

    % Feature importance heatmap
    create_feature_importance_heatmap(feature_matrix_valid, cluster_labels, ...
        feature_names, feature_categories, feature_rankings, sess_name);

    % Pairwise discrimination heatmap
    create_pairwise_discrimination_heatmap(pairwise_discrimination, feature_names, ...
        n_clusters, sess_name);

    % Top features bar plot (univariate)
    create_top_features_plot(feature_rankings, feature_names, feature_pvalues, ...
        feature_effect_sizes, sess_name);

    % RFE quality curve visualization
    create_rfe_quality_plot(rfe_quality_curve, rfe_selected_features, feature_names, sess_name);

    % Feature importance comparison plot
    create_feature_comparison_plot(feature_rankings, rf_rankings, rfe_rankings, ...
        feature_names, feature_effect_sizes, rf_importance, sess_name);

    % Store results
    feature_importance_results.(sess_name).univariate_rankings = feature_rankings;
    feature_importance_results.(sess_name).pvalues = feature_pvalues;
    feature_importance_results.(sess_name).effect_sizes = feature_effect_sizes;
    feature_importance_results.(sess_name).rf_rankings = rf_rankings;
    feature_importance_results.(sess_name).rf_importance = rf_importance;
    feature_importance_results.(sess_name).rfe_selected = rfe_selected_features;
    feature_importance_results.(sess_name).rfe_rankings = rfe_rankings;
    feature_importance_results.(sess_name).rfe_quality_curve = rfe_quality_curve;
    feature_importance_results.(sess_name).pairwise_discrimination = pairwise_discrimination;
    feature_importance_results.(sess_name).top_features_per_pair = top_features_per_pair;

    % Print top features from all three methods
    fprintf('\n  === METHOD 1: UNIVARIATE (Top 15) ===\n');
    for i = 1:min(15, length(feature_rankings))
        feat_idx = feature_rankings(i);
        fprintf('  %2d. %-30s (η²=%.3f, p=%.2e)\n', ...
            i, feature_names{feat_idx}, feature_effect_sizes(feat_idx), feature_pvalues(feat_idx));
    end

    fprintf('\n  === METHOD 2: RANDOM FOREST (Top 15) ===\n');
    for i = 1:min(15, length(rf_rankings))
        feat_idx = rf_rankings(i);
        fprintf('  %2d. %-30s (Importance=%.4f)\n', ...
            i, feature_names{feat_idx}, rf_importance(feat_idx));
    end

    fprintf('\n  === METHOD 3: RFE (Selected %d features) ===\n', length(rfe_selected_features));
    for i = 1:length(rfe_selected_features)
        feat_idx = rfe_selected_features(i);
        fprintf('  %2d. %s\n', i, feature_names{feat_idx});
    end
    fprintf('\n');

    fprintf('✓ %s feature importance analysis complete\n\n', sess_name);
end

% Print recommended minimal feature sets from all methods
fprintf('=== RECOMMENDED FEATURE SETS (BY METHOD) ===\n\n');
for sess_type_idx = 1%:length(session_types_to_analyze)
    sess_name = session_types_to_analyze{sess_type_idx};

    if ~isfield(feature_importance_results, sess_name)
        continue;
    end

    fprintf('%s sessions:\n', sess_name);

    % Univariate
    univ_rankings = feature_importance_results.(sess_name).univariate_rankings;
    fprintf('  Univariate (Top 10): %s\n', strjoin(feature_names(univ_rankings(1:min(10, length(univ_rankings)))), ', '));

    % Random Forest
    rf_rankings = feature_importance_results.(sess_name).rf_rankings;
    fprintf('  RandomForest (Top 10): %s\n', strjoin(feature_names(rf_rankings(1:min(10, length(rf_rankings)))), ', '));

    % RFE
    rfe_selected = feature_importance_results.(sess_name).rfe_selected;
    fprintf('  RFE (%d features): %s\n\n', length(rfe_selected), strjoin(feature_names(rfe_selected), ', '));
end

fprintf('✓ Feature importance analysis complete\n\n');

%% ========================================================================
%  SECTION 8: SIMPLIFIED CLUSTERING (KEY FEATURES ONLY)
%% ========================================================================

fprintf('=== SIMPLIFIED CLUSTERING (KEY FEATURES) ===\n\n');

% Manual feature list (fallback if feature importance not available)
manual_feature_names = {
    'PPC_LFP_low', 'PPC_LFP_high', ...
    'PPC_Breath_low', 'PPC_Breath_high', ...
    'Coherence_LFP_low', 'Coherence_LFP_high', ...
    'FR', 'CV', 'BurstIndex'
};

% Determine feature selection method
if config.use_data_driven_features && exist('feature_importance_results', 'var') && ~isempty(fieldnames(feature_importance_results))
    fprintf('Using DATA-DRIVEN feature selection: %s\n\n', config.feature_selection_method);
    use_session_specific_features = true;
else
    fprintf('Using MANUAL feature selection (%d predefined features)\n\n', length(manual_feature_names));
    use_session_specific_features = false;

    % Find indices of manual features
    key_feature_idx = [];
    key_feature_names_found = {};
    for i = 1:length(manual_feature_names)
        idx = find(strcmp(feature_names, manual_feature_names{i}), 1);
        if ~isempty(idx)
            key_feature_idx(end+1) = idx;
            key_feature_names_found{end+1} = manual_feature_names{i};
        end
    end
    fprintf('Found %d/%d manual features\n\n', length(key_feature_idx), length(manual_feature_names));
end

for sess_type_idx = 1:length(session_types_to_analyze)
    sess_name = session_types_to_analyze{sess_type_idx};

    fprintf('--- Simplified clustering for %s sessions ---\n', sess_name);

    % Select features for this session
    if use_session_specific_features
        % Use session-specific features based on selected method
        if isfield(feature_importance_results, sess_name)
            switch config.feature_selection_method
                case 'Univariate'
                    rankings = feature_importance_results.(sess_name).univariate_rankings;
                    n_features_to_use = min(config.n_top_features, length(rankings));
                    key_feature_idx = rankings(1:n_features_to_use);
                    fprintf('  Using top %d features from UNIVARIATE method\n', n_features_to_use);

                case 'RandomForest'
                    rankings = feature_importance_results.(sess_name).rf_rankings;
                    n_features_to_use = min(config.n_top_features, length(rankings));
                    key_feature_idx = rankings(1:n_features_to_use);
                    fprintf('  Using top %d features from RANDOM FOREST method\n', n_features_to_use);

                case 'RFE'
                    key_feature_idx = feature_importance_results.(sess_name).rfe_selected;
                    n_features_to_use = length(key_feature_idx);
                    fprintf('  Using %d features from RFE method\n', n_features_to_use);

                otherwise
                    error('Unknown feature selection method: %s', config.feature_selection_method);
            end

            key_feature_names_found = feature_names(key_feature_idx);

            fprintf('  Selected features:\n');
            for i = 1:min(5, n_features_to_use)
                fprintf('    %d. %s\n', i, key_feature_names_found{i});
            end
            if n_features_to_use > 5
                fprintf('    ... and %d more\n', n_features_to_use - 5);
            end
        else
            fprintf('  No feature importance results available, skipping...\n\n');
            continue;
        end
    end

    % Check if we have enough features
    if length(key_feature_idx) < 3
        fprintf('  Not enough features found, skipping...\n\n');
        continue;
    end

    % Select units
    if strcmp(sess_name, 'Aversive')
        unit_mask = is_aversive;
    elseif strcmp(sess_name, 'Reward')
        unit_mask = ~is_aversive;
    else
        unit_mask = true(n_units, 1);
    end

    % Extract simplified feature matrix
    simplified_matrix = feature_matrix(:, key_feature_idx);
    simplified_names = key_feature_names_found;

    sess_simple_matrix = simplified_matrix(unit_mask, :);
    sess_session_ids = session_ids(unit_mask);
    sess_is_aversive = is_aversive(unit_mask);

    % Select session-specific cluster threshold
    if strcmp(sess_name, 'Aversive')
        cluster_threshold = config.cluster_threshold_aversive;
    elseif strcmp(sess_name, 'Reward')
        cluster_threshold = config.cluster_threshold_reward;
    else
        cluster_threshold = config.cluster_threshold_aversive;
    end

    % Perform clustering
    [unit_sort_idx, linkage_tree, cluster_assignments] = perform_clustering(...
        sess_simple_matrix, cluster_threshold);

    n_clusters = max(cluster_assignments(~isnan(cluster_assignments)));
    fprintf('  Clusters: %d\n', n_clusters);

    % Create simplified heatmap
    create_simplified_heatmap(sess_simple_matrix, simplified_names, ...
        unit_sort_idx, linkage_tree, cluster_assignments, sess_is_aversive, ...
        sess_session_ids, sess_name, config, cluster_threshold);

    fprintf('✓ Simplified %s complete\n\n', sess_name);
end

fprintf('✓ Simplified clustering complete\n\n');

%% ========================================================================
%  SECTION 9: SUMMARY
%% ========================================================================

fprintf('\n========================================\n');
fprintf('CLUSTERING AND VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Total units analyzed: %d\n', n_units);
fprintf('Total features: %d\n', length(feature_names));
fprintf('Session types: %s\n', strjoin(session_types_to_analyze, ', '));
fprintf('\nFiles saved:\n');
fprintf('  - unit_cluster_assignments.mat (UniqueUnitID + ClusterID)\n');
fprintf('\nFigures created:\n');
fprintf('  - Comprehensive heatmaps with dendrograms + cluster IDs\n');
fprintf('  - Cluster composition analysis\n');
fprintf('  - Feature importance analysis (Univariate, RF, RFE)\n');
fprintf('  - Simplified clustering with key features\n');
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS
%% ========================================================================

function feature_categories = categorize_features(feature_names)
% Categorize features based on their names

    feature_categories = cell(size(feature_names));

    for i = 1:length(feature_names)
        name = feature_names{i};

        if contains(name, 'PPC')
            feature_categories{i} = 'PPC';
        elseif contains(name, 'Coherence')
            feature_categories{i} = 'Coherence';
        elseif contains(name, 'PearsonR')
            feature_categories{i} = 'PearsonR';
        elseif contains(name, 'MutualInfo')
            feature_categories{i} = 'MutualInfo';
        elseif contains(name, 'PSTH')
            feature_categories{i} = 'PSTH';
        elseif ismember(name, {'FR', 'CV', 'ISI_FanoFactor', 'ISI_ACF_peak', 'ISI_ACF_lag'})
            feature_categories{i} = 'Spike_Basic';
        elseif contains(name, 'ACF') || contains(name, 'Fano')
            feature_categories{i} = 'Spike_Temporal';
        elseif contains(name, 'Burst')
            feature_categories{i} = 'Spike_Burst';
        elseif ismember(name, {'LV', 'CV2', 'ISI_Skewness', 'ISI_Kurtosis', 'ISI_Mode', 'RefracViolations'})
            feature_categories{i} = 'Spike_Variability';
        else
            feature_categories{i} = 'Other';
        end
    end
end

function [unit_sort_idx, linkage_tree, cluster_assignments] = perform_clustering(feature_matrix, cluster_threshold)
% Perform hierarchical clustering on feature matrix

    n_units = size(feature_matrix, 1);

    % Remove units with too many NaNs
    valid_units = sum(~isnan(feature_matrix), 2) > size(feature_matrix, 2) * 0.5;

    if sum(valid_units) < 3
        fprintf('    WARNING: Not enough valid units, using original order\n');
        unit_sort_idx = 1:n_units;
        linkage_tree = [];
        cluster_assignments = nan(n_units, 1);
        return;
    end

    feature_matrix_clean = feature_matrix(valid_units, :);

    % Remove features with too many NaNs
    valid_features = sum(~isnan(feature_matrix_clean), 1) > size(feature_matrix_clean, 1) * 0.3;
    feature_matrix_clean = feature_matrix_clean(:, valid_features);

    % Replace remaining NaNs with column mean
    for f = 1:size(feature_matrix_clean, 2)
        col = feature_matrix_clean(:, f);
        if ~all(isnan(col))
            col(isnan(col)) = nanmean(col);
            feature_matrix_clean(:, f) = col;
        end
    end

    % Compute distance and linkage
    distances = pdist(feature_matrix_clean, 'euclidean');
    linkage_tree = linkage(distances, 'ward');

    % Get dendrogram order
    [~, ~, sort_idx_clean] = dendrogram(linkage_tree, 0);

    % Cluster assignments
    cluster_assignments_clean = cluster(linkage_tree, 'cutoff', cluster_threshold, 'criterion', 'distance');

    % Map back to original indices
    valid_idx = find(valid_units);
    unit_sort_idx = valid_idx(sort_idx_clean);

    cluster_assignments = nan(n_units, 1);
    cluster_assignments(valid_idx) = cluster_assignments_clean;
end

function create_comprehensive_heatmap(feature_matrix, feature_names, feature_categories, ...
    unit_sort_idx, linkage_tree, is_aversive, session_name, config, cluster_threshold)
% Create comprehensive heatmap with dendrogram

    % Reorder matrix
    feature_matrix_sorted = feature_matrix(unit_sort_idx, :);
    is_aversive_sorted = is_aversive(unit_sort_idx);

    % Get cluster assignments for sorted units
    if ~isempty(linkage_tree)
        cluster_assignments = cluster(linkage_tree, 'cutoff', cluster_threshold, 'criterion', 'distance');
        cluster_assignments_sorted = cluster_assignments(unit_sort_idx);
        n_clusters = max(cluster_assignments);
    else
        cluster_assignments_sorted = ones(length(unit_sort_idx), 1);
        n_clusters = 1;
    end

    % Create figure
    fig = figure('Position', [50 50 2000 1200], ...
        'Name', sprintf('%s Sessions - Comprehensive Unit Features', session_name));

    % Subplot positions
    dendro_width = 0.10;
    heatmap_left = 0.05 + dendro_width + 0.02;
    heatmap_width = 0.70;
    heatmap_bottom = 0.15;
    heatmap_height = 0.70;

    % Draw dendrogram
    if config.show_dendrogram && ~isempty(linkage_tree)
        subplot('Position', [0.05 heatmap_bottom dendro_width heatmap_height]);
        dendrogram(linkage_tree, 0, 'Orientation', 'left', ...
            'ColorThreshold', cluster_threshold);
        set(gca, 'YDir', 'reverse');
        set(gca, 'XTickLabel', []);
        ylabel('Units', 'FontSize', 11, 'FontWeight', 'bold');
        title('Clustering', 'FontSize', 12);

        % Add threshold line
        hold on;
        xlims = xlim;
        plot([cluster_threshold, cluster_threshold], ylim, 'r--', 'LineWidth', 2);
        hold off;
    end

    % Draw main heatmap
    subplot('Position', [heatmap_left heatmap_bottom heatmap_width heatmap_height]);
    imagesc(feature_matrix_sorted);
    axis tight;

    % Colormap
    if strcmp(config.colormap_name, 'bluewhitered')
        colormap(bluewhitered(256));
        if config.normalize_features
            caxis([-3 3]);
        end
    else
        colormap(config.colormap_name);
    end

    % Colorbar
    cb = colorbar;
    cb.Position = [0.90 heatmap_bottom 0.02 heatmap_height];
    if config.normalize_features
        ylabel(cb, 'Z-score', 'FontSize', 11);
    else
        ylabel(cb, 'Feature Value', 'FontSize', 11);
    end

    % Labels
    xlabel('Features', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Units', 'FontSize', 12, 'FontWeight', 'bold');

    % X-axis: Feature names
    set(gca, 'XTick', 1:length(feature_names));
    set(gca, 'XTickLabel', feature_names);
    set(gca, 'XTickLabelRotation', 90);
    set(gca, 'FontSize', 7);

    % Y-axis
    set(gca, 'YTick', []);

    % Cluster ID color bar (left of heatmap)
    subplot('Position', [heatmap_left-0.025 heatmap_bottom 0.01 heatmap_height]);
    cluster_colormap = lines(n_clusters);  % Distinct colors for each cluster
    cluster_colors = zeros(length(unit_sort_idx), 1, 3);
    for i = 1:length(unit_sort_idx)
        cluster_id = cluster_assignments_sorted(i);
        cluster_colors(i, 1, :) = cluster_colormap(cluster_id, :);
    end
    image(cluster_colors);
    set(gca, 'XTick', [], 'YTick', []);
    ylabel('Cluster ID', 'FontSize', 9, 'FontWeight', 'bold', 'Rotation', 0, ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');

    % Session type color bar
    subplot('Position', [heatmap_left-0.012 heatmap_bottom 0.005 heatmap_height]);
    session_colors = zeros(length(unit_sort_idx), 1, 3);
    for i = 1:length(unit_sort_idx)
        if is_aversive_sorted(i)
            session_colors(i, 1, :) = [0.8 0.2 0.2];  % Red
        else
            session_colors(i, 1, :) = [0.2 0.2 0.8];  % Blue
        end
    end
    image(session_colors);
    set(gca, 'XTick', [], 'YTick', []);

    % Feature category color bar
    subplot('Position', [heatmap_left 0.87 heatmap_width 0.01]);
    unique_categories = unique(feature_categories, 'stable');
    category_colors = lines(length(unique_categories));
    feature_category_img = zeros(1, length(feature_names), 3);

    for i = 1:length(feature_names)
        cat_idx = find(strcmp(unique_categories, feature_categories{i}));
        feature_category_img(1, i, :) = category_colors(cat_idx, :);
    end

    image(feature_category_img);
    set(gca, 'XTick', [], 'YTick', []);

    % Title
    title_str = sprintf('%s Sessions: %d units × %d features | Threshold: %.1f', ...
        session_name, size(feature_matrix_sorted, 1), size(feature_matrix_sorted, 2), ...
        cluster_threshold);
    annotation('textbox', [heatmap_left 0.92 heatmap_width 0.05], 'String', title_str, ...
        'EdgeColor', 'none', 'FontSize', 14, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');

    % Category legend
    legend_str = 'Categories: ';
    for i = 1:length(unique_categories)
        legend_str = [legend_str, sprintf('%s | ', unique_categories{i})];
    end
    annotation('textbox', [0.05 0.08 0.90 0.03], 'String', legend_str, ...
        'EdgeColor', 'none', 'FontSize', 9, 'HorizontalAlignment', 'center');

    % Cluster ID legend with color patches
    cluster_legend_y = 0.02;
    patch_width = 0.015;
    patch_height = 0.025;
    text_offset = 0.02;

    for c = 1:n_clusters
        patch_x = 0.05 + (c-1) * 0.08;
        % Draw color patch
        annotation('rectangle', [patch_x cluster_legend_y patch_width patch_height], ...
            'FaceColor', cluster_colormap(c, :), 'EdgeColor', 'k', 'LineWidth', 1);
        % Add cluster ID text
        annotation('textbox', [patch_x + text_offset cluster_legend_y patch_width*2 patch_height], ...
            'String', sprintf('Cluster %d', c), 'EdgeColor', 'none', ...
            'FontSize', 9, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
    end

    % Save figure
    fig_filename = sprintf('Comprehensive_Heatmap_%s.png', session_name);
    saveas(fig, fig_filename);
    fprintf('  Saved: %s\n', fig_filename);
end

function analyze_cluster_composition(cluster_assignments, session_ids, session_name, linkage_tree, config, cluster_threshold)
% Analyze and visualize cluster composition by session

    valid_clusters = ~isnan(cluster_assignments);
    cluster_assignments = cluster_assignments(valid_clusters);
    session_ids = session_ids(valid_clusters);

    n_clusters = max(cluster_assignments);
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
    cluster_colormap = lines(n_clusters);  % Same colors as comprehensive heatmap
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

function create_pca_heatmap(pca_features, unit_sort_idx, linkage_tree, ...
    cluster_assignments, is_aversive, session_name, explained_variance, ...
    config, cluster_threshold)
% Create heatmap visualization for PCA-based clustering
%
% Inputs:
%   pca_features: [n_units × 5] matrix of PC scores
%   unit_sort_idx: Unit indices after clustering
%   linkage_tree: Hierarchical linkage tree
%   cluster_assignments: Cluster IDs for each unit
%   is_aversive: Boolean array indicating session type
%   session_name: Name of session type
%   explained_variance: [5 × 1] variance explained by each PC
%   config: Configuration structure
%   cluster_threshold: Distance threshold for clustering

    % Reorder features by clustering
    pca_features_sorted = pca_features(unit_sort_idx, :);
    is_aversive_sorted = is_aversive(unit_sort_idx);
    cluster_assignments_sorted = cluster_assignments(unit_sort_idx);

    % Get number of clusters
    n_clusters = max(cluster_assignments(~isnan(cluster_assignments)));
    if isnan(n_clusters)
        n_clusters = 0;
    end

    % Create figure
    fig = figure('Position', [100 100 1200 900], ...
        'Name', sprintf('%s - PCA Clustering', session_name));

    % Dendrogram
    if ~isempty(linkage_tree)
        subplot('Position', [0.05 0.15 0.10 0.75]);
        dendrogram(linkage_tree, 0, 'Orientation', 'left', ...
            'ColorThreshold', cluster_threshold);
        set(gca, 'YDir', 'reverse');
        set(gca, 'XTickLabel', []);
        ylabel('Units', 'FontSize', 11);
        title('Clustering', 'FontSize', 12);
        heatmap_left = 0.18;
    else
        heatmap_left = 0.08;
    end

    % Cluster ID color bar
    subplot('Position', [heatmap_left - 0.02 0.15 0.015 0.75]);
    cluster_colormap = lines(n_clusters);
    cluster_colors = zeros(length(unit_sort_idx), 1, 3);
    for i = 1:length(unit_sort_idx)
        cluster_id = cluster_assignments_sorted(i);
        if ~isnan(cluster_id)
            cluster_colors(i, 1, :) = cluster_colormap(cluster_id, :);
        else
            cluster_colors(i, 1, :) = [0.5 0.5 0.5];  % Gray for unassigned
        end
    end
    image(cluster_colors);
    set(gca, 'XTick', [], 'YTick', []);
    ylabel('Cluster', 'FontSize', 10, 'FontWeight', 'bold', 'Rotation', 0, ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');

    % Heatmap
    heatmap_width = 0.50;
    subplot('Position', [heatmap_left 0.15 heatmap_width 0.75]);
    imagesc(pca_features_sorted);
    colormap(gca, bluewhitered(256));

    % Set color limits based on data range
    data_range = prctile(pca_features_sorted(:), [1 99]);
    clim_val = max(abs(data_range));
    caxis([-clim_val clim_val]);
    axis tight;

    cb = colorbar;
    cb.Position = [heatmap_left + heatmap_width + 0.02 0.15 0.02 0.75];
    ylabel(cb, 'PC Score', 'FontSize', 11);

    ylabel('Units', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Principal Components', 'FontSize', 12, 'FontWeight', 'bold');

    % Create PC labels with variance explained
    pc_labels = cell(5, 1);
    for i = 1:5
        pc_labels{i} = sprintf('PC%d (%.1f%%)', i, explained_variance(i));
    end
    set(gca, 'XTick', 1:5);
    set(gca, 'XTickLabel', pc_labels);
    set(gca, 'YTick', []);
    set(gca, 'FontSize', 10);

    % Session color bar
    subplot('Position', [heatmap_left + heatmap_width + 0.06 0.15 0.01 0.75]);
    session_colors = zeros(length(unit_sort_idx), 1, 3);
    for i = 1:length(unit_sort_idx)
        if is_aversive_sorted(i)
            session_colors(i, 1, :) = [0.8 0.2 0.2];  % Red for aversive
        else
            session_colors(i, 1, :) = [0.2 0.2 0.8];  % Blue for reward
        end
    end
    image(session_colors);
    set(gca, 'XTick', [], 'YTick', []);
    ylabel('Session', 'FontSize', 10, 'FontWeight', 'bold', 'Rotation', 0, ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');

    % Legend
    subplot('Position', [heatmap_left + heatmap_width + 0.10 0.15 0.20 0.75]);
    axis off;

    % Session type legend
    text(0.1, 0.95, 'Session Type:', 'FontSize', 11, 'FontWeight', 'bold');
    rectangle('Position', [0.1 0.90 0.05 0.03], 'FaceColor', [0.8 0.2 0.2], 'EdgeColor', 'k');
    text(0.18, 0.915, 'Aversive', 'FontSize', 10);
    rectangle('Position', [0.1 0.86 0.05 0.03], 'FaceColor', [0.2 0.2 0.8], 'EdgeColor', 'k');
    text(0.18, 0.875, 'Reward', 'FontSize', 10);

    % Cluster legend
    text(0.1, 0.78, sprintf('Clusters (n=%d):', n_clusters), 'FontSize', 11, 'FontWeight', 'bold');
    y_pos = 0.73;
    for c = 1:min(n_clusters, 20)  % Limit to 20 clusters for display
        rectangle('Position', [0.1 y_pos 0.05 0.03], ...
            'FaceColor', cluster_colormap(c, :), 'EdgeColor', 'k');
        n_units_in_cluster = sum(cluster_assignments == c);
        text(0.18, y_pos + 0.015, sprintf('Cluster %d (n=%d)', c, n_units_in_cluster), ...
            'FontSize', 9);
        y_pos = y_pos - 0.04;
        if y_pos < 0.15
            break;
        end
    end

    % Variance summary
    text(0.1, 0.10, sprintf('Total variance: %.1f%%', sum(explained_variance)), ...
        'FontSize', 10, 'FontWeight', 'bold');
    text(0.1, 0.05, sprintf('Threshold: %.1f', cluster_threshold), ...
        'FontSize', 9);

    % Title
    title_str = sprintf('%s Sessions: PCA-Based Clustering\n%d units × 5 PCs | %d clusters', ...
        session_name, size(pca_features_sorted, 1), n_clusters);
    annotation('textbox', [heatmap_left 0.92 heatmap_width 0.05], 'String', title_str, ...
        'EdgeColor', 'none', 'FontSize', 14, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');

    % Save figure
    fig_filename = sprintf('PCA_Heatmap_%s.png', session_name);
    saveas(fig, fig_filename);
    fprintf('  Saved: %s\n', fig_filename);
end

function create_pca_loadings_heatmap(pca_loadings, feature_names, explained_variance, ...
    session_name, feature_categories)
% Create heatmap showing PC loadings (how features contribute to each PC)
%
% Inputs:
%   pca_loadings: [n_features × 5] matrix of PC coefficients
%   feature_names: Cell array of feature names
%   explained_variance: [5 × 1] variance explained by each PC
%   session_name: Name of session type
%   feature_categories: Cell array of feature categories

    n_features = size(pca_loadings, 1);

    % Create figure
    fig = figure('Position', [50 50 2000 600], ...
        'Name', sprintf('%s - PC Loadings', session_name));

    % Main heatmap: PCs (rows) × Features (columns)
    subplot('Position', [0.08 0.25 0.85 0.65]);
    imagesc(pca_loadings');  % Transpose to get PCs as rows
    colormap(bluewhitered(256));

    % Set symmetric color limits
    max_abs_loading = max(abs(pca_loadings(:)));
    caxis([-max_abs_loading max_abs_loading]);

    cb = colorbar;
    cb.Position = [0.94 0.25 0.015 0.65];
    ylabel(cb, 'Loading', 'FontSize', 11);

    % Create PC labels with variance explained
    pc_labels = cell(5, 1);
    for i = 1:5
        pc_labels{i} = sprintf('PC%d (%.1f%%)', i, explained_variance(i));
    end

    ylabel('Principal Components', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Features', 'FontSize', 12, 'FontWeight', 'bold');

    set(gca, 'YTick', 1:5);
    set(gca, 'YTickLabel', pc_labels);
    set(gca, 'XTick', 1:n_features);
    set(gca, 'XTickLabel', feature_names);
    set(gca, 'XTickLabelRotation', 90);
    set(gca, 'FontSize', 8);

    % Add vertical grid lines to separate features
    hold on;
    for i = 1:n_features-1
        plot([i+0.5 i+0.5], [0.5 5.5], 'k-', 'LineWidth', 0.1);
    end
    hold off;

    % Category color bar at bottom
    subplot('Position', [0.08 0.15 0.85 0.05]);

    % Get unique categories and assign colors
    unique_categories = unique(feature_categories, 'stable');
    n_categories = length(unique_categories);
    category_colors = lines(n_categories);

    % Create color matrix for features
    feature_color_matrix = zeros(1, n_features, 3);
    for f = 1:n_features
        cat_idx = find(strcmp(unique_categories, feature_categories{f}));
        feature_color_matrix(1, f, :) = category_colors(cat_idx, :);
    end

    image(feature_color_matrix);
    set(gca, 'XTick', []);
    set(gca, 'YTick', []);
    ylabel('Category', 'FontSize', 10, 'FontWeight', 'bold', 'Rotation', 0, ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');

    % Category legend at bottom
    subplot('Position', [0.08 0.02 0.85 0.08]);
    axis off;

    % Display category legend horizontally
    n_cols = min(n_categories, 6);  % Max 6 categories per row
    n_rows = ceil(n_categories / n_cols);

    for c = 1:n_categories
        row = ceil(c / n_cols) - 1;
        col = mod(c - 1, n_cols);

        x_pos = col / n_cols;
        y_pos = 1 - (row + 0.5) / n_rows;

        rectangle('Position', [x_pos y_pos 0.02 0.3], ...
            'FaceColor', category_colors(c, :), 'EdgeColor', 'k');
        text(x_pos + 0.025, y_pos + 0.15, unique_categories{c}, ...
            'FontSize', 9, 'VerticalAlignment', 'middle');
    end

    % Title
    title_str = sprintf('%s Sessions: PC Loadings (5 PCs × %d Features)', ...
        session_name, n_features);
    annotation('textbox', [0.08 0.92 0.85 0.05], 'String', title_str, ...
        'EdgeColor', 'none', 'FontSize', 14, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');

    % Save figure
    fig_filename = sprintf('PCA_Loadings_%s.png', session_name);
    saveas(fig, fig_filename);
    fprintf('  Saved: %s\n', fig_filename);
end

function create_simplified_heatmap(feature_matrix, feature_names, ...
    unit_sort_idx, linkage_tree, cluster_assignments, is_aversive, ...
    session_ids, session_name, config, cluster_threshold)
% Create simplified heatmap with fewer features

    % Reorder
    feature_matrix_sorted = feature_matrix(unit_sort_idx, :);
    is_aversive_sorted = is_aversive(unit_sort_idx);

    % Create figure
    fig = figure('Position', [150 150 1400 900], ...
        'Name', sprintf('%s - Simplified Clustering', session_name));

    % Dendrogram
    if ~isempty(linkage_tree)
        subplot('Position', [0.05 0.15 0.10 0.75]);
        dendrogram(linkage_tree, 0, 'Orientation', 'left', ...
            'ColorThreshold', cluster_threshold);
        set(gca, 'YDir', 'reverse');
        set(gca, 'XTickLabel', []);
        ylabel('Units', 'FontSize', 11);
        title('Clustering', 'FontSize', 12);
        heatmap_left = 0.18;
    else
        heatmap_left = 0.08;
    end

    % Heatmap
    heatmap_width = 0.65;
    subplot('Position', [heatmap_left 0.15 heatmap_width 0.75]);
    imagesc(feature_matrix_sorted);
    colormap(bluewhitered(256));
    caxis([-3 3]);
    axis tight;

    cb = colorbar;
    cb.Position = [heatmap_left + heatmap_width + 0.02 0.15 0.02 0.75];
    ylabel(cb, 'Z-score', 'FontSize', 11);

    xlabel('Features', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Units', 'FontSize', 12, 'FontWeight', 'bold');

    set(gca, 'XTick', 1:length(feature_names));
    set(gca, 'XTickLabel', feature_names);
    set(gca, 'XTickLabelRotation', 45);
    set(gca, 'YTick', []);
    set(gca, 'FontSize', 10);

    % Session color bar
    subplot('Position', [heatmap_left - 0.02 0.15 0.01 0.75]);
    session_colors = zeros(length(unit_sort_idx), 1, 3);
    for i = 1:length(unit_sort_idx)
        if is_aversive_sorted(i)
            session_colors(i, 1, :) = [0.8 0.2 0.2];
        else
            session_colors(i, 1, :) = [0.2 0.2 0.8];
        end
    end
    image(session_colors);
    set(gca, 'XTick', [], 'YTick', []);

    % Title
    n_clusters = max(cluster_assignments(~isnan(cluster_assignments)));
    if isnan(n_clusters)
        n_clusters = 0;
    end
    title_str = sprintf('%s: Simplified Clustering (%d units, %d clusters)', ...
        session_name, size(feature_matrix_sorted, 1), n_clusters);
    annotation('textbox', [heatmap_left 0.92 heatmap_width 0.05], 'String', title_str, ...
        'EdgeColor', 'none', 'FontSize', 14, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');
end

function cmap = bluewhitered(n)
% Create blue-white-red colormap

    if nargin < 1
        n = 256;
    end

    half = ceil(n/2);

    % Blue to white
    r1 = linspace(0, 1, half)';
    g1 = linspace(0, 1, half)';
    b1 = ones(half, 1);

    % White to red
    r2 = ones(half, 1);
    g2 = linspace(1, 0, half)';
    b2 = linspace(1, 0, half)';

    cmap = [r1 g1 b1; r2 g2 b2];
    cmap = cmap(1:n, :);
end

%% ========================================================================
%  FEATURE IMPORTANCE HELPER FUNCTIONS
%% ========================================================================

function [pvalues, effect_sizes, rankings] = compute_global_feature_importance(feature_matrix, cluster_labels, feature_names)
% Compute global feature importance using ANOVA and effect sizes
%
% Outputs:
%   pvalues: p-values from Kruskal-Wallis test for each feature
%   effect_sizes: eta-squared effect sizes
%   rankings: feature indices sorted by effect size (descending)

    n_features = size(feature_matrix, 2);
    pvalues = nan(n_features, 1);
    effect_sizes = nan(n_features, 1);

    unique_clusters = unique(cluster_labels);
    n_clusters = length(unique_clusters);

    for f = 1:n_features
        feature_vals = feature_matrix(:, f);

        % Remove NaN values
        valid_idx = ~isnan(feature_vals);
        if sum(valid_idx) < n_clusters * 2
            continue;
        end

        feature_vals_clean = feature_vals(valid_idx);
        labels_clean = cluster_labels(valid_idx);

        % Kruskal-Wallis test (non-parametric ANOVA)
        try
            pvalues(f) = kruskalwallis(feature_vals_clean, labels_clean, 'off');

            % Compute eta-squared (effect size)
            % eta^2 = SS_between / SS_total
            grand_mean = mean(feature_vals_clean);
            SS_total = sum((feature_vals_clean - grand_mean).^2);

            SS_between = 0;
            for c = 1:n_clusters
                cluster_vals = feature_vals_clean(labels_clean == unique_clusters(c));
                if ~isempty(cluster_vals)
                    cluster_mean = mean(cluster_vals);
                    SS_between = SS_between + length(cluster_vals) * (cluster_mean - grand_mean)^2;
                end
            end

            if SS_total > 0
                effect_sizes(f) = SS_between / SS_total;
            else
                effect_sizes(f) = 0;
            end
        catch
            pvalues(f) = 1;
            effect_sizes(f) = 0;
        end
    end

    % Rank features by effect size (descending)
    [~, rankings] = sort(effect_sizes, 'descend', 'MissingPlacement', 'last');
end

function [discrimination_matrix, top_features_per_pair] = compute_pairwise_discrimination(feature_matrix, cluster_labels, feature_names, n_clusters)
% Compute pairwise cluster discrimination using Cohen's d
%
% Outputs:
%   discrimination_matrix: [n_pairs × n_features] matrix of Cohen's d values
%   top_features_per_pair: cell array of top feature indices for each pair

    n_features = size(feature_matrix, 2);
    n_pairs = n_clusters * (n_clusters - 1) / 2;

    discrimination_matrix = nan(n_pairs, n_features);
    top_features_per_pair = cell(n_pairs, 1);

    pair_idx = 1;
    for c1 = 1:n_clusters
        for c2 = (c1+1):n_clusters
            % Get data for both clusters
            cluster1_data = feature_matrix(cluster_labels == c1, :);
            cluster2_data = feature_matrix(cluster_labels == c2, :);

            % Compute Cohen's d for each feature
            for f = 1:n_features
                vals1 = cluster1_data(:, f);
                vals2 = cluster2_data(:, f);

                % Remove NaNs
                vals1 = vals1(~isnan(vals1));
                vals2 = vals2(~isnan(vals2));

                if length(vals1) < 2 || length(vals2) < 2
                    continue;
                end

                % Cohen's d = (mean1 - mean2) / pooled_std
                mean1 = mean(vals1);
                mean2 = mean(vals2);
                std1 = std(vals1);
                std2 = std(vals2);

                n1 = length(vals1);
                n2 = length(vals2);

                % Pooled standard deviation
                pooled_std = sqrt(((n1-1)*std1^2 + (n2-1)*std2^2) / (n1 + n2 - 2));

                if pooled_std > 0
                    discrimination_matrix(pair_idx, f) = abs((mean1 - mean2) / pooled_std);
                end
            end

            % Find top features for this pair
            [~, top_idx] = sort(discrimination_matrix(pair_idx, :), 'descend', 'MissingPlacement', 'last');
            top_features_per_pair{pair_idx} = top_idx(1:min(10, n_features));

            pair_idx = pair_idx + 1;
        end
    end
end

function create_feature_importance_heatmap(feature_matrix, cluster_labels, feature_names, feature_categories, feature_rankings, session_name)
% Create heatmap showing mean feature values per cluster

    unique_clusters = unique(cluster_labels);
    n_clusters = length(unique_clusters);
    n_features = length(feature_names);

    % Compute mean feature value per cluster
    cluster_means = nan(n_clusters, n_features);
    for c = 1:n_clusters
        cluster_data = feature_matrix(cluster_labels == unique_clusters(c), :);
        cluster_means(c, :) = nanmean(cluster_data, 1);
    end

    % Reorder features by importance
    top_n = min(30, n_features);
    top_features = feature_rankings(1:top_n);
    cluster_means_sorted = cluster_means(:, top_features);
    feature_names_sorted = feature_names(top_features);

    % Create figure
    fig = figure('Position', [100 100 1400 800], ...
        'Name', sprintf('%s - Feature Importance Heatmap', session_name));

    % Heatmap
    subplot('Position', [0.15 0.15 0.70 0.70]);
    imagesc(cluster_means_sorted');
    colormap(bluewhitered(256));
    caxis([-3 3]);

    % Labels
    xlabel('Cluster', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Features (ranked by importance)', 'FontSize', 12, 'FontWeight', 'bold');

    set(gca, 'XTick', 1:n_clusters);
    set(gca, 'XTickLabel', unique_clusters);
    set(gca, 'YTick', 1:top_n);
    set(gca, 'YTickLabel', feature_names_sorted);
    set(gca, 'FontSize', 9);

    % Colorbar
    cb = colorbar;
    cb.Position = [0.87 0.15 0.02 0.70];
    ylabel(cb, 'Mean Z-score', 'FontSize', 11);

    % Title
    title(sprintf('%s: Top %d Features by Cluster', session_name, top_n), ...
        'FontSize', 14, 'FontWeight', 'bold');
end

function create_pairwise_discrimination_heatmap(discrimination_matrix, feature_names, n_clusters, session_name)
% Create heatmap showing pairwise cluster discrimination

    n_features = length(feature_names);
    n_pairs = size(discrimination_matrix, 1);

    % Find top features overall
    mean_discrimination = nanmean(discrimination_matrix, 1);
    [~, top_idx] = sort(mean_discrimination, 'descend', 'MissingPlacement', 'last');
    top_n = min(20, n_features);
    top_features = top_idx(1:top_n);

    % Create pair labels
    pair_labels = cell(n_pairs, 1);
    pair_idx = 1;
    for c1 = 1:n_clusters
        for c2 = (c1+1):n_clusters
            pair_labels{pair_idx} = sprintf('%d vs %d', c1, c2);
            pair_idx = pair_idx + 1;
        end
    end

    % Create figure
    fig = figure('Position', [150 150 1200 900], ...
        'Name', sprintf('%s - Pairwise Discrimination', session_name));

    % Heatmap
    subplot('Position', [0.15 0.10 0.70 0.75]);
    imagesc(discrimination_matrix(:, top_features)');
    colormap(hot);

    % Labels
    xlabel('Cluster Pairs', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Features', 'FontSize', 12, 'FontWeight', 'bold');

    set(gca, 'XTick', 1:n_pairs);
    set(gca, 'XTickLabel', pair_labels);
    set(gca, 'XTickLabelRotation', 45);
    set(gca, 'YTick', 1:top_n);
    set(gca, 'YTickLabel', feature_names(top_features));
    set(gca, 'FontSize', 9);

    % Colorbar
    cb = colorbar;
    cb.Position = [0.87 0.10 0.02 0.75];
    ylabel(cb, 'Cohen''s d', 'FontSize', 11);

    % Title
    title(sprintf('%s: Pairwise Cluster Discrimination (Top %d Features)', session_name, top_n), ...
        'FontSize', 14, 'FontWeight', 'bold');
end

function create_top_features_plot(feature_rankings, feature_names, pvalues, effect_sizes, session_name)
% Create bar plot of top features by effect size

    top_n = min(20, length(feature_rankings));
    top_indices = feature_rankings(1:top_n);

    % Create figure
    fig = figure('Position', [200 200 1000 700], ...
        'Name', sprintf('%s - Top Features', session_name));

    % Bar plot of effect sizes
    subplot(2, 1, 1);
    barh(1:top_n, effect_sizes(top_indices), 'FaceColor', [0.2 0.5 0.8]);
    set(gca, 'YDir', 'reverse');
    set(gca, 'YTick', 1:top_n);
    set(gca, 'YTickLabel', feature_names(top_indices));
    xlabel('Effect Size (η²)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Features', 'FontSize', 11, 'FontWeight', 'bold');
    title('Top Features by Effect Size', 'FontSize', 13, 'FontWeight', 'bold');
    grid on;
    xlim([0 max(effect_sizes(top_indices)) * 1.1]);

    % Add effect size values
    for i = 1:top_n
        text(effect_sizes(top_indices(i)) + 0.01, i, ...
            sprintf('%.3f', effect_sizes(top_indices(i))), ...
            'FontSize', 9, 'VerticalAlignment', 'middle');
    end

    % Bar plot of -log10(p-values)
    subplot(2, 1, 2);
    neg_log_p = -log10(pvalues(top_indices));
    neg_log_p(isinf(neg_log_p)) = 50; % Cap very small p-values

    barh(1:top_n, neg_log_p, 'FaceColor', [0.8 0.3 0.2]);
    set(gca, 'YDir', 'reverse');
    set(gca, 'YTick', 1:top_n);
    set(gca, 'YTickLabel', feature_names(top_indices));
    xlabel('-log₁₀(p-value)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Features', 'FontSize', 11, 'FontWeight', 'bold');
    title('Statistical Significance', 'FontSize', 13, 'FontWeight', 'bold');
    grid on;

    % Add significance threshold line (p=0.05)
    hold on;
    plot([-log10(0.05) -log10(0.05)], [0 top_n+1], 'r--', 'LineWidth', 2);
    text(-log10(0.05), 0.5, '  p=0.05', 'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold');
    hold off;

    % Overall title
    sgtitle(sprintf('%s: Top %d Discriminative Features', session_name, top_n), ...
        'FontSize', 14, 'FontWeight', 'bold');
end

function [rf_importance, rf_rankings] = compute_random_forest_importance(feature_matrix, cluster_labels, feature_names)
% Compute feature importance using Random Forest classifier
%
% Outputs:
%   rf_importance: importance score for each feature
%   rf_rankings: feature indices sorted by importance (descending)

    n_features = size(feature_matrix, 2);
    n_samples = size(feature_matrix, 1);

    % Remove features with too many NaNs
    valid_features = sum(~isnan(feature_matrix), 1) > n_samples * 0.3;
    feature_matrix_clean = feature_matrix(:, valid_features);

    % Remove samples with too many NaNs
    valid_samples = sum(~isnan(feature_matrix_clean), 2) > sum(valid_features) * 0.5;
    feature_matrix_clean = feature_matrix_clean(valid_samples, :);
    cluster_labels_clean = cluster_labels(valid_samples);

    % Impute remaining NaNs with column mean
    for f = 1:size(feature_matrix_clean, 2)
        col = feature_matrix_clean(:, f);
        if any(isnan(col))
            col(isnan(col)) = nanmean(col);
            feature_matrix_clean(:, f) = col;
        end
    end

    % Train Random Forest
    n_trees = 100;
    try
        rf_model = TreeBagger(n_trees, feature_matrix_clean, cluster_labels_clean, ...
            'Method', 'classification', 'OOBPredictorImportance', 'on', 'NumPrint', 0);

        % Get feature importance
        importance_clean = rf_model.OOBPermutedPredictorDeltaError;

        % Map back to original feature indices
        rf_importance = zeros(n_features, 1);
        valid_idx = find(valid_features);
        rf_importance(valid_idx) = importance_clean;
    catch
        warning('Random Forest failed, using uniform importance');
        rf_importance = ones(n_features, 1) / n_features;
    end

    % Rank features by importance
    [~, rf_rankings] = sort(rf_importance, 'descend');
end

function [selected_features, quality_curve, rankings] = perform_rfe_with_validation(feature_matrix, cluster_labels, feature_names, min_features, quality_threshold)
% Recursive Feature Elimination with cluster quality validation
%
% Outputs:
%   selected_features: indices of selected features
%   quality_curve: cluster quality at each step
%   rankings: elimination order (most important = eliminated last)

    n_features = size(feature_matrix, 2);
    n_samples = size(feature_matrix, 1);

    % Clean data
    valid_features = sum(~isnan(feature_matrix), 1) > n_samples * 0.3;
    feature_matrix_clean = feature_matrix(:, valid_features);
    feature_names_valid = feature_names;
    feature_names_valid = feature_names_valid(valid_features);

    valid_samples = sum(~isnan(feature_matrix_clean), 2) > sum(valid_features) * 0.5;
    feature_matrix_clean = feature_matrix_clean(valid_samples, :);
    cluster_labels_clean = cluster_labels(valid_samples);

    % Impute NaNs
    for f = 1:size(feature_matrix_clean, 2)
        col = feature_matrix_clean(:, f);
        if any(isnan(col))
            col(isnan(col)) = nanmean(col);
            feature_matrix_clean(:, f) = col;
        end
    end

    % Initial feature set
    remaining_features = 1:size(feature_matrix_clean, 2);
    elimination_order = [];
    quality_curve = [];

    % Baseline quality (all features)
    baseline_quality = compute_cluster_quality(feature_matrix_clean, cluster_labels_clean);
    quality_curve(1) = baseline_quality;

    fprintf('    Baseline quality: %.4f\n', baseline_quality);

    % RFE loop
    while length(remaining_features) > min_features
        % Train RF on current features
        current_matrix = feature_matrix_clean(:, remaining_features);

        try
            rf_model = TreeBagger(50, current_matrix, cluster_labels_clean, ...
                'Method', 'classification', 'OOBPredictorImportance', 'on', 'NumPrint', 0);
            importances = rf_model.OOBPermutedPredictorDeltaError;
        catch
            % If RF fails, use univariate importance
            importances = ones(length(remaining_features), 1);
            for f = 1:length(remaining_features)
                feat_vals = current_matrix(:, f);
                if std(feat_vals) > 0
                    [~, p] = kruskalwallis(feat_vals, cluster_labels_clean, 'off');
                    importances(f) = -log10(p + 1e-10);
                end
            end
        end

        % Remove least important feature
        [~, least_important_idx] = min(importances);
        fprintf('    Removed %d features: %s \n', remaining_features(least_important_idx), feature_names_valid{remaining_features(least_important_idx)});
        elimination_order(end+1) = remaining_features(least_important_idx);
        remaining_features(least_important_idx) = [];
        

        % Evaluate quality after removal
        current_matrix = feature_matrix_clean(:, remaining_features);
        current_quality = compute_cluster_quality(current_matrix, cluster_labels_clean);
        quality_curve(end+1) = current_quality;

        % Check if quality degraded too much
        quality_drop = (baseline_quality - current_quality) / baseline_quality;

        if mod(length(remaining_features), 5) == 0 || quality_drop > quality_threshold
            fprintf('    %d features: quality=%.4f (drop=%.1f%%)\n', ...
                length(remaining_features), current_quality, quality_drop * 100);
        end

        % Stop if quality degrades beyond threshold
        if quality_drop > quality_threshold
            fprintf('    Quality threshold exceeded, stopping RFE\n');
            % Add back the last eliminated feature
            remaining_features = [remaining_features, eliminated_feature];
            elimination_order(end) = [];
            quality_curve(end) = [];
            break;
        end
    end

    % Map back to original feature indices
    valid_idx = find(valid_features);
    selected_features = valid_idx(remaining_features);
    eliminated_features = valid_idx(elimination_order(end:-1:1));

    % Rankings: features eliminated last are most important
    rankings = [remaining_features, eliminated_features];

    fprintf('    Selected %d features with quality %.4f\n', length(selected_features), quality_curve(end));
end

function quality = compute_cluster_quality(feature_matrix, cluster_labels)
% Compute cluster quality using silhouette score

    try
        % Compute pairwise distances
        distances = pdist(feature_matrix, 'euclidean');

        % Compute silhouette values
        silhouette_vals = silhouette([], cluster_labels, distances);

        % Average silhouette score
        quality = mean(silhouette_vals);
    catch
        % Fallback: use simpler metric
        unique_clusters = unique(cluster_labels);
        within_var = 0;
        between_var = 0;

        grand_mean = mean(feature_matrix, 1);

        for c = 1:length(unique_clusters)
            cluster_data = feature_matrix(cluster_labels == unique_clusters(c), :);
            cluster_mean = mean(cluster_data, 1);
            within_var = within_var + sum(sum((cluster_data - cluster_mean).^2));
            between_var = between_var + size(cluster_data, 1) * sum((cluster_mean - grand_mean).^2);
        end

        % Variance ratio (higher is better)
        quality = between_var / (between_var + within_var + eps);
    end
end

function create_rfe_quality_plot(quality_curve, selected_features, feature_names, session_name)
% Plot RFE quality curve showing cluster quality vs number of features

    n_points = length(quality_curve);
    n_features_curve = (size(feature_names, 2)):-1:(size(feature_names, 2) - n_points + 1);

    fig = figure('Position', [250 250 1000 600], ...
        'Name', sprintf('%s - RFE Quality Curve', session_name));

    % Quality curve
    plot(n_features_curve, quality_curve, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
    hold on;

    % Mark selected point
    plot(length(selected_features), quality_curve(end), 'ro', ...
        'MarkerSize', 12, 'LineWidth', 3);

    % Add quality drop threshold line
    baseline = quality_curve(1);
    threshold_line = baseline * 0.95; % 5% drop
    plot([min(n_features_curve) max(n_features_curve)], [threshold_line threshold_line], ...
        'r--', 'LineWidth', 1.5);

    hold off;
    grid on;

    xlabel('Number of Features', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Cluster Quality (Silhouette Score)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('%s: RFE Quality Curve (Selected %d features)', session_name, length(selected_features)), ...
        'FontSize', 14, 'FontWeight', 'bold');

    legend({'Quality Curve', sprintf('Selected (%d features)', length(selected_features)), ...
        '5% Quality Threshold'}, 'Location', 'best');
end

function create_feature_comparison_plot(univ_rankings, rf_rankings, rfe_rankings, feature_names, effect_sizes, rf_importance, session_name)
% Create comparison plot showing top features from all three methods

    top_n = 15;

    % Get top features from each method (ensure column vectors)
    univ_top = univ_rankings(1:min(top_n, length(univ_rankings)));
    univ_top = univ_top(:);  % Force column vector

    rf_top = rf_rankings(1:min(top_n, length(rf_rankings)));
    rf_top = rf_top(:);  % Force column vector

    rfe_top = rfe_rankings(1:min(top_n, length(rfe_rankings)));
    rfe_top = rfe_top(:);  % Force column vector

    % Union of all top features
    all_top_features = unique([univ_top; rf_top; rfe_top]);
    n_shown = length(all_top_features);

    % Create ranking matrix (lower rank = better)
    ranking_matrix = nan(n_shown, 3);

    for i = 1:n_shown
        feat_idx = all_top_features(i);

        % Find rank in each method
        univ_rank = find(univ_rankings == feat_idx);
        rf_rank = find(rf_rankings == feat_idx);
        rfe_rank = find(rfe_rankings == feat_idx);

        if ~isempty(univ_rank), ranking_matrix(i, 1) = univ_rank; end
        if ~isempty(rf_rank), ranking_matrix(i, 2) = rf_rank; end
        if ~isempty(rfe_rank), ranking_matrix(i, 3) = rfe_rank; end
    end

    % Create figure
    fig = figure('Position', [300 300 1200 800], ...
        'Name', sprintf('%s - Feature Selection Comparison', session_name));

    % Heatmap of rankings
    imagesc(ranking_matrix');
    colormap(flipud(hot));

    xlabel('Features', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Method', 'FontSize', 12, 'FontWeight', 'bold');

    set(gca, 'YTick', 1:3);
    set(gca, 'YTickLabel', {'Univariate', 'RandomForest', 'RFE'});
    set(gca, 'XTick', 1:n_shown);
    set(gca, 'XTickLabel', feature_names(all_top_features));
    set(gca, 'XTickLabelRotation', 90);
    set(gca, 'FontSize', 9);

    cb = colorbar;
    ylabel(cb, 'Rank (lower = better)', 'FontSize', 11);

    title(sprintf('%s: Feature Selection Method Comparison', session_name), ...
        'FontSize', 14, 'FontWeight', 'bold');
end
