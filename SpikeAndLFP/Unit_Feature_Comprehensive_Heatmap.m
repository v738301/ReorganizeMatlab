%% ========================================================================
%  COMPREHENSIVE UNIT FEATURE HEATMAP
%  Creates a single large heatmap showing all features for all units
%  with multiple sorting and clustering options
%% ========================================================================
%
%  This script creates a comprehensive visualization of all unit features:
%  - Coherence features (narrow + broad bands)
%  - Phase coupling features (MRL for all band × behavior combinations)
%  - PSTH response features (key events)
%
%  Sorting options:
%  1. By session type (Aversive vs Reward)
%  2. Hierarchical clustering (unsupervised)
%  3. By specific feature values
%  4. By principal components
%
%  Input: unit_features_comprehensive.mat
%  Output: Comprehensive heatmap figure
%
%% ========================================================================

clear all;
% close all;

%% ========================================================================
%  SECTION 1: CONFIGURATION
%% ========================================================================

fprintf('=== COMPREHENSIVE UNIT FEATURE HEATMAP ===\n\n');

% Visualization configuration
config = struct();
config.cluster_dimension = 'units';  % Options: 'units', 'features', 'both'
config.sort_method = 'hierarchical';  % Options: 'session_type', 'hierarchical', 'pca', 'feature'
config.feature_to_sort = [];  % Used if sort_method = 'feature'
config.show_dendrogram = true;  % Show hierarchical clustering dendrogram
config.normalize_features = true;  % Z-score normalize each feature column
config.colormap_name = 'bluewhitered';  % 'bluewhitered', 'jet', 'parula', 'redblue'
config.exclude_categories = {};  % Feature categories to exclude, e.g., {'PSTH', 'Coherence'}
config.separate_by_session = false;  % Analyze reward and aversive sessions separately

fprintf('Configuration:\n');
fprintf('  Cluster dimension: %s\n', config.cluster_dimension);
fprintf('  Sort method: %s\n', config.sort_method);
fprintf('  Separate by session: %d\n', config.separate_by_session);
fprintf('  Normalize features: %d\n', config.normalize_features);
fprintf('  Colormap: %s\n', config.colormap_name);
if ~isempty(config.exclude_categories)
    fprintf('  Exclude categories: %s\n', strjoin(config.exclude_categories, ', '));
end
fprintf('\n');

%% ========================================================================
%  SECTION 2: LOAD DATA
%% ========================================================================

fprintf('Loading feature data...\n');

try
    load('unit_features_comprehensive.mat', 'unit_features_comprehensive');
    fprintf('✓ Loaded feature data\n\n');
catch ME
    fprintf('❌ Failed to load: %s\n', ME.message);
    fprintf('Please run Unit_Feature_Extraction.m first\n');
    return;
end

% Extract components
coherence_features = unit_features_comprehensive.coherence_features;
phase_narrow_features = unit_features_comprehensive.phase_narrow_features;
phase_broad_features = unit_features_comprehensive.phase_broad_features;
psth_features = unit_features_comprehensive.psth_features;
feat_config = unit_features_comprehensive.config;

n_units_coherence = length(coherence_features);
n_units_phase_narrow = length(phase_narrow_features);
n_units_phase_broad = length(phase_broad_features);
n_units_psth = length(psth_features);

fprintf('Available units:\n');
fprintf('  Coherence: %d units\n', n_units_coherence);
fprintf('  Phase narrow: %d units\n', n_units_phase_narrow);
fprintf('  Phase broad: %d units\n', n_units_phase_broad);
fprintf('  PSTH: %d units\n\n', n_units_psth);

% Use coherence features as the base (most complete)
n_units = n_units_coherence;

%% ========================================================================
%  SECTION 3: BUILD COMPREHENSIVE FEATURE MATRIX
%% ========================================================================

fprintf('Building comprehensive feature matrix...\n');

% Initialize storage
all_features = [];
feature_names = {};
feature_categories = {};  % Track which category each feature belongs to

% ----- COHERENCE FEATURES -----
fprintf('  Adding coherence features...\n');

% Narrow band coherence (3 features)
coherence_narrow_names = {'Coh_1-3Hz', 'Coh_5-7Hz', 'Coh_8-10Hz'};
coherence_narrow_vars = {'coherence_1_3Hz', 'coherence_5_7Hz', 'coherence_8_10Hz'};

for i = 1:3
    all_features(:, end+1) = [coherence_features.(coherence_narrow_vars{i})]';
    feature_names{end+1} = coherence_narrow_names{i};
    feature_categories{end+1} = 'Coherence';
end

% Broad band coherence (6 features)
coherence_broad_names = {'Coh_Delta', 'Coh_Theta', 'Coh_Beta', 'Coh_LowGamma', 'Coh_HighGamma', 'Coh_UltraGamma'};
coherence_broad_vars = {'coherence_delta', 'coherence_theta', 'coherence_beta', ...
                        'coherence_low_gamma', 'coherence_high_gamma', 'coherence_ultra_gamma'};

for i = 1:6
    all_features(:, end+1) = [coherence_features.(coherence_broad_vars{i})]';
    feature_names{end+1} = coherence_broad_names{i};
    feature_categories{end+1} = 'Coherence';
end

% Peak coherence (2 features)
all_features(:, end+1) = [coherence_features.coherence_peak_freq]';
feature_names{end+1} = 'Coh_PeakFreq';
feature_categories{end+1} = 'Coherence';

all_features(:, end+1) = [coherence_features.coherence_peak_mag]';
feature_names{end+1} = 'Coh_PeakMag';
feature_categories{end+1} = 'Coherence';

fprintf('    Added %d coherence features\n', 11);

% ----- PHASE COUPLING FEATURES (NARROW BANDS) -----
fprintf('  Adding narrow-band phase coupling features...\n');

narrow_bands = feat_config.narrow_bands;  % {'1-3Hz', '5-7Hz', '8-10Hz'}
behaviors = feat_config.behavior_names;    % 7 behaviors

% Extract MRL for each band × behavior (3 × 7 = 21 features)
for band_idx = 1:length(narrow_bands)
    for beh_idx = 1:length(behaviors)
        % Extract MRL from all units
        mrl_values = nan(n_units, 1);
        for u = 1:min(n_units, n_units_phase_narrow)
            mrl_values(u) = phase_narrow_features(u).phase_MRL_narrow(band_idx, beh_idx);
        end

        all_features(:, end+1) = mrl_values;

        % Create feature name
        band_clean = strrep(narrow_bands{band_idx}, '-', '_');
        beh_clean = strrep(behaviors{beh_idx}, '/', '_');
        beh_clean = strrep(beh_clean, ' ', '');
        feature_names{end+1} = sprintf('MRL_N_%s_%s', band_clean, beh_clean);
        feature_categories{end+1} = 'Phase_Narrow';
    end
end

fprintf('    Added %d narrow-band phase features\n', length(narrow_bands) * length(behaviors));

% ----- PHASE COUPLING FEATURES (BROAD BANDS) -----
fprintf('  Adding broad-band phase coupling features...\n');

broad_bands = feat_config.broad_bands;  % {'Delta', 'Theta', 'Beta', 'Low_Gamma', 'High_Gamma', 'Ultra_Gamma'}

% Extract MRL for each band × behavior (6 × 7 = 42 features)
for band_idx = 1:length(broad_bands)
    for beh_idx = 1:length(behaviors)
        % Extract MRL from all units
        mrl_values = nan(n_units, 1);
        for u = 1:min(n_units, n_units_phase_broad)
            mrl_values(u) = phase_broad_features(u).phase_MRL_broad(band_idx, beh_idx);
        end

        all_features(:, end+1) = mrl_values;

        % Create feature name
        beh_clean = strrep(behaviors{beh_idx}, '/', '_');
        beh_clean = strrep(beh_clean, ' ', '');
        feature_names{end+1} = sprintf('MRL_B_%s_%s', broad_bands{band_idx}, beh_clean);
        feature_categories{end+1} = 'Phase_Broad';
    end
end

fprintf('    Added %d broad-band phase features\n', length(broad_bands) * length(behaviors));

% ----- PSTH FEATURES (KEY EVENTS) -----
fprintf('  Adding PSTH response features...\n');

if ~isempty(psth_features)
    % Key events to include
    key_events = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON', 'AversiveOnset', ...
                  'Beh1_Onset', 'Beh2_Onset', 'Beh7_Onset', 'MovementOnset'};
    key_event_labels = {'IR1', 'IR2', 'WP1', 'WP2', 'Aversive', ...
                        'Reward', 'Walking', 'Standing', 'Movement'};

    psth_feature_count = 0;
    for e = 1:length(key_events)
        event_name = key_events{e};
        mean_z_field = [event_name '_mean_z_0to1sec'];
        peak_z_field = [event_name '_peak_z'];

        % Add mean Z-score if available
        if isfield(psth_features, mean_z_field)
            psth_values = nan(n_units, 1);
            for u = 1:min(n_units, n_units_psth)
                psth_values(u) = psth_features(u).(mean_z_field);
            end
            all_features(:, end+1) = psth_values;
            feature_names{end+1} = sprintf('PSTH_%s_MeanZ', key_event_labels{e});
            feature_categories{end+1} = 'PSTH';
            psth_feature_count = psth_feature_count + 1;
        end

        % Add peak Z-score if available
        if isfield(psth_features, peak_z_field)
            psth_values = nan(n_units, 1);
            for u = 1:min(n_units, n_units_psth)
                psth_values(u) = psth_features(u).(peak_z_field);
            end
            all_features(:, end+1) = psth_values;
            feature_names{end+1} = sprintf('PSTH_%s_PeakZ', key_event_labels{e});
            feature_categories{end+1} = 'PSTH';
            psth_feature_count = psth_feature_count + 1;
        end
    end
    fprintf('    Added %d PSTH features\n', psth_feature_count);
else
    fprintf('    Skipped PSTH features (no data)\n');
end

fprintf('\n✓ Feature matrix built: %d units × %d features\n\n', size(all_features, 1), size(all_features, 2));

%% ========================================================================
%  SECTION 4: FILTER FEATURE CATEGORIES (OPTIONAL)
%% ========================================================================

if ~isempty(config.exclude_categories)
    fprintf('Filtering feature categories...\n');
    fprintf('  Excluding: %s\n', strjoin(config.exclude_categories, ', '));

    % Find features to keep (not in excluded categories)
    features_to_keep = true(size(feature_categories));
    for i = 1:length(feature_categories)
        if any(strcmp(feature_categories{i}, config.exclude_categories))
            features_to_keep(i) = false;
        end
    end

    n_excluded = sum(~features_to_keep);
    fprintf('  Removing %d features from excluded categories\n', n_excluded);

    % Filter the feature matrix and metadata
    all_features = all_features(:, features_to_keep);
    feature_names = feature_names(features_to_keep);
    feature_categories = feature_categories(features_to_keep);

    fprintf('✓ Filtered to %d features\n\n', size(all_features, 2));
else
    fprintf('No feature categories excluded\n\n');
end

%% ========================================================================
%  SECTION 5: NORMALIZE FEATURES (OPTIONAL)
%% ========================================================================

if config.normalize_features
    fprintf('Normalizing features (z-score)...\n');
    all_features_normalized = nan(size(all_features));

    for f = 1:size(all_features, 2)
        feature_col = all_features(:, f);
        valid_data = feature_col(~isnan(feature_col));

        if ~isempty(valid_data) && std(valid_data) > 0
            % Z-score normalization
            all_features_normalized(:, f) = (feature_col - mean(valid_data)) / std(valid_data);
        else
            all_features_normalized(:, f) = feature_col;
        end
    end

    feature_matrix = all_features_normalized;
    fprintf('✓ Features normalized\n\n');
else
    feature_matrix = all_features;
end

%% ========================================================================
%  SECTION 6: DETERMINE SORTING ORDER (UNITS AND/OR FEATURES)
%% ========================================================================

fprintf('Determining sorting order...\n');
fprintf('  Cluster dimension: %s\n', config.cluster_dimension);
fprintf('  Sort method: %s\n', config.sort_method);
fprintf('  Separate by session: %d\n', config.separate_by_session);

% Get session types for all units
session_types = {coherence_features.session_type};
is_aversive = contains(session_types, 'Aversive');

% If analyzing separately, we'll need to store results for each session type
if config.separate_by_session
    session_results = struct();
    session_types_list = {'Aversive', 'Reward'};

    for s = 1:length(session_types_list)
        sess_name = session_types_list{s};
        fprintf('\n--- Processing %s sessions ---\n', sess_name);

        % Select units for this session type
        if strcmp(sess_name, 'Aversive')
            sess_units = is_aversive;
        else
            sess_units = ~is_aversive;
        end

        fprintf('  %d units in %s sessions\n', sum(sess_units), sess_name);

        % Extract subset of feature matrix
        sess_feature_matrix = feature_matrix(sess_units, :);
        sess_session_types = session_types(sess_units);
        sess_is_aversive = is_aversive(sess_units);

        % Perform clustering/sorting for this session
        [unit_sort_idx, feature_sort_idx, unit_linkage_tree, feature_linkage_tree] = ...
            perform_sorting(sess_feature_matrix, sess_is_aversive, feature_names, config);

        % Store results
        session_results(s).session_name = sess_name;
        session_results(s).unit_indices = find(sess_units);  % Original indices
        session_results(s).unit_sort_idx = unit_sort_idx;
        session_results(s).feature_sort_idx = feature_sort_idx;
        session_results(s).unit_linkage_tree = unit_linkage_tree;
        session_results(s).feature_linkage_tree = feature_linkage_tree;
        session_results(s).feature_matrix = sess_feature_matrix;
        session_results(s).session_types = sess_session_types;
        session_results(s).is_aversive = sess_is_aversive;
    end

    fprintf('\n✓ Sorting order determined for both session types\n\n');

else
    % Original behavior: analyze all units together
    fprintf('  Analyzing all sessions together\n');

    % Initialize
    unit_sort_idx = 1:size(feature_matrix, 1);
    feature_sort_idx = 1:size(feature_matrix, 2);
    unit_linkage_tree = [];
    feature_linkage_tree = [];

% --- UNIT SORTING (only if clustering units or both) ---
if strcmp(config.cluster_dimension, 'units') || strcmp(config.cluster_dimension, 'both')
    fprintf('  Clustering/sorting units...\n');

    switch config.sort_method
        case 'session_type'
            % Simple sort: Aversive first, then Reward
            aversive_idx = find(is_aversive);
            reward_idx = find(~is_aversive);
            unit_sort_idx = [aversive_idx, reward_idx];

        case 'hierarchical'
            % Hierarchical clustering of units
            % Remove units with too many NaNs
            valid_units = sum(~isnan(feature_matrix), 2) > size(feature_matrix, 2) * 0.5;
            feature_matrix_clean = feature_matrix(valid_units, :);

            % Replace remaining NaNs with column mean
            for f = 1:size(feature_matrix_clean, 2)
                col = feature_matrix_clean(:, f);
                col(isnan(col)) = nanmean(col);
                feature_matrix_clean(:, f) = col;
            end

            % Compute distance and linkage
            distances = pdist(feature_matrix_clean, 'euclidean');
            unit_linkage_tree = linkage(distances, 'ward');

            % Get dendrogram order
            [~, ~, sort_idx_clean] = dendrogram(unit_linkage_tree, 0);

            % Map back to original indices
            valid_idx = find(valid_units);
            unit_sort_idx = valid_idx(sort_idx_clean);

        case 'pca'
            % Sort by first principal component
            % Remove NaNs
            valid_units = sum(~isnan(feature_matrix), 2) > size(feature_matrix, 2) * 0.5;
            feature_matrix_clean = feature_matrix(valid_units, :);

            for f = 1:size(feature_matrix_clean, 2)
                col = feature_matrix_clean(:, f);
                col(isnan(col)) = nanmean(col);
                feature_matrix_clean(:, f) = col;
            end

            % PCA
            [coeff, score, ~] = pca(feature_matrix_clean);

            % Sort by PC1
            [~, sort_idx_clean] = sort(score(:, 1));

            % Map back to original indices
            valid_idx = find(valid_units);
            unit_sort_idx = valid_idx(sort_idx_clean);

        case 'feature'
            % Sort by specific feature
            feature_idx = find(strcmp(feature_names, config.feature_to_sort));
            if isempty(feature_idx)
                fprintf('    WARNING: Feature %s not found, using default sort\n', config.feature_to_sort);
                [~, unit_sort_idx] = sortrows([is_aversive',feature_matrix], [0,3]+1 ,'descend', 'MissingPlacement', 'last');
            else
                [~, unit_sort_idx] = sort(feature_matrix(:, feature_idx), 'descend', 'MissingPlacement', 'last');
            end

        otherwise
            % Default: session type
            aversive_idx = find(is_aversive);
            reward_idx = find(~is_aversive);
            unit_sort_idx = [aversive_idx, reward_idx];
    end
    fprintf('    ✓ Units sorted\n');
end

% --- FEATURE SORTING (only if clustering features or both) ---
if strcmp(config.cluster_dimension, 'features') || strcmp(config.cluster_dimension, 'both')
    fprintf('  Clustering features...\n');

    % Hierarchical clustering of features
    % Transpose the matrix (features as rows)
    % Remove features with too many NaNs
    valid_features = sum(~isnan(feature_matrix), 1) > size(feature_matrix, 1) * 0.5;
    feature_matrix_transposed = feature_matrix(:, valid_features)';

    % Replace remaining NaNs with row mean
    for u = 1:size(feature_matrix_transposed, 2)
        col = feature_matrix_transposed(:, u);
        col(isnan(col)) = nanmean(col);
        feature_matrix_transposed(:, u) = col;
    end

    % Compute distance and linkage
    distances_features = pdist(feature_matrix_transposed, 'euclidean');
    feature_linkage_tree = linkage(distances_features, 'ward');

    % Get dendrogram order
    [~, ~, feature_sort_idx_clean] = dendrogram(feature_linkage_tree, 0);

    % Map back to original indices
    valid_feat_idx = find(valid_features);
    feature_sort_idx = valid_feat_idx(feature_sort_idx_clean);
    fprintf('    ✓ Features sorted\n');
end

    fprintf('✓ Sorting order determined\n\n');
end  % End of else block (combined analysis)

%% ========================================================================
%  SECTION 7: CREATE COMPREHENSIVE HEATMAP
%% ========================================================================

fprintf('Creating comprehensive heatmap...\n');

% Determine how many plots to create
if config.separate_by_session
    n_plots = 2;  % One for each session type
    plot_names = {'Aversive', 'Reward'};
else
    n_plots = 1;  % Combined plot
    plot_names = {'Combined'};
end

for plot_idx = 1:n_plots

    if config.separate_by_session
        fprintf('  Creating heatmap for %s sessions...\n', plot_names{plot_idx});

        % Get data for this session type
        sess_data = session_results(plot_idx);
        feature_matrix_plot = sess_data.feature_matrix;
        unit_sort_idx = sess_data.unit_sort_idx;
        feature_sort_idx = sess_data.feature_sort_idx;
        unit_linkage_tree = sess_data.unit_linkage_tree;
        feature_linkage_tree = sess_data.feature_linkage_tree;
        session_types_plot = sess_data.session_types;
        is_aversive_plot = sess_data.is_aversive;

        % Reorder matrix
        feature_matrix_sorted = feature_matrix_plot(unit_sort_idx, feature_sort_idx);
        session_types_sorted = session_types_plot(unit_sort_idx);
        is_aversive_sorted = is_aversive_plot(unit_sort_idx);
        feature_names_sorted = feature_names(feature_sort_idx);
        feature_categories_sorted = feature_categories(feature_sort_idx);

        % Create figure
        fig = figure('Position', [50 + (plot_idx-1)*100 50 + (plot_idx-1)*50 2000 1200], ...
                     'Name', sprintf('%s Sessions - Unit Feature Heatmap', plot_names{plot_idx}));
    else
        % Combined analysis - original behavior
        feature_matrix_sorted = feature_matrix(unit_sort_idx, feature_sort_idx);
        session_types_sorted = session_types(unit_sort_idx);
        is_aversive_sorted = is_aversive(unit_sort_idx);
        feature_names_sorted = feature_names(feature_sort_idx);
        feature_categories_sorted = feature_categories(feature_sort_idx);

        % Create figure
        fig = figure('Position', [50 50 2000 1200], 'Name', 'Comprehensive Unit Feature Heatmap');
    end

% Determine subplot positions based on clustering dimension
if strcmp(config.cluster_dimension, 'both')
    % Both dendrograms
    has_unit_dendrogram = config.show_dendrogram && ~isempty(unit_linkage_tree);
    has_feature_dendrogram = config.show_dendrogram && ~isempty(feature_linkage_tree);

    heatmap_left = 0.17;
    heatmap_bottom = 0.15;
    heatmap_width = 0.65;
    heatmap_height = 0.65;

elseif strcmp(config.cluster_dimension, 'units')
    % Only unit dendrogram on left
    has_unit_dendrogram = config.show_dendrogram && ~isempty(unit_linkage_tree);
    has_feature_dendrogram = false;

    if has_unit_dendrogram
        heatmap_left = 0.17;
    else
        heatmap_left = 0.08;
    end
    heatmap_bottom = 0.15;
    heatmap_width = 0.70;
    heatmap_height = 0.75;

else  % 'features'
    % Only feature dendrogram on top
    has_unit_dendrogram = false;
    has_feature_dendrogram = config.show_dendrogram && ~isempty(feature_linkage_tree);

    heatmap_left = 0.08;
    if has_feature_dendrogram
        heatmap_bottom = 0.15;
        heatmap_height = 0.65;
    else
        heatmap_bottom = 0.15;
        heatmap_height = 0.75;
    end
    heatmap_width = 0.78;
end

% --- Draw Unit Dendrogram (if applicable) ---
if has_unit_dendrogram
    subplot('Position', [0.05 heatmap_bottom 0.10 heatmap_height]);
    dendrogram(unit_linkage_tree, 0, 'Orientation', 'left');
    set(gca, 'YDir', 'reverse');
    set(gca, 'XTickLabel', []);
    ylabel('Units');
    title('Unit Clustering');
end

% --- Draw Feature Dendrogram (if applicable) ---
if has_feature_dendrogram
    subplot('Position', [heatmap_left 0.82 heatmap_width 0.10]);
    dendrogram(feature_linkage_tree, 0, 'Orientation', 'top');
    set(gca, 'XDir', 'normal');
    set(gca, 'YTickLabel', []);
    xlabel('Features');
    title('Feature Clustering');
end

% --- Draw Main Heatmap ---
subplot('Position', [heatmap_left heatmap_bottom heatmap_width heatmap_height]);

% Plot heatmap (units on Y-axis, features on X-axis)
imagesc(feature_matrix_sorted);
axis tight;

% Set colormap
switch config.colormap_name
    case 'bluewhitered'
        colormap(bluewhitered(256));
        if config.normalize_features
            caxis([-3 3]);  % For z-scored data
        end
    case 'redblue'
        colormap(redblue(256));
        if config.normalize_features
            caxis([-3 3]);
        end
    otherwise
        colormap(config.colormap_name);
end

% Add colorbar
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

% X-axis: Feature names (rotated for readability)
set(gca, 'XTick', 1:length(feature_names_sorted));
set(gca, 'XTickLabel', feature_names_sorted);
set(gca, 'XTickLabelRotation', 90);
set(gca, 'FontSize', 8);

% Y-axis: Unit indices
set(gca, 'YTick', []);

% --- Add session type color bar on LEFT (for units on Y-axis) ---
if has_unit_dendrogram
    session_bar_left = 0.15;
else
    session_bar_left = 0.06;
end
subplot('Position', [session_bar_left heatmap_bottom 0.01 heatmap_height]);
session_type_colors = zeros(length(unit_sort_idx), 1, 3);
for i = 1:length(unit_sort_idx)
    if is_aversive_sorted(i)
        session_type_colors(i, 1, :) = [0.8 0.2 0.2];  % Red for aversive
    else
        session_type_colors(i, 1, :) = [0.2 0.2 0.8];  % Blue for reward
    end
end
image(session_type_colors);
set(gca, 'XTick', [], 'YTick', []);
xlabel('Session', 'FontSize', 9, 'Rotation', 0);

% --- Add feature category color bar (position depends on dendrogram) ---
if has_feature_dendrogram
    category_bar_bottom = 0.80;
else
    category_bar_bottom = 0.91;
end
subplot('Position', [heatmap_left category_bar_bottom heatmap_width 0.01]);
unique_categories = unique(feature_categories, 'stable');
category_colors = lines(length(unique_categories));
feature_category_img = zeros(1, length(feature_names_sorted), 3);

for i = 1:length(feature_names_sorted)
    cat_idx = find(strcmp(unique_categories, feature_categories_sorted{i}));
    feature_category_img(1, i, :) = category_colors(cat_idx, :);
end

image(feature_category_img);
set(gca, 'XTick', [], 'YTick', []);
ylabel('Category', 'FontSize', 9, 'Rotation', 0, 'HorizontalAlignment', 'right');

% --- Add title ---
if has_feature_dendrogram
    title_bottom = 0.94;
else
    title_bottom = 0.94;
end

if config.separate_by_session
    title_str = sprintf('%s Sessions: %d units × %d features | Cluster: %s | Sort: %s', ...
        plot_names{plot_idx}, size(feature_matrix_sorted, 1), size(feature_matrix_sorted, 2), ...
        config.cluster_dimension, config.sort_method);
else
    title_str = sprintf('Comprehensive Unit Features (%d units × %d features) | Cluster: %s | Sort: %s', ...
        size(feature_matrix_sorted, 1), size(feature_matrix_sorted, 2), ...
        config.cluster_dimension, config.sort_method);
end

annotation('textbox', [heatmap_left title_bottom heatmap_width 0.05], 'String', title_str, ...
    'EdgeColor', 'none', 'FontSize', 14, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');

% --- Add legend for categories ---
legend_str = sprintf('Categories: ');
for i = 1:length(unique_categories)
    legend_str = [legend_str, sprintf('%s | ', unique_categories{i})];
end
annotation('textbox', [0.05 0.05 0.90 0.05], 'String', legend_str, ...
    'EdgeColor', 'none', 'FontSize', 10, 'HorizontalAlignment', 'center');

end  % End of for loop over plots (separate sessions or combined)

fprintf('✓ Heatmap(s) created\n\n');

%% extra figure;
figure; 
scatter(feature_matrix(is_aversive,2),feature_matrix(is_aversive,3),12,feature_matrix(is_aversive,83),'fill')
colormap turbo
clim([-2,2])
xlim([-2,2])
ylim([-2,2])

%% ========================================================================
%  SECTION 8: SAVE RESULTS
%% ========================================================================

% fprintf('Saving results...\n');
%
% % Save figure
% saveas(fig, sprintf('Unit_Features_Comprehensive_Heatmap_%s.png', config.sort_method));
% fprintf('✓ Saved figure\n');
%
% % Save feature matrix and metadata
% save('unit_features_matrix.mat', 'feature_matrix', 'all_features', 'feature_names', ...
%      'feature_categories', 'sort_idx', 'session_types', 'is_aversive', '-v7.3');
% fprintf('✓ Saved feature matrix to: unit_features_matrix.mat\n');

%% ========================================================================
%  SECTION 9: SUMMARY STATISTICS
%% ========================================================================

fprintf('\n=== FEATURE MATRIX SUMMARY ===\n');
fprintf('Dimensions: %d units × %d features\n', size(feature_matrix, 1), size(feature_matrix, 2));
fprintf('\nFeature breakdown:\n');

for i = 1:length(unique_categories)
    cat_name = unique_categories{i};
    n_features = sum(strcmp(feature_categories, cat_name));
    fprintf('  %s: %d features\n', cat_name, n_features);
end

fprintf('\nSession types:\n');
fprintf('  Aversive: %d units (%.1f%%)\n', sum(is_aversive), 100*sum(is_aversive)/length(is_aversive));
fprintf('  Reward: %d units (%.1f%%)\n', sum(~is_aversive), 100*sum(~is_aversive)/length(is_aversive));

fprintf('\nData completeness:\n');
fprintf('  Average %% non-NaN per unit: %.1f%%\n', ...
    100 * mean(sum(~isnan(feature_matrix), 2) / size(feature_matrix, 2)));

fprintf('\n========================================\n');
fprintf('COMPREHENSIVE HEATMAP COMPLETE!\n');
fprintf('========================================\n');

%% ========================================================================
%  HELPER FUNCTIONS
%% ========================================================================

function [unit_sort_idx, feature_sort_idx, unit_linkage_tree, feature_linkage_tree] = ...
    perform_sorting(feature_matrix, is_aversive, feature_names, config)
% Helper function to perform clustering/sorting on a feature matrix
%
% Inputs:
%   feature_matrix - Units × Features matrix
%   is_aversive - Boolean array indicating aversive sessions
%   feature_names - Cell array of feature names
%   config - Configuration struct
%
% Outputs:
%   unit_sort_idx - Sorting indices for units
%   feature_sort_idx - Sorting indices for features
%   unit_linkage_tree - Linkage tree for unit clustering (if applicable)
%   feature_linkage_tree - Linkage tree for feature clustering (if applicable)

    % Initialize
    unit_sort_idx = 1:size(feature_matrix, 1);
    feature_sort_idx = 1:size(feature_matrix, 2);
    unit_linkage_tree = [];
    feature_linkage_tree = [];

    % Minimum sample size for clustering
    MIN_UNITS_FOR_CLUSTERING = 3;
    MIN_FEATURES_FOR_CLUSTERING = 3;

    % --- UNIT SORTING (only if clustering units or both) ---
    if strcmp(config.cluster_dimension, 'units') || strcmp(config.cluster_dimension, 'both')
        fprintf('    Clustering/sorting units...\n');

        % Check if we have enough units for clustering
        if size(feature_matrix, 1) < MIN_UNITS_FOR_CLUSTERING
            fprintf('    WARNING: Only %d units available, skipping clustering (minimum: %d)\n', ...
                size(feature_matrix, 1), MIN_UNITS_FOR_CLUSTERING);
            fprintf('      Using original order\n');
        else

        switch config.sort_method
            case 'session_type'
                % Simple sort: Aversive first, then Reward
                aversive_idx = find(is_aversive);
                reward_idx = find(~is_aversive);
                unit_sort_idx = [aversive_idx, reward_idx];

            case 'hierarchical'
                % Hierarchical clustering of units
                % Remove units with too many NaNs
                valid_units = sum(~isnan(feature_matrix), 2) > size(feature_matrix, 2) * 0.5;

                if sum(valid_units) < MIN_UNITS_FOR_CLUSTERING
                    fprintf('    WARNING: Only %d valid units (minimum: %d), using original order\n', ...
                        sum(valid_units), MIN_UNITS_FOR_CLUSTERING);
                else
                    feature_matrix_clean = feature_matrix(valid_units, :);

                    % Remove features (columns) that are entirely or mostly NaN
                    % This is critical for separate session analysis where session-specific features may be all NaN
                    valid_features = sum(~isnan(feature_matrix_clean), 1) > size(feature_matrix_clean, 1) * 0.3;

                    if sum(valid_features) < 2
                        fprintf('    WARNING: Only %d valid features after removing NaN columns, using original order\n', sum(valid_features));
                    else
                        feature_matrix_clean = feature_matrix_clean(:, valid_features);

                        % Replace remaining NaNs with column mean
                        for f = 1:size(feature_matrix_clean, 2)
                            col = feature_matrix_clean(:, f);
                            if ~all(isnan(col))  % Extra safety check
                                col(isnan(col)) = nanmean(col);
                                feature_matrix_clean(:, f) = col;
                            end
                        end

                        % Compute distance and linkage
                        distances = pdist(feature_matrix_clean, 'euclidean');
                        unit_linkage_tree = linkage(distances, 'ward');

                        % Get dendrogram order
                        [~, ~, sort_idx_clean] = dendrogram(unit_linkage_tree, 0);

                        % Map back to original indices
                        valid_idx = find(valid_units);
                        unit_sort_idx = valid_idx(sort_idx_clean);
                    end
                end

            case 'pca'
                % Sort by first principal component
                % Remove NaNs
                valid_units = sum(~isnan(feature_matrix), 2) > size(feature_matrix, 2) * 0.5;

                if sum(valid_units) < MIN_UNITS_FOR_CLUSTERING
                    fprintf('    WARNING: Only %d valid units (minimum: %d), using original order\n', ...
                        sum(valid_units), MIN_UNITS_FOR_CLUSTERING);
                else
                    feature_matrix_clean = feature_matrix(valid_units, :);

                    % Remove features (columns) that are entirely or mostly NaN
                    valid_features = sum(~isnan(feature_matrix_clean), 1) > size(feature_matrix_clean, 1) * 0.3;

                    if sum(valid_features) < 2
                        fprintf('    WARNING: Only %d valid features after removing NaN columns, using original order\n', sum(valid_features));
                    else
                        feature_matrix_clean = feature_matrix_clean(:, valid_features);

                        % Replace remaining NaNs with column mean
                        for f = 1:size(feature_matrix_clean, 2)
                            col = feature_matrix_clean(:, f);
                            if ~all(isnan(col))  % Extra safety check
                                col(isnan(col)) = nanmean(col);
                                feature_matrix_clean(:, f) = col;
                            end
                        end

                        % PCA
                        [~, score, ~] = pca(feature_matrix_clean);

                        % Sort by PC1
                        [~, sort_idx_clean] = sort(score(:, 1));

                        % Map back to original indices
                        valid_idx = find(valid_units);
                        unit_sort_idx = valid_idx(sort_idx_clean);
                    end
                end

            case 'feature'
                % Sort by specific feature
                feature_idx = find(strcmp(feature_names, config.feature_to_sort));
                if isempty(feature_idx)
                    fprintf('    WARNING: Feature %s not found, using default sort\n', config.feature_to_sort);
                    [~, unit_sort_idx] = sortrows([is_aversive',feature_matrix], [0,3]+1 ,'descend', 'MissingPlacement', 'last');
                else
                    [~, unit_sort_idx] = sort(feature_matrix(:, feature_idx), 'descend', 'MissingPlacement', 'last');
                end

            otherwise
                % Default: session type
                aversive_idx = find(is_aversive);
                reward_idx = find(~is_aversive);
                unit_sort_idx = [aversive_idx, reward_idx];
        end
        fprintf('      ✓ Units sorted\n');
        end
    end

    % --- FEATURE SORTING (only if clustering features or both) ---
    if strcmp(config.cluster_dimension, 'features') || strcmp(config.cluster_dimension, 'both')
        fprintf('    Clustering features...\n');

        % Check if we have enough features for clustering
        if size(feature_matrix, 2) < MIN_FEATURES_FOR_CLUSTERING
            fprintf('    WARNING: Only %d features available, skipping feature clustering (minimum: %d)\n', ...
                size(feature_matrix, 2), MIN_FEATURES_FOR_CLUSTERING);
            fprintf('      Using original order\n');
        else

            % Hierarchical clustering of features
            % Transpose the matrix (features as rows)
            % Remove features with too many NaNs
            valid_features = sum(~isnan(feature_matrix), 1) > size(feature_matrix, 1) * 0.5;

            if sum(valid_features) < MIN_FEATURES_FOR_CLUSTERING
                fprintf('    WARNING: Only %d valid features (minimum: %d), using original order\n', ...
                    sum(valid_features), MIN_FEATURES_FOR_CLUSTERING);
            else
                feature_matrix_transposed = feature_matrix(:, valid_features)';

                % Replace remaining NaNs with row mean
                for u = 1:size(feature_matrix_transposed, 2)
                    col = feature_matrix_transposed(:, u);
                    col(isnan(col)) = nanmean(col);
                    feature_matrix_transposed(:, u) = col;
                end

                % Compute distance and linkage
                distances_features = pdist(feature_matrix_transposed, 'euclidean');
                feature_linkage_tree = linkage(distances_features, 'ward');

                % Get dendrogram order
                [~, ~, feature_sort_idx_clean] = dendrogram(feature_linkage_tree, 0);

                % Map back to original indices
                valid_feat_idx = find(valid_features);
                feature_sort_idx = valid_feat_idx(feature_sort_idx_clean);
                fprintf('      ✓ Features sorted\n');
            end
        end
    end
end

function cmap = bluewhitered(n)
    if nargin < 1
        n = 256;
    end

    % Create blue to white to red colormap
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

function cmap = redblue(n)
    if nargin < 1
        n = 256;
    end

    % Red to blue colormap
    half = ceil(n/2);

    % Red to white
    r1 = ones(half, 1);
    g1 = linspace(0, 1, half)';
    b1 = linspace(0, 1, half)';

    % White to blue
    r2 = linspace(1, 0, half)';
    g2 = linspace(1, 0, half)';
    b2 = ones(half, 1);

    cmap = [r1 g1 b1; r2 g2 b2];
    cmap = cmap(1:n, :);
end
