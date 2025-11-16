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
close all;

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
config.exclude_categories = {'Phase_Narrow','Phase_Broad'};  % Feature categories to exclude, e.g., {'PSTH', 'Coherence'}
config.separate_by_session = true;  % Analyze reward and aversive sessions separately
config.simplified_cluster_threshold = 9;  % Distance threshold for simplified clustering (lower = more clusters)
config.exclude_sessions = [3,4,17,18,35,36];  % Session IDs to exclude from analysis, e.g., {'SessionA001', 'SessionR005'}

fprintf('Configuration:\n');
fprintf('  Cluster dimension: %s\n', config.cluster_dimension);
fprintf('  Sort method: %s\n', config.sort_method);
fprintf('  Separate by session: %d\n', config.separate_by_session);
fprintf('  Normalize features: %d\n', config.normalize_features);
fprintf('  Colormap: %s\n', config.colormap_name);
if ~isempty(config.exclude_categories)
    fprintf('  Exclude categories: %s\n', strjoin(config.exclude_categories, ', '));
end
if ~isempty(config.exclude_sessions)
    fprintf('  Exclude sessions: %d\n', config.exclude_sessions);
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
%  SECTION 2.5: FILTER SESSIONS (OPTIONAL)
%% ========================================================================

if ~isempty(config.exclude_sessions)
    fprintf('Filtering sessions...\n');

    % Extract session IDs from unit metadata
    if isfield(coherence_features, 'session_id')
        unit_session_ids = {coherence_features.session_id};
    elseif isfield(coherence_features, 'unit_id')
        % Extract session ID from unit_id (format might be 'SessionID_UnitID')
        unit_ids = {coherence_features.unit_id};
        unit_session_ids = cellfun(@(x) extractBefore(x, '_'), unit_ids, 'UniformOutput', false);
        % If no underscore, use the whole unit_id
        for i = 1:length(unit_session_ids)
            if isempty(unit_session_ids{i})
                unit_session_ids{i} = unit_ids{i};
            end
        end
    else
        warning('No session_id or unit_id field found, cannot filter sessions');
        unit_session_ids = cell(1, n_units);
    end

    % Find units to keep (not in excluded sessions)
    units_to_keep = true(1, n_units);
    for i = 1:n_units
        if any(ismember(unit_session_ids{i}, config.exclude_sessions))
            units_to_keep(i) = false;
        end
    end

    n_excluded = sum(~units_to_keep);
    fprintf('  Excluding %d units from %d session(s)\n', n_excluded, length(config.exclude_sessions));

    % Filter all feature structures
    coherence_features = coherence_features(units_to_keep);

    % Filter phase features if they have the same length
    if n_units_phase_narrow == n_units
        phase_narrow_features = phase_narrow_features(units_to_keep);
    end
    if n_units_phase_broad == n_units
        phase_broad_features = phase_broad_features(units_to_keep);
    end
    if n_units_psth == n_units
        psth_features = psth_features(units_to_keep);
    end

    % Update counts
    n_units_coherence = length(coherence_features);
    n_units_phase_narrow = length(phase_narrow_features);
    n_units_phase_broad = length(phase_broad_features);
    n_units_psth = length(psth_features);
    n_units = n_units_coherence;

    fprintf('  Remaining units:\n');
    fprintf('    Coherence: %d units\n', n_units_coherence);
    fprintf('    Phase narrow: %d units\n', n_units_phase_narrow);
    fprintf('    Phase broad: %d units\n', n_units_phase_broad);
    fprintf('    PSTH: %d units\n\n', n_units_psth);
else
    fprintf('No sessions excluded\n\n');
end

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

% Firing rate (1 feature)
all_features(:, end+1) = [coherence_features.firing_rate_mean]';
feature_names{end+1} = 'FiringRate';
feature_categories{end+1} = 'FiringRate';

% Coefficient of Variation (1 feature)
all_features(:, end+1) = [coherence_features.cv]';
feature_names{end+1} = 'CV';
feature_categories{end+1} = 'CV';

fprintf('    Added %d coherence features\n', 11);
fprintf('    Added 1 firing rate feature\n');
fprintf('    Added 1 CV feature\n');

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

% Get session types and session IDs for all units
session_types = {coherence_features.session_type};
is_aversive = contains(session_types, 'Aversive');

% Extract session IDs - check which field is available
if isfield(coherence_features, 'session_id')
    session_ids = {coherence_features.session_id};
elseif isfield(coherence_features, 'unit_id')
    % Extract session ID from unit_id (format might be 'SessionID_UnitID')
    unit_ids = {coherence_features.unit_id};
    session_ids = cellfun(@(x) extractBefore(x, '_'), unit_ids, 'UniformOutput', false);
    % If no underscore, use the whole unit_id
    for i = 1:length(session_ids)
        if isempty(session_ids{i})
            session_ids{i} = unit_ids{i};
        end
    end
else
    % If no session ID field, create generic IDs based on session type
    warning('No session_id or unit_id field found, using generic session IDs');
    session_ids = arrayfun(@(x) sprintf('Session%03d', x), 1:length(session_types), 'UniformOutput', false);
end

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


%% ========================================================================
%  SIMPLIFIED 6-FEATURE CLUSTERING ANALYSIS
%% ========================================================================

fprintf('\n=== SIMPLIFIED 6-FEATURE CLUSTERING ===\n\n');

simplified_feature_names = {
    'Coh_5-7Hz',           % Coherence 5-7Hz
    'Coh_8-10Hz',          % Coherence 8-10Hz
    'PSTH_Reward_MeanZ',   % PSTH PSTH Reward Mean Z-score
    'PSTH_Aversive_MeanZ', % PSTH Aversive mean z-score
    'FiringRate'           % Mean firing rate (Hz)
};

% Extract these features from the full feature matrix
simplified_matrix = [];
simplified_names_found = {};
simplified_indices = [];

for i = 1:length(simplified_feature_names)
    feat_idx = find(strcmp(feature_names, simplified_feature_names{i}));
    if ~isempty(feat_idx)
        simplified_matrix(:, end+1) = feature_matrix(:, feat_idx);
        simplified_names_found{end+1} = simplified_feature_names{i};
        simplified_indices(end+1) = feat_idx;
    else
        fprintf('  WARNING: Feature "%s" not found, skipping\n', simplified_feature_names{i});
    end
end

fprintf('Found %d/%d features for simplified analysis\n', length(simplified_names_found), length(simplified_feature_names));

if size(simplified_matrix, 2) < 3
    fprintf('ERROR: Not enough features found for simplified clustering (need at least 3)\n');
else
    % Normalize the simplified feature matrix
    simplified_matrix_norm = nan(size(simplified_matrix));
    for f = 1:size(simplified_matrix, 2)
        feature_col = simplified_matrix(:, f);
        valid_data = feature_col(~isnan(feature_col));

        if ~isempty(valid_data) && std(valid_data) > 0
            simplified_matrix_norm(:, f) = (feature_col - mean(valid_data)) / std(valid_data);
        else
            simplified_matrix_norm(:, f) = feature_col;
        end
    end

    % Determine how many plots to create (separate by session or combined)
    if config.separate_by_session
        n_simple_plots = 2;
        simple_plot_names = {'Aversive', 'Reward'};
    else
        n_simple_plots = 1;
        simple_plot_names = {'Combined'};
    end

    for plot_idx = 1:n_simple_plots

        if config.separate_by_session
            fprintf('\n  Creating simplified heatmap for %s sessions...\n', simple_plot_names{plot_idx});

            % Select units for this session type
            if strcmp(simple_plot_names{plot_idx}, 'Aversive')
                sess_units = is_aversive;
            else
                sess_units = ~is_aversive;
            end

            simple_matrix_plot = simplified_matrix_norm(sess_units, :);
            simple_is_aversive = is_aversive(sess_units);
            simple_session_ids = session_ids(sess_units);
        else
            simple_matrix_plot = simplified_matrix_norm;
            simple_is_aversive = is_aversive;
            simple_session_ids = session_ids;
        end

        % Perform hierarchical clustering on simplified matrix
        valid_units = sum(~isnan(simple_matrix_plot), 2) > size(simple_matrix_plot, 2) * 0.5;

        if sum(valid_units) >= 3
            simple_matrix_clean = simple_matrix_plot(valid_units, :);

            % Remove NaN columns
            valid_features = sum(~isnan(simple_matrix_clean), 1) > size(simple_matrix_clean, 1) * 0.3;
            simple_matrix_clean = simple_matrix_clean(:, valid_features);
            simple_names_used = simplified_names_found(valid_features);

            % Replace remaining NaNs with column mean
            for f = 1:size(simple_matrix_clean, 2)
                col = simple_matrix_clean(:, f);
                if ~all(isnan(col))
                    col(isnan(col)) = nanmean(col);
                    simple_matrix_clean(:, f) = col;
                end
            end

            % Compute distance and linkage
            distances = pdist(simple_matrix_clean, 'euclidean');
            simple_linkage_tree = linkage(distances, 'ward');

            % Get dendrogram order
            figure;
            [~, ~, sort_idx_clean] = dendrogram(simple_linkage_tree, 0);

            % Map back to original indices
            valid_idx = find(valid_units);
            simple_sort_idx = valid_idx(sort_idx_clean);

            % Reorder matrix
            simple_matrix_sorted = simple_matrix_plot(simple_sort_idx, valid_features);
            simple_is_aversive_sorted = simple_is_aversive(simple_sort_idx);

            % Create figure
            fig_simple = figure('Position', [100 + (plot_idx-1)*100 100 + (plot_idx-1)*50 1600 1000], ...
                               'Name', sprintf('Simplified 6-Feature Clustering - %s', simple_plot_names{plot_idx}));

            % Left: Dendrogram
            subplot('Position', [0.05 0.15 0.10 0.75]);
            dendrogram(simple_linkage_tree, 0, 'Orientation', 'left');
            set(gca, 'YDir', 'reverse');
            set(gca, 'XTickLabel', []);
            ylabel('Units');
            title('Unit Clustering');

            % Center: Heatmap
            subplot('Position', [0.17 0.15 0.65 0.75]);
            imagesc(simple_matrix_sorted);
            axis tight;
            colormap(bluewhitered(256));
            caxis([-3 3]);

            % Colorbar
            cb = colorbar;
            cb.Position = [0.84 0.15 0.02 0.75];
            ylabel(cb, 'Z-score', 'FontSize', 11);

            % Labels
            xlabel('Features', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('Units', 'FontSize', 12, 'FontWeight', 'bold');

            % X-axis: Feature names
            set(gca, 'XTick', 1:length(simple_names_used));
            set(gca, 'XTickLabel', simple_names_used);
            set(gca, 'XTickLabelRotation', 45);
            set(gca, 'FontSize', 11);

            % Y-axis: Unit indices
            set(gca, 'YTick', []);

            % Add session type color bar on LEFT
            subplot('Position', [0.15 0.15 0.01 0.75]);
            session_type_colors = zeros(length(simple_sort_idx), 1, 3);
            for i = 1:length(simple_sort_idx)
                if simple_is_aversive_sorted(i)
                    session_type_colors(i, 1, :) = [0.8 0.2 0.2];  % Red for aversive
                else
                    session_type_colors(i, 1, :) = [0.2 0.2 0.8];  % Blue for reward
                end
            end
            image(session_type_colors);
            set(gca, 'XTick', [], 'YTick', []);
            xlabel('Session', 'FontSize', 9, 'Rotation', 0);

            % Add title
            if config.separate_by_session
                title_str = sprintf('Simplified 6-Feature Clustering - %s Sessions (%d units)', ...
                    simple_plot_names{plot_idx}, size(simple_matrix_sorted, 1));
            else
                title_str = sprintf('Simplified 6-Feature Clustering (%d units)', ...
                    size(simple_matrix_sorted, 1));
            end
            annotation('textbox', [0.17 0.92 0.65 0.05], 'String', title_str, ...
                'EdgeColor', 'none', 'FontSize', 14, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center');

            fprintf('  ✓ Simplified heatmap created\n');

            % --- CLUSTER THRESHOLDING AND SESSION ID COMPOSITION ANALYSIS ---
            fprintf('  Performing cluster thresholding...\n');

            % Determine number of clusters using distance threshold
            % Adjust config.simplified_cluster_threshold to get different number of clusters
            % Lower threshold = more clusters, Higher threshold = fewer clusters
            cluster_threshold = config.simplified_cluster_threshold;
            cluster_assignments_clean = cluster(simple_linkage_tree, 'cutoff', cluster_threshold, 'criterion', 'distance');

            % Option 2: Manual - specify number of clusters (uncomment to use)
            % n_clusters = 4;
            % cluster_assignments_clean = cluster(simple_linkage_tree, 'maxclust', n_clusters);

            % Map cluster assignments back to all units
            cluster_assignments = nan(size(simple_matrix_plot, 1), 1);
            cluster_assignments(valid_idx) = cluster_assignments_clean;

            n_clusters = max(cluster_assignments_clean);
            fprintf('    Found %d clusters\n', n_clusters);

            % Sort units by cluster ID for clean visualization
            [cluster_assignments_sorted, cluster_sort_idx] = sort(cluster_assignments, 'ascend', 'MissingPlacement', 'last');

            % Remove NaN entries (units that weren't clustered)
            valid_cluster_idx = ~isnan(cluster_assignments_sorted);
            cluster_sort_idx = cluster_sort_idx(valid_cluster_idx);
            cluster_assignments_sorted = cluster_assignments_sorted(valid_cluster_idx);

            % Reorder the matrix and session IDs by cluster
            simple_matrix_sorted_by_cluster = simple_matrix_plot(cluster_sort_idx, :);
            simple_session_ids_sorted = simple_session_ids(cluster_sort_idx);

            % Get unique session IDs
            simple_session_ids = [simple_session_ids{:}];
            unique_sessions = unique(simple_session_ids);
            n_sessions = length(unique_sessions);
            fprintf('    Analyzing %d unique sessions\n', n_sessions);

            % Analyze session ID composition of each cluster
            cluster_stats = struct();
            session_cluster_matrix = zeros(n_sessions, n_clusters);  % rows=sessions, cols=clusters

            for c = 1:n_clusters
                units_in_cluster = find(cluster_assignments_sorted == c);
                n_units_in_cluster = length(units_in_cluster);
                session_ids_in_cluster = simple_session_ids_sorted(units_in_cluster);
                session_ids_in_cluster = [session_ids_in_cluster{:}];

                cluster_stats(c).cluster_id = c;
                cluster_stats(c).n_total = n_units_in_cluster;
                cluster_stats(c).session_composition = struct();

                fprintf('    Cluster %d: %d units\n', c, n_units_in_cluster);

                % Count units from each session
                for s = 1:n_sessions
                    sess_id = unique_sessions(s);
                    n_from_session = sum(ismember(session_ids_in_cluster, sess_id));
                    session_cluster_matrix(s, c) = n_from_session;

                    if n_from_session > 0
                        cluster_stats(c).session_composition(s).session_id = sess_id;
                        cluster_stats(c).session_composition(s).n_units = n_from_session;
                        cluster_stats(c).session_composition(s).pct = 100 * n_from_session / n_units_in_cluster;
                        fprintf('      %d: %d units (%.1f%%)\n', sess_id, n_from_session, ...
                            100 * n_from_session / n_units_in_cluster);
                    end
                end
            end


            % --- CREATE SESSION ID COMPOSITION FIGURE ---
            % Create figure first to avoid overlap issues
            fig_composition = figure('Position', [200 + (plot_idx-1)*100 200 + (plot_idx-1)*50 1600 800], ...
                                    'Name', sprintf('Cluster Session ID Composition - %s', simple_plot_names{plot_idx}));

            % --- LEFT: Colored Dendrogram sorted by Cluster ID ---
            subplot('Position', [0.05 0.12 0.18 0.75]);            
            
            allUnitID = 1:length(cluster_assignments);
            allUnitID(simple_sort_idx) = allUnitID;
            [~,cluster_double_sort] = sortrows([cluster_assignments(:),allUnitID(:)],[1,2]);
            % Draw dendrogram with cluster coloring and reordering
            H = dendrogram(simple_linkage_tree, 0, 'Orientation', 'left', ...
                          'ColorThreshold', cluster_threshold, ...
                          'Reorder', cluster_double_sort);

            set(gca, 'YDir', 'normal');  % Cluster 1 at bottom
            set(gca, 'XTickLabel', []);
            ylabel('Units (sorted by Cluster ID)', 'FontSize', 11, 'FontWeight', 'bold');
            title('Dendrogram', 'FontSize', 12, 'FontWeight', 'bold');
            set(gca, 'FontSize', 10);

            % Add cluster threshold line
            hold on;
            ylims = ylim;
            plot([cluster_threshold, cluster_threshold], ylims, 'r--', 'LineWidth', 2);
            hold off;
            box on;

            % --- CENTER: Heatmap showing session contribution to each cluster ---
            subplot('Position', [0.28 0.12 0.35 0.75]);
            imagesc(session_cluster_matrix');  % Transpose so clusters are on Y, sessions on X

            % Use a better colormap for counts
            colormap(gca, hot);
            cb = colorbar('Position', [0.64 0.12 0.015 0.75]);
            ylabel(cb, 'Unit Count', 'FontSize', 10);

            % Labels
            xlabel('Session ID', 'FontSize', 11, 'FontWeight', 'bold');
            ylabel('Cluster ID', 'FontSize', 11, 'FontWeight', 'bold');
            title('Units per Session per Cluster', 'FontSize', 12, 'FontWeight', 'bold');

            % X-axis: Session IDs
            set(gca, 'XTick', 1:n_sessions);
            set(gca, 'XTickLabel', unique_sessions);
            set(gca, 'XTickLabelRotation', 45);

            % Y-axis: Cluster IDs (reversed so Cluster 1 is at bottom)
            set(gca, 'YTick', 1:n_clusters);
            set(gca, 'YTickLabel', 1:n_clusters);
            set(gca, 'YDir', 'normal');  % Normal direction: Cluster 1 at bottom

            set(gca, 'FontSize', 10);
            axis tight;

            % Add text annotations showing counts
            for s = 1:n_sessions
                for c = 1:n_clusters
                    count = session_cluster_matrix(s, c);
                    if count > 0
                        % Use white or black text depending on background intensity
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

            % --- RIGHT: Bar chart showing total units per session ---
            subplot('Position', [0.70 0.12 0.25 0.75]);
            units_per_session = sum(session_cluster_matrix, 2);

            % Color bars by session type (if available)
            bar_colors = zeros(n_sessions, 3);
            for s = 1:n_sessions
                % Find first unit from this session to get its type
                sess_id = unique_sessions(s);
                unit_idx = find(ismember(simple_session_ids, sess_id), 1);
                if ~isempty(unit_idx) && simple_is_aversive(unit_idx)
                    bar_colors(s, :) = [0.8 0.2 0.2];  % Red for aversive
                else
                    bar_colors(s, :) = [0.2 0.2 0.8];  % Blue for reward
                end
            end

            bar_handle = bar(1:n_sessions, units_per_session, 'FaceColor', 'flat');
            bar_handle.CData = bar_colors;

            xlabel('Session ID', 'FontSize', 11, 'FontWeight', 'bold');
            ylabel('Total Units', 'FontSize', 11, 'FontWeight', 'bold');
            title('Units per Session', 'FontSize', 12, 'FontWeight', 'bold');

            set(gca, 'XTick', 1:n_sessions);
            set(gca, 'XTickLabel', unique_sessions);
            set(gca, 'XTickLabelRotation', 45);
            set(gca, 'FontSize', 10);
            grid on;

            % Add legend for session types
            hold on;
            h_aversive = plot(nan, nan, 's', 'MarkerSize', 10, 'MarkerFaceColor', [0.8 0.2 0.2], 'MarkerEdgeColor', 'none');
            h_reward = plot(nan, nan, 's', 'MarkerSize', 10, 'MarkerFaceColor', [0.2 0.2 0.8], 'MarkerEdgeColor', 'none');
            legend([h_aversive, h_reward], {'Aversive', 'Reward'}, 'Location', 'best', 'FontSize', 9);
            hold off;

            % Add overall title
            if config.separate_by_session
                sgtitle(sprintf('Session ID Composition - %s Sessions (%d clusters, %d sessions, threshold=%.2f)', ...
                    simple_plot_names{plot_idx}, n_clusters, n_sessions, cluster_threshold), ...
                    'FontSize', 14, 'FontWeight', 'bold');
            else
                sgtitle(sprintf('Session ID Composition - All Sessions (%d clusters, %d sessions, threshold=%.2f)', ...
                    n_clusters, n_sessions, cluster_threshold), ...
                    'FontSize', 14, 'FontWeight', 'bold');
            end

            % Reorder matrix
            simple_matrix_sorted = simple_matrix_plot(cluster_double_sort, valid_features);
            simple_is_aversive_sorted = simple_is_aversive(cluster_double_sort);

            % Create figure
            fig_simple = figure('Position', [100 + (plot_idx-1)*100 100 + (plot_idx-1)*50 1600 1000], ...
                               'Name', sprintf('Simplified 6-Feature Clustering - %s', simple_plot_names{plot_idx}));

            % Left: Dendrogram
            subplot('Position', [0.05 0.15 0.10 0.75]);
            dendrogram(simple_linkage_tree, 0, 'Orientation', 'left', ...
                          'ColorThreshold', cluster_threshold, ...
                          'Reorder', cluster_double_sort);
            set(gca, 'YDir', 'reverse');
            set(gca, 'XTickLabel', []);
            ylabel('Units');
            title('Unit Clustering');

            % Center: Heatmap
            subplot('Position', [0.17 0.15 0.65 0.75]);
            imagesc(simple_matrix_sorted);
            axis tight;
            colormap(bluewhitered(256));
            caxis([-3 3]);

            % Colorbar
            cb = colorbar;
            cb.Position = [0.84 0.15 0.02 0.75];
            ylabel(cb, 'Z-score', 'FontSize', 11);

            % Labels
            xlabel('Features', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('Units', 'FontSize', 12, 'FontWeight', 'bold');

            % X-axis: Feature names
            set(gca, 'XTick', 1:length(simple_names_used));
            set(gca, 'XTickLabel', simple_names_used);
            set(gca, 'XTickLabelRotation', 45);
            set(gca, 'FontSize', 11);

            % Y-axis: Unit indices
            set(gca, 'YTick', []);

            % Add session type color bar on LEFT
            subplot('Position', [0.15 0.15 0.01 0.75]);
            session_type_colors = zeros(length(simple_sort_idx), 1, 3);
            for i = 1:length(simple_sort_idx)
                if simple_is_aversive_sorted(i)
                    session_type_colors(i, 1, :) = [0.8 0.2 0.2];  % Red for aversive
                else
                    session_type_colors(i, 1, :) = [0.2 0.2 0.8];  % Blue for reward
                end
            end
            image(session_type_colors);
            set(gca, 'XTick', [], 'YTick', []);
            xlabel('Session', 'FontSize', 9, 'Rotation', 0);

            % Add title
            if config.separate_by_session
                title_str = sprintf('Simplified 6-Feature Clustering - %s Sessions (%d units)', ...
                    simple_plot_names{plot_idx}, size(simple_matrix_sorted, 1));
            else
                title_str = sprintf('Simplified 6-Feature Clustering (%d units)', ...
                    size(simple_matrix_sorted, 1));
            end
            annotation('textbox', [0.17 0.92 0.65 0.05], 'String', title_str, ...
                'EdgeColor', 'none', 'FontSize', 14, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center');

            fprintf('  ✓ Sorted heatmap created\n');

            fprintf('  ✓ Session ID composition figure created\n');

            % ========== SAVE ALL RESULTS FOR LATER USE ==========
            fprintf('  Saving clustering results for later analysis...\n');

            simplified_clustering_results = struct();

            % 1. CLUSTERING DATA (for both figures)
            simplified_clustering_results.clustering = struct();
            simplified_clustering_results.clustering.linkage_tree = simple_linkage_tree;
            simplified_clustering_results.clustering.cluster_threshold = cluster_threshold;
            simplified_clustering_results.clustering.cluster_assignments = cluster_assignments;
            simplified_clustering_results.clustering.n_clusters = n_clusters;
            simplified_clustering_results.clustering.cluster_double_sort = cluster_double_sort;
            simplified_clustering_results.clustering.simple_sort_idx = simple_sort_idx;
            simplified_clustering_results.clustering.valid_units = valid_units;
            simplified_clustering_results.clustering.distances = distances;

            % 2. FEATURE MATRIX DATA (for heatmap figure)
            simplified_clustering_results.features = struct();
            simplified_clustering_results.features.matrix = simple_matrix_plot;
            simplified_clustering_results.features.matrix_sorted = simple_matrix_sorted;
            simplified_clustering_results.features.names = simple_names_used;
            simplified_clustering_results.features.valid_features = valid_features;
            simplified_clustering_results.features.simplified_indices = simplified_indices;

            % 3. UNIT METADATA (for both figures)
            simplified_clustering_results.units = [];
            for u = 1:length(cluster_assignments)
                if ~isnan(cluster_assignments(u))
                    unit_info = struct();
                    unit_info.global_unit_id = u;
                    unit_info.session_id = simple_session_ids(u);
                    unit_info.is_aversive = simple_is_aversive(u);
                    unit_info.cluster_id = cluster_assignments(u);
                    unit_info.features = simple_matrix_plot(u, valid_features);
                    unit_info.session_filename = coherence_features(u).session_filename;
                    unit_info.unit_id = coherence_features(u).unit_id;
                    unit_info.session_type = coherence_features(u).session_type;
                    simplified_clustering_results.units = [simplified_clustering_results.units; unit_info];
                end
            end

            % 4. SESSION COMPOSITION DATA (for session composition figure)
            simplified_clustering_results.session_composition = struct();
            simplified_clustering_results.session_composition.session_cluster_matrix = session_cluster_matrix;
            simplified_clustering_results.session_composition.unique_sessions = unique_sessions;
            simplified_clustering_results.session_composition.n_sessions = n_sessions;
            simplified_clustering_results.session_composition.cluster_stats = cluster_stats;
            simplified_clustering_results.session_composition.units_per_session = sum(session_cluster_matrix, 2);

            % 5. VISUALIZATION DATA (for both figures)
            simplified_clustering_results.visualization = struct();
            simplified_clustering_results.visualization.is_aversive_sorted = simple_is_aversive_sorted;
            simplified_clustering_results.visualization.simple_session_ids = simple_session_ids;
            simplified_clustering_results.visualization.simple_is_aversive = simple_is_aversive;

            % 6. CONFIGURATION & METADATA
            simplified_clustering_results.metadata = struct();
            simplified_clustering_results.metadata.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
            simplified_clustering_results.metadata.feature_names = simple_names_used;
            simplified_clustering_results.metadata.n_features = length(simple_names_used);
            simplified_clustering_results.metadata.config = config;
            if config.separate_by_session
                simplified_clustering_results.metadata.session_type = simple_plot_names{plot_idx};
            else
                simplified_clustering_results.metadata.session_type = 'combined';
            end

            % 7. RAW DATA PATHS (for interaction analysis later)
            simplified_clustering_results.data_paths = struct();
            simplified_clustering_results.data_paths.feature_file = 'unit_features_comprehensive.mat';
            simplified_clustering_results.data_paths.spike_data_folder = '/Volumes/ExpansionBackup/Data/Struct_spike';

            % 8. CLUSTER-TO-UNIT LOOKUP (for interaction analysis)
            simplified_clustering_results.cluster_lookup = [];
            for c = 1:n_clusters
                cluster_info = struct();
                unit_indices = find(cluster_assignments == c);
                cluster_info.cluster_id = c;
                cluster_info.unit_indices = unit_indices;
                cluster_info.n_units = length(unit_indices);

                % Store session and unit IDs for each unit in this cluster
                cluster_unit_details = [];
                for i = 1:length(unit_indices)
                    u_idx = unit_indices(i);
                    unit_detail = struct();
                    unit_detail.global_unit_id = u_idx;
                    unit_detail.session_id = simple_session_ids(u_idx);
                    unit_detail.session_filename = coherence_features(u_idx).session_filename;
                    unit_detail.unit_id = coherence_features(u_idx).unit_id;
                    unit_detail.session_type = coherence_features(u_idx).session_type;
                    cluster_unit_details = [cluster_unit_details; unit_detail];
                end
                cluster_info.unit_details = cluster_unit_details;

                % Add cluster centroid (mean feature values)
                cluster_features = simple_matrix_plot(unit_indices, valid_features);
                cluster_info.centroid = mean(cluster_features, 1, 'omitnan');

                simplified_clustering_results.cluster_lookup = [simplified_clustering_results.cluster_lookup; cluster_info];
            end

            % Save to file
            timestamp_file = datestr(now, 'yyyy-mm-dd_HHMMSS');
            if config.separate_by_session
                save_filename = sprintf('simplified_clustering_%s_%s.mat', ...
                    simple_plot_names{plot_idx}, timestamp_file);
            else
                save_filename = sprintf('simplified_clustering_combined_%s.mat', timestamp_file);
            end

            save(save_filename, 'simplified_clustering_results', '-v7.3');
            fprintf('  ✓ Clustering results saved to: %s\n', save_filename);
            fprintf('     - Contains all data to regenerate figures and analyze cluster interactions\n');

        else
            fprintf('  WARNING: Not enough valid units (%d) for simplified clustering\n', sum(valid_units));
        end
    end

    fprintf('\n✓ Simplified clustering complete\n\n');
end

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
