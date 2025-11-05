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
config.sort_method = 'hierarchical';  % Options: 'session_type', 'hierarchical', 'pca', 'feature'
config.sort_method = 'feature';  % Options: 'session_type', 'hierarchical', 'pca', 'feature'
config.feature_to_sort = [];  % Used if sort_method = 'feature'
config.show_dendrogram = true;  % Show hierarchical clustering dendrogram
config.normalize_features = true;  % Z-score normalize each feature column
config.colormap_name = 'bluewhitered';  % 'bluewhitered', 'jet', 'parula', 'redblue'

fprintf('Configuration:\n');
fprintf('  Sort method: %s\n', config.sort_method);
fprintf('  Normalize features: %d\n', config.normalize_features);
fprintf('  Colormap: %s\n\n', config.colormap_name);

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
%  SECTION 4: NORMALIZE FEATURES (OPTIONAL)
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
%  SECTION 5: DETERMINE UNIT SORTING ORDER
%% ========================================================================

fprintf('Determining unit sorting order...\n');
fprintf('  Method: %s\n', config.sort_method);

% Get session types for all units
session_types = {coherence_features.session_type};
is_aversive = contains(session_types, 'Aversive');

switch config.sort_method
    case 'session_type'
        % Simple sort: Aversive first, then Reward
        aversive_idx = find(is_aversive);
        reward_idx = find(~is_aversive);
        sort_idx = [aversive_idx, reward_idx];
        linkage_tree = [];

    case 'hierarchical'
        % Hierarchical clustering
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
        linkage_tree = linkage(distances, 'ward');

        % Get dendrogram order
        [~, ~, sort_idx_clean] = dendrogram(linkage_tree, 0);

        % Map back to original indices
        valid_idx = find(valid_units);
        sort_idx = valid_idx(sort_idx_clean);

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
        sort_idx = valid_idx(sort_idx_clean);
        linkage_tree = [];

    case 'feature'
        % Sort by specific feature
        feature_idx = find(strcmp(feature_names, config.feature_to_sort));
        if isempty(feature_idx)
            fprintf('  WARNING: Feature %s not found, using session_type instead\n', config.feature_to_sort);
%             aversive_idx = find(is_aversive);
%             reward_idx = find(~is_aversive);
%             sort_idx = [aversive_idx, reward_idx];
%             [~, sort_idx] = sortrows([is_aversive',feature_matrix], [0,85,83,2,3]+1 ,'descend', 'MissingPlacement', 'last');
            [~, sort_idx] = sortrows([is_aversive',feature_matrix], [0,3]+1 ,'descend', 'MissingPlacement', 'last');
        else
            [~, sort_idx] = sort(feature_matrix(:, feature_idx), 'descend', 'MissingPlacement', 'last');
        end
        linkage_tree = [];

    otherwise
        % Default: session type
        aversive_idx = find(is_aversive);
        reward_idx = find(~is_aversive);
        sort_idx = [aversive_idx, reward_idx];
        linkage_tree = [];
end

fprintf('✓ Sorting order determined\n\n');

%% ========================================================================
%  SECTION 6: CREATE COMPREHENSIVE HEATMAP
%% ========================================================================

fprintf('Creating comprehensive heatmap...\n');

% Reorder matrix
feature_matrix_sorted = feature_matrix(sort_idx, :);
session_types_sorted = session_types(sort_idx);
is_aversive_sorted = is_aversive(sort_idx);

% Create main figure
fig = figure('Position', [50 50 2000 1200], 'Name', 'Comprehensive Unit Feature Heatmap');

% Create subplot layout: dendrogram (optional) + heatmap + colorbar
if config.show_dendrogram && ~isempty(linkage_tree)
    % With dendrogram
    subplot('Position', [0.05 0.15 0.10 0.75]);  % Dendrogram on left
    dendrogram(linkage_tree, 0, 'Orientation', 'left');
    set(gca, 'YDir', 'reverse');
    set(gca, 'XTickLabel', []);
    ylabel('Units');
    title('Clustering');

    % Heatmap
    subplot('Position', [0.17 0.15 0.70 0.75]);
else
    % No dendrogram - full width heatmap
    subplot('Position', [0.08 0.15 0.80 0.75]);
end

% Plot heatmap
imagesc(feature_matrix_sorted');
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
cb.Position = [0.90 0.15 0.02 0.75];
if config.normalize_features
    ylabel(cb, 'Z-score', 'FontSize', 11);
else
    ylabel(cb, 'Feature Value', 'FontSize', 11);
end

% Labels
xlabel('Units', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Features', 'FontSize', 12, 'FontWeight', 'bold');

% Y-axis: Feature names
set(gca, 'YTick', 1:length(feature_names));
set(gca, 'YTickLabel', feature_names);
set(gca, 'FontSize', 12);

% X-axis: Show session type boundaries
set(gca, 'XTick', []);

% Add session type color bar at top
subplot('Position', [0.17 0.91 0.70 0.02]);
session_type_colors = zeros(1, length(sort_idx), 3);
for i = 1:length(sort_idx)
    if is_aversive_sorted(i)
        session_type_colors(1, i, :) = [0.8 0.2 0.2];  % Red for aversive
    else
        session_type_colors(1, i, :) = [0.2 0.2 0.8];  % Blue for reward
    end
end
image(session_type_colors);
set(gca, 'XTick', [], 'YTick', []);
ylabel('Session', 'FontSize', 9, 'Rotation', 0, 'HorizontalAlignment', 'right');

% Add feature category color bar on right
subplot('Position', [0.88 0.15 0.01 0.75]);
unique_categories = unique(feature_categories, 'stable');
category_colors = lines(length(unique_categories));
feature_category_img = zeros(length(feature_names), 1, 3);

for i = 1:length(feature_names)
    cat_idx = find(strcmp(unique_categories, feature_categories{i}));
    feature_category_img(i, 1, :) = category_colors(cat_idx, :);
end

image(feature_category_img);
set(gca, 'XTick', [], 'YTick', []);
xlabel('Category', 'FontSize', 12, 'Rotation', 0);

% Add title
annotation('textbox', [0.17 0.94 0.70 0.05], 'String', ...
    sprintf('Comprehensive Unit Features (%d units × %d features) | Sort: %s', ...
    size(feature_matrix_sorted, 1), size(feature_matrix_sorted, 2), config.sort_method), ...
    'EdgeColor', 'none', 'FontSize', 14, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');

% Add legend for categories
legend_str = sprintf('Categories: ');
for i = 1:length(unique_categories)
    legend_str = [legend_str, sprintf('%s | ', unique_categories{i})];
end
annotation('textbox', [0.05 0.05 0.90 0.05], 'String', legend_str, ...
    'EdgeColor', 'none', 'FontSize', 10, 'HorizontalAlignment', 'center');

fprintf('✓ Heatmap created\n\n');

%% extra figure;
figure; 
scatter(feature_matrix(is_aversive,2),feature_matrix(is_aversive,3),12,feature_matrix(is_aversive,83),'fill')
colormap turbo
clim([-2,2])
xlim([-2,2])
ylim([-2,2])

%% ========================================================================
%  SECTION 7: SAVE RESULTS
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
%  SECTION 8: SUMMARY STATISTICS
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
