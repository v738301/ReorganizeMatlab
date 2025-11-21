%% ========================================================================
%  PSTH TIME SERIES FEATURE EXTRACTION FOR CLUSTERING
%  Extracts the Z-scored PSTH time series as features for unit clustering
%  ========================================================================
%
%  This script loads results from:
%  1. PSTH_Survey_Analysis - Extracts the Z-scored PSTH trace within a specified window.
%     Each time bin within this window becomes a separate feature.
%  2. Unit_Features_Analysis - Used only to create a master list of all unique units.
%
%  CRITICAL: Uses unique unit IDs (session_filename + unit_number) to prevent feature mismatches.
%
%  Output: unit_features_for_clustering_PSTH_TimeSeries.mat
%    - One row per unit with unique identifier.
%    - Features consist of the concatenated PSTH time series for all events.
%
%% ========================================================================

clear all;
close all;

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== PSTH TIME SERIES FEATURE EXTRACTION FOR CLUSTERING ===\n\n');

config = struct();

% PSTH time window for feature extraction
config.psth_response_window = [-1, 2];  % Use -1 to 2 sec post-event time series

% Which PSTH events to use for features. Set to 'all' to use all available events,
% or a cell array of strings to select specific ones, e.g., {'AversiveOnset', 'IR1ON'}.
% config.psth_events_to_use = 'all';
config.psth_events_to_use = {'CenterArea_Onset','GoalDirected_Onset','HighFreqBreathing_Onset','HighSpeed_Onset','IR1ON','IR2ON','MovementOnset','AversiveOnset'};

fprintf('Configuration:\n');
fprintf('  PSTH window: [%.1f, %.1f] sec\n', config.psth_response_window(1), config.psth_response_window(2));
if ischar(config.psth_events_to_use) && strcmp(config.psth_events_to_use, 'all')
    fprintf('  PSTH events: all\n\n');
else
    fprintf('  PSTH events: %s\n\n', strjoin(config.psth_events_to_use, ', '));
end

%% ========================================================================
%  SECTION 2: LOAD MASTER UNIT LIST AND PSTH DATA
%  ========================================================================

fprintf('Loading Unit Features Analysis to build master unit list...\n');

% Find most recent Unit_features_analysis file
feature_files = dir('Unit_features_analysis_*.mat');
if isempty(feature_files)
    error('No Unit_features_analysis_*.mat file found');
end

[~, idx] = max([feature_files.datenum]);
feature_file = feature_files(idx).name;
fprintf('  Loading: %s\n', feature_file);

feature_data = load(feature_file);
unit_features_tbl = feature_data.results.tbl_data;
fprintf('✓ Loaded: %d entries from unit features\n\n', height(unit_features_tbl));

% Load PSTH results
fprintf('Loading PSTH results...\n');
psth_file = 'PSTH_Survey_Results.mat';
if exist(psth_file, 'file')
    psth_loaded = load(psth_file);
    psth_data = psth_loaded.results;
    fprintf('✓ Loaded PSTH data: %d units\n\n', psth_data.n_units_total);
else
    fprintf('  WARNING: PSTH file not found\n');
    psth_data = [];
end

%% ========================================================================
%  SECTION 3: CREATE MASTER UNIT LIST WITH UNIQUE IDs
%  ========================================================================

fprintf('=== CREATING MASTER UNIT LIST WITH UNIQUE IDs ===\n');

% Get unique (Session, Unit) combinations
unique_units_tbl = unique(unit_features_tbl(:, {'session_name', 'Unit', 'SessionType'}), 'rows');
n_master_units = height(unique_units_tbl);

% Create unique unit IDs: "SessionID_UnitX"
unique_unit_ids = cell(n_master_units, 1);
for i = 1:n_master_units
    session_str = char(unique_units_tbl.session_name(i));
    unit_num = rem(double(string(unique_units_tbl.Unit(i))),1000);
    unique_unit_ids{i} = sprintf('%s_Unit%d', session_str, unit_num);
end

% Initialize master features table
master_features = table();
master_features.UniqueUnitID = unique_unit_ids;
master_features.Session = unique_units_tbl.session_name;
master_features.Unit = rem(double(string(unique_units_tbl.Unit)),1000);
master_features.SessionType = unique_units_tbl.SessionType;

fprintf('✓ Master unit list created: %d unique units\n', n_master_units);
fprintf('  Example UniqueUnitID: %s\n\n', unique_unit_ids{1});

%% ========================================================================
%  SECTION 4: EXTRACT PSTH TIME SERIES FEATURES (EFFICIENT METHOD)
%  ========================================================================

fprintf('=== EXTRACTING PSTH TIME SERIES FEATURES (EFFICIENT METHOD) ===\n');

if ~isempty(psth_data) && ~isempty(psth_data.unit_data)
    % --- Step 1: Define feature space from a sample unit ---
    sample_unit = psth_data.unit_data(1);
    all_time_centers = sample_unit.time_centers;
    
    time_bins_mask = all_time_centers >= config.psth_response_window(1) & all_time_centers <= config.psth_response_window(2);
    feature_time_centers = all_time_centers(time_bins_mask);
    n_time_bins = length(feature_time_centers);

    event_fields = fieldnames(sample_unit);
    all_detected_events = {};
    for i = 1:length(event_fields)
        if contains(event_fields{i}, '_zscore')
            event_name = strrep(event_fields{i}, '_zscore', '');
            all_detected_events{end+1} = event_name;
        end
    end
    
    % --- Filter for user-specified events if provided ---
    if ischar(config.psth_events_to_use) && strcmp(config.psth_events_to_use, 'all')
        event_types = all_detected_events;
        fprintf('  Using all %d detected event types.\n', length(event_types));
    else
        event_types = {};
        for i = 1:length(config.psth_events_to_use)
            if ismember(config.psth_events_to_use{i}, all_detected_events)
                event_types{end+1} = config.psth_events_to_use{i};
            else
                fprintf('  WARNING: Specified event "%s" not found in PSTH data. It will be ignored.\n', config.psth_events_to_use{i});
            end
        end
        fprintf('  Filtered to %d user-specified event types.\n', length(event_types));
    end
    
    n_events = length(event_types);
    n_features = n_events * n_time_bins;
    fprintf('  Found %d event types\n', n_events);
    fprintf('  Extracting %d time bins per event, for a total of %d features\n', n_time_bins, n_features);

    % --- Step 2: Create a fast lookup map for unit indices ---
    id_to_index_map = containers.Map(master_features.UniqueUnitID, 1:n_master_units);
    fprintf('  Created a fast lookup map for %d units.\n', n_master_units);

    % --- Step 3: Pre-allocate a matrix for performance ---
    psth_feature_matrix = nan(n_master_units, n_features);
    
    % --- Step 4: Populate the matrix using the map ---
    fprintf('  Processing PSTH data...\n');
    for unit_idx = 1:length(psth_data.unit_data)
        unit = psth_data.unit_data(unit_idx);
        [~, session_base, ext] = fileparts(unit.session_name);
        psth_unique_id = sprintf('%s%s_Unit%d', session_base, ext, unit.unit_id);
        
        if id_to_index_map.isKey(psth_unique_id)
            master_idx = id_to_index_map(psth_unique_id);
            
            % Concatenate all event traces for this unit into a single row vector
            feature_row = nan(1, n_features);
            for e_idx = 1:n_events
                event_name = event_types{e_idx};
                zscore_field = [event_name '_zscore'];
                
                if isfield(unit, zscore_field) && isfield(unit, [event_name '_n_events']) && unit.([event_name '_n_events']) > 0 && ~all(isnan(unit.(zscore_field)))
                    psth_segment = unit.(zscore_field)(time_bins_mask);
                    start_col = (e_idx - 1) * n_time_bins + 1;
                    end_col = e_idx * n_time_bins;
                    feature_row(start_col:end_col) = psth_segment;
                end
            end
            psth_feature_matrix(master_idx, :) = feature_row;
        end
        
        if mod(unit_idx, 100) == 0
            fprintf('    ...processed %d/%d PSTH units\n', unit_idx, length(psth_data.unit_data));
        end
    end

    % --- Step 5: Create feature names and convert matrix to table ---
    feature_names = cell(1, n_features);
    feature_idx = 1;
    for e_idx = 1:n_events
        for t_idx = 1:n_time_bins
            time_str = strrep(sprintf('t%.2f', feature_time_centers(t_idx)), '-', 'neg');
            feature_names{feature_idx} = ['PSTH_' event_types{e_idx} '_' time_str];
            feature_idx = feature_idx + 1;
        end
    end
    
    psth_feature_table = array2table(psth_feature_matrix, 'VariableNames', feature_names);

    % --- Step 6: Combine metadata with feature data ---
    master_features = [master_features, psth_feature_table];

    fprintf('✓ Efficiently extracted PSTH time series features.\n');
else
    fprintf('  Skipping PSTH features (no data).\n');
end

%% ========================================================================
%  SECTION 5: SAVE RESULTS
%  ========================================================================

fprintf('\n=== SAVING RESULTS ===\n');

% Package results
results = struct();
results.master_features = master_features;
results.config = config;
results.n_units = n_master_units;
results.feature_names = master_features.Properties.VariableNames(5:end);  % Skip metadata columns
results.n_features = length(results.feature_names);

% Save
save_filename = 'unit_features_for_clustering_PSTH_TimeSeries.mat';
save(save_filename, 'results', '-v7.3');

fprintf('✓ Saved to: %s\n', save_filename);

%% ========================================================================
%  SECTION 6: SUMMARY
%  ========================================================================

fprintf('\n========================================\n');
fprintf('PSTH TIME SERIES FEATURE EXTRACTION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Total units: %d\n', results.n_units);
fprintf('Total features: %d\n', results.n_features);
fprintf('\nFeature categories:\n');
if ~isempty(psth_data)
    fprintf('  - PSTH time series: %d event types × %d time bins\n', length(event_types), n_time_bins);
end
fprintf('\nData integrity:\n');
fprintf('  ✓ Using unique unit IDs (session + unit number)\n');
fprintf('  ✓ Prevents feature mismatches across data sources\n');
fprintf('\nNext step: Run Unit_Feature_Comprehensive_Heatmap_PSTH_TimeSeries.m\n');
fprintf('========================================\n');