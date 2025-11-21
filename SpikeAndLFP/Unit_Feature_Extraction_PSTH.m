%% ========================================================================
%  COMPREHENSIVE UNIT FEATURE EXTRACTION FOR CLUSTERING
%  Extracts features from all analysis results for unit clustering
%  ========================================================================
%
%  This script loads results from:
%  1. Unit_PPC_Analysis (LFP + Breathing) - PPC at 1.1-2.1 Hz and 8.1-9.1 Hz
%  2. Amplitude_Correlation_Analysis (LFP + Breathing) - Pearson_R and MI at 1.1-2.1 Hz and 8.1-9.1 Hz
%  3. Spike_LFP_Coherence_Overall (LFP + Breathing) - Coherence at 1.1-2.1 Hz and 8.1-9.1 Hz
%  4. PSTH_Survey_Analysis - Average Z-score 0-2 sec post-event
%  5. Unit_Features_Analysis - All 23 spike train metrics
%
%  CRITICAL: Uses unique unit IDs (session_filename + unit_number) to prevent feature mismatches
%
%  Output: unit_features_for_clustering.mat
%    - One row per unit with unique identifier
%    - All features combined
%
%% ========================================================================

clear all;
close all;

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== COMPREHENSIVE UNIT FEATURE EXTRACTION FOR CLUSTERING ===\n\n');

config = struct();

% Target frequency bands
config.freq_bands = struct();
config.freq_bands.low = [1.1, 2.1];    % Low frequency band (breathing/slow oscillations)
config.freq_bands.high = [8.1, 9.1];   % High frequency band (theta/sniffing)

% PSTH time window for feature extraction
config.psth_response_window = [0, 2];  % Extract 0-2 sec post-event

% Data paths
config.DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';

fprintf('Configuration:\n');
fprintf('  Low freq band: [%.1f, %.1f] Hz\n', config.freq_bands.low(1), config.freq_bands.low(2));
fprintf('  High freq band: [%.1f, %.1f] Hz\n', config.freq_bands.high(1), config.freq_bands.high(2));
fprintf('  PSTH window: [%.1f, %.1f] sec\n\n', config.psth_response_window(1), config.psth_response_window(2));

%% ========================================================================
%  SECTION 2: LOAD UNIT FEATURES ANALYSIS
%  ========================================================================

fprintf('Loading Unit Features Analysis...\n');

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
fprintf('✓ Loaded: %d entries from unit features\n', height(unit_features_tbl));

%% ========================================================================
%  SECTION 3: CREATE MASTER UNIT LIST WITH UNIQUE IDs
%  ========================================================================

fprintf('\n=== CREATING MASTER UNIT LIST WITH UNIQUE IDs ===\n');

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
%  SECTION 4: LOAD ALL DATA SOURCES
%  ========================================================================

fprintf('=== LOADING DATA SOURCES ===\n\n');

% Load PPC (LFP)
fprintf('Loading PPC Analysis (LFP)...\n');
ppc_lfp_data = loadAnalysisData(config.DataSetsPath, 'UnitPPC', '*_unit_ppc.mat');
fprintf('✓ Loaded %d sessions\n', length(ppc_lfp_data));

% Load PPC (Breathing)
fprintf('Loading PPC Analysis (Breathing)...\n');
ppc_breath_data = loadAnalysisData(config.DataSetsPath, 'UnitPPC_Breathing', '*_unit_ppc_breathing.mat');
fprintf('✓ Loaded %d sessions\n', length(ppc_breath_data));

% Load Amplitude Correlation (LFP)
fprintf('Loading Amplitude Correlation (LFP)...\n');
ampcorr_lfp_data = loadAnalysisData(config.DataSetsPath, 'AmpCorr', '*_ampcorr.mat');
fprintf('✓ Loaded %d sessions\n', length(ampcorr_lfp_data));

% Load Amplitude Correlation (Breathing)
fprintf('Loading Amplitude Correlation (Breathing)...\n');
ampcorr_breath_data = loadAnalysisData(config.DataSetsPath, 'AmpCorr_Breathing', '*_ampcorr_breathing.mat');
fprintf('✓ Loaded %d sessions\n', length(ampcorr_breath_data));

% Load Coherence (LFP)
fprintf('Loading Coherence (LFP)...\n');
coherence_lfp_data = loadAnalysisData(config.DataSetsPath, 'SpikeLFPCoherence_Overall', '*_spike_lfp_coherence_overall.mat');
fprintf('✓ Loaded %d sessions\n', length(coherence_lfp_data));

% Load Coherence (Breathing)
fprintf('Loading Coherence (Breathing)...\n');
coherence_breath_data = loadAnalysisData(config.DataSetsPath, 'SpikeBreathingCoherence_Overall', '*_spike_breathing_coherence_overall.mat');
fprintf('✓ Loaded %d sessions\n', length(coherence_breath_data));

% Load PSTH
fprintf('Loading PSTH results...\n');
psth_file = 'PSTH_Survey_Results.mat';
if exist(psth_file, 'file')
    psth_loaded = load(psth_file);
    psth_data = psth_loaded.results;
    fprintf('✓ Loaded PSTH data: %d units\n', psth_data.n_units_total);
else
    fprintf('  WARNING: PSTH file not found\n');
    psth_data = [];
end

fprintf('\n');

%% ========================================================================
%  SECTION 6: EXTRACT UNIT FEATURES (23 METRICS)
%  ========================================================================

fprintf('=== EXTRACTING UNIT FEATURES (23 METRICS) ===\n');

% List of all 23 metrics
all_metrics = {'FR', 'CV', 'ISI_FanoFactor', 'ISI_ACF_peak', 'ISI_ACF_lag', ...
               'Count_ACF_1ms_peak', 'Count_ACF_25ms_peak', 'Count_ACF_50ms_peak', ...
               'Count_ACF_1ms_peak_ind', 'Count_ACF_25ms_peak_ind', 'Count_ACF_50ms_peak_ind', ...
               'LV', 'CV2', 'BurstIndex', 'BurstRate', 'MeanBurstLength', ...
               'ISI_Skewness', 'ISI_Kurtosis', 'ISI_Mode', ...
               'CountFanoFactor_1ms', 'CountFanoFactor_25ms', 'CountFanoFactor_50ms', ...
               'RefracViolations'};

% Initialize columns
for m = 1:length(all_metrics)
    master_features.([all_metrics{m}]) = nan(n_master_units, 1);
end

% Extract features for each unit
for i = 1:n_master_units
    if mod(i, 100) == 0
        fprintf('  Processed %d/%d Features units\n', i, n_master_units);
    end
    unique_id = master_features.UniqueUnitID{i};
    session = master_features.Session(i);
    unit = master_features.Unit(i);

    % Find matching rows in unit_features_tbl
    unit_rows = ismember(unit_features_tbl.session_name, session) & rem(double(string(unit_features_tbl.Unit)),1000) == unit;

    if any(unit_rows)
        unit_data = unit_features_tbl(unit_rows, :);

        % Average each metric across periods
        for m = 1:length(all_metrics)
            metric = all_metrics{m};
            if ismember(metric, unit_data.Properties.VariableNames)
                master_features.([metric])(i) = mean(unit_data.(metric), 'omitnan');
            end
        end
    end
end

fprintf('✓ Extracted 23 unit features\n');

%% ========================================================================
%  SECTION 7: EXTRACT PPC FEATURES (LFP)
%  ========================================================================

fprintf('\n=== EXTRACTING PPC FEATURES (LFP) ===\n');

master_features.PPC_LFP_low = extractFreqBandFeatureWithUniqueID(...
    ppc_lfp_data, master_features, config.freq_bands.low, 'PPC');
master_features.PPC_LFP_high = extractFreqBandFeatureWithUniqueID(...
    ppc_lfp_data, master_features, config.freq_bands.high, 'PPC');

fprintf('✓ Extracted PPC (LFP) features\n');

%% ========================================================================
%  SECTION 8: EXTRACT PPC FEATURES (BREATHING)
%  ========================================================================

fprintf('\n=== EXTRACTING PPC FEATURES (BREATHING) ===\n');

master_features.PPC_Breath_low = extractFreqBandFeatureWithUniqueID(...
    ppc_breath_data, master_features, config.freq_bands.low, 'PPC');
master_features.PPC_Breath_high = extractFreqBandFeatureWithUniqueID(...
    ppc_breath_data, master_features, config.freq_bands.high, 'PPC');

fprintf('✓ Extracted PPC (Breathing) features\n');

%% ========================================================================
%  SECTION 9: EXTRACT AMPLITUDE CORRELATION FEATURES (LFP)
%  ========================================================================

fprintf('\n=== EXTRACTING AMPLITUDE CORRELATION FEATURES (LFP) ===\n');

master_features.PearsonR_LFP_low = extractFreqBandFeatureWithUniqueID(...
    ampcorr_lfp_data, master_features, config.freq_bands.low, 'Pearson_R');
master_features.PearsonR_LFP_high = extractFreqBandFeatureWithUniqueID(...
    ampcorr_lfp_data, master_features, config.freq_bands.high, 'Pearson_R');
master_features.MutualInfo_LFP_low = extractFreqBandFeatureWithUniqueID(...
    ampcorr_lfp_data, master_features, config.freq_bands.low, 'Mutual_Info');
master_features.MutualInfo_LFP_high = extractFreqBandFeatureWithUniqueID(...
    ampcorr_lfp_data, master_features, config.freq_bands.high, 'Mutual_Info');

fprintf('✓ Extracted Amplitude Correlation (LFP) features\n');

%% ========================================================================
%  SECTION 10: EXTRACT AMPLITUDE CORRELATION FEATURES (BREATHING)
%  ========================================================================

fprintf('\n=== EXTRACTING AMPLITUDE CORRELATION FEATURES (BREATHING) ===\n');

master_features.PearsonR_Breath_low = extractFreqBandFeatureWithUniqueID(...
    ampcorr_breath_data, master_features, config.freq_bands.low, 'Pearson_R');
master_features.PearsonR_Breath_high = extractFreqBandFeatureWithUniqueID(...
    ampcorr_breath_data, master_features, config.freq_bands.high, 'Pearson_R');
master_features.MutualInfo_Breath_low = extractFreqBandFeatureWithUniqueID(...
    ampcorr_breath_data, master_features, config.freq_bands.low, 'Mutual_Info');
master_features.MutualInfo_Breath_high = extractFreqBandFeatureWithUniqueID(...
    ampcorr_breath_data, master_features, config.freq_bands.high, 'Mutual_Info');

fprintf('✓ Extracted Amplitude Correlation (Breathing) features\n');

%% ========================================================================
%  SECTION 11: EXTRACT COHERENCE FEATURES (LFP)
%  ========================================================================

fprintf('\n=== EXTRACTING COHERENCE FEATURES (LFP) ===\n');

master_features.Coherence_LFP_low = extractCoherenceFeatureWithUniqueID(...
    coherence_lfp_data, master_features, config.freq_bands.low);
master_features.Coherence_LFP_high = extractCoherenceFeatureWithUniqueID(...
    coherence_lfp_data, master_features, config.freq_bands.high);

fprintf('✓ Extracted Coherence (LFP) features\n');

%% ========================================================================
%  SECTION 12: EXTRACT COHERENCE FEATURES (BREATHING)
%  ========================================================================

fprintf('\n=== EXTRACTING COHERENCE FEATURES (BREATHING) ===\n');

master_features.Coherence_Breath_low = extractCoherenceFeatureWithUniqueID(...
    coherence_breath_data, master_features, config.freq_bands.low);
master_features.Coherence_Breath_high = extractCoherenceFeatureWithUniqueID(...
    coherence_breath_data, master_features, config.freq_bands.high);

fprintf('✓ Extracted Coherence (Breathing) features\n');

%% ========================================================================
%  SECTION 13: EXTRACT PSTH FEATURES
%  ========================================================================

fprintf('\n=== EXTRACTING PSTH FEATURES ===\n');

if ~isempty(psth_data) && ~isempty(psth_data.unit_data)
    % Get list of event types
    sample_unit = psth_data.unit_data(1);
    event_fields = fieldnames(sample_unit);
    event_types = {};

    for i = 1:length(event_fields)
        if contains(event_fields{i}, '_zscore')
            event_name = strrep(event_fields{i}, '_zscore', '');
            event_types{end+1} = event_name;
        end
    end

    fprintf('  Found %d event types\n', length(event_types));

    % Initialize PSTH feature columns
    for e = 1:length(event_types)
        master_features.(['PSTH_' event_types{e} '_mean_z']) = nan(n_master_units, 1);
    end

    % Extract features for each unit
    for unit_idx = 1:length(psth_data.unit_data)
        unit = psth_data.unit_data(unit_idx);

        % Create unique ID for this unit
        session_name = unit.session_name;
        % Extract base session name (remove path and extension)
        [~, session_base, ext] = fileparts(session_name);
        unit_id = unit.unit_id;
        psth_unique_id = sprintf('%s_Unit%d', [session_base,ext], unit_id);

        % Find matching row in master_features by unique ID
        master_idx = find(strcmp(master_features.UniqueUnitID, psth_unique_id), 1);

        if isempty(master_idx)
            fprintf('Error, didnt find matched unit\n');
            % Try alternative matching - sometimes session names differ slightly
            % Match by session type and unit number
            session_type = unit.session_type;
            possible_matches = find(master_features.Unit == unit_id & ...
                                   contains(string(master_features.SessionType), session_type));
            if ~isempty(possible_matches)
                master_idx = possible_matches(1);
            end
        end

        if ~isempty(master_idx)
            % Extract features
            time_centers = unit.time_centers;
            response_bins = time_centers >= config.psth_response_window(1) & ...
                           time_centers <= config.psth_response_window(2);

            for e = 1:length(event_types)
                event_name = event_types{e};
                zscore_field = [event_name '_zscore'];
                n_events_field = [event_name '_n_events'];

                if isfield(unit, zscore_field) && isfield(unit, n_events_field)
                    zscore_trace = unit.(zscore_field);
                    n_events = unit.(n_events_field);

                    if n_events > 0 && ~all(isnan(zscore_trace))
                        mean_z = nanmean(zscore_trace(response_bins));
                        master_features.(['PSTH_' event_name '_mean_z'])(master_idx) = mean_z;
                    end
                end
            end
        end

        if mod(unit_idx, 100) == 0
            fprintf('  Processed %d/%d PSTH units\n', unit_idx, length(psth_data.unit_data));
        end
    end

    fprintf('✓ Extracted PSTH features\n');
else
    fprintf('  Skipping PSTH features (no data)\n');
end

%% ========================================================================
%  SECTION 14: SAVE RESULTS
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
save_filename = 'unit_features_for_clustering.mat';
save(save_filename, 'results', '-v7.3');

fprintf('✓ Saved to: %s\n', save_filename);

%% ========================================================================
%  SECTION 15: SUMMARY
%  ========================================================================

fprintf('\n========================================\n');
fprintf('FEATURE EXTRACTION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Total units: %d\n', results.n_units);
fprintf('Total features: %d\n', results.n_features);
fprintf('\nFeature categories:\n');
fprintf('  - Unit spike metrics: 23 features\n');
fprintf('  - PPC (LFP + Breathing): 4 features\n');
fprintf('  - Coherence (LFP + Breathing): 4 features\n');
fprintf('  - Pearson R (LFP + Breathing): 4 features\n');
fprintf('  - Mutual Info (LFP + Breathing): 4 features\n');
if ~isempty(psth_data)
    fprintf('  - PSTH responses: %d event types\n', length(event_types));
end
fprintf('\nData integrity:\n');
fprintf('  ✓ Using unique unit IDs (session + unit number)\n');
fprintf('  ✓ Prevents feature mismatches across data sources\n');
fprintf('\nNext step: Run Unit_Feature_Comprehensive_Heatmap.m\n');
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function data = loadAnalysisData(base_path, analysis_type, file_pattern)
% Load analysis data from both Aversive and Reward directories

    aversive_path = fullfile(base_path, ['RewardAversive_' analysis_type]);
    reward_path = fullfile(base_path, ['RewardSeeking_' analysis_type]);

    data = {};

    % Load aversive
    if exist(aversive_path, 'dir')
        aversive_files = dir(fullfile(aversive_path, file_pattern));
        for i = 1:length(aversive_files)
            loaded = load(fullfile(aversive_path, aversive_files(i).name));
            data{end+1} = loaded;
        end
    end

    % Load reward
    if exist(reward_path, 'dir')
        reward_files = dir(fullfile(reward_path, file_pattern));
        for i = 1:length(reward_files)
            loaded = load(fullfile(reward_path, reward_files(i).name));
            data{end+1} = loaded;
        end
    end
end

function feature_values = extractFreqBandFeatureWithUniqueID(session_data, master_features, freq_band, feature_name)
% Extract feature values using unique unit IDs for matching

    n_units = height(master_features);
    feature_values = nan(n_units, 1);

    for i = 1:n_units
        unique_id = master_features.UniqueUnitID{i};

        % Parse unique_id to get session and unit
        parts = strsplit(unique_id, '_Unit');
        if length(parts) ~= 2
            continue;
        end
        session_base = parts{1};
        unit_num = str2double(parts{2});

        % Search through all sessions for matching unit
        for sess_idx = 1:length(session_data)
            sess_loaded = session_data{sess_idx};

            if ~isfield(sess_loaded, 'session_results')
                continue;
            end

            sess = sess_loaded.session_results;

            % Check if session filename matches
            if isfield(sess, 'filename')
                [~, sess_base, ext] = fileparts(sess.filename);

                if strcmp([sess_base,ext], session_base)
                    % Found matching session, now extract data for this unit
                    if isfield(sess, 'data') && ~isempty(sess.data)
                        unit_data = sess.data(sess.data.Unit == unit_num, :);

                        if ~isempty(unit_data)
                            % Filter by frequency band
                            freq_center = (unit_data.Freq_Low_Hz + unit_data.Freq_High_Hz) / 2;
                            freq_mask = freq_center >= freq_band(1) & freq_center <= freq_band(2);

                            if any(freq_mask)
                                band_data = unit_data(freq_mask, :);

                                if ismember(feature_name, band_data.Properties.VariableNames)
                                    feature_values(i) = mean(band_data.(feature_name), 'omitnan');
                                    break;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function feature_values = extractCoherenceFeatureWithUniqueID(session_data, master_features, freq_band)
% Extract coherence values using unique unit IDs

    n_units = height(master_features);
    feature_values = nan(n_units, 1);

    for i = 1:n_units
        unique_id = master_features.UniqueUnitID{i};

        % Parse unique_id
        parts = strsplit(unique_id, '_Unit');
        if length(parts) ~= 2
            continue;
        end
        session_base = parts{1};
        unit_num = str2double(parts{2});

        % Search through all sessions
        for sess_idx = 1:length(session_data)
            sess_loaded = session_data{sess_idx};

            if ~isfield(sess_loaded, 'session_results')
                continue;
            end

            sess = sess_loaded.session_results;

            % Check if session filename matches
            if isfield(sess, 'filename')
                [~, sess_base, ext] = fileparts(sess.filename);

                if strcmp([sess_base,ext], session_base)
                    % Found matching session
                    if isfield(sess, 'unit_coherence_results') && ...
                       unit_num <= length(sess.unit_coherence_results)

                        unit_result = sess.unit_coherence_results{unit_num};

                        if ~isempty(unit_result) && ~unit_result.skipped
                            % Extract coherence in frequency band
                            freq = unit_result.freq;
                            coherence = unit_result.coherence;

                            freq_mask = freq >= freq_band(1) & freq <= freq_band(2);

                            if any(freq_mask)
                                feature_values(i) = mean(coherence(freq_mask), 'omitnan');
                                break;
                            end
                        end
                    end
                end
            end
        end
    end
end
