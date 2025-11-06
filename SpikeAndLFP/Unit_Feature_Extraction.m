%% ========================================================================
%  COMPREHENSIVE UNIT FEATURE EXTRACTION
%  Extracts all features from existing analysis results
%  ========================================================================
%
%  This script loads results from:
%  1. PSTH Survey Analysis (event-triggered responses)
%  2. Spike-LFP Coherence (frequency-specific coupling)
%  3. Spike-Phase Coupling (narrow and broad bands)
%  4. Firing Rate & CV Analysis
%
%  Output: unit_features_comprehensive.mat
%
%% ========================================================================

clear all;
close all;

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== COMPREHENSIVE UNIT FEATURE EXTRACTION ===\n\n');

config = struct();
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
config.n_behaviors = 7;

% Narrow frequency bands
config.narrow_bands = {'1-3Hz', '5-7Hz', '8-10Hz'};
config.n_narrow_bands = 3;

% Broad frequency bands
config.broad_bands = {'Delta', 'Theta', 'Beta', 'Low_Gamma', 'High_Gamma', 'Ultra_Gamma'};
config.n_broad_bands = 6;

% PSTH time window for feature extraction
config.psth_response_window = [0, 1];  % Extract 0-1 sec post-event

fprintf('Configuration:\n');
fprintf('  Behaviors: %d\n', config.n_behaviors);
fprintf('  Narrow bands: %d\n', config.n_narrow_bands);
fprintf('  Broad bands: %d\n', config.n_broad_bands);
fprintf('  PSTH window: [%.1f, %.1f] sec\n\n', config.psth_response_window(1), config.psth_response_window(2));

%% ========================================================================
%  SECTION 2: LOAD PSTH RESULTS
%  ========================================================================

fprintf('Loading PSTH results...\n');

psth_file = 'PSTH_Survey_Results.mat';
if ~exist(psth_file, 'file')
    fprintf('  WARNING: %s not found - PSTH features will be skipped\n', psth_file);
    psth_data = [];
else
    try
        psth_loaded = load(psth_file);
        psth_data = psth_loaded.results;
        fprintf('✓ Loaded PSTH data: %d units\n', psth_data.n_units_total);
    catch ME
        fprintf('  ERROR loading PSTH: %s\n', ME.message);
        psth_data = [];
    end
end

%% ========================================================================
%  SECTION 3: LOAD COHERENCE RESULTS
%  ========================================================================

fprintf('\nLoading Coherence results...\n');

DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingCoherencePath = fullfile(DataSetsPath, 'RewardSeeking_SpikeLFPCoherence_Overall');
RewardAversiveCoherencePath = fullfile(DataSetsPath, 'RewardAversive_SpikeLFPCoherence_Overall');

% Load aversive coherence
aversive_coh_files = dir(fullfile(RewardAversiveCoherencePath, '*_spike_lfp_coherence_overall.mat'));
fprintf('  Found %d aversive coherence sessions\n', length(aversive_coh_files));

aversive_coherence = cell(length(aversive_coh_files), 1);
for i = 1:length(aversive_coh_files)
    data = load(fullfile(RewardAversiveCoherencePath, aversive_coh_files(i).name));
    aversive_coherence{i} = data.session_results;
end

% Load reward coherence
reward_coh_files = dir(fullfile(RewardSeekingCoherencePath, '*_spike_lfp_coherence_overall.mat'));
fprintf('  Found %d reward coherence sessions\n', length(reward_coh_files));

reward_coherence = cell(length(reward_coh_files), 1);
for i = 1:length(reward_coh_files)
    data = load(fullfile(RewardSeekingCoherencePath, reward_coh_files(i).name));
    reward_coherence{i} = data.session_results;
end

fprintf('✓ Loaded coherence data\n');

%% ========================================================================
%  SECTION 4: LOAD PHASE COUPLING RESULTS (NARROW BANDS)
%  ========================================================================

fprintf('\nLoading Phase Coupling results (Narrow Bands)...\n');

RewardSeekingPhasePath = fullfile(DataSetsPath, 'RewardSeeking_SpikePhase_BehaviorSpecific_NarrowBands');
RewardAversivePhasePath = fullfile(DataSetsPath, 'RewardAversive_SpikePhase_BehaviorSpecific_NarrowBands');

% Load aversive phase
aversive_phase_files = dir(fullfile(RewardAversivePhasePath, '*_spike_phase_coupling_narrowbands.mat'));
fprintf('  Found %d aversive phase sessions (narrow bands)\n', length(aversive_phase_files));

aversive_phase_narrow = cell(length(aversive_phase_files), 1);
for i = 1:length(aversive_phase_files)
    data = load(fullfile(RewardAversivePhasePath, aversive_phase_files(i).name));
    aversive_phase_narrow{i} = data.session_results;
end

% Load reward phase
reward_phase_files = dir(fullfile(RewardSeekingPhasePath, '*_spike_phase_coupling_narrowbands.mat'));
fprintf('  Found %d reward phase sessions (narrow bands)\n', length(reward_phase_files));

reward_phase_narrow = cell(length(reward_phase_files), 1);
for i = 1:length(reward_phase_files)
    data = load(fullfile(RewardSeekingPhasePath, reward_phase_files(i).name));
    reward_phase_narrow{i} = data.session_results;
end

fprintf('✓ Loaded phase coupling data (narrow bands)\n');

%% ========================================================================
%  SECTION 5: LOAD PHASE COUPLING RESULTS (BROAD BANDS)
%  ========================================================================

fprintf('\nLoading Phase Coupling results (Broad Bands)...\n');

RewardSeekingPhaseBroadPath = fullfile(DataSetsPath, 'RewardSeeking_SpikePhase_BehaviorSpecific');
RewardAversivePhaseBroadPath = fullfile(DataSetsPath, 'RewardAversive_SpikePhase_BehaviorSpecific');

% Load aversive phase (broad)
aversive_phase_broad_files = dir(fullfile(RewardAversivePhaseBroadPath, '*_spike_phase_coupling_by_behavior.mat'));
fprintf('  Found %d aversive phase sessions (broad bands)\n', length(aversive_phase_broad_files));

aversive_phase_broad = cell(length(aversive_phase_broad_files), 1);
for i = 1:length(aversive_phase_broad_files)
    data = load(fullfile(RewardAversivePhaseBroadPath, aversive_phase_broad_files(i).name));
    aversive_phase_broad{i} = data.session_results;
end

% Load reward phase (broad)
reward_phase_broad_files = dir(fullfile(RewardSeekingPhaseBroadPath, '*_spike_phase_coupling_by_behavior.mat'));
fprintf('  Found %d reward phase sessions (broad bands)\n', length(reward_phase_broad_files));

reward_phase_broad = cell(length(reward_phase_broad_files), 1);
for i = 1:length(reward_phase_broad_files)
    data = load(fullfile(RewardSeekingPhaseBroadPath, reward_phase_broad_files(i).name));
    reward_phase_broad{i} = data.session_results;
end

fprintf('✓ Loaded phase coupling data (broad bands)\n');

%% ========================================================================
%  SECTION 6: EXTRACT FEATURES - COHERENCE
%  ========================================================================

fprintf('\n=== EXTRACTING COHERENCE FEATURES ===\n');

% Combine all sessions
all_coherence_sessions = [aversive_coherence; reward_coherence];
n_coh_sessions = length(all_coherence_sessions);

% Storage for coherence features
coherence_features = [];
unit_counter = 0;

for sess_idx = 1:n_coh_sessions
    session = all_coherence_sessions{sess_idx};

    fprintf('[%d/%d] Processing: %s\n', sess_idx, n_coh_sessions, session.filename);

    n_units = session.n_units;

    for unit_idx = 1:n_units
        unit_result = session.unit_coherence_results{unit_idx};

        if isempty(unit_result) || unit_result.skipped
            continue;
        end

        unit_counter = unit_counter + 1;

        % Initialize feature struct
        feat = struct();
        feat.global_unit_id = unit_counter;
        feat.session_id = sess_idx;
        feat.unit_id = unit_idx;
        feat.session_type = session.session_type;
        feat.session_filename = session.filename;
        feat.n_spikes = unit_result.n_spikes;

        % Extract narrow band coherence (1-3, 5-7, 8-10 Hz)
        freq = unit_result.freq;
        coherence = unit_result.coherence;

        % 1-3 Hz
        mask_1_3 = freq >= 1 & freq <= 3;
        feat.coherence_1_3Hz = mean(coherence(mask_1_3));

        % 5-7 Hz
        mask_5_7 = freq >= 5 & freq <= 7;
        feat.coherence_5_7Hz = mean(coherence(mask_5_7));

        % 8-10 Hz
        mask_8_10 = freq >= 8 & freq <= 10;
        feat.coherence_8_10Hz = mean(coherence(mask_8_10));

        % Peak coherence
        [peak_coh, peak_idx] = max(coherence);
        feat.coherence_peak_mag = peak_coh;
        feat.coherence_peak_freq = freq(peak_idx);

        % Broad band coherence
        feat.coherence_delta = unit_result.band_mean_coherence.Delta;
        feat.coherence_theta = unit_result.band_mean_coherence.Theta;
        feat.coherence_beta = unit_result.band_mean_coherence.Beta;
        feat.coherence_low_gamma = unit_result.band_mean_coherence.Low_Gamma;
        feat.coherence_high_gamma = unit_result.band_mean_coherence.High_Gamma;
        feat.coherence_ultra_gamma = unit_result.band_mean_coherence.Ultra_Gamma;

        % Store
        if unit_counter == 1
            coherence_features = feat;
        else
            coherence_features(unit_counter) = feat;
        end
    end
end

fprintf('✓ Extracted coherence features from %d units\n', unit_counter);

%% ========================================================================
%  SECTION 7: EXTRACT FEATURES - PHASE COUPLING (NARROW BANDS)
%  ========================================================================

fprintf('\n=== EXTRACTING PHASE COUPLING FEATURES (NARROW BANDS) ===\n');

% Combine all sessions
all_phase_narrow_sessions = [aversive_phase_narrow; reward_phase_narrow];
n_phase_sessions = length(all_phase_narrow_sessions);

% Storage for phase features (will match to coherence_features by session + unit)
phase_narrow_features = [];
phase_unit_counter = 0;

for sess_idx = 1:n_phase_sessions
    session = all_phase_narrow_sessions{sess_idx};

    fprintf('[%d/%d] Processing: %s\n', sess_idx, n_phase_sessions, session.filename);

    n_units = session.n_units;

    for unit_idx = 1:n_units
        unit_result = session.unit_results{unit_idx};

        if isempty(unit_result)
            continue;
        end

        phase_unit_counter = phase_unit_counter + 1;

        % Initialize feature struct
        feat = struct();
        feat.phase_unit_id = phase_unit_counter;
        feat.session_id = sess_idx;
        feat.unit_id = unit_idx;
        feat.session_type = session.session_type;
        feat.session_filename = session.filename;

        % Extract MRL and preferred phase for each behavior × narrow band
        % Structure: 3 narrow bands × 7 behaviors
        MRL_narrow = nan(config.n_narrow_bands, config.n_behaviors);
        preferred_phase_narrow = nan(config.n_narrow_bands, config.n_behaviors);
        is_significant_narrow = false(config.n_narrow_bands, config.n_behaviors);

        for band_idx = 1:config.n_narrow_bands
            band_result = unit_result.band_results{band_idx};

            for beh_idx = 1:config.n_behaviors
                beh_result = band_result.behavior_results{beh_idx};

                MRL_narrow(band_idx, beh_idx) = beh_result.MRL;
                preferred_phase_narrow(band_idx, beh_idx) = beh_result.preferred_phase;
                is_significant_narrow(band_idx, beh_idx) = beh_result.is_significant;
            end
        end

        feat.phase_MRL_narrow = MRL_narrow;  % [3 bands × 7 behaviors]
        feat.phase_preferred_narrow = preferred_phase_narrow;
        feat.phase_significant_narrow = is_significant_narrow;

        % Store
        if phase_unit_counter == 1
            phase_narrow_features = feat;
        else
            phase_narrow_features(phase_unit_counter) = feat;
        end
    end
end

fprintf('✓ Extracted narrow-band phase features from %d units\n', phase_unit_counter);

%% ========================================================================
%  SECTION 8: EXTRACT FEATURES - PHASE COUPLING (BROAD BANDS)
%  ========================================================================

fprintf('\n=== EXTRACTING PHASE COUPLING FEATURES (BROAD BANDS) ===\n');

% Combine all sessions
all_phase_broad_sessions = [aversive_phase_broad; reward_phase_broad];
n_phase_broad_sessions = length(all_phase_broad_sessions);

% Storage for broad band phase features
phase_broad_features = [];
phase_broad_counter = 0;

for sess_idx = 1:n_phase_broad_sessions
    session = all_phase_broad_sessions{sess_idx};

    fprintf('[%d/%d] Processing: %s\n', sess_idx, n_phase_broad_sessions, session.filename);

    n_units = session.n_units;

    for unit_idx = 1:n_units
        unit_result = session.unit_results{unit_idx};

        if isempty(unit_result)
            continue;
        end

        phase_broad_counter = phase_broad_counter + 1;

        % Initialize feature struct
        feat = struct();
        feat.phase_broad_unit_id = phase_broad_counter;
        feat.session_id = sess_idx;
        feat.unit_id = unit_idx;
        feat.session_type = session.session_type;
        feat.session_filename = session.filename;

        % Extract MRL and preferred phase for each behavior × broad band
        % Structure: 6 broad bands × 7 behaviors
        MRL_broad = nan(config.n_broad_bands, config.n_behaviors);
        preferred_phase_broad = nan(config.n_broad_bands, config.n_behaviors);
        is_significant_broad = false(config.n_broad_bands, config.n_behaviors);

        for band_idx = 1:config.n_broad_bands
            band_result = unit_result.band_results{band_idx};

            for beh_idx = 1:config.n_behaviors
                beh_result = band_result.behavior_results{beh_idx};

                MRL_broad(band_idx, beh_idx) = beh_result.MRL;
                preferred_phase_broad(band_idx, beh_idx) = beh_result.preferred_phase;
                is_significant_broad(band_idx, beh_idx) = beh_result.is_significant;
            end
        end

        feat.phase_MRL_broad = MRL_broad;  % [6 bands × 7 behaviors]
        feat.phase_preferred_broad = preferred_phase_broad;
        feat.phase_significant_broad = is_significant_broad;

        % Store
        if phase_broad_counter == 1
            phase_broad_features = feat;
        else
            phase_broad_features(phase_broad_counter) = feat;
        end
    end
end

fprintf('✓ Extracted broad-band phase features from %d units\n', phase_broad_counter);

%% ========================================================================
%  SECTION 9: EXTRACT FEATURES - PSTH
%  ========================================================================

fprintf('\n=== EXTRACTING PSTH FEATURES ===\n');

if ~isempty(psth_data)

    psth_features = [];

    % Get list of all event types from first unit
    if ~isempty(psth_data.unit_data)
        sample_unit = psth_data.unit_data(1);
        event_fields = fieldnames(sample_unit);
        event_types = {};

        for i = 1:length(event_fields)
            if contains(event_fields{i}, '_psth')
                event_name = strrep(event_fields{i}, '_psth', '');
                event_types{end+1} = event_name;
            end
        end

        fprintf('  Found %d event types\n', length(event_types));
    else
        fprintf('  WARNING: No PSTH unit data found\n');
        event_types = {};
    end

    % Extract PSTH features for each unit
    for unit_idx = 1:length(psth_data.unit_data)
        unit = psth_data.unit_data(unit_idx);

        % Initialize feature struct
        feat = struct();
        feat.psth_global_unit_id = unit.global_unit_id;
        feat.session_name = unit.session_name;
        feat.session_type = unit.session_type;
        feat.unit_id = unit.unit_id;
        feat.n_spikes = unit.n_spikes;

        % Time vector
        time_centers = unit.time_centers;

        % Find bins in response window [0, 1] sec
        response_bins = time_centers >= config.psth_response_window(1) & ...
                       time_centers <= config.psth_response_window(2);

        % Extract features for each event type
        for e = 1:length(event_types)
            event_name = event_types{e};

            zscore_field = [event_name '_zscore'];
            n_events_field = [event_name '_n_events'];

            if isfield(unit, zscore_field) && isfield(unit, n_events_field)
                zscore_trace = unit.(zscore_field);
                n_events = unit.(n_events_field);

                if n_events > 0 && ~all(isnan(zscore_trace))
                    % Mean Z-score in response window [0-1 sec]
                    mean_z = nanmean(zscore_trace(response_bins));

                    % Peak Z-score in response window
                    [peak_z, peak_idx_local] = max(abs(zscore_trace(response_bins)));
                    peak_z_signed = zscore_trace(find(response_bins, 1) + peak_idx_local - 1);

                    % Response type
                    if peak_z_signed > 2
                        response_type = 1;  % Excitation
                    elseif peak_z_signed < -2
                        response_type = -1;  % Inhibition
                    else
                        response_type = 0;  % None
                    end

                    % Store
                    feat.([event_name '_mean_z_0to1sec']) = mean_z;
                    feat.([event_name '_peak_z']) = peak_z_signed;
                    feat.([event_name '_response_type']) = response_type;
                    feat.([event_name '_n_events']) = n_events;
                else
                    feat.([event_name '_mean_z_0to1sec']) = NaN;
                    feat.([event_name '_peak_z']) = NaN;
                    feat.([event_name '_response_type']) = 0;
                    feat.([event_name '_n_events']) = n_events;
                end
            end
        end

        % Store
        if unit_idx == 1
            psth_features = feat;
        else
            psth_features(unit_idx) = feat;
        end

        if mod(unit_idx, 100) == 0
            fprintf('  Processed %d/%d units\n', unit_idx, length(psth_data.unit_data));
        end
    end

    fprintf('✓ Extracted PSTH features from %d units\n', length(psth_features));
else
    fprintf('  Skipping PSTH features (no data loaded)\n');
    psth_features = [];
end

%% ========================================================================
%  SECTION 10: COMPUTE ADDITIONAL FIRING STATISTICS (FR & CV)
%  ========================================================================

fprintf('\n=== COMPUTING FIRING STATISTICS (FR & CV) ===\n');

% Configuration
min_spikes_for_CV = 10;  % Minimum spikes to calculate CV reliably

% Load sorting parameters (needed for raw spike data loading)
fprintf('Loading sorting parameters for spike data...\n');
[T_sorted] = loadSortingParameters();

% Load raw spike data for aversive sessions
fprintf('Loading raw spike data for FR/CV calculation...\n');
numofsession = 2;
folderpath = "/Volumes/ExpansionBackup/Data/Struct_spike";

% Process aversive sessions
[aversive_files, ~, num_aversive] = selectFilesWithAnimalIDFiltering(folderpath, numofsession, '2025*RewardAversive*.mat');
fprintf('  Found %d aversive session files\n', num_aversive);

% Process reward sessions
[reward_files, ~, num_reward] = selectFilesWithAnimalIDFiltering(folderpath, numofsession, '2025*RewardSeeking*.mat');
fprintf('  Found %d reward session files\n', num_reward);

% Combine all spike files
all_spike_files = [aversive_files; reward_files];
all_spike_sessions = cell(length(all_spike_files), 1);

% Initialize FR/CV fields in coherence_features
for i = 1:length(coherence_features)
    coherence_features(i).n_spikes_total = coherence_features(i).n_spikes;
    coherence_features(i).firing_rate_mean = NaN;
    coherence_features(i).cv = NaN;
end

% Load and process each spike file
fprintf('Processing spike files for FR/CV calculation...\n');
for file_idx = 1:length(all_spike_files)
    fprintf('[%d/%d] %s\n', file_idx, length(all_spike_files), all_spike_files(file_idx).name);

    try
        % Load spike data
        Timelimits = 'No';
        [~, ~, ~, ~, ~, ~, ~, ~, ~, ~, valid_spikes, ~, TriggerMid] = ...
            loadAndPrepareSessionData(all_spike_files(file_idx), T_sorted, Timelimits);

        % Calculate session duration
        session_start = TriggerMid(1);
        session_end = TriggerMid(end);
        session_duration = session_end - session_start;

        % Store session info
        all_spike_sessions{file_idx} = struct();
        all_spike_sessions{file_idx}.filename = all_spike_files(file_idx).name;
        all_spike_sessions{file_idx}.n_units = length(valid_spikes);
        all_spike_sessions{file_idx}.duration = session_duration;

        % Find matching coherence features for this session
        spike_basename = extractBefore(all_spike_files(file_idx).name, '.mat');

        for coh_idx = 1:length(coherence_features)
            % Match by filename
            if contains(coherence_features(coh_idx).session_filename, spike_basename)
                unit_id = coherence_features(coh_idx).unit_id;

                % Check if unit exists in spike data
                if unit_id <= length(valid_spikes)
                    spike_times = valid_spikes{unit_id};

                    if ~isempty(spike_times)
                        % Calculate firing rate (Hz)
                        n_spikes = length(spike_times);
                        firing_rate = n_spikes / session_duration;
                        coherence_features(coh_idx).firing_rate_mean = firing_rate;

                        % Calculate CV from inter-spike intervals
                        if n_spikes >= min_spikes_for_CV
                            ISI = diff(spike_times);
                            if ~isempty(ISI) && mean(ISI) > 0
                                CV = std(ISI) / mean(ISI);
                                coherence_features(coh_idx).cv = CV;
                            end
                        end
                    end
                end
            end
        end

    catch ME
        fprintf('  Warning: Failed to process %s: %s\n', all_spike_files(file_idx).name, ME.message);
        continue;
    end
end

% Count how many units have FR/CV computed
n_with_FR = sum(~isnan([coherence_features.firing_rate_mean]));
n_with_CV = sum(~isnan([coherence_features.cv]));

fprintf('✓ FR/CV statistics computed\n');
fprintf('  Units with firing rate: %d/%d\n', n_with_FR, length(coherence_features));
fprintf('  Units with CV: %d/%d\n', n_with_CV, length(coherence_features));

%% ========================================================================
%  SECTION 11: SAVE EXTRACTED FEATURES
%  ========================================================================

fprintf('\n=== SAVING EXTRACTED FEATURES ===\n');

% Package all features
unit_features_comprehensive = struct();
unit_features_comprehensive.coherence_features = coherence_features;
unit_features_comprehensive.phase_narrow_features = phase_narrow_features;
unit_features_comprehensive.phase_broad_features = phase_broad_features;
unit_features_comprehensive.psth_features = psth_features;
unit_features_comprehensive.config = config;
unit_features_comprehensive.n_units_coherence = length(coherence_features);
unit_features_comprehensive.n_units_phase_narrow = length(phase_narrow_features);
unit_features_comprehensive.n_units_phase_broad = length(phase_broad_features);
unit_features_comprehensive.n_units_psth = length(psth_features);

% Save
save_filename = 'unit_features_comprehensive.mat';
save(save_filename, 'unit_features_comprehensive', '-v7.3');

fprintf('✓ Saved to: %s\n', save_filename);

%% ========================================================================
%  SECTION 12: SUMMARY
%  ========================================================================

fprintf('\n========================================\n');
fprintf('FEATURE EXTRACTION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Features extracted:\n');
fprintf('  Coherence: %d units\n', length(coherence_features));
fprintf('    - Narrow bands (1-3, 5-7, 8-10 Hz)\n');
fprintf('    - Broad bands (Delta, Theta, Beta, Gammas)\n');
fprintf('    - Peak coherence frequency & magnitude\n');
fprintf('    - Firing rate (Hz)\n');
fprintf('    - Coefficient of variation (CV)\n');
fprintf('\n');
fprintf('  Phase Coupling (Narrow): %d units\n', length(phase_narrow_features));
fprintf('    - MRL for 3 narrow bands × 7 behaviors\n');
fprintf('    - Preferred phase angles\n');
fprintf('    - Significance flags\n');
fprintf('\n');
fprintf('  Phase Coupling (Broad): %d units\n', length(phase_broad_features));
fprintf('    - MRL for 6 broad bands × 7 behaviors\n');
fprintf('    - Preferred phase angles\n');
fprintf('    - Significance flags\n');
fprintf('\n');
fprintf('  PSTH: %d units\n', length(psth_features));
fprintf('    - Mean Z-score [0-1 sec] per event\n');
fprintf('    - Peak Z-score per event\n');
fprintf('    - Response type (excite/inhibit/none)\n');
fprintf('    - Number of events\n');
fprintf('\n');
fprintf('  Firing Statistics: %d units with FR, %d units with CV\n', n_with_FR, n_with_CV);
fprintf('\n');
fprintf('Next step: Run Unit_Feature_Visualization.m\n');
fprintf('========================================\n');
