%% ========================================================================
%  PSTH SURVEY ANALYSIS: All Units and Events
%  Events: Reward bouts, Aversive onsets, Behavioral transitions, Movement
%  Analyze reward and aversive sessions separately
%  ========================================================================
%
%  This script calculates PSTH for:
%  1. Reward bouts (IR1ON, IR2ON, WP1ON, WP2ON) - detected using bout clustering
%  2. Aversive sound onsets
%  3. Behavioral state transitions (7 LSTM classes)
%  4. Behavioral matrix transitions (8 features)
%  5. Movement onset/offset
%
%  REWRITTEN: Uses loadBehaviorPredictionsFromSpikeFiles() and
%             loadSessionMetricsFromSpikeFiles() for data loading
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== PSTH SURVEY ANALYSIS ===\n\n');

config = struct();

% Behavioral class names (LSTM predictions)
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
config.n_behaviors = 7;

% Behavioral matrix feature names
config.matrix_feature_names = {'HighSpeed', 'NonRewardCorner', 'RewardCornerNoPort', ...
                               'AtRewardPort', 'CenterArea', 'Rearing_BM', ...
                               'GoalDirected', 'HighFreqBreathing'};
config.n_matrix_features = 8;
config.breathing_freq_threshold = 5;  % Hz - threshold for high frequency breathing

% PSTH parameters
config.psth_window = [-2, 2];           % -2 to +2 sec around event
config.psth_bin_size = 0.05;            % 50 ms bins
config.baseline_window = [-2, -0.5];    % Baseline for z-score

% Event detection parameters
config.bout_epsilon = 0.5;              % Max gap for reward bout detection (sec)
config.bout_minPts = 3;                 % Minimum events per bout
config.movement_percentile = 5;         % 5th percentile for movement threshold
config.camera_fps = 20;                 % 20 Hz

% Statistical parameters
config.zscore_threshold = 2;            % Z > 2 or Z < -2 for significance

% Data paths
config.prediction_folder = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/BehaviorPrediction';
config.spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';
config.numofsession = 999;              % Max sessions per animal

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/NewScripts/');

fprintf('Configuration:\n');
fprintf('  PSTH window: [%.1f, %.1f] sec\n', config.psth_window(1), config.psth_window(2));
fprintf('  Bin size: %.0f ms\n', config.psth_bin_size * 1000);
fprintf('  Bout epsilon: %.2f sec\n', config.bout_epsilon);
fprintf('  Movement threshold: %dth percentile\n', config.movement_percentile);
fprintf('  Z-score threshold: %.1f\n\n', config.zscore_threshold);

%% Create time vector for PSTH
time_vector = config.psth_window(1):config.psth_bin_size:config.psth_window(2);
n_bins = length(time_vector) - 1;
time_centers = time_vector(1:end-1) + config.psth_bin_size/2;

% Baseline bin indices
baseline_bins = (time_centers >= config.baseline_window(1)) & ...
                (time_centers <= config.baseline_window(2));

%% ========================================================================
%  SECTION 2: LOAD SORTING PARAMETERS
%  ========================================================================

fprintf('Loading sorting parameters...\n');
[T_sorted] = loadSortingParameters();
fprintf('✓ Sorting parameters loaded\n\n');

%% ========================================================================
%  SECTION 3: INITIALIZE STORAGE
%  ========================================================================

% Storage for all units across sessions
all_unit_data = [];
unit_counter = 0;

%% ========================================================================
%  SECTION 4: PROCESS BOTH SESSION TYPES
%  ========================================================================

session_types = {'Aversive', 'Reward'};
session_patterns = {'2025*RewardAversive*.mat', '2025*RewardSeeking*.mat'};

for session_type_idx = 1:2
    session_type = session_types{session_type_idx};
    session_pattern = session_patterns{session_type_idx};

    fprintf('\n==== PROCESSING %s SESSIONS ====\n', upper(session_type));

    % Select spike files
    [allfiles, ~, ~, ~] = selectFilesWithAnimalIDFiltering(...
        config.spike_folder, config.numofsession, session_pattern);
    n_sessions = length(allfiles);
    fprintf('Found %d %s sessions\n', n_sessions, session_type);

    % Load LSTM predictions matched to spike files
    try
        prediction_sessions = loadBehaviorPredictionsFromSpikeFiles(allfiles, config.prediction_folder);
        fprintf('✓ Loaded LSTM predictions: %d sessions\n', length(prediction_sessions));
    catch ME
        fprintf('❌ Failed to load LSTM predictions: %s\n', ME.message);
        continue;
    end

    % Load behavioral matrices matched to spike files
    try
        sessions = loadSessionMetricsFromSpikeFiles(allfiles, T_sorted);
        fprintf('✓ Loaded behavioral matrices: %d sessions\n\n', length(sessions));
    catch ME
        fprintf('❌ Failed to load behavioral matrices: %s\n', ME.message);
        continue;
    end

    %% Process each session
    for sess_idx = 1:n_sessions
        fprintf('[%d/%d] Processing: %s\n', sess_idx, n_sessions, allfiles(sess_idx).name);
        tic;

        % Load session data
        Timelimits = 'No';
        try
            [NeuralTime, AdjustedXYZ, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, WP1ON, WP2ON, ...
             AversiveSound, sessionLabels, valid_spikes, Fs, TriggerMid] = ...
                loadAndPrepareSessionData(allfiles(sess_idx), T_sorted, Timelimits);
        catch ME
            fprintf('  ERROR loading session: %s\n', ME.message);
            continue;
        end

        spike_filename = allfiles(sess_idx).name;
        fprintf('  Loaded %d units\n', length(valid_spikes));

        %% Extract behavioral data
        session = sessions{sess_idx};

        % Get LSTM predictions and map to neural time
        BehaviorClass = [];
        if sess_idx <= length(prediction_sessions) && ~isempty(prediction_sessions(sess_idx).prediction_scores)
            prediction_scores = prediction_sessions(sess_idx).prediction_scores;
            [~, dominant_beh] = max(prediction_scores, [], 2);
            n_predictions = length(dominant_beh);

            % Map LSTM predictions (1 Hz) to neural time
            BehaviorClass = zeros(size(NeuralTime));
            for pred_idx = 1:n_predictions
                frame_start = (pred_idx - 1) * config.camera_fps + 1;
                frame_end = min(frame_start + config.camera_fps - 1, length(TriggerMid));

                if frame_end <= length(TriggerMid)
                    pred_time_start = TriggerMid(frame_start);
                    pred_time_end = TriggerMid(frame_end);
                    neural_indices = find(NeuralTime >= pred_time_start & NeuralTime <= pred_time_end);
                    BehaviorClass(neural_indices) = dominant_beh(pred_idx);
                end
            end
            fprintf('  LSTM predictions: %d predictions mapped to neural time\n', n_predictions);
        else
            fprintf('  WARNING: No LSTM predictions available\n');
        end

        % Get behavioral matrix
        behavioral_matrix_full = [];
        if isfield(session, 'behavioral_matrix_full') && ~isempty(session.behavioral_matrix_full)
            behavioral_matrix_full = session.behavioral_matrix_full;
            fprintf('  Behavioral matrix: %d × %d\n', size(behavioral_matrix_full));
        else
            fprintf('  WARNING: No behavioral matrix available\n');
        end

        %% Detect all event types
        event_times = detectAllEvents(NeuralTime, IR1ON, IR2ON, WP1ON, WP2ON, ...
            AversiveSound, BehaviorClass, behavioral_matrix_full, AdjustedXYZ_speed, ...
            config, TriggerMid);

        %% Calculate PSTH for each unit
        % Define all possible event types
        all_event_types = getAllEventTypes(config);

        for unit_idx = 1:length(valid_spikes)
            unit_counter = unit_counter + 1;
            spike_times = valid_spikes{unit_idx};

            if isempty(spike_times)
                continue;
            end

            % Initialize unit data
            unit_data = struct();
            unit_data.unit_id = unit_idx;
            unit_data.global_unit_id = unit_counter;
            unit_data.session_name = spike_filename;
            unit_data.session_type = session_type;
            unit_data.n_spikes = length(spike_times);
            unit_data.time_centers = time_centers;

            % Calculate PSTH for each event type
            for e = 1:length(all_event_types)
                event_type = all_event_types{e};

                % Get event list
                if isfield(event_times, event_type)
                    event_list = event_times.(event_type);
                else
                    event_list = [];
                end

                % Calculate PSTH
                [psth, zscore_psth, significant_bins, n_events, baseline_mean, baseline_std] = ...
                    calculatePSTH(spike_times, event_list, time_vector, time_centers, ...
                    baseline_bins, config.psth_bin_size, config.zscore_threshold);

                % Store results
                unit_data.([event_type '_psth']) = psth;
                unit_data.([event_type '_zscore']) = zscore_psth;
                unit_data.([event_type '_significant']) = significant_bins;
                unit_data.([event_type '_n_events']) = n_events;
                unit_data.([event_type '_baseline_mean']) = baseline_mean;
                unit_data.([event_type '_baseline_std']) = baseline_std;
            end

            % Add to results
            if unit_counter == 1
                all_unit_data = unit_data;
            else
                all_unit_data(unit_counter) = unit_data;
            end
        end

        elapsed = toc;
        fprintf('  ✓ Processed %d units in %.1f sec\n', length(valid_spikes), elapsed);
    end

    fprintf('\n✓ Processed %d %s sessions\n', n_sessions, session_type);
end

%% ========================================================================
%  SECTION 5: SAVE RESULTS
%  ========================================================================

fprintf('\nSaving results...\n');

results = struct();
results.config = config;
results.unit_data = all_unit_data;
results.n_units_total = unit_counter;
results.time_centers = time_centers;

save_filename = 'PSTH_Survey_Results.mat';
save(save_filename, 'results', '-v7.3');

fprintf('✓ Results saved to: %s\n', save_filename);
fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Total units analyzed: %d\n', unit_counter);
fprintf('\nNext step: Run Visualize_PSTH_Survey.m\n');


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function event_times = detectAllEvents(NeuralTime, IR1ON, IR2ON, WP1ON, WP2ON, ...
    AversiveSound, BehaviorClass, behavioral_matrix_full, AdjustedXYZ_speed, config, TriggerMid)
% Detect all event types and return as structure

    event_times = struct();

    fprintf('  Detecting events...\n');

    %% 1. Reward bouts (IR1ON, IR2ON, WP1ON, WP2ON)
    reward_event_names = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON'};
    reward_signals = {IR1ON, IR2ON, WP1ON, WP2ON};

    for r = 1:length(reward_event_names)
        event_name = reward_event_names{r};
        event_signal = reward_signals{r};
        event_indices = find(event_signal == 1);

        if ~isempty(event_indices)
            event_times_raw = NeuralTime(event_indices);
            try
                [bout_starts, ~] = findEventCluster_SuperFast(event_times_raw, ...
                    config.bout_epsilon, config.bout_minPts);
                event_times.(event_name) = bout_starts;
                fprintf('    %s: %d bouts\n', event_name, length(bout_starts));
            catch
                event_times.(event_name) = [];
            end
        else
            event_times.(event_name) = [];
        end
    end

    %% 2. Aversive sound onsets
    aversive_onsets = find(diff(AversiveSound) == 1);
    if ~isempty(aversive_onsets)
        event_times.AversiveOnset = NeuralTime(aversive_onsets);
        fprintf('    Aversive onsets: %d events\n', length(aversive_onsets));
    else
        event_times.AversiveOnset = [];
    end

    %% 3. LSTM behavioral onsets/offsets (7 classes)
    if ~isempty(BehaviorClass)
        for beh_class = 1:7
            % Onset
            onset_indices = find(diff(BehaviorClass == beh_class) == 1);
            if ~isempty(onset_indices)
                event_times.(['Beh' num2str(beh_class) '_Onset']) = NeuralTime(onset_indices + 1);
            else
                event_times.(['Beh' num2str(beh_class) '_Onset']) = [];
            end

            % Offset
            offset_indices = find(diff(BehaviorClass == beh_class) == -1);
            if ~isempty(offset_indices)
                event_times.(['Beh' num2str(beh_class) '_Offset']) = NeuralTime(offset_indices + 1);
            else
                event_times.(['Beh' num2str(beh_class) '_Offset']) = [];
            end
        end
        fprintf('    LSTM behavioral transitions: 7 behaviors\n');
    else
        for beh_class = 1:7
            event_times.(['Beh' num2str(beh_class) '_Onset']) = [];
            event_times.(['Beh' num2str(beh_class) '_Offset']) = [];
        end
    end

    %% 4. Behavioral matrix events (8 features)
    if ~isempty(behavioral_matrix_full)
        n_features = min(8, size(behavioral_matrix_full, 2));

        for feat_idx = 1:n_features
            feat_name = config.matrix_feature_names{feat_idx};

            % Get feature signal
            if feat_idx == 8
                feat_signal = behavioral_matrix_full(:, feat_idx) > config.breathing_freq_threshold;
            else
                feat_signal = behavioral_matrix_full(:, feat_idx);
            end

            % Find onsets
            onset_indices = find(feat_signal == 1);

            if ~isempty(onset_indices)
                onset_times_raw = NeuralTime(onset_indices);
                try
                    [bout_starts, bout_ends] = findEventCluster_SuperFast(onset_times_raw, ...
                        config.bout_epsilon, config.bout_minPts);
                    event_times.([feat_name '_Onset']) = bout_starts;
                    event_times.([feat_name '_Offset']) = bout_ends;
                catch
                    event_times.([feat_name '_Onset']) = [];
                    event_times.([feat_name '_Offset']) = [];
                end
            else
                event_times.([feat_name '_Onset']) = [];
                event_times.([feat_name '_Offset']) = [];
            end
        end
        fprintf('    Behavioral matrix events: 8 features\n');
    else
        for feat_idx = 1:8
            feat_name = config.matrix_feature_names{feat_idx};
            event_times.([feat_name '_Onset']) = [];
            event_times.([feat_name '_Offset']) = [];
        end
    end

    %% 5. Movement onset/offset
    movement_threshold = prctile(AdjustedXYZ_speed, config.movement_percentile);
    is_moving = AdjustedXYZ_speed > movement_threshold;

    movement_onsets = find(diff(is_moving) == 1);
    movement_offsets = find(diff(is_moving) == -1);

    if ~isempty(movement_onsets)
        event_times.MovementOnset = NeuralTime(movement_onsets + 1);
        fprintf('    Movement onsets: %d events\n', length(movement_onsets));
    else
        event_times.MovementOnset = [];
    end

    if ~isempty(movement_offsets)
        event_times.MovementOffset = NeuralTime(movement_offsets + 1);
        fprintf('    Movement offsets: %d events\n', length(movement_offsets));
    else
        event_times.MovementOffset = [];
    end
end

function all_event_types = getAllEventTypes(config)
% Get list of all event types for consistent processing

    all_event_types = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON', 'AversiveOnset', ...
                       'MovementOnset', 'MovementOffset'};

    % Add LSTM behavioral onsets/offsets (7 classes)
    for beh_class = 1:config.n_behaviors
        all_event_types{end+1} = ['Beh' num2str(beh_class) '_Onset'];
        all_event_types{end+1} = ['Beh' num2str(beh_class) '_Offset'];
    end

    % Add behavioral matrix feature onsets/offsets (8 features)
    for feat_idx = 1:config.n_matrix_features
        feat_name = config.matrix_feature_names{feat_idx};
        all_event_types{end+1} = [feat_name '_Onset'];
        all_event_types{end+1} = [feat_name '_Offset'];
    end
end

function [psth, zscore_psth, significant_bins, n_events, baseline_mean, baseline_std] = ...
    calculatePSTH(spike_times, event_list, time_vector, time_centers, baseline_bins, bin_size, zscore_threshold)
% Calculate PSTH for a single unit and event type

    n_bins = length(time_centers);

    if isempty(event_list)
        % No events - return NaN
        psth = nan(n_bins, 1);
        zscore_psth = nan(n_bins, 1);
        significant_bins = false(n_bins, 1);
        n_events = 0;
        baseline_mean = nan;
        baseline_std = nan;
        return;
    end

    % Calculate PSTH across all events
    psth_matrix = zeros(length(event_list), n_bins);

    for ev_idx = 1:length(event_list)
        event_time = event_list(ev_idx);

        % Get relative spike times
        rel_spikes = spike_times - event_time;
        rel_spikes = rel_spikes(rel_spikes >= time_vector(1) & rel_spikes <= time_vector(end));

        % Bin spikes
        trial_psth = histcounts(rel_spikes, time_vector);
        psth_matrix(ev_idx, :) = trial_psth / bin_size;  % Convert to Hz
    end

    % Average across trials
    psth = mean(psth_matrix, 1)';

    % Calculate baseline statistics
    baseline_fr = psth(baseline_bins);
    baseline_mean = mean(baseline_fr);
    baseline_std = std(baseline_fr);

    % Z-score normalization
    if baseline_std > 0
        zscore_psth = (psth - baseline_mean) / baseline_std;
    else
        zscore_psth = zeros(size(psth));
    end

    % Statistical significance
    significant_bins = abs(zscore_psth) > zscore_threshold;
    n_events = length(event_list);
end
