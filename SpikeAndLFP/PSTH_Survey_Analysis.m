%% ========================================================================
%  PSTH SURVEY ANALYSIS: All Units and Events
%  Events: Reward bouts, Aversive onsets, Behavioral transitions, Movement
%  Analyze reward and aversive sessions separately
%  ========================================================================
%
%  This script calculates PSTH for:
%  1. Reward bouts (IR1ON, IR2ON, WP1ON, WP2ON) - detected using bout clustering
%  2. Aversive sound onsets
%  3. Behavioral state transitions (7 classes)
%  4. Movement onset/offset
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== PSTH SURVEY ANALYSIS ===\n\n');

config = struct();
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};

% Behavioral matrix feature names (from create_behavioral_matrix.m)
config.behavioral_matrix_names = {'HighSpeed', 'NonRewardCorner', 'RewardCornerNoPort', ...
                                  'AtRewardPort', 'CenterArea', 'Rearing_BM', 'GoalDirected', ...
                                  'HighFreqBreathing'};
config.breathing_freq_threshold = 5;  % Hz - threshold for high frequency breathing

% PSTH parameters
config.psth_window = [-2, 2];           % -2 to +2 sec around event
config.psth_bin_size = 0.05;            % 50 ms bins
config.baseline_window = [-2, -0.5];    % Baseline for z-score

% Event detection parameters
config.bout_epsilon = 0.5;              % Max gap for reward bout detection
config.bout_minPts = 3;                 % Minimum events per bout
config.movement_percentile = 5;         % 5th percentile for movement threshold
config.camera_fps = 20;                 % 20 Hz

% Statistical parameters
config.zscore_threshold = 2;            % Z > 2 or Z < -2 for significance

% Add path to bout detection function
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');

fprintf('Configuration:\n');
fprintf('  PSTH window: [%.1f, %.1f] sec\n', config.psth_window(1), config.psth_window(2));
fprintf('  Bin size: %.0f ms\n', config.psth_bin_size * 1000);
fprintf('  Bout epsilon: %.2f sec\n', config.bout_epsilon);
fprintf('  Movement threshold: %dth percentile\n\n', config.movement_percentile);

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
%  SECTION 3: LOAD BEHAVIOR PREDICTION DATA
%  ========================================================================

fprintf('Loading LSTM behavior predictions...\n');

% Load aversive behavior predictions
try
    pred_data_aversive = load('lstm_prediction_results_aversive_27-Oct-2025');
    prediction_sessions_aversive = pred_data_aversive.final_results.session_predictions;
    fprintf('✓ Loaded aversive behavior predictions: %d sessions\n', length(prediction_sessions_aversive));
catch ME
    fprintf('❌ Failed to load aversive behavior data: %s\n', ME.message);
    return;
end

% Load reward behavior predictions
try
    pred_data_reward = load('lstm_prediction_results_reward_27-Oct-2025');
    prediction_sessions_reward = pred_data_reward.final_results.session_predictions;
    fprintf('✓ Loaded reward behavior predictions: %d sessions\n\n', length(prediction_sessions_reward));
catch ME
    fprintf('❌ Failed to load reward behavior data: %s\n', ME.message);
    return;
end

% Also load coupling data for session info
try
    coupling_data_aversive = load('27-Oct-2025_RewardAversive_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_aversive = coupling_data_aversive.all_session_metrics;
    coupling_data_reward = load('27-Oct-2025_RewardSeeking_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_reward = coupling_data_reward.all_session_metrics;
    fprintf('✓ Loaded coupling data for session matching\n\n');
catch ME
    fprintf('❌ Failed to load coupling data: %s\n', ME.message);
    return;
end

%% ========================================================================
%  SECTION 4: INITIALIZE STORAGE
%  ========================================================================

% Storage for all units across sessions
all_unit_data = [];
unit_counter = 0;

%% ========================================================================
%  SECTION 5: PROCESS AVERSIVE SESSIONS
%  ========================================================================

fprintf('\n==== PROCESSING AVERSIVE SESSIONS ====\n');

numofsession = 2;
folderpath = "/Volumes/ExpansionBackUp/Data/Struct_spike";
[allfiles, folderpath, num_aversive_sessions] = selectFilesWithAnimalIDFiltering(folderpath, numofsession, '2025*RewardAversive*.mat');

fprintf('Found %d aversive sessions\n', num_aversive_sessions);

for spike_sess_idx = 1:num_aversive_sessions
    fprintf('\n[%d/%d] Processing: %s\n', spike_sess_idx, num_aversive_sessions, allfiles(spike_sess_idx).name);
    tic;

    % Load session data
    Timelimits = 'No';
    try
        [NeuralTime, AdjustedXYZ, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, WP1ON, WP2ON, ...
         AversiveSound, sessionLabels, valid_spikes, Fs, TriggerMid] = ...
            loadAndPrepareSessionData(allfiles(spike_sess_idx), T_sorted, Timelimits);
    catch ME
        fprintf('  ERROR loading session: %s\n', ME.message);
        continue;
    end

    spike_filename = allfiles(spike_sess_idx).name;
    session_type = 'Aversive';
    SpeedSmoothed = AdjustedXYZ_speed;  % Rename for clarity

    fprintf('  Loaded %d units\n', length(valid_spikes));

    % Match with behavior prediction session by filename
    matched = false;
    BehaviorClass = [];  % Will be filled if match found

    for beh_sess_idx = 1:length(sessions_aversive)
        beh_session = sessions_aversive{beh_sess_idx};

        % Simple filename matching
        if contains(spike_filename, extractBefore(beh_session.filename, '.mat'))
            matched = true;

            % Get behavior predictions (1 Hz - 20 frames per prediction)
            prediction_scores = prediction_sessions_aversive(beh_sess_idx).prediction_scores;
            n_predictions = size(prediction_scores, 1);

            % Get dominant behavior at each prediction time
            [max_confidence, dominant_beh] = max(prediction_scores, [], 2);

            % Map predictions to NeuralTime
            % Each prediction uses 20 frames at 20 fps = 1 second
            % Prediction i corresponds to frames [(i-1)*20+1 : i*20]
            BehaviorClass = zeros(size(NeuralTime));  % Initialize
            for pred_idx = 1:n_predictions
                frame_start = (pred_idx - 1) * 20 + 1;
                frame_end = min(frame_start + 19, length(TriggerMid));

                if frame_end <= length(TriggerMid)
                    % Get time range for this prediction
                    pred_time_start = TriggerMid(frame_start);
                    pred_time_end = TriggerMid(frame_end);

                    % Find NeuralTime indices within this range
                    neural_indices = find(NeuralTime >= pred_time_start & NeuralTime <= pred_time_end);

                    % Assign behavior class to all neural samples in this prediction window
                    BehaviorClass(neural_indices) = dominant_beh(pred_idx);
                end
            end

            fprintf('  Matched with behavior session: %s\n', beh_session.filename);
            fprintf('  Behavior predictions: %d predictions (1 Hz)\n', n_predictions);

            % Extract behavioral matrix if available
            if isfield(beh_session, 'behavioral_matrix_full')
                behavioral_matrix_full = beh_session.behavioral_matrix_full;
                fprintf('  Behavioral matrix: %d × %d\n', size(behavioral_matrix_full, 1), size(behavioral_matrix_full, 2));
            else
                behavioral_matrix_full = [];
                fprintf('  WARNING: No behavioral_matrix_full found\n');
            end

            break;
        end
    end

    if ~matched
        fprintf('  WARNING: No behavior match for %s - skipping behavioral transitions\n', spike_filename);
        behavioral_matrix_full = [];
    end

    %% Detect Events
    event_times = struct();

    % 1. Reward bouts (IR1ON, IR2ON, WP1ON, WP2ON)
    % Note: These signals are at NeuralTime sampling rate, NOT camera frame rate
    fprintf('  Detecting reward bouts...\n');
    reward_event_names = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON'};
    reward_signals = {IR1ON, IR2ON, WP1ON, WP2ON};

    for r = 1:length(reward_event_names)
        event_name = reward_event_names{r};
        event_signal = reward_signals{r};

        event_indices = find(event_signal == 1);

        if ~isempty(event_indices)
            % Use NeuralTime (not TriggerMid) since IR/WP signals are at neural sampling rate
            event_times_raw = NeuralTime(event_indices);

            % Detect bouts
            try
                [bout_starts, ~] = findEventCluster_SuperFast(event_times_raw, ...
                    config.bout_epsilon, config.bout_minPts);
                event_times.(event_name) = bout_starts;
                fprintf('    %s: %d bouts\n', event_name, length(bout_starts));
            catch ME
                fprintf('    %s: ERROR in bout detection - %s\n', event_name, ME.message);
                event_times.(event_name) = [];
            end
        else
            event_times.(event_name) = [];
        end
    end

    % 2. Aversive sound onsets
    % AversiveSound is at NeuralTime sampling rate
    fprintf('  Detecting aversive sound onsets...\n');
    aversive_onsets = find(diff(AversiveSound) == 1);
    if ~isempty(aversive_onsets)
        event_times.AversiveOnset = NeuralTime(aversive_onsets);
        fprintf('    Aversive onsets: %d events\n', length(event_times.AversiveOnset));
    else
        event_times.AversiveOnset = [];
    end

    % 3. Behavioral onsets and offsets for each of the 7 classes
    % BehaviorClass is at NeuralTime sampling rate (only if behavior data matched)
    fprintf('  Detecting behavioral onsets and offsets for 7 classes...\n');
    if matched && ~isempty(BehaviorClass)
        for beh_class = 1:7
            % Find onset: transition TO this behavior from any other
            onset_indices = find(diff(BehaviorClass == beh_class) == 1);
            if ~isempty(onset_indices)
                event_times.(['Beh' num2str(beh_class) '_Onset']) = NeuralTime(onset_indices + 1);
                fprintf('    Behavior %d (%s) onset: %d events\n', ...
                    beh_class, config.behavior_names{beh_class}, length(onset_indices));
            else
                event_times.(['Beh' num2str(beh_class) '_Onset']) = [];
            end

            % Find offset: transition FROM this behavior to any other
            offset_indices = find(diff(BehaviorClass == beh_class) == -1);
            if ~isempty(offset_indices)
                event_times.(['Beh' num2str(beh_class) '_Offset']) = NeuralTime(offset_indices + 1);
                fprintf('    Behavior %d (%s) offset: %d events\n', ...
                    beh_class, config.behavior_names{beh_class}, length(offset_indices));
            else
                event_times.(['Beh' num2str(beh_class) '_Offset']) = [];
            end
        end
    else
        % No behavior data - set all to empty
        for beh_class = 1:7
            event_times.(['Beh' num2str(beh_class) '_Onset']) = [];
            event_times.(['Beh' num2str(beh_class) '_Offset']) = [];
        end
        fprintf('    Behavioral onsets/offsets: Skipped (no behavior data)\n');
    end

    % 3b. Behavioral matrix events with bout detection (7 binary features + 1 breathing freq)
    fprintf('  Detecting behavioral matrix events with bout detection...\n');
    if matched && ~isempty(behavioral_matrix_full)
        % Behavioral matrix is at NeuralTime sampling rate
        n_features = min(8, size(behavioral_matrix_full, 2));

        for feat_idx = 1:n_features
            feat_name = config.behavioral_matrix_names{feat_idx};

            % Get the signal for this feature
            if feat_idx == 8
                % Column 8 is breathing frequency - threshold at 5 Hz
                feat_signal = behavioral_matrix_full(:, feat_idx) > config.breathing_freq_threshold;
            else
                % Columns 1-7 are binary signals
                feat_signal = behavioral_matrix_full(:, feat_idx);
            end

            % Find all onsets
            onset_indices = feat_signal == 1;

            if ~isempty(onset_indices)
                onset_times_raw = NeuralTime(onset_indices);

                % Apply bout detection to cluster nearby onsets
                try
                    [bout_starts, bout_ends] = findEventCluster_SuperFast(onset_times_raw, ...
                        config.bout_epsilon, config.bout_minPts);
                    event_times.([feat_name '_Onset']) = bout_starts;
                    fprintf('    %s onset: %d bouts (from %d raw events)\n', ...
                        feat_name, length(bout_starts), length(onset_times_raw));
                    event_times.([feat_name '_Offset']) = bout_ends;
                    fprintf('    %s offset: %d bouts (from %d raw events)\n', ...
                        feat_name, length(bout_ends), length(onset_times_raw));
                catch ME
                    fprintf('    %s onset: ERROR in bout detection - %s\n', feat_name, ME.message);
                    event_times.([feat_name '_Onset']) = [];  % Use raw if bout detection fails
                    event_times.([feat_name '_Offset']) = [];
                end
            else
                event_times.([feat_name '_Onset']) = [];
                event_times.([feat_name '_Offset']) = [];
            end        
        end
    else
        % No behavioral matrix - set all to empty
        for feat_idx = 1:8
            feat_name = config.behavioral_matrix_names{feat_idx};
            event_times.([feat_name '_Onset']) = [];
            event_times.([feat_name '_Offset']) = [];
        end
        fprintf('    Behavioral matrix events: Skipped (no behavioral matrix data)\n');
    end

    % 4. Movement onset/offset
    % SpeedSmoothed is at NeuralTime sampling rate
    fprintf('  Detecting movement events...\n');
    movement_threshold = prctile(SpeedSmoothed, config.movement_percentile);
    is_moving = SpeedSmoothed > movement_threshold;

    movement_onsets = find(diff(is_moving) == 1);
    movement_offsets = find(diff(is_moving) == -1);

    if ~isempty(movement_onsets)
        event_times.MovementOnset = NeuralTime(movement_onsets + 1);
        fprintf('    Movement onsets: %d events\n', length(event_times.MovementOnset));
    else
        event_times.MovementOnset = [];
    end

    if ~isempty(movement_offsets)
        event_times.MovementOffset = NeuralTime(movement_offsets + 1);
        fprintf('    Movement offsets: %d events\n', length(event_times.MovementOffset));
    else
        event_times.MovementOffset = [];
    end

    %% Calculate PSTH for each unit
    % Define all possible event types (ensure consistency across all units)
    all_event_types = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON', 'AversiveOnset', ...
                       'MovementOnset', 'MovementOffset'};

    % Add all 7 behavioral onset/offset types (LSTM predictions)
    for beh_class = 1:7
        all_event_types{end+1} = ['Beh' num2str(beh_class) '_Onset'];
        all_event_types{end+1} = ['Beh' num2str(beh_class) '_Offset'];
    end

    % Add all 8 behavioral matrix feature onset/offset types
    for feat_idx = 1:8
        feat_name = config.behavioral_matrix_names{feat_idx};
        all_event_types{end+1} = [feat_name '_Onset'];
        all_event_types{end+1} = [feat_name '_Offset'];
    end

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

        % Calculate PSTH for each event type (loop over ALL possible event types)
        for e = 1:length(all_event_types)
            event_type = all_event_types{e};

            % Check if this event type exists in event_times
            if isfield(event_times, event_type)
                event_list = event_times.(event_type);
            else
                event_list = [];  % Event type doesn't exist for this session
            end

            if isempty(event_list)
                % No events - store NaN
                unit_data.([event_type '_psth']) = nan(n_bins, 1);
                unit_data.([event_type '_zscore']) = nan(n_bins, 1);
                unit_data.([event_type '_significant']) = false(n_bins, 1);
                unit_data.([event_type '_n_events']) = 0;
                unit_data.([event_type '_baseline_mean']) = nan(1, 1);
                unit_data.([event_type '_baseline_std']) = nan(1, 1);
                continue;
            end

            % Calculate PSTH across all events
            psth_matrix = zeros(length(event_list), n_bins);

            for ev_idx = 1:length(event_list)
                event_time = event_list(ev_idx);

                % Get relative spike times
                rel_spikes = spike_times - event_time;
                rel_spikes = rel_spikes(rel_spikes >= config.psth_window(1) & ...
                                       rel_spikes <= config.psth_window(2));

                % Bin spikes
                trial_psth = histcounts(rel_spikes, time_vector);
                psth_matrix(ev_idx, :) = trial_psth / config.psth_bin_size;  % Convert to Hz
            end

            % Average across trials
            mean_psth = mean(psth_matrix, 1)';

            % Calculate baseline statistics
            baseline_fr = mean_psth(baseline_bins);
            baseline_mean = mean(baseline_fr);
            baseline_std = std(baseline_fr);

            % Z-score normalization
            if baseline_std > 0
                zscore_psth = (mean_psth - baseline_mean) / baseline_std;
            else
                zscore_psth = zeros(size(mean_psth));
            end

            % Statistical significance
            significant_bins = abs(zscore_psth) > config.zscore_threshold;

            % Store results
            unit_data.([event_type '_psth']) = mean_psth;
            unit_data.([event_type '_zscore']) = zscore_psth;
            unit_data.([event_type '_significant']) = significant_bins;
            unit_data.([event_type '_n_events']) = length(event_list);
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
    fprintf('  Processed %d units in %.1f sec\n', length(valid_spikes), elapsed);
end

fprintf('\n✓ Processed %d aversive sessions, %d units total\n', num_aversive_sessions, unit_counter);

%% ========================================================================
%  SECTION 6: PROCESS REWARD SESSIONS
%  ========================================================================

fprintf('\n==== PROCESSING REWARD SESSIONS ====\n');

[allfiles, folderpath, num_reward_sessions] = selectFilesWithAnimalIDFiltering(folderpath, numofsession, '2025*RewardSeeking*.mat');

fprintf('Found %d reward sessions\n', num_reward_sessions);

for spike_sess_idx = 1:num_reward_sessions
    fprintf('\n[%d/%d] Processing: %s\n', spike_sess_idx, num_reward_sessions, allfiles(spike_sess_idx).name);
    tic;

    % Load session data
    Timelimits = 'No';
    try
        [NeuralTime, AdjustedXYZ, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, WP1ON, WP2ON, ...
         AversiveSound, sessionLabels, valid_spikes, Fs, TriggerMid] = ...
            loadAndPrepareSessionData(allfiles(spike_sess_idx), T_sorted, Timelimits);
    catch ME
        fprintf('  ERROR loading session: %s\n', ME.message);
        continue;
    end

    spike_filename = allfiles(spike_sess_idx).name;
    session_type = 'Reward';
    SpeedSmoothed = AdjustedXYZ_speed;  % Rename for clarity

    fprintf('  Loaded %d units\n', length(valid_spikes));

    % Match with behavior prediction session by filename
    matched = false;
    BehaviorClass = [];  % Will be filled if match found

    for beh_sess_idx = 1:length(sessions_reward)
        beh_session = sessions_reward{beh_sess_idx};

        % Simple filename matching
        if contains(spike_filename, extractBefore(beh_session.filename, '.mat'))
            matched = true;

            % Get behavior predictions (1 Hz - 20 frames per prediction)
            prediction_scores = prediction_sessions_reward(beh_sess_idx).prediction_scores;
            n_predictions = size(prediction_scores, 1);

            % Get dominant behavior at each prediction time
            [max_confidence, dominant_beh] = max(prediction_scores, [], 2);

            % Map predictions to NeuralTime
            % Each prediction uses 20 frames at 20 fps = 1 second
            % Prediction i corresponds to frames [(i-1)*20+1 : i*20]
            BehaviorClass = zeros(size(NeuralTime));  % Initialize
            for pred_idx = 1:n_predictions
                frame_start = (pred_idx - 1) * 20 + 1;
                frame_end = min(frame_start + 19, length(TriggerMid));

                if frame_end <= length(TriggerMid)
                    % Get time range for this prediction
                    pred_time_start = TriggerMid(frame_start);
                    pred_time_end = TriggerMid(frame_end);

                    % Find NeuralTime indices within this range
                    neural_indices = find(NeuralTime >= pred_time_start & NeuralTime <= pred_time_end);

                    % Assign behavior class to all neural samples in this prediction window
                    BehaviorClass(neural_indices) = dominant_beh(pred_idx);
                end
            end

            fprintf('  Matched with behavior session: %s\n', beh_session.filename);
            fprintf('  Behavior predictions: %d predictions (1 Hz)\n', n_predictions);

            % Extract behavioral matrix if available
            if isfield(beh_session, 'behavioral_matrix_full')
                behavioral_matrix_full = beh_session.behavioral_matrix_full;
                fprintf('  Behavioral matrix: %d × %d\n', size(behavioral_matrix_full, 1), size(behavioral_matrix_full, 2));
            else
                behavioral_matrix_full = [];
                fprintf('  WARNING: No behavioral_matrix_full found\n');
            end

            break;
        end
    end

    if ~matched
        fprintf('  WARNING: No behavior match for %s - skipping behavioral transitions\n', spike_filename);
        behavioral_matrix_full = [];
    end

    %% Detect Events
    event_times = struct();

    % 1. Reward bouts
    % Note: These signals are at NeuralTime sampling rate, NOT camera frame rate
    fprintf('  Detecting reward bouts...\n');
    reward_event_names = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON'};
    reward_signals = {IR1ON, IR2ON, WP1ON, WP2ON};

    for r = 1:length(reward_event_names)
        event_name = reward_event_names{r};
        event_signal = reward_signals{r};

        event_indices = find(event_signal == 1);

        if ~isempty(event_indices)
            % Use NeuralTime (not TriggerMid) since IR/WP signals are at neural sampling rate
            event_times_raw = NeuralTime(event_indices);

            % Detect bouts
            try
                [bout_starts, ~] = findEventCluster_SuperFast(event_times_raw, ...
                    config.bout_epsilon, config.bout_minPts);
                event_times.(event_name) = bout_starts;
                fprintf('    %s: %d bouts\n', event_name, length(bout_starts));
            catch ME
                fprintf('    %s: ERROR in bout detection - %s\n', event_name, ME.message);
                event_times.(event_name) = [];
            end
        else
            event_times.(event_name) = [];
        end
    end

    % 2. No aversive events in reward sessions
    event_times.AversiveOnset = [];

    % 3. Behavioral onsets and offsets for each of the 7 classes
    % BehaviorClass is at NeuralTime sampling rate (only if behavior data matched)
    fprintf('  Detecting behavioral onsets and offsets for 7 classes...\n');
    if matched && ~isempty(BehaviorClass)
        for beh_class = 1:7
            % Find onset: transition TO this behavior from any other
            onset_indices = find(diff(BehaviorClass == beh_class) == 1);
            if ~isempty(onset_indices)
                event_times.(['Beh' num2str(beh_class) '_Onset']) = NeuralTime(onset_indices + 1);
                fprintf('    Behavior %d (%s) onset: %d events\n', ...
                    beh_class, config.behavior_names{beh_class}, length(onset_indices));
            else
                event_times.(['Beh' num2str(beh_class) '_Onset']) = [];
            end

            % Find offset: transition FROM this behavior to any other
            offset_indices = find(diff(BehaviorClass == beh_class) == -1);
            if ~isempty(offset_indices)
                event_times.(['Beh' num2str(beh_class) '_Offset']) = NeuralTime(offset_indices + 1);
                fprintf('    Behavior %d (%s) offset: %d events\n', ...
                    beh_class, config.behavior_names{beh_class}, length(offset_indices));
            else
                event_times.(['Beh' num2str(beh_class) '_Offset']) = [];
            end
        end
    else
        % No behavior data - set all to empty
        for beh_class = 1:7
            event_times.(['Beh' num2str(beh_class) '_Onset']) = [];
            event_times.(['Beh' num2str(beh_class) '_Offset']) = [];
        end
        fprintf('    Behavioral onsets/offsets: Skipped (no behavior data)\n');
    end

    % 3b. Behavioral matrix events with bout detection (7 binary features + 1 breathing freq)
    fprintf('  Detecting behavioral matrix events with bout detection...\n');
    if matched && ~isempty(behavioral_matrix_full)
        % Behavioral matrix is at NeuralTime sampling rate
        n_features = min(8, size(behavioral_matrix_full, 2));

        for feat_idx = 1:n_features
            feat_name = config.behavioral_matrix_names{feat_idx};

            % Get the signal for this feature
            if feat_idx == 8
                % Column 8 is breathing frequency - threshold at 5 Hz
                feat_signal = behavioral_matrix_full(:, feat_idx) > config.breathing_freq_threshold;
            else
                % Columns 1-7 are binary signals
                feat_signal = behavioral_matrix_full(:, feat_idx);
            end

            % Find all onsets
            onset_indices = feat_signal == 1;

            if ~isempty(onset_indices)
                onset_times_raw = NeuralTime(onset_indices);

                % Apply bout detection to cluster nearby onsets
                try
                    [bout_starts, bout_ends] = findEventCluster_SuperFast(onset_times_raw, ...
                        config.bout_epsilon, config.bout_minPts);
                    event_times.([feat_name '_Onset']) = bout_starts;
                    fprintf('    %s onset: %d bouts (from %d raw events)\n', ...
                        feat_name, length(bout_starts), length(onset_times_raw));
                    event_times.([feat_name '_Offset']) = bout_ends;
                    fprintf('    %s offset: %d bouts (from %d raw events)\n', ...
                        feat_name, length(bout_ends), length(onset_times_raw));
                catch ME
                    fprintf('    %s onset: ERROR in bout detection - %s\n', feat_name, ME.message);
                    event_times.([feat_name '_Onset']) = [];  % Use raw if bout detection fails
                    event_times.([feat_name '_Offset']) = [];
                end
            else
                event_times.([feat_name '_Onset']) = [];
                event_times.([feat_name '_Offset']) = [];
            end        
        end
    else
        % No behavioral matrix - set all to empty
        for feat_idx = 1:8
            feat_name = config.behavioral_matrix_names{feat_idx};
            event_times.([feat_name '_Onset']) = [];
            event_times.([feat_name '_Offset']) = [];
        end
        fprintf('    Behavioral matrix events: Skipped (no behavioral matrix data)\n');
    end

    % 4. Movement onset/offset
    % SpeedSmoothed is at NeuralTime sampling rate
    fprintf('  Detecting movement events...\n');
    movement_threshold = prctile(SpeedSmoothed, config.movement_percentile);
    is_moving = SpeedSmoothed > movement_threshold;

    movement_onsets = find(diff(is_moving) == 1);
    movement_offsets = find(diff(is_moving) == -1);

    if ~isempty(movement_onsets)
        event_times.MovementOnset = NeuralTime(movement_onsets + 1);
        fprintf('    Movement onsets: %d events\n', length(event_times.MovementOnset));
    else
        event_times.MovementOnset = [];
    end

    if ~isempty(movement_offsets)
        event_times.MovementOffset = NeuralTime(movement_offsets + 1);
        fprintf('    Movement offsets: %d events\n', length(event_times.MovementOffset));
    else
        event_times.MovementOffset = [];
    end

    %% Calculate PSTH for each unit
    % Define all possible event types (ensure consistency across all units)
    all_event_types = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON', 'AversiveOnset', ...
                       'MovementOnset', 'MovementOffset'};

    % Add all 7 behavioral onset/offset types (LSTM predictions)
    for beh_class = 1:7
        all_event_types{end+1} = ['Beh' num2str(beh_class) '_Onset'];
        all_event_types{end+1} = ['Beh' num2str(beh_class) '_Offset'];
    end

    % Add all 8 behavioral matrix feature onset/offset types
    for feat_idx = 1:8
        feat_name = config.behavioral_matrix_names{feat_idx};
        all_event_types{end+1} = [feat_name '_Onset'];
        all_event_types{end+1} = [feat_name '_Offset'];
    end

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

        % Calculate PSTH for each event type (loop over ALL possible event types)
        for e = 1:length(all_event_types)
            event_type = all_event_types{e};

            % Check if this event type exists in event_times
            if isfield(event_times, event_type)
                event_list = event_times.(event_type);
            else
                event_list = [];  % Event type doesn't exist for this session
            end

            if isempty(event_list)
                % No events - store NaN
                unit_data.([event_type '_psth']) = nan(n_bins, 1);
                unit_data.([event_type '_zscore']) = nan(n_bins, 1);
                unit_data.([event_type '_significant']) = false(n_bins, 1);
                unit_data.([event_type '_n_events']) = 0;
                unit_data.([event_type '_baseline_mean']) = nan(1, 1);
                unit_data.([event_type '_baseline_std']) = nan(1, 1);
                continue;
            end

            % Calculate PSTH across all events
            psth_matrix = zeros(length(event_list), n_bins);

            for ev_idx = 1:length(event_list)
                event_time = event_list(ev_idx);

                % Get relative spike times
                rel_spikes = spike_times - event_time;
                rel_spikes = rel_spikes(rel_spikes >= config.psth_window(1) & ...
                                       rel_spikes <= config.psth_window(2));

                % Bin spikes
                trial_psth = histcounts(rel_spikes, time_vector);
                psth_matrix(ev_idx, :) = trial_psth / config.psth_bin_size;  % Convert to Hz
            end

            % Average across trials
            mean_psth = mean(psth_matrix, 1)';

            % Calculate baseline statistics
            baseline_fr = mean_psth(baseline_bins);
            baseline_mean = mean(baseline_fr);
            baseline_std = std(baseline_fr);

            % Z-score normalization
            if baseline_std > 0
                zscore_psth = (mean_psth - baseline_mean) / baseline_std;
            else
                zscore_psth = zeros(size(mean_psth));
            end

            % Statistical significance
            significant_bins = abs(zscore_psth) > config.zscore_threshold;

            % Store results
            unit_data.([event_type '_psth']) = mean_psth;
            unit_data.([event_type '_zscore']) = zscore_psth;
            unit_data.([event_type '_significant']) = significant_bins;
            unit_data.([event_type '_n_events']) = length(event_list);
            unit_data.([event_type '_baseline_mean']) = baseline_mean;
            unit_data.([event_type '_baseline_std']) = baseline_std;
        end

        % Add to results
        all_unit_data(unit_counter) = unit_data;
    end

    elapsed = toc;
    fprintf('  Processed %d units in %.1f sec\n', length(valid_spikes), elapsed);
end

fprintf('\n✓ Processed %d reward sessions\n', num_reward_sessions);

%% ========================================================================
%  SECTION 7: SAVE RESULTS
%  ========================================================================

fprintf('\nSaving results...\n');

results = struct();
results.config = config;
results.unit_data = all_unit_data;
results.n_units_total = unit_counter;
results.n_aversive_sessions = num_aversive_sessions;
results.n_reward_sessions = num_reward_sessions;
results.time_centers = time_centers;

save_filename = 'PSTH_Survey_Results.mat';
save(save_filename, 'results', '-v7.3');

fprintf('✓ Results saved to: %s\n', save_filename);
fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Total units analyzed: %d\n', unit_counter);
fprintf('Aversive sessions: %d\n', num_aversive_sessions);
fprintf('Reward sessions: %d\n', num_reward_sessions);
fprintf('\nNext step: Run Visualize_PSTH_Survey.m\n');
