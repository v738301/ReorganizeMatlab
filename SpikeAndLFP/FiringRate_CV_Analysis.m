%% ========================================================================
%  FIRING RATE & CV ANALYSIS: Period × Behavior × SessionType
%  5-second sliding window approach for robust FR/CV estimation
%  Then assign to prediction bins and aggregate by Period and Behavior
%  ========================================================================
%
%  Analysis: FR/CV ~ Period × Behavior × SessionType
%  SessionType: Aversive vs Reward
%  Periods: P1-P4 (matched across both session types)
%  Behaviors: 7 classes from LSTM predictions
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== FIRING RATE & CV ANALYSIS: AVERSIVE vs REWARD ===\n');
fprintf('Period × Behavior × SessionType\n\n');

config = struct();
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
config.n_behaviors = 7;
config.confidence_threshold = 0.3;  % Minimum confidence for behavior assignment
config.min_spikes_for_CV = 10;      % Minimum spikes to calculate CV
config.frames_per_prediction = 20;  % Each prediction uses 20 frames
config.camera_fps = 20;             % Camera frame rate: 20 Hz (20 frames = 1 sec)
config.window_ize = 5;             % 5-second windows for FR/CV calculation
config.window_slide = 1;            % Slide by 1 second

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
    coupling_data_aversive = load('27-Oct-2025_RewardAversive_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_aversive = coupling_data_aversive.all_session_metrics;
    pred_data_aversive = load('lstm_prediction_results_aversive_27-Oct-2025');
    prediction_sessions_aversive = pred_data_aversive.final_results.session_predictions;
    fprintf('✓ Loaded aversive behavior predictions: %d sessions\n', length(sessions_aversive));
catch ME
    fprintf('❌ Failed to load aversive behavior data: %s\n', ME.message);
    return;
end

% Load reward behavior predictions
try
    coupling_data_reward = load('27-Oct-2025_RewardSeeking_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_reward = coupling_data_reward.all_session_metrics;
    pred_data_reward = load('lstm_prediction_results_reward_27-Oct-2025');
    prediction_sessions_reward = pred_data_reward.final_results.session_predictions;
    fprintf('✓ Loaded reward behavior predictions: %d sessions\n\n', length(sessions_reward));
catch ME
    fprintf('❌ Failed to load reward behavior data: %s\n', ME.message);
    return;
end

%% ========================================================================
%  SECTION 4: PROCESS AVERSIVE SESSIONS
%  ========================================================================

fprintf('\n==== PROCESSING AVERSIVE SESSIONS ====\n');

% Initialize storage for prediction-level data
aversive_predictions = struct();
aversive_predictions.session_id = [];
aversive_predictions.unit_id = [];
aversive_predictions.prediction_idx = [];
aversive_predictions.period = [];
aversive_predictions.behavior = [];
aversive_predictions.FR = [];
aversive_predictions.CV = [];
aversive_predictions.n_windows_averaged = [];
aversive_predictions.session_name = {};

n_valid_aversive = 0;

% Load raw spike data files
numofsession = 2;
folderpath = "/Volumes/ExpansionBackup/Data/Struct_spike";
[allfiles, folderpath, num_aversive_sessions] = selectFilesWithAnimalIDFiltering(folderpath, numofsession, '2025*RewardAversive*.mat');

for spike_sess_idx = 1:num_aversive_sessions
    fprintf('\n[%d/%d] Processing: %s\n', spike_sess_idx, num_aversive_sessions, allfiles(spike_sess_idx).name);
    tic;

    % Load raw spike data
    Timelimits = 'No';
    [NeuralTime, ~, ~, ~, ~, ~, ~, ~, AversiveSound, ~, valid_spikes, Fs, TriggerMid] = ...
        loadAndPrepareSessionData(allfiles(spike_sess_idx), T_sorted, Timelimits);

    % Get all aversive sound timepoints
    aversive_onsets = find(diff(AversiveSound) == 1);
    all_aversive_time = NeuralTime(aversive_onsets);

    if length(all_aversive_time) < 6
        fprintf('  Skipping: insufficient aversive events (%d)\n', length(all_aversive_time));
        continue;
    end

    % Match with behavior prediction session by filename
    spike_filename = allfiles(spike_sess_idx).name;
    matched = false;

    for beh_sess_idx = 1:length(sessions_aversive)
        beh_session = sessions_aversive{beh_sess_idx};

        % Simple filename matching
        if contains(spike_filename, extractBefore(beh_session.filename, '.mat'))
            matched = true;
            n_valid_aversive = n_valid_aversive + 1;

            % Get behavior predictions
            prediction_scores = prediction_sessions_aversive(beh_sess_idx).prediction_scores;
            n_predictions = size(prediction_scores, 1);

            % Get dominant behavior at each prediction time
            [max_confidence, dominant_beh] = max(prediction_scores, [], 2);

            % Define period boundaries (P1-P4)
            period_boundaries = [TriggerMid(1), ...
                                 all_aversive_time(1:3)' + TriggerMid(1), ...
                                 all_aversive_time(4) + TriggerMid(1)];

            fprintf('  Processing %d units with 5-sec sliding windows...\n', length(valid_spikes));

            % Process each unit
            n_units = length(valid_spikes);

            for unit_idx = 1:n_units
                spike_times = valid_spikes{unit_idx};

                if isempty(spike_times)
                    continue;
                end

                % STEP 1: Create 5-second sliding windows and calculate FR/CV
                session_start = TriggerMid(1);
                session_end = TriggerMid(end);
                session_duration = session_end - session_start;

                % Generate window start times (slide by 1 sec)
                window_starts = session_start:config.window_slide:(session_end - config.window_size);
                n_windows = length(window_starts);

                % Pre-allocate window results
                window_FR = nan(n_windows, 1);
                window_CV = nan(n_windows, 1);

                for w = 1:n_windows
                    win_start = window_starts(w);
                    win_end = win_start + config.window_size;

                    % Find spikes in this 5-sec window
                    spikes_in_win = spike_times(spike_times >= win_start & spike_times < win_end);
                    n_spikes = length(spikes_in_win);

                    % Calculate FR
                    window_FR(w) = n_spikes / config.window_size;

                    % Calculate CV
                    if n_spikes >= config.min_spikes_for_CV
                        ISI = diff(spikes_in_win);
                        if ~isempty(ISI) && mean(ISI) > 0
                            window_CV(w) = std(ISI) / mean(ISI);
                        end
                    end
                end

                % STEP 2: For each prediction, find overlapping windows and average
                for pred_idx = 1:n_predictions
                    % Get prediction time range (20 frames = 1 sec)
                    frame_start = (pred_idx - 1) * config.frames_per_prediction + 1;
                    frame_end = min(frame_start + config.frames_per_prediction - 1, length(TriggerMid));

                    if frame_end > length(TriggerMid)
                        continue;
                    end

                    pred_time_start = TriggerMid(frame_start);
                    pred_time_end = TriggerMid(frame_end);

                    % Get behavior for this prediction
                    pred_behavior = dominant_beh(pred_idx);
                    pred_confidence = max_confidence(pred_idx);

                    % Skip if confidence too low
                    if pred_confidence < config.confidence_threshold
                        continue;
                    end

                    % Determine which period this prediction belongs to
                    pred_period = 0;
                    for p = 1:4
                        if pred_time_start >= period_boundaries(p) && pred_time_start < period_boundaries(p+1)
                            pred_period = p;
                            break;
                        end
                    end

                    if pred_period == 0
                        continue;
                    end

                    % Find overlapping 5-sec windows
                    % A window overlaps if: window_start < pred_time_end AND window_end > pred_time_start
                    window_ends = window_starts + config.window_size;
                    overlap_mask = (window_starts < pred_time_end) & (window_ends > pred_time_start);
                    overlapping_indices = find(overlap_mask);

                    if isempty(overlapping_indices)
                        continue;
                    end

                    % Average FR and CV from overlapping windows
                    avg_FR = mean(window_FR(overlapping_indices), 'omitnan');
                    avg_CV = mean(window_CV(overlapping_indices), 'omitnan');

                    % Store prediction-level data
                    aversive_predictions.session_id(end+1) = n_valid_aversive;
                    aversive_predictions.unit_id(end+1) = (n_valid_aversive - 1) * 1000 + unit_idx;
                    aversive_predictions.prediction_idx(end+1) = pred_idx;
                    aversive_predictions.period(end+1) = pred_period;
                    aversive_predictions.behavior(end+1) = pred_behavior;
                    aversive_predictions.FR(end+1) = avg_FR;
                    aversive_predictions.CV(end+1) = avg_CV;
                    aversive_predictions.n_windows_averaged(end+1) = length(overlapping_indices);
                    aversive_predictions.session_name{end+1} = spike_filename;
                end
            end

            fprintf('  Session %d: %s - %d units, %d predictions\n', ...
                    n_valid_aversive, spike_filename, n_units, n_predictions);
            break;
        end
    end

    if ~matched
        fprintf('  Warning: No behavior match for %s\n', spike_filename);
    end

    elapsed = toc;
    fprintf('  Elapsed time: %.1f sec\n', elapsed);
end

fprintf('\n✓ Processed %d aversive sessions\n', n_valid_aversive);
fprintf('  Total prediction-level data points: %d\n\n', length(aversive_predictions.session_id));

%% ========================================================================
%  SECTION 5: PROCESS REWARD SESSIONS
%  ========================================================================

fprintf('\n==== PROCESSING REWARD SESSIONS ====\n');

% Initialize storage for prediction-level data
reward_predictions = struct();
reward_predictions.session_id = [];
reward_predictions.unit_id = [];
reward_predictions.prediction_idx = [];
reward_predictions.period = [];
reward_predictions.behavior = [];
reward_predictions.FR = [];
reward_predictions.CV = [];
reward_predictions.n_windows_averaged = [];
reward_predictions.session_name = {};

n_valid_reward = 0;

% Load raw spike data files
[allfiles, folderpath, num_reward_sessions] = selectFilesWithAnimalIDFiltering(folderpath, numofsession, '2025*RewardSeeking*.mat');

for spike_sess_idx = 1:num_reward_sessions
    fprintf('\n[%d/%d] Processing: %s\n', spike_sess_idx, num_reward_sessions, allfiles(spike_sess_idx).name);
    tic;

    % Load raw spike data
    Timelimits = 'No';
    [NeuralTime, ~, ~, ~, ~, ~, ~, ~, ~, ~, valid_spikes, Fs, TriggerMid] = ...
        loadAndPrepareSessionData(allfiles(spike_sess_idx), T_sorted, Timelimits);

    % Match with behavior prediction session by filename
    spike_filename = allfiles(spike_sess_idx).name;
    matched = false;

    for beh_sess_idx = 1:length(sessions_reward)
        beh_session = sessions_reward{beh_sess_idx};

        % Simple filename matching
        if contains(spike_filename, extractBefore(beh_session.filename, '.mat'))
            matched = true;
            n_valid_reward = n_valid_reward + 1;

            % Get behavior predictions
            prediction_scores = prediction_sessions_reward(beh_sess_idx).prediction_scores;
            n_predictions = size(prediction_scores, 1);

            % Get dominant behavior at each prediction time
            [max_confidence, dominant_beh] = max(prediction_scores, [], 2);

            % Define period boundaries based on time (0-8, 8-16, 16-24, 24-30 min)
            time_boundaries = [0, 8*60, 16*60, 24*60, 30*60];
            period_boundaries = [TriggerMid(1), ...
                                 time_boundaries(2:end) + TriggerMid(1)];

            fprintf('  Processing %d units with 5-sec sliding windows...\n', length(valid_spikes));

            % Process each unit
            n_units = length(valid_spikes);

            for unit_idx = 1:n_units
                spike_times = valid_spikes{unit_idx};

                if isempty(spike_times)
                    continue;
                end

                % STEP 1: Create 5-second sliding windows and calculate FR/CV
                session_start = TriggerMid(1);
                session_end = TriggerMid(end);

                % Generate window start times (slide by 1 sec)
                window_starts = session_start:config.window_slide:(session_end - config.window_size);
                n_windows = length(window_starts);

                % Pre-allocate window results
                window_FR = nan(n_windows, 1);
                window_CV = nan(n_windows, 1);

                for w = 1:n_windows
                    win_start = window_starts(w);
                    win_end = win_start + config.window_size;

                    % Find spikes in this 5-sec window
                    spikes_in_win = spike_times(spike_times >= win_start & spike_times < win_end);
                    n_spikes = length(spikes_in_win);

                    % Calculate FR
                    window_FR(w) = n_spikes / config.window_size;

                    % Calculate CV
                    if n_spikes >= config.min_spikes_for_CV
                        ISI = diff(spikes_in_win);
                        if ~isempty(ISI) && mean(ISI) > 0
                            window_CV(w) = std(ISI) / mean(ISI);
                        end
                    end
                end

                % STEP 2: For each prediction, find overlapping windows and average
                for pred_idx = 1:n_predictions
                    % Get prediction time range (20 frames = 1 sec)
                    frame_start = (pred_idx - 1) * config.frames_per_prediction + 1;
                    frame_end = min(frame_start + config.frames_per_prediction - 1, length(TriggerMid));

                    if frame_end > length(TriggerMid)
                        continue;
                    end

                    pred_time_start = TriggerMid(frame_start);
                    pred_time_end = TriggerMid(frame_end);

                    % Get behavior for this prediction
                    pred_behavior = dominant_beh(pred_idx);
                    pred_confidence = max_confidence(pred_idx);

                    % Skip if confidence too low
                    if pred_confidence < config.confidence_threshold
                        continue;
                    end

                    % Determine which period this prediction belongs to
                    pred_period = 0;
                    for p = 1:4
                        if pred_time_start >= period_boundaries(p) && pred_time_start < period_boundaries(p+1)
                            pred_period = p;
                            break;
                        end
                    end

                    if pred_period == 0
                        continue;
                    end

                    % Find overlapping 5-sec windows
                    window_ends = window_starts + config.window_size;
                    overlap_mask = (window_starts < pred_time_end) & (window_ends > pred_time_start);
                    overlapping_indices = find(overlap_mask);

                    if isempty(overlapping_indices)
                        continue;
                    end

                    % Average FR and CV from overlapping windows
                    avg_FR = mean(window_FR(overlapping_indices), 'omitnan');
                    avg_CV = mean(window_CV(overlapping_indices), 'omitnan');

                    % Store prediction-level data
                    reward_predictions.session_id(end+1) = n_valid_reward;
                    reward_predictions.unit_id(end+1) = (n_valid_reward + 10000) * 1000 + unit_idx;
                    reward_predictions.prediction_idx(end+1) = pred_idx;
                    reward_predictions.period(end+1) = pred_period;
                    reward_predictions.behavior(end+1) = pred_behavior;
                    reward_predictions.FR(end+1) = avg_FR;
                    reward_predictions.CV(end+1) = avg_CV;
                    reward_predictions.n_windows_averaged(end+1) = length(overlapping_indices);
                    reward_predictions.session_name{end+1} = spike_filename;
                end
            end

            fprintf('  Session %d: %s - %d units, %d predictions\n', ...
                    n_valid_reward, spike_filename, n_units, n_predictions);
            break;
        end
    end

    if ~matched
        fprintf('  Warning: No behavior match for %s\n', spike_filename);
    end

    elapsed = toc;
    fprintf('  Elapsed time: %.1f sec\n', elapsed);
end

fprintf('\n✓ Processed %d reward sessions\n', n_valid_reward);
fprintf('  Total prediction-level data points: %d\n\n', length(reward_predictions.session_id));

%% ========================================================================
%  SECTION 6: COMBINE AND AGGREGATE DATA
%  ========================================================================

fprintf('Combining and aggregating data...\n');

% Add SessionType
aversive_predictions.session_type = repmat({'Aversive'}, length(aversive_predictions.session_id), 1);
reward_predictions.session_type = repmat({'Reward'}, length(reward_predictions.session_id), 1);

% Combine prediction-level data
combined_predictions = struct();
combined_predictions.session_id = [aversive_predictions.session_id(:); reward_predictions.session_id(:)];
combined_predictions.unit_id = [aversive_predictions.unit_id(:); reward_predictions.unit_id(:)];
combined_predictions.prediction_idx = [aversive_predictions.prediction_idx(:); reward_predictions.prediction_idx(:)];
combined_predictions.period = [aversive_predictions.period(:); reward_predictions.period(:)];
combined_predictions.behavior = [aversive_predictions.behavior(:); reward_predictions.behavior(:)];
combined_predictions.FR = [aversive_predictions.FR(:); reward_predictions.FR(:)];
combined_predictions.CV = [aversive_predictions.CV(:); reward_predictions.CV(:)];
combined_predictions.n_windows_averaged = [aversive_predictions.n_windows_averaged(:); reward_predictions.n_windows_averaged(:)];
combined_predictions.session_type = [aversive_predictions.session_type; reward_predictions.session_type];

% Convert to table
tbl_predictions = table(combined_predictions.session_id, ...
                        combined_predictions.unit_id, ...
                        combined_predictions.prediction_idx, ...
                        combined_predictions.period, ...
                        combined_predictions.behavior, ...
                        combined_predictions.FR, ...
                        combined_predictions.CV, ...
                        combined_predictions.n_windows_averaged, ...
                        combined_predictions.session_type, ...
                        'VariableNames', {'Session', 'Unit', 'Prediction', 'Period', 'Behavior', ...
                                         'FR', 'CV', 'N_windows', 'SessionType'});

% Convert to categorical
tbl_predictions.Session = categorical(tbl_predictions.Session);
tbl_predictions.Unit = categorical(tbl_predictions.Unit);
tbl_predictions.Period = categorical(tbl_predictions.Period);
tbl_predictions.Behavior = categorical(tbl_predictions.Behavior, 1:7, config.behavior_names);
tbl_predictions.SessionType = categorical(tbl_predictions.SessionType);

fprintf('✓ Combined prediction-level dataset created\n');
fprintf('  Total prediction data points: %d\n', height(tbl_predictions));
fprintf('  Sessions: %d aversive, %d reward\n', n_valid_aversive, n_valid_reward);
fprintf('  Mean windows averaged per prediction: %.1f\n\n', mean(tbl_predictions.N_windows, 'omitnan'));

%% Aggregate by Unit × Period × Behavior
fprintf('Aggregating by Unit × Period × Behavior...\n');

% Create aggregated table
unique_combinations = unique(tbl_predictions(:, {'Unit', 'Period', 'Behavior', 'SessionType'}), 'rows');
n_combinations = height(unique_combinations);

% Initialize aggregated data
agg_FR = zeros(n_combinations, 1);
agg_CV = zeros(n_combinations, 1);
agg_n_predictions = zeros(n_combinations, 1);

for i = 1:n_combinations
    mask = (tbl_predictions.Unit == unique_combinations.Unit(i)) & ...
           (tbl_predictions.Period == unique_combinations.Period(i)) & ...
           (tbl_predictions.Behavior == unique_combinations.Behavior(i)) & ...
           (tbl_predictions.SessionType == unique_combinations.SessionType(i));

    % Average FR and CV across predictions
    agg_FR(i) = mean(tbl_predictions.FR(mask), 'omitnan');
    agg_CV(i) = mean(tbl_predictions.CV(mask), 'omitnan');
    agg_n_predictions(i) = sum(mask);
end

% Create aggregated table
tbl_aggregated = [unique_combinations, ...
                  table(agg_FR, agg_CV, agg_n_predictions, ...
                        'VariableNames', {'FR', 'CV', 'N_predictions'})];

fprintf('✓ Aggregated dataset created\n');
fprintf('  Aggregated data points: %d\n', height(tbl_aggregated));
fprintf('  Mean predictions per combination: %.1f\n\n', mean(agg_n_predictions));

%% ========================================================================
%  SECTION 7: SAVE RESULTS
%  ========================================================================

fprintf('Saving results...\n');

results = struct();
results.config = config;
results.tbl_predictions = tbl_predictions;       % Prediction-level data
results.tbl_aggregated = tbl_aggregated;         % Aggregated by Unit × Period × Behavior
results.n_aversive_sessions = n_valid_aversive;
results.n_reward_sessions = n_valid_reward;

timestamp = datestr(now, 'dd-mmm-yyyy');
save_filename = sprintf('FR_CV_analysis_results_%s.mat', timestamp);
save(save_filename, 'results', '-v7.3');

fprintf('✓ Results saved to: %s\n', save_filename);
fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Next step: Run Visualize_FiringRate_CV.m\n');
