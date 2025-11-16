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
config.window_size = 5;             % 5-second windows for FR/CV calculation
config.window_slide = 1;            % Slide by 1 second

% New metric parameters
config.acf_max_lag = 0.1;           % Auto-correlation up to 100ms
config.burst_isi_threshold = 0.01;  % 10ms threshold for burst detection
config.refrac_threshold = 0.002;    % 2ms for refractory violations
config.count_bin_sizes = [0.001, 0.025, 0.050];  % 1ms, 25ms, 50ms for spike count ACF

% Data paths
config.prediction_folder = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/BehaviorPrediction';
config.spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';
config.numofsession = 999;  % Max sessions per animal

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/NewScripts/');

%% ========================================================================
%  SECTION 2: LOAD SORTING PARAMETERS
%  ========================================================================

fprintf('Loading sorting parameters...\n');
[T_sorted] = loadSortingParameters();
fprintf('✓ Sorting parameters loaded\n\n');

%% ========================================================================
%  SECTION 3: SELECT AND LOAD SPIKE FILES
%  ========================================================================


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

% New metrics
aversive_predictions.ISI_FanoFactor = [];
aversive_predictions.ISI_ACF_peak = [];
aversive_predictions.ISI_ACF_lag = [];
aversive_predictions.ISI_ACF_decay = [];
aversive_predictions.Count_ACF_1ms_peak = [];
aversive_predictions.Count_ACF_25ms_peak = [];
aversive_predictions.Count_ACF_50ms_peak = [];
aversive_predictions.LV = [];
aversive_predictions.CV2 = [];
aversive_predictions.LVR = [];
aversive_predictions.BurstIndex = [];
aversive_predictions.BurstRate = [];
aversive_predictions.MeanBurstLength = [];
aversive_predictions.ISI_Skewness = [];
aversive_predictions.ISI_Kurtosis = [];
aversive_predictions.ISI_Mode = [];
aversive_predictions.CountFanoFactor_1ms = [];
aversive_predictions.CountFanoFactor_25ms = [];
aversive_predictions.CountFanoFactor_50ms = [];
aversive_predictions.RefracViolations = [];

n_valid_aversive = 0;

for spike_sess_idx = 1:num_aversive_sessions
    fprintf('\n[%d/%d] Processing: %s\n', spike_sess_idx, num_aversive_sessions, allfiles_aversive(spike_sess_idx).name);
    tic;

    % Load raw spike data
    Timelimits = 'No';
    [NeuralTime, ~, ~, ~, ~, ~, ~, ~, AversiveSound, ~, valid_spikes, Fs, TriggerMid] = ...
        loadAndPrepareSessionData(allfiles_aversive(spike_sess_idx), T_sorted, Timelimits);

    % Get all aversive sound timepoints
    aversive_onsets = find(diff(AversiveSound) == 1);
    all_aversive_time = NeuralTime(aversive_onsets);

    if length(all_aversive_time) < 6
        fprintf('  Skipping: insufficient aversive events (%d)\n', length(all_aversive_time));
        continue;
    end

    % Get behavior predictions for this session
    spike_filename = allfiles_aversive(spike_sess_idx).name;

    % Check if predictions are available for this session
    if spike_sess_idx <= length(prediction_sessions_aversive) && ...
       ~isempty(prediction_sessions_aversive(spike_sess_idx).prediction_scores)

        n_valid_aversive = n_valid_aversive + 1;

        % Get behavior predictions
        prediction_scores = prediction_sessions_aversive(spike_sess_idx).prediction_scores;
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

                % Pre-allocate window results - store all metrics
                window_metrics = cell(n_windows, 1);

                for w = 1:n_windows
                    win_start = window_starts(w);
                    win_end = win_start + config.window_size;

                    % Calculate ALL metrics for this window
                    window_metrics{w} = calculateAllMetrics(spike_times, win_start, win_end, config);
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

                    % Average ALL metrics from overlapping windows
                    avg_metrics = averageWindowMetrics(window_metrics(overlapping_indices));

                    % Store prediction-level data
                    aversive_predictions.session_id(end+1) = n_valid_aversive;
                    aversive_predictions.unit_id(end+1) = (n_valid_aversive - 1) * 1000 + unit_idx;
                    aversive_predictions.prediction_idx(end+1) = pred_idx;
                    aversive_predictions.period(end+1) = pred_period;
                    aversive_predictions.behavior(end+1) = pred_behavior;
                    aversive_predictions.FR(end+1) = avg_metrics.FR;
                    aversive_predictions.CV(end+1) = avg_metrics.CV;
                    aversive_predictions.n_windows_averaged(end+1) = length(overlapping_indices);
                    aversive_predictions.session_name{end+1} = spike_filename;

                    % Store new metrics
                    aversive_predictions.ISI_FanoFactor(end+1) = avg_metrics.ISI_FanoFactor;
                    aversive_predictions.ISI_ACF_peak(end+1) = avg_metrics.ISI_ACF_peak;
                    aversive_predictions.ISI_ACF_lag(end+1) = avg_metrics.ISI_ACF_lag;
                    aversive_predictions.ISI_ACF_decay(end+1) = avg_metrics.ISI_ACF_decay;
                    aversive_predictions.Count_ACF_1ms_peak(end+1) = avg_metrics.Count_ACF_1ms_peak;
                    aversive_predictions.Count_ACF_25ms_peak(end+1) = avg_metrics.Count_ACF_25ms_peak;
                    aversive_predictions.Count_ACF_50ms_peak(end+1) = avg_metrics.Count_ACF_50ms_peak;
                    aversive_predictions.LV(end+1) = avg_metrics.LV;
                    aversive_predictions.CV2(end+1) = avg_metrics.CV2;
                    aversive_predictions.LVR(end+1) = avg_metrics.LVR;
                    aversive_predictions.BurstIndex(end+1) = avg_metrics.BurstIndex;
                    aversive_predictions.BurstRate(end+1) = avg_metrics.BurstRate;
                    aversive_predictions.MeanBurstLength(end+1) = avg_metrics.MeanBurstLength;
                    aversive_predictions.ISI_Skewness(end+1) = avg_metrics.ISI_Skewness;
                    aversive_predictions.ISI_Kurtosis(end+1) = avg_metrics.ISI_Kurtosis;
                    aversive_predictions.ISI_Mode(end+1) = avg_metrics.ISI_Mode;
                    aversive_predictions.CountFanoFactor_1ms(end+1) = avg_metrics.CountFanoFactor_1ms;
                    aversive_predictions.CountFanoFactor_25ms(end+1) = avg_metrics.CountFanoFactor_25ms;
                    aversive_predictions.CountFanoFactor_50ms(end+1) = avg_metrics.CountFanoFactor_50ms;
                    aversive_predictions.RefracViolations(end+1) = avg_metrics.RefracViolations;
                end
            end

        fprintf('  Session %d: %s - %d units, %d predictions\n', ...
                n_valid_aversive, spike_filename, n_units, n_predictions);
    else
        fprintf('  Skipping: No behavior predictions for %s\n', spike_filename);
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

% New metrics
reward_predictions.ISI_FanoFactor = [];
reward_predictions.ISI_ACF_peak = [];
reward_predictions.ISI_ACF_lag = [];
reward_predictions.ISI_ACF_decay = [];
reward_predictions.Count_ACF_1ms_peak = [];
reward_predictions.Count_ACF_25ms_peak = [];
reward_predictions.Count_ACF_50ms_peak = [];
reward_predictions.LV = [];
reward_predictions.CV2 = [];
reward_predictions.LVR = [];
reward_predictions.BurstIndex = [];
reward_predictions.BurstRate = [];
reward_predictions.MeanBurstLength = [];
reward_predictions.ISI_Skewness = [];
reward_predictions.ISI_Kurtosis = [];
reward_predictions.ISI_Mode = [];
reward_predictions.CountFanoFactor_1ms = [];
reward_predictions.CountFanoFactor_25ms = [];
reward_predictions.CountFanoFactor_50ms = [];
reward_predictions.RefracViolations = [];

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

                % Pre-allocate window results - store all metrics
                window_metrics = cell(n_windows, 1);

                for w = 1:n_windows
                    win_start = window_starts(w);
                    win_end = win_start + config.window_size;

                    % Calculate ALL metrics for this window
                    window_metrics{w} = calculateAllMetrics(spike_times, win_start, win_end, config);
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

% Combine new metrics
combined_predictions.ISI_FanoFactor = [aversive_predictions.ISI_FanoFactor(:); reward_predictions.ISI_FanoFactor(:)];
combined_predictions.ISI_ACF_peak = [aversive_predictions.ISI_ACF_peak(:); reward_predictions.ISI_ACF_peak(:)];
combined_predictions.ISI_ACF_lag = [aversive_predictions.ISI_ACF_lag(:); reward_predictions.ISI_ACF_lag(:)];
combined_predictions.ISI_ACF_decay = [aversive_predictions.ISI_ACF_decay(:); reward_predictions.ISI_ACF_decay(:)];
combined_predictions.Count_ACF_1ms_peak = [aversive_predictions.Count_ACF_1ms_peak(:); reward_predictions.Count_ACF_1ms_peak(:)];
combined_predictions.Count_ACF_25ms_peak = [aversive_predictions.Count_ACF_25ms_peak(:); reward_predictions.Count_ACF_25ms_peak(:)];
combined_predictions.Count_ACF_50ms_peak = [aversive_predictions.Count_ACF_50ms_peak(:); reward_predictions.Count_ACF_50ms_peak(:)];
combined_predictions.LV = [aversive_predictions.LV(:); reward_predictions.LV(:)];
combined_predictions.CV2 = [aversive_predictions.CV2(:); reward_predictions.CV2(:)];
combined_predictions.LVR = [aversive_predictions.LVR(:); reward_predictions.LVR(:)];
combined_predictions.BurstIndex = [aversive_predictions.BurstIndex(:); reward_predictions.BurstIndex(:)];
combined_predictions.BurstRate = [aversive_predictions.BurstRate(:); reward_predictions.BurstRate(:)];
combined_predictions.MeanBurstLength = [aversive_predictions.MeanBurstLength(:); reward_predictions.MeanBurstLength(:)];
combined_predictions.ISI_Skewness = [aversive_predictions.ISI_Skewness(:); reward_predictions.ISI_Skewness(:)];
combined_predictions.ISI_Kurtosis = [aversive_predictions.ISI_Kurtosis(:); reward_predictions.ISI_Kurtosis(:)];
combined_predictions.ISI_Mode = [aversive_predictions.ISI_Mode(:); reward_predictions.ISI_Mode(:)];
combined_predictions.CountFanoFactor_1ms = [aversive_predictions.CountFanoFactor_1ms(:); reward_predictions.CountFanoFactor_1ms(:)];
combined_predictions.CountFanoFactor_25ms = [aversive_predictions.CountFanoFactor_25ms(:); reward_predictions.CountFanoFactor_25ms(:)];
combined_predictions.CountFanoFactor_50ms = [aversive_predictions.CountFanoFactor_50ms(:); reward_predictions.CountFanoFactor_50ms(:)];
combined_predictions.RefracViolations = [aversive_predictions.RefracViolations(:); reward_predictions.RefracViolations(:)];

% Convert to table - use struct2table for efficiency
tbl_predictions = struct2table(combined_predictions);

% Rename columns for consistency
tbl_predictions.Properties.VariableNames{'session_id'} = 'Session';
tbl_predictions.Properties.VariableNames{'unit_id'} = 'Unit';
tbl_predictions.Properties.VariableNames{'prediction_idx'} = 'Prediction';
tbl_predictions.Properties.VariableNames{'period'} = 'Period';
tbl_predictions.Properties.VariableNames{'behavior'} = 'Behavior';
tbl_predictions.Properties.VariableNames{'n_windows_averaged'} = 'N_windows';
tbl_predictions.Properties.VariableNames{'session_type'} = 'SessionType';

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


%% ========================================================================
%  HELPER FUNCTIONS FOR METRIC CALCULATION
%  ========================================================================

function metrics = calculateAllMetrics(spike_times, win_start, win_end, config)
% Calculate all metrics for a single window
% Returns struct with all computed metrics

    metrics = struct();

    % Get spikes in window
    spikes_in_win = spike_times(spike_times >= win_start & spike_times < win_end);
    n_spikes = length(spikes_in_win);

    % Basic metrics
    metrics.FR = n_spikes / (win_end - win_start);
    metrics.n_spikes = n_spikes;

    % Initialize all other metrics as NaN
    metrics.CV = NaN;
    metrics.ISI_FanoFactor = NaN;
    metrics.ISI_ACF_peak = NaN;
    metrics.ISI_ACF_lag = NaN;
    metrics.ISI_ACF_decay = NaN;
    metrics.Count_ACF_1ms_peak = NaN;
    metrics.Count_ACF_25ms_peak = NaN;
    metrics.Count_ACF_50ms_peak = NaN;
    metrics.LV = NaN;
    metrics.CV2 = NaN;
    metrics.LVR = NaN;
    metrics.BurstIndex = NaN;
    metrics.BurstRate = NaN;
    metrics.MeanBurstLength = NaN;
    metrics.ISI_Skewness = NaN;
    metrics.ISI_Kurtosis = NaN;
    metrics.ISI_Mode = NaN;
    metrics.CountFanoFactor_1ms = NaN;
    metrics.CountFanoFactor_25ms = NaN;
    metrics.CountFanoFactor_50ms = NaN;
    metrics.RefracViolations = NaN;

    % Need enough spikes for ISI-based metrics
    if n_spikes < config.min_spikes_for_CV
        return;
    end

    % Calculate ISI
    ISI = diff(spikes_in_win);

    if isempty(ISI)
        return;
    end

    % === ISI-based metrics ===
    isi_mean = mean(ISI);
    isi_std = std(ISI);

    if isi_mean > 0
        % 1. CV
        metrics.CV = isi_std / isi_mean;

        % 2. ISI Fano Factor
        metrics.ISI_FanoFactor = var(ISI) / isi_mean;

        % 3. ISI Auto-correlation
        [acf_vals, acf_lags] = calculateISI_ACF(ISI, config.acf_max_lag);
        if ~isempty(acf_vals)
            [metrics.ISI_ACF_peak, peak_idx] = max(acf_vals(2:end));  % Skip lag 0
            metrics.ISI_ACF_lag = acf_lags(peak_idx + 1);
            metrics.ISI_ACF_decay = calculateACFDecay(acf_vals, acf_lags);
        end

        % 4. Local Variation (LV)
        metrics.LV = calculateLV(ISI);

        % 5. CV2
        metrics.CV2 = calculateCV2(ISI);

        % 6. LVR (Revised Local Variation)
        metrics.LVR = calculateLVR(ISI, config.refrac_threshold);

        % 7-9. Burst metrics
        [metrics.BurstIndex, metrics.BurstRate, metrics.MeanBurstLength] = ...
            calculateBurstMetrics(ISI, config.burst_isi_threshold, win_end - win_start);

        % 10-12. ISI distribution shape
        if length(ISI) >= 3
            metrics.ISI_Skewness = skewness(ISI);
            metrics.ISI_Kurtosis = kurtosis(ISI);
            metrics.ISI_Mode = mode(round(ISI, 4));  % Round to avoid floating point issues
        end

        % 14. Refractory violations
        metrics.RefracViolations = 100 * sum(ISI < config.refrac_threshold) / length(ISI);
    end

    % === Spike count based metrics ===
    % 13. Spike count Fano Factor and ACF for different bin sizes
    for bin_idx = 1:length(config.count_bin_sizes)
        bin_size = config.count_bin_sizes(bin_idx);
        bin_label = sprintf('%.0fms', bin_size * 1000);

        [fano, acf_peak] = calculateCountMetrics(spikes_in_win, win_start, win_end, ...
                                                  bin_size, config.acf_max_lag);

        if bin_size == 0.001
            metrics.CountFanoFactor_1ms = fano;
            metrics.Count_ACF_1ms_peak = acf_peak;
        elseif bin_size == 0.025
            metrics.CountFanoFactor_25ms = fano;
            metrics.Count_ACF_25ms_peak = acf_peak;
        elseif bin_size == 0.050
            metrics.CountFanoFactor_50ms = fano;
            metrics.Count_ACF_50ms_peak = acf_peak;
        end
    end
end

function [acf_vals, acf_lags] = calculateISI_ACF(ISI, max_lag)
% Calculate auto-correlation of ISI up to max_lag

    if length(ISI) < 3
        acf_vals = [];
        acf_lags = [];
        return;
    end

    % Determine number of lags
    mean_isi = mean(ISI);
    if mean_isi == 0
        acf_vals = [];
        acf_lags = [];
        return;
    end

    max_lag_samples = min(floor(max_lag / mean_isi), length(ISI) - 1);
    max_lag_samples = max(max_lag_samples, 1);

    % Calculate ACF
    try
        [acf_vals, ~, ~] = autocorr(ISI, max_lag_samples);
        acf_lags = (0:max_lag_samples) * mean_isi;
    catch
        acf_vals = [];
        acf_lags = [];
    end
end

function decay_time = calculateACFDecay(acf_vals, acf_lags)
% Find time to reach 50% of peak ACF value

    if length(acf_vals) < 2
        decay_time = NaN;
        return;
    end

    peak_val = max(acf_vals(2:end));  % Exclude lag 0
    threshold = peak_val * 0.5;

    % Find first crossing
    crossing_idx = find(acf_vals(2:end) < threshold, 1, 'first');

    if isempty(crossing_idx)
        decay_time = acf_lags(end);
    else
        decay_time = acf_lags(crossing_idx + 1);
    end
end

function LV = calculateLV(ISI)
% Local Variation: sensitive to rate changes

    n = length(ISI);
    if n < 2
        LV = NaN;
        return;
    end

    sum_term = 0;
    for i = 1:(n-1)
        sum_term = sum_term + ((ISI(i+1) - ISI(i))^2) / ((ISI(i+1) + ISI(i))^2);
    end

    LV = (3 / (n - 1)) * sum_term;
end

function CV2 = calculateCV2(ISI)
% CV2: Local coefficient of variation

    n = length(ISI);
    if n < 2
        CV2 = NaN;
        return;
    end

    sum_term = 0;
    for i = 1:(n-1)
        sum_term = sum_term + abs(ISI(i+1) - ISI(i)) / (ISI(i+1) + ISI(i));
    end

    CV2 = 2 * sum_term / (n - 1);
end

function LVR = calculateLVR(ISI, refrac_period)
% Revised Local Variation: corrected for refractoriness

    n = length(ISI);
    if n < 2
        LVR = NaN;
        return;
    end

    sum_term = 0;
    valid_count = 0;

    for i = 1:(n-1)
        % Only include pairs where both ISIs > refractory period
        if ISI(i) > refrac_period && ISI(i+1) > refrac_period
            sum_term = sum_term + ((ISI(i+1) - ISI(i))^2) / ((ISI(i+1) + ISI(i))^2);
            valid_count = valid_count + 1;
        end
    end

    if valid_count > 0
        LVR = (3 / valid_count) * sum_term;
    else
        LVR = NaN;
    end
end

function [burst_index, burst_rate, mean_burst_length] = calculateBurstMetrics(ISI, threshold, duration)
% Calculate burst-related metrics

    if isempty(ISI)
        burst_index = NaN;
        burst_rate = NaN;
        mean_burst_length = NaN;
        return;
    end

    % Burst index: fraction of ISIs below threshold
    burst_index = sum(ISI < threshold) / length(ISI);

    % Detect bursts: sequences of ISIs < threshold
    is_burst_isi = ISI < threshold;
    burst_starts = find(diff([0; is_burst_isi]) == 1);
    burst_ends = find(diff([is_burst_isi; 0]) == -1);

    n_bursts = length(burst_starts);

    if n_bursts > 0
        burst_rate = n_bursts / duration;

        % Calculate burst lengths (number of spikes)
        burst_lengths = zeros(n_bursts, 1);
        for b = 1:n_bursts
            burst_lengths(b) = burst_ends(b) - burst_starts(b) + 2;  % +2 for first and last spike
        end
        mean_burst_length = mean(burst_lengths);
    else
        burst_rate = 0;
        mean_burst_length = NaN;
    end
end

function [fano_factor, acf_peak] = calculateCountMetrics(spike_times, win_start, win_end, bin_size, max_lag)
% Calculate spike count Fano factor and ACF for given bin size

    % Create bins
    bin_edges = win_start:bin_size:win_end;

    if length(bin_edges) < 3
        fano_factor = NaN;
        acf_peak = NaN;
        return;
    end

    % Count spikes in bins
    spike_counts = histcounts(spike_times, bin_edges);

    % Fano factor
    if mean(spike_counts) > 0
        fano_factor = var(spike_counts) / mean(spike_counts);
    else
        fano_factor = NaN;
    end

    % ACF
    if length(spike_counts) >= 3
        max_lag_bins = min(floor(max_lag / bin_size), length(spike_counts) - 1);
        max_lag_bins = max(max_lag_bins, 1);

        try
            acf_vals = autocorr(spike_counts, max_lag_bins);
            acf_peak = max(acf_vals(2:end));  % Exclude lag 0
        catch
            acf_peak = NaN;
        end
    else
        acf_peak = NaN;
    end
end

function avg_metrics = averageWindowMetrics(window_metrics_cell)
% Average metrics across multiple windows

    avg_metrics = struct();

    if isempty(window_metrics_cell)
        % Return all NaN
        avg_metrics.FR = NaN;
        avg_metrics.CV = NaN;
        avg_metrics.ISI_FanoFactor = NaN;
        avg_metrics.ISI_ACF_peak = NaN;
        avg_metrics.ISI_ACF_lag = NaN;
        avg_metrics.ISI_ACF_decay = NaN;
        avg_metrics.Count_ACF_1ms_peak = NaN;
        avg_metrics.Count_ACF_25ms_peak = NaN;
        avg_metrics.Count_ACF_50ms_peak = NaN;
        avg_metrics.LV = NaN;
        avg_metrics.CV2 = NaN;
        avg_metrics.LVR = NaN;
        avg_metrics.BurstIndex = NaN;
        avg_metrics.BurstRate = NaN;
        avg_metrics.MeanBurstLength = NaN;
        avg_metrics.ISI_Skewness = NaN;
        avg_metrics.ISI_Kurtosis = NaN;
        avg_metrics.ISI_Mode = NaN;
        avg_metrics.CountFanoFactor_1ms = NaN;
        avg_metrics.CountFanoFactor_25ms = NaN;
        avg_metrics.CountFanoFactor_50ms = NaN;
        avg_metrics.RefracViolations = NaN;
        return;
    end

    % Extract all metric values
    metric_names = fieldnames(window_metrics_cell{1});

    for m = 1:length(metric_names)
        metric_name = metric_names{m};
        values = [];

        for w = 1:length(window_metrics_cell)
            if ~isempty(window_metrics_cell{w}) && isfield(window_metrics_cell{w}, metric_name)
                values(end+1) = window_metrics_cell{w}.(metric_name);
            end
        end

        % Average across windows
        avg_metrics.(metric_name) = mean(values, 'omitnan');
    end
end
