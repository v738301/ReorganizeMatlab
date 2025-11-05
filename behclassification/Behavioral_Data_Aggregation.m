%% ========================================================================
%  BEHAVIORAL DATA AGGREGATION
%  Loads, processes, and groups all behavioral data for analysis
%  ========================================================================
%
%  This script:
%  1. Loads LSTM behavioral predictions (7 classes, 1 Hz)
%  2. Loads behavioral matrices (8 features, neural sampling rate)
%  3. Computes summary statistics for each session
%  4. Saves aggregated data for visualization
%
%  Output: behavioral_data_summary.mat
%
%% ========================================================================

clear all;
close all;

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== BEHAVIORAL DATA AGGREGATION ===\n\n');

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

% Camera frame rate
config.camera_fps = 20;  % 20 Hz

fprintf('Configuration:\n');
fprintf('  LSTM behaviors: %d\n', config.n_behaviors);
fprintf('  Matrix features: %d\n', config.n_matrix_features);
fprintf('  Camera FPS: %d\n\n', config.camera_fps);

%% ========================================================================
%  SECTION 2: LOAD LSTM PREDICTIONS
%  ========================================================================

fprintf('Loading LSTM behavioral predictions...\n');

% Load aversive predictions
try
    pred_data_aversive = load('lstm_prediction_results_aversive_27-Oct-2025');
    prediction_sessions_aversive = pred_data_aversive.final_results.session_predictions;
    fprintf('✓ Loaded aversive predictions: %d sessions\n', length(prediction_sessions_aversive));
catch ME
    fprintf('❌ Failed to load aversive predictions: %s\n', ME.message);
    return;
end

% Load reward predictions
try
    pred_data_reward = load('lstm_prediction_results_reward_27-Oct-2025');
    prediction_sessions_reward = pred_data_reward.final_results.session_predictions;
    fprintf('✓ Loaded reward predictions: %d sessions\n', length(prediction_sessions_reward));
catch ME
    fprintf('❌ Failed to load reward predictions: %s\n', ME.message);
    return;
end

%% ========================================================================
%  SECTION 3: LOAD BEHAVIORAL MATRICES
%  ========================================================================

fprintf('\nLoading behavioral matrices...\n');

% Load aversive coupling data (contains behavioral matrices)
try
    coupling_data_aversive = load('27-Oct-2025_RewardAversive_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_aversive = coupling_data_aversive.all_session_metrics;
    fprintf('✓ Loaded aversive behavioral matrices: %d sessions\n', length(sessions_aversive));
catch ME
    fprintf('❌ Failed to load aversive matrices: %s\n', ME.message);
    return;
end

% Load reward coupling data
try
    coupling_data_reward = load('27-Oct-2025_RewardSeeking_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_reward = coupling_data_reward.all_session_metrics;
    fprintf('✓ Loaded reward behavioral matrices: %d sessions\n', length(sessions_reward));
catch ME
    fprintf('❌ Failed to load reward matrices: %s\n', ME.message);
    return;
end

%% ========================================================================
%  SECTION 4: PROCESS AVERSIVE SESSIONS
%  ========================================================================

fprintf('\n==== PROCESSING AVERSIVE SESSIONS ====\n');

n_aversive = length(sessions_aversive);
aversive_data = [];

for sess_idx = 1:n_aversive
    fprintf('[%d/%d] Processing: %s\n', sess_idx, n_aversive, sessions_aversive{sess_idx}.filename);

    session = sessions_aversive{sess_idx};

    % Initialize session data structure
    sess_data = struct();
    sess_data.session_id = sess_idx;
    sess_data.filename = session.filename;
    sess_data.session_type = 'Aversive';

    % Check required fields
    if ~isfield(session, 'NeuralTime') || ~isfield(session, 'TriggerMid')
        fprintf('  WARNING: Missing required fields - skipping\n');
        continue;
    end

    sess_data.duration_min = (session.NeuralTime(end) - session.NeuralTime(1)) / 60;

    %% Process LSTM predictions
    if sess_idx <= length(prediction_sessions_aversive)
        prediction_scores = prediction_sessions_aversive(sess_idx).prediction_scores;
        [max_confidence, dominant_beh] = max(prediction_scores, [], 2);

        % Calculate time budget (% time in each behavior)
        behavior_time_percent = zeros(config.n_behaviors, 1);
        for beh = 1:config.n_behaviors
            behavior_time_percent(beh) = sum(dominant_beh == beh) / length(dominant_beh) * 100;
        end

        sess_data.behavior_time_percent = behavior_time_percent;
        sess_data.behavior_predictions = dominant_beh;
        sess_data.prediction_confidence = max_confidence;

        fprintf('  LSTM behaviors extracted\n');
    else
        fprintf('  WARNING: No LSTM predictions for this session\n');
        sess_data.behavior_time_percent = nan(config.n_behaviors, 1);
        sess_data.behavior_predictions = [];
        sess_data.prediction_confidence = [];
    end

    %% Process behavioral matrix
    if isfield(session, 'behavioral_matrix_full') && ~isempty(session.behavioral_matrix_full)
        behavioral_matrix = session.behavioral_matrix_full;
        neural_time = session.NeuralTime;

        % Calculate state occupancy for binary features (1-7)
        matrix_feature_percent = zeros(config.n_matrix_features, 1);
        for feat = 1:7
            matrix_feature_percent(feat) = sum(behavioral_matrix(:, feat) == 1) / length(neural_time) * 100;
        end

        % Feature 8 is breathing frequency (continuous)
        breathing_freq = behavioral_matrix(:, 8);
        breathing_freq_clean = breathing_freq(breathing_freq > 0 & breathing_freq < 20);  % Remove outliers

        sess_data.matrix_feature_percent = matrix_feature_percent;
        sess_data.breathing_overall_mean = mean(breathing_freq_clean);
        sess_data.breathing_overall_std = std(breathing_freq_clean);
        sess_data.breathing_overall_median = median(breathing_freq_clean);
        sess_data.behavioral_matrix = behavioral_matrix;

        fprintf('  Behavioral matrix extracted\n');

        %% Calculate breathing rate per LSTM behavior
        if ~isempty(sess_data.behavior_predictions)
            breathing_by_behavior = nan(config.n_behaviors, 1);
            breathing_std_by_behavior = nan(config.n_behaviors, 1);

            % Map LSTM predictions to neural time
            n_predictions = length(sess_data.behavior_predictions);
            behavior_at_neural = zeros(size(neural_time));
            trigger_mid = session.TriggerMid;

            for pred_idx = 1:n_predictions
                frame_start = (pred_idx - 1) * config.camera_fps + 1;
                frame_end = min(frame_start + config.camera_fps - 1, length(trigger_mid));

                if frame_end <= length(trigger_mid)
                    pred_time_start = trigger_mid(frame_start);
                    pred_time_end = trigger_mid(frame_end);

                    neural_indices = find(neural_time >= pred_time_start & neural_time <= pred_time_end);
                    behavior_at_neural(neural_indices) = sess_data.behavior_predictions(pred_idx);
                end
            end

            % Calculate breathing rate for each behavior
            for beh = 1:config.n_behaviors
                beh_mask = behavior_at_neural == beh;
                if sum(beh_mask) > 0
                    beh_breathing = breathing_freq(beh_mask);
                    beh_breathing_clean = beh_breathing(beh_breathing > 0 & beh_breathing < 20);

                    if ~isempty(beh_breathing_clean)
                        breathing_by_behavior(beh) = mean(beh_breathing_clean);
                        breathing_std_by_behavior(beh) = std(beh_breathing_clean);
                    end
                end
            end

            sess_data.breathing_by_behavior = breathing_by_behavior;
            sess_data.breathing_std_by_behavior = breathing_std_by_behavior;

            fprintf('  Breathing by behavior computed\n');
        else
            sess_data.breathing_by_behavior = nan(config.n_behaviors, 1);
            sess_data.breathing_std_by_behavior = nan(config.n_behaviors, 1);
        end
    else
        fprintf('  WARNING: No behavioral matrix for this session\n');
        sess_data.matrix_feature_percent = nan(config.n_matrix_features, 1);
        sess_data.breathing_overall_mean = NaN;
        sess_data.breathing_overall_std = NaN;
        sess_data.breathing_overall_median = NaN;
        sess_data.breathing_by_behavior = nan(config.n_behaviors, 1);
        sess_data.breathing_std_by_behavior = nan(config.n_behaviors, 1);
        sess_data.behavioral_matrix = [];
    end

    % Add to array
    if sess_idx == 1
        aversive_data = sess_data;
    else
        aversive_data(sess_idx) = sess_data;
    end

    fprintf('  ✓ Session %d processed\n', sess_idx);
end

fprintf('\n✓ Processed %d aversive sessions\n', length(aversive_data));

%% ========================================================================
%  SECTION 5: PROCESS REWARD SESSIONS
%  ========================================================================

fprintf('\n==== PROCESSING REWARD SESSIONS ====\n');

n_reward = length(sessions_reward);
reward_data = [];

for sess_idx = 1:n_reward
    fprintf('[%d/%d] Processing: %s\n', sess_idx, n_reward, sessions_reward{sess_idx}.filename);

    session = sessions_reward{sess_idx};

    % Initialize session data structure
    sess_data = struct();
    sess_data.session_id = sess_idx;
    sess_data.filename = session.filename;
    sess_data.session_type = 'Reward';

    % Check required fields
    if ~isfield(session, 'NeuralTime') || ~isfield(session, 'TriggerMid')
        fprintf('  WARNING: Missing required fields - skipping\n');
        continue;
    end

    sess_data.duration_min = (session.NeuralTime(end) - session.NeuralTime(1)) / 60;

    %% Process LSTM predictions
    if sess_idx <= length(prediction_sessions_reward)
        prediction_scores = prediction_sessions_reward(sess_idx).prediction_scores;
        [max_confidence, dominant_beh] = max(prediction_scores, [], 2);

        % Calculate time budget
        behavior_time_percent = zeros(config.n_behaviors, 1);
        for beh = 1:config.n_behaviors
            behavior_time_percent(beh) = sum(dominant_beh == beh) / length(dominant_beh) * 100;
        end

        sess_data.behavior_time_percent = behavior_time_percent;
        sess_data.behavior_predictions = dominant_beh;
        sess_data.prediction_confidence = max_confidence;

        fprintf('  LSTM behaviors extracted\n');
    else
        fprintf('  WARNING: No LSTM predictions for this session\n');
        sess_data.behavior_time_percent = nan(config.n_behaviors, 1);
        sess_data.behavior_predictions = [];
        sess_data.prediction_confidence = [];
    end

    %% Process behavioral matrix
    if isfield(session, 'behavioral_matrix_full') && ~isempty(session.behavioral_matrix_full)
        behavioral_matrix = session.behavioral_matrix_full;
        neural_time = session.NeuralTime;

        % Calculate state occupancy
        matrix_feature_percent = zeros(config.n_matrix_features, 1);
        for feat = 1:7
            matrix_feature_percent(feat) = sum(behavioral_matrix(:, feat) == 1) / length(neural_time) * 100;
        end

        % Breathing frequency
        breathing_freq = behavioral_matrix(:, 8);
        breathing_freq_clean = breathing_freq(breathing_freq > 0 & breathing_freq < 20);

        sess_data.matrix_feature_percent = matrix_feature_percent;
        sess_data.breathing_overall_mean = mean(breathing_freq_clean);
        sess_data.breathing_overall_std = std(breathing_freq_clean);
        sess_data.breathing_overall_median = median(breathing_freq_clean);
        sess_data.behavioral_matrix = behavioral_matrix;

        fprintf('  Behavioral matrix extracted\n');

        %% Calculate breathing rate per LSTM behavior
        if ~isempty(sess_data.behavior_predictions)
            breathing_by_behavior = nan(config.n_behaviors, 1);
            breathing_std_by_behavior = nan(config.n_behaviors, 1);

            % Map LSTM predictions to neural time
            n_predictions = length(sess_data.behavior_predictions);
            behavior_at_neural = zeros(size(neural_time));
            trigger_mid = session.TriggerMid;

            for pred_idx = 1:n_predictions
                frame_start = (pred_idx - 1) * config.camera_fps + 1;
                frame_end = min(frame_start + config.camera_fps - 1, length(trigger_mid));

                if frame_end <= length(trigger_mid)
                    pred_time_start = trigger_mid(frame_start);
                    pred_time_end = trigger_mid(frame_end);

                    neural_indices = find(neural_time >= pred_time_start & neural_time <= pred_time_end);
                    behavior_at_neural(neural_indices) = sess_data.behavior_predictions(pred_idx);
                end
            end

            % Calculate breathing rate for each behavior
            for beh = 1:config.n_behaviors
                beh_mask = behavior_at_neural == beh;
                if sum(beh_mask) > 0
                    beh_breathing = breathing_freq(beh_mask);
                    beh_breathing_clean = beh_breathing(beh_breathing > 0 & beh_breathing < 20);

                    if ~isempty(beh_breathing_clean)
                        breathing_by_behavior(beh) = mean(beh_breathing_clean);
                        breathing_std_by_behavior(beh) = std(beh_breathing_clean);
                    end
                end
            end

            sess_data.breathing_by_behavior = breathing_by_behavior;
            sess_data.breathing_std_by_behavior = breathing_std_by_behavior;

            fprintf('  Breathing by behavior computed\n');
        else
            sess_data.breathing_by_behavior = nan(config.n_behaviors, 1);
            sess_data.breathing_std_by_behavior = nan(config.n_behaviors, 1);
        end
    else
        fprintf('  WARNING: No behavioral matrix for this session\n');
        sess_data.matrix_feature_percent = nan(config.n_matrix_features, 1);
        sess_data.breathing_overall_mean = NaN;
        sess_data.breathing_overall_std = NaN;
        sess_data.breathing_overall_median = NaN;
        sess_data.breathing_by_behavior = nan(config.n_behaviors, 1);
        sess_data.breathing_std_by_behavior = nan(config.n_behaviors, 1);
        sess_data.behavioral_matrix = [];
    end

    % Add to array
    if sess_idx == 1
        reward_data = sess_data;
    else
        reward_data(sess_idx) = sess_data;
    end

    fprintf('  ✓ Session %d processed\n', sess_idx);
end

fprintf('\n✓ Processed %d reward sessions\n', length(reward_data));

%% ========================================================================
%  SECTION 6: SAVE AGGREGATED DATA
%  ========================================================================

fprintf('\nSaving aggregated data...\n');

% Package everything
behavioral_summary = struct();
behavioral_summary.aversive_sessions = aversive_data;
behavioral_summary.reward_sessions = reward_data;
behavioral_summary.config = config;
behavioral_summary.n_aversive = length(aversive_data);
behavioral_summary.n_reward = length(reward_data);
behavioral_summary.behavior_names = config.behavior_names;
behavioral_summary.matrix_feature_names = config.matrix_feature_names;

% Save
save_filename = 'behavioral_data_summary.mat';
save(save_filename, 'behavioral_summary', '-v7.3');

fprintf('✓ Saved to: %s\n', save_filename);

%% ========================================================================
%  SECTION 7: SUMMARY STATISTICS
%  ========================================================================

fprintf('\n=== SUMMARY ===\n');
fprintf('Total sessions processed: %d\n', length(aversive_data) + length(reward_data));
fprintf('  Aversive: %d\n', length(aversive_data));
fprintf('  Reward: %d\n', length(reward_data));

fprintf('\nData fields per session:\n');
fprintf('  - Session metadata (ID, filename, type, duration)\n');
fprintf('  - LSTM behavior time budget (7 behaviors)\n');
fprintf('  - Behavioral matrix state occupancy (8 features)\n');
fprintf('  - Overall breathing statistics\n');
fprintf('  - Breathing rate per behavior (7 behaviors)\n');
fprintf('  - Raw predictions and matrices (for detailed analysis)\n');

fprintf('\n========================================\n');
fprintf('DATA AGGREGATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Next step: Run Behavioral_Comprehensive_Visualization.m\n');
fprintf('========================================\n');
