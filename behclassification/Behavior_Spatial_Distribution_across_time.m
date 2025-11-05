%% ========================================================================
%  BEHAVIOR SPATIAL DISTRIBUTION ANALYSIS: Period × Behavior × SessionType
%  Using LSTM predictions for all 7 behavior classes
%  ========================================================================
%
%  Analysis: Spatial distribution of all behaviors across time periods
%  SessionType: Aversive vs Reward
%  Periods: P1-P4 (matched across both session types)
%  Source: LSTM prediction scores
%
%% ========================================================================

clear all
% close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== BEHAVIOR SPATIAL DISTRIBUTION ANALYSIS ===\n');
fprintf('Period × Behavior × SessionType\n\n');

config = struct();
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
config.n_behaviors = 7;
config.spatial_bin_size = 20;  % cm - 20x20 cm bins
config.confidence_threshold = 0.3;  % Minimum confidence for behavior assignment
config.camera_fps = 20;  % Camera frame rate

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');

fprintf('Configuration:\n');
fprintf('  Behaviors: %d\n', config.n_behaviors);
fprintf('  Spatial bin size: %d cm\n', config.spatial_bin_size);
fprintf('  Confidence threshold: %.2f\n', config.confidence_threshold);

%% ========================================================================
%  SECTION 2: LOAD SORTING PARAMETERS
%  ========================================================================

fprintf('\nLoading sorting parameters...\n');
[T_sorted] = loadSortingParameters();
fprintf('✓ Sorting parameters loaded\n');

%% ========================================================================
%  SECTION 3: LOAD BEHAVIOR PREDICTION DATA
%  ========================================================================

fprintf('Loading LSTM behavior predictions...\n');

try
    pred_data_aversive = load('lstm_prediction_results_aversive_27-Oct-2025');
    prediction_sessions_aversive = pred_data_aversive.final_results.session_predictions;
    fprintf('✓ Loaded aversive predictions: %d sessions\n', length(prediction_sessions_aversive));
catch ME
    fprintf('❌ Failed to load aversive predictions: %s\n', ME.message);
    return;
end

try
    pred_data_reward = load('lstm_prediction_results_reward_27-Oct-2025');
    prediction_sessions_reward = pred_data_reward.final_results.session_predictions;
    fprintf('✓ Loaded reward predictions: %d sessions\n', length(prediction_sessions_reward));
catch ME
    fprintf('❌ Failed to load reward predictions: %s\n', ME.message);
    return;
end

% Load coupling data for session matching
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
%  SECTION 4: PROCESS AVERSIVE SESSIONS
%  ========================================================================

fprintf('\n==== PROCESSING AVERSIVE SESSIONS ====\n');

numofsession = 2;
folderpath = "/Volumes/Expansion/Data/Struct_spike";
[allfiles, folderpath, num_aversive_sessions] = selectFilesWithAnimalIDFiltering(folderpath, numofsession, '2025*RewardAversive*.mat');

fprintf('Found %d aversive sessions\n', num_aversive_sessions);

% Storage for all behaviors: session_id, period, behavior, x, y, time
aversive_behavior_data = struct();
aversive_behavior_data.session_id = [];
aversive_behavior_data.period = [];
aversive_behavior_data.behavior = [];
aversive_behavior_data.x_position = [];
aversive_behavior_data.y_position = [];
aversive_behavior_data.prediction_time = [];

% Arena bounds
arena_x_min = inf;
arena_x_max = -inf;
arena_y_min = inf;
arena_y_max = -inf;

n_valid_aversive = 0;

for spike_sess_idx = 1:num_aversive_sessions
    fprintf('\n[%d/%d] Processing: %s\n', spike_sess_idx, num_aversive_sessions, allfiles(spike_sess_idx).name);

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

    % Match with behavior prediction and coupling sessions
    matched = false;
    prediction_scores = [];
    period_boundaries = [];

    for beh_sess_idx = 1:length(sessions_aversive)
        beh_session = sessions_aversive{beh_sess_idx};

        if contains(spike_filename, extractBefore(beh_session.filename, '.mat'))
            matched = true;

            % Get behavior predictions
            if beh_sess_idx <= length(prediction_sessions_aversive)
                prediction_scores = prediction_sessions_aversive(beh_sess_idx).prediction_scores;
                fprintf('  Matched with behavior session: %s\n', beh_session.filename);
                fprintf('  Predictions: %d windows (1 Hz)\n', size(prediction_scores, 1));
            else
                fprintf('  WARNING: No prediction data for this session\n');
            end

            % Get period boundaries
            if isfield(beh_session, 'all_aversive_time') && length(beh_session.all_aversive_time) >= 6
                aversive_times = beh_session.all_aversive_time;
                period_boundaries = [TriggerMid(1), ...
                                     aversive_times(1:3)' + TriggerMid(1), ...
                                     aversive_times(4) + TriggerMid(1)];
                fprintf('  Period boundaries defined\n');
            else
                fprintf('  WARNING: Invalid aversive_times\n');
            end

            break;
        end
    end

    if ~matched || isempty(prediction_scores) || isempty(period_boundaries)
        fprintf('  Skipping session - no match or missing data\n');
        continue;
    end

    n_valid_aversive = n_valid_aversive + 1;

    %% Map predictions to camera frames and XY positions
    % Each prediction corresponds to 20 frames (1 second at 20 fps)
    n_predictions = size(prediction_scores, 1);
    prediction_ind = 1:20:length(TriggerMid);
    prediction_ind = prediction_ind + 10;  % Use middle frame of each 1-second window

    for pred_idx = 1:n_predictions
        if pred_idx > length(prediction_ind)
            break;
        end

        frame_idx = prediction_ind(pred_idx);

        if frame_idx <= 0 || frame_idx > length(TriggerMid) || frame_idx > size(AdjustedXYZ, 1)
            continue;
        end

        % Get time and position
        pred_time = TriggerMid(frame_idx);
        x_pos = AdjustedXYZ(frame_idx, 1);
        y_pos = AdjustedXYZ(frame_idx, 2);

        % Get dominant behavior
        [max_confidence, dominant_beh] = max(prediction_scores(pred_idx, :));

        % Filter by confidence threshold
        if max_confidence < config.confidence_threshold
            continue;
        end

        % Determine period
        period = 0;
        for p = 1:4
            if pred_time >= period_boundaries(p) && pred_time < period_boundaries(p+1)
                period = p;
                break;
            end
        end

        if period > 0
            % Store data
            aversive_behavior_data.session_id(end+1) = n_valid_aversive;
            aversive_behavior_data.period(end+1) = period;
            aversive_behavior_data.behavior(end+1) = dominant_beh;
            aversive_behavior_data.x_position(end+1) = x_pos;
            aversive_behavior_data.y_position(end+1) = y_pos;
            aversive_behavior_data.prediction_time(end+1) = pred_time;

            % Update arena bounds
            arena_x_min = min(arena_x_min, x_pos);
            arena_x_max = max(arena_x_max, x_pos);
            arena_y_min = min(arena_y_min, y_pos);
            arena_y_max = max(arena_y_max, y_pos);
        end
    end
end

fprintf('\n✓ Processed %d aversive sessions\n', n_valid_aversive);
fprintf('  Total prediction windows: %d\n', length(aversive_behavior_data.session_id));

%% ========================================================================
%  SECTION 5: PROCESS REWARD SESSIONS
%  ========================================================================

fprintf('\n==== PROCESSING REWARD SESSIONS ====\n');

[allfiles, folderpath, num_reward_sessions] = selectFilesWithAnimalIDFiltering(folderpath, numofsession, '2025*RewardSeeking*.mat');

fprintf('Found %d reward sessions\n', num_reward_sessions);

% Storage for reward sessions
reward_behavior_data = struct();
reward_behavior_data.session_id = [];
reward_behavior_data.period = [];
reward_behavior_data.behavior = [];
reward_behavior_data.x_position = [];
reward_behavior_data.y_position = [];
reward_behavior_data.prediction_time = [];

n_valid_reward = 0;

for spike_sess_idx = 1:num_reward_sessions
    fprintf('\n[%d/%d] Processing: %s\n', spike_sess_idx, num_reward_sessions, allfiles(spike_sess_idx).name);

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

    % Match with behavior prediction
    matched = false;
    prediction_scores = [];

    for beh_sess_idx = 1:length(sessions_reward)
        beh_session = sessions_reward{beh_sess_idx};

        if contains(spike_filename, extractBefore(beh_session.filename, '.mat'))
            matched = true;

            if beh_sess_idx <= length(prediction_sessions_reward)
                prediction_scores = prediction_sessions_reward(beh_sess_idx).prediction_scores;
                fprintf('  Matched with behavior session: %s\n', beh_session.filename);
                fprintf('  Predictions: %d windows (1 Hz)\n', size(prediction_scores, 1));
            else
                fprintf('  WARNING: No prediction data for this session\n');
            end

            % Period boundaries based on time
            time_boundaries = [0, 8*60, 16*60, 24*60, 30*60];
            period_boundaries = [TriggerMid(1), time_boundaries(2:end) + TriggerMid(1)];
            fprintf('  Period boundaries defined\n');

            break;
        end
    end

    if ~matched || isempty(prediction_scores)
        fprintf('  Skipping session - no match or missing data\n');
        continue;
    end

    n_valid_reward = n_valid_reward + 1;

    %% Map predictions to positions
    n_predictions = size(prediction_scores, 1);
    prediction_ind = 1:20:length(TriggerMid);
    prediction_ind = prediction_ind + 10;

    for pred_idx = 1:n_predictions
        if pred_idx > length(prediction_ind)
            break;
        end

        frame_idx = prediction_ind(pred_idx);

        if frame_idx <= 0 || frame_idx > length(TriggerMid) || frame_idx > size(AdjustedXYZ, 1)
            continue;
        end

        pred_time = TriggerMid(frame_idx);
        x_pos = AdjustedXYZ(frame_idx, 1);
        y_pos = AdjustedXYZ(frame_idx, 2);

        [max_confidence, dominant_beh] = max(prediction_scores(pred_idx, :));

        if max_confidence < config.confidence_threshold
            continue;
        end

        % Determine period
        period = 0;
        for p = 1:4
            if pred_time >= period_boundaries(p) && pred_time < period_boundaries(p+1)
                period = p;
                break;
            end
        end

        if period > 0
            reward_behavior_data.session_id(end+1) = n_valid_reward;
            reward_behavior_data.period(end+1) = period;
            reward_behavior_data.behavior(end+1) = dominant_beh;
            reward_behavior_data.x_position(end+1) = x_pos;
            reward_behavior_data.y_position(end+1) = y_pos;
            reward_behavior_data.prediction_time(end+1) = pred_time;

            % Update arena bounds
            arena_x_min = min(arena_x_min, x_pos);
            arena_x_max = max(arena_x_max, x_pos);
            arena_y_min = min(arena_y_min, y_pos);
            arena_y_max = max(arena_y_max, y_pos);
        end
    end
end

fprintf('\n✓ Processed %d reward sessions\n', n_valid_reward);
fprintf('  Total prediction windows: %d\n', length(reward_behavior_data.session_id));

%% ========================================================================
%  SECTION 6: CREATE SPATIAL BINS
%  ========================================================================

fprintf('\n==== CREATING SPATIAL BINS ====\n');

% Round arena bounds to nearest bin size
arena_x_min = floor(arena_x_min / config.spatial_bin_size) * config.spatial_bin_size;
arena_x_max = ceil(arena_x_max / config.spatial_bin_size) * config.spatial_bin_size;
arena_y_min = floor(arena_y_min / config.spatial_bin_size) * config.spatial_bin_size;
arena_y_max = ceil(arena_y_max / config.spatial_bin_size) * config.spatial_bin_size;

x_edges = arena_x_min:config.spatial_bin_size:arena_x_max;
y_edges = arena_y_min:config.spatial_bin_size:arena_y_max;

n_x_bins = length(x_edges) - 1;
n_y_bins = length(y_edges) - 1;

fprintf('Arena bounds: X=[%.1f, %.1f], Y=[%.1f, %.1f]\n', arena_x_min, arena_x_max, arena_y_min, arena_y_max);
fprintf('Spatial bins: %d × %d (%.1f cm bins)\n', n_x_bins, n_y_bins, config.spatial_bin_size);

%% ========================================================================
%  SECTION 7: COMPUTE SPATIAL MAPS
%  ========================================================================

fprintf('\n==== COMPUTING SPATIAL MAPS ====\n');

% Initialize: [n_y_bins × n_x_bins × 4 periods × 2 session types × 7 behaviors]
behavior_spatial_counts = zeros(n_y_bins, n_x_bins, 4, 2, config.n_behaviors);
occupancy_counts = zeros(n_y_bins, n_x_bins, 4, 2);  % Total occupancy (all behaviors)
behavior_spatial_density = zeros(n_y_bins, n_x_bins, 4, 2, config.n_behaviors);  % Normalized

% Process aversive data
fprintf('Processing aversive spatial data...\n');
for i = 1:length(aversive_behavior_data.session_id)
    x = aversive_behavior_data.x_position(i);
    y = aversive_behavior_data.y_position(i);
    p = aversive_behavior_data.period(i);
    b = aversive_behavior_data.behavior(i);

    x_bin = discretize(x, x_edges);
    y_bin = discretize(y, y_edges);

    if ~isnan(x_bin) && ~isnan(y_bin)
        behavior_spatial_counts(y_bin, x_bin, p, 1, b) = behavior_spatial_counts(y_bin, x_bin, p, 1, b) + 1;
        occupancy_counts(y_bin, x_bin, p, 1) = occupancy_counts(y_bin, x_bin, p, 1) + 1;
    end
end

% Process reward data
fprintf('Processing reward spatial data...\n');
for i = 1:length(reward_behavior_data.session_id)
    x = reward_behavior_data.x_position(i);
    y = reward_behavior_data.y_position(i);
    p = reward_behavior_data.period(i);
    b = reward_behavior_data.behavior(i);

    x_bin = discretize(x, x_edges);
    y_bin = discretize(y, y_edges);

    if ~isnan(x_bin) && ~isnan(y_bin)
        behavior_spatial_counts(y_bin, x_bin, p, 2, b) = behavior_spatial_counts(y_bin, x_bin, p, 2, b) + 1;
        occupancy_counts(y_bin, x_bin, p, 2) = occupancy_counts(y_bin, x_bin, p, 2) + 1;
    end
end

% Normalize by occupancy (probability of behavior given location)
fprintf('Normalizing by occupancy...\n');
for session_type = 1:2
    for p = 1:4
        for b = 1:config.n_behaviors
            % Normalize: behavior_count / total_occupancy in each bin
            occ_map = occupancy_counts(:, :, p, session_type);
            beh_map = behavior_spatial_counts(:, :, p, session_type, b);

            % Avoid division by zero
            valid_bins = occ_map > 0;
            behavior_spatial_density(:, :, p, session_type, b) = zeros(n_y_bins, n_x_bins);
            behavior_spatial_density(valid_bins, 1, p, session_type, b) = ...
                beh_map(valid_bins) ./ occ_map(valid_bins);

            % Fix dimension issue
            temp_density = zeros(n_y_bins, n_x_bins);
            temp_density(valid_bins) = beh_map(valid_bins) ./ occ_map(valid_bins);
            behavior_spatial_density(:, :, p, session_type, b) = temp_density;
        end
    end
end

fprintf('✓ Spatial maps computed\n');

%% ========================================================================
%  SECTION 8: VISUALIZATION - One figure per behavior
%  ========================================================================

fprintf('\n==== CREATING VISUALIZATIONS ====\n');

session_type_names = {'Aversive', 'Reward'};

for b = 1:config.n_behaviors
    fprintf('Creating figure for %s...\n', config.behavior_names{b});

    figure('Position', [50 + (b-1)*50, 50 + (b-1)*50, 1400, 700], ...
           'Name', sprintf('Spatial Distribution - %s', config.behavior_names{b}));

    for session_type = 1:2
        for p = 1:4
            subplot_idx = (session_type - 1) * 4 + p;
            subplot(2, 4, subplot_idx);

            % Get spatial density map
            spatial_map = behavior_spatial_density(:, :, p, session_type, b);

            % Plot
            imagesc(x_edges(1:end-1) + config.spatial_bin_size/2, ...
                    y_edges(1:end-1) + config.spatial_bin_size/2, ...
                    spatial_map);

            colormap(hot);
            colorbar;
            caxis([0, max(spatial_map(:))]);
            axis equal tight;

            % Title
            if session_type == 1
                title_color = [1, 0, 0];  % Red for aversive
            else
                title_color = [0, 0.6, 0];  % Green for reward
            end

            title(sprintf('%s - P%d', session_type_names{session_type}, p), ...
                'FontSize', 11, 'FontWeight', 'bold', 'Color', title_color);

            xlabel('X (cm)', 'FontSize', 9);
            ylabel('Y (cm)', 'FontSize', 9);

            % Add count
            total_count = sum(sum(behavior_spatial_counts(:, :, p, session_type, b)));
            text(0.02, 0.98, sprintf('n=%d', total_count), ...
                'Units', 'normalized', 'VerticalAlignment', 'top', ...
                'BackgroundColor', 'w', 'FontSize', 9);
        end
    end

    sgtitle(sprintf('Spatial Distribution: %s (Occupancy-Normalized)', config.behavior_names{b}), ...
        'FontSize', 13, 'FontWeight', 'bold');
end

fprintf('✓ All visualizations complete\n');

%% ========================================================================
%  SECTION 9: SUMMARY
%  ========================================================================

fprintf('\n========================================\n');
fprintf('ANALYSIS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Sessions processed:\n');
fprintf('  Aversive: %d sessions, %d prediction windows\n', n_valid_aversive, length(aversive_behavior_data.session_id));
fprintf('  Reward: %d sessions, %d prediction windows\n', n_valid_reward, length(reward_behavior_data.session_id));
fprintf('Behaviors analyzed: %d\n', config.n_behaviors);
fprintf('Spatial resolution: %d × %d bins (%.1f cm)\n', n_x_bins, n_y_bins, config.spatial_bin_size);
fprintf('Figures created: %d (one per behavior)\n', config.n_behaviors);
fprintf('========================================\n');
