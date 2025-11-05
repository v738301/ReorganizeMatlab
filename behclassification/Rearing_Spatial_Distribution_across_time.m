%% ========================================================================
%  REARING SPATIAL DISTRIBUTION ANALYSIS: Period × SessionType
%  Using ground-truth rearing annotations from behavioral_matrix
%  ========================================================================
%
%  Analysis: Spatial distribution of rearing onsets across time periods
%  SessionType: Aversive vs Reward
%  Periods: P1-P4 (matched across both session types)
%  Source: behavioral_matrix_full column 6 ("Rearing_BM")
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== REARING SPATIAL DISTRIBUTION ANALYSIS ===\n');
fprintf('Period × SessionType\n\n');

config = struct();
config.spatial_bin_size = 20;  % cm - 20x20 cm bins
config.bout_epsilon = 1.0;  % seconds - max gap within a bout
config.bout_minPts = 2;  % minimum points to form a bout
config.camera_fps = 20;  % Camera frame rate

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');

fprintf('Configuration:\n');
fprintf('  Spatial bin size: %d cm\n', config.spatial_bin_size);
fprintf('  Bout detection: epsilon=%.1fs, minPts=%d\n', config.bout_epsilon, config.bout_minPts);

%% ========================================================================
%  SECTION 2: LOAD SORTING PARAMETERS
%  ========================================================================

fprintf('\nLoading sorting parameters...\n');
[T_sorted] = loadSortingParameters();
fprintf('✓ Sorting parameters loaded\n');

%% ========================================================================
%  SECTION 3: LOAD COUPLING DATA FOR SESSION MATCHING
%  ========================================================================

fprintf('Loading coupling session data...\n');

try
    coupling_data_aversive = load('27-Oct-2025_RewardAversive_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_aversive = coupling_data_aversive.all_session_metrics;
    fprintf('✓ Loaded aversive coupling data: %d sessions\n', length(sessions_aversive));
catch ME
    fprintf('❌ Failed to load aversive coupling data: %s\n', ME.message);
    return;
end

try
    coupling_data_reward = load('27-Oct-2025_RewardSeeking_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_reward = coupling_data_reward.all_session_metrics;
    fprintf('✓ Loaded reward coupling data: %d sessions\n\n', length(sessions_reward));
catch ME
    fprintf('❌ Failed to load reward coupling data: %s\n', ME.message);
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

% Storage for all sessions
aversive_rearing_data = struct();
aversive_rearing_data.session_id = [];
aversive_rearing_data.period = [];
aversive_rearing_data.x_position = [];
aversive_rearing_data.y_position = [];
aversive_rearing_data.onset_time = [];

% Arena bounds (will be updated from first session)
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

    % Match with coupling session to get behavioral_matrix_full
    matched = false;
    behavioral_matrix_full = [];

    for beh_sess_idx = 1:length(sessions_aversive)
        beh_session = sessions_aversive{beh_sess_idx};

        if contains(spike_filename, extractBefore(beh_session.filename, '.mat'))
            matched = true;

            if isfield(beh_session, 'behavioral_matrix_full')
                behavioral_matrix_full = beh_session.behavioral_matrix_full;
                fprintf('  Matched with coupling session: %s\n', beh_session.filename);
                fprintf('  Behavioral matrix: %d × %d\n', size(behavioral_matrix_full, 1), size(behavioral_matrix_full, 2));
            else
                fprintf('  WARNING: No behavioral_matrix_full found\n');
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
                period_boundaries = [];
            end

            break;
        end
    end

    if ~matched || isempty(behavioral_matrix_full) || isempty(period_boundaries)
        fprintf('  Skipping session - no match or missing data\n');
        continue;
    end

    n_valid_aversive = n_valid_aversive + 1;

    %% Extract rearing onsets from behavioral_matrix
    % Column 6 = "Rearing_BM"
    rearing_signal = behavioral_matrix_full(:, 6);

    % Find rearing bout onsets using bout detection
    rearing_indices = find(rearing_signal == 1);

    if isempty(rearing_indices)
        fprintf('  No rearing events detected\n');
        continue;
    end

    rearing_times_raw = NeuralTime(rearing_indices);

    try
        [rearing_onset_times, ~] = findEventCluster_SuperFast(rearing_times_raw, ...
            config.bout_epsilon, config.bout_minPts);
        fprintf('  Rearing onsets: %d bouts (from %d raw events)\n', ...
            length(rearing_onset_times), length(rearing_times_raw));
    catch ME
        fprintf('  ERROR in bout detection: %s\n', ME.message);
        continue;
    end

    if isempty(rearing_onset_times)
        fprintf('  No rearing onsets after bout detection\n');
        continue;
    end

    %% Map rearing onset times to XY positions
    % For each onset time, find nearest camera frame and get position

    for onset_idx = 1:length(rearing_onset_times)
        onset_time = rearing_onset_times(onset_idx);

        % Find nearest camera frame
        [~, closest_frame] = min(abs(TriggerMid - onset_time));

        if closest_frame > 0 && closest_frame <= size(AdjustedXYZ, 1)
            x_pos = AdjustedXYZ(closest_frame, 1);
            y_pos = AdjustedXYZ(closest_frame, 2);

            % Determine which period this onset belongs to
            period = 0;
            for p = 1:4
                if onset_time >= period_boundaries(p) && onset_time < period_boundaries(p+1)
                    period = p;
                    break;
                end
            end

            if period > 0
                % Store data
                aversive_rearing_data.session_id(end+1) = n_valid_aversive;
                aversive_rearing_data.period(end+1) = period;
                aversive_rearing_data.x_position(end+1) = x_pos;
                aversive_rearing_data.y_position(end+1) = y_pos;
                aversive_rearing_data.onset_time(end+1) = onset_time;

                % Update arena bounds
                arena_x_min = min(arena_x_min, x_pos);
                arena_x_max = max(arena_x_max, x_pos);
                arena_y_min = min(arena_y_min, y_pos);
                arena_y_max = max(arena_y_max, y_pos);
            end
        end
    end
end

fprintf('\n✓ Processed %d aversive sessions\n', n_valid_aversive);
fprintf('  Total rearing onsets: %d\n', length(aversive_rearing_data.session_id));

%% ========================================================================
%  SECTION 5: PROCESS REWARD SESSIONS
%  ========================================================================

fprintf('\n==== PROCESSING REWARD SESSIONS ====\n');

[allfiles, folderpath, num_reward_sessions] = selectFilesWithAnimalIDFiltering(folderpath, numofsession, '2025*RewardSeeking*.mat');

fprintf('Found %d reward sessions\n', num_reward_sessions);

% Storage for reward sessions
reward_rearing_data = struct();
reward_rearing_data.session_id = [];
reward_rearing_data.period = [];
reward_rearing_data.x_position = [];
reward_rearing_data.y_position = [];
reward_rearing_data.onset_time = [];

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

    % Match with coupling session
    matched = false;
    behavioral_matrix_full = [];

    for beh_sess_idx = 1:length(sessions_reward)
        beh_session = sessions_reward{beh_sess_idx};

        if contains(spike_filename, extractBefore(beh_session.filename, '.mat'))
            matched = true;

            if isfield(beh_session, 'behavioral_matrix_full')
                behavioral_matrix_full = beh_session.behavioral_matrix_full;
                fprintf('  Matched with coupling session: %s\n', beh_session.filename);
                fprintf('  Behavioral matrix: %d × %d\n', size(behavioral_matrix_full, 1), size(behavioral_matrix_full, 2));
            else
                fprintf('  WARNING: No behavioral_matrix_full found\n');
            end

            % Period boundaries based on time
            time_boundaries = [0, 8*60, 16*60, 24*60, 30*60];
            period_boundaries = [TriggerMid(1), time_boundaries(2:end) + TriggerMid(1)];
            fprintf('  Period boundaries defined\n');

            break;
        end
    end

    if ~matched || isempty(behavioral_matrix_full)
        fprintf('  Skipping session - no match or missing data\n');
        continue;
    end

    n_valid_reward = n_valid_reward + 1;

    %% Extract rearing onsets
    rearing_signal = behavioral_matrix_full(:, 6);
    rearing_indices = find(rearing_signal == 1);

    if isempty(rearing_indices)
        fprintf('  No rearing events detected\n');
        continue;
    end

    rearing_times_raw = NeuralTime(rearing_indices);

    try
        [rearing_onset_times, ~] = findEventCluster_SuperFast(rearing_times_raw, ...
            config.bout_epsilon, config.bout_minPts);
        fprintf('  Rearing onsets: %d bouts (from %d raw events)\n', ...
            length(rearing_onset_times), length(rearing_times_raw));
    catch ME
        fprintf('  ERROR in bout detection: %s\n', ME.message);
        continue;
    end

    if isempty(rearing_onset_times)
        fprintf('  No rearing onsets after bout detection\n');
        continue;
    end

    %% Map to XY positions
    for onset_idx = 1:length(rearing_onset_times)
        onset_time = rearing_onset_times(onset_idx);

        [~, closest_frame] = min(abs(TriggerMid - onset_time));

        if closest_frame > 0 && closest_frame <= size(AdjustedXYZ, 1)
            x_pos = AdjustedXYZ(closest_frame, 1);
            y_pos = AdjustedXYZ(closest_frame, 2);

            % Determine period
            period = 0;
            for p = 1:4
                if onset_time >= period_boundaries(p) && onset_time < period_boundaries(p+1)
                    period = p;
                    break;
                end
            end

            if period > 0
                reward_rearing_data.session_id(end+1) = n_valid_reward;
                reward_rearing_data.period(end+1) = period;
                reward_rearing_data.x_position(end+1) = x_pos;
                reward_rearing_data.y_position(end+1) = y_pos;
                reward_rearing_data.onset_time(end+1) = onset_time;

                % Update arena bounds
                arena_x_min = min(arena_x_min, x_pos);
                arena_x_max = max(arena_x_max, x_pos);
                arena_y_min = min(arena_y_min, y_pos);
                arena_y_max = max(arena_y_max, y_pos);
            end
        end
    end
end

fprintf('\n✓ Processed %d reward sessions\n', n_valid_reward);
fprintf('  Total rearing onsets: %d\n', length(reward_rearing_data.session_id));

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

% Initialize storage: [n_y_bins × n_x_bins × 4 periods × 2 session types]
spatial_counts = zeros(n_y_bins, n_x_bins, 4, 2);  % Raw counts
spatial_density = zeros(n_y_bins, n_x_bins, 4, 2);  % Normalized by occupancy

% Note: We don't have occupancy data per bin in this script
% So we'll show raw counts normalized by total onsets per period/session
% For true occupancy normalization, would need to track time spent in each bin

% Process aversive data
fprintf('Processing aversive spatial data...\n');
for i = 1:length(aversive_rearing_data.session_id)
    x = aversive_rearing_data.x_position(i);
    y = aversive_rearing_data.y_position(i);
    p = aversive_rearing_data.period(i);

    % Find bin indices
    x_bin = discretize(x, x_edges);
    y_bin = discretize(y, y_edges);

    if ~isnan(x_bin) && ~isnan(y_bin)
        spatial_counts(y_bin, x_bin, p, 1) = spatial_counts(y_bin, x_bin, p, 1) + 1;
    end
end

% Process reward data
fprintf('Processing reward spatial data...\n');
for i = 1:length(reward_rearing_data.session_id)
    x = reward_rearing_data.x_position(i);
    y = reward_rearing_data.y_position(i);
    p = reward_rearing_data.period(i);

    x_bin = discretize(x, x_edges);
    y_bin = discretize(y, y_edges);

    if ~isnan(x_bin) && ~isnan(y_bin)
        spatial_counts(y_bin, x_bin, p, 2) = spatial_counts(y_bin, x_bin, p, 2) + 1;
    end
end

% Normalize by total onsets per period/session
fprintf('Normalizing spatial maps...\n');
for session_type = 1:2
    for p = 1:4
        total_counts = sum(sum(spatial_counts(:, :, p, session_type)));
        if total_counts > 0
            spatial_density(:, :, p, session_type) = spatial_counts(:, :, p, session_type) / total_counts;
        end
    end
end

fprintf('✓ Spatial maps computed\n');

%% ========================================================================
%  SECTION 8: VISUALIZATION
%  ========================================================================

fprintf('\n==== CREATING VISUALIZATIONS ====\n');

% Create figure: 2 rows (aversive, reward) × 4 columns (periods)
figure('Position', [50, 50, 1600, 800], 'Name', 'Rearing Spatial Distribution');

session_type_names = {'Aversive', 'Reward'};
colors = {[1, 0, 0], [0, 0.6, 0]};  % Red for aversive, green for reward

for session_type = 1:2
    for p = 1:4
        subplot_idx = (session_type - 1) * 4 + p;
        subplot(2, 4, subplot_idx);

        % Get spatial map for this period/session
        spatial_map = spatial_density(:, :, p, session_type);

        % Plot as heatmap
        imagesc(x_edges(1:end-1) + config.spatial_bin_size/2, ...
                y_edges(1:end-1) + config.spatial_bin_size/2, ...
                spatial_map);

        colormap(hot);
        colorbar;
        axis equal tight;

        % Title
        title(sprintf('%s - Period %d', session_type_names{session_type}, p), ...
            'FontSize', 12, 'FontWeight', 'bold', 'Color', colors{session_type});

        xlabel('X position (cm)', 'FontSize', 10);
        ylabel('Y position (cm)', 'FontSize', 10);

        % Add onset count
        total_onsets = sum(sum(spatial_counts(:, :, p, session_type)));
        text(0.02, 0.98, sprintf('n=%d onsets', total_onsets), ...
            'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'BackgroundColor', 'w', 'FontSize', 9);
    end
end

sgtitle('Rearing Onset Spatial Distribution Across Time', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('✓ Visualization complete\n');

%% ========================================================================
%  SECTION 9: QUANTITATIVE METRICS
%  ========================================================================

fprintf('\n==== QUANTITATIVE METRICS ====\n');

% Calculate center of arena
arena_center_x = (arena_x_min + arena_x_max) / 2;
arena_center_y = (arena_y_min + arena_y_max) / 2;

fprintf('Arena center: (%.1f, %.1f) cm\n\n', arena_center_x, arena_center_y);

% For each session type and period, calculate:
% 1. Mean distance from center
% 2. Spatial entropy (dispersion)

for session_type = 1:2
    fprintf('%s sessions:\n', session_type_names{session_type});

    if session_type == 1
        data = aversive_rearing_data;
    else
        data = reward_rearing_data;
    end

    for p = 1:4
        period_mask = data.period == p;
        x_positions = data.x_position(period_mask);
        y_positions = data.y_position(period_mask);

        if ~isempty(x_positions)
            % Mean distance from center
            distances = sqrt((x_positions - arena_center_x).^2 + (y_positions - arena_center_y).^2);
            mean_distance = mean(distances);

            % Spatial entropy
            spatial_map = spatial_density(:, :, p, session_type);
            spatial_map_flat = spatial_map(:);
            spatial_map_flat = spatial_map_flat(spatial_map_flat > 0);  % Remove zeros
            spatial_entropy = -sum(spatial_map_flat .* log2(spatial_map_flat));

            fprintf('  P%d: Mean dist from center = %.1f cm, Spatial entropy = %.2f bits, n = %d onsets\n', ...
                p, mean_distance, spatial_entropy, sum(period_mask));
        else
            fprintf('  P%d: No data\n', p);
        end
    end
    fprintf('\n');
end

%% ========================================================================
%  SECTION 10: SUMMARY
%  ========================================================================

fprintf('========================================\n');
fprintf('ANALYSIS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Sessions processed:\n');
fprintf('  Aversive: %d sessions, %d rearing onsets\n', n_valid_aversive, length(aversive_rearing_data.session_id));
fprintf('  Reward: %d sessions, %d rearing onsets\n', n_valid_reward, length(reward_rearing_data.session_id));
fprintf('Spatial resolution: %d × %d bins (%.1f cm)\n', n_x_bins, n_y_bins, config.spatial_bin_size);
fprintf('========================================\n');
