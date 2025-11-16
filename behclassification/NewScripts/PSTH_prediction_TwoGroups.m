%% Aversive Noise Response Group Analysis with Behavioral PSTH
% This script:
% 1. Classifies animals into two groups based on Goal-Directed Movement response
%    to the first aversive noise (responders vs non-responders)
% 2. Creates PSTH of 7 behavioral types aligned to each aversive noise
% 3. Compares behavioral changes between groups

clear all
% close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== AVERSIVE RESPONSE GROUP ANALYSIS ===\n\n');

% Analysis parameters
config = struct();
config.confidence_threshold = 0.3;           % Threshold for dominant behavior
config.goal_movement_column = 7;             % Column for Goal-Directed Movement
config.psth_time_window = [-60, 300];        % Time window around aversive noise (seconds)
config.psth_bin_size = 5;                    % PSTH bin size (seconds)
config.significance_alpha = 0.05;            % Statistical significance level

% Behavioral definitions
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
config.n_behaviors = 7;

% Define colors
config.group_colors = struct('responder', [0.9 0.3 0.2], ...  % Red for responders
                             'non_responder', [0.1 0.4 0.7]);  % Blue for non-responders

%% ========================================================================
%  SECTION 2: LOAD DATA
%  ========================================================================

fprintf('Loading data...\n');

% Configuration
prediction_folder = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/BehaviorPrediction';
spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';

% Load sorting parameters
[T_sorted] = loadSortingParameters();

try
    % Select spike files
    [allfiles_aversive, ~, ~, ~] = selectFilesWithAnimalIDFiltering(spike_folder, 999, '2025*RewardAversive*.mat');

    % Calculate behavioral matrices from spike files
    sessions = loadSessionMetricsFromSpikeFiles(allfiles_aversive, T_sorted);

    % Load predictions
    prediction_sessions = loadBehaviorPredictionsFromSpikeFiles(allfiles_aversive, prediction_folder);

    fprintf('✓ Loaded %d sessions\n\n', length(sessions));
catch ME
    fprintf('❌ Failed to load data: %s\n', ME.message);
    return;
end

%% ========================================================================
%  SECTION 3: CLASSIFY ANIMALS INTO RESPONSE GROUPS
%  ========================================================================

fprintf('Classifying animals into responder/non-responder groups...\n');

% Calculate Goal-Directed Movement change for each session
n_sessions = length(sessions);
goal_movement_change = nan(n_sessions, 1);

for sess_idx = 1:n_sessions
    session = sessions{sess_idx};
    
    % Check required fields
    if ~isfield(session, 'behavioral_matrix_full') || ...
       ~isfield(session, 'all_aversive_time') || ...
       ~isfield(session, 'NeuralTime')
        continue;
    end
    
    behavioral_matrix = session.behavioral_matrix_full;
    aversive_times = session.all_aversive_time;
    neural_time = session.NeuralTime;
    
    % Period 1: Before first aversive noise
    period1_idx = neural_time < aversive_times(1);
    
    % Period 2: After first aversive noise (until second noise or end)
    if length(aversive_times) > 1
        period2_idx = neural_time >= aversive_times(1) & neural_time < aversive_times(2);
    else
        period2_idx = neural_time >= aversive_times(1);
    end
    
    % Calculate frequency of Goal-Directed Movement
    goal_movement = behavioral_matrix(:, config.goal_movement_column);
    freq_before = (sum(goal_movement(period1_idx)) / sum(period1_idx)) * 100;
    freq_after = (sum(goal_movement(period2_idx)) / sum(period2_idx)) * 100;
    
    % Change in Goal-Directed Movement (negative = decrease)
    goal_movement_change(sess_idx) = freq_after - freq_before;
end

% Classify groups based on significant drop in Goal-Directed Movement
% Use statistical test: sessions with significant decrease are "responders"
group_assignment = struct();
group_assignment.session_ids = (1:n_sessions)';
group_assignment.goal_movement_change = goal_movement_change;
group_assignment.is_responder = goal_movement_change < -5;  % >0% decrease threshold

% Summary
n_responders = sum(group_assignment.is_responder);
n_non_responders = sum(~group_assignment.is_responder & ~isnan(goal_movement_change));

fprintf('✓ Group classification complete:\n');
fprintf('  - Responders (decreased Goal-Directed Movement): %d sessions\n', n_responders);
fprintf('  - Non-responders: %d sessions\n\n', n_non_responders);

%% ========================================================================
%  SECTION 4: EXTRACT BEHAVIORAL DATA AROUND EACH AVERSIVE NOISE
%  ========================================================================

fprintf('Extracting behavioral data aligned to aversive noises...\n');

% Initialize storage for PSTH data
psth_data = struct();
psth_data.responder = [];
psth_data.non_responder = [];

for sess_idx = 1:n_sessions
    session = sessions{sess_idx};
    
    % Skip if session not properly classified
    if isnan(goal_movement_change(sess_idx))
        continue;
    end
    
    % Check required fields
    if ~isfield(session, 'behavioral_matrix_full') || ...
       ~isfield(session, 'all_aversive_time') || ...
       ~isfield(session, 'NeuralTime') || ...
       sess_idx > length(prediction_sessions) || ...
       ~isfield(prediction_sessions, 'prediction_scores')
        continue;
    end
    
    behavioral_matrix = session.behavioral_matrix_full;
    aversive_times = session.all_aversive_time;
    neural_time = session.NeuralTime;
    prediction_scores = prediction_sessions(sess_idx).prediction_scores;
    camera_time = session.TriggerMid;
    
    % Determine group
    if group_assignment.is_responder(sess_idx)
        group_name = 'responder';
    else
        group_name = 'non_responder';
    end
    
    % Extract data around each aversive noise
    for noise_idx = 1:length(aversive_times)
        noise_time = aversive_times(noise_idx);
        
        % Find time window around this aversive noise
        time_window = [noise_time + config.psth_time_window(1), ...
                       noise_time + config.psth_time_window(2)];
        
        % Get camera frames in this window
        camera_idx = camera_time >= time_window(1) & camera_time <= time_window(2);
        
        if sum(camera_idx) < 10  % Need at least 10 frames
            continue;
        end
        
        % Extract prediction scores for this window
        % Predictions are at 1 Hz, every 20 frames
        prediction_indices = 1:20:length(camera_time)+1;
        
        for pred_idx = 1:(length(prediction_indices)-1)
            frame_start = prediction_indices(pred_idx);
            
            % Check if this prediction is in the time window
            if camera_time(frame_start) < time_window(1) || ...
               camera_time(frame_start) > time_window(2)
                continue;
            end
            
            % Time relative to aversive noise
            rel_time = camera_time(frame_start) - noise_time;
            
            % Get dominant behavior
            [max_conf, dominant_beh] = max(prediction_scores(pred_idx, :));
            
            if max_conf > config.confidence_threshold
                % Store this data point
                data_point = struct();
                data_point.session_id = sess_idx;
                data_point.noise_number = noise_idx;
                data_point.relative_time = rel_time;
                data_point.behavior = dominant_beh;
                data_point.confidence = max_conf;
                
                % Add to appropriate group
                psth_data.(group_name) = [psth_data.(group_name); data_point];
            end
        end
    end
end

fprintf('✓ Data extraction complete\n');
fprintf('  - Responder data points: %d\n', length(psth_data.responder));
fprintf('  - Non-responder data points: %d\n\n', length(psth_data.non_responder));

%% ========================================================================
%  SECTION 5: CREATE PSTH FOR EACH BEHAVIORAL TYPE
%  ========================================================================

fprintf('Creating behavioral PSTH...\n');

% Define time bins
time_bins = config.psth_time_window(1):config.psth_bin_size:config.psth_time_window(2);
n_bins = length(time_bins) - 1;
bin_centers = time_bins(1:end-1) + config.psth_bin_size/2;

% Calculate PSTH for each group and behavior
psth_responder = zeros(config.n_behaviors, n_bins);
psth_non_responder = zeros(config.n_behaviors, n_bins);

% Responders
if ~isempty(psth_data.responder)
    for beh = 1:config.n_behaviors
        beh_data = psth_data.responder([psth_data.responder.behavior] == beh);
        
        for bin_idx = 1:n_bins
            bin_start = time_bins(bin_idx);
            bin_end = time_bins(bin_idx + 1);
            
            count = sum([beh_data.relative_time] >= bin_start & ...
                       [beh_data.relative_time] < bin_end);
            psth_responder(beh, bin_idx) = count;
        end
    end
    
    % Normalize to percentage (per bin)
    psth_responder = psth_responder ./ sum(psth_responder, 1) * 100;
end

% Non-responders
if ~isempty(psth_data.non_responder)
    for beh = 1:config.n_behaviors
        beh_data = psth_data.non_responder([psth_data.non_responder.behavior] == beh);
        
        for bin_idx = 1:n_bins
            bin_start = time_bins(bin_idx);
            bin_end = time_bins(bin_idx + 1);
            
            count = sum([beh_data.relative_time] >= bin_start & ...
                       [beh_data.relative_time] < bin_end);
            psth_non_responder(beh, bin_idx) = count;
        end
    end
    
    % Normalize to percentage (per bin)
    psth_non_responder = psth_non_responder ./ sum(psth_non_responder, 1) * 100;
end

fprintf('✓ PSTH calculation complete\n\n');

%% ========================================================================
%  SECTION 6: VISUALIZATION - PSTH COMPARISON
%  ========================================================================

fprintf('Creating PSTH visualizations...\n');

% Create figure with subplots for each behavior
fig1 = figure('Position', [50, 50, 1800, 1000]);

for beh = 1:config.n_behaviors
    subplot(3, 3, beh);
    hold on;
    
    % Plot PSTH for both groups
    plot(bin_centers, psth_responder(beh, :), 'LineWidth', 2.5, ...
         'Color', config.group_colors.responder, 'DisplayName', 'Responders');
    plot(bin_centers, psth_non_responder(beh, :), 'LineWidth', 2.5, ...
         'Color', config.group_colors.non_responder, 'DisplayName', 'Non-responders');
    
    % Add vertical line at aversive noise onset
    xline(0, 'k--', 'LineWidth', 2, 'DisplayName', 'Aversive Noise');
    
    % Formatting
    xlabel('Time from Aversive Noise (s)', 'FontSize', 11);
    ylabel('Behavioral Frequency (%)', 'FontSize', 11);
    title(config.behavior_names{beh}, 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    xlim(config.psth_time_window);
    set(gca, 'FontSize', 10);
end

sgtitle('Behavioral PSTH Aligned to Aversive Noise: Responders vs Non-responders', ...
        'FontSize', 15, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 7: STATISTICAL COMPARISON
%  ========================================================================

fprintf('Performing statistical comparisons...\n\n');

% Compare pre vs post aversive noise for each group and behavior
pre_window = [-60, 0];   % Pre-aversive window
post_window = [0, 60];   % Post-aversive window

fprintf('=== STATISTICAL COMPARISONS: PRE vs POST AVERSIVE NOISE ===\n\n');

for beh = 1:config.n_behaviors
    fprintf('%s:\n', config.behavior_names{beh});
    
    % Responders
    if ~isempty(psth_data.responder)
        beh_data = psth_data.responder([psth_data.responder.behavior] == beh);
        
        pre_data = [beh_data([beh_data.relative_time] >= pre_window(1) & ...
                             [beh_data.relative_time] < pre_window(2)).relative_time];
        post_data = [beh_data([beh_data.relative_time] >= post_window(1) & ...
                              [beh_data.relative_time] < post_window(2)).relative_time];
        
        freq_pre = length(pre_data) / abs(pre_window(2) - pre_window(1));
        freq_post = length(post_data) / abs(post_window(2) - post_window(1));
        
        fprintf('  Responders:     Pre=%.2f/s, Post=%.2f/s, Change=%.1f%%\n', ...
                freq_pre, freq_post, ((freq_post - freq_pre) / freq_pre * 100));
    end
    
    % Non-responders
    if ~isempty(psth_data.non_responder)
        beh_data = psth_data.non_responder([psth_data.non_responder.behavior] == beh);
        
        pre_data = [beh_data([beh_data.relative_time] >= pre_window(1) & ...
                             [beh_data.relative_time] < pre_window(2)).relative_time];
        post_data = [beh_data([beh_data.relative_time] >= post_window(1) & ...
                              [beh_data.relative_time] < post_window(2)).relative_time];
        
        freq_pre = length(pre_data) / abs(pre_window(2) - pre_window(1));
        freq_post = length(post_data) / abs(post_window(2) - post_window(1));
        
        fprintf('  Non-responders: Pre=%.2f/s, Post=%.2f/s, Change=%.1f%%\n', ...
                freq_pre, freq_post, ((freq_post - freq_pre) / freq_pre * 100));
    end
    
    fprintf('\n');
end

%% ========================================================================
%  SECTION 8: SAVE RESULTS
%  ========================================================================

fprintf('Saving results...\n');

results = struct();
results.config = config;
results.group_assignment = group_assignment;
results.psth_data = psth_data;
results.psth_responder = psth_responder;
results.psth_non_responder = psth_non_responder;
results.bin_centers = bin_centers;

save('aversive_response_group_analysis_results.mat', 'results');

fprintf('✓ Results saved to: aversive_response_group_analysis_results.mat\n');
fprintf('\n=== ANALYSIS COMPLETE ===\n');

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================
% Add your custom analysis functions below this line
% This section is reserved for future extensions