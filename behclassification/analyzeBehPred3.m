%% Multi-Period Variance Analysis: Reward-Only Sessions Divided by Time
% This script compares behavioral metrics across 4 periods divided by time intervals

clear all

%% Configuration
fprintf('=== REWARD-ONLY ANALYSIS: 4 TIME PERIODS (0-8, 8-16, 16-24, 24-30 mins) ===\n');

confidence_threshold_dominant = 0.3;

%% Load data
fprintf('Loading data...\n');

try
%     coupling_data_before = load('RewardSeeking_session_metrics_breathing_LFPCcouple(10-1).mat');
%     sessions_before = coupling_data_before.all_session_metrics;
%     pred_data = load('lstm_prediction_results_reward.mat');
%     prediction_sessions = pred_data.final_results.session_predictions;
%     fprintf('✓ Loaded data: %d sessions\n', length(sessions_before));

    coupling_data_before = load('27-Oct-2025_RewardSeeking_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_before = coupling_data_before.all_session_metrics;
    pred_data = load('lstm_prediction_results_reward_27-Oct-2025');
    prediction_sessions = pred_data.final_results.session_predictions;
    fprintf('✓ Loaded data: %d sessions\n', length(sessions_before));
catch ME
    fprintf('❌ Failed to load data: %s\n', ME.message);
    return;
end

%% Define animal colors (8 sessions = 4 animals, 2 sessions each)
n_animals = 9;
animal_colors = lines(n_animals);  % Generate 4 distinct colors

% Map sessions to animals (sessions 1,2 -> animal 1; sessions 3,4 -> animal 2, etc.)
session_to_animal = zeros(18, 1);
for i = 1:18
    session_to_animal(i) = ceil(i/2);
end

%% Define time boundaries (in seconds)
time_boundaries = [0, 8*60, 16*60, 24*60, 30*60];  % [0, 480, 960, 1440, 1800] seconds

%% Extract data for all 4 periods
fprintf('Extracting data for 4 time periods...\n');
all_period_data = cell(4, 1);

for period = 1:4
    fprintf('Processing period %d (%.1f-%.1f mins)...\n', period, ...
        time_boundaries(period)/60, time_boundaries(period+1)/60);
    [all_period_data{period}, ~] = extract_period_data_by_time(sessions_before, prediction_sessions, ...
        time_boundaries(period), time_boundaries(period+1));
end

%% Define variables
behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', 'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
n_behaviors = 7;
period_names = {'Period 1 (0-8 min)', 'Period 2 (8-16 min)', ...
                'Period 3 (16-24 min)', 'Period 4 (24-30 min)'};

% Define colors for 4 periods (for median lines)
period_colors = [0.1 0.4 0.7;   % Period 1 - Blue
                 0.5 0.8 0.2;   % Period 2 - Green
                 0.9 0.6 0.1;   % Period 3 - Orange
                 0.9 0.3 0.2];  % Period 4 - Red

%% FIGURE 1: Behavioral Percentage Comparison Across All Periods
fprintf('Creating behavioral percentage comparison across all periods...\n');

% Calculate session-level behavioral percentages for all periods
session_behavior_all = cell(4, 1);
unique_sessions_all = cell(4, 1);

for period = 1:4
    data_period = all_period_data{period};
    [~, ~, ~, predictions_period] = extract_variables(data_period);
    session_ids_period = [data_period.session_id];
    unique_sessions_all{period} = unique(session_ids_period);
    
    session_behavior_all{period} = zeros(length(unique_sessions_all{period}), n_behaviors);
    
    for i = 1:length(unique_sessions_all{period})
        sess_idx = session_ids_period == unique_sessions_all{period}(i);
        sess_predictions = predictions_period(sess_idx, :);
        
        if ~isempty(sess_predictions)
            [max_confidence, dominant_behavior] = max(sess_predictions, [], 2);
            valid_predictions = max_confidence > confidence_threshold_dominant;
            
            if sum(valid_predictions) > 0
                dominant_behavior_valid = dominant_behavior(valid_predictions);
                for beh_idx = 1:n_behaviors
                    behavior_count = sum(dominant_behavior_valid == beh_idx);
                    session_behavior_all{period}(i, beh_idx) = behavior_count / sum(valid_predictions) * 100;
                end
            end
        end
    end
end

% Create figure with subplots for each behavior
fig1 = figure('Position', [100, 100, 1800, 1000]);

for beh_idx = 1:n_behaviors
    subplot(3, 3, beh_idx);
    hold on;
    
    % Collect data for this behavior across all periods
    all_medians = zeros(4, 1);
    all_q25 = zeros(4, 1);
    all_q75 = zeros(4, 1);
    
    for period = 1:4
        behavior_data = session_behavior_all{period}(:, beh_idx);
        all_medians(period) = median(behavior_data, 'omitnan');
        all_q25(period) = prctile(behavior_data, 25);
        all_q75(period) = prctile(behavior_data, 75);
    end
    
    % Plot with error bars (median lines)
    x_pos = 1:4;
    err_low = all_medians - all_q25;
    err_high = all_q75 - all_medians;
    
    for period = 1:4
        errorbar(x_pos(period), all_medians(period), err_low(period), err_high(period), ...
            'o', 'MarkerSize', 10, 'MarkerFaceColor', period_colors(period, :), ...
            'Color', period_colors(period, :), 'LineWidth', 2.5);
    end
    
    % Add individual session points with animal-specific colors
    for period = 1:4
        behavior_data = session_behavior_all{period}(:, beh_idx);
        sessions_in_period = unique_sessions_all{period};
        
        for sess_i = 1:length(sessions_in_period)
            if ~isnan(behavior_data(sess_i))
                session_id = sessions_in_period(sess_i);
                animal_id = session_to_animal(session_id);
                
                % Add jitter for visibility
                jitter = (rand - 0.5) * 0.15;
                
                scatter(period + jitter, behavior_data(sess_i), 50, ...
                    animal_colors(animal_id, :), 'filled', 'MarkerFaceAlpha', 0.7, ...
                    'LineWidth', 1.5);
            end
        end
    end
    
    % Add connecting lines between same animals across periods
    for animal_id = 1:n_animals
        % Get sessions for this animal
        animal_sessions = find(session_to_animal == animal_id);
        
        % For each session of this animal, connect across periods
        for sess = animal_sessions'
            period_values = nan(4, 1);
            period_positions = nan(4, 1);
            
            for period = 1:4
                sessions_in_period = unique_sessions_all{period};
                sess_idx_in_period = find(sessions_in_period == sess);
                
                if ~isempty(sess_idx_in_period)
                    period_values(period) = session_behavior_all{period}(sess_idx_in_period, beh_idx);
                    period_positions(period) = period;
                end
            end
            
            % Plot connecting lines for this session
            valid_periods = ~isnan(period_values);
            if sum(valid_periods) > 1
                plot(period_positions(valid_periods), period_values(valid_periods), ...
                    '-', 'Color', [animal_colors(animal_id, :), 0.4], 'LineWidth', 1.5);
            end
        end
    end
    
    % Statistical comparison: Period 1 vs each other period
    p_values = zeros(3, 1);
    for period = 2:4
        data_p1 = session_behavior_all{1}(:, beh_idx);
        data_p = session_behavior_all{period}(:, beh_idx);
        
        % Find paired sessions
        valid_pairs = ~isnan(data_p1) & ~isnan(data_p);
        if sum(valid_pairs) >= 3
            [p_values(period-1), ~] = signrank(data_p1(valid_pairs), data_p(valid_pairs));
        else
            p_values(period-1) = NaN;
        end
    end
    
    % Add significance stars comparing to Period 1
    y_max = max(all_q75) + 2;
    for period = 2:4
        if ~isnan(p_values(period-1))
            if p_values(period-1) < 0.001
                star_text = '***';
            elseif p_values(period-1) < 0.01
                star_text = '**';
            elseif p_values(period-1) < 0.05
                star_text = '*';
            else
                star_text = '';
            end
            
            if ~isempty(star_text)
                text(period, y_max, star_text, 'HorizontalAlignment', 'center', ...
                    'FontSize', 14, 'FontWeight', 'bold', 'Color', 'red');
            end
        end
    end
    
    ylabel('Percentage (%)', 'FontSize', 11);
    xlabel('Time Period', 'FontSize', 11);
    title(behavior_names{beh_idx}, 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'XTick', 1:4, 'XTickLabel', {'0-8', '8-16', '16-24', '24-30'});
    grid on;
    ylim([0, y_max + 3]);
    set(gca, 'FontSize', 10);
end

sgtitle('Behavioral Percentages Across Time Periods - Reward-Only Sessions (Median ± IQR)', ...
    'FontSize', 16, 'FontWeight', 'bold');

%% FIGURE 2: Behavioral Duration Comparison Across All Periods
fprintf('Creating behavioral duration comparison across all periods...\n');

Fs = 1;

% Calculate session-level average bout durations for all periods
session_duration_all = cell(4, 1);

for period = 1:4
    data_period = all_period_data{period};
    [~, ~, ~, predictions_period] = extract_variables(data_period);
    session_ids_period = [data_period.session_id];
    unique_sessions_period = unique(session_ids_period);
    
    session_duration_all{period} = zeros(length(unique_sessions_period), n_behaviors);
    
    for i = 1:length(unique_sessions_period)
        sess_idx = session_ids_period == unique_sessions_period(i);
        sess_predictions = predictions_period(sess_idx, :);
        
        if ~isempty(sess_predictions)
            [max_confidence, dominant_behavior] = max(sess_predictions, [], 2);
            valid_predictions = max_confidence > confidence_threshold_dominant;
            
            dominant_behavior_double = double(dominant_behavior);
            dominant_behavior_double(~valid_predictions) = NaN;
            dominant_behavior_filled = fillmissing(dominant_behavior_double, 'nearest');
            
            for beh_idx = 1:n_behaviors
                beh_binary = (dominant_behavior_filled == beh_idx);
                onset = find(diff([0; beh_binary(:)]) == 1);
                offset = find(diff([beh_binary(:); 0]) == -1);
                
                if ~isempty(onset) && ~isempty(offset)
                    bout_durations = (offset - onset + 1) * Fs;
                    session_duration_all{period}(i, beh_idx) = mean(bout_durations);
                else
                    session_duration_all{period}(i, beh_idx) = 0;
                end
            end
        else
            session_duration_all{period}(i, :) = NaN;
        end
    end
end

% Create figure with subplots for each behavior
fig2 = figure('Position', [100, 100, 1800, 1000]);

for beh_idx = 1:n_behaviors
    subplot(3, 3, beh_idx);
    hold on;
    
    % Collect data for this behavior across all periods
    all_medians = zeros(4, 1);
    all_q25 = zeros(4, 1);
    all_q75 = zeros(4, 1);
    
    for period = 1:4
        duration_data = session_duration_all{period}(:, beh_idx);
        all_medians(period) = median(duration_data, 'omitnan');
        all_q25(period) = prctile(duration_data, 25);
        all_q75(period) = prctile(duration_data, 75);
    end
    
    % Plot with error bars (median lines)
    x_pos = 1:4;
    err_low = all_medians - all_q25;
    err_high = all_q75 - all_medians;
    
    for period = 1:4
        errorbar(x_pos(period), all_medians(period), err_low(period), err_high(period), ...
            'o', 'MarkerSize', 10, 'MarkerFaceColor', period_colors(period, :), ...
            'Color', period_colors(period, :), 'LineWidth', 2.5);
    end
    
    % Add individual session points with animal-specific colors
    for period = 1:4
        duration_data = session_duration_all{period}(:, beh_idx);
        sessions_in_period = unique_sessions_all{period};
        
        for sess_i = 1:length(sessions_in_period)
            if ~isnan(duration_data(sess_i))
                session_id = sessions_in_period(sess_i);
                animal_id = session_to_animal(session_id);
                
                % Add jitter for visibility
                jitter = (rand - 0.5) * 0.15;
                
                scatter(period + jitter, duration_data(sess_i), 50, ...
                    animal_colors(animal_id, :), 'filled', 'MarkerFaceAlpha', 0.7, ...
                    'LineWidth', 1.5);
            end
        end
    end
    
    % Add connecting lines between same animals across periods
    for animal_id = 1:n_animals
        % Get sessions for this animal
        animal_sessions = find(session_to_animal == animal_id);
        
        % For each session of this animal, connect across periods
        for sess = animal_sessions'
            period_values = nan(4, 1);
            period_positions = nan(4, 1);
            
            for period = 1:4
                sessions_in_period = unique_sessions_all{period};
                sess_idx_in_period = find(sessions_in_period == sess);
                
                if ~isempty(sess_idx_in_period)
                    period_values(period) = session_duration_all{period}(sess_idx_in_period, beh_idx);
                    period_positions(period) = period;
                end
            end
            
            % Plot connecting lines for this session
            valid_periods = ~isnan(period_values);
            if sum(valid_periods) > 1
                plot(period_positions(valid_periods), period_values(valid_periods), ...
                    '-', 'Color', [animal_colors(animal_id, :), 0.4], 'LineWidth', 1.5);
            end
        end
    end
    
    % Statistical comparison: Period 1 vs each other period
    p_values = zeros(3, 1);
    for period = 2:4
        data_p1 = session_duration_all{1}(:, beh_idx);
        data_p = session_duration_all{period}(:, beh_idx);
        
        valid_pairs = ~isnan(data_p1) & ~isnan(data_p);
        if sum(valid_pairs) >= 3
            [p_values(period-1), ~] = signrank(data_p1(valid_pairs), data_p(valid_pairs));
        else
            p_values(period-1) = NaN;
        end
    end
    
    % Add significance stars
    y_max = max(all_q75) + 0.5;
    for period = 2:4
        if ~isnan(p_values(period-1))
            if p_values(period-1) < 0.001
                star_text = '***';
            elseif p_values(period-1) < 0.01
                star_text = '**';
            elseif p_values(period-1) < 0.05
                star_text = '*';
            else
                star_text = '';
            end
            
            if ~isempty(star_text)
                text(period, y_max, star_text, 'HorizontalAlignment', 'center', ...
                    'FontSize', 14, 'FontWeight', 'bold', 'Color', 'red');
            end
        end
    end
    
    ylabel('Duration (s)', 'FontSize', 11);
    xlabel('Time Period', 'FontSize', 11);
    title(behavior_names{beh_idx}, 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'XTick', 1:4, 'XTickLabel', {'0-8', '8-16', '16-24', '24-30'});
    grid on;
    ylim([0, y_max + 1]);
    set(gca, 'FontSize', 10);
end

sgtitle('Average Bout Durations Across Time Periods - Reward-Only Sessions (Median ± IQR)', ...
    'FontSize', 16, 'FontWeight', 'bold');

%% Print Summary Statistics
fprintf('\n=== SUMMARY STATISTICS ===\n');
fprintf('\nBehavioral Percentage - Period 1 vs Other Periods:\n');
for beh_idx = 1:n_behaviors
    fprintf('\n%s:\n', behavior_names{beh_idx});
    fprintf('Period\tMedian\tp-value (vs P1)\n');
    
    for period = 1:4
        med = median(session_behavior_all{period}(:, beh_idx), 'omitnan');
        if period == 1
            fprintf('P%d\t%.2f\t-\n', period, med);
        else
            data_p1 = session_behavior_all{1}(:, beh_idx);
            data_p = session_behavior_all{period}(:, beh_idx);
            valid_pairs = ~isnan(data_p1) & ~isnan(data_p);
            if sum(valid_pairs) >= 3
                p_val = signrank(data_p1(valid_pairs), data_p(valid_pairs));
                fprintf('P%d\t%.2f\t%.4f\n', period, med, p_val);
            else
                fprintf('P%d\t%.2f\tN/A\n', period, med);
            end
        end
    end
end

fprintf('\nBehavioral Duration - Period 1 vs Other Periods:\n');
for beh_idx = 1:n_behaviors
    fprintf('\n%s:\n', behavior_names{beh_idx});
    fprintf('Period\tMedian (s)\tp-value (vs P1)\n');
    
    for period = 1:4
        med = median(session_duration_all{period}(:, beh_idx), 'omitnan');
        if period == 1
            fprintf('P%d\t%.2f\t\t-\n', period, med);
        else
            data_p1 = session_duration_all{1}(:, beh_idx);
            data_p = session_duration_all{period}(:, beh_idx);
            valid_pairs = ~isnan(data_p1) & ~isnan(data_p);
            if sum(valid_pairs) >= 3
                p_val = signrank(data_p1(valid_pairs), data_p(valid_pairs));
                fprintf('P%d\t%.2f\t\t%.4f\n', period, med, p_val);
            else
                fprintf('P%d\t%.2f\t\tN/A\n', period, med);
            end
        end
    end
end

%% Helper Function: Extract period data based on time boundaries
function [extracted_data, baseline_info] = extract_period_data_by_time(sessions, prediction_sessions, time_start, time_end)
    % time_start, time_end: boundaries in seconds
    
    n_sessions = length(sessions);
    reward_baselines = struct('mean', cell(n_sessions, 1), 'std', cell(n_sessions, 1), 'valid', cell(n_sessions, 1));
    
    % First pass: Calculate baselines
    for i = 1:n_sessions
        session = sessions{i};
        reward_baselines(i).valid = false;
        
        if ~isfield(session, 'behavioral_matrix_full') || ~isfield(session, 'coupling_results_multiband')
            continue;
        end
        
        behavioral_matrix = session.behavioral_matrix_full;
        coupling_data_session = session.coupling_results_multiband;
        
        if isempty(coupling_data_session) || isempty(coupling_data_session.summary)
            continue;
        end
        
        coupling_time = coupling_data_session.summary.window_times;
        %         coupling_MI = coupling_data_session.summary.all_MI_values(4, :);
        coupling_MI = coupling_data_session.summary.all_MI_values;

        neural_time = session.NeuralTime;
        
        if size(behavioral_matrix, 2) >= 1
            reward_periods = behavioral_matrix(:, 1) == 1;
        else
            reward_periods = true(size(neural_time));
        end
        
        try
            coupling_neural = interp1(coupling_time, coupling_MI, neural_time, 'linear', 'extrap');
            reward_coupling = coupling_neural(reward_periods);
            
            mu = mean(reward_coupling);
            sigma = std(reward_coupling);
            reward_coupling_clean = reward_coupling(abs(reward_coupling - mu) < 3 * sigma);
            
            if length(reward_coupling_clean) > 10
                baseline_mean = mean(reward_coupling_clean);
                baseline_std = std(reward_coupling_clean);
                if baseline_std == 0, baseline_std = 0.001; end
                
                reward_baselines(i).mean = baseline_mean;
                reward_baselines(i).std = baseline_std;
                reward_baselines(i).valid = true;
            end
        catch
        end
    end
    
    % Second pass: Extract data for specified time period
    all_data_cell = cell(n_sessions, 1);
    
    for i = 1:n_sessions
        session = sessions{i};
        
        if ~reward_baselines(i).valid || ...
           ~isfield(session, 'behavioral_matrix_full') || ...
           ~isfield(session, 'coupling_results_multiband') || ...
           ~isfield(session, 'TriggerMid')
            continue;
        end
        
        neural_time = session.NeuralTime;
        
        % Find indices for this time period
        period_indices = find(neural_time >= time_start & neural_time <= time_end);
        
        if isempty(period_indices)
            continue;
        end
        
        % Extract data using existing logic
        behavioral_matrix = session.behavioral_matrix_full;
        speed = session.Speed;
        camera_time = session.TriggerMid;
        coupling_data_session = session.coupling_results_multiband;
        
        if i <= length(prediction_sessions) && isfield(prediction_sessions(1), 'prediction_scores')
            prediction_scores = prediction_sessions(i).prediction_scores;
        else
            continue;
        end
        
        coupling_time = coupling_data_session.summary.window_times;
        %         coupling_MI = coupling_data_session.summary.all_MI_values(4, :);
        coupling_MI = coupling_data_session.summary.all_MI_values;
        
        try
            speed_camera = interp1(neural_time, speed, camera_time, 'linear', 'extrap');
            
            if size(behavioral_matrix, 2) >= 8
                breathing_neural = behavioral_matrix(:, 8);
                breathing_camera = interp1(neural_time, breathing_neural, camera_time, 'linear', 'extrap');
            else
                continue;
            end
            
            coupling_camera_raw = interp1(coupling_time, coupling_MI, camera_time, 'linear', 'extrap');
            baseline_mean = reward_baselines(i).mean;
            baseline_std = reward_baselines(i).std;
            coupling_camera_normalized = (coupling_camera_raw - baseline_mean) / baseline_std;
            
        catch
            continue;
        end
        
        prediction_indices = 1:20:length(camera_time)+1;
        
        session_data = struct(...
            'session_id', [], 'prediction_id', [], 'camera_time', [], ...
            'speed_median', [], 'breathing_median', [], ...
            'coupling_median_normalized', [], 'coupling_median_raw', [], ...
            'prediction_scores', [], 'reward_baseline_mean', [], ...
            'reward_baseline_std', [], 'is_in_temporal_overlap', []);
        
        session_data = repmat(session_data, size(prediction_scores, 1), 1);
        
        valid_count = 0;
        for p = 1:length(prediction_indices)-1
            window_start = prediction_indices(p);
            window_end = prediction_indices(p+1)-1;
            
            % Check if window is within the time period
            if camera_time(window_start) < neural_time(period_indices(1)) || ...
               camera_time(window_start) > neural_time(period_indices(end))
                continue;
            end
            
            valid_count = valid_count + 1;
            
            speed_window = speed_camera(window_start:window_end);
            breathing_window = breathing_camera(window_start:window_end);
            coupling_norm_window = coupling_camera_normalized(window_start:window_end);
            coupling_raw_window = coupling_camera_raw(window_start:window_end);
            
            session_data(valid_count).session_id = i;
            session_data(valid_count).prediction_id = p;
            session_data(valid_count).camera_time = camera_time(prediction_indices(p));
            session_data(valid_count).speed_median = median(speed_window, 'omitnan');
            session_data(valid_count).breathing_median = median(breathing_window, 'omitnan');
            session_data(valid_count).coupling_median_normalized = median(coupling_norm_window, 'omitnan');
            session_data(valid_count).coupling_median_raw = median(coupling_raw_window, 'omitnan');
            session_data(valid_count).prediction_scores = prediction_scores(p, :);
            session_data(valid_count).reward_baseline_mean = baseline_mean;
            session_data(valid_count).reward_baseline_std = baseline_std;
        end
        
        if valid_count > 0
            all_data_cell{i} = session_data(1:valid_count);
        end
    end
    
    all_data_cell = all_data_cell(~cellfun(@isempty, all_data_cell));
    
    if ~isempty(all_data_cell)
        extracted_data = vertcat(all_data_cell{:});
    else
        extracted_data = [];
    end
    
    baseline_info = reward_baselines;
end

% Reuse existing helper function
function [speed, coupling, breathing, predictions] = extract_variables(data)
    if isempty(data)
        speed = []; coupling = []; breathing = []; predictions = [];
        return;
    end
    speed = [data.speed_median]';
    coupling = [data.coupling_median_normalized]';
    breathing = [data.breathing_median]';
    predictions = vertcat(data.prediction_scores);
end