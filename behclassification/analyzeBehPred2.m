%% Multi-Period Variance Analysis: Period 1 (Pure Reward) vs Periods 2-7 (After Each Aversive Noise)
% This script compares behavioral metrics across 7 periods divided by 6 aversive noises

clear all

%% Configuration
fprintf('=== MULTI-PERIOD ANALYSIS: 7 PERIODS DIVIDED BY 6 AVERSIVE NOISES ===\n');

confidence_threshold_dominant = 0.3;

%% Load data
fprintf('Loading data...\n');

try
%     coupling_data_before = load('RewardAversive_session_metrics_breathing_LFPCcouple(10-1).mat');
%     sessions_before = coupling_data_before.all_session_metrics;
%     pred_data = load('lstm_prediction_results_aversive.mat');
%     prediction_sessions = pred_data.final_results.session_predictions;
%     fprintf('✓ Loaded data: %d sessions\n', length(sessions_before));

    coupling_data_before = load('27-Oct-2025_RewardAversive_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_before = coupling_data_before.all_session_metrics;
    pred_data = load('lstm_prediction_results_aversive_27-Oct-2025');
    prediction_sessions = pred_data.final_results.session_predictions;
    fprintf('✓ Loaded data: %d sessions\n', length(sessions_before));
catch ME
    fprintf('❌ Failed to load data: %s\n', ME.message);
    return;
end

%% find aversive noise onset time
% all_verisve_time = [];
% for i = 1:length(coupling_data_before.all_session_metrics)
%     all_verisve_time = [all_verisve_time, coupling_data_before.all_session_metrics{i}.all_aversive_time];
% end
% 
% mean(all_verisve_time,2)./60;
% 
% ans =
%     8.5580
%    16.4116
%    24.2626
%    32.0810
%    39.9190
%    47.7272
%% Define animal colors (14 sessions = 7 animals, 2 sessions each)
n_animals = 12;
animal_colors = lines(n_animals);  % Generate 7 distinct colors

% Map sessions to animals (sessions 1,2 -> animal 1; sessions 3,4 -> animal 2, etc.)
session_to_animal = zeros(24, 1);
for i = 1:24
    session_to_animal(i) = ceil(i/2);
end

%% Extract data for all 7 periods
fprintf('Extracting data for 7 periods...\n');
all_period_data = cell(7, 1);

for period = 1:7
    fprintf('Processing period %d...\n', period);
    [all_period_data{period}, ~] = extract_period_data_by_aversive(sessions_before, prediction_sessions, period);
end

%% Define variables
behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', 'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
n_behaviors = 7;
period_names = {'Period 1 (Pure Reward)', 'Period 2 (After Noise 1)', 'Period 3 (After Noise 2)', ...
                'Period 4 (After Noise 3)', 'Period 5 (After Noise 4)', 'Period 6 (After Noise 5)', ...
                'Period 7 (After Noise 6)'};

% Define colors for 7 periods (for median lines)
period_colors = [0.1 0.4 0.7;   % Period 1 - Blue
                 0.9 0.3 0.2;   % Period 2 - Red
                 0.9 0.6 0.1;   % Period 3 - Orange
                 0.5 0.8 0.2;   % Period 4 - Green
                 0.7 0.2 0.7;   % Period 5 - Purple
                 0.2 0.7 0.7;   % Period 6 - Cyan
                 0.9 0.5 0.6];  % Period 7 - Pink

%% FIGURE 1: Behavioral Percentage Comparison Across All Periods
fprintf('Creating behavioral percentage comparison across all periods...\n');

% Calculate session-level behavioral percentages for all periods
session_behavior_all = cell(7, 1);
unique_sessions_all = cell(7, 1);

for period = 1:7
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
    all_medians = zeros(7, 1);
    all_q25 = zeros(7, 1);
    all_q75 = zeros(7, 1);
    
    for period = 1:7
        behavior_data = session_behavior_all{period}(:, beh_idx);
        all_medians(period) = median(behavior_data, 'omitnan');
        all_q25(period) = prctile(behavior_data, 25);
        all_q75(period) = prctile(behavior_data, 75);
    end
    
    % Plot with error bars (median lines)
    x_pos = 1:7;
    err_low = all_medians - all_q25;
    err_high = all_q75 - all_medians;
    
    for period = 1:7
        errorbar(x_pos(period), all_medians(period), err_low(period), err_high(period), ...
            'o', 'MarkerSize', 8, 'MarkerFaceColor', period_colors(period, :), ...
            'Color', period_colors(period, :), 'LineWidth', 2);
    end
    
    % Add individual session points with animal-specific colors
    for period = 1:7
        behavior_data = session_behavior_all{period}(:, beh_idx);
        sessions_in_period = unique_sessions_all{period};
        
        for sess_i = 1:length(sessions_in_period)
            if ~isnan(behavior_data(sess_i))
                session_id = sessions_in_period(sess_i);
                animal_id = session_to_animal(session_id);
                
                % Add jitter for visibility
                jitter = (rand - 0.5) * 0.15;
                
                scatter(period + jitter, behavior_data(sess_i), 40, ...
                    animal_colors(animal_id, :), 'filled', 'MarkerFaceAlpha', 0.7);
            end
        end
    end
    
    % Add connecting lines between same animals across periods
    for animal_id = 1:n_animals
        % Get sessions for this animal
        animal_sessions = find(session_to_animal == animal_id);
        
        % For each session of this animal, connect across periods
        for sess = animal_sessions'
            period_values = nan(7, 1);
            period_positions = nan(7, 1);
            
            for period = 1:7
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
                    '-', 'Color', [animal_colors(animal_id, :), 0.3], 'LineWidth', 1);
            end
        end
    end
    
    % Statistical comparison: Period 1 vs each other period
    p_values = zeros(6, 1);
    for period = 2:7
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
    for period = 2:7
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
                    'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');
            end
        end
    end
    
    ylabel('Percentage (%)');
    xlabel('Period');
    title(behavior_names{beh_idx});
    set(gca, 'XTick', 1:7, 'XTickLabel', {'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'});
    grid on;
    ylim([0, y_max + 3]);
end

sgtitle('Behavioral Percentages Across 7 Periods (Median ± IQR)', 'FontSize', 14, 'FontWeight', 'bold');

% Add legend for animals
legend_entries = cell(n_animals, 1);
for i = 1:n_animals
    legend_entries{i} = sprintf('Animal %d', i);
end

%% FIGURE 2: Behavioral Duration Comparison Across All Periods
fprintf('Creating behavioral duration comparison across all periods...\n');

Fs = 1;

% Calculate session-level average bout durations for all periods
session_duration_all = cell(7, 1);

for period = 1:7
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
    all_medians = zeros(7, 1);
    all_q25 = zeros(7, 1);
    all_q75 = zeros(7, 1);
    
    for period = 1:7
        duration_data = session_duration_all{period}(:, beh_idx);
        all_medians(period) = median(duration_data, 'omitnan');
        all_q25(period) = prctile(duration_data, 25);
        all_q75(period) = prctile(duration_data, 75);
    end
    
    % Plot with error bars (median lines)
    x_pos = 1:7;
    err_low = all_medians - all_q25;
    err_high = all_q75 - all_medians;
    
    for period = 1:7
        errorbar(x_pos(period), all_medians(period), err_low(period), err_high(period), ...
            'o', 'MarkerSize', 8, 'MarkerFaceColor', period_colors(period, :), ...
            'Color', period_colors(period, :), 'LineWidth', 2);
    end
    
    % Add individual session points with animal-specific colors
    for period = 1:7
        duration_data = session_duration_all{period}(:, beh_idx);
        sessions_in_period = unique_sessions_all{period};
        
        for sess_i = 1:length(sessions_in_period)
            if ~isnan(duration_data(sess_i))
                session_id = sessions_in_period(sess_i);
                animal_id = session_to_animal(session_id);
                
                % Add jitter for visibility
                jitter = (rand - 0.5) * 0.15;
                
                scatter(period + jitter, duration_data(sess_i), 40, ...
                    animal_colors(animal_id, :), 'filled', 'MarkerFaceAlpha', 0.7);
            end
        end
    end
    
    % Add connecting lines between same animals across periods
    for animal_id = 1:n_animals
        % Get sessions for this animal
        animal_sessions = find(session_to_animal == animal_id);
        
        % For each session of this animal, connect across periods
        for sess = animal_sessions'
            period_values = nan(7, 1);
            period_positions = nan(7, 1);
            
            for period = 1:7
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
                    '-', 'Color', [animal_colors(animal_id, :), 0.3], 'LineWidth', 1);
            end
        end
    end
    
    % Statistical comparison: Period 1 vs each other period
    p_values = zeros(6, 1);
    for period = 2:7
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
    for period = 2:7
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
                    'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');
            end
        end
    end
    
    ylabel('Duration (s)');
    xlabel('Period');
    title(behavior_names{beh_idx});
    set(gca, 'XTick', 1:7, 'XTickLabel', {'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'});
    grid on;
    ylim([0, y_max + 1]);
end

sgtitle('Average Bout Durations Across 7 Periods (Median ± IQR)', 'FontSize', 14, 'FontWeight', 'bold');

%% Print Summary Statistics
fprintf('\n=== SUMMARY STATISTICS ===\n');
fprintf('\nBehavioral Percentage - Period 1 vs Other Periods:\n');
for beh_idx = 1:n_behaviors
    fprintf('\n%s:\n', behavior_names{beh_idx});
    fprintf('Period\tMedian\tp-value (vs P1)\n');
    
    for period = 1:7
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

%% Helper Function: Extract period data based on aversive noise timing
function [extracted_data, baseline_info] = extract_period_data_by_aversive(sessions, prediction_sessions, period_num)
    % period_num: 1 = before first noise, 2-7 = after each of the 6 noises
    
    n_sessions = length(sessions);
    reward_baselines = struct('mean', cell(n_sessions, 1), 'std', cell(n_sessions, 1), 'valid', cell(n_sessions, 1));
    
    % First pass: Calculate baselines (reuse existing logic)
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
    
    % Second pass: Extract data for specified period
    all_data_cell = cell(n_sessions, 1);
    
    for i = 1:n_sessions
        session = sessions{i};
        
        if ~reward_baselines(i).valid || ...
           ~isfield(session, 'all_aversive_time') || ...
           ~isfield(session, 'behavioral_matrix_full') || ...
           ~isfield(session, 'coupling_results_multiband') || ...
           ~isfield(session, 'TriggerMid')
            continue;
        end
        
        % Determine period boundaries based on aversive noise times
        aversive_times = session.all_aversive_time;
        neural_time = session.NeuralTime;
        
        if period_num == 1
            % Period 1: From start to first aversive noise
            period_start = neural_time(1);
            period_end = aversive_times(1);
        elseif period_num <= length(aversive_times)
            % Periods 2-6: Between consecutive noises
            period_start = aversive_times(period_num - 1);
            if period_num < length(aversive_times) + 1
                period_end = aversive_times(period_num);
            else
                period_end = neural_time(end);
            end
        else
            % Period 7: After last noise
            period_start = aversive_times(end);
            period_end = neural_time(end);
        end
        
        % Find indices for this period
        period_indices = find(neural_time >= period_start & neural_time <= period_end);
        
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
            
            % Check if window is within the period
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