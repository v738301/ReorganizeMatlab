%% Session-Level Variance Analysis: Before vs After First Noise
% This script properly accounts for inter-session variance and implements
% rigorous CDF comparison methods with confidence bands

clear all

%% Configuration
fprintf('=== SESSION-LEVEL VARIANCE ANALYSIS: BEFORE vs AFTER FIRST NOISE ===\n');

% Define analysis parameters
confidence_threshold = 0.8;
confidence_threshold_dominant = 0.3;

%% Load data
fprintf('Loading data for before and after conditions...\n');

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

%% Extract data for before and after periods
fprintf('Extracting Before and After Aversive data...\n');
[data_before, ~] = extract_period_data(sessions_before, prediction_sessions, 'before_indices');
[data_after, ~] = extract_period_data(sessions_before, prediction_sessions, 'after_indices');

%% Extract variables and organize by session
[speed_before, coupling_before, breathing_before, predictions_before] = extract_variables(data_before);
[speed_after, coupling_after, breathing_after, predictions_after] = extract_variables(data_after);

% Get session organization
session_ids_before = [data_before.session_id];
session_ids_after = [data_after.session_id];
unique_sessions_before = unique(session_ids_before);
unique_sessions_after = unique(session_ids_after);

fprintf('Unique sessions: Before=%d, After=%d\n', length(unique_sessions_before), length(unique_sessions_after));

%% Define variables
behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', 'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
n_behaviors = min(7, size(predictions_before, 2));
colors = struct('before', [0.1 0.4 0.7], 'after', [0.7 0.2 0.1]);
condition_names = {'Before First Noise', 'After First Noise'};

%% FIGURE 1: Session-Level Dominant Behavior Percentage Analysis (Single Plot)
fprintf('Creating session-level dominant behavior analysis...\n');

% Calculate session-level behavioral percentages
session_behavior_before = zeros(length(unique_sessions_before), n_behaviors);
session_behavior_after = zeros(length(unique_sessions_after), n_behaviors);

for i = 1:length(unique_sessions_before)
    sess_idx = session_ids_before == unique_sessions_before(i);
    sess_predictions = predictions_before(sess_idx, :);
    
    if ~isempty(sess_predictions)
        [max_confidence, dominant_behavior] = max(sess_predictions, [], 2);
        valid_predictions = max_confidence > confidence_threshold_dominant;
        
        if sum(valid_predictions) > 0
            dominant_behavior_valid = dominant_behavior(valid_predictions);
            for beh_idx = 1:n_behaviors
                behavior_count = sum(dominant_behavior_valid == beh_idx);
                session_behavior_before(i, beh_idx) = behavior_count / sum(valid_predictions) * 100;
            end
        end
    end
end

for i = 1:length(unique_sessions_after)
    sess_idx = session_ids_after == unique_sessions_after(i);
    sess_predictions = predictions_after(sess_idx, :);
    
    if ~isempty(sess_predictions)
        [max_confidence, dominant_behavior] = max(sess_predictions, [], 2);
        valid_predictions = max_confidence > confidence_threshold_dominant;
        
        if sum(valid_predictions) > 0
            dominant_behavior_valid = dominant_behavior(valid_predictions);
            for beh_idx = 1:n_behaviors
                behavior_count = sum(dominant_behavior_valid == beh_idx);
                session_behavior_after(i, beh_idx) = behavior_count / sum(valid_predictions) * 100;
            end
        end
    end
end

% Statistical analysis using non-parametric tests
fig = figure('Position', [100, 100, 1200, 600]);

% Calculate medians and IQR for plotting
session_medians_before = median(session_behavior_before, 1, 'omitnan');
session_medians_after = median(session_behavior_after, 1, 'omitnan');

% Calculate IQR for error bars
session_q25_before = prctile(session_behavior_before, 25, 1);
session_q75_before = prctile(session_behavior_before, 75, 1);
session_q25_after = prctile(session_behavior_after, 25, 1);
session_q75_after = prctile(session_behavior_after, 75, 1);

% For asymmetric error bars
session_err_low_before = session_medians_before - session_q25_before;
session_err_high_before = session_q75_before - session_medians_before;
session_err_low_after = session_medians_after - session_q25_after;
session_err_high_after = session_q75_after - session_medians_after;

session_p_values = zeros(1, n_behaviors);
session_effect_sizes = zeros(1, n_behaviors);

for beh_idx = 1:n_behaviors
    before_vals = session_behavior_before(:, beh_idx);
    after_vals = session_behavior_after(:, beh_idx);
    
    % Find paired sessions (both before and after have valid data)
    valid_pairs = ~isnan(before_vals) & ~isnan(after_vals);
    
    before_vals_paired = before_vals(valid_pairs);
    after_vals_paired = after_vals(valid_pairs);
    
    if length(before_vals_paired) >= 3
        % Wilcoxon Signed-Rank Test (paired non-parametric test)
        [session_p_values(beh_idx), ~, stats] = signrank(before_vals_paired, after_vals_paired);
        
        % Calculate matched-pairs rank-biserial correlation as effect size
        n_pairs = length(before_vals_paired);
        
        % Calculate differences
        differences = after_vals_paired - before_vals_paired;
        abs_diffs = abs(differences);
        ranks = tiedrank(abs_diffs);
        
        % Sum of positive ranks (where after > before)
        W_plus = sum(ranks(differences > 0));
        % Total possible sum of ranks
        W_max = n_pairs * (n_pairs + 1) / 2;
        
        % Matched-pairs rank-biserial correlation
        % r = (W+ / W_max) * 2 - 1
        % Range: -1 (all negative) to +1 (all positive)
        session_effect_sizes(beh_idx) = (W_plus / W_max) * 2 - 1;
        
    else
        session_p_values(beh_idx) = NaN;
        session_effect_sizes(beh_idx) = NaN;
    end
end

% Main plot: Session-level behavioral percentages
x_pos = 1:n_behaviors;
bar_width = 0.35;

% Plot error bars with median ± IQR (asymmetric)
errorbar(x_pos - bar_width/2, session_medians_before, session_err_low_before, session_err_high_before, ...
    'o', 'MarkerSize', 8, 'MarkerFaceColor', colors.before, 'Color', colors.before, 'LineWidth', 2);
hold on;
errorbar(x_pos + bar_width/2, session_medians_after, session_err_low_after, session_err_high_after, ...
    'o', 'MarkerSize', 8, 'MarkerFaceColor', colors.after, 'Color', colors.after, 'LineWidth', 2);

% Add individual session points with jitter
sessionColor = repelem(lines(size(session_behavior_before,1)./2),2,1);

for beh_idx = 1:n_behaviors
    before_vals = session_behavior_before(:, beh_idx);
    after_vals = session_behavior_after(:, beh_idx);
    
    before_vals_clean = before_vals(~isnan(before_vals));
    after_vals_clean = after_vals(~isnan(after_vals));
    
    jitter_before = (rand(size(before_vals_clean)) - 0.5) * 0.2;
    jitter_after = (rand(size(after_vals_clean)) - 0.5) * 0.2;
    
%     scatter(beh_idx - bar_width/2 + jitter_before, before_vals_clean, 25, colors.before, 'filled', 'MarkerFaceAlpha', 0.6);
%     scatter(beh_idx + bar_width/2 + jitter_after, after_vals_clean, 25, colors.after, 'filled', 'MarkerFaceAlpha', 0.6);
    
    for animalID = 1:length(before_vals_clean)
        scatter(beh_idx - bar_width/2 - 0.05, before_vals_clean(animalID), 25, sessionColor(animalID,:), 'filled', 'MarkerFaceAlpha', 0.6);
        scatter(beh_idx + bar_width/2 + 0.05, after_vals_clean(animalID), 25, sessionColor(animalID,:), 'filled', 'MarkerFaceAlpha', 0.6);
    end
end

% Add connecting lines between paired sessions
for beh_idx = 1:n_behaviors
    before_vals = session_behavior_before(:, beh_idx);
    after_vals = session_behavior_after(:, beh_idx);
    
    % Only connect sessions with data in both conditions
    valid_sessions = ~isnan(before_vals) & ~isnan(after_vals);
    
    if sum(valid_sessions) > 0
        before_vals_paired = before_vals(valid_sessions);
        after_vals_paired = after_vals(valid_sessions);
        
        for sess_i = 1:length(before_vals_paired)
            plot([beh_idx - bar_width/2, beh_idx + bar_width/2], ...
                 [before_vals_paired(sess_i), after_vals_paired(sess_i)], ...
                 'k-', 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5 0.7]);
        end
    end
end

% Add significance stars
y_max = max([session_q75_before; session_q75_after]);
for beh_idx = 1:n_behaviors
    if ~isnan(session_p_values(beh_idx))
        % Determine star marker based on p-value
        if session_p_values(beh_idx) < 0.001
            star_text = '***';
        elseif session_p_values(beh_idx) < 0.01
            star_text = '**';
        elseif session_p_values(beh_idx) < 0.05
            star_text = '*';
        else
            star_text = '';  % No star for non-significant
        end
        
        if ~isempty(star_text)
            % Position star above the higher error bar
            star_y = y_max(beh_idx) + 2;
            
            % Add horizontal line connecting the two conditions
            line_y = y_max(beh_idx) + 1;
            plot([beh_idx - bar_width/2, beh_idx + bar_width/2], [line_y, line_y], 'k-', 'LineWidth', 1);
            
            % Add star above the line
            text(beh_idx, star_y, star_text, 'HorizontalAlignment', 'center', ...
                'FontSize', 14, 'FontWeight', 'bold', 'Color', 'red');
        end
    end
end

ylabel('Dominant Behavior Percentage (%)');
xlabel('Behavior');
title('Session-Level Behavioral Percentages: Before vs After First Noise (Median ± IQR)');
set(gca, 'XTick', x_pos, 'XTickLabel', behavior_names, 'XTickLabelRotation', 45);
legend(condition_names, 'Location', 'best', 'FontSize', 10);
grid on;

% Adjust y-axis to accommodate stars
ylim([0, max(y_max) + 20]);

% Print statistics summary
fprintf('\nNon-parametric statistics (Signed-rank test):\n');
fprintf('Behavior\t\tMedian Before\tMedian After\tp-value\t\tEffect Size (r)\n');
fprintf('------------------------------------------------------------------------------------\n');
for beh_idx = 1:n_behaviors
    fprintf('%s\t\t%.2f\t\t%.2f\t\t%.4f\t\t%.3f\n', ...
        behavior_names{beh_idx}, session_medians_before(beh_idx), ...
        session_medians_after(beh_idx), session_p_values(beh_idx), ...
        session_effect_sizes(beh_idx));
end

%% FIGURE 2: Session-Level Dominant Behavior Duration Analysis (Single Plot)
fprintf('Creating session-level average behavior bout duration analysis...\n');

Fs = 1;

% Calculate session-level average behavioral bout durations (in seconds)
session_avg_duration_before = zeros(length(unique_sessions_before), n_behaviors);
session_avg_duration_after = zeros(length(unique_sessions_after), n_behaviors);

% Process BEFORE sessions
for i = 1:length(unique_sessions_before)
    sess_idx = session_ids_before == unique_sessions_before(i);
    sess_predictions = predictions_before(sess_idx, :);
    
    if ~isempty(sess_predictions)
        [max_confidence, dominant_behavior] = max(sess_predictions, [], 2);
        valid_predictions = max_confidence > confidence_threshold_dominant;
        
        % Mark invalid predictions as NaN
        dominant_behavior_double = double(dominant_behavior);
        dominant_behavior_double(~valid_predictions) = NaN;
        
        % Fill missing values with nearest neighbor
        dominant_behavior_filled = fillmissing(dominant_behavior_double, 'nearest');
        
        % Calculate average bout duration for each behavior
        for beh_idx = 1:n_behaviors
            % Create binary vector for this behavior
            beh_binary = (dominant_behavior_filled == beh_idx);
            
            % Find onsets and offsets
            onset = find(diff([0; beh_binary(:)]) == 1);
            offset = find(diff([beh_binary(:); 0]) == -1);
            
            % Calculate bout durations
            if ~isempty(onset) && ~isempty(offset)
                bout_durations = (offset - onset + 1) * Fs;  % +1 because both onset and offset frames are included
                session_avg_duration_before(i, beh_idx) = mean(bout_durations);
            else
                session_avg_duration_before(i, beh_idx) = 0;
            end
        end
    else
        session_avg_duration_before(i, :) = NaN;
    end
end

% Process AFTER sessions
for i = 1:length(unique_sessions_after)
    sess_idx = session_ids_after == unique_sessions_after(i);
    sess_predictions = predictions_after(sess_idx, :);
    
    if ~isempty(sess_predictions)
        [max_confidence, dominant_behavior] = max(sess_predictions, [], 2);
        valid_predictions = max_confidence > confidence_threshold_dominant;
        
        % Mark invalid predictions as NaN
        dominant_behavior_double = double(dominant_behavior);
        dominant_behavior_double(~valid_predictions) = NaN;
        
        % Fill missing values with nearest neighbor
        dominant_behavior_filled = fillmissing(dominant_behavior_double, 'nearest');
        
        % Calculate average bout duration for each behavior
        for beh_idx = 1:n_behaviors
            % Create binary vector for this behavior
            beh_binary = (dominant_behavior_filled == beh_idx);
            
            % Find onsets and offsets
            onset = find(diff([0; beh_binary(:)]) == 1);
            offset = find(diff([beh_binary(:); 0]) == -1);
            
            % Calculate bout durations
            if ~isempty(onset) && ~isempty(offset)
                bout_durations = (offset - onset + 1) * Fs;
                session_avg_duration_after(i, beh_idx) = mean(bout_durations);
            else
                session_avg_duration_after(i, beh_idx) = 0;
            end
        end
    else
        session_avg_duration_after(i, :) = NaN;
    end
end

% Statistical analysis using non-parametric tests
fig = figure('Position', [100, 100, 1200, 600]);

% Calculate medians and IQR for plotting
session_medians_before = median(session_avg_duration_before, 1, 'omitnan');
session_medians_after = median(session_avg_duration_after, 1, 'omitnan');

% Calculate IQR for error bars
session_q25_before = prctile(session_avg_duration_before, 25, 1);
session_q75_before = prctile(session_avg_duration_before, 75, 1);
session_q25_after = prctile(session_avg_duration_after, 25, 1);
session_q75_after = prctile(session_avg_duration_after, 75, 1);

% For asymmetric error bars
session_err_low_before = session_medians_before - session_q25_before;
session_err_high_before = session_q75_before - session_medians_before;
session_err_low_after = session_medians_after - session_q25_after;
session_err_high_after = session_q75_after - session_medians_after;

session_p_values = zeros(1, n_behaviors);
session_effect_sizes = zeros(1, n_behaviors);

for beh_idx = 1:n_behaviors
    before_vals = session_avg_duration_before(:, beh_idx);
    after_vals = session_avg_duration_after(:, beh_idx);
    
    % Find paired sessions (both before and after have valid data)
    valid_pairs = ~isnan(before_vals) & ~isnan(after_vals);
    
    before_vals_paired = before_vals(valid_pairs);
    after_vals_paired = after_vals(valid_pairs);
    
    if length(before_vals_paired) >= 3
        % Wilcoxon Signed-Rank Test (paired non-parametric test)
        [session_p_values(beh_idx), ~, stats] = signrank(before_vals_paired, after_vals_paired);
        
        % Calculate matched-pairs rank-biserial correlation as effect size
        n_pairs = length(before_vals_paired);
        
        % Calculate differences
        differences = after_vals_paired - before_vals_paired;
        abs_diffs = abs(differences);
        ranks = tiedrank(abs_diffs);
        
        % Sum of positive ranks (where after > before)
        W_plus = sum(ranks(differences > 0));
        % Total possible sum of ranks
        W_max = n_pairs * (n_pairs + 1) / 2;
        
        % Matched-pairs rank-biserial correlation
        % r = (W+ / W_max) * 2 - 1
        % Range: -1 (all negative) to +1 (all positive)
        session_effect_sizes(beh_idx) = (W_plus / W_max) * 2 - 1;
        
    else
        session_p_values(beh_idx) = NaN;
        session_effect_sizes(beh_idx) = NaN;
    end
end

% Main plot: Session-level average bout durations
x_pos = 1:n_behaviors;
bar_width = 0.35;

% Plot error bars with median ± IQR
errorbar(x_pos - bar_width/2, session_medians_before, session_err_low_before, session_err_high_before, ...
    'o', 'MarkerSize', 8, 'MarkerFaceColor', colors.before, 'Color', colors.before, 'LineWidth', 2);
hold on;
errorbar(x_pos + bar_width/2, session_medians_after, session_err_low_after, session_err_high_after, ...
    'o', 'MarkerSize', 8, 'MarkerFaceColor', colors.after, 'Color', colors.after, 'LineWidth', 2);

% Add individual session points with jitter
for beh_idx = 1:n_behaviors
    before_vals = session_avg_duration_before(:, beh_idx);
    after_vals = session_avg_duration_after(:, beh_idx);
    
    before_vals_clean = before_vals(~isnan(before_vals));
    after_vals_clean = after_vals(~isnan(after_vals));
    
    jitter_before = (rand(size(before_vals_clean)) - 0.5) * 0.2;
    jitter_after = (rand(size(after_vals_clean)) - 0.5) * 0.2;
    
    scatter(beh_idx - bar_width/2 + jitter_before, before_vals_clean, 25, colors.before, 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(beh_idx + bar_width/2 + jitter_after, after_vals_clean, 25, colors.after, 'filled', 'MarkerFaceAlpha', 0.6);
end

% Add connecting lines between paired sessions
for beh_idx = 1:n_behaviors
    before_vals = session_avg_duration_before(:, beh_idx);
    after_vals = session_avg_duration_after(:, beh_idx);
    
    valid_sessions = ~isnan(before_vals) & ~isnan(after_vals);
    
    if sum(valid_sessions) > 0
        before_vals_paired = before_vals(valid_sessions);
        after_vals_paired = after_vals(valid_sessions);
        
        for sess_i = 1:length(before_vals_paired)
            plot([beh_idx - bar_width/2, beh_idx + bar_width/2], ...
                 [before_vals_paired(sess_i), after_vals_paired(sess_i)], ...
                 'k-', 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5 0.7]);
        end
    end
end

% Add significance stars
y_max = max([session_q75_before; session_q75_after]);
for beh_idx = 1:n_behaviors
    if ~isnan(session_p_values(beh_idx))
        if session_p_values(beh_idx) < 0.001
            star_text = '***';
        elseif session_p_values(beh_idx) < 0.01
            star_text = '**';
        elseif session_p_values(beh_idx) < 0.05
            star_text = '*';
        else
            star_text = '';
        end
        
        if ~isempty(star_text)
            star_y = y_max(beh_idx) + 0.5;
            line_y = y_max(beh_idx) + 0.3;
            plot([beh_idx - bar_width/2, beh_idx + bar_width/2], [line_y, line_y], 'k-', 'LineWidth', 1);
            text(beh_idx, star_y, star_text, 'HorizontalAlignment', 'center', ...
                'FontSize', 14, 'FontWeight', 'bold', 'Color', 'red');
        end
    end
end

ylabel('Average Bout Duration (seconds)');
xlabel('Behavior');
title('Session-Level Average Behavior Bout Durations: Before vs After First Noise (Median ± IQR)');
set(gca, 'XTick', x_pos, 'XTickLabel', behavior_names, 'XTickLabelRotation', 45);
legend(condition_names, 'Location', 'best', 'FontSize', 10);
grid on;

ylim([0, max(y_max) + 2]);

% Print statistics summary
fprintf('\nNon-parametric statistics (Signed-rank test) - Average Bout Duration Analysis:\n');
fprintf('Behavior\t\tMedian Before (s)\tMedian After (s)\tp-value\t\tEffect Size (r)\n');
fprintf('------------------------------------------------------------------------------------\n');
for beh_idx = 1:n_behaviors
    fprintf('%s\t\t%.2f\t\t\t%.2f\t\t\t%.4f\t\t%.3f\n', ...
        behavior_names{beh_idx}, session_medians_before(beh_idx), ...
        session_medians_after(beh_idx), session_p_values(beh_idx), ...
        session_effect_sizes(beh_idx));
end

%% Helper Functions
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

function [extracted_data, baseline_info] = extract_period_data(sessions, prediction_sessions, period_field)
    n_sessions = length(sessions);
    reward_baselines = struct('mean', cell(n_sessions, 1), 'std', cell(n_sessions, 1), 'valid', cell(n_sessions, 1));
    
    %% First pass: Calculate baselines
    for i = 1:n_sessions
        fprintf("%d Session \n", i)
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
%         coupling_MI = coupling_data_session.summary.all_MI_values(4, :); % 7Hz band
        coupling_MI = coupling_data_session.summary.all_MI_values; % 7Hz band
        neural_time = session.NeuralTime;
        
        if size(behavioral_matrix, 2) >= 1
            [reward_periods, ~, ~] = group_behavioral_states(behavioral_matrix);
        else
            reward_periods = true(size(neural_time));
        end
        
        try
            coupling_neural = interp1(coupling_time, coupling_MI, neural_time, 'linear', 'extrap');
            reward_coupling = coupling_neural(reward_periods);
            
            % Vectorized outlier removal
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
            % Already set to invalid
        end
    end
    
    %% Second pass: Extract data
    % Pre-allocate cell array for speed
    all_data_cell = cell(n_sessions, 1);
    
    for i = 1:n_sessions
        fprintf("%d Session \n", i)
        session = sessions{i};
        
        if ~reward_baselines(i).valid || ...
           ~isfield(session, 'behavioral_matrix_full') || ...
           ~isfield(session, 'coupling_results_multiband') || ...
           ~isfield(session, 'TriggerMid')
            continue;
        end
        
        % Extract period data
        if ~isempty(period_field) && isfield(session, period_field)
            period_indices = session.(period_field);
        end

        behavioral_matrix = session.behavioral_matrix_full;
        neural_time = session.NeuralTime;
        speed = session.Speed;
        camera_time = session.TriggerMid;
        coupling_data_session = session.coupling_results_multiband;
        
        % Get prediction scores
        if i <= length(prediction_sessions) && isfield(prediction_sessions(1), 'prediction_scores')
            prediction_scores = prediction_sessions(i).prediction_scores;
        else
            continue;
        end
        
        coupling_time = coupling_data_session.summary.window_times;
%         coupling_MI = coupling_data_session.summary.all_MI_values(4, :);
        coupling_MI = coupling_data_session.summary.all_MI_values;
        
        try
            % Interpolate all data at once
            speed_camera = interp1(neural_time, speed, camera_time, 'linear', 'extrap');
            
            % Handle breathing data
            if size(behavioral_matrix, 2) >= 8
                breathing_neural = behavioral_matrix(:, 8);
                breathing_camera = interp1(neural_time, breathing_neural, camera_time, 'linear', 'extrap');
            else
                continue;
            end
            
            % Interpolate coupling data
            coupling_camera_raw = interp1(coupling_time, coupling_MI, camera_time, 'linear', 'extrap');
            baseline_mean = reward_baselines(i).mean;
            baseline_std = reward_baselines(i).std;
            coupling_camera_normalized = (coupling_camera_raw - baseline_mean) / baseline_std;
            
        catch
            continue;
        end

        % Pre-calculate prediction windows
        prediction_indices = 1:20:length(camera_time)+1;
        
        % Pre-allocate struct array for this session
        session_data = struct(...
            'session_id', [], ...
            'prediction_id', [], ...
            'camera_time', [], ...
            'speed_median', [], ...
            'breathing_median', [], ...
            'coupling_median_normalized', [], ...
            'coupling_median_raw', [], ...
            'prediction_scores', [], ...
            'reward_baseline_mean', [], ...
            'reward_baseline_std', [], ...
            'is_in_temporal_overlap', []);
        
        session_data = repmat(session_data, size(prediction_scores, 1), 1);
        
        % Vectorized computation where possible
        valid_count = 0;
        for p = 1:length(prediction_indices)-1
            window_start = prediction_indices(p);
            window_end = prediction_indices(p+1)-1;
            
            if or(camera_time(window_start) <= neural_time(period_indices(1)),camera_time(window_start) >= neural_time(period_indices(end))), continue; end
            
            valid_count = valid_count + 1;
            
            % Extract window data
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
        
        % Trim to valid entries
        if valid_count > 0
            all_data_cell{i} = session_data(1:valid_count);
        end
    end
    
    %% Concatenate all data efficiently
    % Remove empty cells
    all_data_cell = all_data_cell(~cellfun(@isempty, all_data_cell));
    
    % Concatenate all at once
    if ~isempty(all_data_cell)
        extracted_data = vertcat(all_data_cell{:});
    else
        extracted_data = [];
    end
    
    baseline_info = reward_baselines;
end

