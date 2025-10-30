%% Comprehensive Behavioral Analysis with Response Groups
% Compares across:
%   - 7 behavioral types
%   - Before vs After first aversive noise
%   - Responder vs Non-responder groups
% Metrics analyzed:
%   - Breathing-LFP coupling strength
%   - Breathing rate
%   - Movement speed

clear all
% close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== COMPREHENSIVE BEHAVIORAL ANALYSIS WITH RESPONSE GROUPS ===\n\n');

% Analysis parameters
config = struct();
config.confidence_threshold = 0.3;           % Behavioral prediction threshold
config.goal_movement_column = 4;             % Column 7 for Goal-Directed Movement
config.drop_threshold_percent = 5;           % Threshold for responder classification

% Behavioral definitions
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
config.n_behaviors = 7;

% Colors
config.colors = struct();
config.colors.before = [0.1 0.4 0.7];        % Blue
config.colors.after = [0.9 0.3 0.2];         % Red
config.colors.responder = [0.9 0.3 0.2];     % Red
config.colors.non_responder = [0.1 0.4 0.7]; % Blue

%% ========================================================================
%  SECTION 2: LOAD DATA
%  ========================================================================

fprintf('Loading data...\n');

try
    coupling_data = load('27-Oct-2025_RewardAversive_session_metrics_breathing_LFPCcouple(10-1)');
    sessions = coupling_data.all_session_metrics;
    pred_data = load('lstm_prediction_results_aversive_27-Oct-2025');
    prediction_sessions = pred_data.final_results.session_predictions;
    fprintf('✓ Loaded %d sessions\n\n', length(sessions));
catch ME
    fprintf('❌ Failed to load data: %s\n', ME.message);
    return;
end

%% ========================================================================
%  SECTION 3: CLASSIFY RESPONSE GROUPS
%  ========================================================================

fprintf('Classifying response groups...\n');

[group_assignment, goal_movement_change] = classify_response_groups(sessions, config);

n_responders = sum(group_assignment.is_responder);
n_non_responders = sum(~group_assignment.is_responder & ~isnan(goal_movement_change));

fprintf('✓ Classification complete:\n');
fprintf('  - Responders: %d sessions\n', n_responders);
fprintf('  - Non-responders: %d sessions\n\n', n_non_responders);

%% ========================================================================
%  SECTION 4: EXTRACT DATA BY GROUP AND PERIOD
%  ========================================================================

fprintf('Extracting data for each group and period...\n');

% Extract data for all sessions (before and after)
[data_before, ~] = extract_period_data(sessions, prediction_sessions, 'before_indices');
[data_after, ~] = extract_period_data(sessions, prediction_sessions, 'after_indices');

fprintf('✓ Data extracted:\n');
fprintf('  - Before: %d data points\n', length(data_before));
fprintf('  - After: %d data points\n\n', length(data_after));

%% ========================================================================
%  SECTION 5: ORGANIZE DATA BY BEHAVIOR AND GROUP
%  ========================================================================

fprintf('Organizing data by behavior and group...\n');
% Initialize storage for all three metrics
% For each metric: [behaviors × sessions × conditions]
% Conditions: 1=Before, 2=After
metrics = struct();
metrics.coupling = struct('responder', [], 'non_responder', []);
metrics.breathing = struct('responder', [], 'non_responder', []);
metrics.speed = struct('responder', [], 'non_responder', []);

% Pre-allocate cell arrays to store data for each behavior
metrics.coupling.responder = cell(config.n_behaviors, 1);
metrics.coupling.non_responder = cell(config.n_behaviors, 1);
metrics.breathing.responder = cell(config.n_behaviors, 1);
metrics.breathing.non_responder = cell(config.n_behaviors, 1);
metrics.speed.responder = cell(config.n_behaviors, 1);
metrics.speed.non_responder = cell(config.n_behaviors, 1);

% Organize data for responders
fprintf('Processing responders...\n');
responder_sessions = find(group_assignment.is_responder);
for beh = 1:config.n_behaviors
    [metrics.coupling.responder{beh}, ...
     metrics.breathing.responder{beh}, ...
     metrics.speed.responder{beh}] = ...
        extract_metrics_for_behavior(data_before, data_after, beh, ...
                                     responder_sessions, config);
end

% Organize data for non-responders
fprintf('Processing non-responders...\n');
non_responder_sessions = find(~group_assignment.is_responder & ...
                              ~isnan(goal_movement_change));
for beh = 1:config.n_behaviors
    [metrics.coupling.non_responder{beh}, ...
     metrics.breathing.non_responder{beh}, ...
     metrics.speed.non_responder{beh}] = ...
        extract_metrics_for_behavior(data_before, data_after, beh, ...
                                     non_responder_sessions, config);
end
fprintf('✓ Data organization complete\n\n');

%% ========================================================================
%  SECTION 6: CALCULATE SESSION-LEVEL STATISTICS
%  ========================================================================

fprintf('Calculating session-level statistics...\n');

stats_results = calculate_session_statistics(metrics, config);

fprintf('✓ Statistics calculated\n\n');

%% ========================================================================
%  SECTION 7: VISUALIZATION - COUPLING STRENGTH
%  ========================================================================

fprintf('Creating coupling strength visualizations...\n');

fig1 = figure('Position', [50, 50, 1400, 900]);

for beh = 1:config.n_behaviors
    subplot(3, 3, beh);
    hold on;
    
    % Get data
    resp_before = stats_results.coupling.responder.before_mean(beh);
    resp_after = stats_results.coupling.responder.after_mean(beh);
    resp_before_sem = stats_results.coupling.responder.before_sem(beh);
    resp_after_sem = stats_results.coupling.responder.after_sem(beh);
    
    nonresp_before = stats_results.coupling.non_responder.before_mean(beh);
    nonresp_after = stats_results.coupling.non_responder.after_mean(beh);
    nonresp_before_sem = stats_results.coupling.non_responder.before_sem(beh);
    nonresp_after_sem = stats_results.coupling.non_responder.after_sem(beh);
    
    % Plot
    x_pos = [1, 2, 4, 5];
    means = [resp_before, resp_after, nonresp_before, nonresp_after];
    sems = [resp_before_sem, resp_after_sem, nonresp_before_sem, nonresp_after_sem];
    colors_plot = [config.colors.before; config.colors.after; ...
                   config.colors.before; config.colors.after];
    
    for i = 1:4
        errorbar(x_pos(i), means(i), sems(i), 'o', ...
                'MarkerSize', 10, 'MarkerFaceColor', colors_plot(i,:), ...
                'Color', colors_plot(i,:), 'LineWidth', 2);
    end
    
    % Add connecting lines
    plot([1, 2], [resp_before, resp_after], '-', ...
         'Color', config.colors.responder, 'LineWidth', 1.5);
    plot([4, 5], [nonresp_before, nonresp_after], '-', ...
         'Color', config.colors.non_responder, 'LineWidth', 1.5);
    
    % Labels
    ylabel('Coupling Strength (z-score)', 'FontSize', 10);
    title(config.behavior_names{beh}, 'FontSize', 11, 'FontWeight', 'bold');
    set(gca, 'XTick', [1.5, 4.5], 'XTickLabel', {'Responders', 'Non-responders'});
    xlim([0, 6]);
    grid on;
    
    % Add significance markers
    p_resp = stats_results.coupling.responder.p_values(beh);
    p_nonresp = stats_results.coupling.non_responder.p_values(beh);
    
    if p_resp < 0.001
        text(1.5, max([resp_before, resp_after]) + 0.2, '***', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'red');
    elseif p_resp < 0.01
        text(1.5, max([resp_before, resp_after]) + 0.2, '**', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'red');
    elseif p_resp < 0.05
        text(1.5, max([resp_before, resp_after]) + 0.2, '*', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'red');
    end
    
    if p_nonresp < 0.001
        text(4.5, max([nonresp_before, nonresp_after]) + 0.2, '***', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'blue');
    elseif p_nonresp < 0.01
        text(4.5, max([nonresp_before, nonresp_after]) + 0.2, '**', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'blue');
    elseif p_nonresp < 0.05
        text(4.5, max([nonresp_before, nonresp_after]) + 0.2, '*', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'blue');
    end
end

sgtitle('Breathing-LFP Coupling Strength: Before vs After First Noise by Group', ...
        'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 8: VISUALIZATION - BREATHING RATE
%  ========================================================================

fprintf('Creating breathing rate visualizations...\n');

fig2 = figure('Position', [100, 50, 1400, 900]);

for beh = 1:config.n_behaviors
    subplot(3, 3, beh);
    hold on;
    
    % Get data
    resp_before = stats_results.breathing.responder.before_mean(beh);
    resp_after = stats_results.breathing.responder.after_mean(beh);
    resp_before_sem = stats_results.breathing.responder.before_sem(beh);
    resp_after_sem = stats_results.breathing.responder.after_sem(beh);
    
    nonresp_before = stats_results.breathing.non_responder.before_mean(beh);
    nonresp_after = stats_results.breathing.non_responder.after_mean(beh);
    nonresp_before_sem = stats_results.breathing.non_responder.before_sem(beh);
    nonresp_after_sem = stats_results.breathing.non_responder.after_sem(beh);
    
    % Plot
    x_pos = [1, 2, 4, 5];
    means = [resp_before, resp_after, nonresp_before, nonresp_after];
    sems = [resp_before_sem, resp_after_sem, nonresp_before_sem, nonresp_after_sem];
    colors_plot = [config.colors.before; config.colors.after; ...
                   config.colors.before; config.colors.after];
    
    for i = 1:4
        errorbar(x_pos(i), means(i), sems(i), 'o', ...
                'MarkerSize', 10, 'MarkerFaceColor', colors_plot(i,:), ...
                'Color', colors_plot(i,:), 'LineWidth', 2);
    end
    
    % Add connecting lines
    plot([1, 2], [resp_before, resp_after], '-', ...
         'Color', config.colors.responder, 'LineWidth', 1.5);
    plot([4, 5], [nonresp_before, nonresp_after], '-', ...
         'Color', config.colors.non_responder, 'LineWidth', 1.5);
    
    % Labels
    ylabel('Breathing Rate (Hz)', 'FontSize', 10);
    title(config.behavior_names{beh}, 'FontSize', 11, 'FontWeight', 'bold');
    set(gca, 'XTick', [1.5, 4.5], 'XTickLabel', {'Responders', 'Non-responders'});
    xlim([0, 6]);
    grid on;
    
    % Add significance markers
    p_resp = stats_results.breathing.responder.p_values(beh);
    p_nonresp = stats_results.breathing.non_responder.p_values(beh);
    
    if p_resp < 0.001
        text(1.5, max([resp_before, resp_after]) + 0.1, '***', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'red');
    elseif p_resp < 0.01
        text(1.5, max([resp_before, resp_after]) + 0.1, '**', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'red');
    elseif p_resp < 0.05
        text(1.5, max([resp_before, resp_after]) + 0.1, '*', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'red');
    end
    
    if p_nonresp < 0.001
        text(4.5, max([nonresp_before, nonresp_after]) + 0.1, '***', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'blue');
    elseif p_nonresp < 0.01
        text(4.5, max([nonresp_before, nonresp_after]) + 0.1, '**', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'blue');
    elseif p_nonresp < 0.05
        text(4.5, max([nonresp_before, nonresp_after]) + 0.1, '*', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'blue');
    end
end

sgtitle('Breathing Rate: Before vs After First Noise by Group', ...
        'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 9: VISUALIZATION - MOVEMENT SPEED
%  ========================================================================

fprintf('Creating movement speed visualizations...\n');

fig3 = figure('Position', [150, 50, 1400, 900]);

for beh = 1:config.n_behaviors
    subplot(3, 3, beh);
    hold on;
    
    % Get data
    resp_before = stats_results.speed.responder.before_mean(beh);
    resp_after = stats_results.speed.responder.after_mean(beh);
    resp_before_sem = stats_results.speed.responder.before_sem(beh);
    resp_after_sem = stats_results.speed.responder.after_sem(beh);
    
    nonresp_before = stats_results.speed.non_responder.before_mean(beh);
    nonresp_after = stats_results.speed.non_responder.after_mean(beh);
    nonresp_before_sem = stats_results.speed.non_responder.before_sem(beh);
    nonresp_after_sem = stats_results.speed.non_responder.after_sem(beh);
    
    % Plot
    x_pos = [1, 2, 4, 5];
    means = [resp_before, resp_after, nonresp_before, nonresp_after];
    sems = [resp_before_sem, resp_after_sem, nonresp_before_sem, nonresp_after_sem];
    colors_plot = [config.colors.before; config.colors.after; ...
                   config.colors.before; config.colors.after];
    
    for i = 1:4
        errorbar(x_pos(i), means(i), sems(i), 'o', ...
                'MarkerSize', 10, 'MarkerFaceColor', colors_plot(i,:), ...
                'Color', colors_plot(i,:), 'LineWidth', 2);
    end
    
    % Add connecting lines
    plot([1, 2], [resp_before, resp_after], '-', ...
         'Color', config.colors.responder, 'LineWidth', 1.5);
    plot([4, 5], [nonresp_before, nonresp_after], '-', ...
         'Color', config.colors.non_responder, 'LineWidth', 1.5);
    
    % Labels
    ylabel('Speed (cm/s)', 'FontSize', 10);
    title(config.behavior_names{beh}, 'FontSize', 11, 'FontWeight', 'bold');
    set(gca, 'XTick', [1.5, 4.5], 'XTickLabel', {'Responders', 'Non-responders'});
    xlim([0, 6]);
    grid on;
    
    % Add significance markers
    p_resp = stats_results.speed.responder.p_values(beh);
    p_nonresp = stats_results.speed.non_responder.p_values(beh);
    
    if p_resp < 0.001
        text(1.5, max([resp_before, resp_after]) + 2, '***', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'red');
    elseif p_resp < 0.01
        text(1.5, max([resp_before, resp_after]) + 2, '**', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'red');
    elseif p_resp < 0.05
        text(1.5, max([resp_before, resp_after]) + 2, '*', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'red');
    end
    
    if p_nonresp < 0.001
        text(4.5, max([nonresp_before, nonresp_after]) + 2, '***', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'blue');
    elseif p_nonresp < 0.01
        text(4.5, max([nonresp_before, nonresp_after]) + 2, '**', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'blue');
    elseif p_nonresp < 0.05
        text(4.5, max([nonresp_before, nonresp_after]) + 2, '*', ...
             'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'blue');
    end
end

sgtitle('Movement Speed: Before vs After First Noise by Group', ...
        'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 10: PRINT STATISTICS
%  ========================================================================

fprintf('\n=== STATISTICAL RESULTS ===\n\n');
print_statistics(stats_results, config);

%% ========================================================================
%  SECTION 11: SAVE RESULTS
%  ========================================================================

fprintf('\nSaving results...\n');

results = struct();
results.config = config;
results.group_assignment = group_assignment;
results.metrics = metrics;
results.stats_results = stats_results;

save('coupling_breathing_speed_analysis_with_groups.mat', 'results');

fprintf('✓ Results saved\n');
fprintf('\n=== ANALYSIS COMPLETE ===\n');

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function [group_assignment, goal_movement_change] = classify_response_groups(sessions, config)
    % Classify sessions into responder/non-responder groups
    
    n_sessions = length(sessions);
    goal_movement_change = nan(n_sessions, 1);
    
    for sess_idx = 1:n_sessions
        session = sessions{sess_idx};
        
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
        
        % Period 2: After first aversive noise
        if length(aversive_times) > 1
            period2_idx = neural_time >= aversive_times(1) & neural_time < aversive_times(2);
        else
            period2_idx = neural_time >= aversive_times(1);
        end
        
        % Calculate Goal-Directed Movement frequency
        goal_movement = behavioral_matrix(:, config.goal_movement_column);
        freq_before = (sum(goal_movement(period1_idx)) / sum(period1_idx)) * 100;
        freq_after = (sum(goal_movement(period2_idx)) / sum(period2_idx)) * 100;
        
        goal_movement_change(sess_idx) = freq_after - freq_before;
    end
    
    % Classify
    group_assignment = struct();
    group_assignment.session_ids = (1:n_sessions)';
    group_assignment.goal_movement_change = goal_movement_change;
    group_assignment.is_responder = goal_movement_change < -config.drop_threshold_percent;
end

function [coupling_data, breathing_data, speed_data] = ...
    extract_metrics_for_behavior(data_before, data_after, behavior_idx, ...
                                 session_ids, config)
    % Extract coupling, breathing, and speed for specific behavior and sessions
    
    n_sessions = length(session_ids);
    
    % Initialize storage: [sessions × 2] where columns are [before, after]
    coupling_data = nan(n_sessions, 2);
    breathing_data = nan(n_sessions, 2);
    speed_data = nan(n_sessions, 2);
    
    % Process each session
    for i = 1:n_sessions
        sess_id = session_ids(i);
        
        % Before period
        sess_data_before = data_before([data_before.session_id] == sess_id);
        if ~isempty(sess_data_before)
            [coupling_before, breathing_before, speed_before] = ...
                get_behavior_metrics(sess_data_before, behavior_idx, config);
            coupling_data(i, 1) = coupling_before;
            breathing_data(i, 1) = breathing_before;
            speed_data(i, 1) = speed_before;
        end
        
        % After period
        sess_data_after = data_after([data_after.session_id] == sess_id);
        if ~isempty(sess_data_after)
            [coupling_after, breathing_after, speed_after] = ...
                get_behavior_metrics(sess_data_after, behavior_idx, config);
            coupling_data(i, 2) = coupling_after;
            breathing_data(i, 2) = breathing_after;
            speed_data(i, 2) = speed_after;
        end
    end
end

function [coupling_mean, breathing_mean, speed_mean] = ...
    get_behavior_metrics(session_data, behavior_idx, config)
    % Get average metrics for a specific behavior in a session
    
    % Extract predictions
    predictions = vertcat(session_data.prediction_scores);
    
    % Find dominant behavior
    [max_confidence, dominant_behavior] = max(predictions, [], 2);
    valid_predictions = max_confidence > config.confidence_threshold;
    
    % Filter for this specific behavior
    is_target_behavior = (dominant_behavior == behavior_idx) & valid_predictions;
    
    if sum(is_target_behavior) > 0
        % Get metrics for this behavior
        coupling_vals = [session_data(is_target_behavior).coupling_median_normalized];
        breathing_vals = [session_data(is_target_behavior).breathing_median];
        speed_vals = [session_data(is_target_behavior).speed_median];
        
        coupling_mean = mean(coupling_vals, 'omitnan');
        breathing_mean = mean(breathing_vals, 'omitnan');
        speed_mean = mean(speed_vals, 'omitnan');
    else
        coupling_mean = NaN;
        breathing_mean = NaN;
        speed_mean = NaN;
    end
end

function stats_results = calculate_session_statistics(metrics, config)
    % Calculate statistics for all metrics and groups
    
    stats_results = struct();
    metric_names = {'coupling', 'breathing', 'speed'};
    
    for m = 1:length(metric_names)
        metric_name = metric_names{m};
        
        for group = {'responder', 'non_responder'}
            group_name = group{1};
            
            before_means = zeros(config.n_behaviors, 1);
            after_means = zeros(config.n_behaviors, 1);
            before_sems = zeros(config.n_behaviors, 1);
            after_sems = zeros(config.n_behaviors, 1);
            p_values = zeros(config.n_behaviors, 1);
            
            for beh = 1:config.n_behaviors
                data = metrics.(metric_name).(group_name){beh};
                
                before_vals = data(:, 1);
                after_vals = data(:, 2);
                
                % Calculate means and SEMs
                before_means(beh) = mean(before_vals, 'omitnan');
                after_means(beh) = mean(after_vals, 'omitnan');
                before_sems(beh) = std(before_vals, 'omitnan') / sqrt(sum(~isnan(before_vals)));
                after_sems(beh) = std(after_vals, 'omitnan') / sqrt(sum(~isnan(after_vals)));
                
                % Statistical test (paired)
                valid_pairs = ~isnan(before_vals) & ~isnan(after_vals);
                if sum(valid_pairs) >= 3
                    p_values(beh) = signrank(before_vals(valid_pairs), ...
                                            after_vals(valid_pairs));
                else
                    p_values(beh) = NaN;
                end
            end
            
            % Store results
            stats_results.(metric_name).(group_name).before_mean = before_means;
            stats_results.(metric_name).(group_name).after_mean = after_means;
            stats_results.(metric_name).(group_name).before_sem = before_sems;
            stats_results.(metric_name).(group_name).after_sem = after_sems;
            stats_results.(metric_name).(group_name).p_values = p_values;
        end
    end
end

function print_statistics(stats_results, config)
    % Print formatted statistical results
    
    metric_names = {'Coupling Strength', 'Breathing Rate', 'Speed'};
    metric_fields = {'coupling', 'breathing', 'speed'};
    
    for m = 1:length(metric_names)
        fprintf('\n=== %s ===\n', metric_names{m});
        fprintf('\nRESPONDERS:\n');
        fprintf('%-25s  Before    After     p-value\n', 'Behavior');
        fprintf('-------------------------------------------------------------\n');
        
        for beh = 1:config.n_behaviors
            before = stats_results.(metric_fields{m}).responder.before_mean(beh);
            after = stats_results.(metric_fields{m}).responder.after_mean(beh);
            p = stats_results.(metric_fields{m}).responder.p_values(beh);
            
            fprintf('%-25s  %.3f     %.3f     %s\n', ...
                    config.behavior_names{beh}, before, after, format_pvalue(p));
        end
        
        fprintf('\nNON-RESPONDERS:\n');
        fprintf('%-25s  Before    After     p-value\n', 'Behavior');
        fprintf('-------------------------------------------------------------\n');
        
        for beh = 1:config.n_behaviors
            before = stats_results.(metric_fields{m}).non_responder.before_mean(beh);
            after = stats_results.(metric_fields{m}).non_responder.after_mean(beh);
            p = stats_results.(metric_fields{m}).non_responder.p_values(beh);
            
            fprintf('%-25s  %.3f     %.3f     %s\n', ...
                    config.behavior_names{beh}, before, after, format_pvalue(p));
        end
    end
end

function str = format_pvalue(p)
    % Format p-value for display
    if isnan(p)
        str = 'N/A';
    elseif p < 0.001
        str = '<0.001***';
    elseif p < 0.01
        str = sprintf('%.3f**', p);
    elseif p < 0.05
        str = sprintf('%.3f*', p);
    else
        str = sprintf('%.3f', p);
    end
end

function [extracted_data, baseline_info] = extract_period_data(sessions, prediction_sessions, period_field)
    % Extract data for before or after first aversive noise
    % period_field: 'before_indices' or 'after_indices'
    
    n_sessions = length(sessions);
    
    % First pass: Calculate reward baselines
    reward_baselines = struct('mean', cell(n_sessions, 1), ...
                             'std', cell(n_sessions, 1), ...
                             'valid', cell(n_sessions, 1));
    
    for i = 1:n_sessions
        session = sessions{i};
        reward_baselines(i).valid = false;
        
        if ~isfield(session, 'behavioral_matrix_full') || ...
           ~isfield(session, 'coupling_results_multiband')
            continue;
        end
        
        behavioral_matrix = session.behavioral_matrix_full;
        coupling_data_session = session.coupling_results_multiband;
        
        if isempty(coupling_data_session) || isempty(coupling_data_session.summary)
            continue;
        end
        
        coupling_time = coupling_data_session.summary.window_times;
        coupling_MI = coupling_data_session.summary.all_MI_values;
        neural_time = session.NeuralTime;
        
        % Get reward periods
        if size(behavioral_matrix, 2) >= 1
            reward_periods = behavioral_matrix(:, 1) == 1;
        else
            reward_periods = true(size(neural_time));
        end
        
        try
            coupling_neural = interp1(coupling_time, coupling_MI, neural_time, 'linear', 'extrap');
            reward_coupling = coupling_neural(reward_periods);
            
            % Remove outliers
            mu = mean(reward_coupling);
            sigma = std(reward_coupling);
            reward_coupling_clean = reward_coupling(abs(reward_coupling - mu) < 3 * sigma);
            
            if length(reward_coupling_clean) > 10
                reward_baselines(i).mean = mean(reward_coupling_clean);
                reward_baselines(i).std = std(reward_coupling_clean);
                if reward_baselines(i).std == 0
                    reward_baselines(i).std = 0.001;
                end
                reward_baselines(i).valid = true;
            end
        catch
        end
    end
    
    % Second pass: Extract data
    all_data_cell = cell(n_sessions, 1);
    
    for i = 1:n_sessions
        session = sessions{i};
        
        if ~reward_baselines(i).valid || ...
           ~isfield(session, 'behavioral_matrix_full') || ...
           ~isfield(session, 'coupling_results_multiband') || ...
           ~isfield(session, 'TriggerMid')
            continue;
        end
        
        % Get period indices
        if ~isempty(period_field) && isfield(session, period_field)
            period_indices = session.(period_field);
        else
            continue;
        end
        
        behavioral_matrix = session.behavioral_matrix_full;
        neural_time = session.NeuralTime;
        speed = session.Speed;
        camera_time = session.TriggerMid;
        coupling_data_session = session.coupling_results_multiband;
        
        % Get prediction scores
        if i <= length(prediction_sessions) && ...
           isfield(prediction_sessions, 'prediction_scores')
            prediction_scores = prediction_sessions(i).prediction_scores;
        else
            continue;
        end
        
        coupling_time = coupling_data_session.summary.window_times;
        coupling_MI = coupling_data_session.summary.all_MI_values;
        
        try
            % Interpolate data
            speed_camera = interp1(neural_time, speed, camera_time, 'linear', 'extrap');
            
            if size(behavioral_matrix, 2) >= 8
                breathing_neural = behavioral_matrix(:, 8);
                breathing_camera = interp1(neural_time, breathing_neural, ...
                                          camera_time, 'linear', 'extrap');
            else
                continue;
            end
            
            coupling_camera_raw = interp1(coupling_time, coupling_MI, ...
                                         camera_time, 'linear', 'extrap');
            baseline_mean = reward_baselines(i).mean;
            baseline_std = reward_baselines(i).std;
            coupling_camera_normalized = (coupling_camera_raw - baseline_mean) / baseline_std;
            
        catch
            continue;
        end
        
        % Extract prediction windows
        prediction_indices = 1:20:length(camera_time)+1;
        
        session_data = struct(...
            'session_id', [], 'prediction_id', [], 'camera_time', [], ...
            'speed_median', [], 'breathing_median', [], ...
            'coupling_median_normalized', [], 'coupling_median_raw', [], ...
            'prediction_scores', [], 'reward_baseline_mean', [], ...
            'reward_baseline_std', []);
        
        session_data = repmat(session_data, size(prediction_scores, 1), 1);
        
        valid_count = 0;
        for p = 1:length(prediction_indices)-1
            window_start = prediction_indices(p);
            window_end = prediction_indices(p+1)-1;
            
            % Check if window is in period
            if camera_time(window_start) <= neural_time(period_indices(1)) || ...
               camera_time(window_start) >= neural_time(period_indices(end))
                continue;
            end
            
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
        
        if valid_count > 0
            all_data_cell{i} = session_data(1:valid_count);
        end
    end
    
    % Concatenate all data
    all_data_cell = all_data_cell(~cellfun(@isempty, all_data_cell));
    
    if ~isempty(all_data_cell)
        extracted_data = vertcat(all_data_cell{:});
    else
        extracted_data = [];
    end
    
    baseline_info = reward_baselines;
end