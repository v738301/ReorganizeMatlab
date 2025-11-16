%% Enhanced Aversive Noise Response Analysis with Period-Based Behavioral Changes
% This script extends the basic analysis with:
% 1. Group classification based on Goal-Directed Movement
% 2. PSTH aligned to aversive noise
% 3. Behavioral type percentage comparison across 7 periods for each group
% 4. Statistical testing of group differences

clear all
% close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== ENHANCED AVERSIVE RESPONSE GROUP ANALYSIS ===\n\n');

% Analysis parameters
config = struct();
config.confidence_threshold = 0.3;
config.goal_movement_column = 7;
config.psth_time_window = [-60, 300];
config.psth_bin_size = 5;
config.significance_alpha = 0.05;
config.drop_threshold_percent = 5;  % Threshold for defining "responder" (% drop)

% Behavioral definitions
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
config.n_behaviors = 7;

% Period definitions
config.period_names = {'P1: Before Noise 1', 'P2: After Noise 1', 'P3: After Noise 2', ...
                       'P4: After Noise 3', 'P5: After Noise 4', 'P6: After Noise 5', ...
                       'P7: After Noise 6'};

% Colors
config.group_colors = struct('responder', [0.9 0.3 0.2], ...
                             'non_responder', [0.1 0.4 0.7]);

config.period_colors = [0.1 0.4 0.7;   % P1 - Blue
                        0.9 0.3 0.2;    % P2 - Red
                        0.9 0.6 0.1;    % P3 - Orange
                        0.5 0.8 0.2;    % P4 - Green
                        0.7 0.2 0.7;    % P5 - Purple
                        0.2 0.7 0.7;    % P6 - Cyan
                        0.9 0.5 0.6];   % P7 - Pink

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

fprintf('Classifying animals into response groups...\n');

n_sessions = length(sessions);
[group_assignment, goal_movement_change] = classify_response_groups(sessions, config);

fprintf('✓ Classification complete:\n');
fprintf('  - Responders: %d sessions\n', sum(group_assignment.is_responder));
fprintf('  - Non-responders: %d sessions\n\n', sum(~group_assignment.is_responder));

%% ========================================================================
%  SECTION 4: EXTRACT BEHAVIORAL PERCENTAGES AND DURATIONS ACROSS 7 PERIODS
%  ========================================================================

fprintf('Extracting behavioral percentages and durations for 7 periods...\n');

[behavior_percentages, behavior_durations] = extract_period_percentages(sessions, prediction_sessions, ...
                                                                         group_assignment, config);

fprintf('✓ Period extraction complete\n\n');

%% ========================================================================
%  SECTION 5: EXTRACT PSTH DATA
%  ========================================================================

fprintf('Extracting PSTH data aligned to aversive noises...\n');

psth_data = extract_psth_data(sessions, prediction_sessions, group_assignment, config);

fprintf('✓ PSTH extraction complete\n');
fprintf('  - Responder data points: %d\n', length(psth_data.responder));
fprintf('  - Non-responder data points: %d\n\n', length(psth_data.non_responder));

%% ========================================================================
%  SECTION 6: CALCULATE PSTH
%  ========================================================================

fprintf('Calculating behavioral PSTH...\n');

[psth_responder, psth_non_responder, bin_centers] = calculate_psth(psth_data, config);

fprintf('✓ PSTH calculation complete\n\n');

%% ========================================================================
%  SECTION 7: VISUALIZATION - BEHAVIORAL PERCENTAGES ACROSS PERIODS
%  ========================================================================

fprintf('Creating behavioral percentage plots...\n');

fig1 = figure('Position', [50, 50, 1800, 1000]);

for beh = 1:config.n_behaviors
    subplot(3, 3, beh);
    hold on;
    
    % Extract data for this behavior
    responder_data = squeeze(behavior_percentages.responder(:, beh, :));
    non_responder_data = squeeze(behavior_percentages.non_responder(:, beh, :));
    
    % Calculate medians and IQR
    resp_median = median(responder_data, 1, 'omitnan');
    resp_q25 = prctile(responder_data, 25, 1);
    resp_q75 = prctile(responder_data, 75, 1);
    
    nonresp_median = median(non_responder_data, 1, 'omitnan');
    nonresp_q25 = prctile(non_responder_data, 25, 1);
    nonresp_q75 = prctile(non_responder_data, 75, 1);
    
    % Plot with error bars
    x_pos = 1:7;
    
    % Responders
    errorbar(x_pos, resp_median, resp_median - resp_q25, resp_q75 - resp_median, ...
             'o-', 'LineWidth', 2, 'MarkerSize', 8, ...
             'Color', config.group_colors.responder, ...
             'MarkerFaceColor', config.group_colors.responder, ...
             'DisplayName', 'Responders');
    
    % Non-responders
    errorbar(x_pos, nonresp_median, nonresp_median - nonresp_q25, nonresp_q75 - nonresp_median, ...
             'o-', 'LineWidth', 2, 'MarkerSize', 8, ...
             'Color', config.group_colors.non_responder, ...
             'MarkerFaceColor', config.group_colors.non_responder, ...
             'DisplayName', 'Non-responders');
    
    % Add vertical line after Period 1
    xline(1.5, 'k--', 'LineWidth', 1.5, 'Alpha', 0.5);
    
    xlabel('Period', 'FontSize', 11);
    ylabel('Frequency (%)', 'FontSize', 11);
    title(config.behavior_names{beh}, 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    xlim([0.5, 7.5]);
    set(gca, 'XTick', 1:7, 'FontSize', 10);
end

sgtitle('Behavioral Frequency Across 7 Periods: Responders vs Non-responders', ...
        'FontSize', 15, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 8: VISUALIZATION - BEHAVIORAL DURATION ACROSS PERIODS
%  ========================================================================

fprintf('Creating behavioral duration plots...\n');

fig2 = figure('Position', [50, 50, 1800, 1000]);

for beh = 1:config.n_behaviors
    subplot(3, 3, beh);
    hold on;
    
    % Extract data for this behavior
    responder_data = squeeze(behavior_durations.responder(:, beh, :));
    non_responder_data = squeeze(behavior_durations.non_responder(:, beh, :));
    
    % Calculate medians and IQR
    resp_median = median(responder_data, 1, 'omitnan');
    resp_q25 = prctile(responder_data, 25, 1);
    resp_q75 = prctile(responder_data, 75, 1);
    
    nonresp_median = median(non_responder_data, 1, 'omitnan');
    nonresp_q25 = prctile(non_responder_data, 25, 1);
    nonresp_q75 = prctile(non_responder_data, 75, 1);
    
    % Plot with error bars
    x_pos = 1:7;
    
    % Responders
    errorbar(x_pos, resp_median, resp_median - resp_q25, resp_q75 - resp_median, ...
             'o-', 'LineWidth', 2, 'MarkerSize', 8, ...
             'Color', config.group_colors.responder, ...
             'MarkerFaceColor', config.group_colors.responder, ...
             'DisplayName', 'Responders');
    
    % Non-responders
    errorbar(x_pos, nonresp_median, nonresp_median - nonresp_q25, nonresp_q75 - nonresp_median, ...
             'o-', 'LineWidth', 2, 'MarkerSize', 8, ...
             'Color', config.group_colors.non_responder, ...
             'MarkerFaceColor', config.group_colors.non_responder, ...
             'DisplayName', 'Non-responders');
    
    % Add vertical line after Period 1
    xline(1.5, 'k--', 'LineWidth', 1.5, 'Alpha', 0.5);
    
    xlabel('Period', 'FontSize', 11);
    ylabel('Average Bout Duration (s)', 'FontSize', 11);
    title(config.behavior_names{beh}, 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    xlim([0.5, 7.5]);
    set(gca, 'XTick', 1:7, 'FontSize', 10);
end

sgtitle('Behavioral Bout Duration Across 7 Periods: Responders vs Non-responders', ...
        'FontSize', 15, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 9: VISUALIZATION - PSTH
%  ========================================================================

fprintf('Creating PSTH visualizations...\n');

fig3 = figure('Position', [100, 50, 1800, 1000]);

for beh = 1:config.n_behaviors
    subplot(3, 3, beh);
    hold on;
    
    plot(bin_centers, psth_responder(beh, :), 'LineWidth', 2.5, ...
         'Color', config.group_colors.responder, 'DisplayName', 'Responders');
    plot(bin_centers, psth_non_responder(beh, :), 'LineWidth', 2.5, ...
         'Color', config.group_colors.non_responder, 'DisplayName', 'Non-responders');
    
    xline(0, 'k--', 'LineWidth', 2, 'DisplayName', 'Aversive Noise');
    
    xlabel('Time from Aversive Noise (s)', 'FontSize', 11);
    ylabel('Behavioral Frequency (%)', 'FontSize', 11);
    title(config.behavior_names{beh}, 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    xlim(config.psth_time_window);
    set(gca, 'FontSize', 10);
end

sgtitle('Behavioral PSTH Aligned to Aversive Noise', 'FontSize', 15, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 10: STATISTICAL ANALYSIS
%  ========================================================================

fprintf('Performing statistical analyses...\n\n');

stats_results = perform_statistical_tests(behavior_percentages, psth_data, config);

print_statistics(stats_results, config);

%% ========================================================================
%  SECTION 11: SAVE RESULTS
%  ========================================================================

fprintf('\nSaving results...\n');

results = struct();
results.config = config;
results.group_assignment = group_assignment;
results.behavior_percentages = behavior_percentages;
results.behavior_durations = behavior_durations;
results.psth_data = psth_data;
results.psth_responder = psth_responder;
results.psth_non_responder = psth_non_responder;
results.bin_centers = bin_centers;
results.stats_results = stats_results;

save('enhanced_aversive_response_analysis.mat', 'results');

fprintf('✓ Results saved\n');
fprintf('\n=== ANALYSIS COMPLETE ===\n');

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function [group_assignment, goal_movement_change] = classify_response_groups(sessions, config)
    % Classify sessions into responder/non-responder groups based on
    % Goal-Directed Movement change after first aversive noise
    
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
    
    % Classify: responders have significant decrease in Goal-Directed Movement
    group_assignment = struct();
    group_assignment.session_ids = (1:n_sessions)';
    group_assignment.goal_movement_change = goal_movement_change;
    group_assignment.is_responder = goal_movement_change < -config.drop_threshold_percent;
end

function [behavior_percentages, behavior_durations] = extract_period_percentages(sessions, prediction_sessions, ...
                                                            group_assignment, config)
    % Extract behavioral percentages and durations for each session across 7 periods
    
    n_sessions = length(sessions);
    
    % Initialize storage [sessions x behaviors x periods]
    resp_sessions = find(group_assignment.is_responder);
    nonresp_sessions = find(~group_assignment.is_responder);
    
    behavior_percentages = struct();
    behavior_percentages.responder = nan(length(resp_sessions), config.n_behaviors, 7);
    behavior_percentages.non_responder = nan(length(nonresp_sessions), config.n_behaviors, 7);
    
    behavior_durations = struct();
    behavior_durations.responder = nan(length(resp_sessions), config.n_behaviors, 7);
    behavior_durations.non_responder = nan(length(nonresp_sessions), config.n_behaviors, 7);
    
    % Process responders
    for idx = 1:length(resp_sessions)
        sess_idx = resp_sessions(idx);
        [percentages, durations] = extract_session_periods(sessions{sess_idx}, ...
                                              prediction_sessions(sess_idx), config);
        if ~isempty(percentages)
            behavior_percentages.responder(idx, :, :) = percentages;
            behavior_durations.responder(idx, :, :) = durations;
        end
    end
    
    % Process non-responders
    for idx = 1:length(nonresp_sessions)
        sess_idx = nonresp_sessions(idx);
        [percentages, durations] = extract_session_periods(sessions{sess_idx}, ...
                                              prediction_sessions(sess_idx), config);
        if ~isempty(percentages)
            behavior_percentages.non_responder(idx, :, :) = percentages;
            behavior_durations.non_responder(idx, :, :) = durations;
        end
    end
end

function [percentages, durations] = extract_session_periods(session, prediction_session, config)
    % Extract behavioral percentages and bout durations for one session across 7 periods
    
    percentages = nan(config.n_behaviors, 7);
    durations = nan(config.n_behaviors, 7);
    
    if ~isfield(session, 'behavioral_matrix_full') || ...
       ~isfield(session, 'all_aversive_time') || ...
       ~isfield(session, 'NeuralTime') || ...
       ~isfield(session, 'TriggerMid') || ...
       ~isfield(prediction_session, 'prediction_scores')
        return;
    end
    
    aversive_times = session.all_aversive_time;
    neural_time = session.NeuralTime;
    camera_time = session.TriggerMid;
    prediction_scores = prediction_session.prediction_scores;
    
    % Define period boundaries
    period_boundaries = zeros(7, 2);
    period_boundaries(1, :) = [neural_time(1), aversive_times(1)];
    
    for period = 2:min(length(aversive_times), 6)
        period_boundaries(period, :) = [aversive_times(period-1), aversive_times(period)];
    end
    
    if length(aversive_times) >= 6
        period_boundaries(7, :) = [aversive_times(6), neural_time(end)];
    end
    
    % Process each period
    for period = 1:7
        if all(period_boundaries(period, :) == 0)
            continue;
        end
        
        % Find predictions in this period
        prediction_indices = 1:20:length(camera_time)+1;
        period_predictions = [];
        
        for pred_idx = 1:(length(prediction_indices)-1)
            frame_start = prediction_indices(pred_idx);
            
            if camera_time(frame_start) >= period_boundaries(period, 1) && ...
               camera_time(frame_start) < period_boundaries(period, 2)
                
                [max_conf, dominant_beh] = max(prediction_scores(pred_idx, :));
                
                if max_conf > config.confidence_threshold
                    period_predictions = [period_predictions; dominant_beh];
                end
            end
        end
        
        % Calculate percentages and durations
        if ~isempty(period_predictions)
            for beh = 1:config.n_behaviors
                % Percentage (frequency)
                percentages(beh, period) = sum(period_predictions == beh) / ...
                                          length(period_predictions) * 100;
                
                % Duration (average bout length)
                % period_predictions is like [1, 1, 1, 7, 7, 7, 2, 2, ...]
                % where each value is a behavior at 1 second intervals
                behavior_binary = (period_predictions == beh);
                
                % Find bout onsets and offsets
                onset = find(diff([0; behavior_binary(:)]) == 1);
                offset = find(diff([behavior_binary(:); 0]) == -1);
                
                if ~isempty(onset) && ~isempty(offset)
                    % Calculate bout durations in seconds (predictions are at 1 Hz)
                    bout_durations = offset - onset + 1;
                    durations(beh, period) = mean(bout_durations);
                else
                    durations(beh, period) = 0;
                end
            end
        end
    end
end

function psth_data = extract_psth_data(sessions, prediction_sessions, group_assignment, config)
    % Extract behavioral data aligned to aversive noise onset
    
    psth_data = struct();
    psth_data.responder = [];
    psth_data.non_responder = [];
    
    n_sessions = length(sessions);
    
    for sess_idx = 1:n_sessions
        session = sessions{sess_idx};
        
        if isnan(group_assignment.goal_movement_change(sess_idx))
            continue;
        end
        
        if ~isfield(session, 'behavioral_matrix_full') || ...
           ~isfield(session, 'all_aversive_time') || ...
           ~isfield(session, 'NeuralTime') || ...
           sess_idx > length(prediction_sessions) || ...
           ~isfield(prediction_sessions, 'prediction_scores')
            continue;
        end
        
        aversive_times = session.all_aversive_time;
        camera_time = session.TriggerMid;
        prediction_scores = prediction_sessions(sess_idx).prediction_scores;
        
        % Determine group
        if group_assignment.is_responder(sess_idx)
            group_name = 'responder';
        else
            group_name = 'non_responder';
        end
        
        % Extract data around each aversive noise
        for noise_idx = 1:length(aversive_times)
            noise_time = aversive_times(noise_idx);
            time_window = [noise_time + config.psth_time_window(1), ...
                          noise_time + config.psth_time_window(2)];
            
            prediction_indices = 1:20:length(camera_time)+1;
            
            for pred_idx = 1:(length(prediction_indices)-1)
                frame_start = prediction_indices(pred_idx);
                
                if camera_time(frame_start) < time_window(1) || ...
                   camera_time(frame_start) > time_window(2)
                    continue;
                end
                
                rel_time = camera_time(frame_start) - noise_time;
                [max_conf, dominant_beh] = max(prediction_scores(pred_idx, :));
                
                if max_conf > config.confidence_threshold
                    data_point = struct();
                    data_point.session_id = sess_idx;
                    data_point.noise_number = noise_idx;
                    data_point.relative_time = rel_time;
                    data_point.behavior = dominant_beh;
                    data_point.confidence = max_conf;
                    
                    psth_data.(group_name) = [psth_data.(group_name); data_point];
                end
            end
        end
    end
end

function [psth_responder, psth_non_responder, bin_centers] = calculate_psth(psth_data, config)
    % Calculate PSTH for both groups
    
    time_bins = config.psth_time_window(1):config.psth_bin_size:config.psth_time_window(2);
    n_bins = length(time_bins) - 1;
    bin_centers = time_bins(1:end-1) + config.psth_bin_size/2;
    
    psth_responder = zeros(config.n_behaviors, n_bins);
    psth_non_responder = zeros(config.n_behaviors, n_bins);
    
    % Calculate for responders
    if ~isempty(psth_data.responder)
        for beh = 1:config.n_behaviors
            beh_data = psth_data.responder([psth_data.responder.behavior] == beh);
            
            for bin_idx = 1:n_bins
                count = sum([beh_data.relative_time] >= time_bins(bin_idx) & ...
                           [beh_data.relative_time] < time_bins(bin_idx + 1));
                psth_responder(beh, bin_idx) = count;
            end
        end
        psth_responder = psth_responder ./ sum(psth_responder, 1) * 100;
    end
    
    % Calculate for non-responders
    if ~isempty(psth_data.non_responder)
        for beh = 1:config.n_behaviors
            beh_data = psth_data.non_responder([psth_data.non_responder.behavior] == beh);
            
            for bin_idx = 1:n_bins
                count = sum([beh_data.relative_time] >= time_bins(bin_idx) & ...
                           [beh_data.relative_time] < time_bins(bin_idx + 1));
                psth_non_responder(beh, bin_idx) = count;
            end
        end
        psth_non_responder = psth_non_responder ./ sum(psth_non_responder, 1) * 100;
    end
end

function stats_results = perform_statistical_tests(behavior_percentages, psth_data, config)
    % Perform statistical comparisons between groups
    
    stats_results = struct();
    
    % Test 1: Compare Period 1 vs Period 2 for each group
    stats_results.within_group = cell(config.n_behaviors, 1);
    
    for beh = 1:config.n_behaviors
        resp_p1 = behavior_percentages.responder(:, beh, 1);
        resp_p2 = behavior_percentages.responder(:, beh, 2);
        nonresp_p1 = behavior_percentages.non_responder(:, beh, 1);
        nonresp_p2 = behavior_percentages.non_responder(:, beh, 2);
        
        valid_resp = ~isnan(resp_p1) & ~isnan(resp_p2);
        valid_nonresp = ~isnan(nonresp_p1) & ~isnan(nonresp_p2);
        
        stats_results.within_group{beh}.responder_pval = [];
        stats_results.within_group{beh}.nonresponder_pval = [];
        
        if sum(valid_resp) >= 3
            stats_results.within_group{beh}.responder_pval = ...
                signrank(resp_p1(valid_resp), resp_p2(valid_resp));
        end
        
        if sum(valid_nonresp) >= 3
            stats_results.within_group{beh}.nonresponder_pval = ...
                signrank(nonresp_p1(valid_nonresp), nonresp_p2(valid_nonresp));
        end
    end
    
    % Test 2: Compare groups at each period
    stats_results.between_group = cell(config.n_behaviors, 7);
    
    for beh = 1:config.n_behaviors
        for period = 1:7
            resp_data = behavior_percentages.responder(:, beh, period);
            nonresp_data = behavior_percentages.non_responder(:, beh, period);
            
            valid = ~isnan(resp_data) & ~isnan(nonresp_data);
            
            if sum(valid) >= 3
                stats_results.between_group{beh, period} = ...
                    ranksum(resp_data(valid), nonresp_data(valid));
            else
                stats_results.between_group{beh, period} = nan;
            end
        end
    end
end

function print_statistics(stats_results, config)
    % Print statistical results
    
    fprintf('=== STATISTICAL RESULTS ===\n\n');
    
    fprintf('Within-group comparisons (Period 1 vs Period 2):\n');
    fprintf('%-25s  Responders  Non-responders\n', 'Behavior');
    fprintf('----------------------------------------------------------\n');
    
    for beh = 1:config.n_behaviors
        resp_p = stats_results.within_group{beh}.responder_pval;
        nonresp_p = stats_results.within_group{beh}.nonresponder_pval;
        
        resp_str = format_pvalue(resp_p);
        nonresp_str = format_pvalue(nonresp_p);
        
        fprintf('%-25s  %-10s  %-10s\n', config.behavior_names{beh}, resp_str, nonresp_str);
    end
    
    fprintf('\nBetween-group comparisons across periods:\n');
    for beh = 1:config.n_behaviors
        fprintf('\n%s:\n', config.behavior_names{beh});
        fprintf('  Period:  ');
        for period = 1:7
            fprintf('P%-2d      ', period);
        end
        fprintf('\n  p-value: ');
        for period = 1:7
            p_val = stats_results.between_group{beh, period};
            fprintf('%-8s', format_pvalue(p_val));
        end
        fprintf('\n');
    end
end

function str = format_pvalue(p)
    % Format p-value for display
    if isempty(p) || isnan(p)
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