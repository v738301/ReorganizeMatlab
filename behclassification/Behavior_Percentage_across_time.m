%% ========================================================================
%  BEHAVIORAL PERCENTAGE ANALYSIS: Period × Behavior × SessionType
%  Session-level aggregation (one percentage per session/period/behavior)
%  ========================================================================
%
%  Analysis: Percentage ~ Period × Behavior × SessionType (Aversive/Reward)
%  Periods: P1-P4 (matched across both session types)
%  Method: Session-level aggregation, dominant behavior (confidence > 0.3)
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== BEHAVIORAL PERCENTAGE ANALYSIS ===\n');
fprintf('Period × Behavior × SessionType\n\n');

config = struct();
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
config.n_behaviors = 7;
config.confidence_threshold = 0.3;

%% ========================================================================
%  SECTION 2: LOAD DATA
%  ========================================================================

fprintf('Loading data...\n');

% Load aversive sessions
try
    coupling_data_aversive = load('27-Oct-2025_RewardAversive_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_aversive = coupling_data_aversive.all_session_metrics;
    pred_data_aversive = load('lstm_prediction_results_aversive_27-Oct-2025');
    prediction_sessions_aversive = pred_data_aversive.final_results.session_predictions;
    fprintf('✓ Loaded aversive data: %d sessions\n', length(sessions_aversive));
catch ME
    fprintf('❌ Failed to load aversive data: %s\n', ME.message);
    return;
end

% Load reward sessions
try
    coupling_data_reward = load('27-Oct-2025_RewardSeeking_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_reward = coupling_data_reward.all_session_metrics;
    pred_data_reward = load('lstm_prediction_results_reward_27-Oct-2025');
    prediction_sessions_reward = pred_data_reward.final_results.session_predictions;
    fprintf('✓ Loaded reward data: %d sessions\n\n', length(sessions_reward));
catch ME
    fprintf('❌ Failed to load reward data: %s\n', ME.message);
    return;
end

%% ========================================================================
%  SECTION 3: EXTRACT AVERSIVE DATA (P1-P4 only)
%  ========================================================================

fprintf('Extracting aversive session data (P1-P4)...\n');

% Initialize storage
aversive_data = struct();
aversive_data.session_id = [];
aversive_data.period = [];
aversive_data.behavior = [];
aversive_data.percentage = [];

n_valid_aversive = 0;

for sess_idx = 1:length(sessions_aversive)
    session = sessions_aversive{sess_idx};

    % Check required fields
    if ~isfield(session, 'all_aversive_time') || ...
       ~isfield(session, 'NeuralTime') || ...
       ~isfield(session, 'TriggerMid') || ...
       sess_idx > length(prediction_sessions_aversive)
        continue;
    end

    aversive_times = session.all_aversive_time;
    if length(aversive_times) < 6
        continue;
    end

    n_valid_aversive = n_valid_aversive + 1;

    neural_time = session.NeuralTime;
    prediction_scores = prediction_sessions_aversive(sess_idx).prediction_scores;
    prediction_ind = 1:20:length(session.TriggerMid);
    prediction_ind = prediction_ind + 10;
    prediction_time = session.TriggerMid(prediction_ind);

    % Define period boundaries (P1-P4 only)
    % P1: start to noise 1
    % P2: noise 1 to noise 2
    % P3: noise 2 to noise 3
    % P4: noise 3 to noise 4
    period_boundaries = [session.TriggerMid(1), ...
                         aversive_times(1:3)' + session.TriggerMid(1), ...
                         aversive_times(4) + session.TriggerMid(1)];

    % Process each period
    for period = 1:4
        period_start = period_boundaries(period);
        period_end = period_boundaries(period + 1);

        % Find prediction windows in this period
        prediction_idx = prediction_time >= period_start & prediction_time < period_end;

        if sum(prediction_idx) < 10
            continue;
        end

        % Get predictions for this period
        predictions_in_period = prediction_scores(prediction_idx, :);

        % Find dominant behavior for each window
        [max_confidence, dominant_beh] = max(predictions_in_period, [], 2);

        % Filter by confidence threshold
        valid_mask = max_confidence > config.confidence_threshold;
        valid_dominant = dominant_beh(valid_mask);

        total_valid = sum(valid_mask);

        if total_valid > 0
            % Calculate percentage for each behavior
            for beh = 1:config.n_behaviors
                count = sum(valid_dominant == beh);
                percentage = (count / total_valid) * 100;

                % Store data point
                aversive_data.session_id(end+1) = n_valid_aversive;
                aversive_data.period(end+1) = period;
                aversive_data.behavior(end+1) = beh;
                aversive_data.percentage(end+1) = percentage;
            end
        end
    end
end

fprintf('✓ Processed %d valid aversive sessions\n', n_valid_aversive);
fprintf('  Data points: %d\n\n', length(aversive_data.session_id));

%% ========================================================================
%  SECTION 4: EXTRACT REWARD DATA (P1-P4)
%  ========================================================================

fprintf('Extracting reward session data (P1-P4)...\n');

% Initialize storage
reward_data = struct();
reward_data.session_id = [];
reward_data.period = [];
reward_data.behavior = [];
reward_data.percentage = [];

n_valid_reward = 0;

for sess_idx = 1:length(sessions_reward)
    session = sessions_reward{sess_idx};

    % Check required fields
    if ~isfield(session, 'NeuralTime') || ...
       ~isfield(session, 'TriggerMid') || ...
       sess_idx > length(prediction_sessions_reward)
        continue;
    end

    n_valid_reward = n_valid_reward + 1;

    neural_time = session.NeuralTime;
    prediction_scores = prediction_sessions_reward(sess_idx).prediction_scores;
    prediction_ind = 1:20:length(session.TriggerMid);
    prediction_ind = prediction_ind + 10;
    prediction_time = session.TriggerMid(prediction_ind);

    % Define period boundaries based on time (in seconds)
    time_boundaries = [0, 8*60, 16*60, 24*60, 30*60];
    period_boundaries = [session.TriggerMid(1), ...
                         time_boundaries(2:end) + session.TriggerMid(1)];

    % Process each period
    for period = 1:4
        period_start = period_boundaries(period);
        period_end = period_boundaries(period + 1);

        % Find prediction windows in this period
        prediction_idx = prediction_time >= period_start & prediction_time < period_end;

        if sum(prediction_idx) < 10
            continue;
        end

        % Get predictions for this period
        predictions_in_period = prediction_scores(prediction_idx, :);

        % Find dominant behavior for each window
        [max_confidence, dominant_beh] = max(predictions_in_period, [], 2);

        % Filter by confidence threshold
        valid_mask = max_confidence > config.confidence_threshold;
        valid_dominant = dominant_beh(valid_mask);

        total_valid = sum(valid_mask);

        if total_valid > 0
            % Calculate percentage for each behavior
            for beh = 1:config.n_behaviors
                count = sum(valid_dominant == beh);
                percentage = (count / total_valid) * 100;

                % Store data point
                reward_data.session_id(end+1) = n_valid_reward;
                reward_data.period(end+1) = period;
                reward_data.behavior(end+1) = beh;
                reward_data.percentage(end+1) = percentage;
            end
        end
    end
end

fprintf('✓ Processed %d valid reward sessions\n', n_valid_reward);
fprintf('  Data points: %d\n\n', length(reward_data.session_id));

%% ========================================================================
%  SECTION 5: COMBINE DATASETS
%  ========================================================================

fprintf('Combining datasets...\n');

% Add SessionType column
aversive_data.session_type = repmat({'Aversive'}, length(aversive_data.session_id), 1);
reward_data.session_type = repmat({'Reward'}, length(reward_data.session_id), 1);

% Make session IDs unique across session types
max_aversive_id = max(aversive_data.session_id);
reward_data.session_id = reward_data.session_id + max_aversive_id;

% Combine all fields
combined_data = struct();
combined_data.session_id = [aversive_data.session_id(:); reward_data.session_id(:)];
combined_data.period = [aversive_data.period(:); reward_data.period(:)];
combined_data.behavior = [aversive_data.behavior(:); reward_data.behavior(:)];
combined_data.percentage = [aversive_data.percentage(:); reward_data.percentage(:)];
combined_data.session_type = [aversive_data.session_type; reward_data.session_type];

% Convert to table
tbl = table(combined_data.session_id, ...
            combined_data.period, ...
            combined_data.behavior, ...
            combined_data.percentage, ...
            combined_data.session_type, ...
            'VariableNames', {'Session', 'Period', 'Behavior', 'Percentage', 'SessionType'});

% Convert to categorical
tbl.Session = categorical(tbl.Session);
tbl.Period = categorical(tbl.Period);
tbl.Behavior = categorical(tbl.Behavior, 1:7, config.behavior_names);
tbl.SessionType = categorical(tbl.SessionType);

fprintf('✓ Combined dataset created\n');
fprintf('  Total rows: %d\n', height(tbl));
fprintf('  Sessions: %d (%d aversive + %d reward)\n', ...
        length(unique(tbl.Session)), n_valid_aversive, n_valid_reward);
fprintf('  Periods: %d\n', length(unique(tbl.Period)));
fprintf('  Behaviors: %d\n\n', length(unique(tbl.Behavior)));

% Display first 20 rows
fprintf('First 20 rows of data:\n');
disp(tbl(1:min(20, height(tbl)), :));

%% ========================================================================
%  SECTION 6: FIT LME MODEL
%  ========================================================================

fprintf('\n=== FITTING LINEAR MIXED-EFFECTS MODEL ===\n');
fprintf('Formula: Percentage ~ Period * Behavior * SessionType + (1|Session)\n');
fprintf('This may take a few minutes...\n\n');

try
    lme_full = fitlme(tbl, 'Percentage ~ Period * Behavior * SessionType + (1|Session)', ...
                      'FitMethod', 'REML');

    fprintf('✓ Full model fitted successfully\n\n');
    disp(lme_full);

catch ME
    fprintf('❌ ERROR fitting model: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for k = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
    end
    return;
end

%% ========================================================================
%  SECTION 7: GET PREDICTIONS FOR VISUALIZATION
%  ========================================================================

fprintf('\n=== GENERATING PREDICTIONS FOR VISUALIZATION ===\n');

% Create prediction grid: 4 periods × 7 behaviors × 2 session types
n_pred_total = 2 * config.n_behaviors * 4;  % = 56 predictions
pred_session = ones(n_pred_total, 1);
pred_period = zeros(n_pred_total, 1);
pred_behavior = zeros(n_pred_total, 1);
pred_sessiontype = cell(n_pred_total, 1);
session_type_names = {'Aversive', 'Reward'};

idx = 1;
for st = 1:2
    for b = 1:config.n_behaviors
        for p = 1:4
            pred_period(idx) = p;
            pred_behavior(idx) = b;
            pred_sessiontype{idx} = session_type_names{st};
            idx = idx + 1;
        end
    end
end

% Create prediction table
pred_grid = table(categorical(pred_session), ...
                  categorical(pred_period), ...
                  categorical(pred_behavior, 1:7, config.behavior_names), ...
                  categorical(pred_sessiontype), ...
                  'VariableNames', {'Session', 'Period', 'Behavior', 'SessionType'});

% Get predictions
[pred_full, pred_CI] = predict(lme_full, pred_grid, 'Conditional', false);

% Reshape to [period × behavior × session_type]
pred_array = reshape(pred_full, 4, config.n_behaviors, 2);

fprintf('✓ Generated predictions: %d data points\n', length(pred_full));
fprintf('  Reshaped to: [4 periods × 7 behaviors × 2 session types]\n\n');

%% ========================================================================
%  SECTION 8: EXTRACT PERIOD-SPECIFIC SIGNIFICANCE
%  ========================================================================

fprintf('=== TESTING PERIOD × SESSIONTYPE INTERACTION BY BEHAVIOR ===\n\n');

coef_table = lme_full.Coefficients;

% Store p-values for each behavior × period combination
% Rows = behaviors, Columns = periods (2, 3, 4)
% Period 1 is reference, so no test needed
period_pvals = nan(config.n_behaviors, 3);  % Periods 2, 3, 4

% For each behavior, find its three-way interaction terms
for b = 1:config.n_behaviors
    beh_name = config.behavior_names{b};

    fprintf('%s:\n', beh_name);

    % Find all three-way interaction terms for this behavior
    interaction_idx = contains(coef_table.Name, ['Behavior_', beh_name]) & ...
                     contains(coef_table.Name, 'Period') & ...
                     contains(coef_table.Name, 'SessionType');

    if any(interaction_idx)
        beh_interactions = coef_table(interaction_idx, :);
        fprintf('  Three-way interaction terms found: %d\n', height(beh_interactions));

        % Parse each interaction term to extract which period it refers to
        for i = 1:height(beh_interactions)
            term_name = beh_interactions.Name{i};

            % Extract period number from term name
            % Term format: "Period_X:Behavior_Y:SessionType_Z"
            period_match = regexp(term_name, 'Period_(\d+)', 'tokens');

            if ~isempty(period_match)
                period_num = str2double(period_match{1}{1});

                % Store p-value (Period 2 -> column 1, Period 3 -> column 2, etc.)
                if period_num >= 2 && period_num <= 4
                    period_pvals(b, period_num - 1) = beh_interactions.pValue(i);

                    fprintf('    Period %d: β=%.4f, p=%.4f', ...
                           period_num, ...
                           beh_interactions.Estimate(i), ...
                           beh_interactions.pValue(i));

                    if beh_interactions.pValue(i) < 0.05
                        fprintf(' ***');
                    end
                    fprintf('\n');
                end
            end
        end
    else
        fprintf('  No interaction terms found (reference category)\n');
    end
    fprintf('\n');
end

%% ========================================================================
%  SECTION 9: VISUALIZE - Three-way interaction plot
%  ========================================================================

fprintf('Creating visualization...\n');

figure('Position', [50, 50, 1800, 1000]);

ax = [];
for b = 1:config.n_behaviors
    ax(end+1) = subplot(3, 3, b);
    hold on;

    % Plot Aversive line (red circles)
    h_av = plot(1:4, pred_array(:, b, 1), 'o-', ...
                'LineWidth', 2.5, 'MarkerSize', 8, ...
                'Color', [0.8, 0.2, 0.2], ...
                'MarkerFaceColor', [0.8, 0.2, 0.2], ...
                'DisplayName', 'Aversive');

    % Plot Reward line (green squares)
    h_rw = plot(1:4, pred_array(:, b, 2), 's-', ...
                'LineWidth', 2.5, 'MarkerSize', 8, ...
                'Color', [0.2, 0.6, 0.2], ...
                'MarkerFaceColor', [0.2, 0.6, 0.2], ...
                'DisplayName', 'Reward');

    % Get y-axis limits for star placement
    y_vals = pred_array(:, b, :);
    y_min = min(y_vals(:));
    y_max = max(y_vals(:));
    y_range = y_max - y_min;

    % Add significance stars for each period (P2, P3, P4)
    star_y = y_max + 0.15 * y_range;  % Position stars above data

    for p = 2:4  % Periods 2, 3, 4 (Period 1 is reference)
        p_val = period_pvals(b, p - 1);

        if ~isnan(p_val)
            if p_val < 0.001
                star_text = '***';
                star_size = 14;
                star_color = [0.8, 0, 0];
            elseif p_val < 0.01
                star_text = '**';
                star_size = 13;
                star_color = [0.8, 0.2, 0];
            elseif p_val < 0.05
                star_text = '*';
                star_size = 12;
                star_color = [0.8, 0.4, 0];
            else
                star_text = '';
                star_size = 0;
            end

            % Plot star if significant
            if ~isempty(star_text)
                text(p, star_y, star_text, ...
                     'FontSize', star_size, ...
                     'FontWeight', 'bold', ...
                     'Color', star_color, ...
                     'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'bottom');
            end
        end
    end

    % Title - red if any period is significant
    title_str = config.behavior_names{b};
    title_color = 'k';

    if any(period_pvals(b, :) < 0.05)
        title_color = 'r';
    end

    title(title_str, 'FontSize', 13, 'FontWeight', 'bold', 'Color', title_color);

    % Formatting
    xlabel('Period', 'FontSize', 11);
    ylabel('Percentage (%)', 'FontSize', 11);
    xticks(1:4);
    xticklabels({'P1', 'P2', 'P3', 'P4'});

    % Adjust y-limits to accommodate stars
    ylim([y_min - 0.1 * y_range, y_max + 0.25 * y_range]);

    if b == 1
        legend([h_av, h_rw], 'Location', 'northwest', 'FontSize', 10);
    end

    grid on;
    set(gca, 'FontSize', 10);
    hold off;
end

% Link axes for better comparison
linkaxes(ax, 'xy');

% Add overall title
sgtitle({'Behavioral Percentages: Period × Behavior × SessionType', ...
         'Stars above each period indicate significant Period×SessionType interaction for that behavior', ...
         '* p<0.05, ** p<0.01, *** p<0.001'}, ...
        'FontSize', 15, 'FontWeight', 'bold');

fprintf('✓ Visualization complete\n\n');
fprintf('=== ANALYSIS COMPLETE ===\n');
