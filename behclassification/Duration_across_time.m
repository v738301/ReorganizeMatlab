%% ========================================================================
%  BEHAVIORAL PATCH DURATION ANALYSIS: Period × Behavior × SessionType
%  Analyzes duration/length of consecutive behavioral bouts
%  MODIFIED: Uses post-hoc pairwise comparisons with FDR correction
%  ========================================================================
%
%  Analysis: Duration ~ Period × Behavior × SessionType
%  SessionType: Aversive vs Reward
%  Periods: P1-P4 (matched across both session types)
%  Method: Extract all behavioral patches and measure their duration
%  Duration unit: Seconds (prediction frame rate = 1Hz, 1 window = 1 second)
%
%% ========================================================================

clear all
% close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== BEHAVIORAL PATCH DURATION ANALYSIS: AVERSIVE vs REWARD ===\n');
fprintf('Period × Behavior × SessionType\n\n');

config = struct();
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
config.n_behaviors = 7;
config.confidence_threshold = 0.3;
config.prediction_framerate = 1;  % 1Hz, so 1 window = 1 second

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
%  SECTION 3: EXTRACT AVERSIVE PATCH DURATIONS (P1-P4 only)
%  ========================================================================

fprintf('Extracting aversive patch durations (P1-P4)...\n');

% Initialize storage
aversive_data = struct();
aversive_data.session_id = [];
aversive_data.period = [];
aversive_data.behavior = [];
aversive_data.duration = [];  % in seconds

n_valid_aversive = 0;
total_patches_aversive = 0;

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

        if length(valid_dominant) < 2
            continue;
        end

        % Find consecutive patches (where behavior doesn't change)
        % Add sentinel values to detect changes at boundaries
        behavior_changes = [1; find(diff(valid_dominant) ~= 0) + 1; length(valid_dominant) + 1];

        % Extract each patch
        for patch_idx = 1:(length(behavior_changes) - 1)
            patch_start = behavior_changes(patch_idx);
            patch_end = behavior_changes(patch_idx + 1) - 1;

            patch_behavior = valid_dominant(patch_start);
            patch_duration_windows = patch_end - patch_start + 1;

            % Convert to seconds (1 window = 1 second at 1Hz)
            patch_duration_seconds = patch_duration_windows * config.prediction_framerate;

            % Store patch
            aversive_data.session_id(end+1) = n_valid_aversive;
            aversive_data.period(end+1) = period;
            aversive_data.behavior(end+1) = patch_behavior;
            aversive_data.duration(end+1) = patch_duration_seconds;

            total_patches_aversive = total_patches_aversive + 1;
        end
    end
end

fprintf('✓ Processed %d aversive sessions\n', n_valid_aversive);
fprintf('  Total patches extracted: %d\n', total_patches_aversive);
fprintf('  Mean patches per session: %.1f\n\n', total_patches_aversive / n_valid_aversive);

%% ========================================================================
%  SECTION 4: EXTRACT REWARD PATCH DURATIONS (P1-P4)
%  ========================================================================

fprintf('Extracting reward patch durations (P1-P4)...\n');

% Initialize storage
reward_data = struct();
reward_data.session_id = [];
reward_data.period = [];
reward_data.behavior = [];
reward_data.duration = [];  % in seconds

n_valid_reward = 0;
total_patches_reward = 0;

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

        if length(valid_dominant) < 2
            continue;
        end

        % Find consecutive patches
        behavior_changes = [1; find(diff(valid_dominant) ~= 0) + 1; length(valid_dominant) + 1];

        % Extract each patch
        for patch_idx = 1:(length(behavior_changes) - 1)
            patch_start = behavior_changes(patch_idx);
            patch_end = behavior_changes(patch_idx + 1) - 1;

            patch_behavior = valid_dominant(patch_start);
            patch_duration_windows = patch_end - patch_start + 1;

            % Convert to seconds
            patch_duration_seconds = patch_duration_windows * config.prediction_framerate;

            % Store patch
            reward_data.session_id(end+1) = n_valid_reward;
            reward_data.period(end+1) = period;
            reward_data.behavior(end+1) = patch_behavior;
            reward_data.duration(end+1) = patch_duration_seconds;

            total_patches_reward = total_patches_reward + 1;
        end
    end
end

fprintf('✓ Processed %d reward sessions\n', n_valid_reward);
fprintf('  Total patches extracted: %d\n', total_patches_reward);
fprintf('  Mean patches per session: %.1f\n\n', total_patches_reward / n_valid_reward);

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
combined_data.duration = [aversive_data.duration(:); reward_data.duration(:)];
combined_data.session_type = [aversive_data.session_type; reward_data.session_type];

% Convert to table
tbl = table(combined_data.session_id, ...
            combined_data.period, ...
            combined_data.behavior, ...
            combined_data.duration, ...
            combined_data.session_type, ...
            'VariableNames', {'Session', 'Period', 'Behavior', 'Duration', 'SessionType'});

% Convert to categorical
tbl.Session = categorical(tbl.Session);
tbl.Period = categorical(tbl.Period);
tbl.Behavior = categorical(tbl.Behavior, 1:7, config.behavior_names);
tbl.SessionType = categorical(tbl.SessionType);

fprintf('✓ Combined dataset created\n');
fprintf('  Total rows (patches): %d\n', height(tbl));
fprintf('  Sessions: %d total\n', length(unique(tbl.Session)));
fprintf('    - Aversive: %d sessions, %d patches\n', n_valid_aversive, total_patches_aversive);
fprintf('    - Reward: %d sessions, %d patches\n', n_valid_reward, total_patches_reward);
fprintf('  Periods: %d\n', length(unique(tbl.Period)));
fprintf('  Behaviors: %d\n\n', length(unique(tbl.Behavior)));

% Display duration statistics
fprintf('Duration statistics (seconds):\n');
fprintf('  Overall mean: %.2f ± %.2f\n', mean(tbl.Duration), std(tbl.Duration));
fprintf('  Median: %.2f\n', median(tbl.Duration));
fprintf('  Range: [%.2f, %.2f]\n\n', min(tbl.Duration), max(tbl.Duration));

% Display first 20 rows
fprintf('First 20 rows of data:\n');
disp(tbl(1:min(20, height(tbl)), :));

%% ========================================================================
%  SECTION 6: FIT LME MODEL
%  ========================================================================

fprintf('\n=== FITTING LINEAR MIXED-EFFECTS MODEL ===\n');
fprintf('Formula: Duration ~ Period * Behavior * SessionType + (1|Session)\n');
fprintf('This may take a few minutes...\n\n');

try
    lme_full = fitlme(tbl, 'Duration ~ Period * Behavior * SessionType + (1|Session)', ...
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
%  SECTION 7: POST-HOC PAIRWISE COMPARISONS
%  Direct test: Aversive vs Reward at each Period × Behavior
%  ========================================================================

fprintf('=== POST-HOC PAIRWISE COMPARISONS ===\n');
fprintf('Testing: Aversive vs Reward at each Period × Behavior combination\n\n');

% Initialize results storage
comparison_results = struct();
comparison_results.behavior = [];
comparison_results.behavior_name = {};
comparison_results.period = [];
comparison_results.aversive_mean = [];
comparison_results.reward_mean = [];
comparison_results.difference = [];      % Aversive - Reward
comparison_results.SE_diff = [];
comparison_results.t_stat = [];
comparison_results.pvalue = [];
comparison_results.CI_lower = [];
comparison_results.CI_upper = [];

% Get degrees of freedom for t-tests
df = lme_full.DFE;

% Loop through all behavior × period combinations
for b = 1:config.n_behaviors
    beh_name = config.behavior_names{b};

    fprintf('%s:\n', beh_name);

    for p = 1:4
        % Create prediction table for both session types at this period/behavior
        pred_tbl_compare = table(...
            categorical([1; 1]), ...  % Dummy session
            categorical([p; p]), ...  % This period
            categorical([b; b], 1:7, config.behavior_names), ...  % This behavior
            categorical({'Aversive'; 'Reward'}), ...
            'VariableNames', {'Session', 'Period', 'Behavior', 'SessionType'});

        % Get predictions with confidence intervals
        [pred_vals, pred_CI] = predict(lme_full, pred_tbl_compare, 'Conditional', false);

        % Extract values
        aversive_mean = pred_vals(1);
        reward_mean = pred_vals(2);
        difference = aversive_mean - reward_mean;

        % Calculate SE from confidence intervals
        % CI = estimate ± 1.96*SE, so SE = (CI_upper - CI_lower) / (2*1.96)
        SE_aversive = (pred_CI(1,2) - pred_CI(1,1)) / (2 * 1.96);
        SE_reward = (pred_CI(2,2) - pred_CI(2,1)) / (2 * 1.96);

        % SE of difference (assuming independence)
        SE_diff = sqrt(SE_aversive^2 + SE_reward^2);

        % T-test for difference
        t_stat = difference / SE_diff;
        pval = 2 * (1 - tcdf(abs(t_stat), df));

        % 95% CI for difference
        CI_lower = difference - 1.96 * SE_diff;
        CI_upper = difference + 1.96 * SE_diff;

        % Store results
        comparison_results.behavior(end+1) = b;
        comparison_results.behavior_name{end+1} = beh_name;
        comparison_results.period(end+1) = p;
        comparison_results.aversive_mean(end+1) = aversive_mean;
        comparison_results.reward_mean(end+1) = reward_mean;
        comparison_results.difference(end+1) = difference;
        comparison_results.SE_diff(end+1) = SE_diff;
        comparison_results.t_stat(end+1) = t_stat;
        comparison_results.pvalue(end+1) = pval;
        comparison_results.CI_lower(end+1) = CI_lower;
        comparison_results.CI_upper(end+1) = CI_upper;

        % Print results
        fprintf('  P%d: Aver=%.2fs, Rew=%.2fs, Diff=%.2fs (SE=%.2f), t=%.2f, p=%.4f', ...
               p, aversive_mean, reward_mean, difference, SE_diff, t_stat, pval);

        if pval < 0.001
            fprintf(' ***\n');
        elseif pval < 0.01
            fprintf(' **\n');
        elseif pval < 0.05
            fprintf(' *\n');
        else
            fprintf('\n');
        end
    end
    fprintf('\n');
end

% Apply FDR correction for multiple comparisons
fprintf('Applying FDR correction for multiple comparisons...\n');
try
    % Use MATLAB's built-in mafdr function (Bioinformatics Toolbox)
    comparison_results.pvalue_fdr = mafdr(comparison_results.pvalue, 'BHFDR', true);
catch ME
    fprintf('Warning: mafdr function not available. Using Bonferroni correction instead.\n');
    % Fallback to Bonferroni if mafdr not available
    comparison_results.pvalue_fdr = min(comparison_results.pvalue * length(comparison_results.pvalue), 1);
end

% Count significant comparisons
n_sig_uncorrected = sum(comparison_results.pvalue < 0.05);
n_sig_fdr = sum(comparison_results.pvalue_fdr < 0.05);

fprintf('  Uncorrected: %d/%d comparisons significant (p<0.05)\n', ...
        n_sig_uncorrected, length(comparison_results.pvalue));
fprintf('  FDR-corrected: %d/%d comparisons significant (q<0.05)\n\n', ...
        n_sig_fdr, length(comparison_results.pvalue));

% Reshape p-values into matrix for plotting
% Rows = behaviors, Columns = periods
period_pvals = reshape(comparison_results.pvalue_fdr, [4, config.n_behaviors])';

fprintf('Summary of significant comparisons (FDR-corrected):\n');
for b = 1:config.n_behaviors
    sig_periods = find(period_pvals(b, :) < 0.05);
    if ~isempty(sig_periods)
        fprintf('  %s: Periods %s\n', config.behavior_names{b}, ...
                mat2str(sig_periods));
    end
end
fprintf('\n');

%% ========================================================================
%  SECTION 8: VISUALIZE - Session-level mean durations
%  ========================================================================

fprintf('Creating visualization with session-level mean durations...\n');

% Calculate session-level mean durations
session_means_aversive = groupsummary(tbl(tbl.SessionType == 'Aversive', :), ...
                                      {'Session', 'Period', 'Behavior'}, 'mean', 'Duration');
session_means_reward = groupsummary(tbl(tbl.SessionType == 'Reward', :), ...
                                    {'Session', 'Period', 'Behavior'}, 'mean', 'Duration');

% Define colors
color_aversive = [1, 0.6, 0.6];      % Red (lighter for individual sessions)
color_reward = [0.6, 1, 0.6];        % Green (lighter for individual sessions)

figure('Position', [50, 50, 1800, 1000]);

ax = [];
for b = 1:config.n_behaviors
    ax(end+1) = subplot(3, 3, b);
    hold on;

    % Extract data for this behavior
    behavior_name = config.behavior_names{b};

    % Plot individual AVERSIVE sessions
    aversive_beh = session_means_aversive(session_means_aversive.Behavior == behavior_name, :);
    aversive_sessions = unique(aversive_beh.Session);

    for s = 1:length(aversive_sessions)
        sess_mask = aversive_beh.Session == aversive_sessions(s);
        sess_data = aversive_beh(sess_mask, :);

        % Sort by period
        [~, sort_idx] = sort(double(sess_data.Period));
        periods = double(sess_data.Period(sort_idx));
        duration_vals = sess_data.mean_Duration(sort_idx);

        % Plot with transparency
        plot(periods, duration_vals, 'o-', ...
             'Color', color_aversive, ...
             'LineWidth', 1, ...
             'MarkerSize', 4, ...
             'MarkerFaceColor', color_aversive, ...
             'HandleVisibility', 'off');
    end

    % Plot individual REWARD sessions
    reward_beh = session_means_reward(session_means_reward.Behavior == behavior_name, :);
    reward_sessions = unique(reward_beh.Session);

    for s = 1:length(reward_sessions)
        sess_mask = reward_beh.Session == reward_sessions(s);
        sess_data = reward_beh(sess_mask, :);

        % Sort by period
        [~, sort_idx] = sort(double(sess_data.Period));
        periods = double(sess_data.Period(sort_idx));
        duration_vals = sess_data.mean_Duration(sort_idx);

        % Plot with transparency
        plot(periods, duration_vals, 's-', ...
             'Color', color_reward, ...
             'LineWidth', 1, ...
             'MarkerSize', 4, ...
             'MarkerFaceColor', color_reward, ...
             'HandleVisibility', 'off');
    end

    % Calculate and plot mean lines across sessions
    mean_aversive = zeros(4, 1);
    mean_reward = zeros(4, 1);

    for p = 1:4
        % Aversive mean
        aversive_period_data = aversive_beh.mean_Duration(double(aversive_beh.Period) == p);
        if ~isempty(aversive_period_data)
            mean_aversive(p) = nanmean(aversive_period_data);
        else
            mean_aversive(p) = NaN;
        end

        % Reward mean
        reward_period_data = reward_beh.mean_Duration(double(reward_beh.Period) == p);
        if ~isempty(reward_period_data)
            mean_reward(p) = nanmean(reward_period_data);
        else
            mean_reward(p) = NaN;
        end
    end

    % Plot mean lines (thick, opaque)
    h_aver = plot(1:4, mean_aversive, 'o-', ...
                'LineWidth', 3, 'MarkerSize', 10, ...
                'Color', [1,0,0], ...
                'MarkerFaceColor', [1,0,0], ...
                'DisplayName', 'Aversive (mean)');

    h_rew = plot(1:4, mean_reward, 's-', ...
                'LineWidth', 3, 'MarkerSize', 10, ...
                'Color', [0,0.6,0], ...
                'MarkerFaceColor', [0,0.6,0], ...
                'DisplayName', 'Reward (mean)');

    % Get y-axis limits for star placement
    all_vals = [mean_aversive; mean_reward; ...
                aversive_beh.mean_Duration; reward_beh.mean_Duration];
    y_min = min(all_vals);
    y_max = max(all_vals);
    y_range = y_max - y_min;

    if y_range == 0
        y_range = 1;
    end

    % Add significance stars for each period
    star_y = y_max + 0.15 * y_range;

    for p = 1:4
        p_val = period_pvals(b, p);

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
    ylabel('Patch Duration (seconds)', 'FontSize', 11);
    xticks(1:4);
    xticklabels({'P1', 'P2', 'P3', 'P4'});

    % Adjust y-limits to accommodate stars
    ylim([max(0, y_min - 0.1 * y_range), y_max + 0.25 * y_range]);

    if b == 1
        legend([h_aver, h_rew], 'Location', 'northwest', 'FontSize', 10);
    end

    grid on;
    set(gca, 'FontSize', 10);
    hold off;
end

% Link axes for better comparison
linkaxes(ax, 'xy');

% Add overall title
sgtitle({'Behavioral Patch Duration: Aversive vs Reward', ...
         'Aversive (red); Reward (green)', ...
         'Transparent lines = individual sessions; Thick lines = group means', ...
         'Stars: * q<0.05, ** q<0.01, *** q<0.001 (FDR-corrected pairwise comparisons)'}, ...
        'FontSize', 14, 'FontWeight', 'bold');

fprintf('✓ Visualization complete\n');
fprintf('  Plotted %d aversive sessions (red circles)\n', length(aversive_sessions));
fprintf('  Plotted %d reward sessions (green squares)\n\n', length(reward_sessions));

%% ========================================================================
%  SECTION 9: MODEL PREDICTIONS FIGURE
%  ========================================================================

fprintf('\nCreating model predictions figure...\n');

% Generate predictions from the model
periods = [1, 2, 3, 4];
behaviors = 1:config.n_behaviors;
session_types = {'Aversive', 'Reward'};

% Create prediction grid
pred_grid = [];
for st = 1:length(session_types)
    for b = 1:length(behaviors)
        for p = 1:length(periods)
            pred_grid = [pred_grid; p, b, st];
        end
    end
end

% Create table with all required variables
pred_tbl = table(categorical(ones(size(pred_grid, 1), 1)), ...  % Dummy session
                 categorical(pred_grid(:,1)), ...
                 categorical(pred_grid(:,2), 1:7, config.behavior_names), ...
                 categorical(pred_grid(:,3), 1:2, session_types), ...
                 'VariableNames', {'Session', 'Period', 'Behavior', 'SessionType'});

% Get predictions
try
    [pred_duration, pred_CI] = predict(lme_full, pred_tbl, 'Conditional', false);
    fprintf('✓ Model predictions generated (population-level)\n');
catch ME
    fprintf('❌ Failed to generate predictions: %s\n', ME.message);
    pred_duration = [];
end

if ~isempty(pred_duration)
    % Reshape predictions for plotting
    pred_matrix = reshape(pred_duration, [4, config.n_behaviors, 2]);
    pred_CI_lower = reshape(pred_CI(:,1), [4, config.n_behaviors, 2]);
    pred_CI_upper = reshape(pred_CI(:,2), [4, config.n_behaviors, 2]);

    % Create figure
    figure('Position', [100, 50, 1800, 1000]);

    ax_pred = [];
    for b = 1:config.n_behaviors
        ax_pred(end+1) = subplot(3, 3, b);
        hold on;

        % Extract predictions for this behavior
        aversive_pred = pred_matrix(:, b, 1);
        aversive_CI_lower = pred_CI_lower(:, b, 1);
        aversive_CI_upper = pred_CI_upper(:, b, 1);

        reward_pred = pred_matrix(:, b, 2);
        reward_CI_lower = pred_CI_lower(:, b, 2);
        reward_CI_upper = pred_CI_upper(:, b, 2);

        % Plot confidence intervals as shaded areas
        x_fill = [1:4, fliplr(1:4)];

        % Aversive CI
        y_fill_aver = [aversive_CI_lower', fliplr(aversive_CI_upper')];
        fill(x_fill, y_fill_aver, [1, 0.8, 0.8], ...
             'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');

        % Reward CI
        y_fill_rew = [reward_CI_lower', fliplr(reward_CI_upper')];
        fill(x_fill, y_fill_rew, [0.8, 1, 0.8], ...
             'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');

        % Plot prediction lines
        h_aver_pred = plot(1:4, aversive_pred, 'o-', ...
                    'LineWidth', 3, 'MarkerSize', 10, ...
                    'Color', [1,0,0], ...
                    'MarkerFaceColor', [1,0,0], ...
                    'DisplayName', 'Aversive (predicted)');

        h_rew_pred = plot(1:4, reward_pred, 's-', ...
                    'LineWidth', 3, 'MarkerSize', 10, ...
                    'Color', [0,0.6,0], ...
                    'MarkerFaceColor', [0,0.6,0], ...
                    'DisplayName', 'Reward (predicted)');

        % Get y-axis limits for star placement
        all_preds = [aversive_pred; reward_pred; ...
                     aversive_CI_lower; aversive_CI_upper; ...
                     reward_CI_lower; reward_CI_upper];
        y_min = min(all_preds);
        y_max = max(all_preds);
        y_range = y_max - y_min;

        if y_range == 0
            y_range = 1;
        end

        % Add significance stars
        star_y = y_max + 0.15 * y_range;

        for p = 1:4
            p_val = period_pvals(b, p);

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

        % Add text box with key statistics for significant behaviors
        if any(period_pvals(b, :) < 0.05)
            sig_periods = find(period_pvals(b, :) < 0.05);
            [min_pval, most_sig_period] = min(period_pvals(b, :));
            effect_size = abs(aversive_pred(most_sig_period) - reward_pred(most_sig_period));

            text_str = sprintf('P%d: q=%.3f\nΔ=%.1fs', ...
                             most_sig_period, min_pval, effect_size);

            text(0.98, 0.02, text_str, ...
                 'Units', 'normalized', ...
                 'FontSize', 9, ...
                 'BackgroundColor', [1, 1, 0.9], ...
                 'EdgeColor', [0.8, 0.4, 0], ...
                 'LineWidth', 1.5, ...
                 'HorizontalAlignment', 'right', ...
                 'VerticalAlignment', 'bottom');
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
        ylabel('Patch Duration (seconds)', 'FontSize', 11);
        xticks(1:4);
        xticklabels({'P1', 'P2', 'P3', 'P4'});

        % Adjust y-limits
        ylim([max(0, y_min - 0.1 * y_range), y_max + 0.25 * y_range]);

        if b == 1
            legend([h_aver_pred, h_rew_pred], 'Location', 'northwest', 'FontSize', 10);
        end

        grid on;
        set(gca, 'FontSize', 10);
        hold off;
    end

    % Link axes
    linkaxes(ax_pred, 'xy');

    % Add overall title
    sgtitle({'Model Predictions: Behavioral Patch Duration (Aversive vs Reward)', ...
             'Lines = LME model predictions; Shaded areas = 95% confidence intervals', ...
             'Stars: * q<0.05, ** q<0.01, *** q<0.001 (FDR-corrected pairwise comparisons)', ...
             'Text boxes show most significant period with q-value and effect size (Δ)'}, ...
            'FontSize', 14, 'FontWeight', 'bold');

    fprintf('✓ Model predictions figure complete\n\n');
end

%% ========================================================================
%  SECTION 10: SAVE RESULTS
%  ========================================================================

% fprintf('Saving results...\n');
%
% results = struct();
% results.config = config;
% results.aversive_data = aversive_data;
% results.reward_data = reward_data;
% results.combined_data = combined_data;
% results.tbl = tbl;
% results.lme_full = lme_full;
% results.comparison_results = comparison_results;
% results.period_pvals = period_pvals;
% results.n_aversive = n_valid_aversive;
% results.n_reward = n_valid_reward;
% results.total_patches_aversive = total_patches_aversive;
% results.total_patches_reward = total_patches_reward;
%
% save('duration_aversive_vs_reward_results.mat', 'results');
%
% fprintf('✓ Results saved to: duration_aversive_vs_reward_results.mat\n');
fprintf('\n=== ANALYSIS COMPLETE ===\n');
