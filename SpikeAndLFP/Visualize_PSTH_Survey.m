%% ========================================================================
%  VISUALIZE PSTH SURVEY RESULTS
%  Loads and visualizes PSTH analysis results for all event types
%  ========================================================================
%
%  REWRITTEN: Modular helper functions for better maintainability
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: VISUALIZATION CONFIGURATION
%  ========================================================================

fprintf('=== VISUALIZING PSTH SURVEY RESULTS ===\n\n');

viz_config = struct();

% ========== HEATMAP SORTING MODE ==========
% 'independent': Each event heatmap sorted by its own peak times
% 'aligned': All event heatmaps sorted by the same reference event's peak times
viz_config.sort_mode = 'aligned';           % 'independent' or 'aligned'
viz_config.reference_event = 'IR1ON';       % Reference event for 'aligned' mode

% Figure appearance
viz_config.heatmap_clim = [-5, 5];              % Color limits for z-score heatmaps
viz_config.trace_ylim = [-2, 2];                % Y-axis limits for mean traces ([] = auto)
viz_config.trace_xlim = [];                     % X-axis limits for traces ([] = auto)

% Event comparison plots
viz_config.responsive_ylim = [0, 100];          % Y-axis for % responsive units
viz_config.avg_response_ylim = [];              % Y-axis for avg response [0-0.5s] ([] = auto)
viz_config.peak_magnitude_ylim = [-20, 20];     % Y-axis for peak z-score ([] = auto)
viz_config.latency_xlim = [];                   % X-axis for latency histograms ([] = auto)
viz_config.latency_ylim = [];                   % Y-axis for latency histograms ([] = auto)
viz_config.n_events_to_show_latency = 7;        % Number of events in latency plots

% Session type colors
viz_config.colors = struct();
viz_config.colors.aversive = [1, 0.6, 0.6];     % Light red
viz_config.colors.reward = [0.6, 1, 0.6];       % Light green
viz_config.colors.aversive_dark = [1, 0, 0];    % Red
viz_config.colors.reward_dark = [0, 0.6, 0];    % Green

fprintf('Configuration:\n');
fprintf('  Sorting mode: %s', viz_config.sort_mode);
if strcmp(viz_config.sort_mode, 'aligned')
    fprintf(' (reference: %s)', viz_config.reference_event);
end
fprintf('\n');
fprintf('  Heatmap CLim: [%.1f, %.1f]\n', viz_config.heatmap_clim(1), viz_config.heatmap_clim(2));
fprintf('  Latency plots: showing first %d events\n\n', viz_config.n_events_to_show_latency);

%% ========================================================================
%  SECTION 2: LOAD RESULTS
%  ========================================================================

fprintf('Loading PSTH results...\n');

if ~exist('PSTH_Survey_Results.mat', 'file')
    error('PSTH_Survey_Results.mat not found! Run PSTH_Survey_Analysis.m first.');
end

load('PSTH_Survey_Results.mat', 'results');

% Extract data
unit_data = results.unit_data;
config = results.config;
time_centers = results.time_centers;
n_units = results.n_units_total;

fprintf('✓ Data loaded\n');
fprintf('  Total units: %d\n', n_units);
fprintf('  Time window: [%.1f, %.1f] sec\n', config.psth_window(1), config.psth_window(2));
fprintf('  Bin size: %.0f ms\n\n', config.psth_bin_size * 1000);

%% ========================================================================
%  SECTION 3: PREPARE EVENT TYPES AND LABELS
%  ========================================================================

[event_types, event_labels] = getEventTypesAndLabels(config);
fprintf('Total event types to visualize: %d\n\n', length(event_types));

%% ========================================================================
%  SECTION 4: SEPARATE UNITS BY SESSION TYPE
%  ========================================================================

fprintf('Separating units by session type...\n');

aversive_units = find(strcmp({unit_data.session_type}, 'Aversive'));
reward_units = find(strcmp({unit_data.session_type}, 'Reward'));

fprintf('  Aversive units: %d\n', length(aversive_units));
fprintf('  Reward units: %d\n\n', length(reward_units));

%% ========================================================================
%  SECTION 5: CALCULATE REFERENCE SORTING ORDER (IF ALIGNED MODE)
%  ========================================================================

[reference_sort_aver, reference_sort_rew] = calculateReferenceSorting(...
    viz_config, unit_data, n_units, aversive_units, reward_units);

%% ========================================================================
%  SECTION 6: CREATE HEATMAP FIGURES FOR EACH EVENT TYPE
%  ========================================================================

fprintf('Creating heatmap figures...\n');

for ev = 1:length(event_types)
    event_type = event_types{ev};
    event_label = event_labels{ev};

    % Collect PSTH data for this event
    [aversive_psth, reward_psth] = collectPSTHData(unit_data, event_type, ...
        aversive_units, reward_units);

    % Skip if no data
    if isempty(aversive_psth) && isempty(reward_psth)
        fprintf('  Skipping %s: No data\n', event_label);
        continue;
    end

    % Create figure
    createEventHeatmapFigure(event_type, event_label, aversive_psth, reward_psth, ...
        time_centers, config, viz_config, reference_sort_aver, reference_sort_rew, ev);

    fprintf('  ✓ Created figure for %s\n', event_label);
end

fprintf('✓ All heatmap figures complete\n\n');

%% ========================================================================
%  SECTION 7: CALCULATE RESPONSE METRICS
%  ========================================================================

fprintf('Calculating response metrics across all events...\n');

event_metrics = calculateEventMetrics(unit_data, event_types, event_labels, ...
    time_centers, n_units);

fprintf('✓ Metrics calculated\n\n');

%% ========================================================================
%  SECTION 8: CREATE EVENT COMPARISON FIGURE
%  ========================================================================

fprintf('Creating event comparison figure...\n');

createEventComparisonFigure(event_metrics, event_types, event_labels, ...
    config, viz_config, time_centers);

fprintf('✓ Event comparison figure complete\n\n');

%% ========================================================================
%  SECTION 9: PRINT SUMMARY STATISTICS
%  ========================================================================

printSummaryStatistics(event_metrics, event_labels);

fprintf('\n=== VISUALIZATION COMPLETE ===\n');
fprintf('Total heatmap figures: %d\n', length(event_types));
fprintf('Event comparison figure: 1 (9 panels - 3×3 layout)\n');
fprintf('\nResponsiveness criteria:\n');
fprintf('  Positive (excitatory): Average z-score [0, 0.5]s > 2\n');
fprintf('  Negative (inhibitory): Average z-score [0, 0.5]s < -2\n');


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function [event_types, event_labels] = getEventTypesAndLabels(config)
% Generate lists of all event types and their display labels

    % Basic events
    event_types = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON', 'AversiveOnset', ...
                   'MovementOnset', 'MovementOffset'};
    event_labels = {'IR1ON Bouts', 'IR2ON Bouts', 'WP1ON Bouts', 'WP2ON Bouts', ...
                    'Aversive Onset', 'Movement Onset', 'Movement Offset'};

    % Add LSTM behavioral onsets/offsets (7 classes)
    for beh_class = 1:config.n_behaviors
        event_types{end+1} = ['Beh' num2str(beh_class) '_Onset'];
        event_labels{end+1} = [config.behavior_names{beh_class} ' Onset'];

        event_types{end+1} = ['Beh' num2str(beh_class) '_Offset'];
        event_labels{end+1} = [config.behavior_names{beh_class} ' Offset'];
    end

    % Add behavioral matrix feature onsets/offsets (8 features)
    if isfield(config, 'matrix_feature_names')
        for feat_idx = 1:config.n_matrix_features
            event_types{end+1} = [config.matrix_feature_names{feat_idx} '_Onset'];
            event_labels{end+1} = [config.matrix_feature_names{feat_idx} ' Onset'];

            event_types{end+1} = [config.matrix_feature_names{feat_idx} '_Offset'];
            event_labels{end+1} = [config.matrix_feature_names{feat_idx} ' Offset'];
        end
    end
end

function [reference_sort_aver, reference_sort_rew] = calculateReferenceSorting(...
    viz_config, unit_data, n_units, aversive_units, reward_units)
% Calculate reference sorting order for aligned mode

    reference_sort_aver = [];
    reference_sort_rew = [];

    if ~strcmp(viz_config.sort_mode, 'aligned')
        return;
    end

    fprintf('Calculating reference sorting order from: %s\n', viz_config.reference_event);

    % Collect reference event PSTH data
    field_name = [viz_config.reference_event '_zscore'];

    % Aversive units
    ref_psth_aver = [];
    for u = aversive_units
        if isfield(unit_data(u), field_name)
            psth = unit_data(u).(field_name);
            ref_psth_aver = [ref_psth_aver; psth'];
        end
    end

    % Reward units
    ref_psth_rew = [];
    for u = reward_units
        if isfield(unit_data(u), field_name)
            psth = unit_data(u).(field_name);
            ref_psth_rew = [ref_psth_rew; psth'];
        end
    end

    % Calculate sorting order
    if ~isempty(ref_psth_aver)
        [~, peak_idx] = max(abs(ref_psth_aver), [], 2);
        [~, reference_sort_aver] = sort(peak_idx);
        fprintf('  Aversive: %d units sorted\n', length(reference_sort_aver));
    else
        fprintf('  WARNING: No aversive data for reference event\n');
    end

    if ~isempty(ref_psth_rew)
        [~, peak_idx] = max(abs(ref_psth_rew), [], 2);
        [~, reference_sort_rew] = sort(peak_idx);
        fprintf('  Reward: %d units sorted\n', length(reference_sort_rew));
    else
        fprintf('  WARNING: No reward data for reference event\n');
    end
end

function [aversive_psth, reward_psth] = collectPSTHData(unit_data, event_type, ...
    aversive_units, reward_units)
% Collect z-scored PSTH data for a specific event type

    field_name = [event_type '_zscore'];

    % Collect aversive data
    aversive_psth = [];
    for u = aversive_units
        if isfield(unit_data(u), field_name)
            psth = unit_data(u).(field_name);
            aversive_psth = [aversive_psth; psth'];
        end
    end

    % Collect reward data
    reward_psth = [];
    for u = reward_units
        if isfield(unit_data(u), field_name)
            psth = unit_data(u).(field_name);
            reward_psth = [reward_psth; psth'];
        end
    end
end

function createEventHeatmapFigure(event_type, event_label, aversive_psth, reward_psth, ...
    time_centers, config, viz_config, reference_sort_aver, reference_sort_rew, fig_offset)
% Create 4-panel figure with heatmaps and average traces

    figure('Position', [50 + fig_offset*20, 50 + fig_offset*20, 1400, 800], 'Name', event_label);
    set(gcf, 'Color', 'white');

    % Top left: Aversive heatmap
    subplot(2, 2, 1);
    plotHeatmap(aversive_psth, time_centers, viz_config, reference_sort_aver, 'Aversive', event_label);

    % Top right: Reward heatmap
    subplot(2, 2, 2);
    plotHeatmap(reward_psth, time_centers, viz_config, reference_sort_rew, 'Reward', event_label);

    % Bottom left: Aversive average trace
    subplot(2, 2, 3);
    plotAverageTrace(aversive_psth, time_centers, config, viz_config, 'Aversive');

    % Bottom right: Reward average trace
    subplot(2, 2, 4);
    plotAverageTrace(reward_psth, time_centers, config, viz_config, 'Reward');

    sgtitle(sprintf('PSTH Analysis: %s', event_label), 'FontSize', 14, 'FontWeight', 'bold');
end

function plotHeatmap(psth_data, time_centers, viz_config, reference_sort, session_label, event_label)
% Plot heatmap for one session type

    if isempty(psth_data)
        text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
        title([session_label ' Sessions'], 'FontWeight', 'bold');
        return;
    end

    % Determine sorting order
    if strcmp(viz_config.sort_mode, 'aligned') && ~isempty(reference_sort)
        sort_order = reference_sort;
        sort_label = sprintf('sorted by %s peak', viz_config.reference_event);
    else
        [~, peak_idx] = max(abs(psth_data), [], 2);
        [~, sort_order] = sort(peak_idx);
        sort_label = 'sorted by peak';
    end

    % Plot heatmap
    imagesc(time_centers, 1:size(psth_data, 1), psth_data(sort_order, :));
    colormap(jet);
    colorbar;
    set(gca, 'CLim', viz_config.heatmap_clim);
    xlabel('Time from event (s)');
    ylabel(sprintf('Unit (%s)', sort_label));
    title(sprintf('%s Sessions - %s\n%d units', session_label, event_label, size(psth_data, 1)), ...
          'FontWeight', 'bold');

    % Event onset line
    hold on;
    plot([0, 0], ylim, 'w--', 'LineWidth', 2);
end

function plotAverageTrace(psth_data, time_centers, config, viz_config, session_label)
% Plot average PSTH trace with SEM

    if isempty(psth_data)
        return;
    end

    % Calculate mean and SEM
    mean_psth = mean(psth_data, 1, 'omitnan');
    sem_psth = std(psth_data, 0, 1, 'omitnan') / sqrt(size(psth_data, 1));

    hold on;

    % Shade baseline window
    patch([config.baseline_window(1), config.baseline_window(2), ...
           config.baseline_window(2), config.baseline_window(1)], ...
          [-100, -100, 100, 100], [0.9, 0.9, 0.9], 'EdgeColor', 'none');

    % Select color based on session type
    if strcmp(session_label, 'Aversive')
        fill_color = [1, 0.6, 0.6];
        line_color = 'r';
    else
        fill_color = [0.6, 1, 0.6];
        line_color = 'g';
    end

    % Plot mean ± SEM
    fill([time_centers, fliplr(time_centers)], ...
         [mean_psth + sem_psth, fliplr(mean_psth - sem_psth)], ...
         fill_color, 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    plot(time_centers, mean_psth, line_color, 'LineWidth', 2);

    % Event onset line
    plot([0, 0], ylim, 'k--', 'LineWidth', 2);

    xlabel('Time from event (s)');
    ylabel('Z-scored firing rate');
    title([session_label ': Mean ± SEM'], 'FontWeight', 'bold');
    grid on;

    % Apply limits
    if ~isempty(viz_config.trace_xlim)
        xlim(viz_config.trace_xlim);
    else
        xlim(config.psth_window);
    end
    if ~isempty(viz_config.trace_ylim)
        ylim(viz_config.trace_ylim);
    end
end

function event_metrics = calculateEventMetrics(unit_data, event_types, event_labels, ...
    time_centers, n_units)
% Calculate response metrics for all event types
% A unit is considered responsive if average z-score in [0, 0.5]s window > 2

    event_metrics = struct();

    % Define response window [0, 0.5]s after event
    response_window_bins = (time_centers >= 0) & (time_centers <= 0.5);

    for ev = 1:length(event_types)
        event_type = event_types{ev};

        % Initialize counters - separated by response type AND session type
        n_total_aver = 0;
        n_total_rew = 0;

        % Positive responders (excitatory)
        n_responsive_pos_aver = 0;
        n_responsive_pos_rew = 0;
        peak_response_pos_aver = [];
        peak_response_pos_rew = [];
        peak_latency_pos_aver = [];
        peak_latency_pos_rew = [];
        avg_response_pos_aver = [];
        avg_response_pos_rew = [];

        % Negative responders (inhibitory)
        n_responsive_neg_aver = 0;
        n_responsive_neg_rew = 0;
        peak_response_neg_aver = [];
        peak_response_neg_rew = [];
        peak_latency_neg_aver = [];
        peak_latency_neg_rew = [];
        avg_response_neg_aver = [];
        avg_response_neg_rew = [];

        % Loop through units
        for u = 1:n_units
            field_name = [event_type '_zscore'];
            if ~isfield(unit_data(u), field_name)
                continue;
            end

            n_events = unit_data(u).([event_type '_n_events']);
            if n_events == 0
                continue;
            end

            zscore_psth = unit_data(u).(field_name);

            % Calculate average response in [0, 0.5]s window
            avg_response_window = mean(zscore_psth(response_window_bins), 'omitnan');

            % CRITERION: Separate positive and negative responders
            is_positive = avg_response_window > 2;   % Excitatory
            is_negative = avg_response_window < -2;  % Inhibitory

            if strcmp(unit_data(u).session_type, 'Aversive')
                n_total_aver = n_total_aver + 1;

                if is_positive
                    n_responsive_pos_aver = n_responsive_pos_aver + 1;
                    [~, peak_idx] = max(abs(zscore_psth));
                    peak_response_pos_aver = [peak_response_pos_aver; zscore_psth(peak_idx)];
                    peak_latency_pos_aver = [peak_latency_pos_aver; time_centers(peak_idx)];
                    avg_response_pos_aver = [avg_response_pos_aver; avg_response_window];
                elseif is_negative
                    n_responsive_neg_aver = n_responsive_neg_aver + 1;
                    [~, peak_idx] = max(abs(zscore_psth));
                    peak_response_neg_aver = [peak_response_neg_aver; zscore_psth(peak_idx)];
                    peak_latency_neg_aver = [peak_latency_neg_aver; time_centers(peak_idx)];
                    avg_response_neg_aver = [avg_response_neg_aver; avg_response_window];
                end
            else
                n_total_rew = n_total_rew + 1;

                if is_positive
                    n_responsive_pos_rew = n_responsive_pos_rew + 1;
                    [~, peak_idx] = max(abs(zscore_psth));
                    peak_response_pos_rew = [peak_response_pos_rew; zscore_psth(peak_idx)];
                    peak_latency_pos_rew = [peak_latency_pos_rew; time_centers(peak_idx)];
                    avg_response_pos_rew = [avg_response_pos_rew; avg_response_window];
                elseif is_negative
                    n_responsive_neg_rew = n_responsive_neg_rew + 1;
                    [~, peak_idx] = max(abs(zscore_psth));
                    peak_response_neg_rew = [peak_response_neg_rew; zscore_psth(peak_idx)];
                    peak_latency_neg_rew = [peak_latency_neg_rew; time_centers(peak_idx)];
                    avg_response_neg_rew = [avg_response_neg_rew; avg_response_window];
                end
            end
        end

        % Store metrics
        event_metrics(ev).event_type = event_type;
        event_metrics(ev).event_label = event_labels{ev};
        event_metrics(ev).n_total_aver = n_total_aver;
        event_metrics(ev).n_total_rew = n_total_rew;

        % Positive responders
        event_metrics(ev).n_responsive_pos_aver = n_responsive_pos_aver;
        event_metrics(ev).n_responsive_pos_rew = n_responsive_pos_rew;
        event_metrics(ev).peak_response_pos_aver = peak_response_pos_aver;
        event_metrics(ev).peak_response_pos_rew = peak_response_pos_rew;
        event_metrics(ev).peak_latency_pos_aver = peak_latency_pos_aver;
        event_metrics(ev).peak_latency_pos_rew = peak_latency_pos_rew;
        event_metrics(ev).avg_response_pos_aver = avg_response_pos_aver;
        event_metrics(ev).avg_response_pos_rew = avg_response_pos_rew;

        % Negative responders
        event_metrics(ev).n_responsive_neg_aver = n_responsive_neg_aver;
        event_metrics(ev).n_responsive_neg_rew = n_responsive_neg_rew;
        event_metrics(ev).peak_response_neg_aver = peak_response_neg_aver;
        event_metrics(ev).peak_response_neg_rew = peak_response_neg_rew;
        event_metrics(ev).peak_latency_neg_aver = peak_latency_neg_aver;
        event_metrics(ev).peak_latency_neg_rew = peak_latency_neg_rew;
        event_metrics(ev).avg_response_neg_aver = avg_response_neg_aver;
        event_metrics(ev).avg_response_neg_rew = avg_response_neg_rew;
    end
end

function createEventComparisonFigure(event_metrics, event_types, event_labels, ...
    config, viz_config, time_centers)
% Create 9-panel comparison figure across all event types
% Separated by positive (excitatory) and negative (inhibitory) responses

    figure('Position', [100, 100, 2400, 1200], 'Name', 'Event Comparison');
    set(gcf, 'Color', 'white');

    % Row 1: Responsive units and average responses
    % Panel 1: Percentage of responsive units (positive and negative)
    subplot(3, 3, 1);
    plotResponsivePercentage(event_metrics, event_types, event_labels, viz_config);

    % Panel 2: Average response [0-0.5s] - POSITIVE responders
    subplot(3, 3, 2);
    plotAverageResponse(event_metrics, event_labels, viz_config, 'positive');

    % Panel 3: Average response [0-0.5s] - NEGATIVE responders
    subplot(3, 3, 3);
    plotAverageResponse(event_metrics, event_labels, viz_config, 'negative');

    % Row 2: Peak response magnitude
    % Panel 4: Peak response - POSITIVE responders
    subplot(3, 3, 4);
    plotPeakMagnitude(event_metrics, event_labels, viz_config, 'positive');

    % Panel 5: Peak response - NEGATIVE responders
    subplot(3, 3, 5);
    plotPeakMagnitude(event_metrics, event_labels, viz_config, 'negative');

    % Row 3: Peak latency
    % Panel 7: Peak latency - Aversive
    subplot(3, 3, 7);
    plotPeakLatency(event_metrics, event_labels, time_centers, config, viz_config, 'Aversive');

    % Panel 8: Peak latency - Reward
    subplot(3, 3, 8);
    plotPeakLatency(event_metrics, event_labels, time_centers, config, viz_config, 'Reward');

    sgtitle('Comparison Across Event Types (Positive vs Negative Responders)', 'FontSize', 14, 'FontWeight', 'bold');
end

function plotResponsivePercentage(event_metrics, event_types, event_labels, viz_config)
% Plot percentage of positive and negative responsive units

    pct_pos_aver = zeros(length(event_types), 1);
    pct_pos_rew = zeros(length(event_types), 1);
    pct_neg_aver = zeros(length(event_types), 1);
    pct_neg_rew = zeros(length(event_types), 1);

    for ev = 1:length(event_types)
        if event_metrics(ev).n_total_aver > 0
            pct_pos_aver(ev) = 100 * event_metrics(ev).n_responsive_pos_aver / event_metrics(ev).n_total_aver;
            pct_neg_aver(ev) = 100 * event_metrics(ev).n_responsive_neg_aver / event_metrics(ev).n_total_aver;
        end
        if event_metrics(ev).n_total_rew > 0
            pct_pos_rew(ev) = 100 * event_metrics(ev).n_responsive_pos_rew / event_metrics(ev).n_total_rew;
            pct_neg_rew(ev) = 100 * event_metrics(ev).n_responsive_neg_rew / event_metrics(ev).n_total_rew;
        end
    end

    hold on;
    bar_x = 1:length(event_types);
    bar_width = 0.2;

    % Positive responders
    bar(bar_x - 1.5*bar_width, pct_pos_aver, bar_width, ...
        'FaceColor', viz_config.colors.aversive_dark, 'DisplayName', 'Pos Aver');
    bar(bar_x - 0.5*bar_width, pct_pos_rew, bar_width, ...
        'FaceColor', viz_config.colors.reward_dark, 'DisplayName', 'Pos Rew');
    % Negative responders
    bar(bar_x + 0.5*bar_width, pct_neg_aver, bar_width, ...
        'FaceColor', viz_config.colors.aversive, 'DisplayName', 'Neg Aver');
    bar(bar_x + 1.5*bar_width, pct_neg_rew, bar_width, ...
        'FaceColor', viz_config.colors.reward, 'DisplayName', 'Neg Rew');

    set(gca, 'XTick', 1:length(event_types), 'XTickLabel', event_labels, 'XTickLabelRotation', 45);
    ylabel('% Responsive Units');
    title('Responsive Units (Pos/Neg)', 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 8);
    grid on;

    if ~isempty(viz_config.responsive_ylim)
        ylim(viz_config.responsive_ylim);
    end
end

function plotAverageResponse(event_metrics, event_labels, viz_config, response_type)
% Plot average response in [0, 0.5]s window - separated by response type
% response_type: 'positive' or 'negative'

    % Select appropriate fields based on response type
    if strcmp(response_type, 'positive')
        avg_field_aver = 'avg_response_pos_aver';
        avg_field_rew = 'avg_response_pos_rew';
        threshold_line = 2;
        title_str = 'Avg Response [0-0.5s] - POSITIVE';
    else
        avg_field_aver = 'avg_response_neg_aver';
        avg_field_rew = 'avg_response_neg_rew';
        threshold_line = -2;
        title_str = 'Avg Response [0-0.5s] - NEGATIVE';
    end

    % Collect average responses
    all_avg_aver = [];
    all_avg_rew = [];
    labels_aver = {};
    labels_rew = {};

    for ev = 1:length(event_metrics)
        all_avg_aver = [all_avg_aver; event_metrics(ev).(avg_field_aver)];
        labels_aver = [labels_aver; repmat({event_labels{ev}}, length(event_metrics(ev).(avg_field_aver)), 1)];

        all_avg_rew = [all_avg_rew; event_metrics(ev).(avg_field_rew)];
        labels_rew = [labels_rew; repmat({event_labels{ev}}, length(event_metrics(ev).(avg_field_rew)), 1)];
    end

    if isempty(all_avg_aver) && isempty(all_avg_rew)
        text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
        title(title_str, 'FontWeight', 'bold');
        return;
    end

    % Convert to categorical
    labels_aver_cat = categorical(labels_aver);
    labels_rew_cat = categorical(labels_rew);

    % Get all categories
    all_cats = unique([labels_aver; labels_rew]);

    % Find positions
    [~, pos_aver] = ismember(cellstr(unique(labels_aver_cat)), all_cats);
    [~, pos_rew] = ismember(cellstr(unique(labels_rew_cat)), all_cats);

    % Plot boxplots
    hold on;
    if ~isempty(all_avg_aver)
        boxplot(all_avg_aver, labels_aver_cat, 'Positions', pos_aver - 0.2, 'Widths', 0.3, ...
            'Colors', viz_config.colors.aversive_dark, 'Symbol', '');
    end
    if ~isempty(all_avg_rew)
        boxplot(all_avg_rew, labels_rew_cat, 'Positions', pos_rew + 0.2, 'Widths', 0.3, ...
            'Colors', viz_config.colors.reward_dark, 'Symbol', '');
    end

    % Add horizontal threshold line
    plot(xlim, [threshold_line, threshold_line], 'k--', 'LineWidth', 1.5);

    set(gca, 'XTick', 1:length(all_cats), 'XTickLabel', all_cats, 'XTickLabelRotation', 45);
    ylabel('Average Z-score [0-0.5s]');
    title(title_str, 'FontWeight', 'bold');
    grid on;

    if ~isempty(viz_config.avg_response_ylim)
        ylim(viz_config.avg_response_ylim);
    end

    if ~isempty(all_avg_aver) && ~isempty(all_avg_rew)
        legend_handles = [patch(NaN, NaN, viz_config.colors.aversive_dark), ...
                         patch(NaN, NaN, viz_config.colors.reward_dark)];
        legend(legend_handles, {'Aversive', 'Reward'}, 'Location', 'best');
    end
end

function plotPeakMagnitude(event_metrics, event_labels, viz_config, response_type)
% Plot peak response magnitude - separated by response type
% response_type: 'positive' or 'negative'

    % Select appropriate fields based on response type
    if strcmp(response_type, 'positive')
        peak_field_aver = 'peak_response_pos_aver';
        peak_field_rew = 'peak_response_pos_rew';
        title_str = 'Peak Response - POSITIVE';
    else
        peak_field_aver = 'peak_response_neg_aver';
        peak_field_rew = 'peak_response_neg_rew';
        title_str = 'Peak Response - NEGATIVE';
    end

    % Collect peak responses
    all_peaks_aver = [];
    all_peaks_rew = [];
    labels_aver = {};
    labels_rew = {};

    for ev = 1:length(event_metrics)
        all_peaks_aver = [all_peaks_aver; event_metrics(ev).(peak_field_aver)];
        labels_aver = [labels_aver; repmat({event_labels{ev}}, length(event_metrics(ev).(peak_field_aver)), 1)];

        all_peaks_rew = [all_peaks_rew; event_metrics(ev).(peak_field_rew)];
        labels_rew = [labels_rew; repmat({event_labels{ev}}, length(event_metrics(ev).(peak_field_rew)), 1)];
    end

    if isempty(all_peaks_aver) && isempty(all_peaks_rew)
        text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
        title(title_str, 'FontWeight', 'bold');
        return;
    end

    % Convert to categorical
    labels_aver_cat = categorical(labels_aver);
    labels_rew_cat = categorical(labels_rew);

    % Get all categories
    all_cats = unique([labels_aver; labels_rew]);

    % Find positions
    [~, pos_aver] = ismember(cellstr(unique(labels_aver_cat)), all_cats);
    [~, pos_rew] = ismember(cellstr(unique(labels_rew_cat)), all_cats);

    % Plot boxplots
    hold on;
    if ~isempty(all_peaks_aver)
        boxplot(all_peaks_aver, labels_aver_cat, 'Positions', pos_aver - 0.2, 'Widths', 0.3, ...
            'Colors', viz_config.colors.aversive_dark, 'Symbol', '');
    end
    if ~isempty(all_peaks_rew)
        boxplot(all_peaks_rew, labels_rew_cat, 'Positions', pos_rew + 0.2, 'Widths', 0.3, ...
            'Colors', viz_config.colors.reward_dark, 'Symbol', '');
    end

    set(gca, 'XTick', 1:length(all_cats), 'XTickLabel', all_cats, 'XTickLabelRotation', 45);
    ylabel('Peak Z-score');
    title(title_str, 'FontWeight', 'bold');
    grid on;

    if ~isempty(viz_config.peak_magnitude_ylim)
        ylim(viz_config.peak_magnitude_ylim);
    end

    if ~isempty(all_peaks_aver) && ~isempty(all_peaks_rew)
        legend_handles = [patch(NaN, NaN, viz_config.colors.aversive_dark), ...
                         patch(NaN, NaN, viz_config.colors.reward_dark)];
        legend(legend_handles, {'Aversive', 'Reward'}, 'Location', 'best');
    end
end

function plotPeakLatency(event_metrics, event_labels, time_centers, config, viz_config, session_label)
% Plot peak latency histograms - combines positive and negative responders

    hold on;

    n_to_show = min(viz_config.n_events_to_show_latency, length(event_metrics));
    colors = jet(n_to_show);

    for ev = 1:n_to_show
        % Combine positive and negative responders for the session type
        if strcmp(session_label, 'Aversive')
            latency_pos = event_metrics(ev).peak_latency_pos_aver;
            latency_neg = event_metrics(ev).peak_latency_neg_aver;
        else
            latency_pos = event_metrics(ev).peak_latency_pos_rew;
            latency_neg = event_metrics(ev).peak_latency_neg_rew;
        end

        % Combine positive and negative
        latency_data = [latency_pos; latency_neg];

        if ~isempty(latency_data)
            histogram(latency_data, time_centers, ...
                     'FaceColor', colors(ev, :), 'FaceAlpha', 0.4, 'EdgeColor', 'none', ...
                     'DisplayName', event_labels{ev});
        end
    end

    xlabel('Peak latency (s)');
    ylabel('Count');
    title(['Peak Response Latency - ' session_label], 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 8);
    grid on;

    if ~isempty(viz_config.latency_xlim)
        xlim(viz_config.latency_xlim);
    else
        xlim(config.psth_window);
    end
    if ~isempty(viz_config.latency_ylim)
        ylim(viz_config.latency_ylim);
    end
end

function printSummaryStatistics(event_metrics, event_labels)
% Print summary statistics to console

    fprintf('=== SUMMARY STATISTICS ===\n\n');
    fprintf('NOTE: Positive responsive if avg z-score [0-0.5s] > 2\n');
    fprintf('      Negative responsive if avg z-score [0-0.5s] < -2\n\n');

    for ev = 1:length(event_metrics)
        fprintf('%s:\n', event_labels{ev});

        % Positive responsive units
        pct_pos_aver = 100 * event_metrics(ev).n_responsive_pos_aver / max(event_metrics(ev).n_total_aver, 1);
        pct_pos_rew = 100 * event_metrics(ev).n_responsive_pos_rew / max(event_metrics(ev).n_total_rew, 1);

        fprintf('  POSITIVE Aversive: %d/%d units (%.1f%%)\n', ...
                event_metrics(ev).n_responsive_pos_aver, event_metrics(ev).n_total_aver, pct_pos_aver);
        fprintf('  POSITIVE Reward: %d/%d units (%.1f%%)\n', ...
                event_metrics(ev).n_responsive_pos_rew, event_metrics(ev).n_total_rew, pct_pos_rew);

        % Negative responsive units
        pct_neg_aver = 100 * event_metrics(ev).n_responsive_neg_aver / max(event_metrics(ev).n_total_aver, 1);
        pct_neg_rew = 100 * event_metrics(ev).n_responsive_neg_rew / max(event_metrics(ev).n_total_rew, 1);

        fprintf('  NEGATIVE Aversive: %d/%d units (%.1f%%)\n', ...
                event_metrics(ev).n_responsive_neg_aver, event_metrics(ev).n_total_aver, pct_neg_aver);
        fprintf('  NEGATIVE Reward: %d/%d units (%.1f%%)\n', ...
                event_metrics(ev).n_responsive_neg_rew, event_metrics(ev).n_total_rew, pct_neg_rew);

        % Average response - positive
        if ~isempty(event_metrics(ev).avg_response_pos_aver)
            fprintf('  Pos Aver avg z-score [0-0.5s]: %.2f ± %.2f\n', ...
                    mean(event_metrics(ev).avg_response_pos_aver), ...
                    std(event_metrics(ev).avg_response_pos_aver));
        end
        if ~isempty(event_metrics(ev).avg_response_pos_rew)
            fprintf('  Pos Rew avg z-score [0-0.5s]: %.2f ± %.2f\n', ...
                    mean(event_metrics(ev).avg_response_pos_rew), ...
                    std(event_metrics(ev).avg_response_pos_rew));
        end

        % Average response - negative
        if ~isempty(event_metrics(ev).avg_response_neg_aver)
            fprintf('  Neg Aver avg z-score [0-0.5s]: %.2f ± %.2f\n', ...
                    mean(event_metrics(ev).avg_response_neg_aver), ...
                    std(event_metrics(ev).avg_response_neg_aver));
        end
        if ~isempty(event_metrics(ev).avg_response_neg_rew)
            fprintf('  Neg Rew avg z-score [0-0.5s]: %.2f ± %.2f\n', ...
                    mean(event_metrics(ev).avg_response_neg_rew), ...
                    std(event_metrics(ev).avg_response_neg_rew));
        end

        fprintf('\n');
    end
end
