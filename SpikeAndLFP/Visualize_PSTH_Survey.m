%% ========================================================================
%  VISUALIZE PSTH SURVEY RESULTS
%  Loads and visualizes PSTH analysis results for all event types
%  ========================================================================

clear all
close all

%% ========================================================================
%  VISUALIZATION CONFIGURATION - EASY TO ADJUST
%  ========================================================================

viz_config = struct();

% ========== HEATMAP SORTING MODE ==========
% 'independent': Each event heatmap sorted by its own peak times
% 'aligned': All event heatmaps sorted by the same reference event's peak times
% viz_config.sort_mode = 'independent';           % 'independent' or 'aligned'
viz_config.sort_mode = 'aligned';           % 'independent' or 'aligned'
viz_config.reference_event = 'IR1ON';   % Reference event for 'aligned' mode
                                                 % Options: 'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON',
                                                 %          'AversiveOnset', 'MovementOnset', 'MovementOffset'
                                                 %          'Beh1_Onset', 'Beh2_Onset', etc.

% Figure 1: PSTH Heatmaps and Traces (per event type)
viz_config.heatmap_clim = [-5, 5];              % Color limits for z-score heatmaps
viz_config.trace_ylim = [-2, 2];                % Y-axis limits for mean traces ([] = auto)
viz_config.trace_xlim = [];                     % X-axis limits for traces ([] = use psth_window)

% Figure 2: Event Comparison
viz_config.responsive_ylim = [0, 100];          % Y-axis for % responsive units
viz_config.peak_magnitude_ylim = [-20, 20];     % Y-axis for peak z-score boxplot ([] = auto)
viz_config.latency_xlim = [];                   % X-axis for latency histograms ([] = use psth_window)
viz_config.latency_ylim = [];                   % Y-axis for latency histograms ([] = auto)
viz_config.n_events_to_show_latency = 7;        % Number of event types to show in latency plots

fprintf('=== VISUALIZING PSTH SURVEY RESULTS ===\n\n');
fprintf('Visualization Configuration:\n');
fprintf('  Sorting mode: %s', viz_config.sort_mode);
if strcmp(viz_config.sort_mode, 'aligned')
    fprintf(' (reference: %s)', viz_config.reference_event);
end
fprintf('\n');
fprintf('  Heatmap CLim: [%.1f, %.1f]\n', viz_config.heatmap_clim(1), viz_config.heatmap_clim(2));
fprintf('  Responsive units Y-limit: [%.0f, %.0f]%%\n', viz_config.responsive_ylim(1), viz_config.responsive_ylim(2));
fprintf('  Latency plots: showing first %d events\n\n', viz_config.n_events_to_show_latency);

%% ========================================================================
%  SECTION 1: LOAD RESULTS
%  ========================================================================

fprintf('Loading PSTH results...\n');

% Load results
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
fprintf('  Aversive sessions: %d\n', results.n_aversive_sessions);
fprintf('  Reward sessions: %d\n\n', results.n_reward_sessions);

% Event types to visualize
% Reward bouts, aversive onset, movement events
basic_event_types = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON', 'AversiveOnset', ...
                     'MovementOnset', 'MovementOffset'};
basic_event_labels = {'IR1ON Bouts', 'IR2ON Bouts', 'WP1ON Bouts', 'WP2ON Bouts', ...
                      'Aversive Onset', 'Movement Onset', 'Movement Offset'};

% Add behavioral onset/offset for 7 LSTM prediction classes
behavior_names = config.behavior_names;
event_types = basic_event_types;
event_labels = basic_event_labels;

for beh_class = 1:7
    event_types{end+1} = ['Beh' num2str(beh_class) '_Onset'];
    event_labels{end+1} = [behavior_names{beh_class} ' Onset'];

    event_types{end+1} = ['Beh' num2str(beh_class) '_Offset'];
    event_labels{end+1} = [behavior_names{beh_class} ' Offset'];
end

% Add behavioral matrix feature onset/offset for 8 features
if isfield(config, 'behavioral_matrix_names')
    behavioral_matrix_names = config.behavioral_matrix_names;
    n_bm_features = length(behavioral_matrix_names);
    for feat_idx = 1:n_bm_features
        event_types{end+1} = [behavioral_matrix_names{feat_idx} '_Onset'];
        event_labels{end+1} = [behavioral_matrix_names{feat_idx} ' Onset'];

        event_types{end+1} = [behavioral_matrix_names{feat_idx} '_Offset'];
        event_labels{end+1} = [behavioral_matrix_names{feat_idx} ' Offset'];
    end
end

fprintf('Total event types to visualize: %d\n', length(event_types));

%% ========================================================================
%  SECTION 2: SEPARATE AVERSIVE AND REWARD UNITS
%  ========================================================================

fprintf('Separating aversive and reward units...\n');

aversive_units = [];
reward_units = [];

for u = 1:n_units
    if strcmp(unit_data(u).session_type, 'Aversive')
        aversive_units = [aversive_units, u];
    else
        reward_units = [reward_units, u];
    end
end

fprintf('  Aversive units: %d\n', length(aversive_units));
fprintf('  Reward units: %d\n\n', length(reward_units));

%% ========================================================================
%  SECTION 3: FIGURE 1 - HEATMAPS FOR EACH EVENT TYPE
%  ========================================================================

fprintf('Creating Figure 1: PSTH heatmaps...\n');

% ========== Calculate reference sorting order if aligned mode ==========
reference_sort_aver = [];
reference_sort_rew = [];

if strcmp(viz_config.sort_mode, 'aligned')
    fprintf('  Calculating reference sorting order from: %s\n', viz_config.reference_event);

    % Collect reference event PSTH data
    ref_psth_aver = [];
    ref_psth_rew = [];

    for u = 1:n_units
        field_name = [viz_config.reference_event '_zscore'];
        if isfield(unit_data(u), field_name)
            psth = unit_data(u).(field_name);
            n_events = unit_data(u).([viz_config.reference_event '_n_events']);

            % Only include if there are events
%             if n_events > 0 && ~all(isnan(psth))
                if strcmp(unit_data(u).session_type, 'Aversive')
                    ref_psth_aver = [ref_psth_aver; psth'];
                else
                    ref_psth_rew = [ref_psth_rew; psth'];
                end
%             end
        end
    end

    % Calculate sorting order for reference event
    if ~isempty(ref_psth_aver)
        [~, peak_idx] = max(abs(ref_psth_aver), [], 2);
        [~, reference_sort_aver] = sort(peak_idx);
        fprintf('    Aversive: %d units sorted\n', length(reference_sort_aver));
    else
        fprintf('    WARNING: No aversive data for reference event\n');
    end

    if ~isempty(ref_psth_rew)
        [~, peak_idx] = max(abs(ref_psth_rew), [], 2);
        [~, reference_sort_rew] = sort(peak_idx);
        fprintf('    Reward: %d units sorted\n', length(reference_sort_rew));
    else
        fprintf('    WARNING: No reward data for reference event\n');
    end
end

% ========== Plot heatmaps for each event type ==========
for ev = 1:length(event_types)
    event_type = event_types{ev};
    event_label = event_labels{ev};

    % Collect z-scored PSTHs for all units
    aversive_psth = [];
    reward_psth = [];

    for u = 1:n_units
        field_name = [event_type '_zscore'];
        if isfield(unit_data(u), field_name)
            psth = unit_data(u).(field_name);
            n_events = unit_data(u).([event_type '_n_events']);

            % Only include if there are events
%             if n_events > 0 && ~all(isnan(psth))
                if strcmp(unit_data(u).session_type, 'Aversive')
                    aversive_psth = [aversive_psth; psth'];
                else
                    reward_psth = [reward_psth; psth'];
                end
%         end
        end
    end

    % Skip if no data
    if isempty(aversive_psth) && isempty(reward_psth)
        fprintf('  Skipping %s: No data\n', event_label);
        continue;
    end

    % Create figure
    figure('Position', [50 + ev*20, 50 + ev*20, 1400, 800], 'Name', event_label);
    set(gcf, 'Color', 'white');

    % Aversive heatmap
    subplot(2, 2, 1);
    if ~isempty(aversive_psth)
        % Determine sorting order based on mode
        if strcmp(viz_config.sort_mode, 'aligned') && ~isempty(reference_sort_aver)
            % Use reference event sorting
            sort_order = reference_sort_aver;
            sort_label = sprintf('sorted by %s peak', viz_config.reference_event);
        else
            % Sort by this event's own peak response time
            [~, peak_idx] = max(abs(aversive_psth), [], 2);
            [~, sort_order] = sort(peak_idx);
            sort_label = 'sorted by peak';
        end

        imagesc(time_centers, 1:size(aversive_psth, 1), aversive_psth(sort_order, :));
        colormap(jet);
        colorbar;
        set(gca, 'CLim', viz_config.heatmap_clim);
        xlabel('Time from event (s)');
        ylabel(sprintf('Unit (%s)', sort_label));
        title(sprintf('Aversive Sessions - %s\n%d units', event_label, size(aversive_psth, 1)), ...
              'FontWeight', 'bold');
        hold on;
        plot([0, 0], ylim, 'w--', 'LineWidth', 2);
    else
        text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
        title('Aversive Sessions', 'FontWeight', 'bold');
    end

    % Reward heatmap
    subplot(2, 2, 2);
    if ~isempty(reward_psth)
        % Determine sorting order based on mode
        if strcmp(viz_config.sort_mode, 'aligned') && ~isempty(reference_sort_rew)
            % Use reference event sorting
            sort_order = reference_sort_rew;
            sort_label = sprintf('sorted by %s peak', viz_config.reference_event);
        else
            % Sort by this event's own peak response time
            [~, peak_idx] = max(abs(reward_psth), [], 2);
            [~, sort_order] = sort(peak_idx);
            sort_label = 'sorted by peak';
        end

        imagesc(time_centers, 1:size(reward_psth, 1), reward_psth(sort_order, :));
        colormap(jet);
        colorbar;
        set(gca, 'CLim', viz_config.heatmap_clim);
        xlabel('Time from event (s)');
        ylabel(sprintf('Unit (%s)', sort_label));
        title(sprintf('Reward Sessions - %s\n%d units', event_label, size(reward_psth, 1)), ...
              'FontWeight', 'bold');
        hold on;
        plot([0, 0], ylim, 'w--', 'LineWidth', 2);
    else
        text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
        title('Reward Sessions', 'FontWeight', 'bold');
    end

    % Average PSTH traces - Aversive
    subplot(2, 2, 3);
    if ~isempty(aversive_psth)
        mean_psth = mean(aversive_psth, 1, 'omitnan');
        sem_psth = std(aversive_psth, 0, 1, 'omitnan') / sqrt(size(aversive_psth, 1));

        hold on;
        % Shade baseline window
        patch([config.baseline_window(1), config.baseline_window(2), ...
               config.baseline_window(2), config.baseline_window(1)], ...
              [-10, -10, 10, 10], [0.9, 0.9, 0.9], 'EdgeColor', 'none');

        % Plot mean ± SEM
        fill([time_centers, fliplr(time_centers)], ...
             [mean_psth + sem_psth, fliplr(mean_psth - sem_psth)], ...
             [1, 0.6, 0.6], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
        plot(time_centers, mean_psth, 'r-', 'LineWidth', 2);

        % Event onset
        plot([0, 0], ylim, 'k--', 'LineWidth', 2);

        xlabel('Time from event (s)');
        ylabel('Z-scored firing rate');
        title('Aversive: Mean ± SEM', 'FontWeight', 'bold');
        grid on;

        % Apply xlim and ylim from config
        if ~isempty(viz_config.trace_xlim)
            xlim(viz_config.trace_xlim);
        else
            xlim(config.psth_window);
        end
        if ~isempty(viz_config.trace_ylim)
            ylim(viz_config.trace_ylim);
        end
    end

    % Average PSTH traces - Reward
    subplot(2, 2, 4);
    if ~isempty(reward_psth)
        mean_psth = mean(reward_psth, 1, 'omitnan');
        sem_psth = std(reward_psth, 0, 1, 'omitnan') / sqrt(size(reward_psth, 1));

        hold on;
        % Shade baseline window
        patch([config.baseline_window(1), config.baseline_window(2), ...
               config.baseline_window(2), config.baseline_window(1)], ...
              [-10, -10, 10, 10], [0.9, 0.9, 0.9], 'EdgeColor', 'none');

        % Plot mean ± SEM
        fill([time_centers, fliplr(time_centers)], ...
             [mean_psth + sem_psth, fliplr(mean_psth - sem_psth)], ...
             [0.6, 1, 0.6], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
        plot(time_centers, mean_psth, 'g-', 'LineWidth', 2);

        % Event onset
        plot([0, 0], ylim, 'k--', 'LineWidth', 2);

        xlabel('Time from event (s)');
        ylabel('Z-scored firing rate');
        title('Reward: Mean ± SEM', 'FontWeight', 'bold');
        grid on;

        % Apply xlim and ylim from config
        if ~isempty(viz_config.trace_xlim)
            xlim(viz_config.trace_xlim);
        else
            xlim(config.psth_window);
        end
        if ~isempty(viz_config.trace_ylim)
            ylim(viz_config.trace_ylim);
        end
    end

    sgtitle(sprintf('PSTH Analysis: %s', event_label), 'FontSize', 14, 'FontWeight', 'bold');

    fprintf('  ✓ Created figure for %s\n', event_label);
end

fprintf('✓ Figure 1 complete (8 event types)\n\n');

%% ========================================================================
%  SECTION 4: FIGURE 2 - RESPONSE COMPARISON ACROSS EVENT TYPES
%  ========================================================================

fprintf('Creating Figure 2: Response comparison across events...\n');

figure('Position', [150, 150, 1800, 1000], 'Name', 'Event Comparison');
set(gcf, 'Color', 'white');

% Calculate response metrics for each event type
event_metrics = struct();

for ev = 1:length(event_types)
    event_type = event_types{ev};

    % Count responsive units (units with any significant bin)
    n_responsive_aver = 0;
    n_responsive_rew = 0;
    n_total_aver = 0;
    n_total_rew = 0;

    peak_response_aver = [];
    peak_response_rew = [];
    peak_latency_aver = [];
    peak_latency_rew = [];

    for u = 1:n_units
        field_name = [event_type '_significant'];
        if isfield(unit_data(u), field_name)
            n_events = unit_data(u).([event_type '_n_events']);

            if n_events > 0
                significant = unit_data(u).(field_name);
                zscore_psth = unit_data(u).([event_type '_zscore']);

                if strcmp(unit_data(u).session_type, 'Aversive')
                    n_total_aver = n_total_aver + 1;
                    if any(significant)
                        n_responsive_aver = n_responsive_aver + 1;

                        % Peak response and latency
                        [~, peak_idx] = max(abs(zscore_psth));
                        peak_response_aver = [peak_response_aver; zscore_psth(peak_idx)];
                        peak_latency_aver = [peak_latency_aver; time_centers(peak_idx)];
                    end
                else
                    n_total_rew = n_total_rew + 1;
                    if any(significant)
                        n_responsive_rew = n_responsive_rew + 1;

                        % Peak response and latency
                        [~, peak_idx] = max(abs(zscore_psth));
                        peak_response_rew = [peak_response_rew; zscore_psth(peak_idx)];
                        peak_latency_rew = [peak_latency_rew; time_centers(peak_idx)];
                    end
                end
            end
        end
    end

    % Store metrics
    event_metrics(ev).event_type = event_type;
    event_metrics(ev).event_label = event_labels{ev};
    event_metrics(ev).n_responsive_aver = n_responsive_aver;
    event_metrics(ev).n_total_aver = n_total_aver;
    event_metrics(ev).n_responsive_rew = n_responsive_rew;
    event_metrics(ev).n_total_rew = n_total_rew;
    event_metrics(ev).peak_response_aver = peak_response_aver;
    event_metrics(ev).peak_response_rew = peak_response_rew;
    event_metrics(ev).peak_latency_aver = peak_latency_aver;
    event_metrics(ev).peak_latency_rew = peak_latency_rew;
end

% Plot 1: Percentage of responsive units
subplot(2, 2, 1);
hold on;

pct_responsive_aver = zeros(length(event_types), 1);
pct_responsive_rew = zeros(length(event_types), 1);

for ev = 1:length(event_types)
    if event_metrics(ev).n_total_aver > 0
        pct_responsive_aver(ev) = 100 * event_metrics(ev).n_responsive_aver / event_metrics(ev).n_total_aver;
    end
    if event_metrics(ev).n_total_rew > 0
        pct_responsive_rew(ev) = 100 * event_metrics(ev).n_responsive_rew / event_metrics(ev).n_total_rew;
    end
end

bar_x = 1:length(event_types);
bar_width = 0.35;
bar(bar_x - bar_width/2, pct_responsive_aver, bar_width, 'FaceColor', [1, 0.6, 0.6], ...
    'DisplayName', 'Aversive');
bar(bar_x + bar_width/2, pct_responsive_rew, bar_width, 'FaceColor', [0.6, 1, 0.6], ...
    'DisplayName', 'Reward');

set(gca, 'XTick', 1:length(event_types), 'XTickLabel', event_labels, 'XTickLabelRotation', 45);
ylabel('% Responsive Units');
title('Responsive Units by Event Type', 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

% Apply ylim from config
if ~isempty(viz_config.responsive_ylim)
    ylim(viz_config.responsive_ylim);
end

% Plot 2: Peak response magnitude distribution
subplot(2, 2, 2);
hold on;

% Collect all peak responses
all_peaks_aver = [];
all_peaks_rew = [];
event_labels_aver = {};
event_labels_rew = {};

for ev = 1:length(event_types)
    all_peaks_aver = [all_peaks_aver; event_metrics(ev).peak_response_aver];
    event_labels_aver = [event_labels_aver; repmat({event_labels{ev}}, length(event_metrics(ev).peak_response_aver), 1)];

    all_peaks_rew = [all_peaks_rew; event_metrics(ev).peak_response_rew];
    event_labels_rew = [event_labels_rew; repmat({event_labels{ev}}, length(event_metrics(ev).peak_response_rew), 1)];
end

% Convert to categorical
event_labels_aver_cat = categorical(event_labels_aver);
event_labels_rew_cat = categorical(event_labels_rew);

% Get unique categories in EACH dataset
unique_cats_aver = unique(event_labels_aver_cat);
unique_cats_rew = unique(event_labels_rew_cat);

% Get all possible categories (union of both)
all_cats = unique([event_labels_aver; event_labels_rew]);

% Find positions for each dataset based on the full category list
[~, pos_aver] = ismember(cellstr(unique_cats_aver), all_cats);
[~, pos_rew] = ismember(cellstr(unique_cats_rew), all_cats);

% Plot with appropriate positions for each dataset
if ~isempty(all_peaks_aver)
    boxplot(all_peaks_aver, event_labels_aver_cat, 'Positions', pos_aver - 0.2, 'Widths', 0.3, ...
        'Colors', [1, 0, 0], 'Symbol', '');
end
hold on;
if ~isempty(all_peaks_rew)
    boxplot(all_peaks_rew, event_labels_rew_cat, 'Positions', pos_rew + 0.2, 'Widths', 0.3, ...
        'Colors', [0, 0.6, 0], 'Symbol', '');
end

% Set x-axis labels to show all categories
set(gca, 'XTick', 1:length(all_cats), 'XTickLabel', all_cats, 'XTickLabelRotation', 45);
ylabel('Peak Z-score');
title('Peak Response Magnitude', 'FontWeight', 'bold');
grid on;

% Apply ylim from config
if ~isempty(viz_config.peak_magnitude_ylim)
    ylim(viz_config.peak_magnitude_ylim);
end

% Add legend manually
if ~isempty(all_peaks_aver) && ~isempty(all_peaks_rew)
    legend_handles = [patch(NaN, NaN, [1, 0, 0]), patch(NaN, NaN, [0, 0.6, 0])];
    legend(legend_handles, {'Aversive', 'Reward'}, 'Location', 'best');
end

% Plot 3: Peak latency distribution
subplot(2, 2, 3);
hold on;

% Number of events to show (from config)
n_to_show = min(viz_config.n_events_to_show_latency, length(event_types));
colors_aver = jet(n_to_show);

for ev = 1:n_to_show
    if ~isempty(event_metrics(ev).peak_latency_aver)
        histogram(event_metrics(ev).peak_latency_aver, time_centers, ...
                 'FaceColor', colors_aver(ev, :), 'FaceAlpha', 0.4, 'EdgeColor', 'none', ...
                 'DisplayName', event_labels{ev});
    end
end

xlabel('Peak latency (s)');
ylabel('Count');
title('Peak Response Latency - Aversive', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Apply xlim and ylim from config
if ~isempty(viz_config.latency_xlim)
    xlim(viz_config.latency_xlim);
else
    xlim(config.psth_window);
end
if ~isempty(viz_config.latency_ylim)
    ylim(viz_config.latency_ylim);
end

subplot(2, 2, 4);
hold on;

for ev = 1:n_to_show
    if ~isempty(event_metrics(ev).peak_latency_rew)
        histogram(event_metrics(ev).peak_latency_rew, time_centers, ...
                 'FaceColor', colors_aver(ev, :), 'FaceAlpha', 0.4, 'EdgeColor', 'none', ...
                 'DisplayName', event_labels{ev});
    end
end

xlabel('Peak latency (s)');
ylabel('Count');
title('Peak Response Latency - Reward', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Apply xlim and ylim from config
if ~isempty(viz_config.latency_xlim)
    xlim(viz_config.latency_xlim);
else
    xlim(config.psth_window);
end
if ~isempty(viz_config.latency_ylim)
    ylim(viz_config.latency_ylim);
end

sgtitle('Comparison Across Event Types', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('✓ Figure 2 complete\n\n');

%% ========================================================================
%  SECTION 5: PRINT SUMMARY STATISTICS
%  ========================================================================

fprintf('=== SUMMARY STATISTICS ===\n\n');

for ev = 1:length(event_types)
    fprintf('%s:\n', event_labels{ev});
    fprintf('  Aversive: %d/%d units responsive (%.1f%%)\n', ...
            event_metrics(ev).n_responsive_aver, event_metrics(ev).n_total_aver, ...
            100 * event_metrics(ev).n_responsive_aver / max(event_metrics(ev).n_total_aver, 1));
    fprintf('  Reward: %d/%d units responsive (%.1f%%)\n', ...
            event_metrics(ev).n_responsive_rew, event_metrics(ev).n_total_rew, ...
            100 * event_metrics(ev).n_responsive_rew / max(event_metrics(ev).n_total_rew, 1));

    if ~isempty(event_metrics(ev).peak_response_aver)
        fprintf('  Aversive peak z-score: %.2f ± %.2f\n', ...
                mean(event_metrics(ev).peak_response_aver), ...
                std(event_metrics(ev).peak_response_aver));
    end
    if ~isempty(event_metrics(ev).peak_response_rew)
        fprintf('  Reward peak z-score: %.2f ± %.2f\n', ...
                mean(event_metrics(ev).peak_response_rew), ...
                std(event_metrics(ev).peak_response_rew));
    end

    if ~isempty(event_metrics(ev).peak_latency_aver)
        fprintf('  Aversive peak latency: %.3f ± %.3f s\n', ...
                mean(event_metrics(ev).peak_latency_aver), ...
                std(event_metrics(ev).peak_latency_aver));
    end
    if ~isempty(event_metrics(ev).peak_latency_rew)
        fprintf('  Reward peak latency: %.3f ± %.3f s\n', ...
                mean(event_metrics(ev).peak_latency_rew), ...
                std(event_metrics(ev).peak_latency_rew));
    end
    fprintf('\n');
end

fprintf('=== VISUALIZATION COMPLETE ===\n');
fprintf('Total figures created: %d\n', length(event_types) + 1);
