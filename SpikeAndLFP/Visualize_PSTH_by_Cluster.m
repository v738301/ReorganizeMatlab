%% ========================================================================
%  VISUALIZE PSTH RESPONSES BY CLUSTER
%  Analyzes PSTH responses grouped by cluster assignments
%  Shows whether different cluster types have distinct PSTH profiles
%  ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== VISUALIZING PSTH RESPONSES BY CLUSTER ===\n\n');

viz_config = struct();

% Visualization settings
viz_config.heatmap_clim = [-5, 5];              % Color limits for z-score heatmaps
viz_config.trace_ylim = [-2, 2];                % Y-axis limits for mean traces
viz_config.trace_xlim = [-2, 2];                % X-axis limits for mean traces
viz_config.show_individual_units = false;       % Show individual unit traces (may be cluttered)
viz_config.min_units_per_cluster = 3;           % Minimum units to show cluster

% Event types to visualize (set to 'all' or specify list)
viz_config.events_to_show = 'all';  % Or: {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON', 'AversiveOnset'}

% Cluster colors (will use lines() colormap)
viz_config.cluster_colormap = @lines;

%% ========================================================================
%  SECTION 2: LOAD DATA
%  ========================================================================

fprintf('Loading data...\n');

% Load cluster assignments
% if ~exist('unit_cluster_assignments.mat', 'file')
%     error('unit_cluster_assignments.mat not found! Run Unit_Feature_Comprehensive_Heatmap.m first.');
% end
% load('unit_cluster_assignments.mat', 'cluster_assignments_output');
load('unit_cluster_assignments_PSTH_TimeSeries.mat', 'cluster_assignments_output');

% Load PSTH results
if ~exist('PSTH_Survey_Results.mat', 'file')
    error('PSTH_Survey_Results.mat not found! Run PSTH_Survey_Analysis.m first.');
end
load('PSTH_Survey_Results.mat', 'results');

fprintf('✓ Data loaded\n');
fprintf('  PSTH units: %d\n', results.n_units_total);
fprintf('  Aversive cluster assignments: %d\n', height(cluster_assignments_output.Aversive));
fprintf('  Reward cluster assignments: %d\n\n', height(cluster_assignments_output.Reward));

%% ========================================================================
%  SECTION 3: MATCH UNITS BETWEEN CLUSTER ASSIGNMENTS AND PSTH DATA
%  ========================================================================

fprintf('Matching units between cluster assignments and PSTH data...\n');

unit_data = results.unit_data;
time_centers = results.time_centers;
config = results.config;

% Get all event types dynamically from unit_data field names
% Find all fields ending with "_zscore"
all_fields = fieldnames(unit_data);
zscore_fields = all_fields(contains(all_fields, '_zscore'));

% Extract event type names by removing "_zscore" suffix
all_event_types = cell(length(zscore_fields), 1);
for i = 1:length(zscore_fields)
    event_type_name = strrep(zscore_fields{i}, '_zscore', '');
    all_event_types{i} = event_type_name;
end

fprintf('Found %d event types from PSTH data fields\n', length(all_event_types));

% Filter events if specified
if ~strcmp(viz_config.events_to_show, 'all')
    event_types = viz_config.events_to_show;
else
    event_types = all_event_types;
end

% Process both session types
session_types = {'Aversive', 'Reward'};
matched_data = struct();

for st_idx = 1:length(session_types)
    sess_type = session_types{st_idx};
    fprintf('  Processing %s sessions...\n', sess_type);

    % Get cluster assignments for this session type
    cluster_table = cluster_assignments_output.(sess_type);

    % Find matching units in PSTH data
    n_matched = 0;
    matched_units = [];

    for i = 1:height(cluster_table)
        session_name = cluster_table.Session{i};
        unit_num = cluster_table.Unit(i);
        cluster_id = cluster_table.ClusterID(i);

        % Find corresponding unit in PSTH data
        % Match by session_name and unit_id
        for u = 1:length(unit_data)
            if strcmp(unit_data(u).session_type, sess_type) && ...
               strcmp(unit_data(u).session_name, session_name) && ...
               unit_data(u).unit_id == unit_num

                % Found matching unit
                matched_unit = struct();
                matched_unit.psth_index = u;
                matched_unit.cluster_id = cluster_id;
                matched_unit.session_name = session_name;
                matched_unit.unit_id = unit_num;
                matched_unit.unique_unit_id = cluster_table.UniqueUnitID{i};

                if isempty(matched_units)
                    matched_units = matched_unit;
                else
                    matched_units(end+1) = matched_unit;
                end

                n_matched = n_matched + 1;
                break;
            end
        end
    end

    matched_data.(sess_type) = matched_units;
    fprintf('    Matched %d / %d units\n', n_matched, height(cluster_table));
end

fprintf('✓ Unit matching complete\n\n');

%% ========================================================================
%  SECTION 4: ORGANIZE DATA BY CLUSTER
%  ========================================================================

fprintf('Organizing data by cluster...\n');

cluster_data = struct();

for st_idx = 1:length(session_types)
    sess_type = session_types{st_idx};
    matched_units = matched_data.(sess_type);

    % Get unique cluster IDs
    cluster_ids = unique([matched_units.cluster_id]);
    cluster_ids = cluster_ids(~isnan(cluster_ids));  % Remove NaN
    n_clusters = length(cluster_ids);

    fprintf('  %s: %d clusters\n', sess_type, n_clusters);

    cluster_data.(sess_type).cluster_ids = cluster_ids;
    cluster_data.(sess_type).n_clusters = n_clusters;

    % For each cluster, collect PSTH data
    for c = 1:n_clusters
        cluster_id = cluster_ids(c);

        % Find units in this cluster
        cluster_unit_mask = [matched_units.cluster_id] == cluster_id;
        cluster_unit_indices = find(cluster_unit_mask);
        n_units_in_cluster = length(cluster_unit_indices);

        fprintf('    Cluster %d: %d units\n', cluster_id, n_units_in_cluster);

        % Collect PSTH data for all event types
        cluster_psth_data = struct();
        cluster_psth_data.cluster_id = cluster_id;
        cluster_psth_data.n_units = n_units_in_cluster;
        cluster_psth_data.unit_indices = cluster_unit_indices;

        for e = 1:length(event_types)
            event_type = event_types{e};

            % Collect z-score PSTH for all units in this cluster
            zscore_matrix = [];
            psth_matrix = [];
            n_events_list = [];

            for u_idx = 1:n_units_in_cluster
                unit_idx = matched_units(cluster_unit_indices(u_idx)).psth_index;

                % Get PSTH data
                zscore_field = [event_type '_zscore'];
                psth_field = [event_type '_psth'];
                n_events_field = [event_type '_n_events'];

                if isfield(unit_data(unit_idx), zscore_field)
                    zscore_psth = unit_data(unit_idx).(zscore_field);
                    psth = unit_data(unit_idx).(psth_field);
                    n_events = unit_data(unit_idx).(n_events_field);

                    if ~isempty(zscore_psth)
                        zscore_matrix = [zscore_matrix; zscore_psth(:)'];
                        psth_matrix = [psth_matrix; psth(:)'];
                        n_events_list = [n_events_list; n_events];
                    end
                end
            end

            % Store cluster PSTH data
            cluster_psth_data.([event_type '_zscore_matrix']) = zscore_matrix;
            cluster_psth_data.([event_type '_psth_matrix']) = psth_matrix;
            cluster_psth_data.([event_type '_n_events']) = n_events_list;

            % Calculate mean and SEM
            if ~isempty(zscore_matrix)
                cluster_psth_data.([event_type '_mean_zscore']) = mean(zscore_matrix, 1, 'omitnan');
                cluster_psth_data.([event_type '_sem_zscore']) = std(zscore_matrix, 0, 1, 'omitnan') / sqrt(size(zscore_matrix, 1));
                cluster_psth_data.([event_type '_mean_psth']) = mean(psth_matrix, 1, 'omitnan');
                cluster_psth_data.([event_type '_sem_psth']) = std(psth_matrix, 0, 1, 'omitnan') / sqrt(size(psth_matrix, 1));
            else
                cluster_psth_data.([event_type '_mean_zscore']) = nan(size(time_centers));
                cluster_psth_data.([event_type '_sem_zscore']) = nan(size(time_centers));
                cluster_psth_data.([event_type '_mean_psth']) = nan(size(time_centers));
                cluster_psth_data.([event_type '_sem_psth']) = nan(size(time_centers));
            end
        end

        % Store cluster data
        cluster_data.(sess_type).(['Cluster_' num2str(cluster_id)]) = cluster_psth_data;
    end
end

fprintf('✓ Data organization complete\n\n');

%% ========================================================================
%  SECTION 5: CREATE VISUALIZATIONS
%  ========================================================================

fprintf('Creating visualizations...\n');

% Create simple event labels (replace underscores with spaces)
event_labels = cell(length(event_types), 1);
for i = 1:length(event_types)
    event_labels{i} = strrep(event_types{i}, '_', ' ');
end
event_label_map = containers.Map(event_types, event_labels);

for st_idx = 1:length(session_types)
    sess_type = session_types{st_idx};

    cluster_ids = cluster_data.(sess_type).cluster_ids;
    n_clusters = cluster_data.(sess_type).n_clusters;

    fprintf('  Creating figures for %s sessions (%d clusters)...\n', sess_type, n_clusters);

    % Generate cluster colormap
    cluster_colors = viz_config.cluster_colormap(n_clusters);

    % Create one figure per event type
    for e = 1:length(event_types)
        event_type = event_types{e};
        event_label = event_label_map(event_type);

        % Create figure
        fig = figure('Position', [100, 100, 1600, 800]);
        sgtitle(sprintf('%s - %s: PSTH by Cluster', sess_type, event_label), ...
            'FontSize', 14, 'FontWeight', 'bold');

        % Plot heatmap for each cluster
        for c = 1:n_clusters
            cluster_id = cluster_ids(c);
            cluster_psth_data = cluster_data.(sess_type).(['Cluster_' num2str(cluster_id)]);

            % Skip if too few units
            if cluster_psth_data.n_units < viz_config.min_units_per_cluster
                continue;
            end

            % Get PSTH matrix
            zscore_matrix = cluster_psth_data.([event_type '_zscore_matrix']);
            mean_zscore = cluster_psth_data.([event_type '_mean_zscore']);
            sem_zscore = cluster_psth_data.([event_type '_sem_zscore']);

            if isempty(zscore_matrix)
                continue;
            end

            % Heatmap subplot
            subplot(2, n_clusters, c);
            imagesc(time_centers, 1:size(zscore_matrix, 1), zscore_matrix);
            colormap(gca, 'jet');
            clim(viz_config.heatmap_clim);
            colorbar;
            xlabel('Time (s)');
            ylabel('Unit #');
            title(sprintf('Cluster %d (n=%d)', cluster_id, cluster_psth_data.n_units), ...
                'Color', cluster_colors(c, :), 'FontWeight', 'bold');
            xline(0, 'w--', 'LineWidth', 2);

            % Mean trace subplot
            subplot(2, n_clusters, n_clusters + c);
            hold on;

            % Plot SEM shading
            if ~all(isnan(sem_zscore))
                fill([time_centers, fliplr(time_centers)], ...
                     [mean_zscore - sem_zscore, fliplr(mean_zscore + sem_zscore)], ...
                     cluster_colors(c, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            end

            % Plot mean trace
            plot(time_centers, mean_zscore, 'Color', cluster_colors(c, :), ...
                'LineWidth', 2);

            xline(0, 'k--', 'LineWidth', 1.5);
            yline(0, 'k:', 'LineWidth', 1);
            xlabel('Time (s)');
            ylabel('Mean z-score');
            title(sprintf('Cluster %d Mean Response', cluster_id), ...
                'Color', cluster_colors(c, :), 'FontWeight', 'bold');
            if ~isempty(viz_config.trace_ylim)
                ylim(viz_config.trace_ylim);
            end
            if ~isempty(viz_config.trace_xlim)
                xlim(viz_config.trace_xlim);
            end
            grid on;
            hold off;
        end

        % Save figure
        fig_filename = sprintf('PSTH_Cluster_%s_%s.png', sess_type, event_type);
        saveas(fig, fig_filename);
        fprintf('    Saved: %s\n', fig_filename);
    end
end

%% ========================================================================
%  SECTION 6: CREATE CLUSTER COMPARISON FIGURE
%  ========================================================================

% fprintf('\nCreating cluster comparison figure...\n');
% 
% % For each session type, create a summary figure comparing all clusters
% for st_idx = 1:length(session_types)
%     sess_type = session_types{st_idx};
% 
%     cluster_ids = cluster_data.(sess_type).cluster_ids;
%     n_clusters = cluster_data.(sess_type).n_clusters;
%     n_events = length(event_types);
% 
%     % Generate cluster colormap
%     cluster_colors = viz_config.cluster_colormap(n_clusters);
% 
%     % Create large summary figure
%     fig = figure('Position', [50, 50, 400*n_events, 300*n_clusters]);
%     sgtitle(sprintf('%s Sessions: All Clusters × All Events', sess_type), ...
%         'FontSize', 16, 'FontWeight', 'bold');
% 
%     plot_idx = 1;
%     for c = 1:n_clusters
%         cluster_id = cluster_ids(c);
%         cluster_psth_data = cluster_data.(sess_type).(['Cluster_' num2str(cluster_id)]);
% 
%         % Skip if too few units
%         if cluster_psth_data.n_units < viz_config.min_units_per_cluster
%             continue;
%         end
% 
%         for e = 1:n_events
%             event_type = event_types{e};
%             event_label = event_label_map(event_type);
% 
%             mean_zscore = cluster_psth_data.([event_type '_mean_zscore']);
%             sem_zscore = cluster_psth_data.([event_type '_sem_zscore']);
% 
%             subplot(n_clusters, n_events, plot_idx);
%             hold on;
% 
%             % Plot SEM shading
%             if ~all(isnan(sem_zscore))
%                 fill([time_centers, fliplr(time_centers)], ...
%                      [mean_zscore - sem_zscore, fliplr(mean_zscore + sem_zscore)], ...
%                      cluster_colors(c, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
%             end
% 
%             % Plot mean trace
%             plot(time_centers, mean_zscore, 'Color', cluster_colors(c, :), ...
%                 'LineWidth', 2);
% 
%             xline(0, 'k--', 'LineWidth', 1.5);
%             yline(0, 'k:', 'LineWidth', 1);
% 
%             % Labels
%             if e == 1
%                 ylabel(sprintf('Cluster %d\n(n=%d)', cluster_id, cluster_psth_data.n_units), ...
%                     'FontWeight', 'bold', 'Color', cluster_colors(c, :));
%             end
%             if c == 1
%                 title(event_label, 'FontSize', 11, 'FontWeight', 'bold');
%             end
%             if c == n_clusters
%                 xlabel('Time (s)');
%             end
% 
%             if ~isempty(viz_config.trace_ylim)
%                 ylim(viz_config.trace_ylim);
%             end
%             if ~isempty(viz_config.trace_xlim)
%                 xlim(viz_config.trace_xlim);
%             end
%             grid on;
%             hold off;
% 
%             plot_idx = plot_idx + 1;
%         end
%     end
% 
%     % Save comparison figure
%     fig_filename = sprintf('PSTH_Cluster_Summary_%s.png', sess_type);
%     saveas(fig, fig_filename);
%     fprintf('  Saved: %s\n', fig_filename);
% end
% 
% fprintf('\n✓ Visualization complete\n');
fprintf('\n=== ANALYSIS COMPLETE ===\n');
