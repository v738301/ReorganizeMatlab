%% ========================================================================
%  QUICK GLM TEST: BREATHING AMPLITUDE VERSION
%  ========================================================================
%
%  This version uses breathing AMPLITUDE (1.5-8Hz envelope) instead of rate:
%  - Extracts breathing amplitude from Signal using bandpass + Hilbert transform
%  - Uses amplitude envelope as continuous predictor
%  - Removes Speed × Breathing interaction
%
%  Predictors:
%  - Events: IR1ON, IR2ON, Aversive (3 events)
%  - Continuous: Speed, BreathingAmplitude_1.5Hz, BreathingAmplitude_8Hz
%  - Spike history: 5 lags
%  - No interactions
%% ========================================================================

clear all
close all

fprintf('=== QUICK GLM TEST: BREATHING AMPLITUDE VERSION ===\n\n');

%% ========================================================================
%  CONFIGURATION
%% ========================================================================

config = struct();

% Timing parameters
config.bin_size = 0.05;
config.unit_of_time = 's';

% Basis function parameters - RAISED COSINE LOG-STRETCHED
config.basis_type = 'raised_cosine';    % Use raised cosine basis functions
config.n_basis_funcs = 10;              % Number of basis functions
config.event_window_pre = 0;          % Window for convolution padding (1s before)
config.event_window_post = 2.0;         % Window for convolution padding (2s after)
config.basis_stretch = 0.1;             % Controls log-stretching (smaller = more compression early)

% Speed and breathing smoothing
config.smooth_window = 0.2;

% Spike history parameters
config.include_history = true;       % Include spike history in model
config.history_lags = 5;           % 1-20- lags (50ms bins = 0-1 seconds)

% Interaction terms
config.include_interactions = false;  % No interactions in this version (breathing amplitude only)
config.interactions = {};  % No interactions

% Model fitting parameters
config.max_iter = 1000;
config.display_fitting = 'off';

% Behavioral parameters
config.breathing_freq_threshold = 5;
config.camera_fps = 20;

% Bout detection parameters (for reward events)
config.bout_epsilon = 2.0;              % Maximum time between events in bout (seconds)
config.bout_minPts = 2;                 % Minimum events to form a bout

% Data paths
config.prediction_folder = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/BehaviorPrediction';
config.spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';

% TEST PARAMETERS (limit data for quick test)
config.max_sessions = 10;        % Only process 2 sessions
config.max_units_per_session = 5;  % Only process 3 units per session

% Add paths
addpath(genpath('neuroGLM'));
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/NewScripts/');

fprintf('Test Configuration:\n');
fprintf('  Max sessions: %d\n', config.max_sessions);
fprintf('  Max units per session: %d\n', config.max_units_per_session);
fprintf('  Bin size: %.0f ms\n', config.bin_size * 1000);
fprintf('  Event window: [-%d, +%d] ms\n\n', ...
    config.event_window_pre * 1000, config.event_window_post * 1000);

%% ========================================================================
%  LOAD SORTING PARAMETERS
%% ========================================================================

fprintf('Loading sorting parameters...\n');
[T_sorted] = loadSortingParameters();
fprintf('✓ Loaded\n\n');

%% ========================================================================
%  LOAD CLUSTER ASSIGNMENTS
%% ========================================================================

fprintf('Loading cluster assignments...\n');
cluster_file = 'unit_cluster_assignments_PSTH_TimeSeries.mat';
unit_count = 1;
if exist(cluster_file, 'file')
    cluster_data = load(cluster_file);
    fprintf('✓ Loaded cluster assignments\n');

    % Extract cluster 2 units from reward sessions
    target_cluster = 3;
    cluster_2_units = struct('session_name', {}, 'unit_idx', {});

    if isfield(cluster_data, 'cluster_assignments_output')
        assignments = cluster_data.cluster_assignments_output;

        % Access the Reward table
        if isfield(assignments, 'Reward')
            reward_table = assignments.Reward;

            % Parse the table (columns: UniqueUnitID, Session, Unit, ClusterID)
            for i = 1:height(reward_table)
                session_name = reward_table.Session{i};
                unit_idx = reward_table.Unit(i);
                cluster_id = reward_table.ClusterID(i);
                % Check if target cluster
                if cluster_id == target_cluster
                    cluster_2_units(unit_count).session_name = session_name;
                    cluster_2_units(unit_count).unit_idx = unit_idx;
                    unit_count = unit_count + 1;
                end
            end

            fprintf('Found %d cluster %d units in reward sessions\n', length(cluster_2_units), target_cluster);
        else
            fprintf('Warning: Reward field not found in cluster assignments, using all units\n');
            cluster_2_units = [];
        end
    else
        fprintf('Warning: cluster assignment structure not recognized, using all units\n');
        cluster_2_units = [];
    end
else
    fprintf('Warning: %s not found, using all units\n', cluster_file);
    cluster_2_units = [];
end
fprintf('\n');

%% ========================================================================
%  LOAD DATA (REWARD SESSIONS WITH CLUSTER 2 UNITS)
%% ========================================================================

fprintf('Loading sessions...\n');
if ~isempty(cluster_2_units)
    % Get unique session names from cluster 2 units
    unique_sessions = unique({cluster_2_units.session_name});
    n_sessions_to_load = min(length(unique_sessions), config.max_sessions);
    unique_sessions = unique_sessions(1:n_sessions_to_load);

    % Find full file paths for these sessions
    all_reward_files = dir(fullfile(config.spike_folder, '2025*RewardSeeking*.mat'));
    allfiles = struct('name', {}, 'folder', {}, 'date', {}, 'bytes', {}, 'isdir', {}, 'datenum', {});
    file_count = 0;
    for s = 1:length(unique_sessions)
        sess_name = unique_sessions{s};
        for f = 1:length(all_reward_files)
            if strcmp(all_reward_files(f).name, sess_name)
                file_count = file_count + 1;
                allfiles(file_count) = all_reward_files(f);
                break;
            end
        end
    end
    n_sessions = length(allfiles);
    fprintf('Found %d sessions with cluster 2 units\n', n_sessions);
else
    % Fallback: load regular reward sessions
    session_pattern = '2025*RewardSeeking*.mat';
    [allfiles, ~, ~, ~] = selectFilesWithAnimalIDFiltering(...
        config.spike_folder, config.max_sessions, session_pattern);
    n_sessions = min(length(allfiles), config.max_sessions);
    fprintf('Found %d sessions (no cluster filtering)\n', n_sessions);
end

% Load behavioral data
prediction_sessions = loadBehaviorPredictionsFromSpikeFiles(allfiles(1:n_sessions), config.prediction_folder);
sessions = loadSessionMetricsFromSpikeFiles(allfiles(1:n_sessions), T_sorted);
fprintf('✓ Loaded behavioral data\n\n');

%% ========================================================================
%  PROCESS UNITS
%% ========================================================================

all_glm_results = [];
unit_counter = 0;

for sess_idx = 1:n_sessions
    fprintf('=== Session %d/%d: %s ===\n', sess_idx, n_sessions, allfiles(sess_idx).name);

    % Load session data
    Timelimits = 'No';
    try
        [NeuralTime, AdjustedXYZ, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, WP1ON, WP2ON, ...
         AversiveSound, sessionLabels, valid_spikes, Fs, TriggerMid] = ...
            loadAndPrepareSessionData(allfiles(sess_idx), T_sorted, Timelimits);
    catch ME
        fprintf('ERROR loading session: %s\n', ME.message);
        continue;
    end

    fprintf('Loaded %d units\n', length(valid_spikes));

    % Get behavioral data
    session = sessions{sess_idx};
    behavioral_matrix_full = [];
    if isfield(session, 'behavioral_matrix_full') && ~isempty(session.behavioral_matrix_full)
        behavioral_matrix_full = session.behavioral_matrix_full;
    end

    % Build design matrix
    fprintf('Building design matrix...\n');
    [dm, predictor_info] = buildDesignMatrixForSession(...
        NeuralTime, IR1ON, IR2ON, AversiveSound, ...
        AdjustedXYZ_speed, Signal, Fs, config);
    fprintf('  Design matrix: %d bins × %d predictors\n', size(dm.X, 1), size(dm.X, 2));

    % Determine which units to process
    if ~isempty(cluster_2_units)
        % Process only cluster 2 units for this session
        session_name = allfiles(sess_idx).name;
        units_to_process = [];
        for u = 1:length(cluster_2_units)
            if strcmp(cluster_2_units(u).session_name, session_name)
                units_to_process(end+1) = cluster_2_units(u).unit_idx;
            end
        end
        fprintf('  Processing %d cluster 2 units\n', length(units_to_process));
    else
        % Fallback: process first N units
        units_to_process = 1:min(length(valid_spikes), config.max_units_per_session);
        fprintf('  Processing first %d units\n', length(units_to_process));
    end

    for u = 1:length(units_to_process)
        unit_idx = units_to_process(u);

        if unit_idx > length(valid_spikes)
            fprintf('  Unit %d: Index out of range, skipping\n', unit_idx);
            continue;
        end

        spike_times = valid_spikes{unit_idx};

        if isempty(spike_times) || length(spike_times) < 100
            fprintf('  Unit %d: Too few spikes, skipping\n', unit_idx);
            continue;
        end

        unit_counter = unit_counter + 1;

        % Bin spike train
        spike_counts = binSpikeTrain(spike_times, dm.time_bins);

        fprintf('  Unit %d: Fitting GLM (%d spikes)...\n', unit_idx, sum(spike_counts));

        % Add spike history to design matrix (unit-specific)
        if config.include_history
            [X_with_history, predictor_info_with_history] = addSpikeHistoryToDesignMatrix(...
                dm.X, spike_counts, predictor_info, config);
        else
            X_with_history = dm.X;
            predictor_info_with_history = predictor_info;
        end

        % Fit GLM
        try
            glm_result = fitPoissonGLM(X_with_history, spike_counts, predictor_info_with_history, config);

            % Store results
            glm_result.unit_id = unit_idx;
            glm_result.global_unit_id = unit_counter;
            glm_result.session_name = allfiles(sess_idx).name;
            glm_result.session_idx = sess_idx;
            glm_result.n_spikes = sum(spike_counts);
            glm_result.mean_firing_rate = sum(spike_counts) * config.bin_size / (length(spike_counts) * config.bin_size);
            glm_result.spike_counts = spike_counts;  % Save for plotting
            glm_result.design_matrix = X_with_history;         % FIXED: Use augmented design matrix
            glm_result.time_centers = dm.time_centers;
            glm_result.predictor_info = predictor_info_with_history; % FIXED: Use augmented predictor info

            % Store cluster assignment if available
            glm_result.cluster_id = NaN;
            if ~isempty(cluster_2_units)
                glm_result.cluster_id = target_cluster;
            end

            if unit_counter == 1
                all_glm_results = glm_result;
            else
                all_glm_results(unit_counter) = glm_result;
            end

            fprintf('    ✓ Dev explained: %.2f%%, Mean FR: %.2f spikes/s\n', ...
                glm_result.deviance_explained * 100, glm_result.mean_firing_rate);

        catch ME
            fprintf('    ✗ GLM fitting failed: %s\n', ME.message);
            continue;
        end
    end
    fprintf('\n');
end

fprintf('=== COMPLETED ===\n');
if ~isempty(cluster_2_units)
    fprintf('Total cluster %d units fitted: %d\n\n', target_cluster, unit_counter);
else
    fprintf('Total units fitted: %d\n\n', unit_counter);
end

if unit_counter == 0
    error('No units were successfully fitted!');
end

%% ========================================================================
%  VISUALIZATION
%% ========================================================================

fprintf('Creating visualizations...\n');

% Create output directory
output_dir = 'GLM_Test_Figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% FIGURE 1: Model Performance Summary
figure('Position', [100, 100, 1200, 800]);
if ~isempty(cluster_2_units)
    sgtitle(sprintf('GLM Test Results: Cluster %d Units', target_cluster), 'FontSize', 14, 'FontWeight', 'bold');
else
    sgtitle('GLM Test Results: Model Performance', 'FontSize', 14, 'FontWeight', 'bold');
end

% Panel 1: Deviance explained
subplot(2, 3, 1);
dev_exp = [all_glm_results.deviance_explained] * 100;
bar(dev_exp);
xlabel('Unit', 'FontSize', 11);
ylabel('Deviance Explained (%)', 'FontSize', 11);
title(sprintf('Model Performance\nMean: %.1f%%', mean(dev_exp)), 'FontSize', 12);
grid on;

% Panel 2: Firing rates
subplot(2, 3, 2);
firing_rates = [all_glm_results.mean_firing_rate];
bar(firing_rates);
xlabel('Unit', 'FontSize', 11);
ylabel('Mean Firing Rate (spikes/s)', 'FontSize', 11);
title('Firing Rates', 'FontSize', 12);
grid on;

% Panel 3: Number of spikes
subplot(2, 3, 3);
n_spikes = [all_glm_results.n_spikes];
bar(n_spikes);
xlabel('Unit', 'FontSize', 11);
ylabel('Total Spikes', 'FontSize', 11);
title('Spike Counts', 'FontSize', 12);
grid on;

% Panel 4: Coefficient magnitudes (heatmap)
subplot(2, 3, [4, 5, 6]);
coef_matrix = zeros(unit_counter, length(predictor_info_with_history.names));
for i = 1:unit_counter
    coef_matrix(i, :) = all_glm_results(i).coefficients';
end
imagesc(coef_matrix(:, 2:end)');  % Exclude bias
colormap(redblue);
colorbar;
caxis([-max(abs(coef_matrix(:))), max(abs(coef_matrix(:)))]);
xlabel('Unit', 'FontSize', 11);
ylabel('Predictor', 'FontSize', 11);
title('GLM Coefficients (z-scored)', 'FontSize', 12);
yticks(1:length(predictor_info_with_history.names)-1);
yticklabels(predictor_info_with_history.names(2:end));

saveas(gcf, fullfile(output_dir, 'Test_1_Model_Performance.png'));
fprintf('  ✓ Saved Test_1_Model_Performance.png\n');

%% FIGURE 2: Example Unit - Actual vs Predicted
example_unit_idx = 10;  % Use first unit
unit = all_glm_results(example_unit_idx);

figure('Position', [100, 100, 1400, 600]);
sgtitle(sprintf('Example Unit %d: Actual vs Predicted Firing Rate', example_unit_idx), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Compute predicted firing rate
predicted_log_rate = unit.design_matrix * unit.coefficients;
predicted_rate = exp(predicted_log_rate);

% Smooth for visualization
window_size = 10;  % 1 second smoothing
actual_smooth = smoothdata(unit.spike_counts / config.bin_size, 'gaussian', window_size);
predicted_smooth = smoothdata(predicted_rate / config.bin_size, 'gaussian', window_size);


N3 = length(unit.spike_counts);
sd = 1 / sqrt(N3);
yline(2 * sd, 'r')
yline(-2 * sd, 'r')

% Plot time series
time_vec = (1:length(unit.spike_counts)) * config.bin_size;
subplot(2, 1, 1);
plot(time_vec, actual_smooth, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(time_vec, predicted_smooth, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
xlabel('Time (s)', 'FontSize', 11);
ylabel('Firing Rate (spikes/s)', 'FontSize', 11);
title(sprintf('Time Series (Dev explained: %.1f%%)', unit.deviance_explained * 100), 'FontSize', 12);
legend('Location', 'best');
grid on;
% xlim([0, min(60, max(time_vec))]);  % Show first 60 sec

% Scatter plot: actual vs predicted
subplot(2, 1, 2);
scatter(predicted_smooth, actual_smooth, 10, 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
max_rate = max([predicted_smooth; actual_smooth]);
plot([0 max_rate], [0 max_rate], 'r--', 'LineWidth', 2);
xlabel('Predicted Rate (spikes/s)', 'FontSize', 11);
ylabel('Actual Rate (spikes/s)', 'FontSize', 11);
title('Actual vs Predicted', 'FontSize', 12);
grid on;
axis equal;
xlim([0, max_rate]);
ylim([0, max_rate]);

saveas(gcf, fullfile(output_dir, 'Test_2_Actual_vs_Predicted.png'));
fprintf('  ✓ Saved Test_2_Actual_vs_Predicted.png\n');

%% FIGURE 3: Temporal Filters
figure('Position', [100, 100, 1600, 900]);
sgtitle('Temporal Filters for Event Predictors', 'FontSize', 14, 'FontWeight', 'bold');

n_basis = config.n_basis_funcs;
n_bins_pre = round(config.event_window_pre / config.bin_size);
n_bins_post = round(config.event_window_post / config.bin_size);
event_duration_bins = n_bins_pre + n_bins_post;
time_axis = ((-n_bins_pre):(n_bins_post-1)) * config.bin_size * 1000;  % ms

event_names = {'IR1ON', 'IR2ON', 'Aversive'};

n_bins_pre = round(config.event_window_pre / config.bin_size);   % 20 bins (1 sec before)
n_bins_post = round(config.event_window_post / config.bin_size); % 40 bins (2 sec after)
event_duration_bins = n_bins_pre + n_bins_post;                  % 60 bins total (3 sec window)
% Create basis functions (for reconstruction)
basis_funcs = createRaisedCosineBasis(config.n_basis_funcs, event_duration_bins, config.bin_size, config.basis_stretch);

for ev = 1:length(event_names)
    event_name = event_names{ev};

    % Extract coefficients for this event (average across units)
    start_idx = 2 + (ev - 1) * n_basis;
    end_idx = start_idx + n_basis - 1;

    event_coefs = coef_matrix(:, start_idx:end_idx);
    mean_coefs = mean(event_coefs, 1);

    % Reconstruct temporal filter
    temporal_filter = basis_funcs * event_coefs';

    % Plot
    subplot(2, 3, ev); hold on;
    plot(time_axis, temporal_filter, 'LineWidth', 2, 'Color', [0.2 0.4 0.8]);
    plot(time_axis, mean(temporal_filter,2), 'LineWidth', 2, 'Color', 'k');
    hold on;

    % Add event marker
    plot([0 0], ylim, 'k--', 'LineWidth', 1.5);

    % Shade pre-event region
    yl = ylim;
    fill([min(time_axis) 0 0 min(time_axis)], [yl(1) yl(1) yl(2) yl(2)], ...
         [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    uistack(gca().Children(1), 'bottom');

    xlabel('Time from event (ms)', 'FontSize', 10);
    ylabel('Weight', 'FontSize', 10);
    title(sprintf('%s Filter', event_name), 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    xlim([min(time_axis), max(time_axis)]);
end

saveas(gcf, fullfile(output_dir, 'Test_3_Temporal_Filters.png'));
fprintf('  ✓ Saved Test_3_Temporal_Filters.png\n');

%% FIGURE 4: Feature Importance + Spike History Kernel
figure('Position', [100, 100, 1400, 900]);
sgtitle('Feature Importance and Spike History', 'FontSize', 14, 'FontWeight', 'bold');

% --- Top Subplot: Feature Importance Bar Plot ---
subplot(2, 1, 1);

% Build feature names list dynamically based on what's included
feature_names = {'IR1ON', 'IR2ON','Aversive', 'Speed', 'BreathingAmplitude_1.5Hz', 'BreathingAmplitude_8Hz'};
if config.include_interactions
    feature_names{end+1} = 'Speed_x_BreathingAmplitude';
end
if config.include_history
    feature_names{end+1} = 'History';
end

importance_matrix = zeros(unit_counter, length(feature_names));

for i = 1:unit_counter
    fi = all_glm_results(i).feature_importance;
    for f = 1:length(feature_names)
        fname = feature_names{f};
        if isfield(fi, fname)
            importance_matrix(i, f) = fi.(fname).percent_deviance;
        end
    end
end

% Average importance
mean_importance = mean(importance_matrix, 1);
std_importance = std(importance_matrix, 0, 1);

bar(mean_importance);
hold on;
errorbar(1:length(feature_names), mean_importance, std_importance, 'k.', 'LineWidth', 1.5);

xticks(1:length(feature_names));
xticklabels(feature_names);
xtickangle(45);
ylabel('% Deviance Contribution', 'FontSize', 12);
title(sprintf('Average Feature Importance (n=%d units)', unit_counter), 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% --- Bottom Subplot: Spike History Kernel ---
if config.include_history
    subplot(2, 1, 2);

    % Extract history weights from all units
    n_lags = config.history_lags;
    history_weights = zeros(unit_counter, n_lags);

    for i = 1:unit_counter
        unit = all_glm_results(i);
        predictor_names = unit.predictor_info.names;

        % Find history predictor indices
        history_idx_start = find(strcmp(predictor_names, 'History_lag1'), 1);
        if ~isempty(history_idx_start)
            history_idx_end = history_idx_start + n_lags - 1;
            history_weights(i, :) = unit.coefficients(history_idx_start:history_idx_end);
        end
    end

    % Plot mean ± SEM
    lag_times = (1:n_lags) * config.bin_size * 1000;  % Convert to milliseconds
    mean_weights = mean(history_weights, 1);
    sem_weights = std(history_weights, 0, 1) / sqrt(unit_counter);

    % Plot with shaded error bars
    fill([lag_times, fliplr(lag_times)], ...
         [mean_weights + sem_weights, fliplr(mean_weights - sem_weights)], ...
         [0.7 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    hold on;
    plot(lag_times, mean_weights, 'k-', 'LineWidth', 2);
    plot([0, max(lag_times)], [0, 0], 'r--', 'LineWidth', 1);  % Zero line

    xlabel('Lag (ms)', 'FontSize', 12);
    ylabel('Weight (z-scored units)', 'FontSize', 12);
    title(sprintf('Spike History Kernel (mean ± SEM, n=%d units)', unit_counter), ...
          'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    xlim([0, max(lag_times)]);

    % Add annotations
    text(0.02, 0.95, 'Refractory period / Bursting', 'Units', 'normalized', ...
         'FontSize', 10, 'Color', [0.5 0.5 0.5]);
end

saveas(gcf, fullfile(output_dir, 'Test_4_Feature_Importance.png'));
fprintf('  ✓ Saved Test_4_Feature_Importance.png\n');

%% FIGURE 5: Cumulative Predictor Contribution to Firing Rate
example_unit_idx = 1; % Use the same example unit
unit = all_glm_results(example_unit_idx);

figure('Position', [100, 100, 1400, 900]); % Increased height for two subplots
sgtitle(sprintf('Unit %d: GLM Analysis (Deviance Explained: %.1f%%)', ...
    unit.unit_id, unit.deviance_explained * 100), 'FontSize', 14, 'FontWeight', 'bold');
ax = [];
% --- Top Subplot: Cumulative Predictor Contributions ---
ax(end+1) = subplot(2, 1, 1);
title('Cumulative Predictor Contribution to Firing Rate');

X = unit.design_matrix;
w = unit.coefficients;
time_vec = unit.time_centers;
predictor_names_full = unit.predictor_info.names;
n_basis = unit.predictor_info.n_basis;
event_names_gl = unit.predictor_info.event_names; % Global event names

% Get smoothed actual firing rate for comparison
window_size = 10; % 1 second smoothing (10 bins @ 0.05s/bin = 0.5s)
actual_smooth = smoothdata(unit.spike_counts / config.bin_size, 'gaussian', window_size);

% Define predictor groups to add sequentially for plotting
% Order: Intercept, IR1ON, IR2ON, Aversive, Speed, BreathingAmplitude_1.5Hz, BreathingAmplitude_8Hz, Interaction, History
predictor_groups_to_add = {'Bias', event_names_gl{:}, 'Speed', 'BreathingAmplitude_1.5Hz', 'BreathingAmplitude_8Hz'};
if config.include_interactions
    predictor_groups_to_add{end+1} = 'Speed_x_BreathingAmplitude';
end
if config.include_history
    predictor_groups_to_add{end+1} = 'History';
end

% For plotting aesthetics
line_styles = {'-', '--', ':', '-.', ':', '--', '-', '-.', ':', '-'};
term_colors = [0 0 1;    % Blue for Intercept
               0 0.5 0;  % Dark Green for IR1ON
               0.5 0 0.5;% Purple for IR2ON
               0 0.7 0.7;% Cyan for Aversive
               0.7 0.7 0;% Olive for Speed
               0.5 0.5 0.5;% Gray for BreathingAmplitude_1.5Hz
               0.7 0.7 0.7;% Light Gray for BreathingAmplitude_8Hz
               1 0.5 0;  % Orange for Interaction
               0.7 0 0;  % Dark Red for History
              ];
% Extend colors if more predictor groups than predefined colors
if length(predictor_groups_to_add) > size(term_colors, 1)
    term_colors = [term_colors; lines(length(predictor_groups_to_add) - size(term_colors, 1))];
end

% Initialize cumulative linear predictor for plotting
current_u_plot = zeros(size(X,1),1); 
current_term_label = '';
plot_idx_counter = 1; % For assigning colors/styles

% Plot Actual Firing Rate first
plot(time_vec, actual_smooth, 'k', 'LineWidth', 2, 'DisplayName', 'Actual Firing Rate');
hold on;

for i = 1:length(predictor_groups_to_add)
    group_name = predictor_groups_to_add{i};

    if strcmp(group_name, 'Bias')
        group_indices = 1;
        term_label = 'Intercept';
    elseif ismember(group_name, event_names_gl)
        % Event predictors are groups of basis functions
        ev_start_idx_in_names = find(strcmp(predictor_names_full, [group_name '_basis1']), 1);
        if ~isempty(ev_start_idx_in_names)
             group_indices = ev_start_idx_in_names : (ev_start_idx_in_names + n_basis - 1);
             term_label = group_name;
        else
            continue; % Skip if event not found
        end
    elseif strcmp(group_name, 'History')
        % History is a group of lagged spike counts
        history_idx = find(strcmp(predictor_names_full, 'History_lag1'), 1);
        if ~isempty(history_idx)
            group_indices = history_idx : (history_idx + unit.predictor_info.n_history_lags - 1);
            term_label = 'History';
        else
            continue; % Skip if history not found
        end
    else % Continuous predictors ('Speed', 'BreathingAmplitude', 'Speed_x_BreathingAmplitude')
        group_indices = find(strcmp(predictor_names_full, group_name), 1);
        if ~isempty(group_indices)
            term_label = group_name;
        else
            continue; % Skip if predictor not found
        end
    end
    
    % Add current group's contribution to the cumulative linear predictor
    current_u_plot = current_u_plot + X(:, group_indices) * w(group_indices);
    predicted_rate_term = smoothdata(exp(current_u_plot) / config.bin_size, 'gaussian', window_size);
    
    % Build cumulative label for legend
    if isempty(current_term_label)
        current_term_label = term_label;
    else
        current_term_label = [current_term_label ' + ' term_label];
    end
    
    plot_idx_counter = plot_idx_counter + 1; % Increment for color/style
    plot(time_vec, predicted_rate_term, 'Color', term_colors(plot_idx_counter-1, :), 'LineWidth', 1.5, ...
        'DisplayName', current_term_label, 'LineStyle', line_styles{mod(plot_idx_counter-1, length(line_styles))+1});
end

xlabel('Time (s)', 'FontSize', 11);
ylabel('Firing Rate (spikes/s)', 'FontSize', 11);
legend('Location', 'northwest', 'Interpreter', 'none');
grid on;
% xlim([0, min(60, max(time_vec))]); % Show first 60 seconds or full duration
set(gca, 'YScale', 'linear'); % Ensure y-axis is linear for firing rate
% Adjust Y-limits if rates are very high
max_y = max([actual_smooth(:); cellfun(@(x) max(x(:)), {predicted_rate_term})]); % max of actual and last predicted
ylim([0, max_y * 1.1]);


% --- Bottom Subplot: Selected Design Matrix Predictors (Lines) ---
ax(end+1) = subplot(2, 1, 2);
hold on;
title('Selected Design Matrix Predictors');
xlabel('Time (s)');
ylabel('Predictor Value');
grid on;
% xlim([0, min(120, max(time_vec))]);

% Select key predictors to plot (don't plot all 150 history lags individually!)
plot_groups_names = {'Bias', event_names_gl{1}, 'Speed', 'BreathingAmplitude_1.5Hz', 'BreathingAmplitude_8Hz'};
if config.include_interactions
    plot_groups_names{end+1} = 'Speed_x_BreathingAmplitude';
end
if config.include_history
    plot_groups_names{end+1} = 'History';  % Will plot mean of all lags
end

plot_colors = {'k', 'r', 'b', 'g', 'm', 'c'}; % Colors for the selected predictors
plot_line_styles = {'-', '--', ':', '-.', '-', '--'};

plotted_legend_names = {};

for p_idx = 1:length(plot_groups_names)
    group_name = plot_groups_names{p_idx};

    current_predictor_sum = zeros(size(X,1), 1);
    current_group_label = group_name;

    if strcmp(group_name, 'Bias')
        current_predictor_sum = X(:, 1);
    elseif ismember(group_name, event_names_gl)
        % Sum all basis functions for this event
        ev_basis_col_indices = find(startsWith(predictor_names_full, [group_name '_basis']));
        if ~isempty(ev_basis_col_indices)
            current_predictor_sum = sum(X(:, ev_basis_col_indices), 2);
        else
            continue; % Skip if event group not found
        end
    elseif strcmp(group_name, 'History')
        % For history, plot the mean of all lags (too many to plot individually)
        history_col_indices = find(startsWith(predictor_names_full, 'History_lag'));
        if ~isempty(history_col_indices)
            current_predictor_sum = mean(X(:, history_col_indices), 2);
            current_group_label = 'History (mean)';
        else
            continue; % Skip if history not found
        end
    else % Continuous predictors ('Speed', 'BreathingAmplitude', 'Speed_x_BreathingAmplitude')
        cont_pred_col_idx = find(strcmp(predictor_names_full, group_name), 1);
        if ~isempty(cont_pred_col_idx)
            current_predictor_sum = X(:, cont_pred_col_idx);
        else
            continue; % Skip if continuous predictor not found
        end
    end

    if ~all(current_predictor_sum == 0) % Only plot non-zero predictors
        color_idx = min(p_idx, length(plot_colors));
        plot(time_vec, current_predictor_sum, 'Color', plot_colors{color_idx}, ...
             'LineStyle', plot_line_styles{min(p_idx, length(plot_line_styles))}, 'LineWidth', 1.5, ...
             'DisplayName', strrep(current_group_label, '_', ' '));
        plotted_legend_names{end+1} = strrep(current_group_label, '_', ' ');
    end
end

legend(plotted_legend_names, 'Location', 'northwest', 'Interpreter', 'none');
hold off;
linkaxes(ax,'x')

saveas(gcf, fullfile(output_dir, 'Test_5_Cumulative_Contributions.png'));
fprintf('  ✓ Saved Test_5_Cumulative_Contributions.png\n');

%% FIGURE 6: Example Unit Detailed Analysis (Temporal Filters, Feature Importance, Spike History)
figure('Position', [100, 100, 1600, 1200]);
sgtitle(sprintf('Unit %d: Detailed GLM Analysis', example_unit_idx), 'FontSize', 14, 'FontWeight', 'bold');

% Get unit data
unit = all_glm_results(example_unit_idx);
predictor_names = unit.predictor_info.names;
coefficients = unit.coefficients;
n_basis = unit.predictor_info.n_basis;
event_names = unit.predictor_info.event_names;

% Prepare for temporal filter reconstruction
n_bins_pre = round(config.event_window_pre / config.bin_size);
n_bins_post = round(config.event_window_post / config.bin_size);
event_duration_bins = n_bins_pre + n_bins_post;
time_axis = ((-n_bins_pre):(n_bins_post-1)) * config.bin_size * 1000;  % ms
n_bins_pre = round(config.event_window_pre / config.bin_size);   % 20 bins (1 sec before)
n_bins_post = round(config.event_window_post / config.bin_size); % 40 bins (2 sec after)
event_duration_bins = n_bins_pre + n_bins_post;                  % 60 bins total (3 sec window)
% Create basis functions (for reconstruction)
basis_funcs = createRaisedCosineBasis(config.n_basis_funcs, event_duration_bins, config.bin_size, config.basis_stretch);

% --- ROW 1: Temporal Filters (5 events) ---
for ev = 1:length(event_names)
    subplot(3, 5, ev);

    % Extract coefficients for this event
    start_idx = 2 + (ev - 1) * n_basis;  % +2 for bias
    end_idx = start_idx + n_basis - 1;
    event_coefs = coefficients(start_idx:end_idx);

    % Reconstruct temporal filter
    temporal_filter = basis_funcs * event_coefs;

    % Plot
    plot(time_axis, temporal_filter, 'LineWidth', 2, 'Color', [0.2 0.4 0.8]);
    hold on;
    plot([0 0], ylim, 'k--', 'LineWidth', 1.5);

    % Shade pre-event region
    yl = ylim;
    fill([min(time_axis) 0 0 min(time_axis)], [yl(1) yl(1) yl(2) yl(2)], ...
         [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    uistack(gca().Children(1), 'bottom');

    xlabel('Time (ms)', 'FontSize', 10);
    ylabel('Weight', 'FontSize', 10);
    title(sprintf('%s Filter', event_names{ev}), 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    xlim([min(time_axis), max(time_axis)]);
end

% --- ROW 2: Feature Importance ---
subplot(3, 1, 2);

% Extract feature importance for this unit
fi = unit.feature_importance;
feature_names = {'IR1ON', 'IR2ON', 'Aversive', 'Speed', 'BreathingAmplitude_1.5Hz', 'BreathingAmplitude_8Hz'};
if config.include_interactions
    feature_names{end+1} = 'Speed_x_BreathingAmplitude';
end
if config.include_history
    feature_names{end+1} = 'History';
end

importance_values = zeros(1, length(feature_names));
for f = 1:length(feature_names)
    fname = feature_names{f};
    if isfield(fi, fname)
        importance_values(f) = fi.(fname).percent_deviance;
    end
end

bar(importance_values);
xticks(1:length(feature_names));
xticklabels(feature_names);
xtickangle(45);
ylabel('% Deviance Contribution', 'FontSize', 12);
title('Feature Importance', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% --- ROW 3: Spike History Kernel ---
if config.include_history
    subplot(3, 1, 3);

    % Extract history weights
    n_lags = config.history_lags;
    history_idx_start = find(strcmp(predictor_names, 'History_lag1'), 1);

    if ~isempty(history_idx_start)
        history_idx_end = history_idx_start + n_lags - 1;
        history_weights = coefficients(history_idx_start:history_idx_end);

        % Plot
        lag_times = (1:n_lags) * config.bin_size * 1000;  % Convert to ms
        plot(lag_times, history_weights, 'k-', 'LineWidth', 2);
        hold on;
        plot([0, max(lag_times)], [0, 0], 'r--', 'LineWidth', 1);  % Zero line

        xlabel('Lag (ms)', 'FontSize', 12);
        ylabel('Weight (z-scored units)', 'FontSize', 12);
        title('Spike History Kernel', 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        xlim([0, max(lag_times)]);

        % Add annotations
        text(0.02, 0.95, 'Refractory period / Bursting', 'Units', 'normalized', ...
             'FontSize', 10, 'Color', [0.5 0.5 0.5]);
    end
end

saveas(gcf, fullfile(output_dir, 'Test_6_Example_Unit_Details.png'));
fprintf('  ✓ Saved Test_6_Example_Unit_Details.png\n');

fprintf('\n=== TEST COMPLETE ===\n');
fprintf('All figures saved to: %s/\n', output_dir);
fprintf('\nSummary:\n');
fprintf('  Units fitted: %d\n', unit_counter);
fprintf('  Mean deviance explained: %.2f%% (±%.2f%%)\n', mean(dev_exp), std(dev_exp));
fprintf('  Mean firing rate: %.2f spikes/s (±%.2f)\n', mean(firing_rates), std(firing_rates));

fprintf('\nIf results look good, run full analysis with Unit_Poisson_GLM_Analysis.m\n');


%% ========================================================================
%  HELPER FUNCTIONS (COPY FROM MAIN SCRIPT)
%% ========================================================================

% Include all helper functions from Unit_Poisson_GLM_Analysis.m
% (buildDesignMatrixForSession, createRaisedCosineBasis, etc.)

% [Rest of the helper functions copied from main script...]
save_filename = 'Unit_GLM_Results.mat';
save(save_filename, 'glm_result', '-v7.3');

fprintf('✓ Results saved to: %s\n', save_filename);
fprintf('\n=== GLM ANALYSIS COMPLETE ===\n');
fprintf('Total units analyzed: %d\n', unit_counter);
fprintf('\nNext step: Run Visualize_GLM_Results.m\n');


%% ========================================================================
%  HELPER FUNCTIONS
%% ========================================================================

function [dm, predictor_info] = buildDesignMatrixForSession(...
    NeuralTime, IR1ON, IR2ON, AversiveSound, ...
    AdjustedXYZ_speed, Signal, Fs, config)
% Build design matrix with Gaussian basis for events and breathing amplitude

    % Create time bins
    t_start = NeuralTime(1);
    t_end = NeuralTime(end);
    time_bins = t_start:config.bin_size:t_end;
    n_bins = length(time_bins) - 1;
    time_centers = time_bins(1:end-1) + config.bin_size/2;

    % Initialize predictor storage
    predictor_names = {};
    predictor_matrix = [];

    %% 1. EVENT PREDICTORS with raised cosine basis

    % Define event signals and names
    event_signals = {IR1ON, IR2ON, AversiveSound};
    event_names = {'IR1ON', 'IR2ON', 'Aversive'};

    % Create Gaussian basis functions with specified centers spanning [-1, +2] seconds
    n_basis = config.n_basis_funcs;
    n_bins_pre = round(config.event_window_pre / config.bin_size);   % 20 bins (1 sec before)
    n_bins_post = round(config.event_window_post / config.bin_size); % 40 bins (2 sec after)
    event_duration_bins = n_bins_pre + n_bins_post;                  % 60 bins total (3 sec window)
    %%
    basis_funcs = createRaisedCosineBasis(n_basis, event_duration_bins, config.bin_size, config.basis_stretch);
    %%
    for ev = 1:length(event_signals)
        event_signal = event_signals{ev};
        event_name = event_names{ev};

        % Find event onsets based on event type
        if strcmp(event_name, 'Aversive')
            % Aversive: Use individual onsets
            event_onset_indices = find(diff([0; event_signal(:)]) == 1);
            if ~isempty(event_onset_indices)
                event_times_to_use = NeuralTime(event_onset_indices);
            else
                event_times_to_use = [];
            end
        else
            % Reward events (IR1ON, IR2ON): Use bout detection
            event_onset_indices = find(event_signal == 1);
            if ~isempty(event_onset_indices)
                event_times_raw = NeuralTime(event_onset_indices);
                try
                    [bout_starts, ~] = findEventCluster_SuperFast(event_times_raw, ...
                        config.bout_epsilon, config.bout_minPts);
                    event_times_to_use = bout_starts;
                catch
                    % Fall back to individual events if bout detection fails
                    event_times_to_use = event_times_raw;
                end
            else
                event_times_to_use = [];
            end
        end

        % Create binary event indicator (fully vectorized)
        event_indicator = zeros(n_bins, 1);
        if ~isempty(event_times_to_use)
            % Vectorized: bin all event times at once using histcounts
            event_counts = histcounts(event_times_to_use, time_bins)';
            event_indicator = event_counts;

            % Note: histcounts automatically handles events outside range
            % Multiple events in same bin are summed (correct behavior)
        end

        % FAST: Use convolution to apply basis functions to all events at once
        % This is ~100x faster than nested loops over events and bins
        for b = 1:n_basis
            kernel = basis_funcs(:, b);  % Length: event_duration_bins (60)

            % Pad event indicator to handle edge effects
            event_padded = [zeros(n_bins_pre, 1); event_indicator; zeros(n_bins_post - 1, 1)];

            % Convolve: predictor[t] = sum over delays: event[t-delay] * kernel[delay]
            predictor_full = conv(event_padded, kernel, 'valid');

            % 'valid' returns only the part where kernel fully overlaps with event_padded
            % This automatically gives us the correct length (n_bins)
            predictor = predictor_full;

            predictor_matrix = [predictor_matrix, predictor];
            predictor_names{end+1} = sprintf('%s_basis%d', event_name, b);
        end
    end

    %% 2. CONTINUOUS PREDICTORS

    % Speed
    speed_binned = binContinuousSignal(AdjustedXYZ_speed, NeuralTime, time_centers);
    speed_smoothed = smoothdata(speed_binned, 'gaussian', round(config.smooth_window/config.bin_size));
    speed_normalized = zscore(speed_smoothed);

    predictor_matrix = [predictor_matrix, speed_normalized];
    predictor_names{end+1} = 'Speed';

    % Breathing amplitude at 1.5 Hz and 8 Hz (from LFP Signal)
    if ~isempty(Signal) && Fs > 0
        % --- 1.5 Hz breathing amplitude ---
        band_range_1p5 = [1.0, 2.0];  % 1 Hz wide band centered around 1.5 Hz
        Signal_filtered_1p5 = bandpass(Signal(:,32), band_range_1p5, Fs, 'ImpulseResponse', 'fir', 'Steepness', 0.85);
        amplitude_envelope_1p5 = abs(hilbert(Signal_filtered_1p5));

        % Bin, smooth, and normalize 1.5 Hz amplitude
        amplitude_binned_1p5 = binContinuousSignal(amplitude_envelope_1p5, NeuralTime, time_centers);
        amplitude_smoothed_1p5 = smoothdata(amplitude_binned_1p5, 'gaussian', round(config.smooth_window/config.bin_size));
        amplitude_normalized_1p5 = zscore(amplitude_smoothed_1p5);

        predictor_matrix = [predictor_matrix, amplitude_normalized_1p5];
        predictor_names{end+1} = 'BreathingAmplitude_1.5Hz';

        % --- 8 Hz breathing amplitude ---
        band_range_8 = [7.5, 8.5];  % 1 Hz wide band centered around 8 Hz
        Signal_filtered_8 = bandpass(Signal(:,32), band_range_8, Fs, 'ImpulseResponse', 'fir', 'Steepness', 0.85);
        amplitude_envelope_8 = abs(hilbert(Signal_filtered_8));

        % Bin, smooth, and normalize 8 Hz amplitude
        amplitude_binned_8 = binContinuousSignal(amplitude_envelope_8, NeuralTime, time_centers);
        amplitude_smoothed_8 = smoothdata(amplitude_binned_8, 'gaussian', round(config.smooth_window/config.bin_size));
        amplitude_normalized_8 = zscore(amplitude_smoothed_8);

        predictor_matrix = [predictor_matrix, amplitude_normalized_8];
        predictor_names{end+1} = 'BreathingAmplitude_8Hz';

        breathing_normalized = amplitude_normalized_1p5;  % For interaction term (if enabled)
    else
        % If Signal not available, add zeros for both frequencies
        predictor_matrix = [predictor_matrix, zeros(n_bins, 2)];
        predictor_names{end+1} = 'BreathingAmplitude_1.5Hz';
        predictor_names{end+1} = 'BreathingAmplitude_8Hz';
        breathing_normalized = zeros(n_bins, 1);
    end

    %% 3. INTERACTION TERMS
    if config.include_interactions
        % Speed × BreathingAmplitude interaction
        interaction = speed_normalized .* breathing_normalized;
        predictor_matrix = [predictor_matrix, interaction];
        predictor_names{end+1} = 'Speed_x_BreathingAmplitude';
    end

    %% 4. Add bias term (intercept)
    predictor_matrix = [ones(n_bins, 1), predictor_matrix];
    predictor_names = ['Bias', predictor_names];

    % Package outputs
    dm = struct();
    dm.X = predictor_matrix;
    dm.time_bins = time_bins;
    dm.time_centers = time_centers;

    predictor_info = struct();
    predictor_info.names = predictor_names;
    predictor_info.n_predictors = length(predictor_names);
    predictor_info.n_basis = n_basis;
    predictor_info.event_names = event_names;
end


function basis = createRaisedCosineBasis(n_basis, n_bins, bin_size, stretch_param)
% Create raised cosine log-stretched basis functions
%
% Inputs:
%   n_basis:        Number of basis functions (e.g., 10)
%   n_bins:         Number of time bins to cover (total duration / bin_size)
%   bin_size:       Size of each time bin in seconds (e.g., 0.05 for 50ms)
%   stretch_param:  Log-stretching parameter (smaller = more compression early)
%
% Output:
%   basis: [n_bins × n_basis] matrix where each column is a raised cosine bump
%
% Example:
%   n_basis = 10;
%   n_bins = 60;           % 3 seconds / 0.05s = 60 bins
%   bin_size = 0.05;       % 50ms bins
%   stretch_param = 0.1;
%   basis = createRaisedCosineBasis(n_basis, n_bins, bin_size, stretch_param);

    basis = zeros(n_bins, n_basis);

    % Time vector (0 to n_bins-1, representing window duration)
    t_vec = (0:n_bins-1)' * bin_size;  % Column vector

    % Total window duration in seconds
    window_duration = (n_bins - 1) * bin_size;

    % Define logarithmically spaced peak positions
    % Map peaks from log-space to linear time
    % Fix: Ensure first peak is at t=0 to capture immediate response
    log_min = log(stretch_param);
    log_max = log(window_duration + stretch_param);
    log_peaks = linspace(log_min, log_max, n_basis + 1);
    peaks = exp(log_peaks) - stretch_param;  % Convert back to linear time

    % Take first n_basis peaks (first peak at ~0)
    peaks = peaks(1:n_basis);

    % Width of each cosine bump (distance between peaks)
    widths = diff(exp(log_peaks));

    % Create raised cosine bumps
    for i = 1:n_basis
        % Distance from peak
        distance = (t_vec - peaks(i)) / widths(i);

        % Raised cosine: 0.5 * (1 + cos(pi * distance)) for |distance| <= 1
        % Zero elsewhere
        valid = abs(distance) <= 1;
        basis(valid, i) = 0.5 * (1 + cos(pi * distance(valid)));

        % Normalize to sum to 1
        if sum(basis(:, i)) > 0
            basis(:, i) = basis(:, i) / sum(basis(:, i));
        end
    end
end


function binned_signal = binContinuousSignal(signal, time_stamps, bin_centers)
% Bin a continuous signal into specified time bins (VECTORIZED)
%
% Inputs:
%   signal:      [N × 1] continuous signal
%   time_stamps: [N × 1] time stamps for signal
%   bin_centers: [M × 1] desired bin centers
%
% Output:
%   binned_signal: [M × 1] signal averaged in each bin

    n_bins = length(bin_centers);

    % Create bin edges from bin centers (fully vectorized)
    bin_edges = zeros(n_bins + 1, 1);
    bin_edges(1) = -inf;  % First edge at -inf
    bin_edges(end) = inf;  % Last edge at +inf

    % Middle edges are midpoints between consecutive centers (vectorized)
    bin_edges(2:n_bins) = (bin_centers(1:n_bins-1) + bin_centers(2:n_bins)) / 2;

    % VECTORIZED: Assign each time_stamp to a bin using discretize
    bin_indices = discretize(time_stamps, bin_edges);

    % VECTORIZED: Average signal values within each bin using accumarray
    % accumarray(bin_indices, signal, [n_bins, 1], @mean, 0)
    % - Groups signal by bin_indices
    % - Applies @mean to each group
    % - Returns array of size [n_bins, 1]
    % - Fills empty bins with 0

    % Remove NaN bin assignments (outside range, though shouldn't happen with -inf/inf)
    valid = ~isnan(bin_indices);
    bin_indices = bin_indices(valid);
    signal = signal(valid);

    if ~isempty(bin_indices)
        binned_signal = accumarray(bin_indices, signal, [n_bins, 1], @mean, 0);
    else
        binned_signal = zeros(n_bins, 1);
    end
end


function spike_counts = binSpikeTrain(spike_times, time_bins)
% Bin spike times into discrete counts
%
% Inputs:
%   spike_times: [N × 1] vector of spike times
%   time_bins:   [M × 1] bin edges
%
% Output:
%   spike_counts: [M-1 × 1] spike counts per bin

    spike_counts = histcounts(spike_times, time_bins)';
end


function [X_augmented, predictor_info_augmented] = addSpikeHistoryToDesignMatrix(X, spike_counts, predictor_info, config)
% Add spike history (autoregressive) terms to design matrix
%
% Inputs:
%   X:              [N × P] original design matrix
%   spike_counts:   [N × 1] spike counts for this unit
%   predictor_info: Original predictor info struct
%   config:         Configuration with history_lags parameter
%
% Outputs:
%   X_augmented:           [N × (P + n_lags)] design matrix with history columns
%   predictor_info_augmented: Updated predictor info

    n_bins = length(spike_counts);
    n_lags = config.history_lags;

    % Create lagged spike count matrix
    history_matrix = zeros(n_bins, n_lags);

    for lag = 1:n_lags
        % Shift spike counts by 'lag' bins
        % lag=1 means previous bin, lag=2 means 2 bins ago, etc.
        history_matrix(lag+1:end, lag) = spike_counts(1:end-lag);
    end

    % Normalize each lag column (z-score)
    % This prevents recent lags from dominating just because they're larger
    for lag = 1:n_lags
        if std(history_matrix(:, lag)) > 0
            history_matrix(:, lag) = zscore(history_matrix(:, lag));
        end
    end

    % Insert history columns after all other predictors (but keep bias at position 1)
    % Original structure: [Bias, Events, Speed, BreathingAmplitude_1.5Hz, BreathingAmplitude_8Hz, Interaction]
    % New structure: [Bias, Events, Speed, BreathingAmplitude_1.5Hz, BreathingAmplitude_8Hz, Interaction, History_lag1, ..., History_lag150]
    X_augmented = [X, history_matrix];

    % Update predictor names
    predictor_names_augmented = predictor_info.names;
    for lag = 1:n_lags
        predictor_names_augmented{end+1} = sprintf('History_lag%d', lag);
    end

    % Update predictor info
    predictor_info_augmented = predictor_info;
    predictor_info_augmented.names = predictor_names_augmented;
    predictor_info_augmented.n_predictors = length(predictor_names_augmented);
    predictor_info_augmented.n_history_lags = n_lags;
    predictor_info_augmented.history_start_idx = size(X, 2) + 1;
    predictor_info_augmented.history_end_idx = size(X, 2) + n_lags;
end


function glm_result = fitPoissonGLM(X, y, predictor_info, config)
% Fit Poisson GLM and compute model statistics
%
% Inputs:
%   X:              [N × P] design matrix
%   y:              [N × 1] spike counts
%   predictor_info: Struct with predictor metadata
%   config:         Configuration struct
%
% Output:
%   glm_result:     Struct with coefficients and statistics

    % Remove any NaN or Inf
    valid_idx = all(isfinite(X), 2) & isfinite(y);
    X = X(valid_idx, :);
    y = y(valid_idx);

    %% Fit full model
    opts = optimoptions('fminunc', ...
        'Algorithm', 'quasi-newton', ...
        'MaxIterations', config.max_iter, ...
        'Display', config.display_fitting, ...
        'SpecifyObjectiveGradient', true);

    % Initialize with least squares
    w_init = X \ y;

    % Poisson negative log-likelihood
    objective_func = @(w) poissonNegLogLikelihood(w, X, y);

    % Optimize
    [w_ml, nll, exitflag, output] = fminunc(objective_func, w_init, opts);

    % Compute deviance explained (compare to null model with only intercept)
    mean_rate = mean(y);
    if mean_rate > 0
        % Null model: constant firing rate = mean(y)
        lambda_null = mean_rate;
        u_null = log(lambda_null);
    else
        % If no spikes, use very low rate
        u_null = -10;
        lambda_null = exp(u_null);
    end

    % Compute null model likelihood manually (avoids dimension issues)
    null_ll = -sum(y * u_null - length(y) * lambda_null);

    % Deviance explained (pseudo-R²)
    deviance_explained = 1 - (nll / null_ll);

    %% Fit reduced models (leave-one-predictor-group-out)
    feature_importance = computeFeatureImportance(X, y, w_ml, nll, predictor_info, config);

    %% Package results
    glm_result = struct();
    glm_result.coefficients = w_ml;
    glm_result.predictor_names = predictor_info.names;
    glm_result.nll = nll;
    glm_result.deviance_explained = deviance_explained;
    glm_result.feature_importance = feature_importance;
    glm_result.exitflag = exitflag;
    glm_result.n_iterations = output.iterations;
end


function [nll, grad] = poissonNegLogLikelihood(w, X, y)
% Compute Poisson negative log-likelihood and gradient
%
% Inputs:
%   w: [P × 1] weight vector
%   X: [N × P] design matrix
%   y: [N × 1] spike counts
%
% Outputs:
%   nll:  Scalar negative log-likelihood
%   grad: [P × 1] gradient vector

    % Linear prediction
    u = X * w;

    % Firing rate (exponential nonlinearity)
    lambda = exp(u);

    % Negative log-likelihood
    nll = -sum(y .* u - lambda);

    % Gradient
    if nargout > 1
        grad = -X' * (y - lambda);
    end
end


function feature_importance = computeFeatureImportance(X, y, w_full, nll_full, predictor_info, config)
% Compute feature importance by fitting reduced models
%
% Inputs:
%   X:              Design matrix
%   y:              Spike counts
%   w_full:         Full model weights
%   nll_full:       Full model negative log-likelihood
%   predictor_info: Predictor metadata
%   config:         Configuration
%
% Output:
%   feature_importance: Struct with importance for each predictor group

    feature_groups = struct();

    % Define predictor groups
    n_basis = predictor_info.n_basis;
    event_names = predictor_info.event_names;

    % Event groups (each event has n_basis predictors)
    for ev = 1:length(event_names)
        start_idx = 2 + (ev - 1) * n_basis;  % +2 for bias and previous events
        end_idx = start_idx + n_basis - 1;
        feature_groups.(event_names{ev}) = start_idx:end_idx;
    end

    % Continuous predictors
    speed_idx = 2 + length(event_names) * n_basis;
    breathing_1p5Hz_idx = speed_idx + 1;
    breathing_8Hz_idx = speed_idx + 2;

    feature_groups.Speed = speed_idx;
    feature_groups.BreathingAmplitude_1p5Hz = breathing_1p5Hz_idx;
    feature_groups.BreathingAmplitude_8Hz = breathing_8Hz_idx;

    % Interaction term (if present)
    current_idx = breathing_8Hz_idx + 1;
    if config.include_interactions
        feature_groups.Speed_x_BreathingAmplitude = current_idx;
        current_idx = current_idx + 1;
    end

    % Spike history (if present)
    if config.include_history && isfield(predictor_info, 'n_history_lags')
        history_start = predictor_info.history_start_idx;
        history_end = predictor_info.history_end_idx;
        feature_groups.History = history_start:history_end;
    end

    % Compute importance for each group
    group_names = fieldnames(feature_groups);
    feature_importance = struct();

    for g = 1:length(group_names)
        group_name = group_names{g};
        exclude_idx = feature_groups.(group_name);

        % Create reduced design matrix (exclude this group)
        keep_idx = setdiff(1:size(X, 2), exclude_idx);
        X_reduced = X(:, keep_idx);

        % Fit reduced model
        try
            opts = optimoptions('fminunc', ...
                'Algorithm', 'quasi-newton', ...
                'MaxIterations', config.max_iter, ...
                'Display', 'off', ...
                'SpecifyObjectiveGradient', true);

            w_init = X_reduced \ y;
            objective_func = @(w) poissonNegLogLikelihood(w, X_reduced, y);
            [~, nll_reduced] = fminunc(objective_func, w_init, opts);

            % Deviance difference (how much worse is reduced model?)
            deviance_diff = 2 * (nll_reduced - nll_full);

            feature_importance.(group_name) = struct();
            feature_importance.(group_name).deviance_diff = deviance_diff;
            feature_importance.(group_name).percent_deviance = deviance_diff / (2 * nll_full) * 100;

        catch
            feature_importance.(group_name) = struct();
            feature_importance.(group_name).deviance_diff = NaN;
            feature_importance.(group_name).percent_deviance = NaN;
        end
    end
end


function cmap = redblue(m)
% Red-Blue colormap
    if nargin < 1
        m = 256;
    end

    r = [(0:m/2-1)'/max(m/2-1,1); ones(m/2,1)];
    g = [(0:m/2-1)'/max(m/2-1,1); (m/2-1:-1:0)'/max(m/2-1,1)];
    b = [ones(m/2,1); (m/2-1:-1:0)'/max(m/2-1,1)];

    cmap = [r g b];
end
