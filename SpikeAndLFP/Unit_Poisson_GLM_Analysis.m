%% ========================================================================
%  POISSON GLM ANALYSIS: Nested Model Comparison
%  ========================================================================
%
%  Fits 4 nested Poisson GLMs to disentangle neural encoding:
%    Model 1: Spike History only (autoregressive baseline)
%    Model 2: Spike History + Events (IR1ON, IR2ON, Aversive Sound)
%    Model 3: Spike History + Events + Speed
%    Model 4: Spike History + Events + Speed + Breathing 8Hz
%
%  Features:
%    - Raised cosine log-stretched basis for event predictors
%    - Processes ALL units across all sessions
%    - Computes deviance explained for model comparison
%    - Saves results for integration with existing pipeline
%
%  Output: Unit_GLM_Nested_Results.mat
%% ========================================================================

clear all
close all

fprintf('\n=== POISSON GLM: NESTED MODEL ANALYSIS ===\n\n');

%% ========================================================================
%  CONFIGURATION
%% ========================================================================

config = struct();

% Timing parameters
config.bin_size = 0.05;                 % 50 ms bins
config.unit_of_time = 's';

% Raised cosine basis parameters
config.basis_type = 'raised_cosine';
config.n_basis_funcs = 10;              % Number of basis functions
config.event_window_pre = 0;            % No pre-event window (causal only)
config.event_window_post = 2.0;         % 2s post-event
config.basis_stretch = 0.1;             % Log-stretching parameter

% Continuous predictor smoothing
config.smooth_window = 0.2;             % 200 ms smoothing

% Model fitting
config.max_iter = 500;                 % Increased for better convergence
config.display_fitting = 'iter';

% Breathing amplitude extraction
config.breathing_band = [7.5, 8.5];     % 8Hz ± 0.5Hz band

% Spike history parameters
config.history_lags = 5;                % Number of history lags (250ms)
config.min_FR = 0;

% Bout detection parameters (for reward events)
config.bout_epsilon = 2.0;              % Maximum time between events in bout (seconds)
config.bout_minPts = 2;                 % Minimum events to form a bout

% Robust GLM fitting options (adapted from GLMspiketools)
config.use_regularization = false;      % Set to true to use L2 ridge regularization
config.lambda_grid = 2.^(-5:15);        % Regularization grid for cross-validation
config.cv_folds = 5;                    % Number of cross-validation folds

% Data paths
config.prediction_folder = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/BehaviorPrediction';
config.spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';
config.numofsession = 999;

% Add toolboxes
addpath(genpath('neuroGLM'));
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/NewScripts/');

fprintf('Configuration:\n');
fprintf('  Bin size: %d ms\n', config.bin_size * 1000);
fprintf('  Basis functions: %d raised cosine (log-stretched)\n', config.n_basis_funcs);
fprintf('  Event window: 0 to +%d ms\n', config.event_window_post * 1000);
fprintf('  Breathing band: %.1f-%.1f Hz\n', config.breathing_band(1), config.breathing_band(2));
fprintf('  Fitting method: Robust (GLMspiketools-style)\n');
fprintf('  Regularization: %s\n\n', mat2str(config.use_regularization));

%% ========================================================================
%  LOAD DATA
%% ========================================================================

fprintf('Loading sorting parameters...\n');
[T_sorted] = loadSortingParameters();
fprintf('✓ Loaded\n\n');

%% ========================================================================
%  PROCESS ALL SESSIONS
%% ========================================================================

% Find all sessions
session_types = {'Aversive', 'Reward'};
session_patterns = {'2025*RewardAversive*.mat', '2025*RewardSeeking*.mat'};

all_results = struct();
unit_counter = 0;

for session_type_idx = 1:2
    session_type = session_types{session_type_idx};
    session_pattern = session_patterns{session_type_idx};

    fprintf('=== %s SESSIONS ===\n', upper(session_type));

    % Select files
    [allfiles, ~, ~, sessions] = selectFilesWithAnimalIDFiltering(...
        config.spike_folder, config.numofsession, session_pattern);
    n_sessions = length(allfiles);

    fprintf('Found %d sessions\n\n', n_sessions);

    %% Process each session
    for sess_idx = 1:n_sessions
        fprintf('Session %d/%d: %s\n', sess_idx, n_sessions, allfiles(sess_idx).name);

        % Load session data
        Timelimits = 'No';
        try
            [NeuralTime, ~, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, ~, ~, ...
             AversiveSound, ~, valid_spikes, Fs, ~] = ...
                loadAndPrepareSessionData(allfiles(sess_idx), T_sorted, Timelimits);
        catch ME
            fprintf('  ERROR loading: %s\n', ME.message);
            continue;
        end

        n_units = length(valid_spikes);
        fprintf('  Units: %d\n', n_units);

        % Build design matrices for all 4 models
        fprintf('  Building design matrices...\n');
        [DM1, DM2, DM3, predictor_info] = buildNestedDesignMatrices(...
            NeuralTime, IR1ON, IR2ON, AversiveSound, AdjustedXYZ_speed, Signal, Fs, config);

        fprintf('    Model 1: %d predictors (events only)\n', size(DM1, 2));
        fprintf('    Model 2: %d predictors (events + speed)\n', size(DM2, 2));
        fprintf('    Model 3: %d predictors (events + speed + breathing)\n', size(DM3, 2));
        fprintf('    Model 4: %d predictors (full model + history)\n', size(DM3, 2) + config.history_lags);

        % Process each unit
        for unit_idx = 1:n_units
            spike_times = valid_spikes{unit_idx};
            if isempty(spike_times)
                continue;
            end

            % Bin spikes
            time_bins = predictor_info.time_bins;
            spike_counts = histcounts(spike_times, time_bins)';

            % Skip units with very low firing rates
            mean_fr = mean(spike_counts) / config.bin_size;
            if mean_fr < config.min_FR 
                continue;
            end

            unit_counter = unit_counter + 1;

            % Fit 4 nested models using robust method (GLMspiketools-style)
            try
                [model1, model2, model3, model4] = fitNestedModels_robust(...
                    spike_counts, DM1, DM2, DM3, config);

                % Store results
                all_results(unit_counter).session_name = allfiles(sess_idx).name;
                all_results(unit_counter).session_type = session_type;
                all_results(unit_counter).unit_idx = unit_idx;
                all_results(unit_counter).mean_firing_rate = mean_fr;

                % Model 1: Events only
                all_results(unit_counter).model1 = model1;

                % Model 2: Events + Speed
                all_results(unit_counter).model2 = model2;

                % Model 3: Events + Speed + Breathing
                all_results(unit_counter).model3 = model3;

                % Model 4: Full model + Spike History
                all_results(unit_counter).model4 = model4;

                % Store predictor info
                all_results(unit_counter).predictor_info = predictor_info;

                % Progress
                if mod(unit_counter, 50) == 0
                    fprintf('    Processed %d units...\n', unit_counter);
                end

            catch ME
                fprintf('    Unit %d FAILED: %s\n', unit_idx, ME.message);
                continue;
            end
        end

        fprintf('  ✓ Session complete\n\n');
    end
end

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Total units analyzed: %d\n', unit_counter);

%% ========================================================================
%  SAVE RESULTS
%% ========================================================================

fprintf('\nSaving results...\n');
save('Unit_GLM_Nested_Results.mat', 'all_results', 'config', '-v7.3');
fprintf('✓ Saved to Unit_GLM_Nested_Results.mat\n');

%% ========================================================================
%  SUMMARY STATISTICS
%% ========================================================================

fprintf('\n=== SUMMARY ===\n');

% Compute average deviance explained
dev_model1 = mean(arrayfun(@(s) s.model1.deviance_explained, all_results));
dev_model2 = mean(arrayfun(@(s) s.model2.deviance_explained, all_results));
dev_model3 = mean(arrayfun(@(s) s.model3.deviance_explained, all_results));
dev_model4 = mean(arrayfun(@(s) s.model4.deviance_explained, all_results));

fprintf('Average Deviance Explained:\n');
fprintf('  Model 1 (Events):                      %.2f%%\n', dev_model1);
fprintf('  Model 2 (Events + Speed):              %.2f%%\n', dev_model2);
fprintf('  Model 3 (Events + Speed + Breathing):  %.2f%%\n', dev_model3);
fprintf('  Model 4 (Full + History):              %.2f%%\n', dev_model4);
fprintf('\nAverage Improvement:\n');
fprintf('  Speed adds:                            %.2f%%\n', dev_model2 - dev_model1);
fprintf('  Breathing adds:                        %.2f%%\n', dev_model3 - dev_model2);
fprintf('  Spike history adds:                    %.2f%%\n', dev_model4 - dev_model3);

fprintf('\nDone!\n');


%% ========================================================================
%  HELPER FUNCTIONS
%% ========================================================================

function [DM1, DM2, DM3, predictor_info] = buildNestedDesignMatrices(...
    NeuralTime, IR1ON, IR2ON, AversiveSound, Speed, Signal, Fs, config)
% Build 3 nested design matrices (Model 4 adds spike history per-unit)
%
% Model 1: Events only
% Model 2: Events + Speed
% Model 3: Events + Speed + Breathing 8Hz
% Model 4: Events + Speed + Breathing + Spike History (added in fitNestedModels)

    % Create time bins
    t_start = NeuralTime(1);
    t_end = NeuralTime(end);
    time_bins = t_start:config.bin_size:t_end;
    n_bins = length(time_bins) - 1;
    time_centers = time_bins(1:end-1) + config.bin_size/2;

    %% 1. EVENT PREDICTORS (shared by all models)

    event_signals = {IR1ON, IR2ON, AversiveSound};
    event_names = {'IR1ON', 'IR2ON', 'Aversive'};

    % Create raised cosine basis
    n_basis = config.n_basis_funcs;
    n_bins_pre = round(config.event_window_pre / config.bin_size);
    n_bins_post = round(config.event_window_post / config.bin_size);
    event_duration_bins = n_bins_pre + n_bins_post;

    basis_funcs = createRaisedCosineBasis(n_basis, event_duration_bins, ...
        config.bin_size, config.basis_stretch);

    % Convolve events with basis
    event_predictors = [];
    for ev = 1:length(event_signals)
        event_signal = event_signals{ev};
        event_name = event_names{ev};

        % Find onsets based on event type
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

        % Create event indicator
        event_indicator = zeros(n_bins, 1);
        if ~isempty(event_times_to_use)
            event_counts = histcounts(event_times_to_use, time_bins)';
            event_indicator = event_counts;
        end

        % Convolve with each basis function
        for b = 1:n_basis
            kernel = basis_funcs(:, b);
            event_padded = [zeros(n_bins_pre, 1); event_indicator; zeros(n_bins_post - 1, 1)];
            predictor = conv(event_padded, kernel, 'valid');
            event_predictors = [event_predictors, predictor];
        end
    end

    % Z-score all event predictors for numerical stability and interpretability
    for col = 1:size(event_predictors, 2)
        if std(event_predictors(:, col)) > 0
            event_predictors(:, col) = zscore(event_predictors(:, col));
        end
    end

    %% 2. SPEED PREDICTOR

    speed_binned = binContinuousSignal(Speed, NeuralTime, time_centers);
    speed_smoothed = smoothdata(speed_binned, 'gaussian', round(config.smooth_window/config.bin_size));
    speed_normalized = zscore(speed_smoothed);

    %% 3. BREATHING 8Hz AMPLITUDE

    breathing_8Hz = zeros(n_bins, 1);
    if ~isempty(Signal) && Fs > 0
        % Extract breathing from channel 32
        breathing_signal = Signal(:, 32);

        % Bandpass filter at 8Hz
        Signal_filtered = bandpass(breathing_signal, config.breathing_band, Fs, ...
            'ImpulseResponse', 'fir', 'Steepness', 0.85);

        % Hilbert envelope
        amplitude_envelope = abs(hilbert(Signal_filtered));

        % Bin, smooth, normalize
        amplitude_binned = binContinuousSignal(amplitude_envelope, NeuralTime, time_centers);
        amplitude_smoothed = smoothdata(amplitude_binned, 'gaussian', ...
            round(config.smooth_window/config.bin_size));
        breathing_8Hz = zscore(amplitude_smoothed);
    end

    %% 4. ASSEMBLE NESTED MODELS

    % Model 1: Bias + Events
    DM1 = [ones(n_bins, 1), event_predictors];

    % Model 2: Bias + Events + Speed
    DM2 = [ones(n_bins, 1), event_predictors, speed_normalized(:)];

    % Model 3: Bias + Events + Speed + Breathing
    DM3 = [ones(n_bins, 1), event_predictors, speed_normalized(:), breathing_8Hz(:)];

    %% 5. PREDICTOR INFO

    predictor_info = struct();
    predictor_info.time_bins = time_bins;
    predictor_info.time_centers = time_centers;
    predictor_info.event_names = event_names;
    predictor_info.n_basis = n_basis;
    predictor_info.n_events = length(event_names);

    % Predictor names for Model 3 (full)
    predictor_names = {'Bias'};
    for ev = 1:length(event_names)
        for b = 1:n_basis
            predictor_names{end+1} = sprintf('%s_basis%d', event_names{ev}, b);
        end
    end
    predictor_names{end+1} = 'Speed';
    predictor_names{end+1} = 'Breathing_8Hz';

    predictor_info.predictor_names = predictor_names;
end

function basis = createRaisedCosineBasis(n_basis, n_bins, bin_size, stretch_param)
% Create raised cosine log-stretched basis functions
%
% Inputs:
%   n_basis:        Number of basis functions
%   n_bins:         Number of time bins
%   bin_size:       Size of each bin (seconds)
%   stretch_param:  Log-stretching parameter
%
% Output:
%   basis: [n_bins × n_basis] matrix

    basis = zeros(n_bins, n_basis);
    t_vec = (0:n_bins-1)' * bin_size;
    window_duration = (n_bins - 1) * bin_size;

    % Logarithmically spaced peaks
    log_min = log(stretch_param);
    log_max = log(window_duration + stretch_param);
    log_peaks = linspace(log_min, log_max, n_basis + 2);
    peaks = exp(log_peaks) - stretch_param;
    peaks = peaks(2:end-1);  % Remove boundary peaks

    % Width of each cosine bump
    widths = diff(exp(log_peaks));

    % Create raised cosine bumps
    for i = 1:n_basis
        distance = (t_vec - peaks(i)) / widths(i);
        valid = abs(distance) <= 1;
        basis(valid, i) = 0.5 * (1 + cos(pi * distance(valid)));

        % Normalize
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
%   bin_centers: [M × 1] bin center times
%
% Output:
%   binned_signal: [M × 1] binned signal

    binned_signal = interp1(time_stamps, signal, bin_centers, 'nearest', 'extrap');
end
