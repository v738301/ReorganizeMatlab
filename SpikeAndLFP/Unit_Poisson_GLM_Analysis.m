%% ========================================================================
%  POISSON GLM ANALYSIS: Nested Model Comparison
%  ========================================================================
%
%  Fits 5 nested Poisson GLMs to disentangle neural encoding:
%    Model 1: Bias + Spike History (autoregressive baseline)
%    Model 2: Bias + Spike History + Events (IR1ON, IR2ON, WP1ON, WP2ON with Gaussian; Aversive with raised cosine)
%    Model 3: Bias + Spike History + Events + Coordinates + Spatial kernels (2D XY & 1D Z)
%    Model 4: Bias + Spike History + Events + Coordinates + Spatial + Speeds (X,Y,Z speeds)
%    Model 5: Bias + Spike History + Events + Coordinates + Spatial + Speeds + Breathing (8Hz + 1.5Hz)
%
%  Features:
%    - Symmetric Gaussian kernel (-2~2 sec) for reward events (IR1/2ON, WP1/2ON)
%    - Raised cosine basis (0~2 sec) for aversive events
%    - Continuous kinematics with -1~1 sec kernel (X,Y,Z speeds & coordinates)
%    - Breathing signals (8Hz + 1.5Hz) with -1~1 sec kernel
%    - 2D XY spatial kernel (10x10 grid) and 1D Z spatial kernel (5 bins)
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

% Event kernel parameters
% IR1ON, IR2ON, WP1ON, WP2ON: 24 Gaussian basis functions (240ms HWHH, ±2 sec)
config.reward_kernel_type = 'gaussian_basis';
config.reward_window_pre = 2.0;         % 2s pre-event
config.reward_window_post = 2.0;        % 2s post-event
config.n_reward_kernels = 24;           % 24 evenly spaced Gaussian basis functions
config.reward_kernel_hwhh = 0.24;       % 240 ms half-width at half-height

% Aversive: Raised cosine (causal only)
config.aversive_kernel_type = 'raised_cosine';
config.n_basis_funcs = 10;              % Number of basis functions
config.aversive_window_pre = 0;         % No pre-event window (causal only)
config.aversive_window_post = 2.0;      % 2s post-event
config.basis_stretch = 0.1;             % Log-stretching parameter

% Continuous predictor kernel (-1 to +1 sec with 8 separate Gaussian kernels)
config.continuous_window = 1.0;         % ±1s window for continuous predictors
config.n_continuous_kernels = 8;        % 8 separate Gaussian basis kernels
config.continuous_kernel_hwhh = 0.24;   % 240 ms half-width at half-height

% Continuous predictor smoothing
config.smooth_window = 0.2;             % 200 ms smoothing

% Model fitting
config.max_iter = 500;                 % Increased for better convergence
config.display_fitting = 'iter';

% Breathing amplitude extraction
config.breathing_band_8Hz = [7.5, 8.5];     % 8Hz ± 0.5Hz band
config.breathing_band_1p5Hz = [1.0, 2.0];   % 1.5Hz ± 0.5Hz band

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
fprintf('  Reward events (IR/WP): %d Gaussian basis kernels [-%d, +%d] ms, HWHH=%.1f ms\n', ...
    config.n_reward_kernels, config.reward_window_pre * 1000, config.reward_window_post * 1000, config.reward_kernel_hwhh * 1000);
fprintf('  Aversive events: %d raised cosine basis [0, +%d] ms\n', ...
    config.n_basis_funcs, config.aversive_window_post * 1000);
fprintf('  Continuous predictors: %d Gaussian basis kernels ±%d ms, HWHH=%.1f ms\n', ...
    config.n_continuous_kernels, config.continuous_window * 1000, config.continuous_kernel_hwhh * 1000);
fprintf('  Breathing 8Hz band: %.1f-%.1f Hz\n', config.breathing_band_8Hz(1), config.breathing_band_8Hz(2));
fprintf('  Breathing 1.5Hz band: %.1f-%.1f Hz\n', config.breathing_band_1p5Hz(1), config.breathing_band_1p5Hz(2));
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
            [NeuralTime, AdjustedXYZ, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, WP1ON, WP2ON, ...
             AversiveSound, ~, valid_spikes, Fs, ~] = ...
                loadAndPrepareSessionData(allfiles(sess_idx), T_sorted, Timelimits);
        catch ME
            fprintf('  ERROR loading: %s\n', ME.message);
            continue;
        end

        n_units = length(valid_spikes);
        fprintf('  Units: %d\n', n_units);

        % Build design matrices for all 5 models
        fprintf('  Building design matrices...\n');
        [DM1, DM2, DM3, DM4, predictor_info] = buildNestedDesignMatrices(...
            NeuralTime, IR1ON, IR2ON, WP1ON, WP2ON, AversiveSound, AdjustedXYZ, AdjustedXYZ_speed, Signal, Fs, config);

        fprintf('    Model 1: %d predictors (bias + spike history)\n', 1 + config.history_lags);
        fprintf('    Model 2: %d predictors (+ events)\n', 1 + config.history_lags + size(DM1, 2));
        fprintf('    Model 3: %d predictors (+ coordinates + spatial)\n', 1 + config.history_lags + size(DM2, 2));
        fprintf('    Model 4: %d predictors (+ speeds)\n', 1 + config.history_lags + size(DM3, 2));
        fprintf('    Model 5: %d predictors (+ breathing)\n', 1 + config.history_lags + size(DM4, 2));

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

            % Fit 5 nested models using robust method (GLMspiketools-style)
            try
                [model1, model2, model3, model4, model5] = fitNestedModels_robust(...
                    spike_counts, DM1, DM2, DM3, DM4, config);

                % Store results
                all_results(unit_counter).session_name = allfiles(sess_idx).name;
                all_results(unit_counter).session_type = session_type;
                all_results(unit_counter).unit_idx = unit_idx;
                all_results(unit_counter).mean_firing_rate = mean_fr;

                % Model 1: Bias + Spike History
                all_results(unit_counter).model1 = model1;

                % Model 2: + Events
                all_results(unit_counter).model2 = model2;

                % Model 3: + Coordinates + Spatial
                all_results(unit_counter).model3 = model3;

                % Model 4: + Speeds
                all_results(unit_counter).model4 = model4;

                % Model 5: + Breathing
                all_results(unit_counter).model5 = model5;

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
dev_model5 = mean(arrayfun(@(s) s.model5.deviance_explained, all_results));

fprintf('Average Deviance Explained:\n');
fprintf('  Model 1 (Bias + Spike History):                %.2f%%\n', dev_model1);
fprintf('  Model 2 (+ Events):                            %.2f%%\n', dev_model2);
fprintf('  Model 3 (+ Coordinates + Spatial):             %.2f%%\n', dev_model3);
fprintf('  Model 4 (+ Speeds):                            %.2f%%\n', dev_model4);
fprintf('  Model 5 (+ Breathing):                         %.2f%%\n', dev_model5);
fprintf('\nAverage Improvement:\n');
fprintf('  Events add:                                    %.2f%%\n', dev_model2 - dev_model1);
fprintf('  Coordinates + Spatial add:                     %.2f%%\n', dev_model3 - dev_model2);
fprintf('  Speeds add:                                    %.2f%%\n', dev_model4 - dev_model3);
fprintf('  Breathing adds:                                %.2f%%\n', dev_model5 - dev_model4);

fprintf('\nDone!\n');


%% ========================================================================
%  HELPER FUNCTIONS
%% ========================================================================

function [DM1, DM2, DM3, DM4, predictor_info] = buildNestedDesignMatrices(...
    NeuralTime, IR1ON, IR2ON, WP1ON, WP2ON, AversiveSound, AdjustedXYZ, Speed, Signal, Fs, config)
% Build 4 nested design matrices (Model 1 adds spike history per-unit in fitNestedModels)
%
% DM1: Events only (IR1ON, IR2ON, WP1ON, WP2ON with Gaussian; Aversive with raised cosine)
% DM2: Events + Coordinates + Spatial kernels (2D XY & 1D Z)
% DM3: Events + Coordinates + Spatial + Speeds (X,Y,Z speeds)
% DM4: Events + Coordinates + Spatial + Speeds + Breathing (8Hz + 1.5Hz)
% Model 1 (bias + spike history) is created in fitNestedModels_robust

    % Create time bins
    t_start = NeuralTime(1);
    t_end = NeuralTime(end);
    time_bins = t_start:config.bin_size:t_end;
    n_bins = length(time_bins) - 1;
    time_centers = time_bins(1:end-1) + config.bin_size/2;

    %% 1. EVENT PREDICTORS (shared by all models)

    % Reward events: IR1ON, IR2ON, WP1ON, WP2ON (24 Gaussian basis kernels, -2~2 sec, 240ms HWHH)
    reward_events = {IR1ON, IR2ON, WP1ON, WP2ON};
    reward_names = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON'};

    % Create 24 Gaussian basis kernels for reward events
    reward_window_total = config.reward_window_pre + config.reward_window_post;
    reward_basis_kernels = createGaussianBasisKernels(...
        reward_window_total / 2, config.bin_size, ...
        config.reward_kernel_hwhh, config.n_reward_kernels);

    % Convolve reward events with 24 Gaussian basis kernels
    reward_predictors = [];
    for ev = 1:length(reward_events)
        event_signal = reward_events{ev};
        event_name = reward_names{ev};

        % Use bout detection for reward events
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

        % Create event indicator
        event_indicator = zeros(n_bins, 1);
        if ~isempty(event_times_to_use)
            event_counts = histcounts(event_times_to_use, time_bins)';
            event_indicator = event_counts;
        end

        % Convolve with each of the 24 Gaussian basis kernels
        for k = 1:config.n_reward_kernels
            kernel = reward_basis_kernels(:, k);
            kernel_flipped = flipud(kernel);  % Flip for proper temporal alignment

            % Pad signal for convolution
            n_bins_half = round(reward_window_total / 2 / config.bin_size);
            event_padded = [zeros(n_bins_half, 1); event_indicator; zeros(n_bins_half, 1)];
            predictor_full = conv(event_padded, kernel_flipped, 'valid');

            % Ensure correct length
            if length(predictor_full) > n_bins
                predictor_full = predictor_full(1:n_bins);
            elseif length(predictor_full) < n_bins
                predictor_full = [predictor_full; zeros(n_bins - length(predictor_full), 1)];
            end

            reward_predictors = [reward_predictors, predictor_full];
        end
    end

    % Aversive event: raised cosine basis (0~2 sec, causal only)
    n_basis = config.n_basis_funcs;
    n_bins_pre_aversive = round(config.aversive_window_pre / config.bin_size);
    n_bins_post_aversive = round(config.aversive_window_post / config.bin_size);
    aversive_duration_bins = n_bins_pre_aversive + n_bins_post_aversive;

    basis_funcs = createRaisedCosineBasis(n_basis, aversive_duration_bins, ...
        config.bin_size, config.basis_stretch);

    % Find aversive onsets
    event_onset_indices = find(diff([0; AversiveSound(:)]) == 1);
    if ~isempty(event_onset_indices)
        aversive_times = NeuralTime(event_onset_indices);
    else
        aversive_times = [];
    end

    % Create aversive event indicator
    aversive_indicator = zeros(n_bins, 1);
    if ~isempty(aversive_times)
        aversive_counts = histcounts(aversive_times, time_bins)';
        aversive_indicator = aversive_counts;
    end

    % Convolve aversive with raised cosine basis functions
    % Fix: Flip kernel for proper causal convolution (response AFTER event)
    aversive_predictors = [];
    for b = 1:n_basis
        kernel = basis_funcs(:, b);
        kernel_flipped = flipud(kernel);  % Flip for causal convolution
        % Pad only at the end for causal response
        aversive_padded = [aversive_indicator; zeros(n_bins_post_aversive, 1)];
        predictor_full = conv(aversive_padded, kernel_flipped, 'full');
        % Extract valid portion (same length as n_bins)
        predictor = predictor_full(1:n_bins);
        aversive_predictors = [aversive_predictors, predictor];
    end

    % Combine all event predictors
    event_predictors = [reward_predictors, aversive_predictors];

    % Z-score all event predictors for numerical stability and interpretability
    for col = 1:size(event_predictors, 2)
        if std(event_predictors(:, col)) > 0
            event_predictors(:, col) = zscore(event_predictors(:, col));
        end
    end

    %% 2. KINEMATICS PREDICTORS (with -1~1 sec kernel)

    % Extract X, Y, Z coordinates
    X_coord = AdjustedXYZ(:, 1);
    Y_coord = AdjustedXYZ(:, 2);
    Z_coord = AdjustedXYZ(:, 3);

    % Compute X, Y, Z speeds (derivatives)
    dt = diff(NeuralTime);
    dt(dt == 0) = mean(dt(dt > 0));  % Handle any zero time steps

    X_speed = [0; diff(X_coord) ./ dt];
    Y_speed = [0; diff(Y_coord) ./ dt];
    Z_speed = [0; diff(Z_coord) ./ dt];

    % Bin continuous signals
    X_coord_binned = binContinuousSignal(X_coord, NeuralTime, time_centers);
    Y_coord_binned = binContinuousSignal(Y_coord, NeuralTime, time_centers);
    Z_coord_binned = binContinuousSignal(Z_coord, NeuralTime, time_centers);

    X_speed_binned = binContinuousSignal(X_speed, NeuralTime, time_centers);
    Y_speed_binned = binContinuousSignal(Y_speed, NeuralTime, time_centers);
    Z_speed_binned = binContinuousSignal(Z_speed, NeuralTime, time_centers);

    % Create 8 separate Gaussian basis kernels for continuous predictors (±1 sec, 240ms HWHH each)
    continuous_basis_kernels = createGaussianBasisKernels(...
        config.continuous_window, config.bin_size, ...
        config.continuous_kernel_hwhh, config.n_continuous_kernels);

    % Convolve continuous signals with each of the 8 kernels
    kinematics_predictors = [];
    continuous_signals = {X_speed_binned, Y_speed_binned, Z_speed_binned, ...
                          X_coord_binned, Y_coord_binned, Z_coord_binned};
    continuous_names = {'X_speed', 'Y_speed', 'Z_speed', 'X_coord', 'Y_coord', 'Z_coord'};

    for i = 1:length(continuous_signals)
        signal = continuous_signals{i};

        % Convolve with each of the 8 basis kernels
        for k = 1:config.n_continuous_kernels
            kernel = continuous_basis_kernels(:, k);
            kernel_flipped = flipud(kernel);  % Flip for proper temporal alignment

            % Pad signal for convolution
            n_bins_half = round(config.continuous_window / config.bin_size);
            signal_padded = [zeros(n_bins_half, 1); signal; zeros(n_bins_half, 1)];
            predictor_full = conv(signal_padded, kernel_flipped, 'valid');

            % Ensure correct length
            if length(predictor_full) > n_bins
                predictor_full = predictor_full(1:n_bins);
            elseif length(predictor_full) < n_bins
                predictor_full = [predictor_full; zeros(n_bins - length(predictor_full), 1)];
            end

            % Smooth and normalize
            predictor_smoothed = smoothdata(predictor_full, 'gaussian', round(config.smooth_window/config.bin_size));
            predictor_normalized = zscore(predictor_smoothed);
            kinematics_predictors = [kinematics_predictors, predictor_normalized];
        end
    end

    %% 3. SPATIAL KERNELS

    % 2D XY spatial kernel (position-dependent tuning)
    % Create spatial grid
    n_spatial_bins_xy = 10;  % 10x10 grid
    x_edges = linspace(min(X_coord), max(X_coord), n_spatial_bins_xy + 1);
    y_edges = linspace(min(Y_coord), max(Y_coord), n_spatial_bins_xy + 1);

    % Bin XY positions
    [~, ~, x_bin_idx] = histcounts(X_coord_binned, x_edges);
    [~, ~, y_bin_idx] = histcounts(Y_coord_binned, y_edges);

    % Create one-hot encoding for 2D spatial bins
    xy_spatial_predictors = zeros(n_bins, n_spatial_bins_xy * n_spatial_bins_xy);
    for i = 1:n_bins
        if x_bin_idx(i) > 0 && y_bin_idx(i) > 0
            spatial_idx = (y_bin_idx(i) - 1) * n_spatial_bins_xy + x_bin_idx(i);
            xy_spatial_predictors(i, spatial_idx) = 1;
        end
    end

    % 1D Z spatial kernel (height-dependent tuning)
    n_spatial_bins_z = 5;  % 5 vertical bins
    z_edges = linspace(min(Z_coord), max(Z_coord), n_spatial_bins_z + 1);

    % Bin Z positions
    [~, ~, z_bin_idx] = histcounts(Z_coord_binned, z_edges);

    % Create one-hot encoding for 1D Z spatial bins
    z_spatial_predictors = zeros(n_bins, n_spatial_bins_z);
    for i = 1:n_bins
        if z_bin_idx(i) > 0
            z_spatial_predictors(i, z_bin_idx(i)) = 1;
        end
    end

    % Combine spatial predictors
    spatial_predictors = [xy_spatial_predictors, z_spatial_predictors];

    %% 4. BREATHING AMPLITUDE (8Hz and 1.5Hz) with 8 separate ±1 sec kernels

    breathing_8Hz_predictors = [];
    breathing_1p5Hz_predictors = [];

    if ~isempty(Signal) && Fs > 0
        % Extract breathing from channel 32
        breathing_signal = Signal(:, 32);

        % Bandpass filter at 8Hz
        Signal_filtered_8Hz = bandpass(breathing_signal, config.breathing_band_8Hz, Fs, ...
            'ImpulseResponse', 'fir', 'Steepness', 0.85);

        % Hilbert envelope for 8Hz
        amplitude_envelope_8Hz = abs(hilbert(Signal_filtered_8Hz));

        % Bin 8Hz
        amplitude_binned_8Hz = binContinuousSignal(amplitude_envelope_8Hz, NeuralTime, time_centers);

        % Convolve with each of the 8 basis kernels
        for k = 1:config.n_continuous_kernels
            kernel = continuous_basis_kernels(:, k);
            kernel_flipped = flipud(kernel);

            % Pad signal for convolution
            n_bins_half = round(config.continuous_window / config.bin_size);
            signal_padded = [zeros(n_bins_half, 1); amplitude_binned_8Hz; zeros(n_bins_half, 1)];
            predictor_full = conv(signal_padded, kernel_flipped, 'valid');

            % Ensure correct length
            if length(predictor_full) > n_bins
                predictor_full = predictor_full(1:n_bins);
            elseif length(predictor_full) < n_bins
                predictor_full = [predictor_full; zeros(n_bins - length(predictor_full), 1)];
            end

            % Smooth and normalize
            predictor_smoothed = smoothdata(predictor_full, 'gaussian', ...
                round(config.smooth_window/config.bin_size));
            breathing_8Hz_predictors = [breathing_8Hz_predictors, zscore(predictor_smoothed)];
        end

        % Bandpass filter at 1.5Hz
        Signal_filtered_1p5Hz = bandpass(breathing_signal, config.breathing_band_1p5Hz, Fs, ...
            'ImpulseResponse', 'fir', 'Steepness', 0.85);

        % Hilbert envelope for 1.5Hz
        amplitude_envelope_1p5Hz = abs(hilbert(Signal_filtered_1p5Hz));

        % Bin 1.5Hz
        amplitude_binned_1p5Hz = binContinuousSignal(amplitude_envelope_1p5Hz, NeuralTime, time_centers);

        % Convolve with each of the 8 basis kernels
        for k = 1:config.n_continuous_kernels
            kernel = continuous_basis_kernels(:, k);
            kernel_flipped = flipud(kernel);

            % Pad signal for convolution
            n_bins_half = round(config.continuous_window / config.bin_size);
            signal_padded = [zeros(n_bins_half, 1); amplitude_binned_1p5Hz; zeros(n_bins_half, 1)];
            predictor_full = conv(signal_padded, kernel_flipped, 'valid');

            % Ensure correct length
            if length(predictor_full) > n_bins
                predictor_full = predictor_full(1:n_bins);
            elseif length(predictor_full) < n_bins
                predictor_full = [predictor_full; zeros(n_bins - length(predictor_full), 1)];
            end

            % Smooth and normalize
            predictor_smoothed = smoothdata(predictor_full, 'gaussian', ...
                round(config.smooth_window/config.bin_size));
            breathing_1p5Hz_predictors = [breathing_1p5Hz_predictors, zscore(predictor_smoothed)];
        end
    else
        % If no breathing signal, create zero predictors
        breathing_8Hz_predictors = zeros(n_bins, config.n_continuous_kernels);
        breathing_1p5Hz_predictors = zeros(n_bins, config.n_continuous_kernels);
    end

    %% 5. ASSEMBLE NESTED MODELS

    % Separate coordinates and speeds from kinematics_predictors
    % kinematics_predictors now has 6 signals × 8 kernels = 48 predictors
    % Order: X_speed(8 kernels), Y_speed(8), Z_speed(8), X_coord(8), Y_coord(8), Z_coord(8)
    n_kernels = config.n_continuous_kernels;
    speed_predictors = kinematics_predictors(:, 1:(3*n_kernels));        % X, Y, Z speeds (24 predictors)
    coord_predictors = kinematics_predictors(:, (3*n_kernels+1):end);    % X, Y, Z coordinates (24 predictors)

    % DM1: Events only
    DM1 = event_predictors;

    % DM2: Events + Coordinates + Spatial kernels
    DM2 = [event_predictors, coord_predictors, spatial_predictors];

    % DM3: Events + Coordinates + Spatial + Speeds
    DM3 = [event_predictors, coord_predictors, spatial_predictors, speed_predictors];

    % DM4: Events + Coordinates + Spatial + Speeds + Breathing (8Hz: 8 kernels, 1.5Hz: 8 kernels)
    DM4 = [event_predictors, coord_predictors, spatial_predictors, speed_predictors, breathing_8Hz_predictors, breathing_1p5Hz_predictors];

    %% 6. PREDICTOR INFO

    predictor_info = struct();
    predictor_info.time_bins = time_bins;
    predictor_info.time_centers = time_centers;

    % Predictor names for Model 5 (full model)
    % Note: Bias and spike history are added in fitNestedModels_robust
    predictor_names = {};

    % Reward event predictors (24 Gaussian basis kernels per event)
    for ev = 1:length(reward_names)
        for k = 1:config.n_reward_kernels
            predictor_names{end+1} = sprintf('%s_kernel%d', reward_names{ev}, k);
        end
    end

    % Aversive event predictors (raised cosine basis)
    for b = 1:n_basis
        predictor_names{end+1} = sprintf('Aversive_basis%d', b);
    end

    % Coordinate predictors (each has 8 kernels)
    for i = 1:length({'X_coord', 'Y_coord', 'Z_coord'})
        coord_name = {'X_coord', 'Y_coord', 'Z_coord'}{i};
        for k = 1:n_kernels
            predictor_names{end+1} = sprintf('%s_kernel%d', coord_name, k);
        end
    end

    % Spatial predictors
    for i = 1:n_spatial_bins_xy * n_spatial_bins_xy
        predictor_names{end+1} = sprintf('XY_spatial_%d', i);
    end
    for i = 1:n_spatial_bins_z
        predictor_names{end+1} = sprintf('Z_spatial_%d', i);
    end

    % Speed predictors (each has 8 kernels)
    for i = 1:length({'X_speed', 'Y_speed', 'Z_speed'})
        speed_name = {'X_speed', 'Y_speed', 'Z_speed'}{i};
        for k = 1:n_kernels
            predictor_names{end+1} = sprintf('%s_kernel%d', speed_name, k);
        end
    end

    % Breathing (each has 8 kernels)
    for k = 1:n_kernels
        predictor_names{end+1} = sprintf('Breathing_8Hz_kernel%d', k);
    end
    for k = 1:n_kernels
        predictor_names{end+1} = sprintf('Breathing_1.5Hz_kernel%d', k);
    end

    predictor_info.predictor_names = predictor_names;
    predictor_info.reward_event_names = reward_names;
    predictor_info.continuous_names = continuous_names;
    predictor_info.n_spatial_bins_xy = n_spatial_bins_xy;
    predictor_info.n_spatial_bins_z = n_spatial_bins_z;
    predictor_info.n_reward_kernels = config.n_reward_kernels;
    predictor_info.n_aversive_basis = n_basis;
    predictor_info.n_continuous_kernels = n_kernels;
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


function kernel = createGaussianKernel(n_bins_pre, n_bins_post, bin_size, sigma)
% Create a symmetric Gaussian kernel
%
% Inputs:
%   n_bins_pre:  Number of bins before event
%   n_bins_post: Number of bins after event
%   bin_size:    Size of each bin (seconds)
%   sigma:       Standard deviation of Gaussian (seconds)
%
% Output:
%   kernel: [(n_bins_pre + n_bins_post) × 1] Gaussian kernel

    total_bins = n_bins_pre + n_bins_post;
    t_vec = ((-n_bins_pre):(n_bins_post-1))' * bin_size;

    % Create Gaussian centered at t=0
    kernel = exp(-(t_vec.^2) / (2 * sigma^2));

    % Normalize to sum to 1
    kernel = kernel / sum(kernel);
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

function basis_kernels = createGaussianBasisKernels(window_size, bin_size, hwhh, n_kernels)
% Create multiple Gaussian basis functions spanning ±window_size
%
% Inputs:
%   window_size: Total window (e.g., 1.0 for ±1 sec)
%   bin_size:    Bin size in seconds
%   hwhh:        Half-width at half-height in seconds (e.g., 0.24 for 240ms)
%   n_kernels:   Number of basis kernels (e.g., 8)
%
% Output:
%   basis_kernels: [n_bins × n_kernels] matrix of Gaussian basis functions

    % Convert HWHH to standard deviation
    % For Gaussian: HWHH = sigma * sqrt(2*ln(2))
    sigma = hwhh / sqrt(2 * log(2));

    % Total bins for ±window_size
    n_bins_half = round(window_size / bin_size);
    total_bins = 2 * n_bins_half + 1;  % Include center bin
    time_vec = ((-n_bins_half):n_bins_half)' * bin_size;

    % Create evenly spaced kernel centers across the window
    kernel_centers = linspace(-window_size, window_size, n_kernels);

    % Initialize basis matrix
    basis_kernels = zeros(total_bins, n_kernels);

    % Create each Gaussian kernel
    for k = 1:n_kernels
        center = kernel_centers(k);
        basis_kernels(:, k) = exp(-((time_vec - center).^2) / (2 * sigma^2));
        % Normalize each kernel to sum to 1
        basis_kernels(:, k) = basis_kernels(:, k) / sum(basis_kernels(:, k));
    end
end
