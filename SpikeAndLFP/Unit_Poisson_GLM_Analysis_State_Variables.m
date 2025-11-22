%% ========================================================================
%  POISSON GLM ANALYSIS: 3 Nested Model Comparison WITH STATE VARIABLES
%  ========================================================================
%
%  Fits 3 nested Poisson GLMs to test incremental contributions:
%    Model 1: Bias + Spike History + Events + Coordinates + Spatial + STATES (baseline)
%    Model 2: Model 1 + Speed
%    Model 3: Model 2 + Breathing
%
%  This uses the SAME predictors as the 3-model analysis, but adds STATE
%  variables (Period Indicators) to the BASE MODEL.
%
%  State Definitions:
%    Aversive Sessions: Defined by Aversive Sound onsets (approx 4 periods)
%    Reward Sessions: Defined by 8-minute time blocks
%
%  Output: Unit_GLM_Nested_Results_StateVariables.mat
%% ========================================================================

clear all
close all

fprintf('\n=== POISSON GLM: 3 NESTED MODEL ANALYSIS (WITH STATES) ===\n\n');

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
config.max_iter = 500;                  % Increased for better convergence
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
config.lambda_grid = 2.^7;              % Regularization parameter
config.cv_folds = 1;                    % Number of cross-validation folds

% Data paths
config.prediction_folder = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/BehaviorPrediction';
config.spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';
config.numofsession = 999;

% Output
config.output_file = 'Unit_GLM_Nested_Results_StateVariables.mat';

% Exclude specific sessions (list of session file names to skip)
config.exclude_sessions = {
    '2025-07-09_15-16-04-RK02-RewardAversive-withbreathing_AllStruct_NeuralTime.mat' % aversive session 19
    '2025-07-12_15-34-53-RK02-RewardAversive-withbreathing_AllStruct_NeuralTime.mat' % aversive session 20
    '2025-07-08_14-19-00-RK02-RewardSeeking-withbreathing_AllStruct_NeuralTime.mat' % reward session 19
    '2025-07-11_16-10-40-RK02-RewardSeeking-withbreathing_AllStruct_NeuralTime.mat' % reward session 20
    };

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
fprintf('  Regularization: %s\n', mat2str(config.use_regularization));
fprintf('  State Variables included in Base Model\n');
if ~isempty(config.exclude_sessions)
    fprintf('  Excluded sessions: %d\n', length(config.exclude_sessions));
end
fprintf('\n');

%% ========================================================================
%  LOAD DATA
%% ========================================================================

fprintf('Loading sorting parameters...\n');
[T_sorted] = loadSortingParameters();
fprintf('✓ Loaded\n\n');

%% ========================================================================
%  PROCESS ALL SESSIONS
%% ========================================================================

% Find all sessions (Both Aversive and Reward)
session_types = {'Aversive', 'Reward'};
session_patterns = {'2025*RewardAversive*.mat', '2025*RewardSeeking*.mat'};

all_results = struct();
unit_counter = 0;

for session_type_idx = 1:length(session_types)
    session_type = session_types{session_type_idx};
    session_pattern = session_patterns{session_type_idx};
    
    fprintf('=== %s SESSIONS ===\n', upper(session_type));
    
    % Select files
    [allfiles, ~, ~, sessions] = selectFilesWithAnimalIDFiltering(...
        config.spike_folder, config.numofsession, session_pattern);
    
    % Filter out excluded sessions
    if ~isempty(config.exclude_sessions)
        file_names = {allfiles.name};
        exclude_mask = ismember(file_names, config.exclude_sessions);
        allfiles = allfiles(~exclude_mask);
        if sum(exclude_mask) > 0
            fprintf('  Excluded %d sessions from analysis\n', sum(exclude_mask));
        end
    end
    
    n_sessions = length(allfiles);
    fprintf('Found %d sessions\n\n', n_sessions);
    
    %% Process each session
    for sess_idx = 1:n_sessions
        fprintf('Session %d/%d: %s\n', sess_idx, n_sessions, allfiles(sess_idx).name);
        
        % Load session data
        Timelimits = 'No';
        try
            [NeuralTime, AdjustedXYZ, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, WP1ON, WP2ON, ...
                AversiveSound, ~, valid_spikes, Fs, TriggerMid] = ...
                loadAndPrepareSessionData(allfiles(sess_idx), T_sorted, Timelimits);
        catch ME
            fprintf('  ERROR loading: %s\n', ME.message);
            continue;
        end
        
        n_units = length(valid_spikes);
        fprintf('  Units: %d\n', n_units);
        
        % Build design matrices WITH STATE VARIABLES
        fprintf('  Building design matrices (including State Variables)...\n');
        [DM1, DM2, DM3, DM4, predictor_info] = buildNestedDesignMatricesWithStates(...
            NeuralTime, IR1ON, IR2ON, WP1ON, WP2ON, AversiveSound, AdjustedXYZ, AdjustedXYZ_speed, Signal, Fs, config, session_type, TriggerMid);
        
        fprintf('    Model 1: %d predictors (bias + spike history + events + coords + spatial + STATES)\n', 1 + config.history_lags + size(DM2, 2));
        fprintf('    Model 2: %d predictors (+ speeds)\n', 1 + config.history_lags + size(DM3, 2));
        fprintf('    Model 3: %d predictors (+ breathing)\n', 1 + config.history_lags + size(DM4, 2));
        
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
            
            % Fit 3 nested models using robust method (GLMspiketools-style)
            % Use DM2, DM3, DM4 (not DM1, DM2, DM3) because we want:
            %   Model 1 = Events + Coords + Spatial + STATES (DM2)
            %   Model 2 = Model 1 + Speeds (DM3)
            %   Model 3 = Model 2 + Breathing (DM4)
            try
                [model1, model2, model3] = fitNestedModels_robust_3models(...
                    spike_counts, DM2, DM3, DM4, config);
                
                % Store results
                all_results(unit_counter).session_name = allfiles(sess_idx).name;
                all_results(unit_counter).session_type = session_type;
                all_results(unit_counter).unit_idx = unit_idx;
                all_results(unit_counter).mean_firing_rate = mean_fr;
                
                % Model 1: Bias + Spike History + Events + Coords + Spatial + STATES (baseline)
                all_results(unit_counter).model1 = model1;
                
                % Model 2: + Speed
                all_results(unit_counter).model2 = model2;
                
                % Model 3: + Breathing
                all_results(unit_counter).model3 = model3;
                
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
save(config.output_file, 'all_results', 'config', '-v7.3');
fprintf('✓ Saved to %s\n', config.output_file);

%% ========================================================================
%  SUMMARY STATISTICS
%% ========================================================================

fprintf('\n=== SUMMARY ===\n');

if ~isempty(all_results)
    % Extract deviance explained for each model
    dev_exp_1 = arrayfun(@(x) x.model1.deviance_explained, all_results);
    dev_exp_2 = arrayfun(@(x) x.model2.deviance_explained, all_results);
    dev_exp_3 = arrayfun(@(x) x.model3.deviance_explained, all_results);
    
    fprintf('Deviance Explained (across %d units):\n', length(all_results));
    fprintf('  Model 1 (Baseline + States):  %.2f ± %.2f%%\n', mean(dev_exp_1), std(dev_exp_1));
    fprintf('  Model 2 (+ Speed):            %.2f ± %.2f%%\n', mean(dev_exp_2), std(dev_exp_2));
    fprintf('  Model 3 (+ Breathing):        %.2f ± %.2f%%\n', mean(dev_exp_3), std(dev_exp_3));
    
    % Additional variance explained
    add_var_speed = dev_exp_2 - dev_exp_1;
    add_var_breathing = dev_exp_3 - dev_exp_2;
    
    fprintf('\nAdditional Variance Explained:\n');
    fprintf('  Speed:     %.2f ± %.2f%% (range: %.2f to %.2f%%)\n', ...
        mean(add_var_speed), std(add_var_speed), min(add_var_speed), max(add_var_speed));
    fprintf('  Breathing: %.2f ± %.2f%% (range: %.2f to %.2f%%)\n', ...
        mean(add_var_breathing), std(add_var_breathing), min(add_var_breathing), max(add_var_breathing));
    
    % Count significant LRTs
    if isfield(all_results(1).model2, 'LRT_p_value')
        sig_speed = sum(arrayfun(@(x) x.model2.LRT_p_value < 0.05, all_results));
        sig_breathing = sum(arrayfun(@(x) x.model3.LRT_p_value < 0.05, all_results));
        
        fprintf('\nSignificant LRTs (p < 0.05):\n');
        fprintf('  Speed:     %d/%d (%.1f%%)\n', sig_speed, length(all_results), ...
            100*sig_speed/length(all_results));
        fprintf('  Breathing: %d/%d (%.1f%%)\n', sig_breathing, length(all_results), ...
            100*sig_breathing/length(all_results));
    end
end

fprintf('\n✓ Analysis complete!\n');


%% ========================================================================
%  HELPER FUNCTIONS
%% ========================================================================

function [DM1, DM2, DM3, DM4, predictor_info] = buildNestedDesignMatricesWithStates(...
    NeuralTime, IR1ON, IR2ON, WP1ON, WP2ON, AversiveSound, AdjustedXYZ, Speed, Signal, Fs, config, session_type, TriggerMid)
% Build 4 nested design matrices WITH STATE VARIABLES
%
% DM1: Events only (IR1ON, IR2ON, WP1ON, WP2ON with Gaussian; Aversive with raised cosine)
% DM2: Events + Coords + Spatial + STATES
% DM3: Events + Coords + Spatial + STATES + Speeds
% DM4: Events + Coords + Spatial + STATES + Speeds + Breathing
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
        n_bins_half = floor(length(kernel) / 2);
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
aversive_predictors = [];
for b = 1:n_basis
    kernel = basis_funcs(:, b);
    kernel_flipped = kernel;  % Flip for causal convolution
    aversive_padded = [aversive_indicator; zeros(n_bins_post_aversive, 1)];
    predictor_full = conv(aversive_padded, kernel_flipped, 'full');
    predictor = predictor_full(1:n_bins);
    aversive_predictors = [aversive_predictors, predictor];
end

% Combine all event predictors
event_predictors = [reward_predictors, aversive_predictors];

% Z-score all event predictors
for col = 1:size(event_predictors, 2)
    if std(event_predictors(:, col)) > 0
        event_predictors(:, col) = zscore(event_predictors(:, col));
    end
end

%% 2. STATE VARIABLES (PERIOD INDICATORS)

% Define Period Boundaries based on Session Type
period_boundaries = [];

if strcmpi(session_type, 'Aversive')
    % Aversive: Periods defined by aversive event onsets
    if length(aversive_times) >= 3
        % Standard pattern: Start -> Av1 -> Av2 -> Av3 -> End
        % Periods: P1 (Start-Av1), P2 (Av1-Av2), P3 (Av2-Av3), P4 (Av3-End)
        % Note: FiringRate_CV_Analysis uses TriggerMid(1) as start
        
        % Ensure we use TriggerMid if available, else NeuralTime
        if exist('TriggerMid', 'var') && ~isempty(TriggerMid)
            session_start = TriggerMid(1);
        else
            session_start = NeuralTime(1);
        end
        
        % Define boundaries: [Start, Av1, Av2, Av3, Av4...]
        % We typically expect 4 aversive events, but code should handle variable number
        % FiringRate_CV_Analysis logic:
        % period_boundaries = [TriggerMid(1), all_aversive_time(1:3)' + TriggerMid(1), all_aversive_time(4) + TriggerMid(1)];
        % This implies relative times? Let's assume aversive_times are absolute NeuralTime.
        
        period_boundaries = [session_start; aversive_times(:)];
        
        % If last aversive event is not near end, add end time?
        % Actually, let's define periods as intervals between these points.
        % P1: Start -> Av1
        % P2: Av1 -> Av2
        % ...
        % P_last: Av_last -> End
        period_boundaries = [period_boundaries; NeuralTime(end)];
        
    else
        % Fallback if not enough events
        period_boundaries = [NeuralTime(1), NeuralTime(end)];
    end
    
elseif strcmpi(session_type, 'Reward')
    % Reward: Fixed 8-minute blocks
    % 0, 8, 16, 24, 30 (or 32, 40...)
    
    if exist('TriggerMid', 'var') && ~isempty(TriggerMid)
        session_start = TriggerMid(1);
    else
        session_start = NeuralTime(1);
    end
    
    % Create 8-minute boundaries (8 * 60 seconds)
    block_size = 8 * 60;
    session_duration = NeuralTime(end) - session_start;
    
    % Generate boundaries: 0, 8, 16... until duration
    boundaries_rel = 0:block_size:session_duration;
    
    % Add end time if not exactly on block boundary
    if boundaries_rel(end) < session_duration
        boundaries_rel = [boundaries_rel, session_duration];
    end
    
    period_boundaries = session_start + boundaries_rel';
end

% Create State Predictors (One-hot encoding for each period)
n_periods = length(period_boundaries) - 1;
state_predictors = zeros(n_bins, n_periods);

for p = 1:n_periods
    p_start = period_boundaries(p);
    p_end = period_boundaries(p+1);
    
    % Find bins within this period
    % Use time_centers for assignment
    in_period = (time_centers >= p_start) & (time_centers < p_end);
    state_predictors(in_period, p) = 1;
end

% Remove the first period to avoid multicollinearity with Intercept?
% Or keep all and rely on regularization/constraints?
% Standard GLM practice with intercept: Remove one category (reference level).
% Let's remove the FIRST period (P1) as the reference baseline.
% So if we have 4 periods, we add 3 predictors (P2, P3, P4).
% If n_periods < 2, no state predictors.

if n_periods >= 2
    state_predictors_final = state_predictors(:, 2:end);
    state_names_final = {};
    for p = 2:n_periods
        state_names_final{end+1} = sprintf('State_Period_%d', p);
    end
else
    state_predictors_final = [];
    state_names_final = {};
end

%% 3. KINEMATICS PREDICTORS (with -1~1 sec kernel)

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
        n_bins_half = floor(length(kernel) / 2);
        signal_padded = [zeros(n_bins_half, 1); signal(:); zeros(n_bins_half, 1)];
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

%% 4. SPATIAL KERNELS

% 2D XY spatial kernel (position-dependent tuning)
n_spatial_bins_xy = 10;  % 10x10 grid
x_edges = linspace(min(X_coord), max(X_coord), n_spatial_bins_xy + 1);
y_edges = linspace(min(Y_coord), max(Y_coord), n_spatial_bins_xy + 1);

[~, ~, x_bin_idx] = histcounts(X_coord_binned, x_edges);
[~, ~, y_bin_idx] = histcounts(Y_coord_binned, y_edges);

xy_spatial_predictors = zeros(n_bins, n_spatial_bins_xy * n_spatial_bins_xy);
for i = 1:n_bins
    if x_bin_idx(i) > 0 && y_bin_idx(i) > 0
        spatial_idx = (y_bin_idx(i) - 1) * n_spatial_bins_xy + x_bin_idx(i);
        xy_spatial_predictors(i, spatial_idx) = 1;
    end
end

% 1D Z spatial kernel (height-dependent tuning)
n_spatial_bins_z = 10;  % 10 vertical bins
z_edges = linspace(0, 100, n_spatial_bins_z + 1);
[~, ~, z_bin_idx] = histcounts(Z_coord_binned, z_edges);

z_spatial_predictors = zeros(n_bins, n_spatial_bins_z);
for i = 1:n_bins
    if z_bin_idx(i) > 0
        z_spatial_predictors(i, z_bin_idx(i)) = 1;
    end
end

spatial_predictors = [xy_spatial_predictors, z_spatial_predictors];

%% 5. BREATHING AMPLITUDE (8Hz and 1.5Hz)

breathing_8Hz_predictors = [];
breathing_1p5Hz_predictors = [];

if ~isempty(Signal) && Fs > 0
    % Extract breathing from channel 32
    breathing_signal = Signal(:, 32);
    
    % Bandpass filter at 8Hz
    Signal_filtered_8Hz = bandpass(breathing_signal, config.breathing_band_8Hz, Fs, ...
        'ImpulseResponse', 'fir', 'Steepness', 0.85);
    amplitude_envelope_8Hz = abs(hilbert(Signal_filtered_8Hz));
    amplitude_binned_8Hz = binContinuousSignal(amplitude_envelope_8Hz, NeuralTime, time_centers);
    
    % Convolve with each of the 8 basis kernels
    for k = 1:config.n_continuous_kernels
        kernel = continuous_basis_kernels(:, k);
        kernel_flipped = flipud(kernel);
        n_bins_half = floor(length(kernel) / 2);
        signal_padded = [zeros(n_bins_half, 1); amplitude_binned_8Hz(:); zeros(n_bins_half, 1)];
        predictor_full = conv(signal_padded, kernel_flipped, 'valid');
        if length(predictor_full) > n_bins
            predictor_full = predictor_full(1:n_bins);
        elseif length(predictor_full) < n_bins
            predictor_full = [predictor_full; zeros(n_bins - length(predictor_full), 1)];
        end
        predictor_smoothed = smoothdata(predictor_full, 'gaussian', round(config.smooth_window/config.bin_size));
        breathing_8Hz_predictors = [breathing_8Hz_predictors, zscore(predictor_smoothed)];
    end
    
    % Bandpass filter at 1.5Hz
    Signal_filtered_1p5Hz = bandpass(breathing_signal, config.breathing_band_1p5Hz, Fs, ...
        'ImpulseResponse', 'fir', 'Steepness', 0.85);
    amplitude_envelope_1p5Hz = abs(hilbert(Signal_filtered_1p5Hz));
    amplitude_binned_1p5Hz = binContinuousSignal(amplitude_envelope_1p5Hz, NeuralTime, time_centers);
    
    for k = 1:config.n_continuous_kernels
        kernel = continuous_basis_kernels(:, k);
        kernel_flipped = flipud(kernel);
        n_bins_half = floor(length(kernel) / 2);
        signal_padded = [zeros(n_bins_half, 1); amplitude_binned_1p5Hz(:); zeros(n_bins_half, 1)];
        predictor_full = conv(signal_padded, kernel_flipped, 'valid');
        if length(predictor_full) > n_bins
            predictor_full = predictor_full(1:n_bins);
        elseif length(predictor_full) < n_bins
            predictor_full = [predictor_full; zeros(n_bins - length(predictor_full), 1)];
        end
        predictor_smoothed = smoothdata(predictor_full, 'gaussian', round(config.smooth_window/config.bin_size));
        breathing_1p5Hz_predictors = [breathing_1p5Hz_predictors, zscore(predictor_smoothed)];
    end
else
    breathing_8Hz_predictors = zeros(n_bins, config.n_continuous_kernels);
    breathing_1p5Hz_predictors = zeros(n_bins, config.n_continuous_kernels);
end

%% 6. ASSEMBLE NESTED MODELS

% Separate coordinates and speeds
n_kernels = config.n_continuous_kernels;
speed_predictors = kinematics_predictors(:, 1:(3*n_kernels));        % X, Y, Z speeds
coord_predictors = kinematics_predictors(:, (3*n_kernels+1):end);    % X, Y, Z coordinates

% DM1: Events only
DM1 = event_predictors;

% DM2: Events + Coordinates + Spatial + STATES
% This is the BASE MODEL for comparison
DM2 = [event_predictors, coord_predictors, spatial_predictors, state_predictors_final];

% DM3: Events + Coordinates + Spatial + STATES + Speeds
DM3 = [event_predictors, coord_predictors, spatial_predictors, state_predictors_final, speed_predictors];

% DM4: Events + Coordinates + Spatial + STATES + Speeds + Breathing
DM4 = [event_predictors, coord_predictors, spatial_predictors, state_predictors_final, speed_predictors, breathing_8Hz_predictors, breathing_1p5Hz_predictors];

%% 7. PREDICTOR INFO

predictor_info = struct();
predictor_info.time_bins = time_bins;
predictor_info.time_centers = time_centers;
predictor_info.history_lags = config.history_lags;
predictor_info.history_window_ms = config.history_lags * config.bin_size * 1000;

predictor_names = {};
predictor_names{end+1} = 'Intercept';
for lag = 1:config.history_lags
    predictor_names{end+1} = sprintf('SpikeHistory_lag%d', lag);
end
for ev = 1:length(reward_names)
    for k = 1:config.n_reward_kernels
        predictor_names{end+1} = sprintf('%s_kernel%d', reward_names{ev}, k);
    end
end
for b = 1:n_basis
    predictor_names{end+1} = sprintf('Aversive_basis%d', b);
end
coord_name = {'X_coord', 'Y_coord', 'Z_coord'};
for i = 1:length(coord_name)
    for k = 1:n_kernels
        predictor_names{end+1} = sprintf('%s_kernel%d', coord_name{i}, k);
    end
end
for i = 1:n_spatial_bins_xy * n_spatial_bins_xy
    predictor_names{end+1} = sprintf('XY_spatial_%d', i);
end
for i = 1:n_spatial_bins_z
    predictor_names{end+1} = sprintf('Z_spatial_%d', i);
end

% Add State Names
for i = 1:length(state_names_final)
    predictor_names{end+1} = state_names_final{i};
end

speed_name = {'X_speed', 'Y_speed', 'Z_speed'};
for i = 1:length(speed_name)
    for k = 1:n_kernels
        predictor_names{end+1} = sprintf('%s_kernel%d', speed_name{i}, k);
    end
end
for k = 1:n_kernels
    predictor_names{end+1} = sprintf('Breathing_8Hz_kernel%d', k);
end
for k = 1:n_kernels
    predictor_names{end+1} = sprintf('Breathing_1.5Hz_kernel%d', k);
end

predictor_info.predictor_names = predictor_names;
predictor_info.state_names = state_names_final;
end

function basis = createRaisedCosineBasis(n_basis, n_bins, bin_size, stretch_param)
basis = zeros(n_bins, n_basis);
t_vec = (0:n_bins-1)' * bin_size;
window_duration = (n_bins - 1) * bin_size;
log_min = log(stretch_param);
log_max = log(window_duration + stretch_param);
log_peaks = linspace(log_min, log_max, n_basis + 1);
peaks = exp(log_peaks) - stretch_param;
peaks = peaks(1:n_basis);
widths = diff(exp(log_peaks));
for i = 1:n_basis
    distance = (t_vec - peaks(i)) / widths(i);
    valid = abs(distance) <= 1;
    basis(valid, i) = 0.5 * (1 + cos(pi * distance(valid)));
    if sum(basis(:, i)) > 0
        basis(:, i) = basis(:, i) / sum(basis(:, i));
    end
end
end

function kernel = createGaussianKernel(n_bins_pre, n_bins_post, bin_size, sigma)
total_bins = n_bins_pre + n_bins_post;
t_vec = ((-n_bins_pre):(n_bins_post-1))' * bin_size;
kernel = exp(-(t_vec.^2) / (2 * sigma^2));
kernel = kernel / sum(kernel);
end

function binned_signal = binContinuousSignal(signal, time_stamps, bin_centers)
binned_signal = interp1(time_stamps, signal, bin_centers, 'nearest', 'extrap');
end

function basis_kernels = createGaussianBasisKernels(window_size, bin_size, hwhh, n_kernels)
sigma = hwhh / sqrt(2 * log(2));
padding = 3 * sigma;
extended_window = window_size + padding;
n_bins_half = round(extended_window / bin_size);
total_bins = 2 * n_bins_half + 1;
time_vec = ((-n_bins_half):n_bins_half)' * bin_size;
kernel_centers = linspace(-window_size, window_size, n_kernels);
basis_kernels = zeros(total_bins, n_kernels);
for k = 1:n_kernels
    center = kernel_centers(k);
    basis_kernels(:, k) = exp(-((time_vec - center).^2) / (2 * sigma^2));
    basis_kernels(:, k) = basis_kernels(:, k) / sum(basis_kernels(:, k));
end
end
