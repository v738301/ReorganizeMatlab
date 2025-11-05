%% ========================================================================
%  BEHAVIOR-SPECIFIC SPIKE-PHASE COUPLING ANALYSIS (NARROW BANDS)
%  Analyzes spike-phase coupling separately for each behavioral class
%  Focuses on three narrow frequency bands: 1-3 Hz, 5-7 Hz, 8-10 Hz
%  Includes statistical reliability measures for all units
%  ========================================================================
%
%  This script:
%  1. Maps LSTM behavioral predictions to neural sampling rate
%  2. Calculates spike-phase coupling for each unit × narrow band × behavior
%  3. Computes confidence intervals and reliability scores
%  4. Includes ALL units regardless of firing rate (no filtering)
%
%  Focus on narrow bands to match coherence analysis:
%    - 1-3 Hz:  Delta/slow oscillations
%    - 5-7 Hz:  Lower theta (matches coherence analysis)
%    - 8-10 Hz: Upper theta (matches coherence analysis)
%
%% ========================================================================

clear all;
% close all;

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== BEHAVIOR-SPECIFIC SPIKE-PHASE COUPLING (NARROW BANDS) ===\n\n');

config = struct();

% Narrow frequency bands to analyze (matching coherence analysis)
config.frequency_bands = {
    'Delta_1_3Hz',    [1, 3];
    'Theta_5_7Hz',    [5, 7];
    'Theta_8_10Hz',   [8, 10];
};

% Behavioral classes
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};

% LFP filtering parameters
config.bp_range = [1 300];  % Bandpass filter range for raw signal

% Phase histogram parameters
config.n_phase_bins = 18;  % 18 bins = 20 degrees each

% Statistical parameters
config.alpha = 0.05;  % Significance level for Rayleigh test
config.bootstrap_samples = 500;  % For MRL confidence intervals
config.ci_level = 0.95;  % 95% confidence intervals

% Reliability thresholds (based on spike count)
config.reliability_thresholds = struct();
config.reliability_thresholds.very_low = 10;      % < 10 spikes
config.reliability_thresholds.low = 50;           % 10-49 spikes
config.reliability_thresholds.moderate = 100;     % 50-99 spikes
config.reliability_thresholds.good = 500;         % 100-499 spikes
config.reliability_thresholds.excellent = 500;    % >= 500 spikes

% Camera frame rate
config.camera_fps = 20;  % 20 Hz (20 frames = 1 sec)

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');

fprintf('Configuration:\n');
fprintf('  Narrow frequency bands: %d\n', size(config.frequency_bands, 1));
for i = 1:size(config.frequency_bands, 1)
    fprintf('    %d. %s: %.1f-%.1f Hz\n', i, config.frequency_bands{i,1}, ...
        config.frequency_bands{i,2}(1), config.frequency_bands{i,2}(2));
end
fprintf('  Behavioral classes: %d\n', length(config.behavior_names));
fprintf('  Bootstrap samples: %d\n', config.bootstrap_samples);
fprintf('  All units included (no spike count filtering)\n\n');

%% ========================================================================
%  SECTION 2: LOAD SORTING PARAMETERS
%  ========================================================================

fprintf('Loading sorting parameters...\n');
[T_sorted] = loadSortingParameters();
fprintf('✓ Sorting parameters loaded\n\n');

%% ========================================================================
%  SECTION 3: LOAD BEHAVIOR PREDICTION DATA
%  ========================================================================

fprintf('Loading LSTM behavior predictions...\n');

% Load aversive behavior predictions
try
    pred_data_aversive = load('lstm_prediction_results_aversive_27-Oct-2025');
    prediction_sessions_aversive = pred_data_aversive.final_results.session_predictions;
    fprintf('✓ Loaded aversive behavior predictions: %d sessions\n', length(prediction_sessions_aversive));
catch ME
    fprintf('❌ Failed to load aversive behavior data: %s\n', ME.message);
    return;
end

% Load reward behavior predictions
try
    pred_data_reward = load('lstm_prediction_results_reward_27-Oct-2025');
    prediction_sessions_reward = pred_data_reward.final_results.session_predictions;
    fprintf('✓ Loaded reward behavior predictions: %d sessions\n\n', length(prediction_sessions_reward));
catch ME
    fprintf('❌ Failed to load reward behavior data: %s\n', ME.message);
    return;
end

% Load coupling data for session matching
try
    coupling_data_aversive = load('27-Oct-2025_RewardAversive_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_aversive = coupling_data_aversive.all_session_metrics;
    coupling_data_reward = load('27-Oct-2025_RewardSeeking_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_reward = coupling_data_reward.all_session_metrics;
    fprintf('✓ Loaded coupling data for session matching\n\n');
catch ME
    fprintf('❌ Failed to load coupling data: %s\n', ME.message);
    return;
end

%% ========================================================================
%  SECTION 4: CREATE SAVE DIRECTORIES
%  ========================================================================

DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_SpikePhase_BehaviorSpecific_NarrowBands');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_SpikePhase_BehaviorSpecific_NarrowBands');

if ~exist(RewardSeekingPath, 'dir')
    mkdir(RewardSeekingPath);
end
if ~exist(RewardAversivePath, 'dir')
    mkdir(RewardAversivePath);
end

%% ========================================================================
%  SECTION 5: PROCESS AVERSIVE SESSIONS
%  ========================================================================

fprintf('\n==== PROCESSING AVERSIVE SESSIONS ====\n');

numofsession = 2;
folderpath = "/Volumes/ExpansionBackUp/Data/Struct_spike";
[allfiles, folderpath, num_aversive_sessions] = selectFilesWithAnimalIDFiltering(folderpath, numofsession, '2025*RewardAversive*.mat');

fprintf('Found %d aversive sessions\n', num_aversive_sessions);

for spike_sess_idx = 1:num_aversive_sessions
    fprintf('\n[%d/%d] Processing: %s\n', spike_sess_idx, num_aversive_sessions, allfiles(spike_sess_idx).name);
    tic;

    % Load session data
    Timelimits = 'No';
    try
        [NeuralTime, AdjustedXYZ, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, WP1ON, WP2ON, ...
         AversiveSound, sessionLabels, valid_spikes, Fs, TriggerMid] = ...
            loadAndPrepareSessionData(allfiles(spike_sess_idx), T_sorted, Timelimits);
    catch ME
        fprintf('  ERROR loading session: %s\n', ME.message);
        continue;
    end

    spike_filename = allfiles(spike_sess_idx).name;

    %% Preprocess LFP
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);
    bestChannel = findBestLFPChannel(filtered_data, Fs);
    LFP = filtered_data(:, bestChannel);

    fprintf('  LFP channel: %d, Duration: %.1f min, Units: %d\n', ...
        bestChannel, (NeuralTime(end) - NeuralTime(1))/60, length(valid_spikes));

    %% Match with behavior prediction session
    matched = false;
    BehaviorClass = [];

    for beh_sess_idx = 1:length(sessions_aversive)
        beh_session = sessions_aversive{beh_sess_idx};

        if contains(spike_filename, extractBefore(beh_session.filename, '.mat'))
            matched = true;

            % Get behavior predictions
            prediction_scores = prediction_sessions_aversive(beh_sess_idx).prediction_scores;
            n_predictions = size(prediction_scores, 1);
            [max_confidence, dominant_beh] = max(prediction_scores, [], 2);

            % Map predictions to NeuralTime (1 Hz predictions → neural sampling rate)
            BehaviorClass = zeros(size(NeuralTime));
            for pred_idx = 1:n_predictions
                frame_start = (pred_idx - 1) * config.camera_fps + 1;
                frame_end = min(frame_start + config.camera_fps - 1, length(TriggerMid));

                if frame_end <= length(TriggerMid)
                    pred_time_start = TriggerMid(frame_start);
                    pred_time_end = TriggerMid(frame_end);

                    neural_indices = find(NeuralTime >= pred_time_start & NeuralTime <= pred_time_end);
                    BehaviorClass(neural_indices) = dominant_beh(pred_idx);
                end
            end

            fprintf('  Matched with behavior session: %s\n', beh_session.filename);
            fprintf('  Behavior predictions: %d predictions (1 Hz)\n', n_predictions);
            break;
        end
    end

    if ~matched
        fprintf('  WARNING: No behavior match - skipping session\n');
        continue;
    end

    %% Calculate behavior-specific spike-phase coupling
    session_results = calculate_spike_phase_coupling_by_behavior(valid_spikes, LFP, NeuralTime, Fs, BehaviorClass, config);

    %% Add session metadata
    session_results.session_id = spike_sess_idx;
    session_results.filename = spike_filename;
    session_results.session_type = 'RewardAversive';
    session_results.session_duration_min = (NeuralTime(end) - NeuralTime(1)) / 60;
    session_results.best_channel = bestChannel;
    session_results.Fs = Fs;
    session_results.NeuralTime_start = NeuralTime(1);
    session_results.NeuralTime_end = NeuralTime(end);

    %% Save results
    [~, base_filename, ~] = fileparts(spike_filename);
    save_filename = fullfile(RewardAversivePath, sprintf('%s_spike_phase_coupling_narrowbands.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    elapsed = toc;
    fprintf('  Completed in %.1f seconds. Saved to: %s\n', elapsed, save_filename);
end

fprintf('\n✓ Processed %d aversive sessions\n', num_aversive_sessions);

%% ========================================================================
%  SECTION 6: PROCESS REWARD SESSIONS
%  ========================================================================

fprintf('\n==== PROCESSING REWARD SESSIONS ====\n');

[allfiles, folderpath, num_reward_sessions] = selectFilesWithAnimalIDFiltering(folderpath, numofsession, '2025*RewardSeeking*.mat');

fprintf('Found %d reward sessions\n', num_reward_sessions);

for spike_sess_idx = 1:num_reward_sessions
    fprintf('\n[%d/%d] Processing: %s\n', spike_sess_idx, num_reward_sessions, allfiles(spike_sess_idx).name);
    tic;

    % Load session data
    Timelimits = 'No';
    try
        [NeuralTime, AdjustedXYZ, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, WP1ON, WP2ON, ...
         AversiveSound, sessionLabels, valid_spikes, Fs, TriggerMid] = ...
            loadAndPrepareSessionData(allfiles(spike_sess_idx), T_sorted, Timelimits);
    catch ME
        fprintf('  ERROR loading session: %s\n', ME.message);
        continue;
    end

    spike_filename = allfiles(spike_sess_idx).name;

    %% Preprocess LFP
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);
    bestChannel = findBestLFPChannel(filtered_data, Fs);
    LFP = filtered_data(:, bestChannel);

    fprintf('  LFP channel: %d, Duration: %.1f min, Units: %d\n', ...
        bestChannel, (NeuralTime(end) - NeuralTime(1))/60, length(valid_spikes));

    %% Match with behavior prediction session
    matched = false;
    BehaviorClass = [];

    for beh_sess_idx = 1:length(sessions_reward)
        beh_session = sessions_reward{beh_sess_idx};

        if contains(spike_filename, extractBefore(beh_session.filename, '.mat'))
            matched = true;

            % Get behavior predictions
            prediction_scores = prediction_sessions_reward(beh_sess_idx).prediction_scores;
            n_predictions = size(prediction_scores, 1);
            [max_confidence, dominant_beh] = max(prediction_scores, [], 2);

            % Map predictions to NeuralTime
            BehaviorClass = zeros(size(NeuralTime));
            for pred_idx = 1:n_predictions
                frame_start = (pred_idx - 1) * config.camera_fps + 1;
                frame_end = min(frame_start + config.camera_fps - 1, length(TriggerMid));

                if frame_end <= length(TriggerMid)
                    pred_time_start = TriggerMid(frame_start);
                    pred_time_end = TriggerMid(frame_end);

                    neural_indices = find(NeuralTime >= pred_time_start & NeuralTime <= pred_time_end);
                    BehaviorClass(neural_indices) = dominant_beh(pred_idx);
                end
            end

            fprintf('  Matched with behavior session: %s\n', beh_session.filename);
            fprintf('  Behavior predictions: %d predictions (1 Hz)\n', n_predictions);
            break;
        end
    end

    if ~matched
        fprintf('  WARNING: No behavior match - skipping session\n');
        continue;
    end

    %% Calculate behavior-specific spike-phase coupling
    session_results = calculate_spike_phase_coupling_by_behavior(...
        valid_spikes, LFP, NeuralTime, Fs, BehaviorClass, config);

    %% Add session metadata
    session_results.session_id = spike_sess_idx;
    session_results.filename = spike_filename;
    session_results.session_type = 'RewardSeeking';
    session_results.session_duration_min = (NeuralTime(end) - NeuralTime(1)) / 60;
    session_results.best_channel = bestChannel;
    session_results.Fs = Fs;
    session_results.NeuralTime_start = NeuralTime(1);
    session_results.NeuralTime_end = NeuralTime(end);

    %% Save results
    [~, base_filename, ~] = fileparts(spike_filename);
    save_filename = fullfile(RewardSeekingPath, sprintf('%s_spike_phase_coupling_narrowbands.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    elapsed = toc;
    fprintf('  Completed in %.1f seconds. Saved to: %s\n', elapsed, save_filename);
end

fprintf('\n✓ Processed %d reward sessions\n', num_reward_sessions);

%% ========================================================================
%  SUMMARY
%  ========================================================================

fprintf('\n========================================\n');
fprintf('ANALYSIS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Narrow bands analyzed:\n');
for i = 1:size(config.frequency_bands, 1)
    fprintf('  %s: %.1f-%.1f Hz\n', config.frequency_bands{i,1}, ...
        config.frequency_bands{i,2}(1), config.frequency_bands{i,2}(2));
end
fprintf('\nResults saved to:\n');
fprintf('  Reward-seeking: %s\n', RewardSeekingPath);
fprintf('  Reward-aversive: %s\n', RewardAversivePath);
fprintf('\nThese narrow bands match the coherence analysis bands\n');
fprintf('where differences were observed between session types.\n');
fprintf('========================================\n');
