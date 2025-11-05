%% ========================================================================
%  SPIKE-LFP COHERENCE ANALYSIS (BEHAVIOR-SPECIFIC)
%  Computes coherence between spikes and LFP separately for each behavior
%  Uses multitaper method for robust spectral estimation
%  ========================================================================
%
%  This script:
%  1. Loads spike, LFP, and behavioral prediction data
%  2. For each unit × behavior combination, computes spike-LFP coherence
%  3. Uses multitaper spectral estimation (TW=3, K=5 tapers)
%  4. Saves results with coherence spectra for each unit × behavior
%
%% ========================================================================

clear all;
close all;

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== SPIKE-LFP COHERENCE ANALYSIS (BEHAVIOR-SPECIFIC) ===\n\n');

config = struct();

% Behavioral classes
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
config.n_behaviors = 7;

% Multitaper parameters (Chronux convention)
config.tapers = [3, 5];  % [TW, K]
config.freq_range = [1, 150];  % Hz
config.pad = 0;  % FFT padding: -1=none, 0=next power of 2, 1=2x next power of 2
config.window_size = 10;  % Window size in seconds for segmented coherence (to avoid memory issues)

% LFP filtering
config.bp_range = [1, 300];

% Minimum spikes per behavior (require sufficient data for coherence)
config.min_spikes_per_behavior = 30;  % At least 30 spikes during a behavior

% Camera frame rate
config.camera_fps = 20;

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');
addpath(genpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/chronux_2_12/chronux_2_12/'));

fprintf('Configuration:\n');
fprintf('  Behaviors: %d\n', config.n_behaviors);
fprintf('  Frequency range: %.1f-%.1f Hz\n', config.freq_range(1), config.freq_range(2));
fprintf('  Tapers: TW=%d, K=%d\n', config.tapers(1), config.tapers(2));
fprintf('  Window size: %d sec (windowed coherence to save memory)\n', config.window_size);
fprintf('  Minimum spikes per behavior: %d\n\n', config.min_spikes_per_behavior);

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

try
    pred_data_aversive = load('lstm_prediction_results_aversive_27-Oct-2025');
    prediction_sessions_aversive = pred_data_aversive.final_results.session_predictions;
    fprintf('✓ Loaded aversive predictions: %d sessions\n', length(prediction_sessions_aversive));
catch ME
    fprintf('❌ Failed to load aversive predictions: %s\n', ME.message);
    return;
end

try
    pred_data_reward = load('lstm_prediction_results_reward_27-Oct-2025');
    prediction_sessions_reward = pred_data_reward.final_results.session_predictions;
    fprintf('✓ Loaded reward predictions: %d sessions\n', length(prediction_sessions_reward));
catch ME
    fprintf('❌ Failed to load reward predictions: %s\n', ME.message);
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
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_SpikeLFPCoherence_BehaviorSpecific');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_SpikeLFPCoherence_BehaviorSpecific');

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
folderpath = "/Volumes/Expansion/Data/Struct_spike";
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

    %% Compute behavior-specific coherence for each unit
    n_units = length(valid_spikes);
    unit_coherence_results = cell(n_units, 1);

    coherence_params = struct();
    coherence_params.freq_range = config.freq_range;
    coherence_params.tapers = config.tapers;
    coherence_params.pad = config.pad;
    coherence_params.window_size = config.window_size;
    coherence_params.Fs = Fs;

    fprintf('  Computing behavior-specific coherence for %d units...\n', n_units);

    for unit_idx = 1:n_units
        spike_times = valid_spikes{unit_idx};
        n_spikes_total = length(spike_times);

        % Convert spike times to indices in NeuralTime
        spike_indices = interp1(NeuralTime, 1:length(NeuralTime), spike_times, 'nearest', 'extrap');
        valid_mask = spike_indices > 0 & spike_indices <= length(BehaviorClass);
        spike_indices = spike_indices(valid_mask);
        spike_behaviors = BehaviorClass(spike_indices);

        % Remove spikes with no behavior assignment
        behavior_mask = spike_behaviors > 0;
        spike_indices = spike_indices(behavior_mask);
        spike_behaviors = spike_behaviors(behavior_mask);
        spike_times_with_behavior = spike_times(valid_mask);
        spike_times_with_behavior = spike_times_with_behavior(behavior_mask);

        % Process each behavior
        behavior_results = cell(config.n_behaviors, 1);

        for beh_idx = 1:config.n_behaviors
            % Select spikes during this behavior
            beh_mask = spike_behaviors == beh_idx;
            beh_spike_times = spike_times_with_behavior(beh_mask);
            n_spikes_beh = length(beh_spike_times);

            if n_spikes_beh < config.min_spikes_per_behavior
                behavior_results{beh_idx} = struct('skipped', true, ...
                    'reason', 'insufficient_spikes', 'n_spikes', n_spikes_beh, ...
                    'behavior_name', config.behavior_names{beh_idx});
                continue;
            end

            % Compute coherence using spikes from this behavior only
            try
                [coherence, phase, freq, S_spike, S_lfp] = ...
                    calculate_spike_lfp_coherence_multitaper(beh_spike_times, LFP, NeuralTime, Fs, coherence_params);

                beh_result = struct();
                beh_result.behavior_name = config.behavior_names{beh_idx};
                beh_result.n_spikes = n_spikes_beh;
                beh_result.coherence = coherence;
                beh_result.phase = phase;
                beh_result.freq = freq;
                beh_result.S_spike = S_spike;
                beh_result.S_lfp = S_lfp;

                % Band-specific mean coherence
                band_names = {'Delta', 'Theta', 'Beta', 'Low_Gamma', 'High_Gamma', 'Ultra_Gamma'};
                band_ranges = [1, 4; 5, 12; 15, 30; 30, 60; 80, 100; 100, 150];

                for b = 1:size(band_ranges, 1)
                    band_mask = freq >= band_ranges(b, 1) & freq <= band_ranges(b, 2);
                    beh_result.band_mean_coherence.(band_names{b}) = mean(coherence(band_mask));
                end

                beh_result.skipped = false;

            catch ME
                beh_result = struct('skipped', true, 'reason', 'computation_error', ...
                    'error_message', ME.message, 'n_spikes', n_spikes_beh, ...
                    'behavior_name', config.behavior_names{beh_idx});
            end

            behavior_results{beh_idx} = beh_result;
        end

        unit_coherence_results{unit_idx} = struct('unit_id', unit_idx, ...
            'n_spikes_total', n_spikes_total, ...
            'behavior_results', {behavior_results});

        if mod(unit_idx, 10) == 0 || unit_idx == n_units
            fprintf('    Unit %d/%d processed\n', unit_idx, n_units);
        end
    end

    %% Save results
    session_results = struct();
    session_results.session_id = spike_sess_idx;
    session_results.filename = spike_filename;
    session_results.session_type = 'RewardAversive';
    session_results.session_duration_min = (NeuralTime(end) - NeuralTime(1)) / 60;
    session_results.best_channel = bestChannel;
    session_results.Fs = Fs;
    session_results.n_units = n_units;
    session_results.unit_coherence_results = unit_coherence_results;
    session_results.NeuralTime_start = NeuralTime(1);
    session_results.NeuralTime_end = NeuralTime(end);

    [~, base_filename, ~] = fileparts(spike_filename);
    save_filename = fullfile(RewardAversivePath, sprintf('%s_spike_lfp_coherence_behavior_specific.mat', base_filename));
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

    %% Match with behavior prediction
    matched = false;
    BehaviorClass = [];

    for beh_sess_idx = 1:length(sessions_reward)
        beh_session = sessions_reward{beh_sess_idx};

        if contains(spike_filename, extractBefore(beh_session.filename, '.mat'))
            matched = true;

            prediction_scores = prediction_sessions_reward(beh_sess_idx).prediction_scores;
            n_predictions = size(prediction_scores, 1);
            [max_confidence, dominant_beh] = max(prediction_scores, [], 2);

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

    %% Compute behavior-specific coherence
    n_units = length(valid_spikes);
    unit_coherence_results = cell(n_units, 1);

    coherence_params = struct();
    coherence_params.freq_range = config.freq_range;
    coherence_params.tapers = config.tapers;
    coherence_params.pad = config.pad;
    coherence_params.window_size = config.window_size;
    coherence_params.Fs = Fs;

    fprintf('  Computing behavior-specific coherence for %d units...\n', n_units);

    for unit_idx = 1:n_units
        spike_times = valid_spikes{unit_idx};
        n_spikes_total = length(spike_times);

        spike_indices = interp1(NeuralTime, 1:length(NeuralTime), spike_times, 'nearest', 'extrap');
        valid_mask = spike_indices > 0 & spike_indices <= length(BehaviorClass);
        spike_indices = spike_indices(valid_mask);
        spike_behaviors = BehaviorClass(spike_indices);

        behavior_mask = spike_behaviors > 0;
        spike_indices = spike_indices(behavior_mask);
        spike_behaviors = spike_behaviors(behavior_mask);
        spike_times_with_behavior = spike_times(valid_mask);
        spike_times_with_behavior = spike_times_with_behavior(behavior_mask);

        behavior_results = cell(config.n_behaviors, 1);

        for beh_idx = 1:config.n_behaviors
            beh_mask = spike_behaviors == beh_idx;
            beh_spike_times = spike_times_with_behavior(beh_mask);
            n_spikes_beh = length(beh_spike_times);

            if n_spikes_beh < config.min_spikes_per_behavior
                behavior_results{beh_idx} = struct('skipped', true, ...
                    'reason', 'insufficient_spikes', 'n_spikes', n_spikes_beh, ...
                    'behavior_name', config.behavior_names{beh_idx});
                continue;
            end

            try
                [coherence, phase, freq, S_spike, S_lfp] = ...
                    calculate_spike_lfp_coherence_multitaper(beh_spike_times, LFP, NeuralTime, Fs, coherence_params);

                beh_result = struct();
                beh_result.behavior_name = config.behavior_names{beh_idx};
                beh_result.n_spikes = n_spikes_beh;
                beh_result.coherence = coherence;
                beh_result.phase = phase;
                beh_result.freq = freq;
                beh_result.S_spike = S_spike;
                beh_result.S_lfp = S_lfp;

                band_names = {'Delta', 'Theta', 'Beta', 'Low_Gamma', 'High_Gamma', 'Ultra_Gamma'};
                band_ranges = [1, 4; 5, 12; 15, 30; 30, 60; 80, 100; 100, 150];

                for b = 1:size(band_ranges, 1)
                    band_mask = freq >= band_ranges(b, 1) & freq <= band_ranges(b, 2);
                    beh_result.band_mean_coherence.(band_names{b}) = mean(coherence(band_mask));
                end

                beh_result.skipped = false;

            catch ME
                beh_result = struct('skipped', true, 'reason', 'computation_error', ...
                    'error_message', ME.message, 'n_spikes', n_spikes_beh, ...
                    'behavior_name', config.behavior_names{beh_idx});
            end

            behavior_results{beh_idx} = beh_result;
        end

        unit_coherence_results{unit_idx} = struct('unit_id', unit_idx, ...
            'n_spikes_total', n_spikes_total, ...
            'behavior_results', {behavior_results});

        if mod(unit_idx, 10) == 0 || unit_idx == n_units
            fprintf('    Unit %d/%d processed\n', unit_idx, n_units);
        end
    end

    %% Save results
    session_results = struct();
    session_results.session_id = spike_sess_idx;
    session_results.filename = spike_filename;
    session_results.session_type = 'RewardSeeking';
    session_results.session_duration_min = (NeuralTime(end) - NeuralTime(1)) / 60;
    session_results.best_channel = bestChannel;
    session_results.Fs = Fs;
    session_results.n_units = n_units;
    session_results.unit_coherence_results = unit_coherence_results;
    session_results.NeuralTime_start = NeuralTime(1);
    session_results.NeuralTime_end = NeuralTime(end);

    [~, base_filename, ~] = fileparts(spike_filename);
    save_filename = fullfile(RewardSeekingPath, sprintf('%s_spike_lfp_coherence_behavior_specific.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    elapsed = toc;
    fprintf('  Completed in %.1f seconds. Saved to: %s\n', elapsed, save_filename);
end

fprintf('\n✓ Processed %d reward sessions\n', num_reward_sessions);

%% ========================================================================
%  SECTION 7: SUMMARY
%  ========================================================================

fprintf('\n========================================\n');
fprintf('ANALYSIS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Results saved to:\n');
fprintf('  Reward-seeking: %s\n', RewardSeekingPath);
fprintf('  Reward-aversive: %s\n', RewardAversivePath);
fprintf('\nNext step: Run visualization script\n');
fprintf('========================================\n');
