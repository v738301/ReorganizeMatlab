%% ========================================================================
%  SPIKE-LFP COHERENCE ANALYSIS (OVERALL)
%  Computes coherence between spikes and LFP for all units using all spikes
%  Uses multitaper method for robust spectral estimation
%  ========================================================================
%
%  This script:
%  1. Loads spike and LFP data for all sessions
%  2. For each unit, computes spike-LFP coherence across frequencies (1-150 Hz)
%  3. Uses multitaper spectral estimation (TW=3, K=5 tapers)
%  4. Saves results with coherence spectra for each unit
%
%% ========================================================================

clear all;
% close all;

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== SPIKE-LFP COHERENCE ANALYSIS (OVERALL) ===\n\n');

config = struct();

% Multitaper parameters (Chronux convention)
config.tapers = [3, 5];  % [TW, K] - time-bandwidth product and number of tapers
config.freq_range = [1, 150];  % Hz - frequency range to analyze
config.pad = 0;  % FFT padding: -1=none, 0=next power of 2, 1=2x next power of 2
config.window_size = 10;  % Window size in seconds for segmented coherence (to avoid memory issues)

% LFP filtering parameters
config.bp_range = [1, 300];  % Bandpass filter range for raw LFP

% Minimum spike count threshold (optional - set to 0 to include all units)
config.min_spikes = 50;  % Require at least 50 spikes for coherence calculation

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');
addpath(genpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/chronux_2_12/chronux_2_12/'));

fprintf('Configuration:\n');
fprintf('  Frequency range: %.1f-%.1f Hz\n', config.freq_range(1), config.freq_range(2));
fprintf('  Tapers: TW=%d, K=%d\n', config.tapers(1), config.tapers(2));
fprintf('  Window size: %d sec (windowed coherence to save memory)\n', config.window_size);
fprintf('  Minimum spikes: %d\n', config.min_spikes);
fprintf('  LFP bandpass: %.1f-%.1f Hz\n\n', config.bp_range(1), config.bp_range(2));

%% ========================================================================
%  SECTION 2: LOAD SORTING PARAMETERS
%  ========================================================================

fprintf('Loading sorting parameters...\n');
[T_sorted] = loadSortingParameters();
fprintf('✓ Sorting parameters loaded\n\n');

%% ========================================================================
%  SECTION 3: CREATE SAVE DIRECTORIES
%  ========================================================================

DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_SpikeLFPCoherence_Overall');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_SpikeLFPCoherence_Overall');

if ~exist(RewardSeekingPath, 'dir')
    mkdir(RewardSeekingPath);
end
if ~exist(RewardAversivePath, 'dir')
    mkdir(RewardAversivePath);
end

%% ========================================================================
%  SECTION 4: PROCESS AVERSIVE SESSIONS
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

    %% Compute coherence for each unit
    n_units = length(valid_spikes);
    unit_coherence_results = cell(n_units, 1);

    % Set up params for coherence calculation
    coherence_params = struct();
    coherence_params.freq_range = config.freq_range;
    coherence_params.tapers = config.tapers;
    coherence_params.pad = config.pad;
    coherence_params.window_size = config.window_size;
    coherence_params.Fs = Fs;

    fprintf('  Computing coherence for %d units...\n', n_units);

    for unit_idx = 1:n_units
        spike_times = valid_spikes{unit_idx};
        n_spikes = length(spike_times);

        if n_spikes < config.min_spikes
            if mod(unit_idx, 20) == 0
                fprintf('    Unit %d/%d: %d spikes - skipping (< %d)\n', ...
                    unit_idx, n_units, n_spikes, config.min_spikes);
            end
            unit_coherence_results{unit_idx} = struct('skipped', true, 'reason', 'insufficient_spikes', 'n_spikes', n_spikes);
            continue;
        end

        % Compute coherence
        try
            [coherence, phase, freq, S_spike, S_lfp] = ...
                calculate_spike_lfp_coherence_multitaper(spike_times, LFP, NeuralTime, Fs, coherence_params);

            % Store results
            unit_result = struct();
            unit_result.unit_id = unit_idx;
            unit_result.n_spikes = n_spikes;
            unit_result.coherence = coherence;
            unit_result.phase = phase;
            unit_result.freq = freq;
            unit_result.S_spike = S_spike;
            unit_result.S_lfp = S_lfp;

            % Calculate mean coherence in each frequency band
            band_names = {'Delta', 'Theta', 'Beta', 'Low_Gamma', 'High_Gamma', 'Ultra_Gamma'};
            band_ranges = [1, 4; 5, 12; 15, 30; 30, 60; 80, 100; 100, 150];

            for b = 1:size(band_ranges, 1)
                band_mask = freq >= band_ranges(b, 1) & freq <= band_ranges(b, 2);
                unit_result.band_mean_coherence.(band_names{b}) = mean(coherence(band_mask));
            end

            unit_result.skipped = false;

            if mod(unit_idx, 20) == 0 || unit_idx == n_units
                fprintf('    Unit %d/%d: %d spikes, mean coherence = %.3f\n', ...
                    unit_idx, n_units, n_spikes, mean(coherence));
            end

        catch ME
            fprintf('    Unit %d: ERROR - %s\n', unit_idx, ME.message);
            unit_result = struct('skipped', true, 'reason', 'computation_error', ...
                'error_message', ME.message, 'n_spikes', n_spikes);
        end

        unit_coherence_results{unit_idx} = unit_result;
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
    save_filename = fullfile(RewardAversivePath, sprintf('%s_spike_lfp_coherence_overall.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    elapsed = toc;
    fprintf('  Completed in %.1f seconds. Saved to: %s\n', elapsed, save_filename);
end

fprintf('\n✓ Processed %d aversive sessions\n', num_aversive_sessions);

%% ========================================================================
%  SECTION 5: PROCESS REWARD SESSIONS
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

    %% Compute coherence for each unit
    n_units = length(valid_spikes);
    unit_coherence_results = cell(n_units, 1);

    coherence_params = struct();
    coherence_params.freq_range = config.freq_range;
    coherence_params.tapers = config.tapers;
    coherence_params.pad = config.pad;
    coherence_params.window_size = config.window_size;
    coherence_params.Fs = Fs;

    fprintf('  Computing coherence for %d units...\n', n_units);

    for unit_idx = 1:n_units
        spike_times = valid_spikes{unit_idx};
        n_spikes = length(spike_times);

        if n_spikes < config.min_spikes
            if mod(unit_idx, 20) == 0
                fprintf('    Unit %d/%d: %d spikes - skipping (< %d)\n', ...
                    unit_idx, n_units, n_spikes, config.min_spikes);
            end
            unit_coherence_results{unit_idx} = struct('skipped', true, 'reason', 'insufficient_spikes', 'n_spikes', n_spikes);
            continue;
        end

        try
            [coherence, phase, freq, S_spike, S_lfp] = ...
                calculate_spike_lfp_coherence_multitaper(spike_times, LFP, NeuralTime, Fs, coherence_params);

            unit_result = struct();
            unit_result.unit_id = unit_idx;
            unit_result.n_spikes = n_spikes;
            unit_result.coherence = coherence;
            unit_result.phase = phase;
            unit_result.freq = freq;
            unit_result.S_spike = S_spike;
            unit_result.S_lfp = S_lfp;

            % Band-specific mean coherence
            band_names = {'Delta', 'Theta', 'Beta', 'Low_Gamma', 'High_Gamma', 'Ultra_Gamma'};
            band_ranges = [1, 4; 5, 12; 15, 30; 30, 60; 80, 100; 100, 150];

            for b = 1:size(band_ranges, 1)
                band_mask = freq >= band_ranges(b, 1) & freq <= band_ranges(b, 2);
                unit_result.band_mean_coherence.(band_names{b}) = mean(coherence(band_mask));
            end

            unit_result.skipped = false;

            if mod(unit_idx, 20) == 0 || unit_idx == n_units
                fprintf('    Unit %d/%d: %d spikes, mean coherence = %.3f\n', ...
                    unit_idx, n_units, n_spikes, mean(coherence));
            end

        catch ME
            fprintf('    Unit %d: ERROR - %s\n', unit_idx, ME.message);
            unit_result = struct('skipped', true, 'reason', 'computation_error', ...
                'error_message', ME.message, 'n_spikes', n_spikes);
        end

        unit_coherence_results{unit_idx} = unit_result;
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
    save_filename = fullfile(RewardSeekingPath, sprintf('%s_spike_lfp_coherence_overall.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    elapsed = toc;
    fprintf('  Completed in %.1f seconds. Saved to: %s\n', elapsed, save_filename);
end

fprintf('\n✓ Processed %d reward sessions\n', num_reward_sessions);

%% ========================================================================
%  SECTION 6: SUMMARY
%  ========================================================================

fprintf('\n========================================\n');
fprintf('ANALYSIS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Results saved to:\n');
fprintf('  Reward-seeking: %s\n', RewardSeekingPath);
fprintf('  Reward-aversive: %s\n', RewardAversivePath);
fprintf('\nNext step: Run visualization script\n');
fprintf('========================================\n');
