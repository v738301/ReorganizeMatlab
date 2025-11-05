%% Spike-Phase Coupling Analysis
% Analyzes phase-locking of single units to LFP oscillations across multiple frequency bands
% Processes both reward-seeking and reward-aversive sessions
% Stores full spike times and phases for subsequent time-resolved analyses

clear all;
close all;

%% Configuration Parameters
config = struct();

% Frequency bands to analyze
config.frequency_bands = {
    'Delta',      [1, 4];
    'Theta',      [5, 12];
    'Beta',       [15, 30];
    'Low_Gamma',  [30, 60];
    'High_Gamma', [80, 100];
    'Ultra_Gamma',[100, 150];
};

% LFP filtering parameters
config.bp_range = [1 300];  % Bandpass filter range for raw signal

% Phase histogram parameters
config.n_phase_bins = 18;  % 18 bins = 20 degrees each

% Statistical threshold
config.alpha = 0.05;  % Significance level for Rayleigh test

% Minimum firing rate (use same as breathing analysis)
config.minFR = 0.5;  % Hz

% Dataset paths
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';

% Create save directories
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_SpikePhase');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_SpikePhase');
if ~exist(RewardSeekingPath, 'dir')
    mkdir(RewardSeekingPath);
end
if ~exist(RewardAversivePath, 'dir')
    mkdir(RewardAversivePath);
end

fprintf('\n========================================\n');
fprintf('SPIKE-PHASE COUPLING ANALYSIS\n');
fprintf('========================================\n');
fprintf('Analyzing %d frequency bands:\n', size(config.frequency_bands, 1));
for i = 1:size(config.frequency_bands, 1)
    fprintf('  %d. %s: %.1f-%.1f Hz\n', i, config.frequency_bands{i,1}, ...
        config.frequency_bands{i,2}(1), config.frequency_bands{i,2}(2));
end
fprintf('========================================\n\n');

%% Load Sorting Parameters
[T_sorted] = loadSortingParameters();

%% Process Reward-Seeking Sessions
fprintf('\n==== PROCESSING REWARD-SEEKING SESSIONS ====\n');
numofsession = 2;
folderpath = "/Volumes/ExpansionBackup/Data/Struct_spike";
[allfiles, folderpath, num_sessions] = selectFilesWithAnimalIDFiltering(folderpath, numofsession, '2025*RewardSeeking*.mat');

for sessionID = 1:num_sessions
    fprintf('\n[%d/%d] Processing: %s\n', sessionID, num_sessions, allfiles(sessionID).name);
    tic;

    %% Load session data
    Timelimits = 'No';
    [NeuralTime, AdjustedXYZ, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, WP1ON, WP2ON, ...
        AversiveSound, sessionLabels, valid_spikes, Fs, TriggerMid] = ...
        loadAndPrepareSessionData(allfiles(sessionID), T_sorted, Timelimits);

    %% Preprocess LFP
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);
    bestChannel = findBestLFPChannel(filtered_data, Fs);
    LFP = filtered_data(:, bestChannel);

    fprintf('  LFP channel: %d, Duration: %.1f min, Units: %d\n', ...
        bestChannel, (NeuralTime(end) - NeuralTime(1))/60, length(valid_spikes));

    %% Calculate spike-phase coupling for all units
    session_results = calculate_spike_phase_coupling(valid_spikes, LFP, NeuralTime, Fs, config);

    %% Add session metadata
    session_results.session_id = sessionID;
    session_results.filename = allfiles(sessionID).name;
    session_results.session_type = 'RewardSeeking';
    session_results.session_duration_min = (NeuralTime(end) - NeuralTime(1)) / 60;
    session_results.best_channel = bestChannel;
    session_results.Fs = Fs;
    session_results.NeuralTime_start = NeuralTime(1);
    session_results.NeuralTime_end = NeuralTime(end);

    %% Save results
    [~, base_filename, ~] = fileparts(allfiles(sessionID).name);
    save_filename = fullfile(RewardSeekingPath, sprintf('%s_spike_phase_coupling.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    elapsed = toc;
    fprintf('  Completed in %.1f seconds. Saved to: %s\n', elapsed, save_filename);
end

fprintf('\n==== REWARD-SEEKING SESSIONS COMPLETED ====\n');

%% Process Reward-Aversive Sessions
fprintf('\n==== PROCESSING REWARD-AVERSIVE SESSIONS ====\n');
[allfiles, folderpath, num_sessions] = selectFilesWithAnimalIDFiltering(folderpath, numofsession, '2025*RewardAversive*.mat');

for sessionID = 1:num_sessions
    fprintf('\n[%d/%d] Processing: %s\n', sessionID, num_sessions, allfiles(sessionID).name);
    tic;

    %% Load session data
    Timelimits = 'No';
    [NeuralTime, AdjustedXYZ, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, WP1ON, WP2ON, ...
        AversiveSound, sessionLabels, valid_spikes, Fs, TriggerMid] = ...
        loadAndPrepareSessionData(allfiles(sessionID), T_sorted, Timelimits);

    %% Preprocess LFP
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);
    bestChannel = findBestLFPChannel(filtered_data, Fs);
    LFP = filtered_data(:, bestChannel);

    % Get all aversive sound timepoints (inline detection)
    aversive_onsets = find(diff(AversiveSound) == 1);
    all_aversive_time = NeuralTime(aversive_onsets);

    fprintf('  LFP channel: %d, Duration: %.1f min, Units: %d, Aversive events: %d\n', ...
        bestChannel, (NeuralTime(end) - NeuralTime(1))/60, length(valid_spikes), length(all_aversive_time));

    %% Calculate spike-phase coupling for all units
    session_results = calculate_spike_phase_coupling(valid_spikes, LFP, NeuralTime, Fs, config);

    %% Add session metadata
    session_results.session_id = sessionID;
    session_results.filename = allfiles(sessionID).name;
    session_results.session_type = 'RewardAversive';
    session_results.session_duration_min = (NeuralTime(end) - NeuralTime(1)) / 60;
    session_results.all_aversive_time = all_aversive_time;  % Store all aversive timepoints
    session_results.n_aversive_events = length(all_aversive_time);
    session_results.best_channel = bestChannel;
    session_results.Fs = Fs;
    session_results.NeuralTime_start = NeuralTime(1);
    session_results.NeuralTime_end = NeuralTime(end);

    %% Save results
    [~, base_filename, ~] = fileparts(allfiles(sessionID).name);
    save_filename = fullfile(RewardAversivePath, sprintf('%s_spike_phase_coupling.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    elapsed = toc;
    fprintf('  Completed in %.1f seconds. Saved to: %s\n', elapsed, save_filename);
end

fprintf('\n==== REWARD-AVERSIVE SESSIONS COMPLETED ====\n');

%% Summary
fprintf('\n========================================\n');
fprintf('ANALYSIS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Results saved to:\n');
fprintf('  Reward-seeking: %s\n', RewardSeekingPath);
fprintf('  Reward-aversive: %s\n', RewardAversivePath);
fprintf('\nNext step: Run Visualize_Spike_Phase_Coupling.m\n');
fprintf('========================================\n');
