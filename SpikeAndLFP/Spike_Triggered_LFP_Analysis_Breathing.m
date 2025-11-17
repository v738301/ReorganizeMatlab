%% ========================================================================
%  SPIKE-TRIGGERED BREATHING SIGNAL AVERAGE ANALYSIS
%  Tests for phase-locking vs amplitude modulation with breathing
%  ========================================================================
%
%  Purpose: Analyze spike-breathing coupling using triggered averages
%
%  Method: Compute spike-triggered breathing signal average
%  - High PPC → sharp, consistent waveform shape
%  - Low PPC → flat or noisy average (no consistent phase)
%
%  BREATHING SIGNAL: Uses channel 32 from ephy data
%
%  Output: Spike-triggered averages using broadband breathing signal
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== SPIKE-TRIGGERED BREATHING SIGNAL AVERAGE ANALYSIS ===\n');
fprintf('Using BREATHING SIGNAL (Channel 32)\n\n');

config = struct();

% Spike-triggered average parameters
config.sta_window = 0.5;  % ±500 ms window around spikes
config.min_spikes = 50;   % Minimum spikes for reliable STA

% Breathing signal parameters
config.breathing_channel = 32;       % Channel 32 = breathing signal
config.bp_range = [0.1 300];         % Broadband breathing signal

% Data paths
config.spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';
config.numofsession = 999;

% Output paths
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_STA_Breathing');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_STA_Breathing');
if ~exist(RewardSeekingPath, 'dir'), mkdir(RewardSeekingPath); end
if ~exist(RewardAversivePath, 'dir'), mkdir(RewardAversivePath); end

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/NewScripts/');

fprintf('Configuration:\n');
fprintf('  Breathing channel: %d\n', config.breathing_channel);
fprintf('  Using BROADBAND BREATHING (0.1-300 Hz)\n');
fprintf('  STA window: ±%.1f sec\n', config.sta_window);
fprintf('  Min spikes for STA: %d\n\n', config.min_spikes);

%% ========================================================================
%  SECTION 2: LOAD SORTING PARAMETERS
%  ========================================================================

fprintf('Loading sorting parameters...\n');
[T_sorted] = loadSortingParameters();
fprintf('✓ Loaded\n\n');

%% ========================================================================
%  SECTION 3: SELECT FILES
%  ========================================================================

fprintf('Selecting spike files...\n');

% Aversive
[allfiles_aversive, ~, num_aversive_sessions] = ...
    selectFilesWithAnimalIDFiltering(config.spike_folder, config.numofsession, '2025*RewardAversive*.mat');
fprintf('✓ Aversive: %d sessions\n', num_aversive_sessions);

% Reward
[allfiles_reward, ~, num_reward_sessions] = ...
    selectFilesWithAnimalIDFiltering(config.spike_folder, config.numofsession, '2025*RewardSeeking*.mat');
fprintf('✓ Reward: %d sessions\n\n', num_reward_sessions);

%% ========================================================================
%  SECTION 4: PROCESS AVERSIVE SESSIONS
%  ========================================================================

fprintf('==== PROCESSING AVERSIVE SESSIONS ====\n');

n_valid_aversive = 0;

for sess_idx = 1:num_aversive_sessions
    fprintf('\n[%d/%d] %s\n', sess_idx, num_aversive_sessions, allfiles_aversive(sess_idx).name);
    tic;

    % Load data
    Timelimits = 'No';
    [NeuralTime, ~, ~, Signal, ~, ~, ~, ~, AversiveSound, ~, valid_spikes, Fs, TriggerMid] = ...
        loadAndPrepareSessionData(allfiles_aversive(sess_idx), T_sorted, Timelimits);

    aversive_onsets = find(diff(AversiveSound) == 1);
    all_aversive_time = NeuralTime(aversive_onsets);

    if length(all_aversive_time) < 6
        fprintf('  Skipping: insufficient aversive events\n');
        continue;
    end

    % Extract BREATHING SIGNAL from channel 32
    breathing_raw = Signal(:, config.breathing_channel);
    [filtered_breathing] = preprocessSignals(breathing_raw, Fs, config.bp_range);
    Breathing = filtered_breathing;

    n_valid_aversive = n_valid_aversive + 1;

    % Period boundaries (7 periods)
    period_boundaries = [TriggerMid(1), all_aversive_time(1:6)' + TriggerMid(1), TriggerMid(end)];
    n_periods = 7;
    n_units = length(valid_spikes);

    fprintf('  Units: %d, Periods: %d, Breathing Channel: %d\n', n_units, n_periods, config.breathing_channel);

    % Compute spike-triggered averages
    session_results = compute_STA_session(NeuralTime, Breathing, valid_spikes, period_boundaries, ...
                                          n_periods, n_units, Fs, config);

    session_results.session_id = n_valid_aversive;
    session_results.filename = allfiles_aversive(sess_idx).name;
    session_results.session_type = 'RewardAversive';
    session_results.breathing_channel = config.breathing_channel;

    % Save
    [~, base_filename, ~] = fileparts(allfiles_aversive(sess_idx).name);
    save_filename = fullfile(RewardAversivePath, sprintf('%s_sta_breathing.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    fprintf('  ✓ Complete (%.1f sec)\n', toc);
end

fprintf('\n✓ Processed %d aversive sessions\n', n_valid_aversive);

%% ========================================================================
%  SECTION 5: PROCESS REWARD SESSIONS
%  ========================================================================

fprintf('\n==== PROCESSING REWARD SESSIONS ====\n');

n_valid_reward = 0;

avg_time_boundaries = [0, 8*60, 16*60, 24*60, 30*60];
fprintf('  Average time boundaries: [%.1f, %.1f, %.1f] seconds\n', avg_time_boundaries);

for sess_idx = 1:num_reward_sessions
    fprintf('\n[%d/%d] %s\n', sess_idx, num_reward_sessions, allfiles_reward(sess_idx).name);
    tic;

    % Load data
    Timelimits = 'No';
    [NeuralTime, ~, ~, Signal, ~, ~, ~, ~, ~, ~, valid_spikes, Fs, TriggerMid] = ...
        loadAndPrepareSessionData(allfiles_reward(sess_idx), T_sorted, Timelimits);

    % Extract BREATHING SIGNAL from channel 32
    breathing_raw = Signal(:, config.breathing_channel);
    [filtered_breathing] = preprocessSignals(breathing_raw, Fs, config.bp_range);
    Breathing = filtered_breathing;

    n_valid_reward = n_valid_reward + 1;

    % Period boundaries (4 periods)
    period_boundaries = [avg_time_boundaries + TriggerMid(1)];
    n_periods = 4;
    n_units = length(valid_spikes);

    fprintf('  Units: %d, Periods: %d, Breathing Channel: %d\n', n_units, n_periods, config.breathing_channel);

    % Compute spike-triggered averages
    session_results = compute_STA_session(NeuralTime, Breathing, valid_spikes, period_boundaries, ...
                                          n_periods, n_units, Fs, config);

    session_results.session_id = n_valid_reward;
    session_results.filename = allfiles_reward(sess_idx).name;
    session_results.session_type = 'RewardSeeking';
    session_results.breathing_channel = config.breathing_channel;

    % Save
    [~, base_filename, ~] = fileparts(allfiles_reward(sess_idx).name);
    save_filename = fullfile(RewardSeekingPath, sprintf('%s_sta_breathing.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    fprintf('  ✓ Complete (%.1f sec)\n', toc);
end

fprintf('\n✓ Processed %d reward sessions\n', n_valid_reward);

%% ========================================================================
%  COMPLETION
%  ========================================================================

fprintf('\n========================================\n');
fprintf('SPIKE-TRIGGERED BREATHING SIGNAL ANALYSIS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Summary:\n');
fprintf('  Breathing channel: %d\n', config.breathing_channel);
fprintf('  Aversive sessions: %d\n', n_valid_aversive);
fprintf('  Reward sessions: %d\n', n_valid_reward);
fprintf('\nResults saved to:\n');
fprintf('  %s\n', RewardAversivePath);
fprintf('  %s\n', RewardSeekingPath);
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function session_results = compute_STA_session(NeuralTime, Breathing, valid_spikes, ...
                                               period_boundaries, n_periods, n_units, Fs, config)
% Compute spike-triggered averages using BROADBAND BREATHING SIGNAL

    window_samples = round(config.sta_window * Fs);
    time_vec = (-window_samples:window_samples) / Fs;

    % Pre-allocate results
    sta_data = struct();
    sta_data.unit = [];
    sta_data.period = [];
    sta_data.sta_waveform = {};  % Cell array of waveforms
    sta_data.sta_peak_amplitude = [];
    sta_data.sta_consistency = [];  % Measure of waveform consistency
    sta_data.n_spikes = [];

    % Time mapping
    t0 = NeuralTime(1);
    dt = median(diff(NeuralTime));

    fprintf('  Computing STAs using broadband breathing signal...\n');

    % Process each unit (using BROADBAND BREATHING - no frequency filtering)
    for unit_idx = 1:n_units
        spike_times = valid_spikes{unit_idx};
        if isempty(spike_times), continue; end

        % Process each period
        for period_idx = 1:n_periods
            period_start = period_boundaries(period_idx);
            period_end = period_boundaries(period_idx + 1);

            % Get spikes in period
            spikes_in_period = spike_times(spike_times >= period_start & spike_times < period_end);
            n_spikes = length(spikes_in_period);

            if n_spikes < config.min_spikes
                continue;
            end

            % Convert to indices
            spike_indices = round((spikes_in_period - t0) / dt) + 1;
            spike_indices = spike_indices(spike_indices > window_samples & ...
                                          spike_indices <= length(Breathing) - window_samples);

            % Compute STA from BROADBAND BREATHING SIGNAL
            sta_matrix = zeros(length(spike_indices), 2*window_samples + 1);
            for s = 1:length(spike_indices)
                idx = spike_indices(s);
                sta_matrix(s, :) = Breathing(idx - window_samples : idx + window_samples);
            end

            sta_waveform = mean(sta_matrix, 1);
            sta_peak = max(abs(sta_waveform));
            sta_consistency = std(std(sta_matrix, 0, 2)) / (std(sta_waveform) + eps);  % Lower = more consistent

            % Store
            sta_data.unit(end+1) = unit_idx;
            sta_data.period(end+1) = period_idx;
            sta_data.sta_waveform{end+1} = sta_waveform;
            sta_data.sta_peak_amplitude(end+1) = sta_peak;
            sta_data.sta_consistency(end+1) = sta_consistency;
            sta_data.n_spikes(end+1) = length(spike_indices);
        end

        if mod(unit_idx, 10) == 0
            fprintf('    Processed %d/%d units\n', unit_idx, n_units);
        end
    end

    % Convert to table
    session_results.data = struct2table(sta_data);
    session_results.data.Properties.VariableNames{'unit'} = 'Unit';
    session_results.data.Properties.VariableNames{'period'} = 'Period';
    session_results.data.Properties.VariableNames{'sta_waveform'} = 'STA_Waveform';
    session_results.data.Properties.VariableNames{'sta_peak_amplitude'} = 'STA_Peak';
    session_results.data.Properties.VariableNames{'sta_consistency'} = 'STA_Consistency';
    session_results.data.Properties.VariableNames{'n_spikes'} = 'N_spikes';

    session_results.time_vec = time_vec;
    session_results.Fs = Fs;
end
