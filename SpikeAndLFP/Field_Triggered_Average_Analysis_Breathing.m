%% ========================================================================
%  FIELD-TRIGGERED AVERAGE ANALYSIS WITH BREATHING SIGNAL
%  Spike rate as a function of breathing phase
%  ========================================================================
%
%  Purpose: Test for phase-locking to breathing by sorting spikes by instantaneous phase
%
%  Method: For each frequency band (0.1-20 Hz):
%  1. Filter breathing signal and extract phase (Hilbert transform)
%  2. For each spike, determine the breathing phase at spike time
%  3. Bin spikes by phase (-π to π)
%  4. Compute FTA: spike rate vs phase
%
%  BREATHING SIGNAL: Uses channel 32 from ephy data
%
%  High phase-locking → FTA has clear peak at preferred phase
%  No phase-locking → FTA is flat across phases
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== FIELD-TRIGGERED AVERAGE ANALYSIS (BREATHING) ===\n');
fprintf('Breathing Signal (Channel 32)\n\n');

config = struct();

% Use SAME frequency bands as PPC analysis (0.1-20 Hz, 1 Hz bins)
freq_start = 0.1:1:19;
freq_end = 1:1:20;
n_bands = length(freq_start);

config.frequency_bands = cell(n_bands, 2);
for i = 1:n_bands
    config.frequency_bands{i, 1} = sprintf('%.1f-%.1fHz', freq_start(i), freq_end(i));
    config.frequency_bands{i, 2} = [freq_start(i), freq_end(i)];
end

% FTA parameters
config.n_phase_bins = 18;  % 18 bins = 20° per bin
config.min_spikes = 50;     % Minimum spikes for reliable FTA

% Breathing signal parameters
config.breathing_channel = 32;       % Channel 32 = breathing signal
config.bp_range = [0.1 300];

% Data paths
config.spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';
config.numofsession = 999;

% Output paths
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_FTA_Breathing');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_FTA_Breathing');
if ~exist(RewardSeekingPath, 'dir'), mkdir(RewardSeekingPath); end
if ~exist(RewardAversivePath, 'dir'), mkdir(RewardAversivePath); end

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/NewScripts/');

fprintf('Configuration:\n');
fprintf('  Breathing channel: %d\n', config.breathing_channel);
fprintf('  Frequency bands: %d (0.1-20 Hz, 1 Hz bins)\n', n_bands);
fprintf('  Phase bins: %d (%.1f° per bin)\n', config.n_phase_bins, 360/config.n_phase_bins);
fprintf('  Min spikes: %d\n\n', config.min_spikes);

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
        fprintf('  Skipping: insufficient events\n');
        continue;
    end

    % Extract BREATHING SIGNAL from channel 32
    breathing_raw = Signal(:, config.breathing_channel);
    [filtered_breathing] = preprocessSignals(breathing_raw, Fs, config.bp_range);
    Breathing = filtered_breathing;

    n_valid_aversive = n_valid_aversive + 1;

    % Period boundaries
    period_boundaries = [TriggerMid(1), all_aversive_time(1:6)' + TriggerMid(1), TriggerMid(end)];
    n_periods = 7;
    n_units = length(valid_spikes);

    fprintf('  Units: %d, Periods: %d, Breathing Channel: %d\n', n_units, n_periods, config.breathing_channel);

    % Compute FTA
    session_results = compute_FTA_session(NeuralTime, Breathing, valid_spikes, period_boundaries, ...
                                          n_periods, n_units, Fs, config);

    session_results.session_id = n_valid_aversive;
    session_results.filename = allfiles_aversive(sess_idx).name;
    session_results.session_type = 'RewardAversive';
    session_results.breathing_channel = config.breathing_channel;

    % Save
    [~, base_filename, ~] = fileparts(allfiles_aversive(sess_idx).name);
    save_filename = fullfile(RewardAversivePath, sprintf('%s_fta_breathing.mat', base_filename));
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

    % Period boundaries
    period_boundaries = [avg_time_boundaries + TriggerMid(1)];
    n_periods = 4;
    n_units = length(valid_spikes);

    fprintf('  Units: %d, Periods: %d, Breathing Channel: %d\n', n_units, n_periods, config.breathing_channel);

    % Compute FTA
    session_results = compute_FTA_session(NeuralTime, Breathing, valid_spikes, period_boundaries, ...
                                          n_periods, n_units, Fs, config);

    session_results.session_id = n_valid_reward;
    session_results.filename = allfiles_reward(sess_idx).name;
    session_results.session_type = 'RewardSeeking';
    session_results.breathing_channel = config.breathing_channel;

    % Save
    [~, base_filename, ~] = fileparts(allfiles_reward(sess_idx).name);
    save_filename = fullfile(RewardSeekingPath, sprintf('%s_fta_breathing.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    fprintf('  ✓ Complete (%.1f sec)\n', toc);
end

fprintf('\n✓ Processed %d reward sessions\n', n_valid_reward);

%% ========================================================================
%  COMPLETION
%  ========================================================================

fprintf('\n========================================\n');
fprintf('FIELD-TRIGGERED AVERAGE ANALYSIS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Summary:\n');
fprintf('  Breathing channel: %d\n', config.breathing_channel);
fprintf('  Aversive sessions: %d\n', n_valid_aversive);
fprintf('  Reward sessions: %d\n', n_valid_reward);
fprintf('\nResults saved to:\n');
fprintf('  %s\n', RewardAversivePath);
fprintf('  %s\n', RewardSeekingPath);
fprintf('\nExpected breathing frequencies:\n');
fprintf('  - Resting: 0.5-2 Hz\n');
fprintf('  - Active: 2-4 Hz\n');
fprintf('  - Sniffing: 4-12 Hz\n');
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function session_results = compute_FTA_session(NeuralTime, Breathing, valid_spikes, ...
                                               period_boundaries, n_periods, n_units, Fs, config)
% Compute field-triggered average: spike rate as function of breathing phase
% OPTIMIZED VERSION: Pre-allocated arrays, vectorized operations

    n_bands = size(config.frequency_bands, 1);
    n_phase_bins = config.n_phase_bins;
    phase_edges = linspace(-pi, pi, n_phase_bins + 1);
    phase_centers = (phase_edges(1:end-1) + phase_edges(2:end)) / 2;

    % Estimate maximum number of results (pre-allocate)
    max_results = n_bands * n_units * n_periods;

    % Pre-allocate arrays (MAJOR SPEEDUP: avoid dynamic growth)
    fta_data = struct();
    fta_data.unit = zeros(max_results, 1);
    fta_data.period = zeros(max_results, 1);
    fta_data.band_name = cell(max_results, 1);
    fta_data.freq_low = zeros(max_results, 1);
    fta_data.freq_high = zeros(max_results, 1);
    fta_data.fta_curve = cell(max_results, 1);  % Cell array: spike rate vs phase
    fta_data.phase_centers = cell(max_results, 1);  % Phase bin centers
    fta_data.mean_vector_length = zeros(max_results, 1);  % Measure of phase-locking
    fta_data.preferred_phase = zeros(max_results, 1);
    fta_data.n_spikes = zeros(max_results, 1);

    result_idx = 0;  % Counter for actual results

    % Time mapping
    t0 = NeuralTime(1);
    dt = median(diff(NeuralTime));

    fprintf('  Computing FTA for %d bands...\n', n_bands);

    % Process each frequency band
    % NOTE: To enable parallel processing, change 'for' to 'parfor' below
    for band_idx = 1:n_bands
        band_name = config.frequency_bands{band_idx, 1};
        band_range = config.frequency_bands{band_idx, 2};

        % Filter breathing signal and extract phase
        Breathing_filtered = filter_breathing_for_phase(Breathing, band_range, Fs);
        phase_signal = angle(hilbert(Breathing_filtered));
        clear Breathing_filtered;  % Free memory immediately

        % Process each unit
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

                % Convert spike times to indices (already optimized)
                spike_indices = round((spikes_in_period - t0) / dt) + 1;
                spike_indices = spike_indices(spike_indices > 0 & spike_indices <= length(phase_signal));

                % Get breathing phase at each spike
                spike_phases = phase_signal(spike_indices);

                % Bin spikes by breathing phase
                spike_counts = histcounts(spike_phases, phase_edges);

                % Normalize to spike rate (spikes per bin / total spikes)
                fta_curve = spike_counts / sum(spike_counts);

                % Compute mean vector length (measure of phase-locking to breathing)
                % OPTIMIZED: Vectorized complex exponential
                mean_vector = mean(exp(1i * spike_phases));
                mvl = abs(mean_vector);
                preferred_phase = angle(mean_vector);

                % Store results (pre-allocated, much faster)
                result_idx = result_idx + 1;
                fta_data.unit(result_idx) = unit_idx;
                fta_data.period(result_idx) = period_idx;
                fta_data.band_name{result_idx} = band_name;
                fta_data.freq_low(result_idx) = band_range(1);
                fta_data.freq_high(result_idx) = band_range(2);
                fta_data.fta_curve{result_idx} = fta_curve;
                fta_data.phase_centers{result_idx} = phase_centers;
                fta_data.mean_vector_length(result_idx) = mvl;
                fta_data.preferred_phase(result_idx) = preferred_phase;
                fta_data.n_spikes(result_idx) = n_spikes;
            end
        end

        if mod(band_idx, 5) == 0
            fprintf('    Processed %d/%d bands\n', band_idx, n_bands);
        end
    end

    % Trim pre-allocated arrays to actual size
    fta_data.unit = fta_data.unit(1:result_idx);
    fta_data.period = fta_data.period(1:result_idx);
    fta_data.band_name = fta_data.band_name(1:result_idx);
    fta_data.freq_low = fta_data.freq_low(1:result_idx);
    fta_data.freq_high = fta_data.freq_high(1:result_idx);
    fta_data.fta_curve = fta_data.fta_curve(1:result_idx);
    fta_data.phase_centers = fta_data.phase_centers(1:result_idx);
    fta_data.mean_vector_length = fta_data.mean_vector_length(1:result_idx);
    fta_data.preferred_phase = fta_data.preferred_phase(1:result_idx);
    fta_data.n_spikes = fta_data.n_spikes(1:result_idx);

    % Convert to table
    session_results.data = struct2table(fta_data);
    session_results.data.Properties.VariableNames{'unit'} = 'Unit';
    session_results.data.Properties.VariableNames{'period'} = 'Period';
    session_results.data.Properties.VariableNames{'band_name'} = 'Band';
    session_results.data.Properties.VariableNames{'freq_low'} = 'Freq_Low_Hz';
    session_results.data.Properties.VariableNames{'freq_high'} = 'Freq_High_Hz';
    session_results.data.Properties.VariableNames{'fta_curve'} = 'FTA_Curve';
    session_results.data.Properties.VariableNames{'phase_centers'} = 'Phase_Centers';
    session_results.data.Properties.VariableNames{'mean_vector_length'} = 'Mean_Vector_Length';
    session_results.data.Properties.VariableNames{'preferred_phase'} = 'Preferred_Phase';
    session_results.data.Properties.VariableNames{'n_spikes'} = 'N_spikes';

    session_results.Fs = Fs;
    session_results.n_phase_bins = n_phase_bins;
end

function Breathing_filtered = filter_breathing_for_phase(Breathing, band_range, Fs)
% Filter breathing signal for phase extraction (same as PPC)
    low_freq = band_range(1);
    high_freq = band_range(2);

    if low_freq < 1
        Breathing_detrend = detrend(Breathing);
        Breathing_demean = Breathing_detrend - mean(Breathing_detrend);
        Breathing_filtered = lowpass(Breathing_demean, high_freq, Fs, 'ImpulseResponse', 'fir', 'Steepness', 0.85);
    else
        Breathing_filtered = bandpass(Breathing, band_range, Fs, 'ImpulseResponse', 'fir', 'Steepness', 0.85);
    end
end
