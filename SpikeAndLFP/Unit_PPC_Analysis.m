%% ========================================================================
%  UNIT PPC ANALYSIS: Period × SessionType × FrequencyBand
%  Pairwise Phase Consistency (PPC) for spike-LFP coupling
%  ========================================================================
%
%  Analysis: PPC ~ Period × SessionType × FrequencyBand
%  SessionType: Aversive vs Reward
%  Aversive Periods: P1-P7 (6 aversive noises create 7 periods)
%  Reward Periods: P1-P4 (time-matched to aversive)
%  Frequency Bands: 1 Hz bins from 0.1 to 20 Hz (20 bands)
%
%  PPC (Pairwise Phase Consistency) advantages:
%  - Unbiased by spike count (unlike MRL)
%  - Comparable across neurons with different firing rates
%  - Robust for small sample sizes
%  - Standard measure in spike-LFP coupling literature
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== UNIT PPC ANALYSIS: AVERSIVE vs REWARD ===\n');
fprintf('Pairwise Phase Consistency for Spike-LFP Coupling\n');
fprintf('1 Hz frequency bins from 0.1 to 20 Hz\n\n');

config = struct();

% Generate frequency bands: 1 Hz bins from 0.1 to 20 Hz
freq_start = 0.1:1:19;
freq_end = 1:1:20;
n_bands = length(freq_start);

config.frequency_bands = cell(n_bands, 2);
for i = 1:n_bands
    config.frequency_bands{i, 1} = sprintf('%.1f-%.1fHz', freq_start(i), freq_end(i));
    config.frequency_bands{i, 2} = [freq_start(i), freq_end(i)];
end

% LFP filtering parameters
config.bp_range = [0.1 300];  % Bandpass filter range for raw signal

% Statistical parameters
config.min_spikes = 10;          % Minimum spikes to calculate PPC
config.bootstrap_samples = 0;    % Set to 0 to skip bootstrap (faster), >0 for CI
config.ci_level = 0.95;          % 95% confidence intervals (if bootstrap enabled)

% Data paths
config.spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';
config.numofsession = 999;  % Max sessions per animal

% Create save directories
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_UnitPPC');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_UnitPPC');
if ~exist(RewardSeekingPath, 'dir')
    mkdir(RewardSeekingPath);
end
if ~exist(RewardAversivePath, 'dir')
    mkdir(RewardAversivePath);
end

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/NewScripts/');

fprintf('Configuration:\n');
fprintf('  Frequency bands: %d (1 Hz bins)\n', n_bands);
fprintf('  Range: %.1f - %.1f Hz\n', freq_start(1), freq_end(end));
fprintf('  Aversive periods: 7 (based on 6 noise events)\n');
fprintf('  Reward periods: 4 (time-matched)\n');
fprintf('  Min spikes for PPC: %d\n', config.min_spikes);
if config.bootstrap_samples > 0
    fprintf('  Bootstrap samples: %d (CI enabled)\n\n', config.bootstrap_samples);
else
    fprintf('  Bootstrap: DISABLED (faster computation)\n\n');
end

%% ========================================================================
%  SECTION 2: LOAD SORTING PARAMETERS
%  ========================================================================

fprintf('Loading sorting parameters...\n');
[T_sorted] = loadSortingParameters();
fprintf('✓ Sorting parameters loaded\n\n');

%% ========================================================================
%  SECTION 3: SELECT SPIKE FILES
%  ========================================================================

fprintf('Selecting spike files...\n');

% Aversive sessions
[allfiles_aversive, folderpath, num_aversive_sessions] = ...
    selectFilesWithAnimalIDFiltering(config.spike_folder, config.numofsession, '2025*RewardAversive*.mat');
fprintf('✓ Found %d aversive sessions\n', num_aversive_sessions);

% Reward sessions
[allfiles_reward, ~, num_reward_sessions] = ...
    selectFilesWithAnimalIDFiltering(config.spike_folder, config.numofsession, '2025*RewardSeeking*.mat');
fprintf('✓ Found %d reward sessions\n\n', num_reward_sessions);

%% ========================================================================
%  SECTION 4: PROCESS AVERSIVE SESSIONS
%  ========================================================================

fprintf('==== PROCESSING AVERSIVE SESSIONS ====\n');

n_valid_aversive = 0;

for sess_idx = 1:num_aversive_sessions
    fprintf('\n[%d/%d] Processing: %s\n', sess_idx, num_aversive_sessions, allfiles_aversive(sess_idx).name);
    tic;

    % Load raw spike data
    Timelimits = 'No';
    [NeuralTime, ~, ~, Signal, ~, ~, ~, ~, AversiveSound, ~, valid_spikes, Fs, TriggerMid] = ...
        loadAndPrepareSessionData(allfiles_aversive(sess_idx), T_sorted, Timelimits);

    % Get all aversive sound timepoints
    aversive_onsets = find(diff(AversiveSound) == 1);
    all_aversive_time = NeuralTime(aversive_onsets);

    if length(all_aversive_time) < 6
        fprintf('  Skipping: insufficient aversive events (%d, need 6)\n', length(all_aversive_time));
        continue;
    end

    % Preprocess LFP
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);
    bestChannel = findBestLFPChannel(filtered_data, Fs);
    LFP = filtered_data(:, bestChannel);

    n_valid_aversive = n_valid_aversive + 1;
    spike_filename = allfiles_aversive(sess_idx).name;

    % Define 7 period boundaries using 6 noises
    period_boundaries = [TriggerMid(1), ...
                         all_aversive_time(1:6)' + TriggerMid(1), ...
                         TriggerMid(end)];

    n_periods = 7;
    n_units = length(valid_spikes);

    fprintf('  LFP channel: %d, Units: %d, Periods: %d, Fs: %.0f Hz\n', bestChannel, n_units, n_periods, Fs);

    % Initialize session results storage
    session_results = struct();
    session_results.session_id = n_valid_aversive;
    session_results.filename = spike_filename;
    session_results.session_type = 'RewardAversive';
    session_results.best_channel = bestChannel;
    session_results.Fs = Fs;
    session_results.n_units = n_units;
    session_results.n_periods = n_periods;
    session_results.period_boundaries = period_boundaries;

    % Initialize data storage for this session
    session_data = struct();
    session_data.unit_id = [];
    session_data.period = [];
    session_data.band_name = {};
    session_data.freq_low = [];
    session_data.freq_high = [];
    session_data.period_duration = [];
    session_data.PPC = [];
    session_data.preferred_phase = [];
    session_data.PPC_CI_lower = [];
    session_data.PPC_CI_upper = [];
    session_data.n_spikes = [];
    session_data.reliability = {};

    % Pre-compute phase signals for all frequency bands
    fprintf('  Computing phase signals for %d frequency bands (0.1-20 Hz)...\n', n_bands);
    phase_signals = cell(n_bands, 1);

    for band_idx = 1:n_bands
        fprintf('    Processed %d/%d bands...\n', band_idx, n_bands);
        band_range = config.frequency_bands{band_idx, 2};

        % Apply unified FIR bandpass filtering
        LFP_filtered = filter_LFP_robust(LFP, band_range, Fs);

        % Hilbert transform to get instantaneous phase
        analytic_signal = hilbert(LFP_filtered);
        phase_signals{band_idx} = angle(analytic_signal);  % Phase in radians [-π, π]

    end
    fprintf('  ✓ All phase signals computed\n');

    % Process each unit
    for unit_idx = 1:n_units
        fprintf('    Processed %d/%d Units...\n', unit_idx, n_units);
        spike_times = valid_spikes{unit_idx};

        if isempty(spike_times)
            continue;
        end

        % Process each period
        for period_idx = 1:n_periods
            period_start = period_boundaries(period_idx);
            period_end = period_boundaries(period_idx + 1);
            period_duration = period_end - period_start;

            % Extract spikes in this period
            spikes_in_period = spike_times(spike_times >= period_start & spike_times < period_end);
            n_spikes = length(spikes_in_period);

            % Skip if too few spikes
            if n_spikes < config.min_spikes
                continue;
            end

            % Convert spike times to indices in NeuralTime
            spike_indices = interp1(NeuralTime, 1:length(NeuralTime), spikes_in_period, 'nearest', 'extrap');
            spike_indices = round(spike_indices);
            spike_indices = spike_indices(spike_indices > 0 & spike_indices <= length(NeuralTime));

            % Process each frequency band
            for band_idx = 1:n_bands
                band_name = config.frequency_bands{band_idx, 1};
                band_range = config.frequency_bands{band_idx, 2};

                % Extract spike phases for this band
                spike_phases = phase_signals{band_idx}(spike_indices);

                % Calculate PPC and statistics
                [PPC, preferred_phase, PPC_CI_lower, PPC_CI_upper] = ...
                    calculate_PPC_with_CI(spike_phases, config.bootstrap_samples, config.ci_level);

                % Determine reliability based on spike count
                reliability = determine_reliability(n_spikes);

                % Store data in session_data
                session_data.unit_id(end+1) = unit_idx;
                session_data.period(end+1) = period_idx;
                session_data.band_name{end+1} = band_name;
                session_data.freq_low(end+1) = band_range(1);
                session_data.freq_high(end+1) = band_range(2);
                session_data.period_duration(end+1) = period_duration;
                session_data.PPC(end+1) = PPC;
                session_data.preferred_phase(end+1) = preferred_phase;
                session_data.PPC_CI_lower(end+1) = PPC_CI_lower;
                session_data.PPC_CI_upper(end+1) = PPC_CI_upper;
                session_data.n_spikes(end+1) = n_spikes;
                session_data.reliability{end+1} = reliability;
            end
        end
    end

    % Convert session data to table and save
    session_results.data = struct2table(session_data);
    session_results.data.Properties.VariableNames{'unit_id'} = 'Unit';
    session_results.data.Properties.VariableNames{'period'} = 'Period';
    session_results.data.Properties.VariableNames{'band_name'} = 'Band';
    session_results.data.Properties.VariableNames{'freq_low'} = 'Freq_Low_Hz';
    session_results.data.Properties.VariableNames{'freq_high'} = 'Freq_High_Hz';
    session_results.data.Properties.VariableNames{'period_duration'} = 'Period_Duration_sec';
    session_results.data.Properties.VariableNames{'preferred_phase'} = 'Preferred_Phase_rad';
    session_results.data.Properties.VariableNames{'n_spikes'} = 'N_spikes';
    session_results.data.Properties.VariableNames{'reliability'} = 'Reliability';

    % Save session results
    [~, base_filename, ~] = fileparts(spike_filename);
    save_filename = fullfile(RewardAversivePath, sprintf('%s_unit_ppc.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    elapsed = toc;
    fprintf('  ✓ Session %d complete (%.1f sec). Saved: %s\n', n_valid_aversive, elapsed, save_filename);
end

fprintf('\n✓ Processed %d aversive sessions\n', n_valid_aversive);
fprintf('  Results saved to: %s\n\n', RewardAversivePath);

%% ========================================================================
%  SECTION 5: PROCESS REWARD SESSIONS
%  ========================================================================

fprintf('==== PROCESSING REWARD SESSIONS ====\n');

n_valid_reward = 0;

avg_time_boundaries = [0, 8*60, 16*60, 24*60, 30*60];
fprintf('  Average time boundaries: [%.1f, %.1f, %.1f] seconds\n', avg_time_boundaries);

% Process reward sessions
for sess_idx = 1:num_reward_sessions
    fprintf('\n[%d/%d] Processing: %s\n', sess_idx, num_reward_sessions, allfiles_reward(sess_idx).name);
    tic;

    % Load raw spike data
    Timelimits = 'No';
    [NeuralTime, ~, ~, Signal, ~, ~, ~, ~, ~, ~, valid_spikes, Fs, TriggerMid] = ...
        loadAndPrepareSessionData(allfiles_reward(sess_idx), T_sorted, Timelimits);

    % Preprocess LFP
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);
    bestChannel = findBestLFPChannel(filtered_data, Fs);
    LFP = filtered_data(:, bestChannel);

    n_valid_reward = n_valid_reward + 1;
    spike_filename = allfiles_reward(sess_idx).name;

    % Define 4 period boundaries using time-matched approach
    period_boundaries = [avg_time_boundaries + TriggerMid(1)];

    n_periods = 4;
    n_units = length(valid_spikes);

    fprintf('  LFP channel: %d, Units: %d, Periods: %d, Fs: %.0f Hz\n', bestChannel, n_units, n_periods, Fs);

    % Initialize session results storage
    session_results = struct();
    session_results.session_id = n_valid_reward;
    session_results.filename = spike_filename;
    session_results.session_type = 'RewardSeeking';
    session_results.best_channel = bestChannel;
    session_results.Fs = Fs;
    session_results.n_units = n_units;
    session_results.n_periods = n_periods;
    session_results.period_boundaries = period_boundaries;

    % Initialize data storage for this session
    session_data = struct();
    session_data.unit_id = [];
    session_data.period = [];
    session_data.band_name = {};
    session_data.freq_low = [];
    session_data.freq_high = [];
    session_data.period_duration = [];
    session_data.PPC = [];
    session_data.preferred_phase = [];
    session_data.PPC_CI_lower = [];
    session_data.PPC_CI_upper = [];
    session_data.n_spikes = [];
    session_data.reliability = {};

    % Pre-compute phase signals for all frequency bands
    fprintf('  Computing phase signals for %d frequency bands (0.1-20 Hz)...\n', n_bands);
    phase_signals = cell(n_bands, 1);

    for band_idx = 1:n_bands
        band_range = config.frequency_bands{band_idx, 2};

        % Apply unified FIR bandpass filtering
        LFP_filtered = filter_LFP_robust(LFP, band_range, Fs);

        % Hilbert transform
        analytic_signal = hilbert(LFP_filtered);
        phase_signals{band_idx} = angle(analytic_signal);

        if mod(band_idx, 30) == 0
            fprintf('    Processed %d/%d bands...\n', band_idx, n_bands);
        end
    end
    fprintf('  ✓ All phase signals computed\n');

    % Process each unit
    for unit_idx = 1:n_units
        spike_times = valid_spikes{unit_idx};

        if isempty(spike_times)
            continue;
        end

        % Process each period
        for period_idx = 1:n_periods
            period_start = period_boundaries(period_idx);
            period_end = period_boundaries(period_idx + 1);
            period_duration = period_end - period_start;

            % Extract spikes in this period
            spikes_in_period = spike_times(spike_times >= period_start & spike_times < period_end);
            n_spikes = length(spikes_in_period);

            % Skip if too few spikes
            if n_spikes < config.min_spikes
                continue;
            end

            % Convert spike times to indices in NeuralTime
            spike_indices = interp1(NeuralTime, 1:length(NeuralTime), spikes_in_period, 'nearest', 'extrap');
            spike_indices = round(spike_indices);
            spike_indices = spike_indices(spike_indices > 0 & spike_indices <= length(NeuralTime));

            % Process each frequency band
            for band_idx = 1:n_bands
                band_name = config.frequency_bands{band_idx, 1};
                band_range = config.frequency_bands{band_idx, 2};

                % Extract spike phases for this band
                spike_phases = phase_signals{band_idx}(spike_indices);

                % Calculate PPC and statistics
                [PPC, preferred_phase, PPC_CI_lower, PPC_CI_upper] = ...
                    calculate_PPC_with_CI(spike_phases, config.bootstrap_samples, config.ci_level);

                % Determine reliability
                reliability = determine_reliability(n_spikes);

                % Store data in session_data
                session_data.unit_id(end+1) = unit_idx;
                session_data.period(end+1) = period_idx;
                session_data.band_name{end+1} = band_name;
                session_data.freq_low(end+1) = band_range(1);
                session_data.freq_high(end+1) = band_range(2);
                session_data.period_duration(end+1) = period_duration;
                session_data.PPC(end+1) = PPC;
                session_data.preferred_phase(end+1) = preferred_phase;
                session_data.PPC_CI_lower(end+1) = PPC_CI_lower;
                session_data.PPC_CI_upper(end+1) = PPC_CI_upper;
                session_data.n_spikes(end+1) = n_spikes;
                session_data.reliability{end+1} = reliability;
            end
        end
    end

    % Convert session data to table and save
    session_results.data = struct2table(session_data);
    session_results.data.Properties.VariableNames{'unit_id'} = 'Unit';
    session_results.data.Properties.VariableNames{'period'} = 'Period';
    session_results.data.Properties.VariableNames{'band_name'} = 'Band';
    session_results.data.Properties.VariableNames{'freq_low'} = 'Freq_Low_Hz';
    session_results.data.Properties.VariableNames{'freq_high'} = 'Freq_High_Hz';
    session_results.data.Properties.VariableNames{'period_duration'} = 'Period_Duration_sec';
    session_results.data.Properties.VariableNames{'preferred_phase'} = 'Preferred_Phase_rad';
    session_results.data.Properties.VariableNames{'n_spikes'} = 'N_spikes';
    session_results.data.Properties.VariableNames{'reliability'} = 'Reliability';

    % Save session results
    [~, base_filename, ~] = fileparts(spike_filename);
    save_filename = fullfile(RewardSeekingPath, sprintf('%s_unit_ppc.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    elapsed = toc;
    fprintf('  ✓ Session %d complete (%.1f sec). Saved: %s\n', n_valid_reward, elapsed, save_filename);
end

fprintf('\n✓ Processed %d reward sessions\n', n_valid_reward);
fprintf('  Results saved to: %s\n\n', RewardSeekingPath);

%% ========================================================================
%  SECTION 6: SUMMARY
%  ========================================================================

fprintf('\n========================================\n');
fprintf('ANALYSIS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Summary:\n');
fprintf('  Aversive sessions: %d (7 periods each)\n', n_valid_aversive);
fprintf('  Reward sessions: %d (4 periods each)\n', n_valid_reward);
fprintf('  Frequency bands: %d (1 Hz bins, 0.1-20 Hz)\n', n_bands);
fprintf('\nResults saved to:\n');
fprintf('  Aversive: %s\n', RewardAversivePath);
fprintf('  Reward: %s\n', RewardSeekingPath);
fprintf('\nEach session file contains:\n');
fprintf('  - session_results.data table with columns:\n');
fprintf('    Unit | Period | Band | Freq_Low_Hz | Freq_High_Hz |\n');
fprintf('    PPC | Preferred_Phase_rad | PPC_CI_lower | PPC_CI_upper |\n');
fprintf('    N_spikes | Reliability | Period_Duration_sec\n');
fprintf('  - session_results.session_id, filename, session_type, etc.\n');
fprintf('  - config (analysis parameters)\n');
fprintf('\nPPC (Pairwise Phase Consistency):\n');
fprintf('  - Range: [0, 1]\n');
fprintf('  - Unbiased by spike count\n');
fprintf('  - Comparable across neurons\n');
if config.bootstrap_samples > 0
    fprintf('  - Bootstrap CI enabled (%d samples)\n', config.bootstrap_samples);
else
    fprintf('  - Bootstrap CI disabled (faster computation)\n');
end
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function LFP_filtered = filter_LFP_robust(LFP, band_range, Fs)
% Unified LFP filtering using FIR filters
%
% INPUTS:
%   LFP        - LFP signal vector
%   band_range - [low_freq, high_freq] in Hz
%   Fs         - Sampling frequency in Hz
%
% OUTPUTS:
%   LFP_filtered - Filtered LFP signal

    low_freq = band_range(1);
    high_freq = band_range(2);

    % Use FIR filter for all frequency bands (unified strategy)
    % FIR filters provide linear phase response and better stability for low frequencies
    if high_freq < Fs/2
        LFP_filtered = bandpass(LFP, [low_freq, high_freq], Fs, ...
            'ImpulseResponse', 'fir', 'Steepness', 0.85);
    else
        % High-pass only if high_freq exceeds Nyquist
        LFP_filtered = highpass(LFP, low_freq, Fs, ...
            'ImpulseResponse', 'fir', 'Steepness', 0.85);
    end
end

function PPC = calculate_PPC(phases)
% Calculate Pairwise Phase Consistency (PPC)
%
% PPC is an unbiased measure of phase-locking strength that is not
% inflated by spike count (unlike MRL).
%
% Formula: PPC = (2 / (N * (N-1))) * Σ_i Σ_{j>i} cos(φ_i - φ_j)
%
% INPUTS:
%   phases - Nx1 vector of phase angles in radians
%
% OUTPUTS:
%   PPC - Pairwise Phase Consistency [0, 1]
%         0 = no phase consistency (uniform)
%         1 = perfect phase consistency (all spikes at same phase)

    N = length(phases);

    if N < 2
        PPC = NaN;
        return;
    end

    % Vectorized calculation of all pairwise phase differences
    phase_diffs = bsxfun(@minus, phases, phases');

    % Sum of cosines over upper triangle (unique pairs)
    cos_diffs = cos(phase_diffs);
    sum_cos = sum(triu(cos_diffs, 1), 'all');

    % PPC formula
    PPC = (2 / (N * (N - 1))) * sum_cos;
end

function [PPC, preferred_phase, PPC_CI_lower, PPC_CI_upper] = calculate_PPC_with_CI(phases, n_bootstrap, ci_level)
% Calculate PPC with bootstrap confidence intervals and preferred phase
%
% INPUTS:
%   phases      - Nx1 vector of phase angles in radians
%   n_bootstrap - Number of bootstrap samples (0 to skip bootstrap)
%   ci_level    - Confidence interval level (e.g., 0.95)
%
% OUTPUTS:
%   PPC           - Pairwise Phase Consistency
%   preferred_phase - Mean phase angle (circular mean)
%   PPC_CI_lower  - Lower bound of CI (NaN if bootstrap disabled)
%   PPC_CI_upper  - Upper bound of CI (NaN if bootstrap disabled)

    N = length(phases);

    if N < 2
        PPC = NaN;
        preferred_phase = NaN;
        PPC_CI_lower = NaN;
        PPC_CI_upper = NaN;
        return;
    end

    % Calculate PPC
    PPC = calculate_PPC(phases);

    % Calculate preferred phase (circular mean)
    mean_x = mean(cos(phases));
    mean_y = mean(sin(phases));
    preferred_phase = atan2(mean_y, mean_x);

    % Bootstrap confidence intervals for PPC (if enabled)
    if n_bootstrap > 0 && N >= 10
        bootstrap_PPC = zeros(n_bootstrap, 1);

        for b = 1:n_bootstrap
            % Resample with replacement
            boot_indices = randi(N, N, 1);
            boot_phases = phases(boot_indices);
            bootstrap_PPC(b) = calculate_PPC(boot_phases);
        end

        % Calculate confidence intervals
        alpha = 1 - ci_level;
        PPC_CI_lower = prctile(bootstrap_PPC, 100 * alpha / 2);
        PPC_CI_upper = prctile(bootstrap_PPC, 100 * (1 - alpha / 2));
    else
        % Bootstrap disabled - fill with NaN
        PPC_CI_lower = NaN;
        PPC_CI_upper = NaN;
    end
end

function reliability = determine_reliability(n_spikes)
% Determine reliability class based on spike count
%
% INPUTS:
%   n_spikes - Number of spikes
%
% OUTPUTS:
%   reliability - String: 'very_low', 'low', 'moderate', 'good', 'excellent'

    if n_spikes < 10
        reliability = 'very_low';
    elseif n_spikes < 50
        reliability = 'low';
    elseif n_spikes < 100
        reliability = 'moderate';
    elseif n_spikes < 500
        reliability = 'good';
    else
        reliability = 'excellent';
    end
end
