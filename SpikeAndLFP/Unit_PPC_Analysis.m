%% ========================================================================
%  UNIT PPC ANALYSIS: Period × SessionType × FrequencyBand
%  Pairwise Phase Consistency (PPC) for spike-LFP coupling
%  ========================================================================
%
%  Analysis: PPC ~ Period × SessionType × FrequencyBand
%  SessionType: Aversive vs Reward
%  Aversive Periods: P1-P7 (6 aversive noises create 7 periods)
%  Reward Periods: P1-P4 (time-matched to aversive)
%  Frequency Bands: Delta, Theta, Beta, Low_Gamma, High_Gamma, Ultra_Gamma
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
fprintf('Pairwise Phase Consistency for Spike-LFP Coupling\n\n');

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

% Statistical parameters
config.min_spikes = 10;          % Minimum spikes to calculate PPC
config.bootstrap_samples = 500;  % Bootstrap iterations for CI
config.ci_level = 0.95;          % 95% confidence intervals

% Data paths
config.spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';
config.numofsession = 999;  % Max sessions per animal

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/NewScripts/');

fprintf('Configuration:\n');
fprintf('  Frequency bands: %d\n', size(config.frequency_bands, 1));
for i = 1:size(config.frequency_bands, 1)
    fprintf('    %d. %s: %.1f-%.1f Hz\n', i, config.frequency_bands{i,1}, ...
        config.frequency_bands{i,2}(1), config.frequency_bands{i,2}(2));
end
fprintf('  Aversive periods: 7 (based on 6 noise events)\n');
fprintf('  Reward periods: 4 (time-matched)\n');
fprintf('  Min spikes for PPC: %d\n', config.min_spikes);
fprintf('  Bootstrap samples: %d\n\n', config.bootstrap_samples);

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

% Initialize storage
aversive_data = struct();
aversive_data.session_id = [];
aversive_data.unit_id = [];
aversive_data.period = [];
aversive_data.band_name = {};
aversive_data.session_name = {};
aversive_data.period_duration = [];

% PPC metrics
aversive_data.PPC = [];
aversive_data.preferred_phase = [];
aversive_data.PPC_CI_lower = [];
aversive_data.PPC_CI_upper = [];
aversive_data.n_spikes = [];
aversive_data.reliability = {};

n_valid_aversive = 0;
n_bands = size(config.frequency_bands, 1);

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

    fprintf('  LFP channel: %d, Units: %d, Periods: %d\n', bestChannel, n_units, n_periods);

    % Pre-compute phase signals for all frequency bands
    fprintf('  Computing phase signals for %d frequency bands...\n', n_bands);
    phase_signals = cell(n_bands, 1);

    for band_idx = 1:n_bands
        band_range = config.frequency_bands{band_idx, 2};
        LFP_filtered = bandpass(LFP, band_range, Fs, 'ImpulseResponse', 'iir', 'Steepness', 0.95);
        analytic_signal = hilbert(LFP_filtered);
        phase_signals{band_idx} = angle(analytic_signal);  % Phase in radians [-π, π]
    end

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

                % Extract spike phases for this band
                spike_phases = phase_signals{band_idx}(spike_indices);

                % Calculate PPC and statistics
                [PPC, preferred_phase, PPC_CI_lower, PPC_CI_upper] = ...
                    calculate_PPC_with_CI(spike_phases, config.bootstrap_samples, config.ci_level);

                % Determine reliability based on spike count
                reliability = determine_reliability(n_spikes);

                % Store data
                aversive_data.session_id(end+1) = n_valid_aversive;
                aversive_data.unit_id(end+1) = (n_valid_aversive - 1) * 1000 + unit_idx;
                aversive_data.period(end+1) = period_idx;
                aversive_data.band_name{end+1} = band_name;
                aversive_data.session_name{end+1} = spike_filename;
                aversive_data.period_duration(end+1) = period_duration;
                aversive_data.PPC(end+1) = PPC;
                aversive_data.preferred_phase(end+1) = preferred_phase;
                aversive_data.PPC_CI_lower(end+1) = PPC_CI_lower;
                aversive_data.PPC_CI_upper(end+1) = PPC_CI_upper;
                aversive_data.n_spikes(end+1) = n_spikes;
                aversive_data.reliability{end+1} = reliability;
            end
        end
    end

    elapsed = toc;
    fprintf('  ✓ Session %d complete (%.1f sec)\n', n_valid_aversive, elapsed);
end

fprintf('\n✓ Processed %d aversive sessions\n', n_valid_aversive);
fprintf('  Total data points: %d\n\n', length(aversive_data.session_id));

%% ========================================================================
%  SECTION 5: PROCESS REWARD SESSIONS
%  ========================================================================

fprintf('==== PROCESSING REWARD SESSIONS ====\n');

% Initialize storage
reward_data = struct();
reward_data.session_id = [];
reward_data.unit_id = [];
reward_data.period = [];
reward_data.band_name = {};
reward_data.session_name = {};
reward_data.period_duration = [];
reward_data.PPC = [];
reward_data.preferred_phase = [];
reward_data.PPC_CI_lower = [];
reward_data.PPC_CI_upper = [];
reward_data.n_spikes = [];
reward_data.reliability = {};

n_valid_reward = 0;

% Need to load aversive sessions to get time-matched periods
fprintf('Loading aversive sessions for time-matching...\n');

aversive_time_boundaries = {};
for sess_idx = 1:num_aversive_sessions
    Timelimits = 'No';
    [NeuralTime, ~, ~, ~, ~, ~, ~, ~, AversiveSound, ~, ~, ~, TriggerMid] = ...
        loadAndPrepareSessionData(allfiles_aversive(sess_idx), T_sorted, Timelimits);

    aversive_onsets = find(diff(AversiveSound) == 1);
    all_aversive_time = NeuralTime(aversive_onsets);

    if length(all_aversive_time) >= 3
        aversive_time_boundaries{sess_idx} = all_aversive_time(1:3)' - TriggerMid(1);
    end
end

% Calculate average time boundaries
all_boundaries = [];
for i = 1:length(aversive_time_boundaries)
    if ~isempty(aversive_time_boundaries{i})
        all_boundaries = [all_boundaries; aversive_time_boundaries{i}];
    end
end
avg_time_boundaries = mean(all_boundaries, 1);
fprintf('  Average time boundaries: [%.1f, %.1f, %.1f] sec\n\n', avg_time_boundaries);

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
    period_boundaries = [TriggerMid(1), ...
                         avg_time_boundaries + TriggerMid(1), ...
                         TriggerMid(end)];

    n_periods = 4;
    n_units = length(valid_spikes);

    fprintf('  LFP channel: %d, Units: %d, Periods: %d\n', bestChannel, n_units, n_periods);

    % Pre-compute phase signals for all frequency bands
    fprintf('  Computing phase signals for %d frequency bands...\n', n_bands);
    phase_signals = cell(n_bands, 1);

    for band_idx = 1:n_bands
        band_range = config.frequency_bands{band_idx, 2};
        LFP_filtered = bandpass(LFP, band_range, Fs, 'ImpulseResponse', 'iir', 'Steepness', 0.95);
        analytic_signal = hilbert(LFP_filtered);
        phase_signals{band_idx} = angle(analytic_signal);
    end

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

                % Extract spike phases for this band
                spike_phases = phase_signals{band_idx}(spike_indices);

                % Calculate PPC and statistics
                [PPC, preferred_phase, PPC_CI_lower, PPC_CI_upper] = ...
                    calculate_PPC_with_CI(spike_phases, config.bootstrap_samples, config.ci_level);

                % Determine reliability
                reliability = determine_reliability(n_spikes);

                % Store data
                reward_data.session_id(end+1) = n_valid_reward;
                reward_data.unit_id(end+1) = (n_valid_reward + 10000) * 1000 + unit_idx;
                reward_data.period(end+1) = period_idx;
                reward_data.band_name{end+1} = band_name;
                reward_data.session_name{end+1} = spike_filename;
                reward_data.period_duration(end+1) = period_duration;
                reward_data.PPC(end+1) = PPC;
                reward_data.preferred_phase(end+1) = preferred_phase;
                reward_data.PPC_CI_lower(end+1) = PPC_CI_lower;
                reward_data.PPC_CI_upper(end+1) = PPC_CI_upper;
                reward_data.n_spikes(end+1) = n_spikes;
                reward_data.reliability{end+1} = reliability;
            end
        end
    end

    elapsed = toc;
    fprintf('  ✓ Session %d complete (%.1f sec)\n', n_valid_reward, elapsed);
end

fprintf('\n✓ Processed %d reward sessions\n', n_valid_reward);
fprintf('  Total data points: %d\n\n', length(reward_data.session_id));

%% ========================================================================
%  SECTION 6: COMBINE AND SAVE RESULTS
%  ========================================================================

fprintf('Combining data...\n');

% Add SessionType
aversive_data.session_type = repmat({'Aversive'}, length(aversive_data.session_id), 1);
reward_data.session_type = repmat({'Reward'}, length(reward_data.session_id), 1);

% Combine all data
combined_data = struct();
combined_data.session_id = [aversive_data.session_id(:); reward_data.session_id(:)];
combined_data.unit_id = [aversive_data.unit_id(:); reward_data.unit_id(:)];
combined_data.period = [aversive_data.period(:); reward_data.period(:)];
combined_data.band_name = [aversive_data.band_name(:); reward_data.band_name(:)];
combined_data.session_type = [aversive_data.session_type; reward_data.session_type];
combined_data.period_duration = [aversive_data.period_duration(:); reward_data.period_duration(:)];
combined_data.PPC = [aversive_data.PPC(:); reward_data.PPC(:)];
combined_data.preferred_phase = [aversive_data.preferred_phase(:); reward_data.preferred_phase(:)];
combined_data.PPC_CI_lower = [aversive_data.PPC_CI_lower(:); reward_data.PPC_CI_lower(:)];
combined_data.PPC_CI_upper = [aversive_data.PPC_CI_upper(:); reward_data.PPC_CI_upper(:)];
combined_data.n_spikes = [aversive_data.n_spikes(:); reward_data.n_spikes(:)];
combined_data.reliability = [aversive_data.reliability(:); reward_data.reliability(:)];

% Convert to table
tbl_data = struct2table(combined_data);

% Rename columns
tbl_data.Properties.VariableNames{'session_id'} = 'Session';
tbl_data.Properties.VariableNames{'unit_id'} = 'Unit';
tbl_data.Properties.VariableNames{'period'} = 'Period';
tbl_data.Properties.VariableNames{'band_name'} = 'Band';
tbl_data.Properties.VariableNames{'session_type'} = 'SessionType';
tbl_data.Properties.VariableNames{'period_duration'} = 'Period_Duration_sec';
tbl_data.Properties.VariableNames{'preferred_phase'} = 'Preferred_Phase_rad';
tbl_data.Properties.VariableNames{'n_spikes'} = 'N_spikes';
tbl_data.Properties.VariableNames{'reliability'} = 'Reliability';

% Convert to categorical
tbl_data.Session = categorical(tbl_data.Session);
tbl_data.Unit = categorical(tbl_data.Unit);
tbl_data.Period = categorical(tbl_data.Period);
tbl_data.Band = categorical(tbl_data.Band);
tbl_data.SessionType = categorical(tbl_data.SessionType);
tbl_data.Reliability = categorical(tbl_data.Reliability);

fprintf('✓ Combined dataset created\n');
fprintf('  Total data points: %d\n', height(tbl_data));
fprintf('  Sessions: %d aversive, %d reward\n', n_valid_aversive, n_valid_reward);
fprintf('  Frequency bands: %d\n\n', n_bands);

%% Save results
fprintf('Saving results...\n');

results = struct();
results.tbl_data = tbl_data;
results.config = config;
results.n_aversive_sessions = n_valid_aversive;
results.n_reward_sessions = n_valid_reward;
results.avg_time_boundaries = avg_time_boundaries;

timestamp = datestr(now, 'dd-mmm-yyyy');
save_filename = sprintf('Unit_PPC_analysis_%s.mat', timestamp);
save(save_filename, 'results', '-v7.3');

fprintf('✓ Results saved to: %s\n', save_filename);

%% ========================================================================
%  SUMMARY
%  ========================================================================

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Summary:\n');
fprintf('  Aversive: %d sessions, 7 periods each\n', n_valid_aversive);
fprintf('  Reward: %d sessions, 4 periods each\n', n_valid_reward);
fprintf('  Total units analyzed: %d\n', length(unique(tbl_data.Unit)));
fprintf('  Frequency bands: %d\n', n_bands);
fprintf('  Total data points: %d\n', height(tbl_data));
fprintf('\nData structure:\n');
fprintf('  Unit | Session | SessionType | Period | Band | PPC | \n');
fprintf('  Preferred_Phase_rad | PPC_CI_lower | PPC_CI_upper | N_spikes | Reliability\n');
fprintf('\nPPC (Pairwise Phase Consistency):\n');
fprintf('  - Range: [0, 1]\n');
fprintf('  - Unbiased by spike count\n');
fprintf('  - Comparable across neurons\n');
fprintf('  - Bootstrap CI (500 samples, 95%% level)\n');
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

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
%   n_bootstrap - Number of bootstrap samples
%   ci_level    - Confidence interval level (e.g., 0.95)
%
% OUTPUTS:
%   PPC           - Pairwise Phase Consistency
%   preferred_phase - Mean phase angle (circular mean)
%   PPC_CI_lower  - Lower bound of CI
%   PPC_CI_upper  - Upper bound of CI

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

    % Bootstrap confidence intervals for PPC
    if N >= 10 && n_bootstrap > 0
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
