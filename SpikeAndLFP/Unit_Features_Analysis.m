%% ========================================================================
%  UNIT FEATURES ANALYSIS: Period × SessionType
%  Direct period-based calculation of spike train metrics
%  ========================================================================
%
%  Analysis: Unit Features ~ Period × SessionType
%  SessionType: Aversive vs Reward
%  Aversive Periods: P1-P7 (6 aversive noises create 7 periods)
%  Reward Periods: P1-P4 (time-matched to aversive)
%
%  Features: 22 spike train metrics (FR, CV, ISI metrics, burst, ACF, etc.)
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== UNIT FEATURES ANALYSIS: AVERSIVE vs REWARD ===\n');
fprintf('Period × SessionType\n\n');

config = struct();

% Metric calculation parameters
config.min_spikes_for_CV = 10;      % Minimum spikes to calculate CV
config.acf_max_lag = 0.1;           % Auto-correlation up to 100ms
config.burst_isi_threshold = 0.01;  % 10ms threshold for burst detection
config.refrac_threshold = 0.002;    % 2ms for refractory violations
config.count_bin_sizes = [0.001, 0.025, 0.050];  % 1ms, 25ms, 50ms for spike count ACF

% Data paths
config.spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';
config.numofsession = 999;  % Max sessions per animal

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/NewScripts/');

fprintf('Configuration:\n');
fprintf('  Min spikes for CV: %d\n', config.min_spikes_for_CV);
fprintf('  Aversive periods: 7 (based on 6 noise events)\n');
fprintf('  Reward periods: 4 (time-matched)\n');
fprintf('  Total features: 22\n\n');

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
aversive_data.session_name = {};
aversive_data.period_duration = [];

% Initialize all 22 metrics
aversive_data.FR = [];
aversive_data.CV = [];
aversive_data.ISI_FanoFactor = [];
aversive_data.ISI_ACF_peak = [];
aversive_data.ISI_ACF_lag = [];
aversive_data.ISI_ACF_decay = [];
aversive_data.Count_ACF_1ms_peak = [];
aversive_data.Count_ACF_25ms_peak = [];
aversive_data.Count_ACF_50ms_peak = [];
aversive_data.LV = [];
aversive_data.CV2 = [];
aversive_data.LVR = [];
aversive_data.BurstIndex = [];
aversive_data.BurstRate = [];
aversive_data.MeanBurstLength = [];
aversive_data.ISI_Skewness = [];
aversive_data.ISI_Kurtosis = [];
aversive_data.ISI_Mode = [];
aversive_data.CountFanoFactor_1ms = [];
aversive_data.CountFanoFactor_25ms = [];
aversive_data.CountFanoFactor_50ms = [];
aversive_data.RefracViolations = [];

n_valid_aversive = 0;

for sess_idx = 1:num_aversive_sessions
    fprintf('\n[%d/%d] Processing: %s\n', sess_idx, num_aversive_sessions, allfiles_aversive(sess_idx).name);
    tic;

    % Load raw spike data
    Timelimits = 'No';
    [NeuralTime, ~, ~, ~, ~, ~, ~, ~, AversiveSound, ~, valid_spikes, Fs, TriggerMid] = ...
        loadAndPrepareSessionData(allfiles_aversive(sess_idx), T_sorted, Timelimits);

    % Get all aversive sound timepoints
    aversive_onsets = find(diff(AversiveSound) == 1);
    all_aversive_time = NeuralTime(aversive_onsets);

    if length(all_aversive_time) < 6
        fprintf('  Skipping: insufficient aversive events (%d, need 6)\n', length(all_aversive_time));
        continue;
    end

    n_valid_aversive = n_valid_aversive + 1;
    spike_filename = allfiles_aversive(sess_idx).name;

    % Define 7 period boundaries using 6 noises
    period_boundaries = [TriggerMid(1), ...
                         all_aversive_time(1:6)' + TriggerMid(1), ...
                         TriggerMid(end)];

    n_periods = 7;
    n_units = length(valid_spikes);

    fprintf('  Processing %d units across %d periods\n', n_units, n_periods);

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

            % Calculate all metrics for this period
            metrics = calculateAllMetrics(spikes_in_period, period_start, period_end, config);

            % Store data
            aversive_data.session_id(end+1) = n_valid_aversive;
            aversive_data.unit_id(end+1) = (n_valid_aversive - 1) * 1000 + unit_idx;
            aversive_data.period(end+1) = period_idx;
            aversive_data.session_name{end+1} = spike_filename;
            aversive_data.period_duration(end+1) = period_duration;

            % Store all metrics
            aversive_data.FR(end+1) = metrics.FR;
            aversive_data.CV(end+1) = metrics.CV;
            aversive_data.ISI_FanoFactor(end+1) = metrics.ISI_FanoFactor;
            aversive_data.ISI_ACF_peak(end+1) = metrics.ISI_ACF_peak;
            aversive_data.ISI_ACF_lag(end+1) = metrics.ISI_ACF_lag;
            aversive_data.ISI_ACF_decay(end+1) = metrics.ISI_ACF_decay;
            aversive_data.Count_ACF_1ms_peak(end+1) = metrics.Count_ACF_1ms_peak;
            aversive_data.Count_ACF_25ms_peak(end+1) = metrics.Count_ACF_25ms_peak;
            aversive_data.Count_ACF_50ms_peak(end+1) = metrics.Count_ACF_50ms_peak;
            aversive_data.LV(end+1) = metrics.LV;
            aversive_data.CV2(end+1) = metrics.CV2;
            aversive_data.LVR(end+1) = metrics.LVR;
            aversive_data.BurstIndex(end+1) = metrics.BurstIndex;
            aversive_data.BurstRate(end+1) = metrics.BurstRate;
            aversive_data.MeanBurstLength(end+1) = metrics.MeanBurstLength;
            aversive_data.ISI_Skewness(end+1) = metrics.ISI_Skewness;
            aversive_data.ISI_Kurtosis(end+1) = metrics.ISI_Kurtosis;
            aversive_data.ISI_Mode(end+1) = metrics.ISI_Mode;
            aversive_data.CountFanoFactor_1ms(end+1) = metrics.CountFanoFactor_1ms;
            aversive_data.CountFanoFactor_25ms(end+1) = metrics.CountFanoFactor_25ms;
            aversive_data.CountFanoFactor_50ms(end+1) = metrics.CountFanoFactor_50ms;
            aversive_data.RefracViolations(end+1) = metrics.RefracViolations;
        end
    end

    elapsed = toc;
    fprintf('  ✓ Session %d complete: %d units × %d periods = %d data points (%.1f sec)\n', ...
            n_valid_aversive, n_units, n_periods, n_units * n_periods, elapsed);
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
reward_data.session_name = {};
reward_data.period_duration = [];

% Initialize all 22 metrics
reward_data.FR = [];
reward_data.CV = [];
reward_data.ISI_FanoFactor = [];
reward_data.ISI_ACF_peak = [];
reward_data.ISI_ACF_lag = [];
reward_data.ISI_ACF_decay = [];
reward_data.Count_ACF_1ms_peak = [];
reward_data.Count_ACF_25ms_peak = [];
reward_data.Count_ACF_50ms_peak = [];
reward_data.LV = [];
reward_data.CV2 = [];
reward_data.LVR = [];
reward_data.BurstIndex = [];
reward_data.BurstRate = [];
reward_data.MeanBurstLength = [];
reward_data.ISI_Skewness = [];
reward_data.ISI_Kurtosis = [];
reward_data.ISI_Mode = [];
reward_data.CountFanoFactor_1ms = [];
reward_data.CountFanoFactor_25ms = [];
reward_data.CountFanoFactor_50ms = [];
reward_data.RefracViolations = [];

n_valid_reward = 0;

avg_time_boundaries = [0, 8*60, 16*60, 24*60, 30*60];
fprintf('  Average time boundaries: [%.1f, %.1f, %.1f] seconds\n', avg_time_boundaries);
fprintf('  Average time boundaries: [%.1f, %.1f, %.1f] minutes\n\n', avg_time_boundaries/60);

% Process reward sessions
for sess_idx = 1:num_reward_sessions
    fprintf('\n[%d/%d] Processing: %s\n', sess_idx, num_reward_sessions, allfiles_reward(sess_idx).name);
    tic;

    % Load raw spike data
    Timelimits = 'No';
    [NeuralTime, ~, ~, ~, ~, ~, ~, ~, ~, ~, valid_spikes, Fs, TriggerMid] = ...
        loadAndPrepareSessionData(allfiles_reward(sess_idx), T_sorted, Timelimits);

    n_valid_reward = n_valid_reward + 1;
    spike_filename = allfiles_reward(sess_idx).name;

    % Define 4 period boundaries using time-matched approach
    % P1-P3: Time-matched to first 3 aversive noises
    % P4: Remaining time
    period_boundaries = [avg_time_boundaries + TriggerMid(1)];

    n_periods = 4;
    n_units = length(valid_spikes);

    fprintf('  Processing %d units across %d periods\n', n_units, n_periods);

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

            % Calculate all metrics for this period
            metrics = calculateAllMetrics(spikes_in_period, period_start, period_end, config);

            % Store data
            reward_data.session_id(end+1) = n_valid_reward;
            reward_data.unit_id(end+1) = (n_valid_reward + 10000) * 1000 + unit_idx;
            reward_data.period(end+1) = period_idx;
            reward_data.session_name{end+1} = spike_filename;
            reward_data.period_duration(end+1) = period_duration;

            % Store all metrics
            reward_data.FR(end+1) = metrics.FR;
            reward_data.CV(end+1) = metrics.CV;
            reward_data.ISI_FanoFactor(end+1) = metrics.ISI_FanoFactor;
            reward_data.ISI_ACF_peak(end+1) = metrics.ISI_ACF_peak;
            reward_data.ISI_ACF_lag(end+1) = metrics.ISI_ACF_lag;
            reward_data.ISI_ACF_decay(end+1) = metrics.ISI_ACF_decay;
            reward_data.Count_ACF_1ms_peak(end+1) = metrics.Count_ACF_1ms_peak;
            reward_data.Count_ACF_25ms_peak(end+1) = metrics.Count_ACF_25ms_peak;
            reward_data.Count_ACF_50ms_peak(end+1) = metrics.Count_ACF_50ms_peak;
            reward_data.LV(end+1) = metrics.LV;
            reward_data.CV2(end+1) = metrics.CV2;
            reward_data.LVR(end+1) = metrics.LVR;
            reward_data.BurstIndex(end+1) = metrics.BurstIndex;
            reward_data.BurstRate(end+1) = metrics.BurstRate;
            reward_data.MeanBurstLength(end+1) = metrics.MeanBurstLength;
            reward_data.ISI_Skewness(end+1) = metrics.ISI_Skewness;
            reward_data.ISI_Kurtosis(end+1) = metrics.ISI_Kurtosis;
            reward_data.ISI_Mode(end+1) = metrics.ISI_Mode;
            reward_data.CountFanoFactor_1ms(end+1) = metrics.CountFanoFactor_1ms;
            reward_data.CountFanoFactor_25ms(end+1) = metrics.CountFanoFactor_25ms;
            reward_data.CountFanoFactor_50ms(end+1) = metrics.CountFanoFactor_50ms;
            reward_data.RefracViolations(end+1) = metrics.RefracViolations;
        end
    end

    elapsed = toc;
    fprintf('  ✓ Session %d complete: %d units × %d periods = %d data points (%.1f sec)\n', ...
            n_valid_reward, n_units, n_periods, n_units * n_periods, elapsed);
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
combined_data.session_type = [aversive_data.session_type; reward_data.session_type];
combined_data.period_duration = [aversive_data.period_duration(:); reward_data.period_duration(:)];

% Combine all 22 metrics
combined_data.FR = [aversive_data.FR(:); reward_data.FR(:)];
combined_data.CV = [aversive_data.CV(:); reward_data.CV(:)];
combined_data.ISI_FanoFactor = [aversive_data.ISI_FanoFactor(:); reward_data.ISI_FanoFactor(:)];
combined_data.ISI_ACF_peak = [aversive_data.ISI_ACF_peak(:); reward_data.ISI_ACF_peak(:)];
combined_data.ISI_ACF_lag = [aversive_data.ISI_ACF_lag(:); reward_data.ISI_ACF_lag(:)];
combined_data.ISI_ACF_decay = [aversive_data.ISI_ACF_decay(:); reward_data.ISI_ACF_decay(:)];
combined_data.Count_ACF_1ms_peak = [aversive_data.Count_ACF_1ms_peak(:); reward_data.Count_ACF_1ms_peak(:)];
combined_data.Count_ACF_25ms_peak = [aversive_data.Count_ACF_25ms_peak(:); reward_data.Count_ACF_25ms_peak(:)];
combined_data.Count_ACF_50ms_peak = [aversive_data.Count_ACF_50ms_peak(:); reward_data.Count_ACF_50ms_peak(:)];
combined_data.LV = [aversive_data.LV(:); reward_data.LV(:)];
combined_data.CV2 = [aversive_data.CV2(:); reward_data.CV2(:)];
combined_data.LVR = [aversive_data.LVR(:); reward_data.LVR(:)];
combined_data.BurstIndex = [aversive_data.BurstIndex(:); reward_data.BurstIndex(:)];
combined_data.BurstRate = [aversive_data.BurstRate(:); reward_data.BurstRate(:)];
combined_data.MeanBurstLength = [aversive_data.MeanBurstLength(:); reward_data.MeanBurstLength(:)];
combined_data.ISI_Skewness = [aversive_data.ISI_Skewness(:); reward_data.ISI_Skewness(:)];
combined_data.ISI_Kurtosis = [aversive_data.ISI_Kurtosis(:); reward_data.ISI_Kurtosis(:)];
combined_data.ISI_Mode = [aversive_data.ISI_Mode(:); reward_data.ISI_Mode(:)];
combined_data.CountFanoFactor_1ms = [aversive_data.CountFanoFactor_1ms(:); reward_data.CountFanoFactor_1ms(:)];
combined_data.CountFanoFactor_25ms = [aversive_data.CountFanoFactor_25ms(:); reward_data.CountFanoFactor_25ms(:)];
combined_data.CountFanoFactor_50ms = [aversive_data.CountFanoFactor_50ms(:); reward_data.CountFanoFactor_50ms(:)];
combined_data.RefracViolations = [aversive_data.RefracViolations(:); reward_data.RefracViolations(:)];

% Convert to table
tbl_data = struct2table(combined_data);

% Rename columns
tbl_data.Properties.VariableNames{'session_id'} = 'Session';
tbl_data.Properties.VariableNames{'unit_id'} = 'Unit';
tbl_data.Properties.VariableNames{'period'} = 'Period';
tbl_data.Properties.VariableNames{'session_type'} = 'SessionType';
tbl_data.Properties.VariableNames{'period_duration'} = 'Period_Duration_sec';

% Convert to categorical
tbl_data.Session = categorical(tbl_data.Session);
tbl_data.Unit = categorical(tbl_data.Unit);
tbl_data.Period = categorical(tbl_data.Period);
tbl_data.SessionType = categorical(tbl_data.SessionType);

fprintf('✓ Combined dataset created\n');
fprintf('  Total data points: %d\n', height(tbl_data));
fprintf('  Sessions: %d aversive, %d reward\n', n_valid_aversive, n_valid_reward);
fprintf('  Features: 22\n\n');

%% Save results
fprintf('Saving results...\n');

results = struct();
results.tbl_data = tbl_data;
results.config = config;
results.n_aversive_sessions = n_valid_aversive;
results.n_reward_sessions = n_valid_reward;
results.avg_time_boundaries = avg_time_boundaries;

timestamp = datestr(now, 'dd-mmm-yyyy');
save_filename = sprintf('Unit_features_analysis_%s.mat', timestamp);
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
fprintf('  Total data points: %d\n', height(tbl_data));
fprintf('\nData structure:\n');
fprintf('  Unit | Session | SessionType | Period | Period_Duration_sec | [22 features]\n');
fprintf('\nFeatures calculated:\n');
fprintf('  1. FR (Firing Rate)\n');
fprintf('  2. CV (Coefficient of Variation)\n');
fprintf('  3. ISI_FanoFactor\n');
fprintf('  4-6. ISI_ACF (peak, lag, decay)\n');
fprintf('  7-9. Count_ACF peaks (1ms, 25ms, 50ms)\n');
fprintf('  10-12. LV, CV2, LVR\n');
fprintf('  13-15. Burst metrics (index, rate, mean length)\n');
fprintf('  16-18. ISI distribution (skewness, kurtosis, mode)\n');
fprintf('  19-21. CountFanoFactor (1ms, 25ms, 50ms)\n');
fprintf('  22. RefracViolations\n');
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS FOR METRIC CALCULATION
%  ========================================================================

function metrics = calculateAllMetrics(spike_times, period_start, period_end, config)
% Calculate all metrics for a single period
% spike_times: already filtered to be within period
% Returns struct with all computed metrics

    metrics = struct();

    n_spikes = length(spike_times);
    period_duration = period_end - period_start;

    % Basic metrics
    metrics.FR = n_spikes / period_duration;

    % Initialize all other metrics as NaN
    metrics.CV = NaN;
    metrics.ISI_FanoFactor = NaN;
    metrics.ISI_ACF_peak = NaN;
    metrics.ISI_ACF_lag = NaN;
    metrics.ISI_ACF_decay = NaN;
    metrics.Count_ACF_1ms_peak = NaN;
    metrics.Count_ACF_25ms_peak = NaN;
    metrics.Count_ACF_50ms_peak = NaN;
    metrics.LV = NaN;
    metrics.CV2 = NaN;
    metrics.LVR = NaN;
    metrics.BurstIndex = NaN;
    metrics.BurstRate = NaN;
    metrics.MeanBurstLength = NaN;
    metrics.ISI_Skewness = NaN;
    metrics.ISI_Kurtosis = NaN;
    metrics.ISI_Mode = NaN;
    metrics.CountFanoFactor_1ms = NaN;
    metrics.CountFanoFactor_25ms = NaN;
    metrics.CountFanoFactor_50ms = NaN;
    metrics.RefracViolations = NaN;

    % Need enough spikes for ISI-based metrics
    if n_spikes < config.min_spikes_for_CV
        return;
    end

    % Calculate ISI
    ISI = diff(spike_times);

    if isempty(ISI)
        return;
    end

    % === ISI-based metrics ===
    isi_mean = mean(ISI);
    isi_std = std(ISI);

    if isi_mean > 0
        % 1. CV
        metrics.CV = isi_std / isi_mean;

        % 2. ISI Fano Factor
        metrics.ISI_FanoFactor = var(ISI) / isi_mean;

        % 3. ISI Auto-correlation
        [acf_vals, acf_lags] = calculateISI_ACF(ISI, config.acf_max_lag);
        if ~isempty(acf_vals) && length(acf_vals) > 1
            [metrics.ISI_ACF_peak, peak_idx] = max(acf_vals(2:end));  % Skip lag 0
            metrics.ISI_ACF_lag = acf_lags(peak_idx + 1);
            metrics.ISI_ACF_decay = calculateACFDecay(acf_vals, acf_lags);
        end

        % 4. Local Variation (LV)
        metrics.LV = calculateLV(ISI);

        % 5. CV2
        metrics.CV2 = calculateCV2(ISI);

        % 6. LVR (Revised Local Variation)
        metrics.LVR = calculateLVR(ISI, config.refrac_threshold);

        % 7-9. Burst metrics
        [metrics.BurstIndex, metrics.BurstRate, metrics.MeanBurstLength] = ...
            calculateBurstMetrics(ISI, config.burst_isi_threshold, period_duration);

        % 10-12. ISI distribution shape
        if length(ISI) >= 3
            metrics.ISI_Skewness = skewness(ISI);
            metrics.ISI_Kurtosis = kurtosis(ISI);
            metrics.ISI_Mode = mode(round(ISI, 4));  % Round to avoid floating point issues
        end

        % 14. Refractory violations
        metrics.RefracViolations = 100 * sum(ISI < config.refrac_threshold) / length(ISI);
    end

    % === Spike count based metrics ===
    % 13. Spike count Fano Factor and ACF for different bin sizes
    for bin_idx = 1:length(config.count_bin_sizes)
        bin_size = config.count_bin_sizes(bin_idx);

        [fano, acf_peak] = calculateCountMetrics(spike_times, period_start, period_end, ...
                                                  bin_size, config.acf_max_lag);

        if bin_size == 0.001
            metrics.CountFanoFactor_1ms = fano;
            metrics.Count_ACF_1ms_peak = acf_peak;
        elseif bin_size == 0.025
            metrics.CountFanoFactor_25ms = fano;
            metrics.Count_ACF_25ms_peak = acf_peak;
        elseif bin_size == 0.050
            metrics.CountFanoFactor_50ms = fano;
            metrics.Count_ACF_50ms_peak = acf_peak;
        end
    end
end

function [acf_vals, acf_lags] = calculateISI_ACF(ISI, max_lag)
% Calculate auto-correlation of ISI up to max_lag

    if length(ISI) < 3
        acf_vals = [];
        acf_lags = [];
        return;
    end

    % Determine number of lags
    mean_isi = mean(ISI);
    if mean_isi == 0
        acf_vals = [];
        acf_lags = [];
        return;
    end

    max_lag_samples = min(floor(max_lag / mean_isi), length(ISI) - 1);
    max_lag_samples = max(max_lag_samples, 1);

    % Calculate ACF
    try
        [acf_vals, ~, ~] = autocorr(ISI, max_lag_samples);
        acf_lags = (0:max_lag_samples) * mean_isi;
    catch
        acf_vals = [];
        acf_lags = [];
    end
end

function decay_time = calculateACFDecay(acf_vals, acf_lags)
% Find time to reach 50% of peak ACF value

    if length(acf_vals) < 2
        decay_time = NaN;
        return;
    end

    peak_val = max(acf_vals(2:end));  % Exclude lag 0
    threshold = peak_val * 0.5;

    % Find first crossing
    crossing_idx = find(acf_vals(2:end) < threshold, 1, 'first');

    if isempty(crossing_idx)
        decay_time = acf_lags(end);
    else
        decay_time = acf_lags(crossing_idx + 1);
    end
end

function LV = calculateLV(ISI)
% Local Variation: sensitive to rate changes

    n = length(ISI);
    if n < 2
        LV = NaN;
        return;
    end

    sum_term = 0;
    for i = 1:(n-1)
        sum_term = sum_term + ((ISI(i+1) - ISI(i))^2) / ((ISI(i+1) + ISI(i))^2);
    end

    LV = (3 / (n - 1)) * sum_term;
end

function CV2 = calculateCV2(ISI)
% CV2: Local coefficient of variation

    n = length(ISI);
    if n < 2
        CV2 = NaN;
        return;
    end

    sum_term = 0;
    for i = 1:(n-1)
        sum_term = sum_term + abs(ISI(i+1) - ISI(i)) / (ISI(i+1) + ISI(i));
    end

    CV2 = 2 * sum_term / (n - 1);
end

function LVR = calculateLVR(ISI, refrac_period)
% Revised Local Variation: corrected for refractoriness

    n = length(ISI);
    if n < 2
        LVR = NaN;
        return;
    end

    sum_term = 0;
    valid_count = 0;

    for i = 1:(n-1)
        % Only include pairs where both ISIs > refractory period
        if ISI(i) > refrac_period && ISI(i+1) > refrac_period
            sum_term = sum_term + ((ISI(i+1) - ISI(i))^2) / ((ISI(i+1) + ISI(i))^2);
            valid_count = valid_count + 1;
        end
    end

    if valid_count > 0
        LVR = (3 / valid_count) * sum_term;
    else
        LVR = NaN;
    end
end

function [burst_index, burst_rate, mean_burst_length] = calculateBurstMetrics(ISI, threshold, duration)
% Calculate burst-related metrics

    if isempty(ISI)
        burst_index = NaN;
        burst_rate = NaN;
        mean_burst_length = NaN;
        return;
    end

    % Burst index: fraction of ISIs below threshold
    burst_index = sum(ISI < threshold) / length(ISI);

    % Detect bursts: sequences of ISIs < threshold
    is_burst_isi = ISI < threshold;
    burst_starts = find(diff([0; is_burst_isi]) == 1);
    burst_ends = find(diff([is_burst_isi; 0]) == -1);

    n_bursts = length(burst_starts);

    if n_bursts > 0
        burst_rate = n_bursts / duration;

        % Calculate burst lengths (number of spikes)
        burst_lengths = zeros(n_bursts, 1);
        for b = 1:n_bursts
            burst_lengths(b) = burst_ends(b) - burst_starts(b) + 2;  % +2 for first and last spike
        end
        mean_burst_length = mean(burst_lengths);
    else
        burst_rate = 0;
        mean_burst_length = NaN;
    end
end

function [fano_factor, acf_peak] = calculateCountMetrics(spike_times, period_start, period_end, bin_size, max_lag)
% Calculate spike count Fano factor and ACF for given bin size

    % Create bins
    bin_edges = period_start:bin_size:period_end;

    if length(bin_edges) < 3
        fano_factor = NaN;
        acf_peak = NaN;
        return;
    end

    % Count spikes in bins
    spike_counts = histcounts(spike_times, bin_edges);

    % Fano factor
    if mean(spike_counts) > 0
        fano_factor = var(spike_counts) / mean(spike_counts);
    else
        fano_factor = NaN;
    end

    % ACF - EXCLUDE LAG 0
    if length(spike_counts) >= 3
        max_lag_bins = min(floor(max_lag / bin_size), length(spike_counts) - 1);
        max_lag_bins = max(max_lag_bins, 1);

        try
            acf_vals = autocorr(spike_counts, max_lag_bins);
            if length(acf_vals) > 1
                acf_peak = max(acf_vals(2:end));  % EXCLUDE LAG 0
            else
                acf_peak = NaN;
            end
        catch
            acf_peak = NaN;
        end
    else
        acf_peak = NaN;
    end
end
