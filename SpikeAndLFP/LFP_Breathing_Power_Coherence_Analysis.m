%% ========================================================================
%  LFP-BREATHING POWER CORRELATION & COHERENCE ANALYSIS
%  Time-resolved power correlation and coherence across frequencies
%  ========================================================================
%
%  Purpose: Analyze time-resolved relationship between LFP and breathing
%
%  Method:
%  1. For each frequency band (0.1-20 Hz in 1 Hz bins):
%     a) Extract power envelopes (amplitude^2) for both LFP and breathing
%     b) Compute sliding window power correlation
%     c) Compute sliding window coherence
%  2. Test whether:
%     - Power correlation varies across session
%     - High coherence is driven by concurrent high power
%
%  Output: Time-resolved metrics for each session, period, and frequency
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== LFP-BREATHING POWER CORRELATION & COHERENCE ANALYSIS ===\n\n');

config = struct();

% Frequency bands (same as other analyses: 0.1-20 Hz in 1 Hz bins)
freq_start = 0.1:1:19;
freq_end = 1:1:20;
n_bands = length(freq_start);

config.frequency_bands = cell(n_bands, 2);
for i = 1:n_bands
    config.frequency_bands{i, 1} = sprintf('%.1f-%.1fHz', freq_start(i), freq_end(i));
    config.frequency_bands{i, 2} = [freq_start(i), freq_end(i)];
end

% Time-resolved analysis parameters
config.window_size = 30;       % 30 second sliding window
config.window_step = 5;        % 5 second step (overlap)
config.min_window_dur = 10;    % Minimum window duration for analysis

% Coherence parameters
config.coherence_nfft = 2048;  % FFT length for coherence
config.coherence_noverlap = 1024;  % Overlap for coherence

% Signal filtering
config.bp_range = [0.1 300];
config.breathing_channel = 32;

% Data paths
config.spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';
config.numofsession = 999;

% Output paths
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_LFP_Breathing_PowerCoh');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_LFP_Breathing_PowerCoh');
if ~exist(RewardSeekingPath, 'dir'), mkdir(RewardSeekingPath); end
if ~exist(RewardAversivePath, 'dir'), mkdir(RewardAversivePath); end

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/NewScripts/');

fprintf('Configuration:\n');
fprintf('  Frequency bands: %d (0.1-20 Hz, 1 Hz bins)\n', n_bands);
fprintf('  Sliding window: %.0f sec\n', config.window_size);
fprintf('  Window step: %.0f sec\n', config.window_step);
fprintf('  Breathing channel: %d\n\n', config.breathing_channel);

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

for sess_idx = 10:num_aversive_sessions
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

    % Preprocess LFP
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);
    bestChannel = findBestLFPChannel(filtered_data, Fs);
    LFP = filtered_data(:, bestChannel);

    % Extract breathing signal
    breathing_raw = Signal(:, config.breathing_channel);
    [filtered_breathing] = preprocessSignals(breathing_raw, Fs, config.bp_range);
    Breathing = filtered_breathing;

    n_valid_aversive = n_valid_aversive + 1;

    % Period boundaries (7 periods)
    period_boundaries = [TriggerMid(1), all_aversive_time(1:6)' + TriggerMid(1), TriggerMid(end)];
    n_periods = 7;

    fprintf('  LFP Channel: %d, Breathing Channel: %d, Periods: %d\n', ...
            bestChannel, config.breathing_channel, n_periods);

    % Compute power correlation and coherence
    session_results = compute_power_correlation_coherence(NeuralTime, LFP, Breathing, ...
                                                           period_boundaries, n_periods, Fs, config);

    session_results.session_id = n_valid_aversive;
    session_results.filename = allfiles_aversive(sess_idx).name;
    session_results.session_type = 'RewardAversive';
    session_results.best_channel = bestChannel;
    session_results.breathing_channel = config.breathing_channel;

    % Save
    [~, base_filename, ~] = fileparts(allfiles_aversive(sess_idx).name);
    save_filename = fullfile(RewardAversivePath, sprintf('%s_lfp_breathing_powercoh.mat', base_filename));
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

for sess_idx = 1:num_reward_sessions
    fprintf('\n[%d/%d] %s\n', sess_idx, num_reward_sessions, allfiles_reward(sess_idx).name);
    tic;

    % Load data
    Timelimits = 'No';
    [NeuralTime, ~, ~, Signal, ~, ~, ~, ~, ~, ~, valid_spikes, Fs, TriggerMid] = ...
        loadAndPrepareSessionData(allfiles_reward(sess_idx), T_sorted, Timelimits);

    % Preprocess LFP
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);
    bestChannel = findBestLFPChannel(filtered_data, Fs);
    LFP = filtered_data(:, bestChannel);

    % Extract breathing signal
    breathing_raw = Signal(:, config.breathing_channel);
    [filtered_breathing] = preprocessSignals(breathing_raw, Fs, config.bp_range);
    Breathing = filtered_breathing;

    n_valid_reward = n_valid_reward + 1;

    % Period boundaries (4 periods)
    period_boundaries = [avg_time_boundaries + TriggerMid(1)];
    n_periods = 4;

    fprintf('  LFP Channel: %d, Breathing Channel: %d, Periods: %d\n', ...
            bestChannel, config.breathing_channel, n_periods);

    % Compute power correlation and coherence
    session_results = compute_power_correlation_coherence(NeuralTime, LFP, Breathing, ...
                                                           period_boundaries, n_periods, Fs, config);

    session_results.session_id = n_valid_reward;
    session_results.filename = allfiles_reward(sess_idx).name;
    session_results.session_type = 'RewardSeeking';
    session_results.best_channel = bestChannel;
    session_results.breathing_channel = config.breathing_channel;

    % Save
    [~, base_filename, ~] = fileparts(allfiles_reward(sess_idx).name);
    save_filename = fullfile(RewardSeekingPath, sprintf('%s_lfp_breathing_powercoh.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    fprintf('  ✓ Complete (%.1f sec)\n', toc);
end

fprintf('\n✓ Processed %d reward sessions\n', n_valid_reward);

%% ========================================================================
%  COMPLETION
%  ========================================================================

fprintf('\n========================================\n');
fprintf('LFP-BREATHING POWER CORRELATION & COHERENCE ANALYSIS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Summary:\n');
fprintf('  Aversive sessions: %d\n', n_valid_aversive);
fprintf('  Reward sessions: %d\n', n_valid_reward);
fprintf('  Frequency bands: %d\n', n_bands);
fprintf('\nResults saved to:\n');
fprintf('  %s\n', RewardAversivePath);
fprintf('  %s\n', RewardSeekingPath);
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function session_results = compute_power_correlation_coherence(NeuralTime, LFP, Breathing, ...
                                                                 period_boundaries, n_periods, Fs, config)
% Compute time-resolved power correlation and coherence between LFP and breathing
%
% For each frequency band:
%   1. Extract power envelopes (amplitude^2)
%   2. Compute sliding window correlations
%   3. Compute sliding window coherence
%
% INPUTS:
%   NeuralTime - time vector
%   LFP - LFP signal vector
%   Breathing - breathing signal vector
%   period_boundaries - period start/end times
%   n_periods - number of periods
%   Fs - sampling frequency
%   config - configuration structure
%
% OUTPUTS:
%   session_results - structure with:
%     .data - table with columns:
%       Period, Band, Freq_Low_Hz, Freq_High_Hz, Window_Start_Time, Window_End_Time,
%       Power_Corr_R, Power_Corr_P, Coherence_Mean, Coherence_Peak_Freq,
%       LFP_Power_Mean, Breathing_Power_Mean, Window_Duration

    n_bands = size(config.frequency_bands, 1);

    % Pre-compute time mapping
    t0 = NeuralTime(1);
    dt = median(diff(NeuralTime));

    % Pre-allocate results storage
    max_windows_per_period = ceil(max(diff(period_boundaries)) / config.window_step);
    max_results = n_bands * n_periods * max_windows_per_period;

    results = struct();
    results.period = zeros(max_results, 1);
    results.band_name = cell(max_results, 1);
    results.freq_low = zeros(max_results, 1);
    results.freq_high = zeros(max_results, 1);
    results.window_start_time = zeros(max_results, 1);
    results.window_end_time = zeros(max_results, 1);
    results.window_center_time = zeros(max_results, 1);
    results.power_corr_r = zeros(max_results, 1);
    results.power_corr_p = zeros(max_results, 1);
    results.coherence_mean = zeros(max_results, 1);
    results.coherence_peak_freq = zeros(max_results, 1);
    results.coherence_peak_value = zeros(max_results, 1);
    results.lfp_power_mean = zeros(max_results, 1);
    results.breathing_power_mean = zeros(max_results, 1);
    results.window_duration = zeros(max_results, 1);

    result_idx = 0;

    fprintf('  Computing power correlation & coherence for %d bands...\n', n_bands);

    % Process each frequency band
    for band_idx = 1:n_bands
        band_name = config.frequency_bands{band_idx, 1};
        band_range = config.frequency_bands{band_idx, 2};

        % Filter LFP and extract power
        LFP_filtered = filter_signal_robust(LFP, band_range, Fs);
        LFP_analytic = hilbert(LFP_filtered);
        LFP_amplitude = abs(LFP_analytic);
        LFP_power = LFP_amplitude.^2;  % Power = amplitude squared

        % Filter breathing and extract power
        Breathing_filtered = filter_signal_robust(Breathing, band_range, Fs);
        Breathing_analytic = hilbert(Breathing_filtered);
        Breathing_amplitude = abs(Breathing_analytic);
        Breathing_power = Breathing_amplitude.^2;

        % Clear intermediate variables
        clear LFP_filtered LFP_analytic Breathing_filtered Breathing_analytic;

        % Process each period
        for period_idx = 1:n_periods
            period_start = period_boundaries(period_idx);
            period_end = period_boundaries(period_idx + 1);
            period_duration = period_end - period_start;

            if period_duration < config.min_window_dur
                continue;
            end

            % Define sliding windows
            window_starts = period_start:config.window_step:(period_end - config.window_size);

            % Process each window
            for win_idx = 1:length(window_starts)
                win_start = window_starts(win_idx);
                win_end = min(win_start + config.window_size, period_end);
                win_duration = win_end - win_start;

                if win_duration < config.min_window_dur
                    continue;
                end

                % Get indices for this window
                start_idx = round((win_start - t0) / dt) + 1;
                end_idx = round((win_end - t0) / dt) + 1;
                start_idx = max(1, min(start_idx, length(LFP_power)));
                end_idx = max(1, min(end_idx, length(LFP_power)));

                if end_idx <= start_idx
                    continue;
                end

                % Extract power in this window
                lfp_pow_window = LFP_power(start_idx:end_idx);
                breath_pow_window = Breathing_power(start_idx:end_idx);

                % Compute power correlation
                [r, p] = corr(lfp_pow_window, breath_pow_window, 'Type', 'Pearson');

                % Extract raw signals for coherence
                lfp_window = LFP_amplitude(start_idx:end_idx);
                breath_window = Breathing_amplitude(start_idx:end_idx);

                % Compute coherence using mscohere
                try
                    [coh, freq_coh] = mscohere(lfp_window, breath_window, ...
                                               hamming(config.coherence_nfft), ...
                                               config.coherence_noverlap, ...
                                               config.coherence_nfft, Fs);

                    % Focus on frequency band of interest
                    freq_mask = freq_coh >= band_range(1) & freq_coh <= band_range(2);
                    if sum(freq_mask) > 0
                        coh_in_band = coh(freq_mask);
                        freq_in_band = freq_coh(freq_mask);

                        coherence_mean = mean(coh_in_band);
                        [coherence_peak_value, peak_idx] = max(coh_in_band);
                        coherence_peak_freq = freq_in_band(peak_idx);
                    else
                        coherence_mean = NaN;
                        coherence_peak_freq = NaN;
                        coherence_peak_value = NaN;
                    end
                catch
                    % If coherence computation fails (e.g., too short window)
                    coherence_mean = NaN;
                    coherence_peak_freq = NaN;
                    coherence_peak_value = NaN;
                end

                % Store results
                result_idx = result_idx + 1;
                results.period(result_idx) = period_idx;
                results.band_name{result_idx} = band_name;
                results.freq_low(result_idx) = band_range(1);
                results.freq_high(result_idx) = band_range(2);
                results.window_start_time(result_idx) = win_start;
                results.window_end_time(result_idx) = win_end;
                results.window_center_time(result_idx) = (win_start + win_end) / 2;
                results.power_corr_r(result_idx) = r;
                results.power_corr_p(result_idx) = p;
                results.coherence_mean(result_idx) = coherence_mean;
                results.coherence_peak_freq(result_idx) = coherence_peak_freq;
                results.coherence_peak_value(result_idx) = coherence_peak_value;
                results.lfp_power_mean(result_idx) = mean(lfp_pow_window);
                results.breathing_power_mean(result_idx) = mean(breath_pow_window);
                results.window_duration(result_idx) = win_duration;
            end
        end

        % Clear power signals before next band
        clear LFP_amplitude LFP_power Breathing_amplitude Breathing_power;

        if mod(band_idx, 5) == 0
            fprintf('    Processed %d/%d bands\n', band_idx, n_bands);
        end
    end

    % Trim pre-allocated arrays
    results.period = results.period(1:result_idx);
    results.band_name = results.band_name(1:result_idx);
    results.freq_low = results.freq_low(1:result_idx);
    results.freq_high = results.freq_high(1:result_idx);
    results.window_start_time = results.window_start_time(1:result_idx);
    results.window_end_time = results.window_end_time(1:result_idx);
    results.window_center_time = results.window_center_time(1:result_idx);
    results.power_corr_r = results.power_corr_r(1:result_idx);
    results.power_corr_p = results.power_corr_p(1:result_idx);
    results.coherence_mean = results.coherence_mean(1:result_idx);
    results.coherence_peak_freq = results.coherence_peak_freq(1:result_idx);
    results.coherence_peak_value = results.coherence_peak_value(1:result_idx);
    results.lfp_power_mean = results.lfp_power_mean(1:result_idx);
    results.breathing_power_mean = results.breathing_power_mean(1:result_idx);
    results.window_duration = results.window_duration(1:result_idx);

    % Convert to table
    session_results.data = struct2table(results);
    session_results.data.Properties.VariableNames{'period'} = 'Period';
    session_results.data.Properties.VariableNames{'band_name'} = 'Band';
    session_results.data.Properties.VariableNames{'freq_low'} = 'Freq_Low_Hz';
    session_results.data.Properties.VariableNames{'freq_high'} = 'Freq_High_Hz';
    session_results.data.Properties.VariableNames{'window_start_time'} = 'Window_Start_Time';
    session_results.data.Properties.VariableNames{'window_end_time'} = 'Window_End_Time';
    session_results.data.Properties.VariableNames{'window_center_time'} = 'Window_Center_Time';
    session_results.data.Properties.VariableNames{'power_corr_r'} = 'Power_Corr_R';
    session_results.data.Properties.VariableNames{'power_corr_p'} = 'Power_Corr_P';
    session_results.data.Properties.VariableNames{'coherence_mean'} = 'Coherence_Mean';
    session_results.data.Properties.VariableNames{'coherence_peak_freq'} = 'Coherence_Peak_Freq';
    session_results.data.Properties.VariableNames{'coherence_peak_value'} = 'Coherence_Peak_Value';
    session_results.data.Properties.VariableNames{'lfp_power_mean'} = 'LFP_Power_Mean';
    session_results.data.Properties.VariableNames{'breathing_power_mean'} = 'Breathing_Power_Mean';
    session_results.data.Properties.VariableNames{'window_duration'} = 'Window_Duration';

    session_results.Fs = Fs;

    fprintf('  ✓ Completed %d time windows across all bands and periods\n', result_idx);
end

function signal_filtered = filter_signal_robust(signal, band_range, Fs)
% Filter signal for power/coherence analysis
% Same filtering approach as other analyses
    low_freq = band_range(1);
    high_freq = band_range(2);

    if low_freq < 1
        % For very low frequencies: detrend + lowpass
        signal_detrend = detrend(signal);
        signal_demean = signal_detrend - mean(signal_detrend);
        if high_freq < Fs/2
            signal_filtered = lowpass(signal_demean, high_freq, Fs, ...
                                     'ImpulseResponse', 'fir', 'Steepness', 0.85);
        else
            signal_filtered = signal_demean;
        end
    else
        % For higher frequencies: bandpass
        if high_freq < Fs/2
            signal_filtered = bandpass(signal, band_range, Fs, ...
                                      'ImpulseResponse', 'fir', 'Steepness', 0.85);
        else
            signal_filtered = highpass(signal, low_freq, Fs, ...
                                      'ImpulseResponse', 'fir', 'Steepness', 0.85);
        end
    end
end
