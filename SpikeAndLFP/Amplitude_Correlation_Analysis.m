%% ========================================================================
%  AMPLITUDE CORRELATION ANALYSIS
%  Spike rate vs LFP amplitude: Linear & Nonlinear coupling
%  ========================================================================
%
%  Purpose: Test for amplitude modulation (explains high SFC + low PPC)
%
%  Method: For each frequency band (0.1-20 Hz):
%  1. Extract LFP amplitude envelope (Hilbert transform)
%  2. Bin spike rate over time
%  3. Compute Pearson correlation (linear coupling)
%  4. Compute mutual information (nonlinear/context-dependent coupling)
%
%  High MI + Low Corr → Context-dependent amplitude modulation
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== AMPLITUDE CORRELATION ANALYSIS ===\n\n');

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

% Analysis parameters
config.bin_size = 0.1;  % 100 ms bins for spike rate & amplitude
config.min_bins = 50;   % Minimum bins for reliable correlation
config.mi_bins = 10;    % Number of bins for mutual information discretization

% LFP filtering
config.bp_range = [0.1 300];

% Data paths
config.spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';
config.numofsession = 999;

% Output paths
DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_AmpCorr');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_AmpCorr');
if ~exist(RewardSeekingPath, 'dir'), mkdir(RewardSeekingPath); end
if ~exist(RewardAversivePath, 'dir'), mkdir(RewardAversivePath); end

% Add paths
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/BreathingLFPSpikeToolbox/');
addpath('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/NewScripts/');

fprintf('Configuration:\n');
fprintf('  Frequency bands: %d (0.1-20 Hz, 1 Hz bins)\n', n_bands);
fprintf('  Bin size: %.0f ms\n', config.bin_size * 1000);
fprintf('  MI discretization: %d bins\n', config.mi_bins);
fprintf('  Min bins: %d\n\n', config.min_bins);

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

    % Preprocess LFP
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);
    bestChannel = findBestLFPChannel(filtered_data, Fs);
    LFP = filtered_data(:, bestChannel);

    n_valid_aversive = n_valid_aversive + 1;

    % Period boundaries
    period_boundaries = [TriggerMid(1), all_aversive_time(1:6)' + TriggerMid(1), TriggerMid(end)];
    n_periods = 7;
    n_units = length(valid_spikes);

    fprintf('  Units: %d, Periods: %d, Channel: %d\n', n_units, n_periods, bestChannel);

    % Compute amplitude correlation
    session_results = compute_amplitude_correlation(NeuralTime, LFP, valid_spikes, period_boundaries, ...
                                                     n_periods, n_units, Fs, config);

    session_results.session_id = n_valid_aversive;
    session_results.filename = allfiles_aversive(sess_idx).name;
    session_results.session_type = 'RewardAversive';
    session_results.best_channel = bestChannel;

    % Save
    [~, base_filename, ~] = fileparts(allfiles_aversive(sess_idx).name);
    save_filename = fullfile(RewardAversivePath, sprintf('%s_ampcorr.mat', base_filename));
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

    % Preprocess LFP
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);
    bestChannel = findBestLFPChannel(filtered_data, Fs);
    LFP = filtered_data(:, bestChannel);

    n_valid_reward = n_valid_reward + 1;

    % Period boundaries
    period_boundaries = [avg_time_boundaries + TriggerMid(1)];
    n_periods = 4;
    n_units = length(valid_spikes);

    fprintf('  Units: %d, Periods: %d, Channel: %d\n', n_units, n_periods, bestChannel);

    % Compute amplitude correlation
    session_results = compute_amplitude_correlation(NeuralTime, LFP, valid_spikes, period_boundaries, ...
                                                     n_periods, n_units, Fs, config);

    session_results.session_id = n_valid_reward;
    session_results.filename = allfiles_reward(sess_idx).name;
    session_results.session_type = 'RewardSeeking';
    session_results.best_channel = bestChannel;

    % Save
    [~, base_filename, ~] = fileparts(allfiles_reward(sess_idx).name);
    save_filename = fullfile(RewardSeekingPath, sprintf('%s_ampcorr.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');

    fprintf('  ✓ Complete (%.1f sec)\n', toc);
end

fprintf('\n✓ Processed %d reward sessions\n', n_valid_reward);

%% ========================================================================
%  COMPLETION
%  ========================================================================

fprintf('\n========================================\n');
fprintf('AMPLITUDE CORRELATION ANALYSIS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Sessions processed:\n');
fprintf('  Aversive: %d\n', n_valid_aversive);
fprintf('  Reward: %d\n', n_valid_reward);
fprintf('\nResults saved to:\n');
fprintf('  %s\n', RewardAversivePath);
fprintf('  %s\n', RewardSeekingPath);
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function session_results = compute_amplitude_correlation(NeuralTime, LFP, valid_spikes, ...
                                                          period_boundaries, n_periods, n_units, Fs, config)
% Compute amplitude-spike rate correlation and mutual information
% OPTIMIZED VERSION: Pre-allocated arrays, vectorized operations

    n_bands = size(config.frequency_bands, 1);

    % Estimate maximum number of results (pre-allocate)
    max_results = n_bands * n_units * n_periods;

    % Pre-allocate arrays (MAJOR SPEEDUP: avoid dynamic growth)
    amp_data = struct();
    amp_data.unit = zeros(max_results, 1);
    amp_data.period = zeros(max_results, 1);
    amp_data.band_name = cell(max_results, 1);
    amp_data.freq_low = zeros(max_results, 1);
    amp_data.freq_high = zeros(max_results, 1);
    amp_data.pearson_r = zeros(max_results, 1);
    amp_data.pearson_p = zeros(max_results, 1);
    amp_data.mutual_info = zeros(max_results, 1);
    amp_data.n_bins = zeros(max_results, 1);

    result_idx = 0;  % Counter for actual results

    fprintf('  Computing amplitude correlations for %d bands...\n', n_bands);

    % Pre-compute time mapping (avoid repeated interp1 calls)
    t0 = NeuralTime(1);
    dt = median(diff(NeuralTime));

    % Process each frequency band
    % NOTE: To enable parallel processing, change 'for' to 'parfor' below
    % and ensure Parallel Computing Toolbox is available
    for band_idx = 1:n_bands
        band_name = config.frequency_bands{band_idx, 1};
        band_range = config.frequency_bands{band_idx, 2};

        % Filter LFP and extract amplitude envelope
        LFP_filtered = filter_LFP_for_amplitude(LFP, band_range, Fs);
        amplitude_envelope = abs(hilbert(LFP_filtered));
        clear LFP_filtered;  % Free memory immediately

        % Process each unit
        for unit_idx = 1:n_units
            spike_times = valid_spikes{unit_idx};
            if isempty(spike_times), continue; end

            % Process each period
            for period_idx = 1:n_periods
                period_start = period_boundaries(period_idx);
                period_end = period_boundaries(period_idx + 1);

                % Create time bins
                bin_edges = period_start:config.bin_size:period_end;
                n_bins = length(bin_edges) - 1;

                if n_bins < config.min_bins
                    continue;
                end

                % Bin spike rate
                spike_counts = histcounts(spike_times, bin_edges);
                spike_rate = spike_counts / config.bin_size;  % Hz

                % Sample amplitude at bin centers (OPTIMIZED: use index math)
                bin_centers = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
                amp_indices = round((bin_centers - t0) / dt) + 1;

                % Clip indices to valid range
                valid_mask = amp_indices > 0 & amp_indices <= length(amplitude_envelope);
                amp_indices = amp_indices(valid_mask);
                spike_rate_clipped = spike_rate(valid_mask);

                if length(amp_indices) < config.min_bins
                    continue;
                end

                amp_values = amplitude_envelope(amp_indices);

                % Compute Pearson correlation (linear)
                [r, p] = corr(amp_values(:), spike_rate_clipped(:));

                % Compute mutual information (nonlinear) - OPTIMIZED VERSION
                mi = compute_mutual_information_fast(amp_values, spike_rate_clipped, config.mi_bins);

                % Store results (pre-allocated, much faster)
                result_idx = result_idx + 1;
                amp_data.unit(result_idx) = unit_idx;
                amp_data.period(result_idx) = period_idx;
                amp_data.band_name{result_idx} = band_name;
                amp_data.freq_low(result_idx) = band_range(1);
                amp_data.freq_high(result_idx) = band_range(2);
                amp_data.pearson_r(result_idx) = r;
                amp_data.pearson_p(result_idx) = p;
                amp_data.mutual_info(result_idx) = mi;
                amp_data.n_bins(result_idx) = length(amp_values);
            end
        end

        if mod(band_idx, 5) == 0
            fprintf('    Processed %d/%d bands\n', band_idx, n_bands);
        end
    end

    % Trim pre-allocated arrays to actual size
    amp_data.unit = amp_data.unit(1:result_idx);
    amp_data.period = amp_data.period(1:result_idx);
    amp_data.band_name = amp_data.band_name(1:result_idx);
    amp_data.freq_low = amp_data.freq_low(1:result_idx);
    amp_data.freq_high = amp_data.freq_high(1:result_idx);
    amp_data.pearson_r = amp_data.pearson_r(1:result_idx);
    amp_data.pearson_p = amp_data.pearson_p(1:result_idx);
    amp_data.mutual_info = amp_data.mutual_info(1:result_idx);
    amp_data.n_bins = amp_data.n_bins(1:result_idx);

    % Convert to table
    session_results.data = struct2table(amp_data);
    session_results.data.Properties.VariableNames{'unit'} = 'Unit';
    session_results.data.Properties.VariableNames{'period'} = 'Period';
    session_results.data.Properties.VariableNames{'band_name'} = 'Band';
    session_results.data.Properties.VariableNames{'freq_low'} = 'Freq_Low_Hz';
    session_results.data.Properties.VariableNames{'freq_high'} = 'Freq_High_Hz';
    session_results.data.Properties.VariableNames{'pearson_r'} = 'Pearson_R';
    session_results.data.Properties.VariableNames{'pearson_p'} = 'Pearson_P';
    session_results.data.Properties.VariableNames{'mutual_info'} = 'Mutual_Info';
    session_results.data.Properties.VariableNames{'n_bins'} = 'N_bins';

    session_results.Fs = Fs;
end

function LFP_filtered = filter_LFP_for_amplitude(LFP, band_range, Fs)
% Filter LFP for amplitude extraction (same as PPC)
    low_freq = band_range(1);
    high_freq = band_range(2);

    if low_freq < 1
        LFP_detrend = detrend(LFP);
        LFP_demean = LFP_detrend - mean(LFP_detrend);
        LFP_filtered = lowpass(LFP_demean, high_freq, Fs, 'ImpulseResponse', 'fir', 'Steepness', 0.85);
    else
        LFP_filtered = bandpass(LFP, band_range, Fs, 'ImpulseResponse', 'fir', 'Steepness', 0.85);
    end
end

function MI = compute_mutual_information_fast(X, Y, n_bins)
% Compute mutual information between two continuous variables
% OPTIMIZED VERSION: Uses accumarray instead of loops
% MI = H(X) + H(Y) - H(X,Y) where H is entropy

    % Discretize into bins
    X_edges = linspace(min(X), max(X), n_bins+1);
    Y_edges = linspace(min(Y), max(Y), n_bins+1);

    X_discrete = discretize(X, X_edges);
    Y_discrete = discretize(Y, Y_edges);

    % Remove NaN values
    valid = ~isnan(X_discrete(:)) & ~isnan(Y_discrete(:));
    X_discrete = X_discrete(valid);
    Y_discrete = Y_discrete(valid);

    if length(X_discrete) < 10
        MI = NaN;
        return;
    end

    % Compute joint probability using accumarray (MUCH FASTER than loops)
    N = length(X_discrete);
    subs = [X_discrete(:), Y_discrete(:)];
    p_xy = accumarray(subs, 1, [n_bins, n_bins]) / N;

    % Compute marginal probabilities
    p_x = sum(p_xy, 2);
    p_y = sum(p_xy, 1);

    % Compute MI using vectorized operations (FASTER)
    % Only compute for non-zero joint probabilities
    [i_idx, j_idx] = find(p_xy > 0);
    MI = 0;
    for k = 1:length(i_idx)
        i = i_idx(k);
        j = j_idx(k);
        MI = MI + p_xy(i,j) * log2(p_xy(i,j) / (p_x(i) * p_y(j)));
    end
end
