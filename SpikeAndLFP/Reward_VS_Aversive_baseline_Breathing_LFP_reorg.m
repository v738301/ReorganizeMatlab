%% Initialize environment
clear all;
% close all;

%% Configuration Parameters
% Analysis parameters
config.bp_range = [1 300];     % Bandpass filter range
config.minFR = 0.5;            % Minimum firing rate (Hz) for valid units
config.min_duration = 2;       % Minimum duration in seconds for breathing events

% Breathing frequency bands
config.f_slow = [0.5, 3];      % slow breathing (0.5-3 Hz)
config.f_mid = [3, 6];         % medium breathing (3-6 Hz)
config.f_fast = [6, 15];       % fast breathing (6-15 Hz)

% Spectral analysis parameters
config.window_length_seconds = 2;  % Length of sliding window in seconds
config.overlap_percentage = 50;    % Overlap between windows (%)

% Behavioral state labels
config.behavior_labels = {'High Speed', 'Non-Reward Corner', 'Reward Corner (No Port)', ...
    'At Reward Port', 'Center Area', 'Rearing', 'Goal-Directed Movement'};

% Breathing type labels
config.breathing_labels = {'Slow (0.5-3Hz)', 'Mid (3-6Hz)', 'Fast (6-15Hz)'};

% dataset path
DataSetsPath = '/Volumes/My980Pro/reorganize/SpikeAndLFP/DataSet';

% Create save directories for individual sessions
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_Individual_Sessions');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_Individual_Sessions');
if ~exist(RewardSeekingPath, 'dir')
    mkdir(RewardSeekingPath);
end
if ~exist(RewardAversivePath, 'dir')
    mkdir(RewardAversivePath);
end

%% Load Sorting Parameters
[T_sorted] = loadSortingParameters();

%% First start from reward seeking sessions
numofsession = 2;
folderpath = "/Volumes/Expansion/Data/Struct_spike";
[allfiles, folderpath, num_sessions] = selectFilesWithAnimalIDFiltering(folderpath,numofsession,'2025*RewardSeeking*.mat');

%% Process Each Session with Behavioral Analysis
for sessionID = 1:num_sessions
    fprintf('\n==== Processing session %d/%d: %s ====\n', sessionID, num_sessions, allfiles(sessionID).name);

    %% Load and preprocess session data
    Timelimits = 'No';
    [NeuralTime, AdjustedXYZ, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, WP1ON, WP2ON, ...
        AversiveSound, sessionLabels, valid_spikes, Fs, TriggerMid] = loadAndPrepareSessionData(allfiles(sessionID), T_sorted, Timelimits);

    %% Preprocess signals
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);

    bestChannel = findBestLFPChannel(filtered_data, Fs);
    % Note: For this simplified version, we use channel 1 for LFP and channel 32 for breathin
    LFP = filtered_data(:, bestChannel);

    breathing = filtered_data(:, 32);
    [dominant_freq_full, is_slow_freq, is_mid_freq, is_fast_freq, slow_events_start, ...
        slow_events_end, mid_events_start, mid_events_end, fast_events_start, fast_events_end] = ...
        identifyBreathingFrequencyBands(breathing, NeuralTime, Fs, ...
        config.f_slow, config.f_mid, config.f_fast, config.min_duration);

    %% Identify behavioral states
    reward_locations = [];
    IR1_indices = find(IR1ON);
    IR2_indices = find(IR2ON);
    if length(IR1_indices) > 5
        reward_locations = [reward_locations; mean(AdjustedXYZ(IR1_indices, 1:2), 1)];
    end
    if length(IR2_indices) > 5
        reward_locations = [reward_locations; mean(AdjustedXYZ(IR2_indices, 1:2), 1)];
    end

    % Create behavioral matrix
    behavioral_matrix = create_behavioral_matrix(AdjustedXYZ, AdjustedXYZ_speed, reward_locations, IR1ON, IR2ON, Fs);

    % Column definitions:
    % 1: High speed (> 50)
    % 2: In non-reward corner
    % 3: In reward corner but not at port
    % 4: At reward port (IR beam break)
    % 5: In center area
    % 6: Rearing
    % 7: Goal-directed movement

    %% Calculate behavioral metrics
    behavioral_matrix_full = [behavioral_matrix, dominant_freq_full];

    %% Multi-Band Breathing-Gamma Coupling Analysis
    fprintf('Computing time-resolved breathing-gamma coupling for specific bands...\n');

    % Define specific breathing and gamma frequency bands to analyze
    breathing_bands = [1, 2, 3, 4, 5, 6, 7, 8]; % Hz
    gamma_bands = [80, 90, 100, 110, 120]; % Hz

    % Initialize storage for multi-band coupling results
    coupling_results_multiband = struct();
    coupling_results_multiband.breathing_bands = breathing_bands;
    coupling_results_multiband.gamma_bands = gamma_bands;
    coupling_results_multiband.band_results = cell(length(breathing_bands), length(gamma_bands));

    % Process each combination of breathing and gamma frequency bands
    for breath_idx = 1:length(breathing_bands)
        breath_freq = breathing_bands(breath_idx);

        for gamma_idx = 1:length(gamma_bands)
            gamma_freq = gamma_bands(gamma_idx);
            fprintf('  Processing breathing band: %.1f Hz with gamma band: %.1f Hz...\n', breath_freq, gamma_freq);

            % Create narrow bands around the target frequencies
            breathing_band = [breath_freq - 0.5, breath_freq + 0.5]; % ±0.5 Hz around target
            gamma_band = [gamma_freq - 5, gamma_freq + 5]; % ±5 Hz around target

            % Analyze coupling for this specific band combination
            try
                tic;
                [band_coupling_results] = analyzeBreathingThetaLFPGammaCoupling(...
                    breathing, LFP, NeuralTime, Fs, ...
                    'breathing_theta_band', breathing_band, ...
                    'lfp_gamma_band', gamma_band, ...
                    'window_size', 10, ...
                    'step_size', 1, ...
                    'n_phase_bins', 18, ...
                    'n_surrogates', 0, ...
                    'plot_results', false, ... % Don't plot individual bands
                    'figure_name', sprintf('Session %d - %.1fHz Breathing-%.1fHz Gamma Coupling', sessionID, breath_freq, gamma_freq));

                % Store results for this band combination
                coupling_results_multiband.band_results{breath_idx, gamma_idx} = band_coupling_results;
                coupling_results_multiband.band_results{breath_idx, gamma_idx}.target_breathing_frequency = breath_freq;
                coupling_results_multiband.band_results{breath_idx, gamma_idx}.target_gamma_frequency = gamma_freq;
                coupling_results_multiband.band_results{breath_idx, gamma_idx}.breathing_band_used = breathing_band;
                coupling_results_multiband.band_results{breath_idx, gamma_idx}.gamma_band_used = gamma_band;

                endTime = toc;
                fprintf('    %.1f Hz breathing - %.1f Hz gamma completed successfully, and spend %.1f secs \n', breath_freq, gamma_freq, endTime);

            catch ME
                fprintf('    Warning: Failed to process %.1f Hz breathing - %.1f Hz gamma - %s\n', breath_freq, gamma_freq, ME.message);
                coupling_results_multiband.band_results{breath_idx, gamma_idx} = [];
            end
        end
    end

    % Compute summary statistics across all bands
    fprintf('Computing summary statistics across frequency bands...\n');

    % Collect valid results
    valid_bands = ~cellfun(@isempty, coupling_results_multiband.band_results);
    valid_indices = find(valid_bands);

    if any(valid_bands(:))
        % Get common time vector (assuming all bands use same windowing)
        [first_breath_idx, first_gamma_idx] = ind2sub(size(coupling_results_multiband.band_results), valid_indices(1));
        reference_result = coupling_results_multiband.band_results{first_breath_idx, first_gamma_idx};
        common_time = reference_result.window_times;
        n_windows = length(common_time);

        % Initialize summary matrices
        all_MI_values = zeros(length(breathing_bands), length(gamma_bands), n_windows);
        all_significant = false(length(breathing_bands), length(gamma_bands), n_windows);
        all_coherence = zeros(length(breathing_bands), length(gamma_bands), n_windows);
        all_breathing_power = zeros(length(breathing_bands), length(gamma_bands), n_windows);
        all_gamma_power = zeros(length(breathing_bands), length(gamma_bands), n_windows);

        % Collect data from each valid band combination
        for idx = 1:length(valid_indices)
            [breath_idx, gamma_idx] = ind2sub(size(coupling_results_multiband.band_results), valid_indices(idx));
            result = coupling_results_multiband.band_results{breath_idx, gamma_idx};

            if length(result.window_times) == n_windows
                all_MI_values(breath_idx, gamma_idx, :) = result.MI_values;
                all_significant(breath_idx, gamma_idx, :) = result.p_values < 0.05;
                all_coherence(breath_idx, gamma_idx, :) = result.breathing_lfp_coherence;
                all_breathing_power(breath_idx, gamma_idx, :) = result.breathing_theta_power;
                all_gamma_power(breath_idx, gamma_idx, :) = result.lfp_gamma_power;
            else
                fprintf('    Warning: Band %.1f Hz breathing - %.1f Hz gamma has different time resolution\n', ...
                    breathing_bands(breath_idx), gamma_bands(gamma_idx));
            end
        end

        % Calculate summary statistics
        coupling_results_multiband.summary = struct();
        coupling_results_multiband.summary.window_times = common_time;
        coupling_results_multiband.summary.all_MI_values = all_MI_values;
        coupling_results_multiband.summary.all_significant = all_significant;
        coupling_results_multiband.summary.all_coherence = all_coherence;
        coupling_results_multiband.summary.all_breathing_power = all_breathing_power;
        coupling_results_multiband.summary.all_gamma_power = all_gamma_power;

        % Peak coupling analysis - find max MI across time for each band combination
        max_MI_per_combo = squeeze(max(all_MI_values, [], 3));
        [max_MI_time_idx_3d] = reshape(1:numel(all_MI_values), size(all_MI_values));
        [~, linear_max_idx] = max(all_MI_values, [], 3);

        coupling_results_multiband.summary.peak_MI_per_combo = max_MI_per_combo;

        % Overall best coupling across all combinations
        [max_MI_per_breath, best_gamma_per_breath] = max(max_MI_per_combo, [], 2);
        [best_MI, best_breath_idx] = max(max_MI_per_breath);

        if ~isempty(best_breath_idx) && best_MI > 0
            best_gamma_idx = best_gamma_per_breath(best_breath_idx);
            coupling_results_multiband.summary.best_breathing_frequency = breathing_bands(best_breath_idx);
            coupling_results_multiband.summary.best_gamma_frequency = gamma_bands(best_gamma_idx);
            coupling_results_multiband.summary.best_MI = best_MI;

            % Find the time of best coupling
            best_result = coupling_results_multiband.band_results{best_breath_idx, best_gamma_idx};
            [~, best_time_idx] = max(best_result.MI_values);
            coupling_results_multiband.summary.best_time = common_time(best_time_idx);
        end

        % Mean coupling strength over time across all valid combinations
        valid_MI = all_MI_values;
        valid_MI(~valid_bands) = NaN;
        coupling_results_multiband.summary.mean_MI_over_time = squeeze(mean(mean(valid_MI, 1, 'omitnan'), 2, 'omitnan'));
        coupling_results_multiband.summary.max_MI_over_time = squeeze(max(max(valid_MI, [], 1), [], 2));

        % Percentage of significant windows per band combination
        coupling_results_multiband.summary.percent_significant_per_combo = squeeze(mean(all_significant, 3)) * 100;

        fprintf('  Multi-band analysis completed\n');
        if isfield(coupling_results_multiband.summary, 'best_breathing_frequency')
            fprintf('  Best coupling: %.1f Hz breathing - %.1f Hz gamma at t=%.1fs (MI=%.4f)\n', ...
                coupling_results_multiband.summary.best_breathing_frequency, ...
                coupling_results_multiband.summary.best_gamma_frequency, ...
                coupling_results_multiband.summary.best_time, ...
                coupling_results_multiband.summary.best_MI);
        end
    else
        fprintf('  Warning: No valid coupling results obtained\n');
        coupling_results_multiband.summary = [];
    end

    fprintf('Multi-band breathing-gamma coupling analysis completed!\n\n');

    %% collect all results
    session_results = calculate_behavioral_metrics(behavioral_matrix, AdjustedXYZ_speed, NeuralTime, Fs);
    session_results.session_id = sessionID;
    session_results.filename = allfiles(sessionID).name;
    session_results.NeuralTime = NeuralTime;
    session_results.behavioral_matrix_full = behavioral_matrix_full;
    session_results.reward_locations = reward_locations;
    session_results.coupling_results_multiband = coupling_results_multiband;
    session_results.Speed = AdjustedXYZ_speed;
    session_results.TriggerMid = TriggerMid;

    %% Save individual session result
    % Create filename based on original data file
    [~, base_filename, ~] = fileparts(allfiles(sessionID).name);
    save_filename = fullfile(RewardSeekingPath, sprintf('%s_coupling_analysis.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');
    fprintf('Session %d results saved to: %s\n', sessionID, save_filename);
    
end

fprintf('\n==== All RewardSeeking sessions completed ====\n\n');

%% Reward aversive sessions
numofsession = 2;
folderpath = "/Volumes/Expansion/Data/Struct_spike";
[allfiles, folderpath, num_sessions] = selectFilesWithAnimalIDFiltering(folderpath,numofsession,'2025*RewardAversive*.mat');

%% Process Each Reward-Aversive Session
for sessionID = 1:num_sessions
    fprintf('\n==== Processing reward-aversive session %d/%d: %s ====\n', sessionID, num_sessions, allfiles(sessionID).name);

    %% Load and preprocess session data
    Timelimits = 'No';
    [NeuralTime, AdjustedXYZ, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, WP1ON, WP2ON, ...
        AversiveSound, sessionLabels, valid_spikes, Fs, TriggerMid] = loadAndPrepareSessionData(allfiles(sessionID), T_sorted, Timelimits);

    %% Preprocess signals
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);

    bestChannel = findBestLFPChannel(filtered_data, Fs);
    % Note: For this simplified version, we use channel 1 for LFP and channel 32 for breathing
    LFP = filtered_data(:, bestChannel);

    breathing = filtered_data(:, 32);
    [dominant_freq_full, is_slow_freq, is_mid_freq, is_fast_freq, slow_events_start, ...
        slow_events_end, mid_events_start, mid_events_end, fast_events_start, fast_events_end] = ...
        identifyBreathingFrequencyBands(breathing, NeuralTime, Fs, ...
        config.f_slow, config.f_mid, config.f_fast, config.min_duration);

    %% Find first aversive sound time
    first_aversive_time = find_first_aversive_sound_time(AversiveSound, NeuralTime);
    all_aversive_time = find_all_aversive_sound_time(AversiveSound, NeuralTime);

    %% Segment data into before and after periods
    [before_indices, after_indices] = segment_data_by_aversive_sound(NeuralTime, first_aversive_time);

    %% Identify behavioral states
    reward_locations = [];
    IR1_indices = find(IR1ON);
    IR2_indices = find(IR2ON);
    if length(IR1_indices) > 5
        reward_locations = [reward_locations; mean(AdjustedXYZ(IR1_indices, 1:2), 1)];
    end
    if length(IR2_indices) > 5
        reward_locations = [reward_locations; mean(AdjustedXYZ(IR2_indices, 1:2), 1)];
    end

    % Create behavioral matrix for entire session
    behavioral_matrix_full = create_behavioral_matrix(AdjustedXYZ, AdjustedXYZ_speed, reward_locations, IR1ON, IR2ON, Fs);

    %% Calculate metrics for BEFORE period
    before_metrics = calculate_behavioral_metrics_for_period(...
        behavioral_matrix_full(before_indices, :), ...
        AdjustedXYZ_speed(before_indices), ...
        NeuralTime(before_indices), Fs, 'BEFORE');

    %% Calculate metrics for AFTER period
    after_metrics = calculate_behavioral_metrics_for_period(...
        behavioral_matrix_full(after_indices, :), ...
        AdjustedXYZ_speed(after_indices), ...
        NeuralTime(after_indices), Fs, 'AFTER');

    %% Store comprehensive session results
    behavioral_matrix_full = [behavioral_matrix_full, dominant_freq_full];

    %% Multi-Band Breathing-Gamma Coupling Analysis
    fprintf('Computing time-resolved breathing-gamma coupling for specific bands...\n');

    % Define specific breathing and gamma frequency bands to analyze
    breathing_bands = [1, 2, 3, 4, 5, 6, 7, 8]; % Hz
    gamma_bands = [80, 90, 100, 110, 120]; % Hz

    % Initialize storage for multi-band coupling results
    coupling_results_multiband = struct();
    coupling_results_multiband.breathing_bands = breathing_bands;
    coupling_results_multiband.gamma_bands = gamma_bands;
    coupling_results_multiband.band_results = cell(length(breathing_bands), length(gamma_bands));

    % Process each combination of breathing and gamma frequency bands
    for breath_idx = 1:length(breathing_bands)
        breath_freq = breathing_bands(breath_idx);

        for gamma_idx = 1:length(gamma_bands)
            gamma_freq = gamma_bands(gamma_idx);
            fprintf('  Processing breathing band: %.1f Hz with gamma band: %.1f Hz...\n', breath_freq, gamma_freq);

            % Create narrow bands around the target frequencies
            breathing_band = [breath_freq - 0.5, breath_freq + 0.5]; % ±0.5 Hz around target
            gamma_band = [gamma_freq - 5, gamma_freq + 5]; % ±5 Hz around target

            % Analyze coupling for this specific band combination
            try
                tic;
                [band_coupling_results] = analyzeBreathingThetaLFPGammaCoupling(...
                    breathing, LFP, NeuralTime, Fs, ...
                    'breathing_theta_band', breathing_band, ...
                    'lfp_gamma_band', gamma_band, ...
                    'window_size', 10, ...
                    'step_size', 1, ...
                    'n_phase_bins', 18, ...
                    'n_surrogates', 0, ...
                    'plot_results', false, ... % Don't plot individual bands
                    'figure_name', sprintf('Session %d - %.1fHz Breathing-%.1fHz Gamma Coupling', sessionID, breath_freq, gamma_freq));

                % Store results for this band combination
                coupling_results_multiband.band_results{breath_idx, gamma_idx} = band_coupling_results;
                coupling_results_multiband.band_results{breath_idx, gamma_idx}.target_breathing_frequency = breath_freq;
                coupling_results_multiband.band_results{breath_idx, gamma_idx}.target_gamma_frequency = gamma_freq;
                coupling_results_multiband.band_results{breath_idx, gamma_idx}.breathing_band_used = breathing_band;
                coupling_results_multiband.band_results{breath_idx, gamma_idx}.gamma_band_used = gamma_band;

                endTime = toc;
                fprintf('    %.1f Hz breathing - %.1f Hz gamma completed successfully, and spend %.1f secs \n', breath_freq, gamma_freq, endTime);

            catch ME
                fprintf('    Warning: Failed to process %.1f Hz breathing - %.1f Hz gamma - %s\n', breath_freq, gamma_freq, ME.message);
                coupling_results_multiband.band_results{breath_idx, gamma_idx} = [];
            end
        end
    end

    % Compute summary statistics across all bands
    fprintf('Computing summary statistics across frequency bands...\n');

    % Collect valid results
    valid_bands = ~cellfun(@isempty, coupling_results_multiband.band_results);
    valid_indices = find(valid_bands);

    if any(valid_bands(:))
        % Get common time vector (assuming all bands use same windowing)
        [first_breath_idx, first_gamma_idx] = ind2sub(size(coupling_results_multiband.band_results), valid_indices(1));
        reference_result = coupling_results_multiband.band_results{first_breath_idx, first_gamma_idx};
        common_time = reference_result.window_times;
        n_windows = length(common_time);

        % Initialize summary matrices
        all_MI_values = zeros(length(breathing_bands), length(gamma_bands), n_windows);
        all_significant = false(length(breathing_bands), length(gamma_bands), n_windows);
        all_coherence = zeros(length(breathing_bands), length(gamma_bands), n_windows);
        all_breathing_power = zeros(length(breathing_bands), length(gamma_bands), n_windows);
        all_gamma_power = zeros(length(breathing_bands), length(gamma_bands), n_windows);

        % Collect data from each valid band combination
        for idx = 1:length(valid_indices)
            [breath_idx, gamma_idx] = ind2sub(size(coupling_results_multiband.band_results), valid_indices(idx));
            result = coupling_results_multiband.band_results{breath_idx, gamma_idx};

            if length(result.window_times) == n_windows
                all_MI_values(breath_idx, gamma_idx, :) = result.MI_values;
                all_significant(breath_idx, gamma_idx, :) = result.p_values < 0.05;
                all_coherence(breath_idx, gamma_idx, :) = result.breathing_lfp_coherence;
                all_breathing_power(breath_idx, gamma_idx, :) = result.breathing_theta_power;
                all_gamma_power(breath_idx, gamma_idx, :) = result.lfp_gamma_power;
            else
                fprintf('    Warning: Band %.1f Hz breathing - %.1f Hz gamma has different time resolution\n', ...
                    breathing_bands(breath_idx), gamma_bands(gamma_idx));
            end
        end

        % Calculate summary statistics
        coupling_results_multiband.summary = struct();
        coupling_results_multiband.summary.window_times = common_time;
        coupling_results_multiband.summary.all_MI_values = all_MI_values;
        coupling_results_multiband.summary.all_significant = all_significant;
        coupling_results_multiband.summary.all_coherence = all_coherence;
        coupling_results_multiband.summary.all_breathing_power = all_breathing_power;
        coupling_results_multiband.summary.all_gamma_power = all_gamma_power;

        % Peak coupling analysis - find max MI across time for each band combination
        max_MI_per_combo = squeeze(max(all_MI_values, [], 3));
        [max_MI_time_idx_3d] = reshape(1:numel(all_MI_values), size(all_MI_values));
        [~, linear_max_idx] = max(all_MI_values, [], 3);

        coupling_results_multiband.summary.peak_MI_per_combo = max_MI_per_combo;

        % Overall best coupling across all combinations
        [max_MI_per_breath, best_gamma_per_breath] = max(max_MI_per_combo, [], 2);
        [best_MI, best_breath_idx] = max(max_MI_per_breath);

        if ~isempty(best_breath_idx) && best_MI > 0
            best_gamma_idx = best_gamma_per_breath(best_breath_idx);
            coupling_results_multiband.summary.best_breathing_frequency = breathing_bands(best_breath_idx);
            coupling_results_multiband.summary.best_gamma_frequency = gamma_bands(best_gamma_idx);
            coupling_results_multiband.summary.best_MI = best_MI;

            % Find the time of best coupling
            best_result = coupling_results_multiband.band_results{best_breath_idx, best_gamma_idx};
            [~, best_time_idx] = max(best_result.MI_values);
            coupling_results_multiband.summary.best_time = common_time(best_time_idx);
        end

        % Mean coupling strength over time across all valid combinations
        valid_MI = all_MI_values;
        valid_MI(~valid_bands) = NaN;
        coupling_results_multiband.summary.mean_MI_over_time = squeeze(mean(mean(valid_MI, 1, 'omitnan'), 2, 'omitnan'));
        coupling_results_multiband.summary.max_MI_over_time = squeeze(max(max(valid_MI, [], 1), [], 2));

        % Percentage of significant windows per band combination
        coupling_results_multiband.summary.percent_significant_per_combo = squeeze(mean(all_significant, 3)) * 100;

        fprintf('  Multi-band analysis completed\n');
        if isfield(coupling_results_multiband.summary, 'best_breathing_frequency')
            fprintf('  Best coupling: %.1f Hz breathing - %.1f Hz gamma at t=%.1fs (MI=%.4f)\n', ...
                coupling_results_multiband.summary.best_breathing_frequency, ...
                coupling_results_multiband.summary.best_gamma_frequency, ...
                coupling_results_multiband.summary.best_time, ...
                coupling_results_multiband.summary.best_MI);
        end
    else
        fprintf('  Warning: No valid coupling results obtained\n');
        coupling_results_multiband.summary = [];
    end

    fprintf('Multi-band breathing-gamma coupling analysis completed!\n\n');

    %%
    session_results = struct();
    session_results.session_id = sessionID;
    session_results.filename = allfiles(sessionID).name;
    session_results.NeuralTime = NeuralTime;
    session_results.first_aversive_time = first_aversive_time;
    session_results.first_aversive_time_min = (first_aversive_time - NeuralTime(1))/60;
    session_results.all_aversive_time = all_aversive_time;
    session_results.before_metrics = before_metrics;
    session_results.after_metrics = after_metrics;
    session_results.behavioral_matrix_full = behavioral_matrix_full;
    session_results.before_indices = before_indices;
    session_results.after_indices = after_indices;
    session_results.reward_locations = reward_locations;
    session_results.coupling_results_multiband = coupling_results_multiband;
    session_results.Speed = AdjustedXYZ_speed;
    session_results.TriggerMid = TriggerMid;

    %% Save individual session result
    % Create filename based on original data file
    [~, base_filename, ~] = fileparts(allfiles(sessionID).name);
    save_filename = fullfile(RewardAversivePath, sprintf('%s_coupling_analysis.mat', base_filename));
    save(save_filename, 'session_results', 'config', '-v7.3');
    fprintf('Session %d results saved to: %s\n', sessionID, save_filename);
end

fprintf('\n==== All RewardAversive sessions completed ====\n');

%% load data
DataSetsPath = '/Volumes/My980Pro/reorganize/SpikeAndLFP/DataSet';

% Define paths to individual session directories
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_Individual_Sessions');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_Individual_Sessions');

% Load config from any session file (assuming all have the same config)
reward_seeking_files = dir(fullfile(RewardSeekingPath, '*_coupling_analysis.mat'));
if ~isempty(reward_seeking_files)
    temp = load(fullfile(RewardSeekingPath, reward_seeking_files(1).name), 'config');
    config = temp.config;
else
    error('No reward seeking session files found');
end

% Load all reward seeking sessions
fprintf('Loading reward seeking sessions...\n');
reward_seeking_results = {};
for i = 1:length(reward_seeking_files)
    fprintf('  Loading %s...\n', reward_seeking_files(i).name);
    temp = load(fullfile(RewardSeekingPath, reward_seeking_files(i).name), 'session_results');
    reward_seeking_results{end+1} = temp.session_results;
    reward_seeking_results{end} = rmfield(reward_seeking_results{end},'coupling_results_multiband');  
end
fprintf('Loaded %d reward seeking sessions\n\n', length(reward_seeking_results));

% Load all reward aversive sessions
fprintf('Loading reward aversive sessions...\n');
reward_aversive_files = dir(fullfile(RewardAversivePath, '*_coupling_analysis.mat'));
reward_aversive_results = {};
for i = 1:length(reward_aversive_files)
    fprintf('  Loading %s...\n', reward_aversive_files(i).name);
    temp = load(fullfile(RewardAversivePath, reward_aversive_files(i).name), 'session_results');
    reward_aversive_results{end+1} = temp.session_results;
    reward_aversive_results{end} = rmfield(reward_aversive_results{end}, 'coupling_results_multiband');  
end
fprintf('Loaded %d reward aversive sessions\n\n', length(reward_aversive_results));

%% Load data - memory-efficient version with matfile references
DataSetsPath = '/Volumes/My980Pro/reorganize/SpikeAndLFP/DataSet';

% Define paths to individual session directories
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_Individual_Sessions');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_Individual_Sessions');

% Load config from any session file (assuming all have the same config)
reward_seeking_files = dir(fullfile(RewardSeekingPath, '*_coupling_analysis.mat'));
if ~isempty(reward_seeking_files)
    temp = load(fullfile(RewardSeekingPath, reward_seeking_files(1).name), 'config');
    config = temp.config;
    clear temp;
else
    error('No reward seeking session files found');
end

% Load only required fields from reward seeking sessions using matfile
fprintf('Loading reward seeking sessions (filename and behavioral_matrix only)...\n');
reward_seeking_results = cell(length(reward_seeking_files), 1);
for i = 1:length(reward_seeking_files)
    fprintf(' Loading %s...\n', reward_seeking_files(i).name);
    
    % Create memory reference to the file
    matObj = matfile(fullfile(RewardSeekingPath, reward_seeking_files(i).name));
    
    % Load the session_results structure
    session_data = matObj.session_results;
    
    % Extract only the specific fields we need
    reward_seeking_results{i} = struct();
    reward_seeking_results{i}.filename = session_data.filename;
    reward_seeking_results{i}.behavioral_matrix = session_data.behavioral_matrix_full;
    
    clear session_data;
end
fprintf('Loaded %d reward seeking sessions\n\n', length(reward_seeking_results));

% Load only required fields from reward aversive sessions using matfile
fprintf('Loading reward aversive sessions (filename and behavioral_matrix only)...\n');
reward_aversive_files = dir(fullfile(RewardAversivePath, '*_coupling_analysis.mat'));
reward_aversive_results = cell(length(reward_aversive_files), 1);
for i = 1:length(reward_aversive_files)
    fprintf(' Loading %s...\n', reward_aversive_files(i).name);
    
    % Create memory reference to the file
    matObj = matfile(fullfile(RewardAversivePath, reward_aversive_files(i).name));
    
    % Load the session_results structure
    session_data = matObj.session_results;
    
    % Extract only the specific fields we need
    reward_aversive_results{i} = struct();
    reward_aversive_results{i}.filename = session_data.filename;
    reward_aversive_results{i}.behavioral_matrix = session_data.behavioral_matrix_full;
    
    clear session_data;
end
fprintf('Loaded %d reward aversive sessions\n\n', length(reward_aversive_results));

%% analyze_breathing_across_conditions
analyze_breathing_across_conditions(reward_seeking_results, reward_aversive_results, config);

%% compare_baseline_before_after_aversive
compare_baseline_before_after_aversive(reward_seeking_results, reward_aversive_results)

%% plotting functions
function analyze_breathing_across_conditions(reward_seeking_results, reward_aversive_results, config)
% Analyze breathing rate distributions across behavioral states and conditions
% Compares: Baseline vs Before-Aversive vs After-Aversive
% For both 3 compound states and 7 individual states

fprintf('Starting breathing rate distribution analysis...\n');

%% Extract breathing data for all conditions
baseline_breathing = extract_baseline_breathing_data(reward_seeking_results);
before_breathing = extract_before_aversive_breathing_data(reward_aversive_results);
after_breathing = extract_after_aversive_breathing_data(reward_aversive_results);

fprintf('Extracted breathing data:\n');
fprintf('  Baseline sessions: %d\n', length(baseline_breathing.sessions));
fprintf('  Before-aversive periods: %d\n', length(before_breathing.sessions));
fprintf('  After-aversive periods: %d\n', length(after_breathing.sessions));

%% Create breathing distribution plots
create_breathing_distribution_plots(baseline_breathing, before_breathing, after_breathing, config);

%% Create breathing statistics summary
create_breathing_statistics_summary(baseline_breathing, before_breathing, after_breathing, config);

%%
fprintf('Breathing analysis completed.\n');
end

%% Extract baseline breathing data
function baseline_breathing = extract_baseline_breathing_data(reward_seeking_results)
n_sessions = length(reward_seeking_results);
baseline_breathing = struct();
baseline_breathing.sessions = {};

for i = 1:n_sessions
    session = reward_seeking_results{i};
    if isfield(session, 'behavioral_matrix_full') && size(session.behavioral_matrix_full, 2) >= 8
        % Breathing is in column 8 (added as dominant_freq_full)
        behavioral_matrix = session.behavioral_matrix_full;
        breathing_freq = behavioral_matrix(:, 8);  % Breathing frequency

        % Create session data structure
        session_data = struct();
        session_data.breathing_freq = breathing_freq;
        session_data.behavioral_states = behavioral_matrix(:, 1:7);
        session_data.filename = session.filename;

        % Calculate compound states
        [a,b,c] = group_behavioral_states(behavioral_matrix);
        session_data.compound_states = [a,b,c];

        baseline_breathing.sessions{end+1} = session_data;
    end
end
end

%% Extract before-aversive breathing data
function before_breathing = extract_before_aversive_breathing_data(reward_aversive_results)
valid_sessions = ~cellfun(@isempty, reward_aversive_results);
reward_aversive_results = reward_aversive_results(valid_sessions);

before_breathing = struct();
before_breathing.sessions = {};

for i = 1:length(reward_aversive_results)
    session = reward_aversive_results{i};
    if isfield(session, 'behavioral_matrix_full') && isfield(session, 'before_indices')
        behavioral_matrix = session.behavioral_matrix_full(session.before_indices, :);

        if size(behavioral_matrix, 2) >= 8
            breathing_freq = behavioral_matrix(:, 8);

            session_data = struct();
            session_data.breathing_freq = breathing_freq;
            session_data.behavioral_states = behavioral_matrix(:, 1:7);
            session_data.filename = session.filename;
            [a,b,c] = group_behavioral_states(behavioral_matrix);
            session_data.compound_states = [a,b,c];

            before_breathing.sessions{end+1} = session_data;
        end
    end
end
end

%% Extract after-aversive breathing data
function after_breathing = extract_after_aversive_breathing_data(reward_aversive_results)
valid_sessions = ~cellfun(@isempty, reward_aversive_results);
reward_aversive_results = reward_aversive_results(valid_sessions);

after_breathing = struct();
after_breathing.sessions = {};

for i = 1:length(reward_aversive_results)
    session = reward_aversive_results{i};
    if isfield(session, 'behavioral_matrix_full') && isfield(session, 'after_indices')
        behavioral_matrix = session.behavioral_matrix_full(session.after_indices, :);

        if size(behavioral_matrix, 2) >= 8
            breathing_freq = behavioral_matrix(:, 8);

            session_data = struct();
            session_data.breathing_freq = breathing_freq;
            session_data.behavioral_states = behavioral_matrix(:, 1:7);
            session_data.filename = session.filename;
            [a,b,c] = group_behavioral_states(behavioral_matrix);
            session_data.compound_states = [a,b,c];
            after_breathing.sessions{end+1} = session_data;
        end
    end
end
end


%% Create breathing distribution plots
function create_breathing_distribution_plots(baseline_breathing, before_breathing, after_breathing, config)

compound_labels = {'Reward Seeking', 'Idling', 'Rearing'};
condition_labels = {'Baseline', 'Before Aversive', 'After Aversive'};
colors = {[0.3 0.7 0.3], [0.2 0.6 0.8], [0.8 0.3 0.2]};

for state = 1:3
    subplot(2, 3, state);

    % Extract breathing data for this compound state
    baseline_data = extract_breathing_for_state(baseline_breathing, 'compound', state);
    before_data = extract_breathing_for_state(before_breathing, 'compound', state);
    after_data = extract_breathing_for_state(after_breathing, 'compound', state);

    % Create violin plot or histogram
    plot_breathing_distribution(baseline_data, before_data, after_data, ...
        compound_labels{state}, condition_labels, colors);
end

% Plot 4-6: Statistical comparisons
for state = 1:3
    subplot(2, 3, state+3);

    % Extract breathing data for this compound state
    baseline_data = extract_breathing_for_state(baseline_breathing, 'compound', state);
    before_data = extract_breathing_for_state(before_breathing, 'compound', state);
    after_data = extract_breathing_for_state(after_breathing, 'compound', state);

    % Create CDF plot
    plot_breathing_cdf(baseline_data, before_data, after_data, ...
        compound_labels{state}, condition_labels, colors);
end

sgtitle('Breathing Rate Distributions: 3 Compound Behavioral States', ...
    'FontSize', 16, 'FontWeight', 'bold');


%% Plot 2: 7 Individual States Breathing Distributions
fig2 = figure('Position', [100, 100, 2000, 1200]);
set(fig2, 'Color', 'white');

individual_labels = config.behavior_labels;

for state = 1:7
    subplot(2, 7, state);
    baseline_data = extract_breathing_for_state(baseline_breathing, 'individual', state);
    before_data = extract_breathing_for_state(before_breathing, 'individual', state);
    after_data = extract_breathing_for_state(after_breathing, 'individual', state);

    plot_breathing_distribution(baseline_data, before_data, after_data, ...
        individual_labels{state}, condition_labels, colors);
end

for state = 1:7
    subplot(2, 7, state+7);
    baseline_data = extract_breathing_for_state(baseline_breathing, 'individual', state);
    before_data = extract_breathing_for_state(before_breathing, 'individual', state);
    after_data = extract_breathing_for_state(after_breathing, 'individual', state);

    plot_breathing_cdf(baseline_data, before_data, after_data, ...
        individual_labels{state}, condition_labels, colors);
end

sgtitle('Breathing Rate Distributions: 7 Individual Behavioral States', ...
    'FontSize', 16, 'FontWeight', 'bold');

end

%% Extract breathing data for specific behavioral state
function breathing_data = extract_breathing_for_state(breathing_struct, state_type, state_idx)
breathing_data = [];

for i = 1:length(breathing_struct.sessions)
    session = breathing_struct.sessions{i};

    if strcmp(state_type, 'compound')
        state_mask = session.compound_states(:, state_idx);
    else  % individual
        state_mask = session.behavioral_states(:, state_idx);
    end

    % Get breathing frequencies during this behavioral state
    breathing_during_state = session.breathing_freq(state_mask==1);
    breathing_during_state = breathing_during_state(~isnan(breathing_during_state) & breathing_during_state > 0);

    breathing_data = [breathing_data; breathing_during_state];
end
end

%% Plot breathing distribution (histogram + overlay)
function plot_breathing_distribution(baseline_data, before_data, after_data, state_name, condition_labels, colors)

% Combine all data to determine common bins
all_data = [baseline_data; before_data; after_data];
if isempty(all_data)
    text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
    title(state_name, 'FontSize', 11, 'FontWeight', 'bold');
    return;
end

edges = linspace(min(all_data), max(all_data), 30);

% Plot histograms
if ~isempty(baseline_data)
    h1 = histogram(baseline_data, edges, 'Normalization', 'probability', ...
        'FaceColor', colors{1}, 'FaceAlpha', 0.6, 'EdgeColor', 'none');
    hold on;
end

if ~isempty(before_data)
    h2 = histogram(before_data, edges, 'Normalization', 'probability', ...
        'FaceColor', colors{2}, 'FaceAlpha', 0.6, 'EdgeColor', 'none');
    hold on;
end

if ~isempty(after_data)
    h3 = histogram(after_data, edges, 'Normalization', 'probability', ...
        'FaceColor', colors{3}, 'FaceAlpha', 0.6, 'EdgeColor', 'none');
    hold on;
end

% Add mean lines
if ~isempty(baseline_data)
    line([mean(baseline_data), mean(baseline_data)], ylim, ...
        'Color', colors{1}, 'LineWidth', 2, 'LineStyle', '--');
end
if ~isempty(before_data)
    line([mean(before_data), mean(before_data)], ylim, ...
        'Color', colors{2}, 'LineWidth', 2, 'LineStyle', '--');
end
if ~isempty(after_data)
    line([mean(after_data), mean(after_data)], ylim, ...
        'Color', colors{3}, 'LineWidth', 2, 'LineStyle', '--');
end

title(state_name, 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Breathing Rate (Hz)');
ylabel('Probability');
legend(condition_labels, 'Location', 'best', 'FontSize', 9);
box off;
end

%% Create breathing statistics summary
function create_breathing_statistics_summary(baseline_breathing, before_breathing, after_breathing, config)

% Print summary statistics to console
fprintf('\n=== BREATHING RATE ANALYSIS SUMMARY ===\n');

% Analyze compound states
compound_labels = {'Reward Seeking', 'Idling', 'Rearing'};

fprintf('\nCOMPOUND STATES BREATHING ANALYSIS:\n');
fprintf('===================================\n');

for state = 1:3
    fprintf('\n%s:\n', compound_labels{state});

    baseline_data = extract_breathing_for_state(baseline_breathing, 'compound', state);
    before_data = extract_breathing_for_state(before_breathing, 'compound', state);
    after_data = extract_breathing_for_state(after_breathing, 'compound', state);

    if ~isempty(baseline_data)
        fprintf('  Baseline: %.2f ± %.2f Hz (n=%d)\n', mean(baseline_data), std(baseline_data), length(baseline_data));
    end
    if ~isempty(before_data)
        fprintf('  Before Aversive: %.2f ± %.2f Hz (n=%d)\n', mean(before_data), std(before_data), length(before_data));
    end
    if ~isempty(after_data)
        fprintf('  After Aversive: %.2f ± %.2f Hz (n=%d)\n', mean(after_data), std(after_data), length(after_data));
    end

    % Statistical test
    if ~isempty(baseline_data) && ~isempty(before_data) && ~isempty(after_data)
        all_data = [baseline_data; before_data; after_data];
        groups = [ones(length(baseline_data), 1); 2*ones(length(before_data), 1); 3*ones(length(after_data), 1)];

        try
            p_val = kruskalwallis(all_data, groups, 'off');
            fprintf('  Statistical test: p = %.4f\n', p_val);
        catch
            fprintf('  Statistical test: Failed\n');
        end
    end
end

% Analyze breathing frequency bands
fprintf('\nBREATHING FREQUENCY BAND ANALYSIS:\n');
fprintf('==================================\n');

% Define breathing bands from config
slow_band = config.f_slow;
mid_band = config.f_mid;
fast_band = config.f_fast;

conditions = {baseline_breathing, before_breathing, after_breathing};
condition_names = {'Baseline', 'Before Aversive', 'After Aversive'};

for c = 1:3
    fprintf('\n%s:\n', condition_names{c});

    all_breathing = [];
    for i = 1:length(conditions{c}.sessions)
        session = conditions{c}.sessions{i};
        breathing = session.breathing_freq(~isnan(session.breathing_freq) & session.breathing_freq > 0);
        all_breathing = [all_breathing; breathing];
    end

    if ~isempty(all_breathing)
        slow_pct = sum(all_breathing >= slow_band(1) & all_breathing <= slow_band(2)) / length(all_breathing) * 100;
        mid_pct = sum(all_breathing >= mid_band(1) & all_breathing <= mid_band(2)) / length(all_breathing) * 100;
        fast_pct = sum(all_breathing >= fast_band(1) & all_breathing <= fast_band(2)) / length(all_breathing) * 100;

        fprintf('  Slow breathing (%.1f-%.1f Hz): %.1f%%\n', slow_band(1), slow_band(2), slow_pct);
        fprintf('  Mid breathing (%.1f-%.1f Hz): %.1f%%\n', mid_band(1), mid_band(2), mid_pct);
        fprintf('  Fast breathing (%.1f-%.1f Hz): %.1f%%\n', fast_band(1), fast_band(2), fast_pct);
        fprintf('  Overall mean: %.2f ± %.2f Hz\n', mean(all_breathing), std(all_breathing));
    end
end

fprintf('\n=== END OF BREATHING ANALYSIS ===\n');
end

%% Helper function for CDF plotting
function plot_breathing_cdf(baseline_data, before_data, after_data, state_name, condition_labels, colors)

% Check if we have data
all_data = [baseline_data; before_data; after_data];
if isempty(all_data)
    text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
    title(state_name, 'FontSize', 11, 'FontWeight', 'bold');
    return;
end

hold on;

% Plot CDF for each condition
if ~isempty(baseline_data)
    [f1, x1] = ecdf(baseline_data);
    plot(x1, f1, 'Color', colors{1}, 'LineWidth', 2, 'DisplayName', condition_labels{1});
end

if ~isempty(before_data)
    [f2, x2] = ecdf(before_data);
    plot(x2, f2, 'Color', colors{2}, 'LineWidth', 2, 'DisplayName', condition_labels{2});
end

if ~isempty(after_data)
    [f3, x3] = ecdf(after_data);
    plot(x3, f3, 'Color', colors{3}, 'LineWidth', 2, 'DisplayName', condition_labels{3});
end

%     Add median lines
if ~isempty(baseline_data)
    median1 = median(baseline_data);
    line([median1, median1], [0, 1], 'Color', colors{1}, 'LineWidth', 1, 'LineStyle', '--');
end
if ~isempty(before_data)
    median2 = median(before_data);
    line([median2, median2], [0, 1], 'Color', colors{2}, 'LineWidth', 1, 'LineStyle', '--');
end
if ~isempty(after_data)
    median3 = median(after_data);
    line([median3, median3], [0, 1], 'Color', colors{3}, 'LineWidth', 1, 'LineStyle', '--');
end

title(state_name, 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Breathing Rate (Hz)');
ylabel('Cumulative Probability');
box off;

if ~isempty(baseline_data) && ~isempty(before_data) && ~isempty(after_data)
    try
        % Original KS tests
        [~, p1, ks1] = kstest2(baseline_data, before_data);
        [~, p2, ks2] = kstest2(baseline_data, after_data);
        [~, p3, ks3] = kstest2(before_data, after_data);

        % Bootstrap for confidence intervals
        n_boot = 1000;
        sample_size = 1000; % Subsample size to avoid computational issues
        ks_boot = zeros(n_boot, 3);

        for i = 1:n_boot
            % Subsample data for bootstrap
            b1 = datasample(baseline_data, min(sample_size, length(baseline_data)));
            b2 = datasample(before_data, min(sample_size, length(before_data)));
            b3 = datasample(after_data, min(sample_size, length(after_data)));

            [~, ~, ks_boot(i,1)] = kstest2(b1, b2);
            [~, ~, ks_boot(i,2)] = kstest2(b1, b3);
            [~, ~, ks_boot(i,3)] = kstest2(b2, b3);
        end

        % Calculate confidence intervals
        ks_ci = prctile(ks_boot, [2.5, 97.5]);

        % Determine relationship with confidence
        baseline_before_mean = mean(ks_boot(:,1));
        baseline_after_mean = mean(ks_boot(:,2));

        % Check if confidence intervals overlap
        ci_overlap = (ks_ci(2,1) >= ks_ci(1,2)) && (ks_ci(1,1) <= ks_ci(2,2));

        if baseline_before_mean < baseline_after_mean && ~ci_overlap
            relationship = 'Before significantly closer to baseline';
        elseif baseline_after_mean < baseline_before_mean && ~ci_overlap
            relationship = 'After significantly closer to baseline';
        else
            relationship = 'No clear difference (CIs overlap)';
        end

        % Display results with bootstrap CIs
        text(0.05, 0.95, sprintf(['Bootstrap KS Statistics (n=%d):\n' ...
            'Base-Before: %.3f [%.3f-%.3f]\n' ...
            'Base-After: %.3f [%.3f-%.3f]\n' ...
            'Before-After: %.3f [%.3f-%.3f]\n' ...
            '%s'], ...
            n_boot, ...
            baseline_before_mean, ks_ci(1,1), ks_ci(2,1), ...
            baseline_after_mean, ks_ci(1,2), ks_ci(2,2), ...
            mean(ks_boot(:,3)), ks_ci(1,3), ks_ci(2,3), ...
            relationship), ...
            'Units', 'normalized', 'FontSize', 7, 'BackgroundColor', 'white');

    catch ME
        % Display error message
        text(0.05, 0.95, sprintf('KS test failed: %s', ME.message), ...
            'Units', 'normalized', 'FontSize', 8, 'BackgroundColor', 'yellow');
    end
end
end

function compare_baseline_before_after_aversive(reward_seeking_results, reward_aversive_results)
% Compare baseline reward-seeking behavior with before AND after aversive periods
% Analyzes 4 behavioral metrics across 3 compound states and 7 individual states
% Three-way comparison: Baseline vs Before-Aversive vs After-Aversive

fprintf('Starting comprehensive baseline vs before vs after aversive comparison...\n');

%% Data Preparation
% Extract baseline data (reward-seeking sessions)
baseline_data = extract_baseline_data(reward_seeking_results);

% Extract before-aversive data
before_aversive_data = extract_before_aversive_data(reward_aversive_results);

% Extract after-aversive data
after_aversive_data = extract_after_aversive_data(reward_aversive_results);

% Print sample information
fprintf('\nSAMPLE INFORMATION:\n');
fprintf('Baseline sessions (pure reward-seeking): %d sessions\n', baseline_data.n_sessions);
fprintf('Before-aversive periods: %d sessions\n', before_aversive_data.n_sessions);
fprintf('After-aversive periods: %d sessions\n', after_aversive_data.n_sessions);
fprintf('Note: Data from different subjects may influence comparisons\n\n');

%% Create Three-way Comparison Analysis
% Part 1: 3 Compound States Analysis
create_3_state_three_way_comparison(baseline_data, before_aversive_data, after_aversive_data);

% Part 2: 7 Individual States Analysis
create_7_state_three_way_comparison(baseline_data, before_aversive_data, after_aversive_data);

fprintf('Three-way comparison analysis completed.\n');
end

%% Extract baseline data from reward-seeking sessions
function baseline_data = extract_baseline_data(reward_seeking_results)
% Extract data from the reward-seeking session results
n_sessions = length(reward_seeking_results);

% Initialize data structure
baseline_data = struct();
baseline_data.n_sessions = n_sessions;

% 3 Compound states data
baseline_data.compound.frequency = zeros(n_sessions, 3);
baseline_data.compound.duration = zeros(n_sessions, 3);
baseline_data.compound.percentage = zeros(n_sessions, 3);
baseline_data.compound.speed = zeros(n_sessions, 3);
baseline_data.compound.labels = {'Reward Seeking', 'Idling', 'Rearing'};

% 7 Individual states data
baseline_data.individual.frequency = zeros(n_sessions, 7);
baseline_data.individual.duration = zeros(n_sessions, 7);
baseline_data.individual.percentage = zeros(n_sessions, 7);
baseline_data.individual.speed = zeros(n_sessions, 7);
baseline_data.individual.labels = {'High Speed', 'Non-Reward Corner', 'Reward Corner (No Port)', ...
    'At Reward Port', 'Center Area', 'Rearing', 'Goal-Directed Movement'};

% Session information
baseline_data.session_info = cell(n_sessions, 1);
baseline_data.session_durations = zeros(n_sessions, 1);

% Extract data from each session
for i = 1:n_sessions
    session = reward_seeking_results{i};

    % Store session info
    if isfield(session, 'filename')
        baseline_data.session_info{i} = session.filename;
    else
        baseline_data.session_info{i} = sprintf('Session_%d', i);
    end
    baseline_data.session_durations(i) = session.session_duration_min;

    % 3 Compound states
    baseline_data.compound.frequency(i, :) = [session.frequency.reward_seeking, ...
        session.frequency.idling, ...
        session.frequency.rearing];
    baseline_data.compound.duration(i, :) = [session.duration.reward_seeking_mean, ...
        session.duration.idling_mean, ...
        session.duration.rearing_mean];
    baseline_data.compound.percentage(i, :) = [session.duration.reward_seeking_percent, ...
        session.duration.idling_percent, ...
        session.duration.rearing_percent];
    baseline_data.compound.speed(i, :) = [session.speed.reward_seeking_mean, ...
        session.speed.idling_mean, ...
        session.speed.rearing_mean];

    % 7 Individual states
    baseline_data.individual.frequency(i, :) = [session.frequency.high_speed, ...
        session.frequency.non_reward_corner, ...
        session.frequency.reward_corner, ...
        session.frequency.reward_port, ...
        session.frequency.center, ...
        session.frequency.rearing, ...
        session.frequency.goal_directed];
    baseline_data.individual.duration(i, :) = [session.duration.high_speed_mean, ...
        session.duration.non_reward_corner_mean, ...
        session.duration.reward_corner_mean, ...
        session.duration.reward_port_mean, ...
        session.duration.center_mean, ...
        session.duration.rearing_mean, ...
        session.duration.goal_directed_mean];
    baseline_data.individual.percentage(i, :) = [session.duration.high_speed_percent, ...
        session.duration.non_reward_corner_percent, ...
        session.duration.reward_corner_percent, ...
        session.duration.reward_port_percent, ...
        session.duration.center_percent, ...
        session.duration.rearing_percent, ...
        session.duration.goal_directed_percent];
    baseline_data.individual.speed(i, :) = [session.speed.high_speed_mean, ...
        session.speed.non_reward_corner_mean, ...
        session.speed.reward_corner_mean, ...
        session.speed.reward_port_mean, ...
        session.speed.center_mean, ...
        session.speed.rearing_mean, ...
        session.speed.goal_directed_mean];
end
end

%% Extract before-aversive data from reward-aversive sessions
function before_aversive_data = extract_before_aversive_data(reward_aversive_results)
% Extract data from the before-aversive periods
valid_sessions = ~cellfun(@isempty, reward_aversive_results);
reward_aversive_results = reward_aversive_results(valid_sessions);
n_sessions = length(reward_aversive_results);

% Initialize data structure
before_aversive_data = struct();
before_aversive_data.n_sessions = n_sessions;

% 3 Compound states data
before_aversive_data.compound.frequency = zeros(n_sessions, 3);
before_aversive_data.compound.duration = zeros(n_sessions, 3);
before_aversive_data.compound.percentage = zeros(n_sessions, 3);
before_aversive_data.compound.speed = zeros(n_sessions, 3);
before_aversive_data.compound.labels = {'Reward Seeking', 'Idling', 'Rearing'};

% 7 Individual states data
before_aversive_data.individual.frequency = zeros(n_sessions, 7);
before_aversive_data.individual.duration = zeros(n_sessions, 7);
before_aversive_data.individual.percentage = zeros(n_sessions, 7);
before_aversive_data.individual.speed = zeros(n_sessions, 7);
before_aversive_data.individual.labels = {'High Speed', 'Non-Reward Corner', 'Reward Corner (No Port)', ...
    'At Reward Port', 'Center Area', 'Rearing', 'Goal-Directed Movement'};

% Session information
before_aversive_data.session_info = cell(n_sessions, 1);
before_aversive_data.session_durations = zeros(n_sessions, 1);
before_aversive_data.aversive_timing = zeros(n_sessions, 1);

% Extract data from each session's before period
for i = 1:n_sessions
    session = reward_aversive_results{i};
    before_metrics = session.before_metrics;

    % Store session info
    if isfield(session, 'filename')
        before_aversive_data.session_info{i} = session.filename;
    else
        before_aversive_data.session_info{i} = sprintf('Session_%d', i);
    end
    before_aversive_data.session_durations(i) = before_metrics.period_duration_min;
    before_aversive_data.aversive_timing(i) = session.first_aversive_time_min;

    % 3 Compound states
    before_aversive_data.compound.frequency(i, :) = [before_metrics.frequency.reward_seeking, ...
        before_metrics.frequency.idling, ...
        before_metrics.frequency.rearing];
    before_aversive_data.compound.duration(i, :) = [before_metrics.duration.reward_seeking_mean, ...
        before_metrics.duration.idling_mean, ...
        before_metrics.duration.rearing_mean];
    before_aversive_data.compound.percentage(i, :) = [before_metrics.duration.reward_seeking_percent, ...
        before_metrics.duration.idling_percent, ...
        before_metrics.duration.rearing_percent];
    before_aversive_data.compound.speed(i, :) = [before_metrics.speed.reward_seeking_mean, ...
        before_metrics.speed.idling_mean, ...
        before_metrics.speed.rearing_mean];

    % 7 Individual states
    before_aversive_data.individual.frequency(i, :) = [before_metrics.frequency.high_speed, ...
        before_metrics.frequency.non_reward_corner, ...
        before_metrics.frequency.reward_corner, ...
        before_metrics.frequency.reward_port, ...
        before_metrics.frequency.center, ...
        before_metrics.frequency.rearing, ...
        before_metrics.frequency.goal_directed];
    before_aversive_data.individual.duration(i, :) = [before_metrics.duration.high_speed_mean, ...
        before_metrics.duration.non_reward_corner_mean, ...
        before_metrics.duration.reward_corner_mean, ...
        before_metrics.duration.reward_port_mean, ...
        before_metrics.duration.center_mean, ...
        before_metrics.duration.rearing_mean, ...
        before_metrics.duration.goal_directed_mean];
    before_aversive_data.individual.percentage(i, :) = [before_metrics.duration.high_speed_percent, ...
        before_metrics.duration.non_reward_corner_percent, ...
        before_metrics.duration.reward_corner_percent, ...
        before_metrics.duration.reward_port_percent, ...
        before_metrics.duration.center_percent, ...
        before_metrics.duration.rearing_percent, ...
        before_metrics.duration.goal_directed_percent];
    before_aversive_data.individual.speed(i, :) = [before_metrics.speed.high_speed_mean, ...
        before_metrics.speed.non_reward_corner_mean, ...
        before_metrics.speed.reward_corner_mean, ...
        before_metrics.speed.reward_port_mean, ...
        before_metrics.speed.center_mean, ...
        before_metrics.speed.rearing_mean, ...
        before_metrics.speed.goal_directed_mean];
end
end

%% Extract after-aversive data from reward-aversive sessions
function after_aversive_data = extract_after_aversive_data(reward_aversive_results)
% Extract data from the after-aversive periods
valid_sessions = ~cellfun(@isempty, reward_aversive_results);
reward_aversive_results = reward_aversive_results(valid_sessions);
n_sessions = length(reward_aversive_results);

% Initialize data structure
after_aversive_data = struct();
after_aversive_data.n_sessions = n_sessions;

% 3 Compound states data
after_aversive_data.compound.frequency = zeros(n_sessions, 3);
after_aversive_data.compound.duration = zeros(n_sessions, 3);
after_aversive_data.compound.percentage = zeros(n_sessions, 3);
after_aversive_data.compound.speed = zeros(n_sessions, 3);
after_aversive_data.compound.labels = {'Reward Seeking', 'Idling', 'Rearing'};

% 7 Individual states data
after_aversive_data.individual.frequency = zeros(n_sessions, 7);
after_aversive_data.individual.duration = zeros(n_sessions, 7);
after_aversive_data.individual.percentage = zeros(n_sessions, 7);
after_aversive_data.individual.speed = zeros(n_sessions, 7);
after_aversive_data.individual.labels = {'High Speed', 'Non-Reward Corner', 'Reward Corner (No Port)', ...
    'At Reward Port', 'Center Area', 'Rearing', 'Goal-Directed Movement'};

% Session information
after_aversive_data.session_info = cell(n_sessions, 1);
after_aversive_data.session_durations = zeros(n_sessions, 1);
after_aversive_data.aversive_timing = zeros(n_sessions, 1);

% Extract data from each session's after period
for i = 1:n_sessions
    session = reward_aversive_results{i};
    after_metrics = session.after_metrics;

    % Store session info
    if isfield(session, 'filename')
        after_aversive_data.session_info{i} = session.filename;
    else
        after_aversive_data.session_info{i} = sprintf('Session_%d', i);
    end
    after_aversive_data.session_durations(i) = after_metrics.period_duration_min;
    after_aversive_data.aversive_timing(i) = session.first_aversive_time_min;

    % 3 Compound states
    after_aversive_data.compound.frequency(i, :) = [after_metrics.frequency.reward_seeking, ...
        after_metrics.frequency.idling, ...
        after_metrics.frequency.rearing];
    after_aversive_data.compound.duration(i, :) = [after_metrics.duration.reward_seeking_mean, ...
        after_metrics.duration.idling_mean, ...
        after_metrics.duration.rearing_mean];
    after_aversive_data.compound.percentage(i, :) = [after_metrics.duration.reward_seeking_percent, ...
        after_metrics.duration.idling_percent, ...
        after_metrics.duration.rearing_percent];
    after_aversive_data.compound.speed(i, :) = [after_metrics.speed.reward_seeking_mean, ...
        after_metrics.speed.idling_mean, ...
        after_metrics.speed.rearing_mean];

    % 7 Individual states
    after_aversive_data.individual.frequency(i, :) = [after_metrics.frequency.high_speed, ...
        after_metrics.frequency.non_reward_corner, ...
        after_metrics.frequency.reward_corner, ...
        after_metrics.frequency.reward_port, ...
        after_metrics.frequency.center, ...
        after_metrics.frequency.rearing, ...
        after_metrics.frequency.goal_directed];
    after_aversive_data.individual.duration(i, :) = [after_metrics.duration.high_speed_mean, ...
        after_metrics.duration.non_reward_corner_mean, ...
        after_metrics.duration.reward_corner_mean, ...
        after_metrics.duration.reward_port_mean, ...
        after_metrics.duration.center_mean, ...
        after_metrics.duration.rearing_mean, ...
        after_metrics.duration.goal_directed_mean];
    after_aversive_data.individual.percentage(i, :) = [after_metrics.duration.high_speed_percent, ...
        after_metrics.duration.non_reward_corner_percent, ...
        after_metrics.duration.reward_corner_percent, ...
        after_metrics.duration.reward_port_percent, ...
        after_metrics.duration.center_percent, ...
        after_metrics.duration.rearing_percent, ...
        after_metrics.duration.goal_directed_percent];
    after_aversive_data.individual.speed(i, :) = [after_metrics.speed.high_speed_mean, ...
        after_metrics.speed.non_reward_corner_mean, ...
        after_metrics.speed.reward_corner_mean, ...
        after_metrics.speed.reward_port_mean, ...
        after_metrics.speed.center_mean, ...
        after_metrics.speed.rearing_mean, ...
        after_metrics.speed.goal_directed_mean];
end
end

%% Create 3-state three-way comparison
function create_3_state_three_way_comparison(baseline_data, before_aversive_data, after_aversive_data)
fig1 = figure('Position', [50, 50, 2000, 1200]);
set(fig1, 'Color', 'white');

% Define colors
color_baseline = [0.3 0.7 0.3];        % Green
color_before = [0.2 0.6 0.8];          % Blue
color_after = [0.8 0.3 0.2];           % Red

behaviors = baseline_data.compound.labels;

%% Plot 1: Frequency Three-way Comparison
subplot(1, 4, 1);
plot_three_way_comparison(baseline_data.compound.frequency, before_aversive_data.compound.frequency, after_aversive_data.compound.frequency, ...
    behaviors, 'Behavioral Frequency (3 Compound States)', 'Bouts per minute', ...
    color_baseline, color_before, color_after, {'Baseline', 'Before Aversive', 'After Aversive'});

%% Plot 2: Duration Three-way Comparison
subplot(1, 4, 2);
plot_three_way_comparison(baseline_data.compound.duration, before_aversive_data.compound.duration, after_aversive_data.compound.duration, ...
    behaviors, 'Average Bout Duration (3 Compound States)', 'Duration (seconds)', ...
    color_baseline, color_before, color_after, {'Baseline', 'Before Aversive', 'After Aversive'});

%% Plot 3: Percentage Three-way Comparison
subplot(1, 4, 3);
plot_three_way_comparison(baseline_data.compound.percentage, before_aversive_data.compound.percentage, after_aversive_data.compound.percentage, ...
    behaviors, 'Time Allocation (3 Compound States)', 'Percentage of period (%)', ...
    color_baseline, color_before, color_after, {'Baseline', 'Before Aversive', 'After Aversive'});

%% Plot 4: Speed Three-way Comparison
subplot(1, 4, 4);
plot_three_way_comparison(baseline_data.compound.speed, before_aversive_data.compound.speed, after_aversive_data.compound.speed, ...
    behaviors, 'Movement Speed (3 Compound States)', 'Speed (units/s)', ...
    color_baseline, color_before, color_after, {'Baseline', 'Before Aversive', 'After Aversive'});

sgtitle('Three-way Comparison: Baseline vs Before vs After Aversive (3 Compound States)', ...
    'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');
end

%% Create 7-state three-way comparison
function create_7_state_three_way_comparison(baseline_data, before_aversive_data, after_aversive_data)
fig2 = figure('Position', [100, 100, 2000, 1400]);
set(fig2, 'Color', 'white');

% Define colors
color_baseline = [0.3 0.7 0.3];        % Green
color_before = [0.2 0.6 0.8];          % Blue
color_after = [0.8 0.3 0.2];           % Red

behaviors = baseline_data.individual.labels;

%% Plot 1-4: Main comparisons
subplot(1, 4, 1);
plot_three_way_comparison(baseline_data.individual.frequency, before_aversive_data.individual.frequency, after_aversive_data.individual.frequency, ...
    behaviors, 'Behavioral Frequency (7 States)', 'Bouts per minute', ...
    color_baseline, color_before, color_after, {'Baseline', 'Before', 'After'});

subplot(1, 4, 2);
plot_three_way_comparison(baseline_data.individual.duration, before_aversive_data.individual.duration, after_aversive_data.individual.duration, ...
    behaviors, 'Bout Duration (7 States)', 'Duration (seconds)', ...
    color_baseline, color_before, color_after, {'Baseline', 'Before', 'After'});

subplot(1, 4, 3);
plot_three_way_comparison(baseline_data.individual.percentage, before_aversive_data.individual.percentage, after_aversive_data.individual.percentage, ...
    behaviors, 'Time Allocation (7 States)', 'Percentage (%)', ...
    color_baseline, color_before, color_after, {'Baseline', 'Before', 'After'});

subplot(1, 4, 4);
plot_three_way_comparison(baseline_data.individual.speed, before_aversive_data.individual.speed, after_aversive_data.individual.speed, ...
    behaviors, 'Movement Speed (7 States)', 'Speed (units/s)', ...
    color_baseline, color_before, color_after, {'Baseline', 'Before', 'After'});


sgtitle('Three-way Comparison: Baseline vs Before vs After Aversive (7 Individual States)', ...
    'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');
end

%% Main three-way plotting function
function plot_three_way_comparison(data1, data2, data3, labels, plot_title, y_label, color1, color2, color3, group_labels)
n_behaviors = length(labels);
x_positions = 1:n_behaviors;
width = 0.25;

% Calculate statistics
means1 = mean(data1, 1, 'omitnan');
means2 = mean(data2, 1, 'omitnan');
means3 = mean(data3, 1, 'omitnan');
sems1 = std(data1, 1, 'omitnan') ./ sqrt(sum(~isnan(data1), 1));
sems2 = std(data2, 1, 'omitnan') ./ sqrt(sum(~isnan(data2), 1));
sems3 = std(data3, 1, 'omitnan') ./ sqrt(sum(~isnan(data3), 1));

% Create bar plot
h1 = bar(x_positions - width, means1, width, 'FaceColor', color1, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
hold on;
h2 = bar(x_positions, means2, width, 'FaceColor', color2, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
h3 = bar(x_positions + width, means3, width, 'FaceColor', color3, 'EdgeColor', 'none', 'FaceAlpha', 0.7);

% Add error bars
errorbar(x_positions - width, means1, sems1, 'k.', 'LineWidth', 1.5, 'CapSize', 3);
errorbar(x_positions, means2, sems2, 'k.', 'LineWidth', 1.5, 'CapSize', 3);
errorbar(x_positions + width, means3, sems3, 'k.', 'LineWidth', 1.5, 'CapSize', 3);

% Add individual data points and perform statistics
for i = 1:n_behaviors
    % Baseline data
    valid1 = ~isnan(data1(:,i));
    if any(valid1)
        jitter1 = (rand(sum(valid1), 1) - 0.5) * 0.08;
        scatter(x_positions(i) - width + jitter1, data1(valid1, i), 20, 'k', 'filled', 'MarkerFaceAlpha', 0.6);
    end

    % Before data
    valid2 = ~isnan(data2(:,i));
    if any(valid2)
        jitter2 = (rand(sum(valid2), 1) - 0.5) * 0.08;
        scatter(x_positions(i) + jitter2, data2(valid2, i), 20, 'k', 'filled', 'MarkerFaceAlpha', 0.6);
    end

    % After data
    valid3 = ~isnan(data3(:,i));
    if any(valid3)
        jitter3 = (rand(sum(valid3), 1) - 0.5) * 0.08;
        scatter(x_positions(i) + width + jitter3, data3(valid3, i), 20, 'k', 'filled', 'MarkerFaceAlpha', 0.6);
    end

    % Statistical tests with pairwise comparisons
    if any(valid1) && any(valid2) && any(valid3)
        % Overall test (Kruskal-Wallis)
        vals = [data1(valid1, i); data2(valid2, i); data3(valid3, i)];
        groups = [ones(sum(valid1), 1); 2*ones(sum(valid2), 1); 3*ones(sum(valid3), 1)];

        try
            p_overall = kruskalwallis(vals, groups, 'off');

            % If overall test is significant, perform pairwise comparisons
            if p_overall < 0.05
                % Pairwise comparisons (Wilcoxon rank-sum tests)
                p12 = ranksum(data1(valid1, i), data2(valid2, i));  % Group 1 vs Group 2
                p13 = ranksum(data1(valid1, i), data3(valid3, i));  % Group 1 vs Group 3
                p23 = ranksum(data2(valid2, i), data3(valid3, i));  % Group 2 vs Group 3

                % Bonferroni correction for multiple comparisons
                p_values = [p12, p13, p23];
                p_corrected = p_values * 3;  % Bonferroni correction
                p_corrected(p_corrected > 1) = 1;  % Cap at 1

                % Get maximum y-value for positioning significance bars
                y_max = max([means1(i) + sems1(i), means2(i) + sems2(i), means3(i) + sems3(i)]);

                % Show overall significance with red star and line if significant
                if p_overall < 0.05
                    overall_y_pos = y_max + y_max * 0.05;
                    % Red horizontal line spanning all three groups
                    line([x_positions(i) - width*1.2, x_positions(i) + width*1.2], [overall_y_pos, overall_y_pos], 'Color', 'r', 'LineWidth', 1.5);
                    % Red star
                    if p_overall < 0.001
                        overall_sig = '***';
                    elseif p_overall < 0.01
                        overall_sig = '**';
                    else
                        overall_sig = '*';
                    end
                    text(x_positions(i), overall_y_pos + y_max * 0.02, overall_sig, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10, 'Color', 'r');
                end

                % Add significance annotations for pairwise comparisons
                line_height = y_max * 0.1;  % Height increment for each comparison line
                base_y = y_max + y_max * 0.15;  % Start pairwise comparisons higher

                % Group 1 vs Group 2
                if p_corrected(1) < 0.05
                    y_pos = base_y + line_height;
                    line([x_positions(i) - width, x_positions(i)], [y_pos, y_pos], 'Color', 'k', 'LineWidth', 1);
                    line([x_positions(i) - width, x_positions(i) - width], [y_pos - line_height*0.5, y_pos], 'Color', 'k', 'LineWidth', 1);
                    line([x_positions(i), x_positions(i)], [y_pos - line_height*0.5, y_pos], 'Color', 'k', 'LineWidth', 1);

                    if p_corrected(1) < 0.001
                        sig_text = '***';
                    elseif p_corrected(1) < 0.01
                        sig_text = '**';
                    else
                        sig_text = '*';
                    end
                    text(x_positions(i) - width/2, y_pos + line_height*0.15, sig_text, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 8);
                end

                % Group 1 vs Group 3
                if p_corrected(2) < 0.05
                    y_pos = base_y + 2*line_height;
                    line([x_positions(i) - width, x_positions(i) + width], [y_pos, y_pos], 'Color', 'k', 'LineWidth', 1);
                    line([x_positions(i) - width, x_positions(i) - width], [y_pos - line_height*0.5, y_pos], 'Color', 'k', 'LineWidth', 1);
                    line([x_positions(i) + width, x_positions(i) + width], [y_pos - line_height*0.5, y_pos], 'Color', 'k', 'LineWidth', 1);

                    if p_corrected(2) < 0.001
                        sig_text = '***';
                    elseif p_corrected(2) < 0.01
                        sig_text = '**';
                    else
                        sig_text = '*';
                    end
                    text(x_positions(i), y_pos + line_height*0.15, sig_text, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 8);
                end

                % Group 2 vs Group 3
                if p_corrected(3) < 0.05
                    y_pos = base_y + 3*line_height;
                    line([x_positions(i), x_positions(i) + width], [y_pos, y_pos], 'Color', 'k', 'LineWidth', 1);
                    line([x_positions(i), x_positions(i)], [y_pos - line_height*0.5, y_pos], 'Color', 'k', 'LineWidth', 1);
                    line([x_positions(i) + width, x_positions(i) + width], [y_pos - line_height*0.5, y_pos], 'Color', 'k', 'LineWidth', 1);

                    if p_corrected(3) < 0.001
                        sig_text = '***';
                    elseif p_corrected(3) < 0.01
                        sig_text = '**';
                    else
                        sig_text = '*';
                    end
                    text(x_positions(i) + width/2, y_pos + line_height*0.15, sig_text, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 8);
                end
            end
        catch
            % Skip if statistical test fails
        end
    end
end

% Formatting
set(gca, 'XTick', x_positions, 'XTickLabel', labels);
xtickangle(45);
title(plot_title, 'FontSize', 11, 'FontWeight', 'bold');
ylabel(y_label, 'FontSize', 10);
legend([h1, h2, h3], group_labels, 'Location', 'best', 'FontSize', 9);
box off;

% Add sample size information
text(0.02, 0.98, sprintf('n: %d/%d/%d', size(data1,1), size(data2,1), size(data3,1)), ...
    'Units', 'normalized', 'FontSize', 8, 'BackgroundColor', 'white');

% Add note about multiple comparisons
text(0.02, 0.02, 'Pairwise comparisons: Bonferroni corrected', ...
    'Units', 'normalized', 'FontSize', 7, 'BackgroundColor', 'white', 'Color', [0.5 0.5 0.5]);
end
