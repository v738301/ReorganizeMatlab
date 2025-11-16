function [all_session_metrics] = loadSessionMetricsFromSpikeFiles(allfiles, T_sorted)
% loadSessionMetricsFromSpikeFiles - Load and calculate session metrics from spike files
%
% This function replaces loading compiled coupling data by calculating
% behavioral matrices and session data directly from spike structure files
%
% Input:
%   allfiles - output from selectFilesWithAnimalIDFiltering
%   T_sorted - sorting parameters from loadSortingParameters()
%
% Output:
%   all_session_metrics - cell array of session structures (compatible with old format)
%                        Each element contains:
%                          .filename
%                          .NeuralTime
%                          .TriggerMid
%                          .behavioral_matrix_full (8 columns)
%                          .all_aversive_time (if aversive session)
%                          .AdjustedXYZ
%                          .AdjustedXYZ_speed
%                          .Fs
%
% Example:
%   [T_sorted] = loadSortingParameters();
%   [allfiles, ~, ~, ~] = selectFilesWithAnimalIDFiltering(...);
%   sessions = loadSessionMetricsFromSpikeFiles(allfiles, T_sorted);

    fprintf('Loading and calculating session metrics from spike files...\n');
    fprintf('  Sessions to process: %d\n', length(allfiles));

    % Configuration
    config.bp_range = [1 300];
    config.min_duration = 2;
    config.f_slow = [0.5, 3];
    config.f_mid = [3, 6];
    config.f_fast = [6, 15];

    % Initialize storage
    all_session_metrics = {};

    % Process each session
    for sessionID = 1:length(allfiles)
        fprintf('  [%d/%d] Processing: %s\n', sessionID, length(allfiles), allfiles(sessionID).name);

        try
            % Load and prepare session data
            Timelimits = 'No';
            [NeuralTime, AdjustedXYZ, AdjustedXYZ_speed, Signal, IR1ON, IR2ON, WP1ON, WP2ON, ...
                AversiveSound, sessionLabels, valid_spikes, Fs, TriggerMid] = ...
                loadAndPrepareSessionData(allfiles(sessionID), T_sorted, Timelimits);

            % Preprocess signals for breathing frequency
            [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);
            breathing = filtered_data(:, 32);

            % Identify breathing frequency
            [dominant_freq_full, ~, ~, ~, ~, ~, ~, ~, ~, ~] = ...
                identifyBreathingFrequencyBands(breathing, NeuralTime, Fs, ...
                config.f_slow, config.f_mid, config.f_fast, config.min_duration);

            % Identify reward locations
            reward_locations = [];
            IR1_indices = find(IR1ON);
            IR2_indices = find(IR2ON);
            if length(IR1_indices) > 5
                reward_locations = [reward_locations; mean(AdjustedXYZ(IR1_indices, 1:2), 1)];
            end
            if length(IR2_indices) > 5
                reward_locations = [reward_locations; mean(AdjustedXYZ(IR2_indices, 1:2), 1)];
            end

            % Create behavioral matrix (7 columns)
            behavioral_matrix = create_behavioral_matrix(AdjustedXYZ, AdjustedXYZ_speed, ...
                reward_locations, IR1ON, IR2ON, Fs);

            % Add breathing frequency as 8th column
            behavioral_matrix_full = [behavioral_matrix, dominant_freq_full];

            % Extract aversive timing if available
            all_aversive_time = [];
            if ~isempty(AversiveSound)
                % Find aversive sound onsets
                aversive_diff = diff([0; AversiveSound]);
                aversive_onsets = find(aversive_diff > 0);

                if ~isempty(aversive_onsets)
                    % Convert to relative times (seconds from start)
                    all_aversive_time = NeuralTime(aversive_onsets) - NeuralTime(1);
                end
            end

            % Create session structure (compatible with old format)
            session_metrics = struct();
            session_metrics.filename = allfiles(sessionID).name;
            session_metrics.NeuralTime = NeuralTime;
            session_metrics.TriggerMid = TriggerMid;
            session_metrics.behavioral_matrix_full = behavioral_matrix_full;
            session_metrics.all_aversive_time = all_aversive_time;
            session_metrics.AdjustedXYZ = AdjustedXYZ;
            session_metrics.AdjustedXYZ_speed = AdjustedXYZ_speed;
            session_metrics.Fs = Fs;
            session_metrics.IR1ON = IR1ON;
            session_metrics.IR2ON = IR2ON;
            session_metrics.WP1ON = WP1ON;
            session_metrics.WP2ON = WP2ON;
            session_metrics.AversiveSound = AversiveSound;

            % Add to cell array
            all_session_metrics{sessionID} = session_metrics;

            fprintf('    ✅ Success: %d timepoints, %d aversive events\n', ...
                length(NeuralTime), length(all_aversive_time));

        catch ME
            fprintf('    ❌ Failed: %s\n', ME.message);

            % Create empty placeholder
            session_metrics = struct();
            session_metrics.filename = allfiles(sessionID).name;
            session_metrics.NeuralTime = [];
            session_metrics.TriggerMid = [];
            session_metrics.behavioral_matrix_full = [];
            session_metrics.all_aversive_time = [];
            session_metrics.AdjustedXYZ = [];
            session_metrics.AdjustedXYZ_speed = [];
            session_metrics.Fs = [];

            all_session_metrics{sessionID} = session_metrics;
        end
    end

    fprintf('✓ Loaded and processed %d sessions\n\n', length(all_session_metrics));
end
