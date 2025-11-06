%% ========================================================================
%  CLUSTER-WAVEFORM-PROPERTY ANALYSIS
%  Investigates relationship between:
%  - Waveform properties (cell type indicators)
%  - Firing rate & CV (temporal dynamics)
%  - Functional cluster assignments (from simplified clustering)
%  ========================================================================
%
%  Research Question:
%  How do intrinsic cellular properties (waveform, FR, CV) relate to
%  functional properties (oscillation coupling, task modulation)?
%
%  Cluster Types Identified:
%  1. Strong local oscillation coupling units
%  2. High firing rate units
%  3. Aversive modulated (inhibited or activated)
%  4. Reward modulated (inhibited or activated)
%  5. Aversive/Reward mix modulated units
%
%% ========================================================================

clear all;
close all;

fprintf('========================================\n');
fprintf('CLUSTER-WAVEFORM-PROPERTY ANALYSIS\n');
fprintf('========================================\n\n');

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

config = struct();

% Paths (UPDATE THESE FOR YOUR SYSTEM)
config.nex_data_path = '/Volumes/ExpansionBackup/Data/Ephy';
config.spike_data_path = '/Volumes/ExpansionBackup/Data/Struct_spike';
config.clustering_results_file = ''; % e.g., 'simplified_clustering_Aversive_2025-01-15.mat'

% Waveform analysis parameters
config.waveform_sampling_rate = 30000;  % Hz (typical for neural recordings)
config.waveform_window_ms = 2.0;        % ms window for waveform analysis
config.trough_to_peak_threshold = 0.4;  % ms - threshold for putative interneurons vs pyramidal

% Firing rate analysis
config.fr_bin_size = 60;  % seconds - bin size for temporal FR analysis
config.cv_min_spikes = 10; % minimum spikes for CV calculation

% Session filtering
config.exclude_sessions = [];  % Add session IDs to exclude if needed

fprintf('Configuration loaded:\n');
fprintf('  NEX data path: %s\n', config.nex_data_path);
fprintf('  Waveform sampling rate: %d Hz\n', config.waveform_sampling_rate);
fprintf('  Trough-to-peak threshold: %.2f ms\n\n', config.trough_to_peak_threshold);

%% ========================================================================
%  SECTION 2: LOAD CLUSTERING RESULTS
%  ========================================================================

fprintf('=== LOADING CLUSTERING RESULTS ===\n');

% Prompt user to select clustering results file if not specified
if isempty(config.clustering_results_file)
    fprintf('Please select the simplified clustering results file...\n');
    [filename, pathname] = uigetfile('*.mat', 'Select Simplified Clustering Results');
    if isequal(filename, 0)
        error('No file selected. Exiting.');
    end
    config.clustering_results_file = fullfile(pathname, filename);
end

fprintf('Loading: %s\n', config.clustering_results_file);
clustering_data = load(config.clustering_results_file);
results = clustering_data.simplified_clustering_results;

fprintf('✓ Clustering results loaded\n');
fprintf('  Total clusters: %d\n', results.clustering.n_clusters);
fprintf('  Total units: %d\n', length(results.units));
fprintf('  Session type: %s\n', results.metadata.session_type);
fprintf('  Features used: %s\n\n', strjoin(results.features.names, ', '));

%% ========================================================================
%  SECTION 3: EXTRACT WAVEFORMS FROM .NEX FILES
%  ========================================================================

fprintf('\n=== EXTRACTING WAVEFORMS FROM .NEX FILES ===\n');

% TODO: USER - Insert your waveform extraction code here
% Reference: TryToIdentifyWaveForm.m
%
% Expected output structure for each unit:
% waveform_data(i).global_unit_id
% waveform_data(i).session_id
% waveform_data(i).unit_id
% waveform_data(i).waveform        % Average waveform (1 x N_samples)
% waveform_data(i).waveform_std    % Standard deviation
% waveform_data(i).n_waveforms     % Number of waveforms averaged
% waveform_data(i).sampling_rate   % Hz
% waveform_data(i).nex_filename    % Source .nex file

fprintf('NOTE: Waveform extraction code needs to be added\n');
fprintf('Please refer to: %s\n', fullfile(config.nex_data_path, 'README'));
fprintf('Expected function signature:\n');
fprintf('  [waveforms] = extractWaveformsFromNex(nex_path, unit_list)\n\n');

% PLACEHOLDER: Example structure
waveform_data = struct();
waveform_data_available = false;

% TODO: Uncomment and implement
% for unit_idx = 1:length(results.units)
%     unit = results.units(unit_idx);
%
%     % Find corresponding .nex file
%     nex_filename = convertMatFilenameToNex(unit.session_filename);
%     nex_filepath = fullfile(config.nex_data_path, nex_filename);
%
%     if exist(nex_filepath, 'file')
%         % Extract waveform for this unit
%         wf = extractSingleUnitWaveform(nex_filepath, unit.unit_id);
%
%         waveform_data(unit_idx).global_unit_id = unit.global_unit_id;
%         waveform_data(unit_idx).session_id = unit.session_id;
%         waveform_data(unit_idx).unit_id = unit.unit_id;
%         waveform_data(unit_idx).waveform = wf.mean_waveform;
%         waveform_data(unit_idx).waveform_std = wf.std_waveform;
%         waveform_data(unit_idx).n_waveforms = wf.n_spikes;
%         waveform_data(unit_idx).sampling_rate = config.waveform_sampling_rate;
%         waveform_data(unit_idx).nex_filename = nex_filename;
%     else
%         fprintf('  Warning: .nex file not found for unit %d\n', unit.global_unit_id);
%     end
% end
%
% waveform_data_available = true;
% fprintf('✓ Extracted waveforms for %d units\n\n', length(waveform_data));

%% ========================================================================
%  SECTION 4: COMPUTE WAVEFORM FEATURES
%  ========================================================================

fprintf('\n=== COMPUTING WAVEFORM FEATURES ===\n');

if waveform_data_available

    waveform_features = struct();

    for i = 1:length(waveform_data)
        wf = waveform_data(i);

        % Time vector (ms)
        n_samples = length(wf.waveform);
        time_ms = (0:n_samples-1) / wf.sampling_rate * 1000;

        % Find trough (minimum)
        [trough_val, trough_idx] = min(wf.waveform);
        trough_time = time_ms(trough_idx);

        % Find peak after trough
        post_trough_wf = wf.waveform(trough_idx:end);
        post_trough_time = time_ms(trough_idx:end);
        [peak_val, peak_idx_rel] = max(post_trough_wf);
        peak_idx = trough_idx + peak_idx_rel - 1;
        peak_time = time_ms(peak_idx);

        % Trough-to-peak time (ms) - key feature for cell type classification
        trough_to_peak = peak_time - trough_time;

        % Peak-to-trough amplitude
        amplitude = peak_val - trough_val;

        % Waveform width at half-maximum (FWHM)
        half_max = trough_val + (peak_val - trough_val) / 2;
        above_half = wf.waveform >= half_max;
        half_max_indices = find(above_half);
        if ~isempty(half_max_indices)
            fwhm = (half_max_indices(end) - half_max_indices(1)) / wf.sampling_rate * 1000;
        else
            fwhm = NaN;
        end

        % Repolarization slope (after peak)
        if peak_idx < length(wf.waveform) - 10
            repol_window = peak_idx:(peak_idx + 10);
            repol_slope = polyfit(time_ms(repol_window), wf.waveform(repol_window), 1);
            repol_rate = repol_slope(1);  % V/ms
        else
            repol_rate = NaN;
        end

        % Asymmetry index (ratio of pre-trough to post-peak)
        pre_trough_area = sum(abs(wf.waveform(1:trough_idx)));
        post_peak_area = sum(abs(wf.waveform(peak_idx:end)));
        asymmetry = post_peak_area / pre_trough_area;

        % Store features
        waveform_features(i).global_unit_id = wf.global_unit_id;
        waveform_features(i).trough_to_peak_ms = trough_to_peak;
        waveform_features(i).amplitude = amplitude;
        waveform_features(i).fwhm_ms = fwhm;
        waveform_features(i).repol_rate = repol_rate;
        waveform_features(i).asymmetry = asymmetry;
        waveform_features(i).trough_time_ms = trough_time;
        waveform_features(i).peak_time_ms = peak_time;

        % Putative cell type classification
        % Interneurons: narrow waveform (trough-to-peak < 0.4 ms)
        % Pyramidal cells: broad waveform (trough-to-peak > 0.4 ms)
        if trough_to_peak < config.trough_to_peak_threshold
            waveform_features(i).putative_cell_type = 'Interneuron';
            waveform_features(i).cell_type_code = 1;
        else
            waveform_features(i).putative_cell_type = 'Pyramidal';
            waveform_features(i).cell_type_code = 2;
        end
    end

    fprintf('✓ Computed waveform features for %d units\n', length(waveform_features));
    fprintf('  Putative interneurons: %d\n', sum([waveform_features.cell_type_code] == 1));
    fprintf('  Putative pyramidal cells: %d\n\n', sum([waveform_features.cell_type_code] == 2));

else
    fprintf('Skipping waveform feature computation (no waveform data)\n\n');
    waveform_features = [];
end

%% ========================================================================
%  SECTION 5: COMPUTE TEMPORAL FIRING RATE & CV
%  ========================================================================

fprintf('\n=== COMPUTING TEMPORAL FR & CV ===\n');

% Load sorting parameters
fprintf('Loading sorting parameters...\n');
[T_sorted] = loadSortingParameters();

% Initialize storage
temporal_fr_cv = struct();
unit_counter = 0;

% Get unique sessions from clustering results
unique_sessions = unique({results.units.session_filename});
fprintf('Processing %d unique sessions...\n', length(unique_sessions));

for sess_idx = 1:length(unique_sessions)
    session_filename = unique_sessions{sess_idx};

    fprintf('\n[%d/%d] Processing: %s\n', sess_idx, length(unique_sessions), session_filename);

    % Find session file
    session_path = fullfile(config.spike_data_path, session_filename);

    if ~exist(session_path, 'file')
        fprintf('  Warning: Session file not found, skipping\n');
        continue;
    end

    % Load session data
    session_struct = dir(session_path);
    Timelimits = 'No';

    try
        [NeuralTime, ~, ~, ~, ~, ~, ~, ~, ~, ~, valid_spikes, Fs, TriggerMid] = ...
            loadAndPrepareSessionData(session_struct, T_sorted, Timelimits);

        % Session duration
        session_start = TriggerMid(1);
        session_end = TriggerMid(end);
        session_duration = session_end - session_start;

        % Time bins for temporal analysis
        time_bins = session_start:config.fr_bin_size:session_end;
        n_bins = length(time_bins) - 1;

        fprintf('  Session duration: %.1f min, %d time bins\n', session_duration/60, n_bins);

        % Get units from this session in clustering results
        session_units = results.units(strcmp({results.units.session_filename}, session_filename));

        fprintf('  Processing %d units\n', length(session_units));

        for u = 1:length(session_units)
            unit = session_units(u);
            unit_counter = unit_counter + 1;

            % Get spike times
            if unit.unit_id <= length(valid_spikes)
                spike_times = valid_spikes{unit.unit_id};
            else
                fprintf('    Warning: Unit %d not found in spike data\n', unit.unit_id);
                continue;
            end

            if isempty(spike_times)
                continue;
            end

            % Compute FR across time
            fr_timecourse = zeros(1, n_bins);
            cv_timecourse = nan(1, n_bins);

            for bin_idx = 1:n_bins
                bin_start = time_bins(bin_idx);
                bin_end = time_bins(bin_idx + 1);

                % Spikes in this bin
                spikes_in_bin = spike_times(spike_times >= bin_start & spike_times < bin_end);
                n_spikes = length(spikes_in_bin);

                % Firing rate (Hz)
                fr_timecourse(bin_idx) = n_spikes / config.fr_bin_size;

                % CV
                if n_spikes >= config.cv_min_spikes
                    ISI = diff(spikes_in_bin);
                    if ~isempty(ISI) && mean(ISI) > 0
                        cv_timecourse(bin_idx) = std(ISI) / mean(ISI);
                    end
                end
            end

            % Overall statistics
            mean_fr = length(spike_times) / session_duration;
            ISI_all = diff(spike_times);
            if length(ISI_all) > 0 && mean(ISI_all) > 0
                mean_cv = std(ISI_all) / mean(ISI_all);
            else
                mean_cv = NaN;
            end

            % Store
            temporal_fr_cv(unit_counter).global_unit_id = unit.global_unit_id;
            temporal_fr_cv(unit_counter).session_id = unit.session_id;
            temporal_fr_cv(unit_counter).unit_id = unit.unit_id;
            temporal_fr_cv(unit_counter).cluster_id = unit.cluster_id;
            temporal_fr_cv(unit_counter).session_type = unit.session_type;
            temporal_fr_cv(unit_counter).session_filename = unit.session_filename;

            temporal_fr_cv(unit_counter).fr_timecourse = fr_timecourse;
            temporal_fr_cv(unit_counter).cv_timecourse = cv_timecourse;
            temporal_fr_cv(unit_counter).time_bins = time_bins(1:end-1);

            temporal_fr_cv(unit_counter).mean_fr = mean_fr;
            temporal_fr_cv(unit_counter).mean_cv = mean_cv;
            temporal_fr_cv(unit_counter).fr_std = std(fr_timecourse);
            temporal_fr_cv(unit_counter).cv_median = nanmedian(cv_timecourse);
            temporal_fr_cv(unit_counter).n_spikes = length(spike_times);
        end

    catch ME
        fprintf('  Error processing session: %s\n', ME.message);
        continue;
    end
end

fprintf('\n✓ Computed temporal FR/CV for %d units\n\n', unit_counter);

%% ========================================================================
%  SECTION 6: INTEGRATE ALL DATA
%  ========================================================================

fprintf('\n=== INTEGRATING ALL DATA ===\n');

% Create comprehensive unit table matching clustering results
n_units = length(results.units);

integrated_data = struct();

for i = 1:n_units
    unit = results.units(i);

    % Basic info
    integrated_data(i).global_unit_id = unit.global_unit_id;
    integrated_data(i).session_id = unit.session_id;
    integrated_data(i).unit_id = unit.unit_id;
    integrated_data(i).cluster_id = unit.cluster_id;
    integrated_data(i).session_type = unit.session_type;
    integrated_data(i).session_filename = unit.session_filename;

    % Functional features (from clustering)
    integrated_data(i).functional_features = unit.features;
    integrated_data(i).feature_names = results.features.names;

    % Waveform features
    if ~isempty(waveform_features)
        wf_idx = find([waveform_features.global_unit_id] == unit.global_unit_id);
        if ~isempty(wf_idx)
            integrated_data(i).wf_trough_to_peak = waveform_features(wf_idx).trough_to_peak_ms;
            integrated_data(i).wf_amplitude = waveform_features(wf_idx).amplitude;
            integrated_data(i).wf_fwhm = waveform_features(wf_idx).fwhm_ms;
            integrated_data(i).wf_asymmetry = waveform_features(wf_idx).asymmetry;
            integrated_data(i).putative_cell_type = waveform_features(wf_idx).putative_cell_type;
            integrated_data(i).cell_type_code = waveform_features(wf_idx).cell_type_code;
        else
            integrated_data(i).wf_trough_to_peak = NaN;
            integrated_data(i).wf_amplitude = NaN;
            integrated_data(i).wf_fwhm = NaN;
            integrated_data(i).wf_asymmetry = NaN;
            integrated_data(i).putative_cell_type = 'Unknown';
            integrated_data(i).cell_type_code = 0;
        end
    else
        integrated_data(i).wf_trough_to_peak = NaN;
        integrated_data(i).wf_amplitude = NaN;
        integrated_data(i).wf_fwhm = NaN;
        integrated_data(i).wf_asymmetry = NaN;
        integrated_data(i).putative_cell_type = 'Unknown';
        integrated_data(i).cell_type_code = 0;
    end

    % Temporal FR/CV
    if ~isempty(temporal_fr_cv)
        temp_idx = find([temporal_fr_cv.global_unit_id] == unit.global_unit_id);
        if ~isempty(temp_idx)
            integrated_data(i).mean_fr = temporal_fr_cv(temp_idx).mean_fr;
            integrated_data(i).mean_cv = temporal_fr_cv(temp_idx).mean_cv;
            integrated_data(i).fr_std = temporal_fr_cv(temp_idx).fr_std;
            integrated_data(i).cv_median = temporal_fr_cv(temp_idx).cv_median;
            integrated_data(i).fr_timecourse = temporal_fr_cv(temp_idx).fr_timecourse;
            integrated_data(i).cv_timecourse = temporal_fr_cv(temp_idx).cv_timecourse;
            integrated_data(i).time_bins = temporal_fr_cv(temp_idx).time_bins;
        else
            integrated_data(i).mean_fr = NaN;
            integrated_data(i).mean_cv = NaN;
            integrated_data(i).fr_std = NaN;
            integrated_data(i).cv_median = NaN;
        end
    else
        integrated_data(i).mean_fr = NaN;
        integrated_data(i).mean_cv = NaN;
        integrated_data(i).fr_std = NaN;
        integrated_data(i).cv_median = NaN;
    end
end

fprintf('✓ Integrated data for %d units\n\n', length(integrated_data));

%% ========================================================================
%  SECTION 7: ANALYZE RELATIONSHIPS
%  ========================================================================

fprintf('\n=== ANALYZING RELATIONSHIPS ===\n');

% Question 1: How does cell type (waveform) relate to cluster assignment?
fprintf('\n--- Cell Type Distribution Across Clusters ---\n');

for c = 1:results.clustering.n_clusters
    cluster_units = integrated_data([integrated_data.cluster_id] == c);

    n_interneuron = sum([cluster_units.cell_type_code] == 1);
    n_pyramidal = sum([cluster_units.cell_type_code] == 2);
    n_unknown = sum([cluster_units.cell_type_code] == 0);

    fprintf('Cluster %d (%d units):\n', c, length(cluster_units));
    fprintf('  Interneurons: %d (%.1f%%)\n', n_interneuron, 100*n_interneuron/length(cluster_units));
    fprintf('  Pyramidal: %d (%.1f%%)\n', n_pyramidal, 100*n_pyramidal/length(cluster_units));
    fprintf('  Unknown: %d\n', n_unknown);
end

% Question 2: How does FR/CV relate to cluster assignment?
fprintf('\n--- FR/CV Properties Across Clusters ---\n');

for c = 1:results.clustering.n_clusters
    cluster_units = integrated_data([integrated_data.cluster_id] == c);

    fr_vals = [cluster_units.mean_fr];
    cv_vals = [cluster_units.mean_cv];

    fprintf('Cluster %d:\n', c);
    fprintf('  Mean FR: %.2f ± %.2f Hz\n', nanmean(fr_vals), nanstd(fr_vals));
    fprintf('  Mean CV: %.2f ± %.2f\n', nanmean(cv_vals), nanstd(cv_vals));
end

% Question 3: Correlation between waveform features and functional features
fprintf('\n--- Waveform-Functional Feature Correlations ---\n');

% Extract matrices for correlation
wf_trough_to_peak = [integrated_data.wf_trough_to_peak]';
valid_wf = ~isnan(wf_trough_to_peak);

if sum(valid_wf) > 10
    functional_matrix = vertcat(integrated_data(valid_wf).functional_features);
    feature_names = integrated_data(1).feature_names;

    for f = 1:length(feature_names)
        [r, p] = corr(wf_trough_to_peak(valid_wf), functional_matrix(:, f), ...
                      'Type', 'Spearman', 'Rows', 'complete');

        if p < 0.05
            fprintf('  %s: r=%.3f, p=%.4f %s\n', feature_names{f}, r, p, ...
                    ternary(p < 0.001, '***', ternary(p < 0.01, '**', '*')));
        end
    end
else
    fprintf('  Insufficient waveform data for correlation analysis\n');
end

%% ========================================================================
%  SECTION 8: SAVE RESULTS
%  ========================================================================

fprintf('\n=== SAVING RESULTS ===\n');

analysis_results = struct();
analysis_results.config = config;
analysis_results.clustering_results = results;
analysis_results.waveform_features = waveform_features;
analysis_results.temporal_fr_cv = temporal_fr_cv;
analysis_results.integrated_data = integrated_data;
analysis_results.analysis_timestamp = datestr(now);

% Save
timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
save_filename = sprintf('cluster_waveform_analysis_%s.mat', timestamp);
save(save_filename, 'analysis_results', '-v7.3');

fprintf('✓ Saved to: %s\n\n', save_filename);

fprintf('========================================\n');
fprintf('ANALYSIS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Next step: Run visualization script\n');
fprintf('  Cluster_Waveform_Visualization.m\n');
fprintf('========================================\n');

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
