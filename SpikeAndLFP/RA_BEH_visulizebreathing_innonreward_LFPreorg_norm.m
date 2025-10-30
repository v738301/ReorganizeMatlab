clear all
% close all

%%

DataSetsPath = '/Volumes/My980Pro/reorganize/SpikeAndLFP/DataSet';
AnalysisType = 'Aversive';
AnalysisType = 'Reward';

analyze_nonreward_coupling_patches(DataSetsPath,AnalysisType)

%%
function analyze_nonreward_coupling_patches(DataSetsPath,AnalysisType)
%% ANALYZE COUPLING DURING NON-REWARD PERIODS - MULTI-BAND COUPLING ANALYSIS WITH SESSION-SPECIFIC BASELINES
% Analyze time-resolved breathing-gamma coupling during non-reward periods
% for multiple frequency bands [4, 5, 6, 7] Hz with session-specific permutation baselines
%
% Features:
% - Session-specific permutation baselines instead of cross-session pooling
% - Normalizes coupling strength by each session's own baseline
%
% Usage:
%   analyze_nonreward_coupling_patches('RewardSeeking_session_metrics_breathing_multiband.mat');

%% Load and process data
% Extract sessions
sessions = extract_sessions_from_data(DataSetsPath,AnalysisType);
n_sessions = length(sessions);
fprintf('Found %d sessions to analyze with session-specific baselines\n', n_sessions);

%% Process patches and coupling data
fprintf('\nExtracting behavioral and coupling data...\n');
[all_behavioral_data, valid_patches] = extract_and_process_patches(sessions);

if isempty(valid_patches)
    fprintf('❌ No valid patches found\n');
    return;
end

%% Perform multi-band analysis with session-specific baselines
fprintf('\n=== MULTI-BAND COUPLING ANALYSIS WITH SESSION-SPECIFIC BASELINES ===\n');
% breathing_bands = [4, 5, 6, 7]; % Hz
breathing_bands = [6, 7]; % Hz
band_results = perform_multiband_analysis_with_session_specific_baselines(valid_patches, all_behavioral_data, breathing_bands);

%% Create comprehensive visualization
if ~isempty(fieldnames(band_results))
%     visualize_multiband_coupling_analysis_simplified(band_results, [4, 5, 6, 7]);
%     visualize_individual_bands_with_breathing_quartiles(band_results, [4, 5, 6, 7], valid_patches);
%     visualize_session_specific_multiband_analysis(band_results, [4, 5, 6, 7]);
%     print_multiband_coupling_summary_session_specific(band_results, [4, 5, 6, 7]);
%     analyze_coupling_duration_relationship(valid_patches);
%     plot_coupling_heatmap(band_results, 'breathing', 'MI', [4,5,6,7], valid_patches);

    visualize_multiband_coupling_analysis_simplified(band_results, [6, 7]);
    visualize_individual_bands_with_breathing_quartiles(band_results, [6, 7], valid_patches);
    visualize_session_specific_multiband_analysis(band_results, [6, 7]);
    print_multiband_coupling_summary_session_specific(band_results, [6, 7]);
    analyze_coupling_duration_relationship(valid_patches);
    plot_coupling_heatmap(band_results, 'breathing', 'MI', [6,7], valid_patches);
else
    fprintf('❌ No results to visualize\n');
end

fprintf('\n✓ Session-specific analysis complete.\n');
end

%% ========================================================================
%% DATA EXTRACTION AND PROCESSING FUNCTIONS
%% ========================================================================

function sessions = extract_sessions_from_data(DataSetsPath,AnalysisType)
%% Extract sessions from loaded data structure
% Define paths to individual session directories
if AnalysisType == 'Reward'
    RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_Individual_Sessions');

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
        reward_seeking_results{i}.NeuralTime = session_data.NeuralTime;
        reward_seeking_results{i}.behavioral_matrix = session_data.behavioral_matrix_full;
        reward_seeking_results{i}.coupling_results_multiband = session_data.coupling_results_multiband;

        clear session_data;
    end
    fprintf('Loaded %d reward seeking sessions\n\n', length(reward_seeking_results));

    sessions = reward_seeking_results;

else
    RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_Individual_Sessions');

    % Load config from any session file (assuming all have the same config)
    reward_seeking_files = dir(fullfile(RewardAversivePath, '*_coupling_analysis.mat'));
    if ~isempty(reward_seeking_files)
        temp = load(fullfile(RewardSeekingPath, reward_seeking_files(1).name), 'config');
        config = temp.config;
        clear temp;
    else
        error('No reward seeking session files found');
    end

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
        reward_aversive_results{i}.NeuralTime = session_data.NeuralTime;
        reward_aversive_results{i}.behavioral_matrix = session_data.behavioral_matrix_full;
        reward_aversive_results{i}.coupling_results_multiband = session_data.coupling_results_multiband;

        clear session_data;
    end
    fprintf('Loaded %d reward aversive sessions\n\n', length(reward_aversive_results));

    sessions = reward_aversive_results;
end

end

function [all_behavioral_data, valid_patches] = extract_and_process_patches(sessions)
%% Extract behavioral data and identify valid patches
all_behavioral_data = [];
nonreward_patches = [];
n_sessions = length(sessions);

for session_idx = 1:n_sessions
    fprintf('  Processing session %d/%d...\n', session_idx, n_sessions);
    
    session = sessions{session_idx};
    
    % Check if coupling results exist
    if ~isfield(session, 'coupling_results_multiband') || ...
       isempty(session.coupling_results_multiband) || ...
       isempty(session.coupling_results_multiband.summary)
        fprintf('    ❌ No multi-band coupling results found, skipping\n');
        continue;
    end
    
    % Extract behavioral matrix
    if isfield(session, 'behavioral_matrix_full')
        % For reward-aversive sessions, use before_indices
        % behavioral_matrix = session.behavioral_matrix_full(session.before_indices,:);
        % neural_time = session.NeuralTime(session.before_indices);
        behavioral_matrix = session.behavioral_matrix_full(session.after_indices,:);
        neural_time = session.NeuralTime(session.after_indices);
        % behavioral_matrix = session.behavioral_matrix_full;
        % neural_time = session.NeuralTime;
    else
        fprintf('    ❌ No behavioral matrix found, skipping\n');
        continue;
    end
    
    % Extract coupling data
    coupling_multiband = session.coupling_results_multiband;
    if isempty(coupling_multiband.summary)
        fprintf('    ❌ No coupling summary found, skipping\n');
        continue;
    end
    
    % Interpolate coupling data to match behavioral timeline
    coupling_interpolated = interpolate_coupling_to_behavioral_timeline(...
        coupling_multiband, neural_time);
    
    if isempty(coupling_interpolated)
        fprintf('    ❌ Failed to interpolate coupling data, skipping\n');
        continue;
    end
    
    % Store all behavioral and coupling data for random patch selection
    session_data = struct();
    session_data.session_id = session_idx;
    session_data.behavioral_matrix = behavioral_matrix;
    session_data.coupling_data = coupling_interpolated;
    session_data.neural_time = neural_time;
    all_behavioral_data = [all_behavioral_data; session_data];
    
    % Identify non-reward patches with coupling data
    patches = identify_nonreward_coupling_patches(behavioral_matrix, coupling_interpolated, session_idx);
    nonreward_patches = [nonreward_patches; patches];
end

% Filter valid patches (≥10 seconds core duration)
valid_patches = nonreward_patches([nonreward_patches.valid] & [nonreward_patches.duration_seconds] >= 10);
fprintf('Found %d valid buffered non-reward patches across all sessions\n', length(valid_patches));
end

function coupling_interpolated = interpolate_coupling_to_behavioral_timeline(coupling_multiband, neural_time)
%% Interpolate multi-band coupling data to match behavioral timeline
fprintf('    Interpolating coupling data to behavioral timeline...\n');

summary = coupling_multiband.summary;
if isempty(summary)
    coupling_interpolated = [];
    return;
end

% Get coupling time vector and data
coupling_time = summary.window_times;
coupling_MI_matrix = summary.all_MI_values; % [n_bands x n_windows]
coupling_coherence_matrix = summary.all_coherence; % [n_bands x n_windows]
breathing_bands = coupling_multiband.breathing_bands;

% Check dimensions
[n_bands, n_coupling_windows] = size(coupling_MI_matrix);
n_behavioral_samples = length(neural_time);

fprintf('      Coupling data: %d bands x %d windows\n', n_bands, n_coupling_windows);
fprintf('      Behavioral data: %d samples\n', n_behavioral_samples);
fprintf('      Time range - Coupling: %.1f-%.1fs, Behavioral: %.1f-%.1fs\n', ...
    coupling_time(1), coupling_time(end), neural_time(1), neural_time(end));

% Initialize interpolated data
coupling_interpolated = struct();
coupling_interpolated.breathing_bands = breathing_bands;
coupling_interpolated.MI_data = zeros(n_bands, n_behavioral_samples);
coupling_interpolated.coherence_data = zeros(n_bands, n_behavioral_samples);
coupling_interpolated.neural_time = neural_time;

% Interpolate each band
for band_idx = 1:n_bands
    freq = breathing_bands(band_idx);
    
    % Extract data for this band
    band_MI = coupling_MI_matrix(band_idx, :);
    band_coherence = coupling_coherence_matrix(band_idx, :);
    
    % Remove NaN values
    valid_coupling = ~isnan(band_MI) & ~isnan(band_coherence);
    if sum(valid_coupling) < 3
        fprintf('      ⚠️  Band %.1f Hz: insufficient valid data points\n', freq);
        continue;
    end
    
    valid_coupling_time = coupling_time(valid_coupling);
    valid_MI = band_MI(valid_coupling);
    valid_coherence = band_coherence(valid_coupling);
    
    % Interpolate to behavioral timeline
    try
        % Use linear interpolation with extrapolation for boundaries
        interp_MI = interp1(valid_coupling_time, valid_MI, neural_time, 'linear', 'extrap');
        interp_coherence = interp1(valid_coupling_time, valid_coherence, neural_time, 'linear', 'extrap');
        
        % Store interpolated data
        coupling_interpolated.MI_data(band_idx, :) = interp_MI;
        coupling_interpolated.coherence_data(band_idx, :) = interp_coherence;
        
        fprintf('      ✓ Band %.1f Hz interpolated successfully\n', freq);
        
    catch ME
        fprintf('      ❌ Band %.1f Hz interpolation failed: %s\n', freq, ME.message);
    end
end

% Validate interpolation quality
valid_bands = ~all(coupling_interpolated.MI_data == 0, 2);
fprintf('    ✓ Successfully interpolated %d/%d bands\n', sum(valid_bands), n_bands);

if sum(valid_bands) == 0
    coupling_interpolated = [];
end
end

function patches = identify_nonreward_coupling_patches(behavioral_matrix, coupling_data, session_id)
%% Identify non-reward periods with coupling data
% Format: [non-reward core] with optional reward buffers

% Extract reward-seeking behavior
if size(behavioral_matrix, 2) >= 7
    reward_seeking = behavioral_matrix(:, 4) | behavioral_matrix(:, 7);
else
    reward_seeking = behavioral_matrix(:, 4);
end

non_reward = ~reward_seeking;
patches = [];
fs = 1250;

% Minimum durations
min_nonreward_core = 10; % 10 seconds minimum non-reward core
reward_buffer = 0; % No required reward buffer

min_core_samples = min_nonreward_core * fs;
buffer_samples = reward_buffer * fs;

% Find non-reward core periods
core_starts = find(diff([0; non_reward]) == 1);
core_ends = find(diff([non_reward; 0]) == -1);

if length(core_starts) ~= length(core_ends)
    min_len = min(length(core_starts), length(core_ends));
    core_starts = core_starts(1:min_len);
    core_ends = core_ends(1:min_len);
end

fprintf('    Found %d potential non-reward cores\n', length(core_starts));

% Create extended patches with reward buffers
for p = 1:length(core_starts)
    core_start = core_starts(p);
    core_end = core_ends(p);
    core_length = core_end - core_start + 1;
    
    % Check if core is long enough
    if core_length < min_core_samples
        continue;
    end
    
    % Calculate extended patch boundaries (with reward buffers)
    patch_start = max(1, core_start - buffer_samples);
    patch_end = min(size(behavioral_matrix, 1), core_end + buffer_samples);
    
    % Verify we have enough data for buffers
    actual_pre_buffer = core_start - patch_start;
    actual_post_buffer = patch_end - core_end;
    
    % Extract breathing data for validation
    if size(behavioral_matrix, 2) >= 8
        breathing_freq = behavioral_matrix(patch_start:patch_end, 8);
        valid_breathing = ~isnan(breathing_freq) & breathing_freq > 0 & breathing_freq <= 20;
        
        % Require good breathing data throughout the patch
        if sum(valid_breathing) >= 0.6 * length(breathing_freq)
            
            % Extract coupling data for this patch
            patch_coupling = struct();
            n_bands = size(coupling_data.MI_data, 1);
            
            for band_idx = 1:n_bands
                band_MI = coupling_data.MI_data(band_idx, patch_start:patch_end);
                band_coherence = coupling_data.coherence_data(band_idx, patch_start:patch_end);
                
                % Check for valid coupling data
                valid_coupling = ~isnan(band_MI) & ~isnan(band_coherence);
                
                patch_coupling.(sprintf('band_%d_MI', band_idx)) = band_MI;
                patch_coupling.(sprintf('band_%d_coherence', band_idx)) = band_coherence;
                patch_coupling.(sprintf('band_%d_valid', band_idx)) = valid_coupling;
            end
            
            % Calculate durations
            total_duration = (patch_end - patch_start + 1) / fs;
            core_duration = core_length / fs;
            pre_buffer_duration = actual_pre_buffer / fs;
            post_buffer_duration = actual_post_buffer / fs;
            
            patch = struct();
            patch.session_id = session_id;
            patch.patch_id = p;
            
            % Extended patch info
            patch.start_idx = patch_start;
            patch.end_idx = patch_end;
            patch.duration_samples = patch_end - patch_start + 1;
            patch.duration_seconds = total_duration;
            
            % Core non-reward info
            patch.core_start_idx = core_start;
            patch.core_end_idx = core_end;
            patch.core_duration_seconds = core_duration;
            
            % Buffer info
            patch.pre_buffer_duration = pre_buffer_duration;
            patch.post_buffer_duration = post_buffer_duration;
            
            % Relative indices
            patch.core_start_relative = core_start - patch_start;
            patch.core_end_relative = core_end - patch_start;
            
            % Breathing and coupling data
            patch.breathing_data = breathing_freq;
            patch.valid_breathing_mask = valid_breathing;
            patch.coupling_data = patch_coupling;
            patch.valid = true;
            patch.patch_type = 'buffered_nonreward_coupling';
            
            patches = [patches; patch];
        end
    end
end

if ~isempty(patches)
    fprintf('    ✓ Created %d buffered coupling patches (%.1f±%.1fs total)\n', ...
        length(patches), mean([patches.duration_seconds]), std([patches.duration_seconds]));
end
end

%% ========================================================================
%% MULTI-BAND ANALYSIS FUNCTIONS WITH SESSION-SPECIFIC BASELINES
%% ========================================================================

function band_results = perform_multiband_analysis_with_session_specific_baselines(valid_patches, all_behavioral_data, breathing_bands)
%% Perform multi-band analysis with session-specific baselines
band_results = struct();

for band_idx = 1:length(breathing_bands)
    freq = breathing_bands(band_idx);
    fprintf('\nAnalyzing %.1f Hz coupling band with session-specific baselines...\n', freq);
    
    band_result = compute_band_analysis_session_specific(valid_patches, all_behavioral_data, band_idx, freq);
    
    % Store results
    band_results.(sprintf('band_%dHz', freq)) = band_result;
    band_results.(sprintf('band_%dHz', freq)).frequency = freq;
    band_results.(sprintf('band_%dHz', freq)).band_index = band_idx;
end
end

function band_result = compute_band_analysis_session_specific(valid_patches, all_behavioral_data, band_idx, freq)
%% Compute analysis for a single band using session-specific baselines
% Extract coupling data for this band from all patches
band_patches = extract_band_coupling_from_patches(valid_patches, band_idx);

if isempty(band_patches)
    fprintf('  ❌ No valid patches for %.1f Hz band\n', freq);
    band_result = [];
    return;
end

% Perform stretched analysis for this band with session-specific permutations
band_result = analyze_stretched_coupling_patches_with_session_specific_permutation(band_patches, all_behavioral_data, band_idx);
end

function results = analyze_stretched_coupling_patches_with_session_specific_permutation(band_patches, all_behavioral_data, band_idx)
%% Analyze stretched coupling patches with session-specific permutation testing
fprintf('  Stretching coupling patches...\n');

target_length = 100;
coupling_stretched = stretch_coupling_patches(band_patches, target_length);

if isempty(coupling_stretched)
    fprintf('  ❌ No valid stretched coupling patches\n');
    results = [];
    return;
end

fprintf('  ✓ Stretched %d coupling patches\n', size(coupling_stretched.MI_matrix, 1));

% Calculate coupling statistics
coupling_stats = calculate_coupling_trajectory_stats(coupling_stretched);

% Perform session-specific permutation test
fprintf('  Running session-specific permutation test...\n');
perm_results = permutation_test_session_specific_coupling_patches(band_patches, all_behavioral_data, target_length, band_idx);

% Combine results
results = struct();
results.coupling = coupling_stretched;
results.coupling_stats = coupling_stats;
results.permutation = perm_results;
results.target_length = target_length;

fprintf('  ✓ Session-specific coupling analysis complete\n');
end

function band_patches = extract_band_coupling_from_patches(patches, band_idx)
%% Extract coupling data for specific band from all patches
band_patches = [];

for p = 1:length(patches)
    patch = patches(p);
    
    % Extract coupling data for this band
    MI_field = sprintf('band_%d_MI', band_idx);
    coherence_field = sprintf('band_%d_coherence', band_idx);
    valid_field = sprintf('band_%d_valid', band_idx);
    
    if isfield(patch.coupling_data, MI_field) && isfield(patch.coupling_data, coherence_field)
        band_MI = patch.coupling_data.(MI_field);
        band_coherence = patch.coupling_data.(coherence_field);
        
        if isfield(patch.coupling_data, valid_field)
            valid_mask = patch.coupling_data.(valid_field);
        else
            valid_mask = ~isnan(band_MI) & ~isnan(band_coherence);
        end
        
        % Require sufficient valid data
        if sum(valid_mask) >= 0.3 * length(band_MI)
            band_patch = patch;
            band_patch.coupling_MI = band_MI;
            band_patch.coupling_coherence = band_coherence;
            band_patch.coupling_valid_mask = valid_mask;
            
            band_patches = [band_patches; band_patch];
        end
    end
end
end

%% ========================================================================
%% SESSION-SPECIFIC PERMUTATION FUNCTIONS
%% ========================================================================

function perm_results = permutation_test_session_specific_coupling_patches(band_patches, all_behavioral_data, target_length, band_idx)
%% Permutation test using session-specific random coupling patches
fprintf('    Generating session-specific permutation test for coupling data...\n');

n_permutations = 10; % Increased for better estimates
n_coupling_patches = length(band_patches);

% Get duration distribution and session info
coupling_durations = [band_patches.duration_seconds];
session_ids = [band_patches.session_id];
unique_sessions = unique(session_ids);

% Observed changes
coupling_stretched = stretch_coupling_patches(band_patches, target_length);
observed_stats = calculate_coupling_trajectory_stats(coupling_stretched);
observed_MI_change = observed_stats.overall_MI_change;
observed_coherence_change = observed_stats.overall_coherence_change;

fprintf('      Observed MI change: %.6f\n', observed_MI_change);
fprintf('      Observed coherence change: %.6f\n', observed_coherence_change);

% Initialize session-specific results
session_baselines = struct();
session_normalized_patches = [];

%% Calculate session-specific baselines
for session_idx = 1:length(unique_sessions)
    session_id = unique_sessions(session_idx);
    
    % Find patches from this session
    session_patch_idx = session_ids == session_id;
    session_patches = band_patches(session_patch_idx);
    session_durations = coupling_durations(session_patch_idx);
    
    if length(session_patches) < 2
        fprintf('      Session %d: Too few patches (%d), skipping\n', session_id, length(session_patches));
        continue;
    end
    
    fprintf('      Session %d: Generating baseline from %d patches\n', session_id, length(session_patches));
    
    % Find the corresponding behavioral data for this session
    session_behavioral_data = [];
    for bd_idx = 1:length(all_behavioral_data)
        if all_behavioral_data(bd_idx).session_id == session_id
            session_behavioral_data = all_behavioral_data(bd_idx);
            break;
        end
    end
    
    if isempty(session_behavioral_data)
        fprintf('      Session %d: No behavioral data found, skipping\n', session_id);
        continue;
    end
    
    % Generate session-specific random patches
    session_random_patches = generate_session_specific_random_patches(...
        session_behavioral_data, session_durations, n_permutations, band_idx);
    
    if isempty(session_random_patches)
        fprintf('      Session %d: Failed to generate random patches\n', session_id);
        continue;
    end
    
    % Calculate session baseline statistics
    session_baseline = calculate_session_baseline_stats(session_random_patches, target_length);
    
    if isempty(session_baseline)
        fprintf('      Session %d: Failed to calculate baseline stats\n', session_id);
        continue;
    end
    
    % Store session baseline
    session_baselines(session_idx).session_id = session_id;
    session_baselines(session_idx).baseline = session_baseline;
    session_baselines(session_idx).n_patches = length(session_patches);
    session_baselines(session_idx).n_random_patches = length(session_random_patches);
    
    % Normalize session patches by their baseline
    normalized_patches = normalize_patches_by_baseline(session_patches, session_baseline, target_length);
    session_normalized_patches = [session_normalized_patches; normalized_patches];
    
    fprintf('      Session %d: Baseline calculated (MI: %.6f±%.6f, Coh: %.6f±%.6f)\n', ...
        session_id, session_baseline.mean_MI_change, session_baseline.std_MI_change, ...
        session_baseline.mean_coherence_change, session_baseline.std_coherence_change);
end

%% Aggregate results across sessions
if isempty(session_normalized_patches)
    fprintf('      ❌ No session baselines calculated\n');
    perm_results = struct();
    return;
end

% Calculate statistics on normalized data
normalized_stretched = stretch_coupling_patches(session_normalized_patches, target_length);
normalized_stats = calculate_coupling_trajectory_stats(normalized_stretched);

% Calculate session-specific p-values and effect sizes
session_specific_results = calculate_session_specific_statistics(band_patches, session_baselines, target_length);

% Store comprehensive results
perm_results = struct();
perm_results.observed_MI_change = observed_MI_change;
perm_results.observed_coherence_change = observed_coherence_change;

% Session-specific baseline information
perm_results.session_baselines = session_baselines;
perm_results.n_sessions_with_baselines = length([session_baselines.session_id]);

% Normalized data results
perm_results.normalized_patches = session_normalized_patches;
perm_results.normalized_stats = normalized_stats;
perm_results.normalized_MI_change = normalized_stats.overall_MI_change;
perm_results.normalized_coherence_change = normalized_stats.overall_coherence_change;

% Session-specific statistical tests
perm_results.session_specific = session_specific_results;

% Overall effect sizes (normalized)
perm_results.MI_effect_size = session_specific_results.overall_MI_effect_size;
perm_results.coherence_effect_size = session_specific_results.overall_coherence_effect_size;
perm_results.MI_p_value = session_specific_results.overall_MI_p_value;
perm_results.coherence_p_value = session_specific_results.overall_coherence_p_value;

% For compatibility with existing visualization code
perm_results.MI_p_value_two_tailed = session_specific_results.overall_MI_p_value;
perm_results.coherence_p_value_two_tailed = session_specific_results.overall_coherence_p_value;
perm_results.n_permutations = n_permutations;

% Trajectory confidence intervals (using aggregated session baselines)
perm_results = add_trajectory_confidence_intervals(perm_results, session_baselines, target_length);

fprintf('      ✓ Session-specific analysis complete (%d sessions)\n', length([session_baselines.session_id]));
end

function session_random_patches = generate_session_specific_random_patches(session_behavioral_data, target_durations, n_permutations, band_idx)
%% Generate random patches from a single session's data
fs = 1250;
total_patches_needed = n_permutations * length(target_durations);
safety_factor = 5; % Higher safety factor for single session

session_random_patches = [];
attempts = 0;
max_attempts = total_patches_needed * safety_factor;

behavioral_matrix = session_behavioral_data.behavioral_matrix;
coupling_data = session_behavioral_data.coupling_data;

while length(session_random_patches) < total_patches_needed && attempts < max_attempts
    attempts = attempts + 1;
    
    % Show progress for long computations
    if mod(attempts, round(max_attempts/10)) == 0
        progress = round(length(session_random_patches) / total_patches_needed * 100);
        success_rate = length(session_random_patches) / attempts * 100;
        fprintf('          Session progress: %d%% (%d/%d patches, %.1f%% success)\n', ...
            progress, length(session_random_patches), total_patches_needed, success_rate);
    end
    
    % Randomly select duration
    duration = target_durations(randi(length(target_durations)));
    target_samples = round(duration * fs);
    
    % Check if session is long enough
    max_start = size(behavioral_matrix, 1) - target_samples + 1;
    if max_start < 1
        continue;
    end
    
    % Random start position
    start_idx = randi(max_start);
    end_idx = start_idx + target_samples - 1;
    
    % Extract coupling data for this band
    if size(coupling_data.MI_data, 1) >= band_idx
        coupling_MI = coupling_data.MI_data(band_idx, start_idx:end_idx);
        coupling_coherence = coupling_data.coherence_data(band_idx, start_idx:end_idx);
        valid_mask = ~isnan(coupling_MI) & ~isnan(coupling_coherence);
        
        if sum(valid_mask) >= 0.3 * length(coupling_MI)
            patch = struct();
            patch.coupling_MI = coupling_MI;
            patch.coupling_coherence = coupling_coherence;
            patch.coupling_valid_mask = valid_mask;
            patch.duration_seconds = duration;
            patch.session_id = session_behavioral_data.session_id;
            patch.valid = true;
            
            session_random_patches = [session_random_patches; patch];
        end
    end
end

fprintf('          Generated %d/%d session-specific random patches (%.1f%% success)\n', ...
    length(session_random_patches), total_patches_needed, ...
    length(session_random_patches)/attempts*100);
end

function baseline_stats = calculate_session_baseline_stats(random_patches, target_length)
%% Calculate baseline statistics from session-specific random patches
if isempty(random_patches)
    baseline_stats = [];
    return;
end

% Stretch random patches
random_stretched = stretch_coupling_patches(random_patches, target_length);

if isempty(random_stretched) || isempty(random_stretched.MI_matrix)
    baseline_stats = [];
    return;
end

% Calculate initial vs final changes for all random patches
MI_matrix = random_stretched.MI_matrix;
coherence_matrix = random_stretched.coherence_matrix;

initial_MI = mean(MI_matrix(:, 1:10), 2);
final_MI = mean(MI_matrix(:, end-9:end), 2);
MI_changes = final_MI - initial_MI;

initial_coherence = mean(coherence_matrix(:, 1:10), 2);
final_coherence = mean(coherence_matrix(:, end-9:end), 2);
coherence_changes = final_coherence - initial_coherence;

% Store baseline statistics
baseline_stats = struct();
baseline_stats.MI_changes = MI_changes;
baseline_stats.coherence_changes = coherence_changes;
baseline_stats.mean_MI_change = mean(MI_changes);
baseline_stats.std_MI_change = std(MI_changes);
baseline_stats.mean_coherence_change = mean(coherence_changes);
baseline_stats.std_coherence_change = std(coherence_changes);
baseline_stats.mean_MI_trajectory = mean(MI_matrix, 1);
baseline_stats.mean_coherence_trajectory = mean(coherence_matrix, 1);
baseline_stats.MI_trajectory_std = std(MI_matrix, 1);
baseline_stats.coherence_trajectory_std = std(coherence_matrix, 1);
baseline_stats.n_random_patches = size(MI_matrix, 1);
end

function normalized_patches = normalize_patches_by_baseline(patches, baseline, target_length)
%% Normalize patches by subtracting session-specific baseline
normalized_patches = [];

for p = 1:length(patches)
    patch = patches(p);
    
    % Stretch the patch
    stretched_patch = stretch_coupling_patches(patch, target_length);
    
    if isempty(stretched_patch) || size(stretched_patch.MI_matrix, 1) == 0
        continue;
    end
    
    % Normalize by subtracting baseline mean trajectory
    normalized_MI = stretched_patch.MI_matrix(1, :) - baseline.mean_MI_trajectory;
    normalized_coherence = stretched_patch.coherence_matrix(1, :) - baseline.mean_coherence_trajectory;
    
    % Create normalized patch
    norm_patch = patch;
    norm_patch.coupling_MI = normalized_MI;
    norm_patch.coupling_coherence = normalized_coherence;
    norm_patch.coupling_valid_mask = true(size(normalized_MI));
    norm_patch.is_normalized = true;
    norm_patch.baseline_MI_change = baseline.mean_MI_change;
    norm_patch.baseline_coherence_change = baseline.mean_coherence_change;
    normalized_patches = [normalized_patches; norm_patch];
end
end

function session_stats = calculate_session_specific_statistics(patches, session_baselines, target_length)
%% Calculate session-specific statistical tests
session_stats = struct();

if isempty(session_baselines)
    return;
end

% Initialize arrays for aggregated statistics
all_normalized_MI_changes = [];
all_normalized_coherence_changes = [];
all_raw_MI_changes = [];
all_raw_coherence_changes = [];
session_effect_sizes_MI = [];
session_effect_sizes_coherence = [];
session_p_values_MI = [];
session_p_values_coherence = [];

% Process each session
session_ids = [session_baselines.session_id];
for i = 1:length(session_baselines)
    session_id = session_ids(i);
    baseline = session_baselines(i).baseline;
    
    % Find patches from this session
    session_patch_idx = [patches.session_id] == session_id;
    session_patches = patches(session_patch_idx);
    
    if isempty(session_patches)
        continue;
    end
    
    % Calculate observed changes for this session
    session_stretched = stretch_coupling_patches(session_patches, target_length);
    if isempty(session_stretched)
        continue;
    end
    
    session_observed_stats = calculate_coupling_trajectory_stats(session_stretched);
    observed_MI_change = session_observed_stats.overall_MI_change;
    observed_coherence_change = session_observed_stats.overall_coherence_change;
    
    % Calculate session-specific effect sizes and p-values
    MI_effect_size = (observed_MI_change - baseline.mean_MI_change) / baseline.std_MI_change;
    coherence_effect_size = (observed_coherence_change - baseline.mean_coherence_change) / baseline.std_coherence_change;
    
    % Calculate p-values based on baseline distribution
    MI_p_value = calculate_empirical_p_value(observed_MI_change, baseline.MI_changes);
    coherence_p_value = calculate_empirical_p_value(observed_coherence_change, baseline.coherence_changes);
    
    % Store session results
    session_stats.sessions(i).session_id = session_id;
    session_stats.sessions(i).observed_MI_change = observed_MI_change;
    session_stats.sessions(i).observed_coherence_change = observed_coherence_change;
    session_stats.sessions(i).normalized_MI_change = observed_MI_change - baseline.mean_MI_change;
    session_stats.sessions(i).normalized_coherence_change = observed_coherence_change - baseline.mean_coherence_change;
    session_stats.sessions(i).MI_effect_size = MI_effect_size;
    session_stats.sessions(i).coherence_effect_size = coherence_effect_size;
    session_stats.sessions(i).MI_p_value = MI_p_value;
    session_stats.sessions(i).coherence_p_value = coherence_p_value;
    session_stats.sessions(i).n_patches = length(session_patches);
    
    % Aggregate for overall statistics
    all_normalized_MI_changes = [all_normalized_MI_changes; observed_MI_change - baseline.mean_MI_change];
    all_normalized_coherence_changes = [all_normalized_coherence_changes; observed_coherence_change - baseline.mean_coherence_change];
    all_raw_MI_changes = [all_raw_MI_changes; observed_MI_change];
    all_raw_coherence_changes = [all_raw_coherence_changes; observed_coherence_change];
    session_effect_sizes_MI = [session_effect_sizes_MI; MI_effect_size];
    session_effect_sizes_coherence = [session_effect_sizes_coherence; coherence_effect_size];
    session_p_values_MI = [session_p_values_MI; MI_p_value];
    session_p_values_coherence = [session_p_values_coherence; coherence_p_value];
end

% Overall aggregated statistics
session_stats.overall_normalized_MI_change = mean(all_normalized_MI_changes);
session_stats.overall_normalized_coherence_change = mean(all_normalized_coherence_changes);
session_stats.overall_MI_effect_size = mean(session_effect_sizes_MI);
session_stats.overall_coherence_effect_size = mean(session_effect_sizes_coherence);

% Meta-analysis across sessions using Fisher's method for p-values
session_stats.overall_MI_p_value = fishers_combined_p_test(session_p_values_MI);
session_stats.overall_coherence_p_value = fishers_combined_p_test(session_p_values_coherence);

% Consistency across sessions
session_stats.MI_effect_size_std = std(session_effect_sizes_MI);
session_stats.coherence_effect_size_std = std(session_effect_sizes_coherence);
session_stats.n_sessions = length(session_baselines);
session_stats.n_significant_sessions_MI = sum(session_p_values_MI < 0.05);
session_stats.n_significant_sessions_coherence = sum(session_p_values_coherence < 0.05);
end

function p_value = calculate_empirical_p_value(observed_value, baseline_distribution)
%% Calculate empirical p-value from baseline distribution
if isempty(baseline_distribution)
    p_value = NaN;
    return;
end

% Two-tailed test
n_permutations = length(baseline_distribution);
n_extreme = sum(abs(baseline_distribution) >= abs(observed_value));
p_value = n_extreme / n_permutations;

% Ensure p-value is not exactly 0
if p_value == 0
    p_value = 1 / (n_permutations + 1);
end
end

function combined_p = fishers_combined_p_test(p_values)
%% Fisher's method for combining p-values across sessions
valid_p = p_values(~isnan(p_values) & p_values > 0);

if isempty(valid_p)
    combined_p = NaN;
    return;
end

% Fisher's chi-square statistic
chi_square_stat = -2 * sum(log(valid_p));
df = 2 * length(valid_p);

% Calculate p-value using chi-square distribution
combined_p = 1 - chi2cdf(chi_square_stat, df);
end

function perm_results = add_trajectory_confidence_intervals(perm_results, session_baselines, target_length)
%% Add trajectory confidence intervals based on session baselines
if isempty(session_baselines)
    return;
end

% Aggregate baseline trajectories
all_MI_trajectories = [];
all_coherence_trajectories = [];

for i = 1:length(session_baselines)
    baseline = session_baselines(i).baseline;
    if ~isempty(baseline)
        all_MI_trajectories = [all_MI_trajectories; baseline.mean_MI_trajectory];
        all_coherence_trajectories = [all_coherence_trajectories; baseline.mean_coherence_trajectory];
    end
end

if ~isempty(all_MI_trajectories)
    perm_results.perm_MI_mean_trajectory = mean(all_MI_trajectories, 1);
    perm_results.perm_coherence_mean_trajectory = mean(all_coherence_trajectories, 1);
    perm_results.MI_ci_lower = prctile(all_MI_trajectories, 2.5, 1);
    perm_results.MI_ci_upper = prctile(all_MI_trajectories, 97.5, 1);
    perm_results.coherence_ci_lower = prctile(all_coherence_trajectories, 2.5, 1);
    perm_results.coherence_ci_upper = prctile(all_coherence_trajectories, 97.5, 1);
else
    % Fallback to empty arrays
    perm_results.perm_MI_mean_trajectory = zeros(1, target_length);
    perm_results.perm_coherence_mean_trajectory = zeros(1, target_length);
    perm_results.MI_ci_lower = zeros(1, target_length);
    perm_results.MI_ci_upper = zeros(1, target_length);
    perm_results.coherence_ci_lower = zeros(1, target_length);
    perm_results.coherence_ci_upper = zeros(1, target_length);
end
end

%% ========================================================================
%% STATISTICAL ANALYSIS FUNCTIONS
%% ========================================================================

function stretched = stretch_coupling_patches(patches, target_length)
%% Stretch all coupling patches to common length
stretched = struct();
stretched_MI_matrix = [];
stretched_coherence_matrix = [];
patch_info = [];

for p = 1:length(patches)
    patch = patches(p);
    coupling_MI = patch.coupling_MI;
    coupling_coherence = patch.coupling_coherence;
    valid_mask = patch.coupling_valid_mask;
    
    if sum(valid_mask) < 10
        continue;
    end
    
    % Extract valid coupling data
    valid_MI = coupling_MI(valid_mask);
    valid_coherence = coupling_coherence(valid_mask);
    valid_indices = find(valid_mask);
    
    % Create normalized time vector
    time_normalized = (valid_indices - valid_indices(1)) / (valid_indices(end) - valid_indices(1));
    target_time = linspace(0, 1, target_length);
    
    try
        % Interpolate both MI and coherence
        stretched_MI = interp1(time_normalized, valid_MI, target_time, 'linear', 'extrap');
        stretched_coherence = interp1(time_normalized, valid_coherence, target_time, 'linear', 'extrap');
        
        % Store results
        stretched_MI_matrix(end+1, :) = stretched_MI;
        stretched_coherence_matrix(end+1, :) = stretched_coherence;
        
        info = struct();
        info.session_id = patch.session_id;
        info.patch_id = patch.patch_id;
        info.duration = patch.duration_seconds;
        info.effective_duration = (valid_indices(end) - valid_indices(1)) / 1250;
        patch_info = [patch_info; info];
        
    catch
        continue;
    end
end

stretched.MI_matrix = stretched_MI_matrix;
stretched.coherence_matrix = stretched_coherence_matrix;
stretched.patch_info = patch_info;
stretched.n_patches = size(stretched_MI_matrix, 1);
end

function stats = calculate_coupling_trajectory_stats(stretched)
%% Calculate statistics for stretched coupling trajectories
MI_matrix = stretched.MI_matrix;
coherence_matrix = stretched.coherence_matrix;

if isempty(MI_matrix)
    stats = struct();
    return;
end

% MI statistics
stats.mean_MI_trajectory = mean(MI_matrix, 1);
stats.std_MI_trajectory = std(MI_matrix, 1);
stats.sem_MI_trajectory = stats.std_MI_trajectory / sqrt(size(MI_matrix, 1));

% Coherence statistics  
stats.mean_coherence_trajectory = mean(coherence_matrix, 1);
stats.std_coherence_trajectory = std(coherence_matrix, 1);
stats.sem_coherence_trajectory = stats.std_coherence_trajectory / sqrt(size(coherence_matrix, 1));

% Initial vs final comparison for MI
initial_MI = mean(MI_matrix(:, 1:10), 2);
final_MI = mean(MI_matrix(:, end-9:end), 2);

stats.initial_MI_mean = mean(initial_MI);
stats.final_MI_mean = mean(final_MI);
stats.overall_MI_change = stats.final_MI_mean - stats.initial_MI_mean;

% Statistical test for MI
[~, stats.MI_p_value, ~, test_stats] = ttest(initial_MI, final_MI);
stats.MI_t_stat = test_stats.tstat;
stats.MI_df = test_stats.df;

% Initial vs final comparison for coherence
initial_coherence = mean(coherence_matrix(:, 1:10), 2);
final_coherence = mean(coherence_matrix(:, end-9:end), 2);

stats.initial_coherence_mean = mean(initial_coherence);
stats.final_coherence_mean = mean(final_coherence);
stats.overall_coherence_change = stats.final_coherence_mean - stats.initial_coherence_mean;

% Statistical test for coherence
[~, stats.coherence_p_value, ~, test_stats] = ttest(initial_coherence, final_coherence);
stats.coherence_t_stat = test_stats.tstat;
stats.coherence_df = test_stats.df;

% Trajectory features for MI
[~, stats.MI_peak_idx] = max(stats.mean_MI_trajectory);
[~, stats.MI_trough_idx] = min(stats.mean_MI_trajectory);
stats.MI_peak_location = stats.MI_peak_idx / length(stats.mean_MI_trajectory) * 100;
stats.MI_trough_location = stats.MI_trough_idx / length(stats.mean_MI_trajectory) * 100;

% Linear trends
time_norm = linspace(0, 1, length(stats.mean_MI_trajectory));
MI_poly_fit = polyfit(time_norm, stats.mean_MI_trajectory, 1);
coherence_poly_fit = polyfit(time_norm, stats.mean_coherence_trajectory, 1);

stats.MI_linear_slope = MI_poly_fit(1);
stats.coherence_linear_slope = coherence_poly_fit(1);
end

%% ========================================================================
%% VISUALIZATION FUNCTIONS WITH SESSION-SPECIFIC RESULTS
%% ========================================================================

function print_multiband_coupling_summary_session_specific(band_results, breathing_bands)
%% Print comprehensive summary for all bands with session-specific baseline normalization
fprintf('\n=== MULTI-BAND COUPLING ANALYSIS WITH SESSION-SPECIFIC BASELINES ===\n');

band_names = fieldnames(band_results);
n_bands = length(band_names);

for i = 1:n_bands
    band_name = band_names{i};
    result = band_results.(band_name);
    
    if isempty(result)
        continue;
    end
    
    freq = result.frequency;
    fprintf('\n--- %.0f Hz BREATHING-GAMMA COUPLING (SESSION-SPECIFIC ANALYSIS) ---\n', freq);
    
    if ~isempty(result.coupling) && ~isempty(result.coupling_stats)
        stats = result.coupling_stats;
        fprintf('Patches analyzed: %d\n', result.coupling.n_patches);
        
        fprintf('\nModulation Index (MI) - Raw Values:\n');
        fprintf('  Initial: %.6f, Final: %.6f\n', stats.initial_MI_mean, stats.final_MI_mean);
        fprintf('  Change: %.6f\n', stats.overall_MI_change);
        fprintf('  t-test: t(%d) = %.3f, p = %.4f', stats.MI_df, stats.MI_t_stat, stats.MI_p_value);
        if stats.MI_p_value < 0.05
            fprintf(' *\n');
        else
            fprintf(' (ns)\n');
        end
        
        fprintf('\nCoherence - Raw Values:\n');
        fprintf('  Initial: %.6f, Final: %.6f\n', stats.initial_coherence_mean, stats.final_coherence_mean);
        fprintf('  Change: %.6f\n', stats.overall_coherence_change);
        fprintf('  t-test: t(%d) = %.3f, p = %.4f', stats.coherence_df, stats.coherence_t_stat, stats.coherence_p_value);
        if stats.coherence_p_value < 0.05
            fprintf(' *\n');
        else
            fprintf(' (ns)\n');
        end
    end
    
    if ~isempty(result.permutation) && isfield(result.permutation, 'session_specific')
        perm = result.permutation;
        session_stats = perm.session_specific;
        
        fprintf('\nSession-Specific Baseline Analysis:\n');
        fprintf('  Sessions with baselines: %d\n', perm.n_sessions_with_baselines);
        
        if isfield(session_stats, 'sessions') && ~isempty(session_stats.sessions)
            fprintf('  Per-session results:\n');
            for s = 1:length(session_stats.sessions)
                sess = session_stats.sessions(s);
                fprintf('    Session %d: MI change = %.6f (ES=%.2f, p=%.3f), Coh change = %.6f (ES=%.2f, p=%.3f)\n', ...
                    sess.session_id, sess.normalized_MI_change, sess.MI_effect_size, sess.MI_p_value, ...
                    sess.normalized_coherence_change, sess.coherence_effect_size, sess.coherence_p_value);
            end
        end
        
        fprintf('\nNormalized (Baseline-Corrected) Results:\n');
        fprintf('  MI: normalized change = %.6f, effect size = %.3f', ...
            session_stats.overall_normalized_MI_change, session_stats.overall_MI_effect_size);
        if session_stats.overall_MI_p_value < 0.05
            fprintf(' *\n');
        else
            fprintf(' (ns)\n');
        end
        fprintf('    Combined p-value: %.4f\n', session_stats.overall_MI_p_value);
        fprintf('    Significant sessions: %d/%d\n', session_stats.n_significant_sessions_MI, session_stats.n_sessions);
        
        fprintf('  Coherence: normalized change = %.6f, effect size = %.3f', ...
            session_stats.overall_normalized_coherence_change, session_stats.overall_coherence_effect_size);
        if session_stats.overall_coherence_p_value < 0.05
            fprintf(' *\n');
        else
            fprintf(' (ns)\n');
        end
        fprintf('    Combined p-value: %.4f\n', session_stats.overall_coherence_p_value);
        fprintf('    Significant sessions: %d/%d\n', session_stats.n_significant_sessions_coherence, session_stats.n_sessions);
        
        fprintf('\nConsistency Across Sessions:\n');
        fprintf('  MI effect size std: %.3f\n', session_stats.MI_effect_size_std);
        fprintf('  Coherence effect size std: %.3f\n', session_stats.coherence_effect_size_std);
    end
end

% Overall summary
fprintf('\n=== OVERALL FINDINGS (SESSION-SPECIFIC BASELINE CORRECTED) ===\n');
significant_MI_bands = 0;
significant_coherence_bands = 0;
strongest_MI_band = '';
strongest_coherence_band = '';
max_MI_effect = 0;
max_coherence_effect = 0;

for i = 1:n_bands
    band_name = band_names{i};
    result = band_results.(band_name);
    
    if isempty(result) || isempty(result.permutation) || ~isfield(result.permutation, 'session_specific')
        continue;
    end
    
    session_stats = result.permutation.session_specific;
    freq = result.frequency;
    
    if session_stats.overall_MI_p_value < 0.05
        significant_MI_bands = significant_MI_bands + 1;
    end
    
    if session_stats.overall_coherence_p_value < 0.05
        significant_coherence_bands = significant_coherence_bands + 1;
    end
    
    if abs(session_stats.overall_MI_effect_size) > max_MI_effect
        max_MI_effect = abs(session_stats.overall_MI_effect_size);
        strongest_MI_band = sprintf('%.0f Hz', freq);
    end
    
    if abs(session_stats.overall_coherence_effect_size) > max_coherence_effect
        max_coherence_effect = abs(session_stats.overall_coherence_effect_size);
        strongest_coherence_band = sprintf('%.0f Hz', freq);
    end
end

fprintf('Significant bands after baseline correction (p < 0.05):\n');
fprintf('  MI: %d/%d bands\n', significant_MI_bands, n_bands);
fprintf('  Coherence: %d/%d bands\n', significant_coherence_bands, n_bands);

if ~isempty(strongest_MI_band)
    fprintf('\nStrongest baseline-corrected effects:\n');
    fprintf('  MI: %s (effect size = %.3f)\n', strongest_MI_band, max_MI_effect);
    fprintf('  Coherence: %s (effect size = %.3f)\n', strongest_coherence_band, max_coherence_effect);
end

fprintf('\n=== SESSION-SPECIFIC ANALYSIS COMPLETE ===\n');
end

function visualize_session_specific_multiband_analysis(band_results, breathing_bands)
%% Create visualization showing session-specific baseline-corrected results
fig = figure('Name', 'Session-Specific Baseline-Corrected Multi-Band Analysis', 'Position', [50, 50, 2000, 1400]);

band_names = fieldnames(band_results);
n_bands = length(band_names);

% Create proper time axis
total_time_seconds = 14;
time_seconds = linspace(0, total_time_seconds, 100);

%% 1. Raw vs Baseline-Corrected Comparison
subplot(3, 4, [1, 2]);
colors = lines(n_bands);
hold on;

for i = 1:n_bands
    band_name = band_names{i};
    result = band_results.(band_name);
    
    if isempty(result) || isempty(result.coupling_stats) || isempty(result.permutation)
        continue;
    end
    
    stats = result.coupling_stats;
    perm = result.permutation;
    freq = result.frequency;
    
    % Plot raw trajectory
    plot(time_seconds, stats.mean_MI_trajectory, '--', 'Color', colors(i,:), 'LineWidth', 2, ...
        'DisplayName', sprintf('%.0f Hz Raw', freq));
    
    % Plot baseline-corrected trajectory if available
    if isfield(perm, 'normalized_stats') && ~isempty(perm.normalized_stats)
        plot(time_seconds, perm.normalized_stats.mean_MI_trajectory, '-', 'Color', colors(i,:), 'LineWidth', 3, ...
            'DisplayName', sprintf('%.0f Hz Corrected', freq));
    end
end

% Highlight core non-reward region
ylims = ylim;
fill([2, 12, 12, 2], [ylims(1), ylims(1), ylims(2), ylims(2)], [1, 1, 0.8], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.2);

xlabel('Time (seconds)');
ylabel('Modulation Index');
title('Raw vs Baseline-Corrected MI Trajectories');
legend('Location', 'best', 'FontSize', 8);
grid on;

%% 2. Session Consistency Plot
subplot(3, 4, [3, 4]);
hold on;

session_effect_sizes_MI = [];
session_effect_sizes_coh = [];
session_labels = {};
band_colors = [];

for i = 1:n_bands
    band_name = band_names{i};
    result = band_results.(band_name);
    
    if isempty(result) || isempty(result.permutation) || ...
       ~isfield(result.permutation, 'session_specific') || ...
       ~isfield(result.permutation.session_specific, 'sessions')
        continue;
    end
    
    sessions = result.permutation.session_specific.sessions;
    freq = result.frequency;
    
    for s = 1:length(sessions)
        session_effect_sizes_MI = [session_effect_sizes_MI; sessions(s).MI_effect_size];
        session_effect_sizes_coh = [session_effect_sizes_coh; sessions(s).coherence_effect_size];
        session_labels{end+1} = sprintf('S%d-%.0fHz', sessions(s).session_id, freq);
        band_colors = [band_colors; colors(i,:)];
    end
end

if ~isempty(session_effect_sizes_MI)
    scatter(session_effect_sizes_MI, session_effect_sizes_coh, 60, band_colors, 'filled');
    
    % Add reference lines
    line([-3, 3], [0, 0], 'Color', 'k', 'LineStyle', '--');
    line([0, 0], [-3, 3], 'Color', 'k', 'LineStyle', '--');
    
    xlabel('MI Effect Size');
    ylabel('Coherence Effect Size');
    title('Session-Specific Effect Sizes');
    grid on;
    
    % Add quadrant labels
    text(2, 2, 'Both↑', 'FontSize', 10, 'HorizontalAlignment', 'center');
    text(-2, 2, 'MI↓ Coh↑', 'FontSize', 10, 'HorizontalAlignment', 'center');
    text(-2, -2, 'Both↓', 'FontSize', 10, 'HorizontalAlignment', 'center');
    text(2, -2, 'MI↑ Coh↓', 'FontSize', 10, 'HorizontalAlignment', 'center');
end

%% 3-10. Individual band session-specific analysis
plot_positions = [5, 6, 7, 8, 9, 10, 11, 12];

for i = 1:min(n_bands, length(plot_positions))
    band_name = band_names{i};
    result = band_results.(band_name);
    
    if isempty(result) || isempty(result.permutation) || ...
       ~isfield(result.permutation, 'session_specific')
        continue;
    end
    
    subplot(3, 4, plot_positions(i));
    plot_session_specific_band_results(result, time_seconds);
end

sgtitle('Session-Specific Baseline-Corrected Multi-Band Coupling Analysis', ...
    'FontSize', 16, 'FontWeight', 'bold');

set(fig, 'Color', 'w');
end

function plot_session_specific_band_results(result, time_seconds)
%% Plot session-specific results for individual band
hold on;

stats = result.coupling_stats;
perm = result.permutation;
freq = result.frequency;

% Plot confidence intervals from session baselines
if isfield(perm, 'MI_ci_lower') && ~isempty(perm.MI_ci_lower)
    fill([time_seconds, fliplr(time_seconds)], [perm.MI_ci_upper, fliplr(perm.MI_ci_lower)], ...
        [0.9, 0.9, 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Session baseline 95% CI');
end

% Plot session baseline mean
if isfield(perm, 'perm_MI_mean_trajectory') && ~isempty(perm.perm_MI_mean_trajectory)
    plot(time_seconds, perm.perm_MI_mean_trajectory, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Session baseline mean');
end

% Plot observed trajectory
plot(time_seconds, stats.mean_MI_trajectory, 'r-', 'LineWidth', 3, 'DisplayName', 'Observed');

% Plot baseline-corrected trajectory if available
if isfield(perm, 'normalized_stats') && ~isempty(perm.normalized_stats)
    plot(time_seconds, perm.normalized_stats.mean_MI_trajectory, 'b-', 'LineWidth', 3, 'DisplayName', 'Baseline-corrected');
end

% Highlight core region
ylims = ylim;
fill([2, 12, 12, 2], [ylims(1), ylims(1), ylims(2), ylims(2)], [1, 1, 0.8], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.2);

xlabel('Time (seconds)');
ylabel('Modulation Index');
title(sprintf('%.0f Hz Session-Specific Analysis', freq));
legend('Location', 'best', 'FontSize', 7);
grid on;

% Add statistical annotation
if isfield(perm, 'session_specific') && ~isempty(perm.session_specific)
    session_stats = perm.session_specific;
    
    % Create annotation text
    annotation_text = {
        sprintf('Sessions: %d', session_stats.n_sessions);
        sprintf('Sig: %d/%d', session_stats.n_significant_sessions_MI, session_stats.n_sessions);
        sprintf('ES: %.2f±%.2f', session_stats.overall_MI_effect_size, session_stats.MI_effect_size_std);
        sprintf('p: %.3f', session_stats.overall_MI_p_value);
    };
    
    % Determine significance color
    if session_stats.overall_MI_p_value < 0.05
        text_color = 'red';
    else
        text_color = 'black';
    end
    
    text(0.02, 0.98, annotation_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontWeight', 'bold', 'Color', text_color, 'BackgroundColor', 'white', ...
        'EdgeColor', text_color, 'FontSize', 8);
end
end

%% ========================================================================
%% ORIGINAL VISUALIZATION FUNCTIONS (KEPT FOR COMPATIBILITY)
%% ========================================================================

function visualize_multiband_coupling_analysis_simplified(band_results, breathing_bands)
%% Create visualization showing all 4 bands individually + multi-band summaries
fig = figure('Name', 'Multi-Band Coupling Non-Reward Analysis', 'Position', [50, 50, 2000, 1200]);

band_names = fieldnames(band_results);
n_bands = length(band_names);

% Create proper time axis
total_time_seconds = 14; % Typical patch duration
time_seconds = linspace(0, total_time_seconds, 100); % target_length = 100

%% 1. Multi-band MI trajectories comparison
subplot(4, 4, [1, 2]);
colors = lines(n_bands);
hold on;

for i = 1:n_bands
    band_name = band_names{i};
    result = band_results.(band_name);
    
    if isempty(result) || isempty(result.coupling_stats)
        continue;
    end
    
    stats = result.coupling_stats;
    freq = result.frequency;
    
    % Plot mean trajectory with confidence interval
    upper_ci = stats.mean_MI_trajectory + 1.96 * stats.sem_MI_trajectory;
    lower_ci = stats.mean_MI_trajectory - 1.96 * stats.sem_MI_trajectory;
    
    fill([time_seconds, fliplr(time_seconds)], [upper_ci, fliplr(lower_ci)], ...
        colors(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    plot(time_seconds, stats.mean_MI_trajectory, 'Color', colors(i,:), 'LineWidth', 3, ...
        'DisplayName', sprintf('%.0f Hz', freq));
end

% Highlight core non-reward region
ylims = ylim;
fill([2, 12, 12, 2], [ylims(1), ylims(1), ylims(2), ylims(2)], [1, 1, 0.8], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.2);

xlabel('Time (seconds)');
ylabel('Modulation Index');
title('Multi-Band Coupling During Non-Reward Periods');
legend('Location', 'best');
grid on;

%% 2. Multi-band coherence trajectories
subplot(4, 4, [5, 6]);
hold on;

for i = 1:n_bands
    band_name = band_names{i};
    result = band_results.(band_name);
    
    if isempty(result) || isempty(result.coupling_stats)
        continue;
    end
    
    stats = result.coupling_stats;
    freq = result.frequency;
    
    upper_ci = stats.mean_coherence_trajectory + 1.96 * stats.sem_coherence_trajectory;
    lower_ci = stats.mean_coherence_trajectory - 1.96 * stats.sem_coherence_trajectory;
    
    fill([time_seconds, fliplr(time_seconds)], [upper_ci, fliplr(lower_ci)], ...
        colors(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    plot(time_seconds, stats.mean_coherence_trajectory, 'Color', colors(i,:), 'LineWidth', 3, ...
        'DisplayName', sprintf('%.0f Hz', freq));
end

% Highlight core region
ylims = ylim;
fill([2, 12, 12, 2], [ylims(1), ylims(1), ylims(2), ylims(2)], [1, 1, 0.8], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.2);

xlabel('Time (seconds)');
ylabel('Coherence');
title('Multi-Band Coherence During Non-Reward Periods');
legend('Location', 'best');
grid on;

%% 3-10. Individual band analysis for ALL bands (MI and Coherence for each)
subplot_positions = [3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; % Positions for individual plots

plot_idx = 1;
for i = 1:n_bands
    band_name = band_names{i};
    result = band_results.(band_name);
    
    if isempty(result) || isempty(result.coupling_stats) || isempty(result.permutation)
        continue;
    end
    
    % Plot MI analysis for this band
    if plot_idx <= length(subplot_positions)
        subplot(4, 4, subplot_positions(plot_idx));
        plot_detailed_band_analysis(result.coupling_stats, result.permutation, ...
            time_seconds, result.frequency, 'MI');
        plot_idx = plot_idx + 1;
    end
    
    % Plot Coherence analysis for this band
    if plot_idx <= length(subplot_positions)
        subplot(4, 4, subplot_positions(plot_idx));
        plot_detailed_band_analysis(result.coupling_stats, result.permutation, ...
            time_seconds, result.frequency, 'Coherence');
        plot_idx = plot_idx + 1;
    end
end

sgtitle('Multi-Band Breathing-Gamma Coupling Analysis During Non-Reward Periods', ...
    'FontSize', 16, 'FontWeight', 'bold');

% Adjust layout for better appearance
set(fig, 'Color', 'w');
end

function plot_detailed_band_analysis(stats, perm, time_seconds, frequency, measure_type)
%% Plot detailed analysis for individual band with enhanced visualization
if strcmp(measure_type, 'MI')
    observed_trajectory = stats.mean_MI_trajectory;
    observed_std = stats.sem_MI_trajectory;
    
    if isfield(perm, 'perm_MI_mean_trajectory') && ~isempty(perm.perm_MI_mean_trajectory)
        perm_trajectory = perm.perm_MI_mean_trajectory;
        ci_lower = perm.MI_ci_lower;
        ci_upper = perm.MI_ci_upper;
    else
        perm_trajectory = zeros(size(observed_trajectory));
        ci_lower = zeros(size(observed_trajectory));
        ci_upper = zeros(size(observed_trajectory));
    end
    
    observed_change = perm.observed_MI_change;
    p_value = perm.MI_p_value;
    effect_size = perm.MI_effect_size;
    ylabel_text = 'Modulation Index';
else
    observed_trajectory = stats.mean_coherence_trajectory;
    observed_std = stats.sem_coherence_trajectory;
    
    if isfield(perm, 'perm_coherence_mean_trajectory') && ~isempty(perm.perm_coherence_mean_trajectory)
        perm_trajectory = perm.perm_coherence_mean_trajectory;
        ci_lower = perm.coherence_ci_lower;
        ci_upper = perm.coherence_ci_upper;
    else
        perm_trajectory = zeros(size(observed_trajectory));
        ci_lower = zeros(size(observed_trajectory));
        ci_upper = zeros(size(observed_trajectory));
    end
    
    observed_change = perm.observed_coherence_change;
    p_value = perm.coherence_p_value;
    effect_size = perm.coherence_effect_size;
    ylabel_text = 'Coherence';
end

% Plot random confidence interval (95% CI)
hold on;
fill([time_seconds, fliplr(time_seconds)], [ci_upper, fliplr(ci_lower)], ...
    [0.9, 0.9, 0.9], 'EdgeColor', 'none', 'DisplayName', 'Random 95% CI');
fill([time_seconds, fliplr(time_seconds)], [observed_trajectory+1.96*observed_std, fliplr(observed_trajectory-1.96*observed_std)], ...
    [0.9, 0, 0], 'EdgeColor', 'none', 'DisplayName', 'MI 95% CI');

% Plot trajectories
plot(time_seconds, perm_trajectory, 'k--', 'LineWidth', 2, 'DisplayName', 'Random mean');
plot(time_seconds, observed_trajectory, 'r-', 'LineWidth', 3, 'DisplayName', 'Non-reward');

% Highlight core non-reward region
ylims = ylim;
fill([2, 12, 12, 2], [ylims(1), ylims(1), ylims(2), ylims(2)], [1, 1, 0.8], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.2, 'DisplayName', 'Non-reward core');

xlabel('Time (seconds)');
ylabel(ylabel_text);
title(sprintf('%.0f Hz %s vs Random', frequency, measure_type));
legend('Location', 'best');
grid on;

% Add comprehensive statistics annotation
if p_value < 0.05
    if observed_change > 0
        direction = '↑';
        text_color = 'red';
        conclusion = 'ABOVE random';
    else
        direction = '↓';
        text_color = 'blue';
        conclusion = 'BELOW random';
    end
    sig_status = 'SIGNIFICANT';
else
    direction = '~';
    text_color = 'black';
    conclusion = 'similar to random';
    sig_status = 'ns';
end

% Create comprehensive annotation
annotation_text = {
    sprintf('%s %s', direction, sig_status);
    sprintf('p = %.4f', p_value);
    sprintf('Effect size = %.2f', effect_size);
    sprintf('Change = %.1e', observed_change);
    sprintf('%s', conclusion);
};

text(0.02, 0.98, annotation_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontWeight', 'bold', 'Color', text_color, 'BackgroundColor', 'white', ...
    'EdgeColor', text_color, 'FontSize', 9);

% Add effect size interpretation
if abs(effect_size) > 2
    effect_interpretation = 'LARGE effect';
elseif abs(effect_size) > 1
    effect_interpretation = 'MEDIUM effect';
elseif abs(effect_size) > 0.5
    effect_interpretation = 'SMALL effect';
else
    effect_interpretation = 'minimal effect';
end

text(0.98, 0.02, effect_interpretation, 'Units', 'normalized', ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right',...
    'FontSize', 8, 'BackgroundColor', 'white');
end

%% ========================================================================
%% ADDITIONAL ANALYSIS FUNCTIONS (KEPT FOR COMPATIBILITY)
%% ========================================================================

function visualize_individual_bands_with_breathing_quartiles(band_results, breathing_bands, valid_patches)
%% Visualize individual bands (4,5,6,7 Hz) with breathing rate quartiles
fig = figure('Name', 'Individual Band Analysis with Breathing Rate Quartiles', ...
    'Position', [50, 50, 2000, 1200]);

% Define colors for quartiles
quartile_colors = [
    0.2, 0.4, 0.8;  % Blue for Q1 (slowest breathing)
    0.0, 0.7, 0.4;  % Green for Q2
    0.9, 0.6, 0.0;  % Orange for Q3
    0.8, 0.2, 0.2   % Red for Q4 (fastest breathing)
];

% Create proper time axis
total_time_seconds = 14;
time_seconds = linspace(0, total_time_seconds, 100);

%% Calculate breathing rate quartiles across all patches
fprintf('Calculating breathing rate quartiles for individual band analysis...\n');
[quartile_data, quartile_thresholds] = calculate_breathing_quartiles_for_bands(band_results, valid_patches);

if isempty(quartile_data)
    fprintf('❌ Could not calculate breathing quartiles\n');
    return;
end

fprintf('Breathing rate quartiles:\n');
fprintf('  Q1 (25%%): ≤%.2f Hz\n', quartile_thresholds(1));
fprintf('  Q2 (50%%): %.2f-%.2f Hz\n', quartile_thresholds(1), quartile_thresholds(2));
fprintf('  Q3 (75%%): %.2f-%.2f Hz\n', quartile_thresholds(2), quartile_thresholds(3));
fprintf('  Q4 (100%%): >%.2f Hz\n', quartile_thresholds(3));

%% Plot individual bands: 2 rows x 4 columns (MI and Coherence for each of 4 bands)
band_names = fieldnames(band_results);
valid_bands = [];

% Find valid bands that match our target frequencies
for i = 1:length(band_names)
    result = band_results.(band_names{i});
    if ~isempty(result) && isfield(result, 'frequency')
        freq = result.frequency;
        if ismember(freq, breathing_bands)
            valid_bands = [valid_bands; struct('name', band_names{i}, 'freq', freq, 'result', result)];
        end
    end
end

% Sort by frequency
[~, sort_idx] = sort([valid_bands.freq]);
valid_bands = valid_bands(sort_idx);

n_valid_bands = length(valid_bands);
if n_valid_bands == 0
    fprintf('❌ No valid bands found\n');
    return;
end

for band_idx = 1:n_valid_bands
    band_info = valid_bands(band_idx);
    freq = band_info.freq;
    result = band_info.result;
    
    % MI plot (top row)
    subplot(2, 4, band_idx);
    plot_band_quartiles_with_permutation(quartile_data, result, time_seconds, ...
        freq, 'MI', quartile_colors, quartile_thresholds);
    
    % Coherence plot (bottom row)
    subplot(2, 4, band_idx + 4);
    plot_band_quartiles_with_permutation(quartile_data, result, time_seconds, ...
        freq, 'Coherence', quartile_colors, quartile_thresholds);
end

% Add main title
sgtitle('Individual Band Analysis: MI and Coherence by Breathing Rate Quartiles', ...
    'FontSize', 16, 'FontWeight', 'bold');

% Adjust layout
set(fig, 'Color', 'w');

% Print summary
print_individual_band_quartile_summary(quartile_data, quartile_thresholds, valid_bands);
end

function [quartile_data, quartile_thresholds] = calculate_breathing_quartiles_for_bands(band_results, valid_patches)
%% Calculate breathing rate quartiles specifically for band analysis using real breathing data
quartile_data = struct();
quartile_thresholds = [];

% Collect all breathing rates and patch data from all bands
all_breathing_rates = [];
all_patch_data = [];

band_names = fieldnames(band_results);

for i = 1:length(band_names)
    band_name = band_names{i};
    result = band_results.(band_name);
    
    if isempty(result) || ~isfield(result, 'coupling') || isempty(result.coupling)
        continue;
    end
    
    coupling = result.coupling;
    
    % Check if we have patch info and matrices
    if ~isfield(coupling, 'patch_info') || ~isfield(coupling, 'MI_matrix') || ...
       ~isfield(coupling, 'coherence_matrix')
        continue;
    end
    
    patch_info = coupling.patch_info;
    MI_matrix = coupling.MI_matrix;
    coherence_matrix = coupling.coherence_matrix;
    frequency = result.frequency;
    band_idx = result.band_index;
    
    for p = 1:length(patch_info)
        % Find the corresponding patch in valid_patches to get real breathing data
        session_id = patch_info(p).session_id;
        patch_id = patch_info(p).patch_id;
        
        % Find matching patch in valid_patches
        matching_patch_idx = [];
        for vp = 1:length(valid_patches)
            if valid_patches(vp).session_id == session_id && valid_patches(vp).patch_id == patch_id
                matching_patch_idx = vp;
                break;
            end
        end
        
        if ~isempty(matching_patch_idx)
            patch = valid_patches(matching_patch_idx);
            
            % Extract actual breathing data from the patch
            if isfield(patch, 'breathing_data') && ~isempty(patch.breathing_data)
                breathing_freq = patch.breathing_data;
                valid_breathing = patch.valid_breathing_mask;
                
                % Calculate mean breathing rate during non-reward core period
                core_start_rel = patch.core_start_relative + 1; % Convert to 1-based indexing
                core_end_rel = patch.core_end_relative + 1;
                
                % Ensure indices are within bounds
                core_start_rel = max(1, core_start_rel);
                core_end_rel = min(length(breathing_freq), core_end_rel);
                
                if core_end_rel > core_start_rel
                    core_breathing = breathing_freq(core_start_rel:core_end_rel);
                    core_valid = valid_breathing(core_start_rel:core_end_rel);
                    
                    if sum(core_valid) > 0
                        breathing_rate = mean(core_breathing(core_valid));
                    else
                        % Fallback to whole patch if core has no valid data
                        if sum(valid_breathing) > 0
                            breathing_rate = mean(breathing_freq(valid_breathing));
                        else
                            continue; % Skip this patch if no valid breathing data
                        end
                    end
                else
                    continue; % Skip if invalid core indices
                end
            else
                continue; % Skip if no breathing data
            end
            
            % Ensure breathing rate is reasonable (1-15 Hz)
            if breathing_rate < 1 || breathing_rate > 15 || isnan(breathing_rate)
                continue;
            end
            
            patch_data = struct();
            patch_data.breathing_rate = breathing_rate;
            patch_data.band_frequency = frequency;
            patch_data.band_name = band_name;
            patch_data.band_idx = band_idx;
            patch_data.patch_idx = p;
            patch_data.MI_trajectory = MI_matrix(p, :);
            patch_data.coherence_trajectory = coherence_matrix(p, :);
            patch_data.session_id = patch_info(p).session_id;
            patch_data.original_patch_id = patch_id;
            
            all_breathing_rates = [all_breathing_rates; breathing_rate];
            all_patch_data = [all_patch_data; patch_data];
        end
    end
end

if isempty(all_breathing_rates)
    fprintf('❌ No breathing rate data found\n');
    return;
end

% Calculate quartile thresholds
quartile_thresholds = prctile(all_breathing_rates, [25, 50, 75, 100]);

% Assign patches to quartiles
quartile_assignments = zeros(length(all_breathing_rates), 1);
for i = 1:length(all_breathing_rates)
    br = all_breathing_rates(i);
    if br <= quartile_thresholds(1)
        quartile_assignments(i) = 1;
    elseif br <= quartile_thresholds(2)
        quartile_assignments(i) = 2;
    elseif br <= quartile_thresholds(3)
        quartile_assignments(i) = 3;
    else
        quartile_assignments(i) = 4;
    end
end

% Organize data by quartiles and bands
for q = 1:4
    quartile_idx = quartile_assignments == q;
    quartile_patches = all_patch_data(quartile_idx);
    
    if isempty(quartile_patches)
        continue;
    end
    
    quartile_data(q).patches = quartile_patches;
    quartile_data(q).n_patches = length(quartile_patches);
    quartile_data(q).breathing_rates = [quartile_patches.breathing_rate];
    quartile_data(q).mean_breathing_rate = mean([quartile_patches.breathing_rate]);
    quartile_data(q).std_breathing_rate = std([quartile_patches.breathing_rate]);
    
    % Organize by band frequency for this quartile
    unique_freqs = unique([quartile_patches.band_frequency]);
    for freq = unique_freqs
        freq_idx = [quartile_patches.band_frequency] == freq;
        freq_patches = quartile_patches(freq_idx);
        
        if ~isempty(freq_patches)
            MI_trajectories = vertcat(freq_patches.MI_trajectory);
            coherence_trajectories = vertcat(freq_patches.coherence_trajectory);
            
            band_field = sprintf('band_%.0fHz', freq);
            quartile_data(q).(band_field).patches = freq_patches;
            quartile_data(q).(band_field).n_patches = length(freq_patches);
            quartile_data(q).(band_field).mean_MI_trajectory = mean(MI_trajectories, 1);
            quartile_data(q).(band_field).sem_MI_trajectory = std(MI_trajectories, 1) / sqrt(size(MI_trajectories, 1));
            quartile_data(q).(band_field).mean_coherence_trajectory = mean(coherence_trajectories, 1);
            quartile_data(q).(band_field).sem_coherence_trajectory = std(coherence_trajectories, 1) / sqrt(size(coherence_trajectories, 1));
            
            % Calculate changes
            initial_MI = mean(MI_trajectories(:, 1:10), 2);
            final_MI = mean(MI_trajectories(:, end-9:end), 2);
            quartile_data(q).(band_field).MI_change = mean(final_MI - initial_MI);
            
            initial_coherence = mean(coherence_trajectories(:, 1:10), 2);
            final_coherence = mean(coherence_trajectories(:, end-9:end), 2);
            quartile_data(q).(band_field).coherence_change = mean(final_coherence - initial_coherence);
        end
    end
end

fprintf('Quartile distribution across all bands:\n');
for q = 1:4
    if q <= length(quartile_data) && ~isempty(quartile_data(q).patches)
        fprintf('  Q%d: %d patches, %.2f±%.2f Hz\n', q, quartile_data(q).n_patches, ...
            quartile_data(q).mean_breathing_rate, quartile_data(q).std_breathing_rate);
    end
end
end

function plot_band_quartiles_with_permutation(quartile_data, band_result, time_seconds, frequency, measure_type, quartile_colors, quartile_thresholds)
%% Plot quartile trajectories for specific band with permutation results
hold on;

band_field = sprintf('band_%.0fHz', frequency);

% Plot permutation confidence interval first (background)
if ~isempty(band_result.permutation)
    perm = band_result.permutation;
    if strcmp(measure_type, 'MI')
        if isfield(perm, 'perm_MI_mean_trajectory') && ~isempty(perm.perm_MI_mean_trajectory)
            perm_mean = perm.perm_MI_mean_trajectory;
            perm_ci_lower = perm.MI_ci_lower;
            perm_ci_upper = perm.MI_ci_upper;
        else
            perm_mean = zeros(size(time_seconds));
            perm_ci_lower = zeros(size(time_seconds));
            perm_ci_upper = zeros(size(time_seconds));
        end
    else
        if isfield(perm, 'perm_coherence_mean_trajectory') && ~isempty(perm.perm_coherence_mean_trajectory)
            perm_mean = perm.perm_coherence_mean_trajectory;
            perm_ci_lower = perm.coherence_ci_lower;
            perm_ci_upper = perm.coherence_ci_upper;
        else
            perm_mean = zeros(size(time_seconds));
            perm_ci_lower = zeros(size(time_seconds));
            perm_ci_upper = zeros(size(time_seconds));
        end
    end
    
    % Plot random permutation confidence interval
    fill([time_seconds, fliplr(time_seconds)], [perm_ci_upper, fliplr(perm_ci_lower)], ...
        [0.9, 0.9, 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Random 95% CI');
    
    % Plot random mean
    plot(time_seconds, perm_mean, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Random mean');
end

% Plot quartile trajectories
quartile_labels = {
    sprintf('Q1: ≤%.1f Hz', quartile_thresholds(1)),
    sprintf('Q2: %.1f-%.1f Hz', quartile_thresholds(1), quartile_thresholds(2)),
    sprintf('Q3: %.1f-%.1f Hz', quartile_thresholds(2), quartile_thresholds(3)),
    sprintf('Q4: >%.1f Hz', quartile_thresholds(3))
};

valid_quartiles = 0;
for q = 1:min(4, length(quartile_data))
    if isempty(quartile_data(q).patches) || ~isfield(quartile_data(q), band_field)
        continue;
    end
    
    band_data = quartile_data(q).(band_field);
    if isempty(band_data)
        continue;
    end
    
    if strcmp(measure_type, 'MI')
        mean_trajectory = band_data.mean_MI_trajectory;
        sem_trajectory = band_data.sem_MI_trajectory;
        ylabel_text = 'Modulation Index';
    else
        mean_trajectory = band_data.mean_coherence_trajectory;
        sem_trajectory = band_data.sem_coherence_trajectory;
        ylabel_text = 'Coherence';
    end
    
    % Plot confidence interval for this quartile
    upper_ci = mean_trajectory + 1.96 * sem_trajectory;
    lower_ci = mean_trajectory - 1.96 * sem_trajectory;
    
    fill([time_seconds, fliplr(time_seconds)], [upper_ci, fliplr(lower_ci)], ...
        quartile_colors(q,:), 'EdgeColor', 'none', 'FaceAlpha', 0.2);
    
    % Plot mean trajectory
    plot(time_seconds, mean_trajectory, 'Color', quartile_colors(q,:), 'LineWidth', 2.5, ...
        'DisplayName', sprintf('%s (n=%d)', quartile_labels{q}, band_data.n_patches));
    
    valid_quartiles = valid_quartiles + 1;
end

% Highlight core non-reward region
ylims = ylim;
fill([2, 12, 12, 2], [ylims(1), ylims(1), ylims(2), ylims(2)], [1, 1, 0.8], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.15);

xlabel('Time (seconds)');
ylabel(ylabel_text);
title(sprintf('%.0f Hz %s by Breathing Rate', frequency, measure_type));

% Add legend only if we have valid quartiles
if valid_quartiles > 0
    legend('Location', 'best', 'FontSize', 8);
end

% Add statistical annotation
if ~isempty(band_result.permutation)
    perm = band_result.permutation;
    if strcmp(measure_type, 'MI')
        p_value = perm.MI_p_value;
        effect_size = perm.MI_effect_size;
        observed_change = perm.observed_MI_change;
    else
        p_value = perm.coherence_p_value;
        effect_size = perm.coherence_effect_size;
        observed_change = perm.observed_coherence_change;
    end
    
    % Determine significance and direction
    if p_value < 0.05
        if observed_change > 0
            sig_text = '↑ SIG';
            text_color = 'red';
        else
            sig_text = '↓ SIG';
            text_color = 'blue';
        end
    else
        sig_text = '~ ns';
        text_color = 'black';
    end
    
    % Add annotation
    annotation_text = {
        sig_text;
        sprintf('p=%.3f', p_value);
        sprintf('ES=%.2f', effect_size)
    };
    
    text(0.02, 0.98, annotation_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontWeight', 'bold', 'Color', text_color, 'BackgroundColor', 'white', ...
        'EdgeColor', text_color, 'FontSize', 8);
end

grid on;
end

function print_individual_band_quartile_summary(quartile_data, quartile_thresholds, valid_bands)
%% Print summary of individual band quartile analysis
fprintf('\n=== INDIVIDUAL BAND QUARTILE ANALYSIS SUMMARY ===\n');

fprintf('\nBreathing Rate Quartiles:\n');
% Q1-Q4 patches count
q_patches = zeros(4, 1);
for q = 1:4
    if length(quartile_data) >= q && ~isempty(quartile_data(q).patches)
        q_patches(q) = quartile_data(q).n_patches;
    end
end

fprintf('  Q1: ≤%.2f Hz (%d patches)\n', quartile_thresholds(1), q_patches(1));
fprintf('  Q2: %.2f-%.2f Hz (%d patches)\n', quartile_thresholds(1), quartile_thresholds(2), q_patches(2));
fprintf('  Q3: %.2f-%.2f Hz (%d patches)\n', quartile_thresholds(2), quartile_thresholds(3), q_patches(3));
fprintf('  Q4: >%.2f Hz (%d patches)\n', quartile_thresholds(3), q_patches(4));

fprintf('\nBy Band and Quartile:\n');
for band_idx = 1:length(valid_bands)
    freq = valid_bands(band_idx).freq;
    band_field = sprintf('band_%.0fHz', freq);
    fprintf('\n--- %.0f Hz Band ---\n', freq);
    for q = 1:4
        if q <= length(quartile_data) && ~isempty(quartile_data(q).patches) && ...
           isfield(quartile_data(q), band_field)
            
            band_data = quartile_data(q).(band_field);
            if ~isempty(band_data)
                fprintf('  Q%d: %d patches, MI change: %.6f, Coherence change: %.6f\n', ...
                    q, band_data.n_patches, band_data.MI_change, band_data.coherence_change);
            end
        end
    end
end

fprintf('\n=== INDIVIDUAL BAND ANALYSIS COMPLETE ===\n');
end

%% ========================================================================
%% COUPLING-DURATION RELATIONSHIP ANALYSIS
%% ========================================================================

function analyze_coupling_duration_relationship(valid_patches)
%% ANALYZE RELATIONSHIP BETWEEN COUPLING AND PATCH DURATION
% This function analyzes the relationship between 4Hz coherence medians
% and patch durations using boundary analysis for triangular distributions

fprintf('\n=== COUPLING-DURATION RELATIONSHIP ANALYSIS ===\n');

%% Extract specific coupling and duration data
% Use 4Hz coherence medians and duration samples as specified
log_duration = log([valid_patches.duration_samples]');
log_coupling = log(arrayfun(@(x) median(x.coupling_data.band_4_coherence(x.valid_breathing_mask)), valid_patches));

% Remove infinite values
valid_idx = isfinite(log_coupling) & isfinite(log_duration);
log_coupling = log_coupling(valid_idx);
log_duration = log_duration(valid_idx);

if length(log_coupling) < 20
    fprintf('❌ Insufficient valid data for analysis (n=%d)\n', length(log_coupling));
    return;
end

fprintf('Analyzing %d patches with valid 4Hz coherence and duration data\n', length(log_coupling));

%% Create figure with boundary analysis
figure('Name', '4Hz Coherence vs Duration Analysis', 'Position', [50, 50, 1200, 800]);

% 1. Quantile regression - analyze upper boundary
tau = [0.5, 0.75, 0.9, 0.95]; % quantiles
quantile_fits = zeros(length(tau), 2); % store slope and intercept

for i = 1:length(tau)
    % Get points at this quantile level
    threshold = quantile(log_duration, tau(i));
    idx = log_duration >= threshold;
    
    if sum(idx) > 5
        [p, ~] = polyfit(log_coupling(idx), log_duration(idx), 1);
        quantile_fits(i, :) = p;
    else
        quantile_fits(i, :) = [NaN, NaN];
    end
end

% 2. Variance analysis
% Bin data and analyze variance in each bin
[~, edges] = histcounts(log_coupling, 10);
bin_centers = (edges(1:end-1) + edges(2:end)) / 2;
bin_vars = arrayfun(@(i) var(log_duration(log_coupling >= edges(i) & log_coupling < edges(i+1))), 1:length(edges)-1);

% 3. Envelope analysis - fit upper boundary
upper_quantile = quantile(log_duration, 0.95);
upper_idx = log_duration >= quantile(log_duration, 0.9);
[p_upper, ~] = polyfit(log_coupling(upper_idx), log_duration(upper_idx), 1);

% 4. Spearman correlation (robust to this distribution)
[rho, p_spearman] = corr(log_coupling, log_duration, 'Type', 'Spearman');

%% Create 4-panel figure

% Plot 1: Main scatter with quantile lines
subplot(2, 2, 1);
scatter(log_coupling, log_duration, 20, [0.5, 0.5, 0.5], 'filled');
hold on;

% Plot quantile regression lines
x_range = linspace(min(log_coupling), max(log_coupling), 100);
colors = {'b', 'g', 'r', 'm'};

for i = 1:length(tau)
    if ~isnan(quantile_fits(i, 1))
        y_line = polyval(quantile_fits(i, :), x_range);
        plot(x_range, y_line, colors{i}, 'LineWidth', 2, ...
            'DisplayName', sprintf('τ=%.2f', tau(i)));
    end
end

xlabel('Log(4Hz Coherence Medians)');
ylabel('Log(Duration Samples)');
title('Quantile Regression Lines');
legend('Location', 'best');
grid on;

% Plot 2: Variance analysis
subplot(2, 2, 2);
plot(bin_centers, bin_vars, 'ro-', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Log(4Hz Coherence) - Bin Centers');
ylabel('Variance in Log(Duration)');
title('Variance vs Coupling Strength');
grid on;

% Fit trend to variance
if sum(~isnan(bin_vars)) > 3
    valid_idx = ~isnan(bin_vars);
    p_var = polyfit(bin_centers(valid_idx), bin_vars(valid_idx), 1);
    hold on;
    plot(bin_centers, polyval(p_var, bin_centers), 'k--', 'LineWidth', 1.5);
    [r_var, p_var_corr] = corr(bin_centers(valid_idx)', bin_vars(valid_idx)');
    text(0.05, 0.95, sprintf('r=%.3f, p=%.3f', r_var, p_var_corr), ...
        'Units', 'normalized', 'BackgroundColor', 'white');
end

% Plot 3: Upper envelope analysis
subplot(2, 2, 3);
scatter(log_coupling, log_duration, 15, [0.7, 0.7, 0.7], 'filled');
hold on;

% Highlight upper boundary points
scatter(log_coupling(upper_idx), log_duration(upper_idx), 40, 'r', 'filled');

% Plot upper boundary line
y_upper = polyval(p_upper, x_range);
plot(x_range, y_upper, 'k-', 'LineWidth', 3, 'DisplayName', 'Upper boundary');

xlabel('Log(4Hz Coherence Medians)');
ylabel('Log(Duration Samples)');
title('Upper Boundary Envelope');
legend('Location', 'best');
grid on;

% Add boundary equation
text(0.05, 0.95, sprintf('y = %.3fx + %.3f', p_upper(1), p_upper(2)), ...
    'Units', 'normalized', 'BackgroundColor', 'white', 'FontWeight', 'bold');

% Plot 4: Summary statistics
subplot(2, 2, 4);
axis off;

% Create text summary
summary_text = {
    '4Hz COHERENCE-DURATION ANALYSIS';
    '';
    sprintf('Spearman correlation: ρ = %.4f', rho);
    sprintf('p-value: %.4f', p_spearman);
    sprintf('Sample size: %d patches', length(log_coupling));
    '';
    'Quantile Slopes:';
    sprintf('  50%%: %.4f', quantile_fits(1, 1));
    sprintf('  75%%: %.4f', quantile_fits(2, 1));
    sprintf('  90%%: %.4f', quantile_fits(3, 1));
    sprintf('  95%%: %.4f', quantile_fits(4, 1));
    '';
    sprintf('Upper boundary slope: %.4f', p_upper(1));
    '';
    'Interpretation:';
    '';
};

% Add interpretation based on results
if p_upper(1) < -0.5
    summary_text{end+1} = '• STRONG negative constraint';
    summary_text{end+1} = '• High coherence → limited duration';
elseif p_upper(1) < -0.1
    summary_text{end+1} = '• Moderate negative constraint';
else
    summary_text{end+1} = '• Weak or no constraint';
end

if abs(rho) > 0.5 && p_spearman < 0.05
    summary_text{end+1} = '• Significant overall correlation';
else
    summary_text{end+1} = '• Weak overall correlation';
end

% Display text
text(0.05, 0.95, summary_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontSize', 10, 'FontName', 'FixedWidth');

sgtitle('4Hz Breathing-Gamma Coherence vs Patch Duration', 'FontSize', 14, 'FontWeight', 'bold');

%% Print results to command window
fprintf('\n=== 4Hz COHERENCE-DURATION ANALYSIS RESULTS ===\n');
fprintf('Sample size: %d patches\n', length(log_coupling));
fprintf('Spearman correlation: ρ = %.4f (p = %.4f)\n', rho, p_spearman);

% Pearson correlation for comparison
[r_pearson, p_pearson] = corr(log_coupling, log_duration, 'Type', 'Pearson');
fprintf('Pearson correlation: r = %.4f (p = %.4f)\n', r_pearson, p_pearson);

fprintf('\nQuantile regression slopes:\n');
for i = 1:length(tau)
    if ~isnan(quantile_fits(i, 1))
        fprintf('  τ=%.2f: slope = %.4f\n', tau(i), quantile_fits(i, 1));
    end
end

fprintf('\nUpper boundary analysis:\n');
fprintf('  90-95%% envelope slope: %.4f\n', p_upper(1));
fprintf('  Points in upper envelope: %d (%.1f%%)\n', sum(upper_idx), sum(upper_idx)/length(upper_idx)*100);

% Variance trend analysis
if exist('r_var', 'var')
    fprintf('\nVariance analysis:\n');
    fprintf('  Variance vs coupling correlation: r = %.4f (p = %.4f)\n', r_var, p_var_corr);
    if r_var < -0.5 && p_var_corr < 0.05
        fprintf('  → HETEROSCEDASTIC: Variance decreases with coupling\n');
    else
        fprintf('  → Variance pattern not significant\n');
    end
end

% Overall interpretation
fprintf('\n=== INTERPRETATION ===\n');
if p_upper(1) < -0.3 && abs(rho) > 0.3
    fprintf('→ Strong evidence for triangular constraint pattern\n');
    fprintf('→ High 4Hz coherence limits maximum patch duration\n');
    fprintf('→ Suggests physiological constraint mechanism\n');
elseif p_upper(1) < -0.1
    fprintf('→ Moderate evidence for constraint pattern\n');
    fprintf('→ Some limitation of duration at high coherence\n');
else
    fprintf('→ Limited evidence for constraint pattern\n');
    fprintf('→ Relationship appears more linear\n');
end

if abs(rho) > 0.5 && p_spearman < 0.05
    fprintf('→ Strong overall correlation detected\n');
elseif abs(rho) > 0.3 && p_spearman < 0.05
    fprintf('→ Moderate overall correlation detected\n');
else
    fprintf('→ Weak or no significant correlation\n');
end

fprintf('\n=== ANALYSIS COMPLETE ===\n');
end

function plot_coupling_heatmap(band_results, sort_by, measure_type, bands_to_plot, valid_patches)
%% PLOT_COUPLING_HEATMAP - Visualize coupling strength of stretched patches
% 
% This function creates imagesc heatmaps of coupling strength for stretched
% patches, with options to sort by median coupling strength, patch length,
% or median breathing rate during patches.
%
% INPUTS:
%   band_results   - Structure containing multi-band coupling results
%   sort_by        - String: 'coupling', 'length', or 'breathing' (default: 'coupling')
%   measure_type   - String: 'MI', 'coherence', or 'both' (default: 'both')
%   bands_to_plot  - Array of frequencies to plot (default: [4, 5, 6, 7])
%   valid_patches  - Array of patch structures (required for 'breathing' sort)

%% Input validation and defaults
if nargin < 2 || isempty(sort_by)
    sort_by = 'coupling';
end

if nargin < 3 || isempty(measure_type)
    measure_type = 'both';
end

if nargin < 4 || isempty(bands_to_plot)
    bands_to_plot = [4, 5, 6, 7];
end

if nargin < 5
    valid_patches = [];
end

% Validate inputs
valid_sort_options = {'coupling', 'length', 'breathing'};
valid_measure_options = {'MI', 'coherence', 'both'};

if ~ismember(sort_by, valid_sort_options)
    error('sort_by must be ''coupling'', ''length'', or ''breathing''');
end

if ~ismember(measure_type, valid_measure_options)
    error('measure_type must be ''MI'', ''coherence'', or ''both''');
end

% Check if breathing sort is requested but valid_patches not provided
if strcmp(sort_by, 'breathing') && isempty(valid_patches)
    error('valid_patches must be provided when sort_by is ''breathing''');
end

fprintf('\n=== COUPLING HEATMAP VISUALIZATION ===\n');
fprintf('Sort by: %s\n', sort_by);
fprintf('Measure: %s\n', measure_type);
fprintf('Bands: %s Hz\n', mat2str(bands_to_plot));

%% Extract and organize data from all bands
band_names = fieldnames(band_results);
available_bands = [];
all_band_data = struct();

% Collect data from all available bands
for i = 1:length(band_names)
    result = band_results.(band_names{i});
    
    if isempty(result) || ~isfield(result, 'frequency') || ...
       ~isfield(result, 'coupling') || isempty(result.coupling)
        continue;
    end
    
    freq = result.frequency;
    
    % Check if this frequency is in our bands_to_plot
    if ismember(freq, bands_to_plot)
        available_bands = [available_bands, freq];
        
        coupling = result.coupling;
        
        % Ensure we have the required data
        if ~isfield(coupling, 'MI_matrix') || ~isfield(coupling, 'coherence_matrix') || ...
           ~isfield(coupling, 'patch_info')
            fprintf('⚠️  Band %.0f Hz: Missing required matrices, skipping\n', freq);
            continue;
        end
        
        band_data = struct();
        band_data.frequency = freq;
        band_data.MI_matrix = coupling.MI_matrix;
        band_data.coherence_matrix = coupling.coherence_matrix;
        band_data.patch_info = coupling.patch_info;
        band_data.n_patches = size(coupling.MI_matrix, 1);
        
        % Calculate sorting metrics for each patch
        band_data.median_MI = median(coupling.MI_matrix, 2);
        band_data.median_coherence = median(coupling.coherence_matrix, 2);
        
        if isfield(coupling.patch_info, 'duration') && ~isempty([coupling.patch_info.duration])
            band_data.patch_lengths = [coupling.patch_info.duration]';
        elseif isfield(coupling.patch_info, 'effective_duration') && ~isempty([coupling.patch_info.effective_duration])
            band_data.patch_lengths = [coupling.patch_info.effective_duration]';
        else
            % Use matrix width as proxy for length
            band_data.patch_lengths = ones(size(coupling.MI_matrix, 1), 1) * size(coupling.MI_matrix, 2);
        end
        
        % Calculate breathing rates if needed and valid_patches provided
        if strcmp(sort_by, 'breathing') && ~isempty(valid_patches)
            band_data.breathing_rates = calculate_patch_breathing_rates(coupling.patch_info, valid_patches);
        else
            band_data.breathing_rates = [];
        end
        
        all_band_data.(sprintf('band_%.0fHz', freq)) = band_data;
    end
end

% Sort available bands for consistent display
available_bands = sort(available_bands);
n_bands = length(available_bands);

if n_bands == 0
    fprintf('❌ No valid bands found for plotting\n');
    return;
end

fprintf('Available bands: %s Hz\n', mat2str(available_bands));

%% Determine subplot layout based on measure_type
if strcmp(measure_type, 'both')
    n_rows = 2;  % MI and Coherence
    measure_types = {'MI', 'coherence'};
    measure_titles = {'Modulation Index (MI)', 'Coherence'};
else
    n_rows = 1;
    if strcmp(measure_type, 'MI')
        measure_types = {'MI'};
        measure_titles = {'Modulation Index (MI)'};
    else
        measure_types = {'coherence'};
        measure_titles = {'Coherence'};
    end
end

n_cols = n_bands;

%% Create figure
fig_width = max(1200, 300 * n_cols);
fig_height = max(800, 400 * n_rows);
fig = figure('Name', sprintf('Coupling Heatmap - Sorted by %s', sort_by), ...
    'Position', [50, 50, fig_width, fig_height]);

%% Create time axis for x-labels
target_length = size(all_band_data.(sprintf('band_%.0fHz', available_bands(1))).MI_matrix, 2);
total_time_seconds = 14; % Typical patch duration
time_seconds = linspace(0, total_time_seconds, target_length);

% Create time tick positions and labels for cleaner display
n_time_ticks = 8;
time_tick_indices = round(linspace(1, target_length, n_time_ticks));
time_tick_labels = arrayfun(@(x) sprintf('%.1f', x), time_seconds(time_tick_indices), 'UniformOutput', false);

%% Plot heatmaps for each measure type and band
for measure_idx = 1:length(measure_types)
    current_measure = measure_types{measure_idx};
    measure_title = measure_titles{measure_idx};
    
    for band_idx = 1:n_bands
        freq = available_bands(band_idx);
        band_field = sprintf('band_%.0fHz', freq);
        band_data = all_band_data.(band_field);
        
        % Determine subplot position
        subplot_idx = (measure_idx - 1) * n_cols + band_idx;
        subplot(n_rows, n_cols, subplot_idx);
        
        % Get the appropriate data matrix
        if strcmp(current_measure, 'MI')
            data_matrix = band_data.MI_matrix;
            median_values = band_data.median_MI;
            colormap_name = 'hot';
        else
            data_matrix = band_data.coherence_matrix;
            median_values = band_data.median_coherence;
            colormap_name = 'parula';
        end
        
        % Determine sorting order
        if strcmp(sort_by, 'coupling')
            [~, sort_idx] = sort(median_values, 'descend');
            sort_metric = median_values;
            sort_label = sprintf('Median %s', current_measure);
        elseif strcmp(sort_by, 'length')
            [~, sort_idx] = sort(band_data.patch_lengths, 'descend');
            sort_metric = band_data.patch_lengths;
            sort_label = 'Patch Length (s)';
        else % sort by breathing rate
            if ~isempty(band_data.breathing_rates)
                [~, sort_idx] = sort(band_data.breathing_rates, 'descend');
                sort_metric = band_data.breathing_rates;
                sort_label = 'Breathing Rate (Hz)';
            else
                fprintf('⚠️  No breathing rate data available for %.0f Hz, using coupling sort\n', freq);
                [~, sort_idx] = sort(median_values, 'descend');
                sort_metric = median_values;
                sort_label = sprintf('Median %s (fallback)', current_measure);
            end
        end
        
        % Sort the data matrix
        sorted_data = data_matrix(sort_idx, :);
        sorted_metric = sort_metric(sort_idx);
        
        % Create the heatmap
        imagesc(sorted_data);
        
        % Set colormap
        colormap(gca, colormap_name);
        
        % Add colorbar with proper labeling
        cb = colorbar;
        cb.Label.String = measure_title;
        cb.Label.FontSize = 10;
        
        % Set axis properties
        xlabel('Time');
        ylabel('Patches (sorted)');
        title(sprintf('%.0f Hz %s\n(sorted by %s)', freq, measure_title, sort_label), ...
            'FontSize', 11, 'FontWeight', 'bold');
        
        % Set x-axis ticks and labels
        set(gca, 'XTick', time_tick_indices, 'XTickLabel', time_tick_labels);
        
        % Set y-axis to show patch numbers
        n_patches = size(sorted_data, 1);
        y_tick_indices = round(linspace(1, n_patches, min(10, n_patches)));
        set(gca, 'YTick', y_tick_indices, 'YTickLabel', y_tick_indices);
        
        % Add grid for better readability
        grid on;
        set(gca, 'GridAlpha', 0.3, 'GridColor', 'white');
        
        % Add statistics annotation
        stats_text = {
            sprintf('N: %d patches', n_patches);
            sprintf('Range: %.3f-%.3f', min(sorted_metric), max(sorted_metric));
            sprintf('Mean: %.3f', mean(sorted_metric));
            sprintf('Std: %.3f', std(sorted_metric))
        };
        
        % Position annotation
        text(0.02, 0.98, stats_text, 'Units', 'normalized', ...
            'VerticalAlignment', 'top', 'FontSize', 8, ...
            'BackgroundColor', 'white', 'EdgeColor', 'black', ...
            'Color', 'black');
        
        % Highlight core non-reward region (if applicable)
        if target_length == 100  % Standard stretched length
            core_start_idx = round(target_length * 2/14);  % 2 seconds
            core_end_idx = round(target_length * 12/14);   % 12 seconds
            
            hold on;
            % Add vertical lines to mark core region
            line([core_start_idx, core_start_idx], [0.5, n_patches + 0.5], ...
                'Color', 'cyan', 'LineWidth', 2, 'LineStyle', '--');
            line([core_end_idx, core_end_idx], [0.5, n_patches + 0.5], ...
                'Color', 'cyan', 'LineWidth', 2, 'LineStyle', '--');
            
            % Add core region label
            text(mean([core_start_idx, core_end_idx]), n_patches * 0.05, 'Core', ...
                'HorizontalAlignment', 'center', 'Color', 'cyan', ...
                'FontWeight', 'bold', 'FontSize', 9);
        end
        
        % Make the plot visually appealing
        box on;
        set(gca, 'LineWidth', 1);
    end
end

%% Add main title and adjust layout
main_title = sprintf('Coupling Strength Heatmaps - Sorted by %s', upper(sort_by));
if strcmp(measure_type, 'both')
    main_title = [main_title, ' (MI and Coherence)'];
else
    main_title = [main_title, sprintf(' (%s)', measure_type)];
end

sgtitle(main_title, 'FontSize', 14, 'FontWeight', 'bold');

fprintf('\n=== HEATMAP COMPLETE ===\n');
end

function breathing_rates = calculate_patch_breathing_rates(patch_info, valid_patches)
%% Calculate median breathing rates for patches based on valid_patches data
% This function matches patch_info with valid_patches to extract breathing rates

breathing_rates = [];

if isempty(patch_info) || isempty(valid_patches)
    return;
end

n_patches = length(patch_info);
breathing_rates = NaN(n_patches, 1);

for p = 1:n_patches
    % Get session and patch IDs from patch_info
    if isfield(patch_info(p), 'session_id') && isfield(patch_info(p), 'patch_id')
        session_id = patch_info(p).session_id;
        patch_id = patch_info(p).patch_id;
        
        % Find matching patch in valid_patches
        matching_patch_idx = [];
        for vp = 1:length(valid_patches)
            if isfield(valid_patches(vp), 'session_id') && isfield(valid_patches(vp), 'patch_id')
                if valid_patches(vp).session_id == session_id && valid_patches(vp).patch_id == patch_id
                    matching_patch_idx = vp;
                    break;
                end
            end
        end
        
        if ~isempty(matching_patch_idx)
            patch = valid_patches(matching_patch_idx);
            
            % Extract breathing data from the patch
            if isfield(patch, 'breathing_data') && isfield(patch, 'valid_breathing_mask') && ...
               ~isempty(patch.breathing_data) && ~isempty(patch.valid_breathing_mask)
                
                breathing_freq = patch.breathing_data;
                valid_breathing = patch.valid_breathing_mask;
                
                % Calculate median breathing rate during non-reward core period
                if isfield(patch, 'core_start_relative') && isfield(patch, 'core_end_relative')
                    core_start_rel = patch.core_start_relative + 1; % Convert to 1-based indexing
                    core_end_rel = patch.core_end_relative + 1;
                    
                    % Ensure indices are within bounds
                    core_start_rel = max(1, core_start_rel);
                    core_end_rel = min(length(breathing_freq), core_end_rel);
                    
                    if core_end_rel > core_start_rel
                        core_breathing = breathing_freq(core_start_rel:core_end_rel);
                        core_valid = valid_breathing(core_start_rel:core_end_rel);
                        
                        if sum(core_valid) > 0
                            valid_core_breathing = core_breathing(core_valid);
                            % Filter out unreasonable breathing rates
                            reasonable_breathing = valid_core_breathing > 1 & valid_core_breathing < 15;
                            if sum(reasonable_breathing) > 0
                                breathing_rates(p) = median(valid_core_breathing(reasonable_breathing));
                            end
                        end
                    end
                end
                
                % Fallback: use whole patch if core calculation failed
                if isnan(breathing_rates(p)) && sum(valid_breathing) > 0
                    valid_patch_breathing = breathing_freq(valid_breathing);
                    reasonable_breathing = valid_patch_breathing > 1 & valid_patch_breathing < 15;
                    if sum(reasonable_breathing) > 0
                        breathing_rates(p) = median(valid_patch_breathing(reasonable_breathing));
                    end
                end
            end
        end
    end
end

% Remove NaN values and return only valid breathing rates
valid_idx = ~isnan(breathing_rates);
if sum(valid_idx) == 0
    fprintf('⚠️  No valid breathing rate data found\n');
    breathing_rates = [];
else
    fprintf('✓ Found breathing rate data for %d/%d patches\n', sum(valid_idx), n_patches);
    % For patches without breathing data, use mean of available data
    if sum(~valid_idx) > 0
        mean_breathing = mean(breathing_rates(valid_idx));
        breathing_rates(~valid_idx) = mean_breathing;
        fprintf('  Filled %d missing values with mean (%.2f Hz)\n', sum(~valid_idx), mean_breathing);
    end
end
end