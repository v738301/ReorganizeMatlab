%% ========================================================================
%  UNIT FEATURE VISUALIZATION
%  Visualizes comprehensive unit features extracted from all analyses
%% ========================================================================
%
%  This script visualizes features extracted by Unit_Feature_Extraction.m:
%  1. Feature distributions (coherence, phase coupling, PSTH responses)
%  2. Feature correlations
%  3. Session type comparisons (Aversive vs Reward)
%  4. Response type summaries
%
%  Input: unit_features_comprehensive.mat
%  Output: Multiple figures showing feature space
%
%% ========================================================================

clear all;
close all;

%% ========================================================================
%  SECTION 1: LOAD DATA
%% ========================================================================

fprintf('=== UNIT FEATURE VISUALIZATION ===\n\n');
fprintf('Loading extracted features...\n');

try
    load('unit_features_comprehensive.mat', 'unit_features_comprehensive');
    fprintf('✓ Loaded feature data\n\n');
catch ME
    fprintf('❌ Failed to load feature data: %s\n', ME.message);
    fprintf('Please run Unit_Feature_Extraction.m first\n');
    return;
end

% Extract feature struct arrays
coherence_features = unit_features_comprehensive.coherence_features;
phase_narrow_features = unit_features_comprehensive.phase_narrow_features;
phase_broad_features = unit_features_comprehensive.phase_broad_features;
psth_features = unit_features_comprehensive.psth_features;
config = unit_features_comprehensive.config;

% Count units (struct arrays)
n_units_coherence = length(coherence_features);
n_units_phase_narrow = length(phase_narrow_features);
n_units_phase_broad = length(phase_broad_features);
n_units_psth = length(psth_features);

fprintf('Feature counts:\n');
fprintf('  Coherence features: %d units\n', n_units_coherence);
fprintf('  Phase coupling (narrow): %d units\n', n_units_phase_narrow);
fprintf('  Phase coupling (broad): %d units\n', n_units_phase_broad);
fprintf('  PSTH features: %d units\n\n', n_units_psth);

%% ========================================================================
%  SECTION 2: COHERENCE FEATURE VISUALIZATION
%% ========================================================================

fprintf('Visualizing coherence features...\n');

% Get session types from struct array
session_types = {coherence_features.session_type};  % Extract as cell array
is_aversive = contains(session_types, 'Aversive');
is_reward = contains(session_types, 'Reward');

% Create figure for coherence features
fig1 = figure('Position', [100 100 1600 1000], 'Name', 'Coherence Features');

% Define narrow band features to plot
narrow_band_names = {'1-3 Hz', '5-7 Hz', '8-10 Hz'};
narrow_band_vars = {'coherence_1_3Hz', 'coherence_5_7Hz', 'coherence_8_10Hz'};

% Plot narrow band coherence distributions
for i = 1:3
    subplot(3, 4, i);
    hold on;

    % Extract values from struct array
    all_values = [coherence_features.(narrow_band_vars{i})];
    data_aversive = all_values(is_aversive);
    data_reward = all_values(is_reward);

    % Remove NaN values
    data_aversive = data_aversive(~isnan(data_aversive));
    data_reward = data_reward(~isnan(data_reward));

    histogram(data_aversive, 20, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    histogram(data_reward, 20, 'FaceColor', [0.2 0.2 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');

    xlabel('Coherence');
    ylabel('Count');
    title(sprintf('Coherence %s', narrow_band_names{i}));
    legend({'Aversive', 'Reward'}, 'Location', 'best');
    grid on;

    % Statistical test
    if ~isempty(data_aversive) && ~isempty(data_reward)
        [~, p] = ttest2(data_aversive, data_reward);
        text(0.05, 0.95, sprintf('p = %.4f', p), 'Units', 'normalized', ...
             'VerticalAlignment', 'top', 'FontSize', 8);
    end
end

% Plot broad band coherence distributions
broad_band_names = {'Delta', 'Theta', 'Beta', 'Low Gamma', 'High Gamma', 'Ultra Gamma'};
broad_band_vars = {'coherence_delta', 'coherence_theta', 'coherence_beta', ...
                   'coherence_low_gamma', 'coherence_high_gamma', 'coherence_ultra_gamma'};

for i = 1:6
    subplot(3, 4, i + 4);
    hold on;

    % Extract values from struct array
    all_values = [coherence_features.(broad_band_vars{i})];
    data_aversive = all_values(is_aversive);
    data_reward = all_values(is_reward);

    data_aversive = data_aversive(~isnan(data_aversive));
    data_reward = data_reward(~isnan(data_reward));

    histogram(data_aversive, 20, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    histogram(data_reward, 20, 'FaceColor', [0.2 0.2 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');

    xlabel('Coherence');
    ylabel('Count');
    title(broad_band_names{i});
    legend({'Aversive', 'Reward'}, 'Location', 'best');
    grid on;

    if ~isempty(data_aversive) && ~isempty(data_reward)
        [~, p] = ttest2(data_aversive, data_reward);
        text(0.05, 0.95, sprintf('p = %.4f', p), 'Units', 'normalized', ...
             'VerticalAlignment', 'top', 'FontSize', 8);
    end
end

% Plot peak coherence frequency distribution
subplot(3, 4, 11);
hold on;

% Extract values from struct array
all_peak_freq = [coherence_features.coherence_peak_freq];
peak_freq_aversive = all_peak_freq(is_aversive);
peak_freq_reward = all_peak_freq(is_reward);

peak_freq_aversive = peak_freq_aversive(~isnan(peak_freq_aversive));
peak_freq_reward = peak_freq_reward(~isnan(peak_freq_reward));

histogram(peak_freq_aversive, 20, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
histogram(peak_freq_reward, 20, 'FaceColor', [0.2 0.2 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');

xlabel('Frequency (Hz)');
ylabel('Count');
title('Peak Coherence Frequency');
legend({'Aversive', 'Reward'}, 'Location', 'best');
grid on;

% Plot peak coherence magnitude
subplot(3, 4, 12);
hold on;

% Extract values from struct array
all_peak_mag = [coherence_features.coherence_peak_mag];
peak_mag_aversive = all_peak_mag(is_aversive);
peak_mag_reward = all_peak_mag(is_reward);

peak_mag_aversive = peak_mag_aversive(~isnan(peak_mag_aversive));
peak_mag_reward = peak_mag_reward(~isnan(peak_mag_reward));

histogram(peak_mag_aversive, 20, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
histogram(peak_mag_reward, 20, 'FaceColor', [0.2 0.2 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');

xlabel('Peak Coherence');
ylabel('Count');
title('Peak Coherence Magnitude');
legend({'Aversive', 'Reward'}, 'Location', 'best');
grid on;

sgtitle('Spike-LFP Coherence Features', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 3: COHERENCE FEATURE CORRELATIONS
%% ========================================================================

fprintf('Computing coherence feature correlations...\n');

fig2 = figure('Position', [150 150 1200 900], 'Name', 'Coherence Correlations');

% Extract numeric coherence features from struct array
coherence_feature_names = {'1-3Hz', '5-7Hz', '8-10Hz', 'Delta', 'Theta', ...
                           'Beta', 'LowGamma', 'HighGamma', 'UltraGamma', ...
                           'PeakFreq', 'PeakMag'};
coherence_matrix = [[coherence_features.coherence_1_3Hz]', ...
                    [coherence_features.coherence_5_7Hz]', ...
                    [coherence_features.coherence_8_10Hz]', ...
                    [coherence_features.coherence_delta]', ...
                    [coherence_features.coherence_theta]', ...
                    [coherence_features.coherence_beta]', ...
                    [coherence_features.coherence_low_gamma]', ...
                    [coherence_features.coherence_high_gamma]', ...
                    [coherence_features.coherence_ultra_gamma]', ...
                    [coherence_features.coherence_peak_freq]', ...
                    [coherence_features.coherence_peak_mag]'];

% Compute correlation
corr_matrix = corr(coherence_matrix, 'rows', 'pairwise');

% Plot correlation matrix
imagesc(corr_matrix);
colorbar;
colormap(bluewhitered(256));
caxis([-1 1]);

set(gca, 'XTick', 1:length(coherence_feature_names), 'XTickLabel', coherence_feature_names, ...
         'XTickLabelRotation', 45);
set(gca, 'YTick', 1:length(coherence_feature_names), 'YTickLabel', coherence_feature_names);

title('Coherence Feature Correlations', 'FontSize', 14, 'FontWeight', 'bold');
axis square;

% Add correlation values
for i = 1:length(coherence_feature_names)
    for j = 1:length(coherence_feature_names)
        if abs(corr_matrix(i, j)) > 0.5
            text(j, i, sprintf('%.2f', corr_matrix(i, j)), ...
                 'HorizontalAlignment', 'center', 'FontSize', 8, ...
                 'Color', 'k', 'FontWeight', 'bold');
        end
    end
end

%% ========================================================================
%  SECTION 3B: PHASE COUPLING FEATURE CORRELATIONS
%% ========================================================================

fprintf('Computing phase coupling feature correlations...\n');

fig2b = figure('Position', [200 100 1800 800], 'Name', 'Phase Coupling Correlations');

% ----- NARROW BAND PHASE CORRELATIONS -----
subplot(1, 2, 1);

% Build matrix of narrow band MRL features
narrow_bands = config.narrow_bands;
behaviors = config.behavior_names;

% Create feature names for narrow bands
narrow_phase_feature_names = {};
narrow_phase_matrix = [];

for band_idx = 1:length(narrow_bands)
    for beh_idx = 1:length(behaviors)
        % Extract MRL values
        mrl_values = [];
        for unit_idx = 1:n_units_phase_narrow
            mrl_values(end+1) = phase_narrow_features(unit_idx).phase_MRL_narrow(band_idx, beh_idx);
        end

        narrow_phase_matrix = [narrow_phase_matrix, mrl_values'];

        % Create short name
        band_short = strrep(narrow_bands{band_idx}, 'Hz', '');
        beh_short = behaviors{beh_idx};
        if length(beh_short) > 8
            beh_short = beh_short(1:8);
        end
        narrow_phase_feature_names{end+1} = sprintf('%s_%s', band_short, beh_short);
    end
end

% Compute correlation
narrow_corr_matrix = corr(narrow_phase_matrix, 'rows', 'pairwise');

% Plot
imagesc(narrow_corr_matrix);
colorbar;
colormap(bluewhitered(256));
caxis([-1 1]);

set(gca, 'XTick', 1:length(narrow_phase_feature_names), ...
         'XTickLabel', narrow_phase_feature_names, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(narrow_phase_feature_names), ...
         'YTickLabel', narrow_phase_feature_names);
set(gca, 'FontSize', 7);

title('Phase Coupling Correlations - Narrow Bands (MRL)', 'FontSize', 12, 'FontWeight', 'bold');
axis square;

% Add strong correlation values
for i = 1:length(narrow_phase_feature_names)
    for j = 1:length(narrow_phase_feature_names)
        if i ~= j && abs(narrow_corr_matrix(i, j)) > 0.6
            text(j, i, sprintf('%.2f', narrow_corr_matrix(i, j)), ...
                 'HorizontalAlignment', 'center', 'FontSize', 6, ...
                 'Color', 'k', 'FontWeight', 'bold');
        end
    end
end

% ----- BROAD BAND PHASE CORRELATIONS -----
subplot(1, 2, 2);

% Build matrix of broad band MRL features
broad_bands = config.broad_bands;

% Create feature names for broad bands
broad_phase_feature_names = {};
broad_phase_matrix = [];

for band_idx = 1:length(broad_bands)
    for beh_idx = 1:length(behaviors)
        % Extract MRL values
        mrl_values = [];
        for unit_idx = 1:n_units_phase_broad
            mrl_values(end+1) = phase_broad_features(unit_idx).phase_MRL_broad(band_idx, beh_idx);
        end

        broad_phase_matrix = [broad_phase_matrix, mrl_values'];

        % Create short name
        beh_short = behaviors{beh_idx};
        if length(beh_short) > 8
            beh_short = beh_short(1:8);
        end
        broad_phase_feature_names{end+1} = sprintf('%s_%s', broad_bands{band_idx}, beh_short);
    end
end

% Compute correlation
broad_corr_matrix = corr(broad_phase_matrix, 'rows', 'pairwise');

% Plot
imagesc(broad_corr_matrix);
colorbar;
colormap(bluewhitered(256));
caxis([-1 1]);

set(gca, 'XTick', 1:length(broad_phase_feature_names), ...
         'XTickLabel', broad_phase_feature_names, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(broad_phase_feature_names), ...
         'YTickLabel', broad_phase_feature_names);
set(gca, 'FontSize', 6);

title('Phase Coupling Correlations - Broad Bands (MRL)', 'FontSize', 12, 'FontWeight', 'bold');
axis square;

% Add strong correlation values
for i = 1:length(broad_phase_feature_names)
    for j = 1:length(broad_phase_feature_names)
        if i ~= j && abs(broad_corr_matrix(i, j)) > 0.6
            text(j, i, sprintf('%.2f', broad_corr_matrix(i, j)), ...
                 'HorizontalAlignment', 'center', 'FontSize', 5, ...
                 'Color', 'k', 'FontWeight', 'bold');
        end
    end
end

sgtitle('Phase Coupling Feature Correlations (MRL)', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 3C: COHERENCE vs PHASE COUPLING CROSS-CORRELATIONS
%% ========================================================================

fprintf('Computing coherence-phase cross-correlations...\n');

fig2c = figure('Position', [250 50 1600 900], 'Name', 'Coherence-Phase Cross-Correlations');

% Build combined matrix: coherence features + phase features
% Use units that have both coherence and phase data
n_units_combined = min(n_units_coherence, n_units_phase_broad);

% Coherence features (11 features)
combined_coherence_matrix = [[coherence_features(1:n_units_combined).coherence_1_3Hz]', ...
                             [coherence_features(1:n_units_combined).coherence_5_7Hz]', ...
                             [coherence_features(1:n_units_combined).coherence_8_10Hz]', ...
                             [coherence_features(1:n_units_combined).coherence_delta]', ...
                             [coherence_features(1:n_units_combined).coherence_theta]', ...
                             [coherence_features(1:n_units_combined).coherence_beta]', ...
                             [coherence_features(1:n_units_combined).coherence_low_gamma]', ...
                             [coherence_features(1:n_units_combined).coherence_high_gamma]', ...
                             [coherence_features(1:n_units_combined).coherence_ultra_gamma]', ...
                             [coherence_features(1:n_units_combined).coherence_peak_freq]', ...
                             [coherence_features(1:n_units_combined).coherence_peak_mag]'];

coherence_feature_names_short = {'Coh_1-3Hz', 'Coh_5-7Hz', 'Coh_8-10Hz', 'Coh_Delta', 'Coh_Theta', ...
                                 'Coh_Beta', 'Coh_LowGam', 'Coh_HighGam', 'Coh_UltraGam', ...
                                 'Coh_PkFreq', 'Coh_PkMag'};

% Phase features (select representative ones: average MRL per band across behaviors)
% For narrow bands
phase_combined_names = {};
phase_combined_matrix = [];

for band_idx = 1:length(narrow_bands)
    % Average MRL across all behaviors for this band
    avg_mrl = [];
    for unit_idx = 1:n_units_combined
        mrl_values = phase_narrow_features(unit_idx).phase_MRL_narrow(band_idx, :);
        avg_mrl(end+1) = nanmean(mrl_values);
    end
    phase_combined_matrix = [phase_combined_matrix, avg_mrl'];
    phase_combined_names{end+1} = sprintf('MRL_N_%s_avg', narrow_bands{band_idx});
end

% For broad bands
for band_idx = 1:length(broad_bands)
    % Average MRL across all behaviors for this band
    avg_mrl = [];
    for unit_idx = 1:n_units_combined
        mrl_values = phase_broad_features(unit_idx).phase_MRL_broad(band_idx, :);
        avg_mrl(end+1) = nanmean(mrl_values);
    end
    phase_combined_matrix = [phase_combined_matrix, avg_mrl'];
    phase_combined_names{end+1} = sprintf('MRL_B_%s_avg', broad_bands{band_idx});
end

% Compute cross-correlation between coherence and phase
cross_corr_matrix = corr([combined_coherence_matrix, phase_combined_matrix], 'rows', 'pairwise');

% Extract the coherence-phase block
n_coh = length(coherence_feature_names_short);
n_phase = length(phase_combined_names);

% Full correlation matrix
subplot(1, 2, 1);
imagesc(cross_corr_matrix);
colorbar;
colormap(bluewhitered(256));
caxis([-1 1]);

all_feature_names = [coherence_feature_names_short, phase_combined_names];
set(gca, 'XTick', 1:length(all_feature_names), ...
         'XTickLabel', all_feature_names, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(all_feature_names), ...
         'YTickLabel', all_feature_names);
set(gca, 'FontSize', 8);

title('Full Correlation Matrix: Coherence + Phase', 'FontSize', 12, 'FontWeight', 'bold');
axis square;

% Add rectangle to highlight cross-correlation block
hold on;
rectangle('Position', [n_coh+0.5, 0.5, n_phase, n_coh], 'EdgeColor', 'k', 'LineWidth', 2);
rectangle('Position', [0.5, n_coh+0.5, n_coh, n_phase], 'EdgeColor', 'k', 'LineWidth', 2);
hold off;

% Coherence-Phase cross-correlation only (zoomed view)
subplot(1, 2, 2);
coh_phase_cross = cross_corr_matrix(1:n_coh, (n_coh+1):end);

imagesc(coh_phase_cross);
colorbar;
colormap(bluewhitered(256));
caxis([-1 1]);

set(gca, 'XTick', 1:n_phase, 'XTickLabel', phase_combined_names, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:n_coh, 'YTickLabel', coherence_feature_names_short);
set(gca, 'FontSize', 9);

title('Coherence-Phase Cross-Correlations', 'FontSize', 12, 'FontWeight', 'bold');

% Add correlation values
for i = 1:n_coh
    for j = 1:n_phase
        if abs(coh_phase_cross(i, j)) > 0.4
            text(j, i, sprintf('%.2f', coh_phase_cross(i, j)), ...
                 'HorizontalAlignment', 'center', 'FontSize', 7, ...
                 'Color', 'k', 'FontWeight', 'bold');
        end
    end
end

sgtitle('Coherence vs Phase Coupling Cross-Correlations', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 4: PHASE COUPLING VISUALIZATION (NARROW BANDS)
%% ========================================================================

fprintf('Visualizing phase coupling features (narrow bands)...\n');

% Get session types for phase features from struct array
session_types_phase = {phase_narrow_features.session_type};
is_aversive_phase = contains(session_types_phase, 'Aversive');
is_reward_phase = contains(session_types_phase, 'Reward');

% Define narrow bands and behaviors
narrow_bands = config.narrow_bands;  % {'1-3Hz', '5-7Hz', '8-10Hz'}
behaviors = config.behavior_names;   % 7 behaviors

% Create figure for narrow band phase coupling
fig3 = figure('Position', [200 200 1800 1000], 'Name', 'Phase Coupling - Narrow Bands');

% Extract MRL data from matrices stored in phase_narrow_features
% Each unit has phase_MRL_narrow which is a [3 bands × 7 behaviors] matrix
for band_idx = 1:length(narrow_bands)
    for beh_idx = 1:length(behaviors)
        subplot(length(narrow_bands), length(behaviors), (band_idx-1)*length(behaviors) + beh_idx);
        hold on;

        % Extract MRL values for this band × behavior from all units
        MRL_aversive = [];
        MRL_reward = [];

        for unit_idx = 1:n_units_phase_narrow
            if is_aversive_phase(unit_idx)
                MRL_aversive(end+1) = phase_narrow_features(unit_idx).phase_MRL_narrow(band_idx, beh_idx);
            else
                MRL_reward(end+1) = phase_narrow_features(unit_idx).phase_MRL_narrow(band_idx, beh_idx);
            end
        end

        % Remove NaN values
        MRL_aversive = MRL_aversive(~isnan(MRL_aversive));
        MRL_reward = MRL_reward(~isnan(MRL_reward));

        if ~isempty(MRL_aversive) || ~isempty(MRL_reward)
            if ~isempty(MRL_aversive)
                histogram(MRL_aversive, 15, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
            end
            if ~isempty(MRL_reward)
                histogram(MRL_reward, 15, 'FaceColor', [0.2 0.2 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
            end

            xlim([0 1]);
            xlabel('MRL');
            ylabel('Count');

            if band_idx == 1
                title(sprintf('%s', behaviors{beh_idx}), 'FontSize', 9);
            end

            if beh_idx == 1
                ylabel(sprintf('%s\nCount', narrow_bands{band_idx}));
            end

            grid on;
        end
    end
end

legend({'Aversive', 'Reward'}, 'Position', [0.92 0.45 0.05 0.05]);
sgtitle('Phase Coupling Strength (MRL) - Narrow Bands × Behaviors', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 5: PHASE COUPLING VISUALIZATION (BROAD BANDS)
%% ========================================================================

fprintf('Visualizing phase coupling features (broad bands)...\n');

% Get session types for broad band phase features from struct array
session_types_broad = {phase_broad_features.session_type};
is_aversive_broad = contains(session_types_broad, 'Aversive');
is_reward_broad = contains(session_types_broad, 'Reward');

% Define broad bands
broad_bands = config.broad_bands;  % {'Delta', 'Theta', 'Beta', 'Low_Gamma', 'High_Gamma', 'Ultra_Gamma'}

% Create figure for broad band phase coupling
fig4 = figure('Position', [250 250 1800 1200], 'Name', 'Phase Coupling - Broad Bands');

% Extract MRL data from matrices stored in phase_broad_features
% Each unit has phase_MRL_broad which is a [6 bands × 7 behaviors] matrix
for band_idx = 1:length(broad_bands)
    for beh_idx = 1:length(behaviors)
        subplot(length(broad_bands), length(behaviors), (band_idx-1)*length(behaviors) + beh_idx);
        hold on;

        % Extract MRL values for this band × behavior from all units
        MRL_aversive = [];
        MRL_reward = [];

        for unit_idx = 1:n_units_phase_broad
            if is_aversive_broad(unit_idx)
                MRL_aversive(end+1) = phase_broad_features(unit_idx).phase_MRL_broad(band_idx, beh_idx);
            else
                MRL_reward(end+1) = phase_broad_features(unit_idx).phase_MRL_broad(band_idx, beh_idx);
            end
        end

        % Remove NaN values
        MRL_aversive = MRL_aversive(~isnan(MRL_aversive));
        MRL_reward = MRL_reward(~isnan(MRL_reward));

        if ~isempty(MRL_aversive) || ~isempty(MRL_reward)
            if ~isempty(MRL_aversive)
                histogram(MRL_aversive, 15, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
            end
            if ~isempty(MRL_reward)
                histogram(MRL_reward, 15, 'FaceColor', [0.2 0.2 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
            end

            xlim([0 1]);
            xlabel('MRL');
            ylabel('Count');

            if band_idx == 1
                title(sprintf('%s', behaviors{beh_idx}), 'FontSize', 9);
            end

            if beh_idx == 1
                ylabel(sprintf('%s\nCount', broad_bands{band_idx}));
            end

            grid on;
        end
    end
end

legend({'Aversive', 'Reward'}, 'Position', [0.92 0.45 0.05 0.05]);
sgtitle('Phase Coupling Strength (MRL) - Broad Bands × Behaviors', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 6: PSTH FEATURE VISUALIZATION
%% ========================================================================

fprintf('Visualizing PSTH features...\n');

if ~isempty(psth_features)
    % Get session types for PSTH features from struct array
    session_types_psth = {psth_features.session_type};
    is_aversive_psth = contains(session_types_psth, 'Aversive');
    is_reward_psth = contains(session_types_psth, 'Reward');

    % Define key event types to visualize
    % Use actual event names from PSTH analysis
    event_types = {'IR1ON', 'IR2ON', 'WP1ON', 'WP2ON', 'AversiveOnset', ...
                   'Beh1_Onset', 'Beh2_Onset', 'Beh7_Onset'};
    event_labels = {'IR1ON Bout', 'IR2ON Bout', 'WP1ON Bout', 'WP2ON Bout', ...
                    'Aversive Onset', 'Reward Onset', 'Walking Onset', 'Standing Onset'};

    % Create figure for PSTH features
    fig5 = figure('Position', [300 300 1600 1000], 'Name', 'PSTH Response Features');

    plot_idx = 1;
    for e = 1:length(event_types)
        event_name = event_types{e};
        var_name = [event_name '_mean_z_0to1sec'];

        % Check if field exists in struct
        if isfield(psth_features, var_name)
            subplot(3, 3, plot_idx);
            hold on;

            % Extract values from struct array
            all_values = [psth_features.(var_name)];
            data_aversive = all_values(is_aversive_psth);
            data_reward = all_values(is_reward_psth);

            data_aversive = data_aversive(~isnan(data_aversive));
            data_reward = data_reward(~isnan(data_reward));

            if ~isempty(data_aversive) || ~isempty(data_reward)
                if ~isempty(data_aversive)
                    histogram(data_aversive, 20, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
                end
                if ~isempty(data_reward)
                    histogram(data_reward, 20, 'FaceColor', [0.2 0.2 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
                end

                xline(0, 'k--', 'LineWidth', 1);
                xline(2, 'r--', 'LineWidth', 0.5);
                xline(-2, 'r--', 'LineWidth', 0.5);

                xlabel('Mean Z-score [0-1 sec]');
                ylabel('Count');
                title(sprintf('%s', event_labels{e}));
                legend({'Aversive', 'Reward', 'Baseline'}, 'Location', 'best', 'FontSize', 8);
                grid on;

                % Statistical test
                if ~isempty(data_aversive) && ~isempty(data_reward)
                    [~, p] = ttest2(data_aversive, data_reward);
                    text(0.05, 0.95, sprintf('p = %.4f', p), 'Units', 'normalized', ...
                         'VerticalAlignment', 'top', 'FontSize', 8);
                end

                plot_idx = plot_idx + 1;
            end
        end
    end

    sgtitle('PSTH Response Magnitudes (Mean Z-score 0-1 sec post-event)', 'FontSize', 14, 'FontWeight', 'bold');

    %% ========================================================================
    %  SECTION 7: RESPONSE TYPE SUMMARY
    %% ========================================================================

    fprintf('Computing response type distributions...\n');

    fig6 = figure('Position', [350 350 1600 1000], 'Name', 'PSTH Response Types');

    plot_idx = 1;
    for e = 1:length(event_types)
        event_name = event_types{e};
        response_var = [event_name '_response_type'];

        % Check if field exists in struct
        if isfield(psth_features, response_var)
            subplot(3, 3, plot_idx);

            % Count response types - extract from struct array
            all_response_types = [psth_features.(response_var)];
            response_types_aversive = all_response_types(is_aversive_psth);
            response_types_reward = all_response_types(is_reward_psth);

            % Remove NaN
            response_types_aversive = response_types_aversive(~isnan(response_types_aversive));
            response_types_reward = response_types_reward(~isnan(response_types_reward));

            if ~isempty(response_types_aversive) || ~isempty(response_types_reward)
                % Count: -1 = inhibition, 0 = no response, 1 = excitation
                counts_aversive = [sum(response_types_aversive == -1), ...
                                  sum(response_types_aversive == 0), ...
                                  sum(response_types_aversive == 1)];
                counts_reward = [sum(response_types_reward == -1), ...
                                sum(response_types_reward == 0), ...
                                sum(response_types_reward == 1)];

                % Convert to percentages
                if sum(counts_aversive) > 0
                    pct_aversive = counts_aversive / sum(counts_aversive) * 100;
                else
                    pct_aversive = [0 0 0];
                end

                if sum(counts_reward) > 0
                    pct_reward = counts_reward / sum(counts_reward) * 100;
                else
                    pct_reward = [0 0 0];
                end

                % Plot grouped bar chart
                bar_data = [pct_aversive; pct_reward]';
                b = bar(bar_data, 'grouped');
                b(1).FaceColor = [0.8 0.2 0.2];
                b(2).FaceColor = [0.2 0.2 0.8];

                set(gca, 'XTickLabel', {'Inhibited', 'None', 'Excited'});
                ylabel('Percentage (%)');
                title(sprintf('%s', event_labels{e}));
                legend({'Aversive', 'Reward'}, 'Location', 'best', 'FontSize', 8);
                ylim([0 100]);
                grid on;

                plot_idx = plot_idx + 1;
            end
        end
    end

    sgtitle('Unit Response Types by Event (% of units)', 'FontSize', 14, 'FontWeight', 'bold');
else
    fprintf('  Skipping PSTH visualizations (no data)\n');
    fig5 = [];
    fig6 = [];
end

%% ========================================================================
%  SECTION 8: SAVE FIGURES
%% ========================================================================

% fprintf('\nSaving figures...\n');
% 
% saveas(fig1, 'Unit_Features_Coherence.png');
% saveas(fig2, 'Unit_Features_Coherence_Correlations.png');
% saveas(fig2b, 'Unit_Features_Phase_Correlations.png');
% saveas(fig2c, 'Unit_Features_Coherence_Phase_CrossCorrelations.png');
% saveas(fig3, 'Unit_Features_PhaseNarrow.png');
% saveas(fig4, 'Unit_Features_PhaseBroad.png');
% 
% if ~isempty(fig5)
%     saveas(fig5, 'Unit_Features_PSTH.png');
% end
% if ~isempty(fig6)
%     saveas(fig6, 'Unit_Features_PSTH_ResponseTypes.png');
% end
% 
% fprintf('✓ Saved all figures\n');

%% ========================================================================
%  SECTION 9: SUMMARY STATISTICS
%% ========================================================================

fprintf('\n=== FEATURE SUMMARY ===\n');

% Coherence summary - extract from struct array
all_coh_1_3 = [coherence_features.coherence_1_3Hz];
all_coh_5_7 = [coherence_features.coherence_5_7Hz];
all_coh_8_10 = [coherence_features.coherence_8_10Hz];
all_peak_mag = [coherence_features.coherence_peak_mag];
all_peak_freq = [coherence_features.coherence_peak_freq];

fprintf('\nCoherence features:\n');
fprintf('  Mean 1-3 Hz coherence: %.4f ± %.4f\n', ...
        nanmean(all_coh_1_3), nanstd(all_coh_1_3));
fprintf('  Mean 5-7 Hz coherence: %.4f ± %.4f\n', ...
        nanmean(all_coh_5_7), nanstd(all_coh_5_7));
fprintf('  Mean 8-10 Hz coherence: %.4f ± %.4f\n', ...
        nanmean(all_coh_8_10), nanstd(all_coh_8_10));
fprintf('  Peak coherence: %.4f ± %.4f at %.2f ± %.2f Hz\n', ...
        nanmean(all_peak_mag), nanstd(all_peak_mag), ...
        nanmean(all_peak_freq), nanstd(all_peak_freq));

% Session type comparison
fprintf('\nSession type comparison (coherence):\n');
fprintf('  Aversive 1-3 Hz: %.4f ± %.4f (n=%d)\n', ...
        nanmean(all_coh_1_3(is_aversive)), ...
        nanstd(all_coh_1_3(is_aversive)), ...
        sum(is_aversive));
fprintf('  Reward 1-3 Hz: %.4f ± %.4f (n=%d)\n', ...
        nanmean(all_coh_1_3(is_reward)), ...
        nanstd(all_coh_1_3(is_reward)), ...
        sum(is_reward));

% Phase coupling summary
fprintf('\nPhase coupling features (narrow bands):\n');
for band_idx = 1:length(narrow_bands)
    fprintf('  %s:\n', narrow_bands{band_idx});
    for beh_idx = 1:min(3, length(behaviors))  % Show first 3 behaviors
        all_MRL = [];
        for unit_idx = 1:n_units_phase_narrow
            all_MRL(end+1) = phase_narrow_features(unit_idx).phase_MRL_narrow(band_idx, beh_idx);
        end
        all_MRL = all_MRL(~isnan(all_MRL));
        fprintf('    %s: MRL = %.3f ± %.3f\n', behaviors{beh_idx}, ...
                nanmean(all_MRL), nanstd(all_MRL));
    end
end

fprintf('\n========================================\n');
fprintf('VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');

%% Helper function for blue-white-red colormap
function cmap = bluewhitered(n)
    if nargin < 1
        n = 256;
    end

    % Create blue to white to red colormap
    half = ceil(n/2);

    % Blue to white
    r1 = linspace(0, 1, half)';
    g1 = linspace(0, 1, half)';
    b1 = ones(half, 1);

    % White to red
    r2 = ones(half, 1);
    g2 = linspace(1, 0, half)';
    b2 = linspace(1, 0, half)';

    cmap = [r1 g1 b1; r2 g2 b2];
    cmap = cmap(1:n, :);
end
