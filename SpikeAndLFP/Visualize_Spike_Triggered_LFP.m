%% ========================================================================
%  VISUALIZE SPIKE-TRIGGERED LFP AVERAGE
%  Check for phase-locking using BROADBAND LFP
%  ========================================================================
%
%  Creates visualizations to understand High SFC + Low PPC
%
%  Figure 1: STA Waveforms (Broadband) - check for consistency
%  Figure 2: STA Power Spectrum - identify dominant frequencies
%  Figure 3: STA Consistency across periods
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: LOAD DATA
%  ========================================================================

fprintf('=== VISUALIZING SPIKE-TRIGGERED LFP AVERAGES ===\n\n');

DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_STA');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_STA');

% Load files
fprintf('Loading STA data...\n');
aversive_files = dir(fullfile(RewardAversivePath, '*_sta.mat'));
reward_files = dir(fullfile(RewardSeekingPath, '*_sta.mat'));

fprintf('  Aversive: %d files\n', length(aversive_files));
fprintf('  Reward: %d files\n', length(reward_files));

% Combine data - collect all data into arrays first, then create single table
all_units = [];
all_periods = [];
all_sta_waveforms = {};
all_sta_peaks = [];
all_sta_consistency = [];
all_n_spikes = [];
all_session_types = {};
all_session_ids = [];  % Track which session each row came from
all_session_names = {};  % Track session filenames

session_id_counter = 0;

for i = 1:length(aversive_files)
    load(fullfile(RewardAversivePath, aversive_files(i).name), 'session_results');
    if ~isempty(session_results.data) && height(session_results.data) > 0
        session_id_counter = session_id_counter + 1;

        % Handle struct2table issue: if table has 1 row with array-valued columns,
        % extract the arrays directly
        if height(session_results.data) == 1
            % Data stored as arrays in single row - extract them
            units = session_results.data.Unit(:);
            periods = session_results.data.Period(:);
            sta_waveforms = session_results.data.STA_Waveform(:);
            sta_peaks = session_results.data.STA_Peak(:);
            sta_consistency = session_results.data.STA_Consistency(:);
            n_spikes = session_results.data.N_spikes(:);
            n_rows = length(units);
        else
            % Normal multi-row table
            n_rows = height(session_results.data);
            units = session_results.data.Unit(:);
            periods = session_results.data.Period(:);
            sta_waveforms = session_results.data.STA_Waveform(:);
            sta_peaks = session_results.data.STA_Peak(:);
            sta_consistency = session_results.data.STA_Consistency(:);
            n_spikes = session_results.data.N_spikes(:);
        end

        all_units = [all_units; units];
        all_periods = [all_periods; periods];
        all_sta_waveforms = [all_sta_waveforms; sta_waveforms];
        all_sta_peaks = [all_sta_peaks; sta_peaks];
        all_sta_consistency = [all_sta_consistency; sta_consistency];
        all_n_spikes = [all_n_spikes; n_spikes];
        all_session_types = [all_session_types; repmat({'Aversive'}, n_rows, 1)];
        all_session_ids = [all_session_ids; repmat(session_id_counter, n_rows, 1)];
        all_session_names = [all_session_names; repmat({aversive_files(i).name}, n_rows, 1)];
    end
end

for i = 1:length(reward_files)
    load(fullfile(RewardSeekingPath, reward_files(i).name), 'session_results', 'config');
    if ~isempty(session_results.data) && height(session_results.data) > 0
        session_id_counter = session_id_counter + 1;

        % Handle struct2table issue: if table has 1 row with array-valued columns,
        % extract the arrays directly
        if height(session_results.data) == 1
            % Data stored as arrays in single row - extract them
            units = session_results.data.Unit(:);
            periods = session_results.data.Period(:);
            sta_waveforms = session_results.data.STA_Waveform(:);
            sta_peaks = session_results.data.STA_Peak(:);
            sta_consistency = session_results.data.STA_Consistency(:);
            n_spikes = session_results.data.N_spikes(:);
            n_rows = length(units);
        else
            % Normal multi-row table
            n_rows = height(session_results.data);
            units = session_results.data.Unit(:);
            periods = session_results.data.Period(:);
            sta_waveforms = session_results.data.STA_Waveform(:);
            sta_peaks = session_results.data.STA_Peak(:);
            sta_consistency = session_results.data.STA_Consistency(:);
            n_spikes = session_results.data.N_spikes(:);
        end

        all_units = [all_units; units];
        all_periods = [all_periods; periods];
        all_sta_waveforms = [all_sta_waveforms; sta_waveforms];
        all_sta_peaks = [all_sta_peaks; sta_peaks];
        all_sta_consistency = [all_sta_consistency; sta_consistency];
        all_n_spikes = [all_n_spikes; n_spikes];
        all_session_types = [all_session_types; repmat({'Reward'}, n_rows, 1)];
        all_session_ids = [all_session_ids; repmat(session_id_counter, n_rows, 1)];
        all_session_names = [all_session_names; repmat({reward_files(i).name}, n_rows, 1)];
        time_vec = session_results.time_vec;  % Get time vector
    end
end

if isempty(all_units)
    error('No data found in any session files');
end

% Create unique unit ID: SessionID * 1000 + Unit (prevents collision across sessions)
all_unique_unit_ids = all_session_ids * 1000 + double(all_units);

% Create single table with guaranteed correct dimensions
tbl = table(all_units, all_periods, all_sta_waveforms, all_sta_peaks, ...
            all_sta_consistency, all_n_spikes, all_session_types, ...
            all_session_ids, all_session_names, all_unique_unit_ids, ...
            'VariableNames', {'Unit', 'Period', 'STA_Waveform', 'STA_Peak', ...
                              'STA_Consistency', 'N_spikes', 'SessionType', ...
                              'SessionID', 'SessionName', 'UniqueUnitID'});
tbl.SessionType = categorical(tbl.SessionType);

fprintf('✓ Data loaded: %d total entries\n\n', height(tbl));

%% ========================================================================
%  SECTION 2: SETUP
%  ========================================================================

output_dir = 'STA_Figures';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

color_aversive = [0.8 0.2 0.2];
color_reward = [0.2 0.4 0.8];

%% ========================================================================
%  FIGURE 1: STA WAVEFORMS (BROADBAND)
%  ========================================================================

fprintf('Creating Figure 1: Broadband STA Waveforms...\n');

fig1 = figure('Position', [50, 50, 1600, 900]);
sgtitle('Spike-Triggered LFP Average (Broadband)', 'FontSize', 16, 'FontWeight', 'bold');

% Plot first 4 periods for both session types
for p = 1:4
    % Aversive
    subplot(2, 4, p);
    aversive_data = tbl(tbl.SessionType == 'Aversive' & tbl.Period == p, :);
    if ~isempty(aversive_data)
        % Average across all units
        all_waveforms = cell2mat(aversive_data.STA_Waveform);
        mean_sta = mean(all_waveforms, 1);
        sem_sta = std(all_waveforms, 0, 1) / sqrt(size(all_waveforms, 1));

        hold on;
        fill([time_vec, fliplr(time_vec)], ...
             [mean_sta - sem_sta, fliplr(mean_sta + sem_sta)], ...
             color_aversive, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(time_vec, mean_sta, 'Color', color_aversive, 'LineWidth', 2);
        plot([0 0], ylim, 'k--', 'LineWidth', 1);
        xlabel('Time (s)'); ylabel('LFP (μV)');
        title(sprintf('Aversive P%d (n=%d)', p, height(aversive_data)), 'FontWeight', 'bold');
        grid on; box on;
    end

    % Reward
    subplot(2, 4, 4 + p);
    reward_data = tbl(tbl.SessionType == 'Reward' & tbl.Period == p, :);
    if ~isempty(reward_data)
        all_waveforms = cell2mat(reward_data.STA_Waveform);
        mean_sta = mean(all_waveforms, 1);
        sem_sta = std(all_waveforms, 0, 1) / sqrt(size(all_waveforms, 1));

        hold on;
        fill([time_vec, fliplr(time_vec)], ...
             [mean_sta - sem_sta, fliplr(mean_sta + sem_sta)], ...
             color_reward, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(time_vec, mean_sta, 'Color', color_reward, 'LineWidth', 2);
        plot([0 0], ylim, 'k--', 'LineWidth', 1);
        xlabel('Time (s)'); ylabel('LFP (μV)');
        title(sprintf('Reward P%d (n=%d)', p, height(reward_data)), 'FontWeight', 'bold');
        grid on; box on;
    end
end

saveas(fig1, fullfile(output_dir, 'Figure1_STA_Waveforms_Broadband.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  FIGURE 2: STA POWER SPECTRA
%  ========================================================================

fprintf('Creating Figure 2: STA Power Spectra...\n');

fig2 = figure('Position', [100, 100, 1600, 800]);
sgtitle('Power Spectrum of Spike-Triggered LFP Average', 'FontSize', 16, 'FontWeight', 'bold');

% Get sampling rate from session_results
Fs = session_results.Fs;

% Aversive
subplot(1, 2, 1);
plot_sta_power_spectrum(tbl(tbl.SessionType == 'Aversive', :), Fs, color_aversive);
title('Aversive Sessions', 'FontSize', 14, 'FontWeight', 'bold');

% Reward
subplot(1, 2, 2);
plot_sta_power_spectrum(tbl(tbl.SessionType == 'Reward', :), Fs, color_reward);
title('Reward Sessions', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig2, fullfile(output_dir, 'Figure2_STA_Power_Spectrum.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  FIGURE 3: STA CONSISTENCY ACROSS PERIODS
%  ========================================================================

fprintf('Creating Figure 3: STA Consistency across Periods...\n');

fig3 = figure('Position', [150, 150, 1600, 800]);
sgtitle('STA Consistency Across Periods (Lower = More Consistent)', 'FontSize', 16, 'FontWeight', 'bold');

% Aversive (7 periods)
subplot(1, 2, 1);
plot_consistency_by_period(tbl(tbl.SessionType == 'Aversive', :), 7, color_aversive);
title('Aversive Sessions', 'FontSize', 14, 'FontWeight', 'bold');

% Reward (4 periods)
subplot(1, 2, 2);
plot_consistency_by_period(tbl(tbl.SessionType == 'Reward', :), 4, color_reward);
title('Reward Sessions', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig3, fullfile(output_dir, 'Figure3_STA_Consistency.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  FIGURE 6: UNIT × TIME HEATMAP (All STA Waveforms)
%  ========================================================================

fprintf('Creating Figure 6: Unit × Time Heatmap...\n');

% CONFIGURABLE: Select which periods to include in heatmap (default: 1-4)
periods_to_include = 1:4;

fig6 = figure('Position', [250, 250, 1800, 900]);
if length(periods_to_include) == 1
    sgtitle(sprintf('All Units × Time: Spike-Triggered Average (Period %d)', periods_to_include), 'FontSize', 16, 'FontWeight', 'bold');
else
    sgtitle(sprintf('All Units × Time: Spike-Triggered Average (Periods %d-%d avg)', min(periods_to_include), max(periods_to_include)), 'FontSize', 16, 'FontWeight', 'bold');
end

% Aversive (average across selected periods)
subplot(1, 2, 1);
aversive_selected = tbl(tbl.SessionType == 'Aversive' & ismember(tbl.Period, periods_to_include), :);
if ~isempty(aversive_selected)
    heatmap_data_av = createSTAHeatmap(aversive_selected);
    imagesc(time_vec, 1:size(heatmap_data_av, 1), heatmap_data_av, [-100,100]);
    set(gca, 'YDir', 'normal');
    colorbar;
    colormap(jet);
    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Unit', 'FontSize', 12, 'FontWeight', 'bold');
    if length(periods_to_include) == 1
        title(sprintf('Aversive P%d (n=%d units)', periods_to_include, size(heatmap_data_av, 1)), 'FontSize', 13, 'FontWeight', 'bold');
    else
        title(sprintf('Aversive P%d-%d avg (n=%d units)', min(periods_to_include), max(periods_to_include), size(heatmap_data_av, 1)), 'FontSize', 13, 'FontWeight', 'bold');
    end
    hold on;
    plot([0 0], ylim, 'k--', 'LineWidth', 2);
    hold off;
end

% Reward (average across selected periods)
subplot(1, 2, 2);
reward_selected = tbl(tbl.SessionType == 'Reward' & ismember(tbl.Period, periods_to_include), :);
if ~isempty(reward_selected)
    heatmap_data_rw = createSTAHeatmap(reward_selected);
    imagesc(time_vec, 1:size(heatmap_data_rw, 1), heatmap_data_rw, [-100,100]);
    set(gca, 'YDir', 'normal');
    colorbar;
    colormap(jet);
    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Unit', 'FontSize', 12, 'FontWeight', 'bold');
    if length(periods_to_include) == 1
        title(sprintf('Reward P%d (n=%d units)', periods_to_include, size(heatmap_data_rw, 1)), 'FontSize', 13, 'FontWeight', 'bold');
    else
        title(sprintf('Reward P%d-%d avg (n=%d units)', min(periods_to_include), max(periods_to_include), size(heatmap_data_rw, 1)), 'FontSize', 13, 'FontWeight', 'bold');
    end
    hold on;
    plot([0 0], ylim, 'k--', 'LineWidth', 2);
    hold off;
end

saveas(fig6, fullfile(output_dir, 'Figure6_Unit_Time_STA_Heatmap.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  COMPLETION
%  ========================================================================

fprintf('\n========================================\n');
fprintf('VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Figures saved to: %s/\n', output_dir);
fprintf('\nInterpretation:\n');
fprintf('  - Sharp, rhythmic STA waveform → Phase-locked (high PPC expected)\n');
fprintf('  - Flat/noisy STA → No phase-locking (low PPC, but could have high SFC)\n');
fprintf('  - Power spectrum peak at 8 Hz → Theta coupling\n');
fprintf('  - Low consistency → Amplitude-modulated, not phase-locked\n');
fprintf('========================================\n');


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function plot_sta_power_spectrum(data, Fs, color)
% Plot average power spectrum of STA waveforms

    % Get all waveforms
    all_waveforms = cell2mat(data.STA_Waveform');

    if isempty(all_waveforms)
        return;
    end

    % Compute power spectrum for each waveform
    n_waveforms = size(all_waveforms, 1);
    nfft = size(all_waveforms, 2);

    % Average power spectrum
    psd_avg = zeros(1, floor(nfft/2) + 1);
    for w = 1:n_waveforms
        [psd, freqs] = periodogram(all_waveforms(w, :), [], nfft, Fs);
        psd_avg = psd_avg + psd';
    end
    psd_avg = psd_avg / n_waveforms;

    % Plot
    hold on;
    plot(freqs, 10*log10(psd_avg), 'Color', color, 'LineWidth', 2);

    % Highlight 8 Hz
    plot([8 8], ylim, 'r--', 'LineWidth', 2);
    text(8, max(10*log10(psd_avg)), ' 8 Hz', 'FontSize', 12, 'Color', 'r', 'FontWeight', 'bold');

    xlabel('Frequency (Hz)', 'FontSize', 12);
    ylabel('Power (dB)', 'FontSize', 12);
    grid on; box on;
    xlim([0 20]);
end

function plot_consistency_by_period(data, n_periods, color)
% Plot STA consistency across periods

    mean_cons = zeros(1, n_periods);
    sem_cons = zeros(1, n_periods);

    for p = 1:n_periods
        period_data = data(data.Period == p, :);
        if ~isempty(period_data)
            mean_cons(p) = mean(period_data.STA_Consistency, 'omitnan');
            sem_cons(p) = std(period_data.STA_Consistency, 'omitnan') / sqrt(sum(~isnan(period_data.STA_Consistency)));
        end
    end

    hold on;
    errorbar(1:n_periods, mean_cons, sem_cons, '-o', 'Color', color, ...
             'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', color);

    xlabel('Period', 'FontSize', 12);
    ylabel('STA Consistency (a.u.)', 'FontSize', 12);
    grid on; box on;
    xlim([0.5 n_periods + 0.5]);
    set(gca, 'XTick', 1:n_periods);
end

function plot_single_session(tbl, session_id, time_vec, output_dir, color, n_periods)
% Plot all units from a single session across periods

    sess_data = tbl(tbl.SessionID == session_id, :);
    session_name = sess_data.SessionName{1};
    session_type = char(sess_data.SessionType(1));

    % Get unique units
    unique_units = unique(sess_data.Unit);
    n_units = length(unique_units);

    % Create figure with subplots for each unit
    n_cols = min(4, n_periods);
    n_rows = ceil(n_units / n_cols);

    fig = figure('Position', [50, 50, 400*n_cols, 300*n_rows]);
    sgtitle(sprintf('%s Session - %s', session_type, strrep(session_name, '_', '\_')), ...
            'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

    for u_idx = 1:n_units
        unit_id = unique_units(u_idx);
        unit_data = sess_data(sess_data.Unit == unit_id, :);

        % Get STA consistency for this unit
        mean_consistency = mean(unit_data.STA_Consistency, 'omitnan');

        for p = 1:min(n_periods, height(unit_data))
            period_row = unit_data(unit_data.Period == p, :);

            if ~isempty(period_row)
                subplot(n_rows, n_cols, u_idx);
                hold on;

                % Plot all periods for this unit
                sta_waveform = period_row.STA_Waveform{1};
                plot(time_vec, sta_waveform, 'Color', color, 'LineWidth', 1.5);
            end
        end

        subplot(n_rows, n_cols, u_idx);
        plot([0 0], ylim, 'k--', 'LineWidth', 1);
        xlabel('Time (s)'); ylabel('LFP (μV)');
        title(sprintf('Unit %d (cons=%.2f)', unit_id, mean_consistency), 'FontSize', 10);
        grid on; box on;
    end

    % Save
    [~, name_only, ~] = fileparts(session_name);
    saveas(fig, fullfile(output_dir, sprintf('Session_%d_%s.png', session_id, name_only)));
    close(fig);
end

function plot_single_unit(tbl, session_id, unit_id, time_vec, output_dir, color, n_periods, consistency_label)
% Plot single unit across all periods

    unit_data = tbl(tbl.SessionID == session_id & tbl.Unit == unit_id, :);

    if isempty(unit_data)
        return;
    end

    session_name = unit_data.SessionName{1};
    session_type = char(unit_data.SessionType(1));
    mean_consistency = mean(unit_data.STA_Consistency, 'omitnan');

    % Create figure
    fig = figure('Position', [50, 50, 1400, 400]);
    sgtitle(sprintf('%s - Unit %d - Consistency: %.3f (%s)', ...
            strrep(session_name, '_', '\_'), unit_id, mean_consistency, consistency_label), ...
            'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');

    % Plot each period
    for p = 1:n_periods
        period_row = unit_data(unit_data.Period == p, :);

        subplot(1, n_periods, p);

        if ~isempty(period_row)
            sta_waveform = period_row.STA_Waveform{1};
            n_spikes = period_row.N_spikes;
            sta_peak = period_row.STA_Peak;

            hold on;
            plot(time_vec, sta_waveform, 'Color', color, 'LineWidth', 2);
            plot([0 0], ylim, 'k--', 'LineWidth', 1);
            xlabel('Time (s)'); ylabel('LFP (μV)');
            title(sprintf('P%d (n=%d, pk=%.1f)', p, n_spikes, sta_peak), 'FontSize', 10);
            grid on; box on;
        else
            title(sprintf('P%d (no data)', p), 'FontSize', 10);
        end
    end

    % Save
    [~, name_only, ~] = fileparts(session_name);
    saveas(fig, fullfile(output_dir, sprintf('Unit_Sess%d_Unit%d_%s_%s.png', ...
                         session_id, unit_id, consistency_label, name_only)));
    close(fig);
end

function heatmap_data = createSTAHeatmap(data)
% Create heatmap matrix: units × time points
%
% INPUTS:
%   data - Table with STA_Waveform for one period
%
% OUTPUTS:
%   heatmap_data - Matrix (n_units × n_timepoints)

    % FIX: Use UniqueUnitID to prevent collision across sessions
    unique_units = unique(data.UniqueUnitID);
    n_units = length(unique_units);

    % Get first waveform to determine size
    n_timepoints = length(data.STA_Waveform{1});

    heatmap_data = nan(n_units, n_timepoints);

    for u = 1:n_units
        % FIX: Filter by UniqueUnitID instead of Unit
        unit_data = data(data.UniqueUnitID == unique_units(u), :);

        if ~isempty(unit_data) && ~isempty(unit_data.STA_Waveform{1})
            % Average across periods if multiple entries for this unique unit
            if height(unit_data) > 1
                all_waveforms = cell2mat(unit_data.STA_Waveform);
                heatmap_data(u, :) = mean(all_waveforms, 1);
            else
                heatmap_data(u, :) = unit_data.STA_Waveform{1};
            end
        end
    end
end
