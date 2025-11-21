%% ========================================================================
%  VISUALIZE SPIKE-TRIGGERED BREATHING SIGNAL AVERAGE
%  Check for phase-locking using BROADBAND BREATHING SIGNAL
%  ========================================================================
%
%  Creates visualizations to understand spike-breathing coupling
%
%  Figure 1: STA Waveforms (Broadband Breathing) - check for consistency
%  Figure 2: STA Power Spectrum - identify dominant breathing frequencies
%  Figure 3: STA Consistency across periods
%
%  BREATHING SIGNAL: Uses channel 32 from ephy data
%
%% ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: LOAD DATA
%  ========================================================================

fprintf('=== VISUALIZING SPIKE-TRIGGERED BREATHING SIGNAL AVERAGES ===\n');
fprintf('Breathing Signal (Channel 32)\n\n');

DataSetsPath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet';
RewardSeekingPath = fullfile(DataSetsPath, 'RewardSeeking_STA_Breathing');
RewardAversivePath = fullfile(DataSetsPath, 'RewardAversive_STA_Breathing');

% Load files
fprintf('Loading STA data...\n');
aversive_files = dir(fullfile(RewardAversivePath, '*_sta_breathing.mat'));
reward_files = dir(fullfile(RewardSeekingPath, '*_sta_breathing.mat'));

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

output_dir = 'STA_Breathing_Figures';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

color_aversive = [0.8 0.2 0.2];
color_reward = [0.2 0.4 0.8];

%% ========================================================================
%  FIGURE 1: STA WAVEFORMS (BROADBAND BREATHING)
%  ========================================================================

fprintf('Creating Figure 1: Broadband Breathing STA Waveforms...\n');

fig1 = figure('Position', [50, 50, 1600, 900]);
sgtitle('Spike-Triggered Breathing Signal Average (Broadband, Channel 32)', 'FontSize', 16, 'FontWeight', 'bold');

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
        xlabel('Time (s)'); ylabel('Breathing Signal (a.u.)');
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
        xlabel('Time (s)'); ylabel('Breathing Signal (a.u.)');
        title(sprintf('Reward P%d (n=%d)', p, height(reward_data)), 'FontWeight', 'bold');
        grid on; box on;
    end
end

saveas(fig1, fullfile(output_dir, 'Figure1_STA_Waveforms_Broadband_Breathing.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  FIGURE 2: STA POWER SPECTRA
%  ========================================================================

fprintf('Creating Figure 2: STA Power Spectra (Breathing)...\n');

fig2 = figure('Position', [100, 100, 1600, 800]);
sgtitle('Power Spectrum of Spike-Triggered Breathing Signal Average', 'FontSize', 16, 'FontWeight', 'bold');

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

saveas(fig2, fullfile(output_dir, 'Figure2_STA_Power_Spectrum_Breathing.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  FIGURE 3: STA CONSISTENCY ACROSS PERIODS
%  ========================================================================

fprintf('Creating Figure 3: STA Consistency across Periods...\n');

fig3 = figure('Position', [150, 150, 1600, 800]);
sgtitle('Breathing STA Consistency Across Periods (Lower = More Consistent)', 'FontSize', 16, 'FontWeight', 'bold');

% Aversive (7 periods)
subplot(1, 2, 1);
plot_consistency_by_period(tbl(tbl.SessionType == 'Aversive', :), 7, color_aversive);
title('Aversive Sessions', 'FontSize', 14, 'FontWeight', 'bold');

% Reward (4 periods)
subplot(1, 2, 2);
plot_consistency_by_period(tbl(tbl.SessionType == 'Reward', :), 4, color_reward);
title('Reward Sessions', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig3, fullfile(output_dir, 'Figure3_STA_Consistency_Breathing.png'));
fprintf('  ✓ Saved\n');

%% ========================================================================
%  COMPLETION
%  ========================================================================

fprintf('\n========================================\n');
fprintf('VISUALIZATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Figures saved to: %s/\n', output_dir);
fprintf('\nInterpretation (Breathing Signal - Channel 32):\n');
fprintf('  - Sharp, rhythmic STA waveform → Phase-locked to breathing\n');
fprintf('  - Flat/noisy STA → No phase-locking to breathing\n');
fprintf('  - Power spectrum peak at 1-4 Hz → Normal breathing coupling\n');
fprintf('  - Power spectrum peak at 4-12 Hz → Sniffing/exploratory breathing coupling\n');
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

    % Highlight breathing-relevant frequencies
    plot([2 2], ylim, 'r--', 'LineWidth', 2);
    text(2, max(10*log10(psd_avg)), ' 2 Hz (breathing)', 'FontSize', 10, 'Color', 'r', 'FontWeight', 'bold');

    plot([8 8], ylim, 'b--', 'LineWidth', 1.5);
    text(8, max(10*log10(psd_avg))*0.9, ' 8 Hz (sniffing)', 'FontSize', 10, 'Color', 'b', 'FontWeight', 'bold');

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
