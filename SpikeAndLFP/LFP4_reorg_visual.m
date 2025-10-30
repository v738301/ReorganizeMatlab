%% Load compiled results
% load('/Volumes/My980Pro/reorganize/SpikeAndLFP/DataSet/AllSessions_CompiledResults_20251015_174854.mat');
load('/Volumes/My980Pro/reorganize/SpikeAndLFP/DataSet/AllSessions_CompiledResults_20251016_004603.mat');

num_sessions = length(all_results.sessions);
fprintf('Analyzing %d sessions...\n', num_sessions);

%% 1. Average Breathing Rate Distribution
fprintf('Computing average breathing distribution...\n');

% Get common x-axis
xi_breathing = all_results.sessions(1).breathing.kde_x;

% Stack all KDE values
all_breathing_kde = zeros(length(xi_breathing), num_sessions);
for s = 1:num_sessions
    all_breathing_kde(:, s) = all_results.sessions(s).breathing.kde_y;
end

mean_breathing_kde = nanmean(all_breathing_kde, 2);
std_breathing_kde = nanstd(all_breathing_kde, 0, 2);
sem_breathing_kde = std_breathing_kde / sqrt(num_sessions);

mean_breathing_kde = fillmissing(mean_breathing_kde,'nearest');
std_breathing_kde = fillmissing(std_breathing_kde,'nearest');
sem_breathing_kde = fillmissing(sem_breathing_kde,'nearest');

%% 2. Average LFP Frequency Distribution
fprintf('Computing average LFP distribution...\n');

% Get common frequency axis
freq_lfp = all_results.sessions(1).lfp.freq;

% Stack all power spectra
all_lfp_power = zeros(length(freq_lfp), num_sessions);
for s = 1:num_sessions
    all_lfp_power(:, s) = all_results.sessions(s).lfp.powspctrm_normalized;
end

mean_lfp_power = nanmean(all_lfp_power, 2);
std_lfp_power = nanstd(all_lfp_power, 0, 2);
sem_lfp_power = std_lfp_power / sqrt(num_sessions);

mean_lfp_power = fillmissing(mean_lfp_power,'nearest');
std_lfp_power = fillmissing(std_lfp_power,'nearest');
sem_lfp_power = fillmissing(sem_lfp_power,'nearest');

%% 3. Average Breathing-LFP Coherence
fprintf('Computing average coherence...\n');

freq_coh = all_results.sessions(1).coherence.freq;
all_coherence = zeros(length(freq_coh), num_sessions);
all_coherence_norm = zeros(length(freq_coh), num_sessions);

for s = 1:num_sessions
    all_coherence(:, s) = all_results.sessions(s).coherence.Cxy;
    all_coherence_norm(:, s) = all_results.sessions(s).coherence.normalized;
end

mean_coherence = nanmean(all_coherence, 2);
std_coherence = nanstd(all_coherence, 0, 2);
mean_coherence_norm = nanmean(all_coherence_norm, 2);
std_coherence_norm = nanstd(all_coherence_norm, 0, 2);

mean_coherence = fillmissing(mean_coherence,'nearest');
std_coherence = fillmissing(std_coherence,'nearest');
mean_coherence_norm = fillmissing(mean_coherence_norm,'nearest');
std_coherence_norm = fillmissing(std_coherence_norm,'nearest');

%% 4. Average Breathing-to-LFP Cross-Frequency Coupling
fprintf('Computing average breathing-LFP coupling...\n');

gammarange = all_results.sessions(1).coupling_breathing_lfp.gammarange;
thetarange = all_results.sessions(1).coupling_breathing_lfp.thetarange;

all_coupling_BL = zeros(length(gammarange), length(thetarange), num_sessions);
all_phase_BL = zeros(length(gammarange), length(thetarange), num_sessions);

for s = 1:num_sessions
    all_coupling_BL(:,:,s) = all_results.sessions(s).coupling_breathing_lfp.normalized_modindex;
    all_phase_BL(:,:,s) = all_results.sessions(s).coupling_breathing_lfp.meanPhaseDeg;
end

mean_coupling_BL = nanmean(all_coupling_BL, 3);
std_coupling_BL = nanstd(all_coupling_BL, 0, 3);

mean_coupling_BL = fillmissing(mean_coupling_BL,'nearest');
std_coupling_BL = fillmissing(std_coupling_BL,'nearest');

% Circular mean for phase (convert to radians, compute, convert back)
phase_rad = deg2rad(all_phase_BL);
mean_phase_BL = atan2(mean(sin(phase_rad), 3), mean(cos(phase_rad), 3));
mean_phase_BL = rad2deg(mean_phase_BL);

%% 5. Average LFP-to-LFP Cross-Frequency Coupling
fprintf('Computing average LFP-LFP coupling...\n');

gammarange_lfp = all_results.sessions(1).coupling_lfp_lfp.gammarange;
thetarange_lfp = all_results.sessions(1).coupling_lfp_lfp.thetarange;

all_coupling_LL = zeros(length(gammarange_lfp), length(thetarange_lfp), num_sessions);

for s = 1:num_sessions
    all_coupling_LL(:,:,s) = all_results.sessions(s).coupling_lfp_lfp.normalized_modindex;
end

mean_coupling_LL = nanmean(all_coupling_LL, 3);
std_coupling_LL = nanstd(all_coupling_LL, 0, 3);

mean_coupling_LL = fillmissing(mean_coupling_LL,'nearest');
std_coupling_LL = fillmissing(std_coupling_LL,'nearest');

%% VISUALIZATION
figure('Position', [100 100 1800 1200]);

%% Row 1: Breathing and LFP Distributions
% 1.1 Breathing Distribution
subplot(3,3,1); hold on;
fill([xi_breathing, flip(xi_breathing)], ...
     [mean_breathing_kde' - sem_breathing_kde', flip(mean_breathing_kde' + sem_breathing_kde')],...
     'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(xi_breathing, mean_breathing_kde, 'b-', 'LineWidth', 2);
xlabel('Breathing Frequency (Hz)');
ylabel('Probability');
title(sprintf('Breathing Distribution (n=%d)', num_sessions));
grid on;

% 1.2 LFP Distribution
subplot(3,3,2);
fill([freq_lfp, flip(freq_lfp)], ...
     [mean_lfp_power' - sem_lfp_power', flip(mean_lfp_power' + sem_lfp_power')], ...
     'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
hold on;
plot(freq_lfp, mean_lfp_power, 'r-', 'LineWidth', 2);
xlabel('LFP Frequency (Hz)');
ylabel('Normalized Power');
title('LFP Power Spectrum');
xlim([0 150]);
grid on;

% 1.3 Coherence
subplot(3,3,3);
fill([freq_coh', flip(freq_coh')], ...
     [mean_coherence' - std_coherence', flip(mean_coherence' + std_coherence')], ...
     'g', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
hold on;
plot(freq_coh, mean_coherence, 'g-', 'LineWidth', 2);
xlabel('Frequency (Hz)');
ylabel('Coherence');
title('Breathing-LFP Coherence');
xlim([0 20]);
grid on;

%% Row 2: Cross-Frequency Coupling Maps (Mean)
% 2.1 Breathing-LFP Coupling Mean
subplot(3,3,4);
imagesc(thetarange, gammarange, mean_coupling_BL);
axis xy; colorbar;
xlabel('Breathing Frequency (Hz)');
ylabel('LFP Frequency (Hz)');
title('Mean Breathing-LFP Coupling');
% caxis([0 prctile(mean_coupling_BL(:), 95)]);

% 2.2 LFP-LFP Coupling Mean
subplot(3,3,5);
imagesc(thetarange_lfp, gammarange_lfp, mean_coupling_LL);
axis xy; colorbar;
xlabel('LFP Low Frequency (Hz)');
ylabel('LFP High Frequency (Hz)');
title('Mean LFP-LFP Coupling');
% caxis([0 prctile(mean_coupling_LL(:), 95)]);

% 2.3 Phase Map
subplot(3,3,6);
mean_phase_masked = mean_phase_BL;
mean_phase_masked(mean_coupling_BL < 2) = NaN;
imagesc(thetarange, gammarange, mean_phase_masked);
axis xy; colorbar;
% caxis([-180 180]);
xlabel('Breathing Frequency (Hz)');
ylabel('LFP Frequency (Hz)');
title('Mean Phase (Coupling > 2SD)');

%% Row 3: Variance Maps
% 3.1 Breathing-LFP Coupling Std
subplot(3,3,7);
imagesc(thetarange, gammarange, std_coupling_BL);
axis xy; colorbar;
xlabel('Breathing Frequency (Hz)');
ylabel('LFP Frequency (Hz)');
title('Std Breathing-LFP Coupling');

% 3.2 LFP-LFP Coupling Std
subplot(3,3,8);
imagesc(thetarange_lfp, gammarange_lfp, std_coupling_LL);
axis xy; colorbar;
xlabel('LFP Low Frequency (Hz)');
ylabel('LFP High Frequency (Hz)');
title('Std LFP-LFP Coupling');

% 3.3 Normalized Coherence
subplot(3,3,9);
fill([freq_coh; flipud(freq_coh)], ...
     [mean_coherence_norm + std_coherence_norm; flipud(mean_coherence_norm - std_coherence_norm)], ...
     'm', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
hold on;
plot(freq_coh, mean_coherence_norm, 'm-', 'LineWidth', 2);
yline(2, 'r--', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Normalized Coherence (Z-score)');
title('Normalized Coherence');
xlim([0 20]);
grid on;

sgtitle(sprintf('Average Metrics Across %d Sessions', num_sessions), 'FontSize', 16, 'FontWeight', 'bold');

%% Plot all Breathing-LFP coupling maps from all sessions

num_sessions = length(all_results.sessions);
gammarange = all_results.sessions(1).coupling_breathing_lfp.gammarange;
thetarange = all_results.sessions(1).coupling_breathing_lfp.thetarange;

% Determine grid size
n_cols = 4;
n_rows = ceil((num_sessions + 1) / n_cols); % +1 for mean

% Get global color limits
all_coupling_BL_stack = cat(3, all_coupling_BL, mean_coupling_BL);
clims = [0, prctile(all_coupling_BL_stack(:), 95)];

figure('Position', [100 100 1600 400*n_rows]);

% Plot each session
for s = 1:num_sessions
    subplot(n_rows, n_cols, s); hold on;
    imagesc(thetarange, gammarange, all_coupling_BL(:,:,s));

    % Overlay breathing frequency distribution at bottom
    % Scale to fit at the bottom of the y-axis
    xi = all_results.sessions(s).breathing.kde_x;
    f_scaled = all_results.sessions(s).breathing.kde_y;
    f_scaled_norm = f_scaled / max(mean_breathing_kde); % Normalize to 0-1
    breathing_y_offset = min(gammarange); % Start at bottom
    breathing_y_scale = (max(gammarange) - min(gammarange)) * 0.8; % Use 15% of y-range
    breathing_y = breathing_y_offset + f_scaled_norm * breathing_y_scale;
    plot(xi, breathing_y, 'r-', 'LineWidth', 2.5);

    % Overlay LFP frequency distribution at left
    % Scale to fit on the left side of x-axis
    powspctrm = all_results.sessions(s).lfp.powspctrm_normalized;
    freq = all_results.sessions(s).lfp.freq;
    power_norm = powspctrm / max(mean_lfp_power); % Normalize
    lfp_x_offset = min(thetarange); % Start at left edge
    lfp_x_scale = (max(thetarange) - min(thetarange)) * 0.5; % Use 15% of x-range
    lfp_x = lfp_x_offset + power_norm * lfp_x_scale;
    plot(lfp_x, freq, 'b-', 'LineWidth', 2.5);

    axis xy;
    colorbar;
    ylim([gammarange(1),gammarange(end)])
    xlim([thetarange(1),thetarange(end)])
%     caxis(clims);
    title(sprintf('Session %d', s));
    xlabel('Breathing Freq (Hz)');
    ylabel('LFP Freq (Hz)');
end

% Plot mean
subplot(n_rows, n_cols, num_sessions + 1); hold on;
imagesc(thetarange, gammarange, mean_coupling_BL);
% Overlay breathing frequency distribution at bottom
% Scale to fit at the bottom of the y-axis
xi = all_results.sessions(s).breathing.kde_x;
f_scaled = mean_breathing_kde;
f_scaled_norm = f_scaled / max(mean_breathing_kde); % Normalize to 0-1
breathing_y_offset = min(gammarange); % Start at bottom
breathing_y_scale = (max(gammarange) - min(gammarange)) * 0.8; % Use 15% of y-range
breathing_y = breathing_y_offset + f_scaled_norm * breathing_y_scale;
plot(xi, breathing_y, 'r-', 'LineWidth', 2.5);

% Overlay LFP frequency distribution at left
% Scale to fit on the left side of x-axis
powspctrm = mean_lfp_power;
freq = all_results.sessions(s).lfp.freq;
power_norm = powspctrm / max(mean_lfp_power); % Normalize
lfp_x_offset = min(thetarange); % Start at left edge
lfp_x_scale = (max(thetarange) - min(thetarange)) * 0.5; % Use 15% of x-range
lfp_x = lfp_x_offset + power_norm * lfp_x_scale;
plot(lfp_x, freq, 'b-', 'LineWidth', 2.5);
axis xy;
colorbar;
% caxis(clims);
ylim([gammarange(1),gammarange(end)])
xlim([thetarange(1),thetarange(end)])
title(sprintf('Mean (n=%d)', num_sessions));
xlabel('Breathing Freq (Hz)');
ylabel('LFP Freq (Hz)');

sgtitle('Breathing-LFP Coupling: All Sessions', 'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(gcf, '/Volumes/My980Pro/reorganize/SpikeAndLFP/DataSet/AllSessions_BreathingLFP_Coupling.png');

%% Save figure
saveas(gcf, '/Volumes/My980Pro/reorganize/SpikeAndLFP/DataSet/AverageMetrics_AllSessions.png');
saveas(gcf, '/Volumes/My980Pro/reorganize/SpikeAndLFP/DataSet/AverageMetrics_AllSessions.fig');

fprintf('Analysis complete!\n');