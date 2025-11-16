%% ========================================================================
%  BREATHING RATE DISTRIBUTION COMPARISON
%  Compare breathing rate distributions between Reward-Seeking and
%  Reward-Aversive sessions
%  ========================================================================
%
%  This script loads coupling strength datasets and extracts breathing
%  rates (column 8 of behavioral_matrix) to visualize distributions
%  across session types.
%
%% ========================================================================

clear all;
close all;

%% ========================================================================
%  SECTION 1: LOAD DATA
%  ========================================================================

fprintf('=== BREATHING RATE DISTRIBUTION COMPARISON ===\n\n');

fprintf('Loading data...\n');

% Load aversive sessions
try
    coupling_data_aversive = load('27-Oct-2025_RewardAversive_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_aversive = coupling_data_aversive.all_session_metrics;
    fprintf('✓ Loaded %d aversive sessions\n', length(sessions_aversive));
catch ME
    fprintf('❌ Failed to load aversive data: %s\n', ME.message);
    return;
end

% Load reward sessions
try
    coupling_data_reward = load('27-Oct-2025_RewardSeeking_session_metrics_breathing_LFPCcouple(10-1)');
    sessions_reward = coupling_data_reward.all_session_metrics;
    fprintf('✓ Loaded %d reward sessions\n\n', length(sessions_reward));
catch ME
    fprintf('❌ Failed to load reward data: %s\n', ME.message);
    return;
end

%% ========================================================================
%  SECTION 2: EXTRACT BREATHING RATES
%  ========================================================================

fprintf('Extracting breathing rates from behavioral matrices...\n');

% Storage for breathing rates
breathing_aversive = [];
breathing_reward = [];

% Extract from aversive sessions
fprintf('Processing aversive sessions...\n');
for sess_idx = 1:length(sessions_aversive)
    session = sessions_aversive{sess_idx};

    if isfield(session, 'behavioral_matrix_full') && ~isempty(session.behavioral_matrix_full)
        behavioral_matrix = session.behavioral_matrix_full;

        % Column 8 is breathing frequency (Hz)
        if size(behavioral_matrix, 2) >= 8
            breathing_freq = behavioral_matrix(:, 8);

            % Remove zeros and NaNs
            breathing_freq = breathing_freq(breathing_freq > 0 & ~isnan(breathing_freq));

            % Also remove extreme outliers (> 20 Hz is physiologically unrealistic)
            breathing_freq = breathing_freq(breathing_freq < 20);

            breathing_aversive = [breathing_aversive; breathing_freq];

            fprintf('  Session %d: %s - %d valid breathing samples\n', ...
                sess_idx, session.filename, length(breathing_freq));
        else
            fprintf('  Session %d: %s - No breathing data (matrix too small)\n', ...
                sess_idx, session.filename);
        end
    else
        fprintf('  Session %d: No behavioral_matrix_full\n', sess_idx);
    end
end

% Extract from reward sessions
fprintf('\nProcessing reward sessions...\n');
for sess_idx = 1:length(sessions_reward)
    session = sessions_reward{sess_idx};

    if isfield(session, 'behavioral_matrix_full') && ~isempty(session.behavioral_matrix_full)
        behavioral_matrix = session.behavioral_matrix_full;

        % Column 8 is breathing frequency (Hz)
        if size(behavioral_matrix, 2) >= 8
            breathing_freq = behavioral_matrix(:, 8);

            % Remove zeros and NaNs
            breathing_freq = breathing_freq(breathing_freq > 0 & ~isnan(breathing_freq));

            % Also remove extreme outliers (> 20 Hz)
            breathing_freq = breathing_freq(breathing_freq < 20);

            breathing_reward = [breathing_reward; breathing_freq];

            fprintf('  Session %d: %s - %d valid breathing samples\n', ...
                sess_idx, session.filename, length(breathing_freq));
        else
            fprintf('  Session %d: %s - No breathing data (matrix too small)\n', ...
                sess_idx, session.filename);
        end
    else
        fprintf('  Session %d: No behavioral_matrix_full\n', sess_idx);
    end
end

fprintf('\n✓ Extraction complete\n');
fprintf('  Aversive samples: %d\n', length(breathing_aversive));
fprintf('  Reward samples: %d\n\n', length(breathing_reward));

if isempty(breathing_aversive) || isempty(breathing_reward)
    error('No breathing data found in one or both session types!');
end

%% ========================================================================
%  SECTION 3: CALCULATE STATISTICS
%  ========================================================================

fprintf('Calculating statistics...\n');

stats = struct();

% Aversive statistics
stats.aversive.mean = mean(breathing_aversive);
stats.aversive.median = median(breathing_aversive);
stats.aversive.std = std(breathing_aversive);
stats.aversive.sem = std(breathing_aversive) / sqrt(length(breathing_aversive));
stats.aversive.n = length(breathing_aversive);
stats.aversive.min = min(breathing_aversive);
stats.aversive.max = max(breathing_aversive);
stats.aversive.q25 = prctile(breathing_aversive, 25);
stats.aversive.q75 = prctile(breathing_aversive, 75);

% Reward statistics
stats.reward.mean = mean(breathing_reward);
stats.reward.median = median(breathing_reward);
stats.reward.std = std(breathing_reward);
stats.reward.sem = std(breathing_reward) / sqrt(length(breathing_reward));
stats.reward.n = length(breathing_reward);
stats.reward.min = min(breathing_reward);
stats.reward.max = max(breathing_reward);
stats.reward.q25 = prctile(breathing_reward, 25);
stats.reward.q75 = prctile(breathing_reward, 75);

% Statistical test (Wilcoxon rank-sum test for non-parametric comparison)
[p_value, h] = ranksum(breathing_aversive, breathing_reward);
stats.ranksum_p = p_value;
stats.ranksum_h = h;

% Effect size (Cohen's d)
pooled_std = sqrt((stats.aversive.std^2 + stats.reward.std^2) / 2);
stats.cohens_d = (stats.aversive.mean - stats.reward.mean) / pooled_std;

fprintf('✓ Statistics calculated\n\n');

%% ========================================================================
%  SECTION 4: VISUALIZATION 1 - HISTOGRAMS
%  ========================================================================

fprintf('Creating histogram comparison...\n');

figure('Position', [50, 50, 1400, 600], 'Name', 'Breathing Rate Distributions');

% Define bins (0 to 15 Hz, 0.25 Hz bins)
edges = 0:0.25:15;

% Subplot 1: Overlaid histograms
subplot(1, 2, 1);
hold on;

histogram(breathing_aversive, edges, 'FaceColor', [1, 0.6, 0.6], ...
    'EdgeColor', [0.8, 0, 0], 'FaceAlpha', 0.6, 'Normalization', 'probability');
histogram(breathing_reward, edges, 'FaceColor', [0.6, 1, 0.6], ...
    'EdgeColor', [0, 0.6, 0], 'FaceAlpha', 0.6, 'Normalization', 'probability');

% Add median lines
ymax = max(ylim);
plot([stats.aversive.median, stats.aversive.median], [0, ymax], 'r--', ...
    'LineWidth', 2, 'DisplayName', sprintf('Aversive median: %.2f Hz', stats.aversive.median));
plot([stats.reward.median, stats.reward.median], [0, ymax], 'g--', ...
    'LineWidth', 2, 'DisplayName', sprintf('Reward median: %.2f Hz', stats.reward.median));

hold off;

xlabel('Breathing Rate (Hz)', 'FontSize', 11);
ylabel('Probability', 'FontSize', 11);
title('Breathing Rate Distribution Comparison', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward', sprintf('Aversive median: %.2f Hz', stats.aversive.median), ...
    sprintf('Reward median: %.2f Hz', stats.reward.median)}, 'Location', 'northeast', 'FontSize', 9);
grid on;
set(gca, 'FontSize', 10);

% Add text with statistics
text(0.98, 0.95, sprintf('p = %.4f', p_value), 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
    'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', 'w');

% Subplot 2: Cumulative distributions
subplot(1, 2, 2);
hold on;

[f_aver, x_aver] = ecdf(breathing_aversive);
[f_rew, x_rew] = ecdf(breathing_reward);

plot(x_aver, f_aver, 'r-', 'LineWidth', 2, 'DisplayName', 'Aversive');
plot(x_rew, f_rew, 'g-', 'LineWidth', 2, 'DisplayName', 'Reward');

% Add median markers
plot(stats.aversive.median, 0.5, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(stats.reward.median, 0.5, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');

hold off;

xlabel('Breathing Rate (Hz)', 'FontSize', 11);
ylabel('Cumulative Probability', 'FontSize', 11);
title('Cumulative Distribution Functions', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);
xlim([0, 15]);

sgtitle('Breathing Rate: Reward-Seeking vs Reward-Aversive Sessions', ...
    'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Histogram comparison created\n');

%% ========================================================================
%  SECTION 5: VISUALIZATION 2 - BOX PLOTS AND VIOLIN PLOTS
%  ========================================================================

fprintf('Creating box plot and density comparison...\n');

figure('Position', [100, 100, 1200, 600], 'Name', 'Breathing Rate Summary');

% Subplot 1: Box plot
subplot(1, 2, 1);

% Prepare data for grouped box plot
all_breathing = [breathing_aversive; breathing_reward];
group_labels = [repmat({'Aversive'}, length(breathing_aversive), 1); ...
                repmat({'Reward'}, length(breathing_reward), 1)];

boxplot(all_breathing, group_labels, 'Colors', 'kr', 'Symbol', '');
hold on;

% Add mean markers
mean_positions = [1, 2];
plot(mean_positions(1), stats.aversive.mean, 'ro', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'r', 'LineWidth', 2);
plot(mean_positions(2), stats.reward.mean, 'go', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'g', 'LineWidth', 2);

hold off;

ylabel('Breathing Rate (Hz)', 'FontSize', 11);
title('Breathing Rate Distributions', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 10);

% Add statistics text
text(0.5, 0.05, sprintf('Mean difference: %.2f Hz\nCohen''s d: %.3f\np = %.4f', ...
    stats.aversive.mean - stats.reward.mean, stats.cohens_d, p_value), ...
    'Units', 'normalized', 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold', ...
    'BackgroundColor', 'w');

% Subplot 2: Kernel density estimation
subplot(1, 2, 2);
hold on;

% Compute kernel density
[f_aver_kde, xi_aver] = ksdensity(breathing_aversive, 'Bandwidth', 0.3);
[f_rew_kde, xi_rew] = ksdensity(breathing_reward, 'Bandwidth', 0.3);

% Plot densities
area(xi_aver, f_aver_kde, 'FaceColor', [1, 0.6, 0.6], 'EdgeColor', [0.8, 0, 0], ...
    'LineWidth', 2, 'FaceAlpha', 0.5);
area(xi_rew, f_rew_kde, 'FaceColor', [0.6, 1, 0.6], 'EdgeColor', [0, 0.6, 0], ...
    'LineWidth', 2, 'FaceAlpha', 0.5);

% Add vertical lines for means
ymax = max([f_aver_kde, f_rew_kde]);
plot([stats.aversive.mean, stats.aversive.mean], [0, ymax], 'r--', 'LineWidth', 2);
plot([stats.reward.mean, stats.reward.mean], [0, ymax], 'g--', 'LineWidth', 2);

hold off;

xlabel('Breathing Rate (Hz)', 'FontSize', 11);
ylabel('Density', 'FontSize', 11);
title('Kernel Density Estimation', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward', 'Aversive mean', 'Reward mean'}, ...
    'Location', 'northeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);
xlim([0, 15]);

sgtitle('Breathing Rate Distribution Summary', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Box plot and density comparison created\n');

%% ========================================================================
%  SECTION 6: VISUALIZATION 3 - NARROW BAND FOCUS (5-10 Hz)
%  ========================================================================

fprintf('Creating narrow-band focus (5-10 Hz)...\n');

figure('Position', [150, 150, 1400, 500], 'Name', 'Breathing Rate: 5-10 Hz Focus');

% Filter data to 5-10 Hz range
breathing_aver_5to10 = breathing_aversive(breathing_aversive >= 5 & breathing_aversive <= 10);
breathing_rew_5to10 = breathing_reward(breathing_reward >= 5 & breathing_reward <= 10);

% Calculate percentage of samples in 5-10 Hz range
pct_aver_5to10 = (length(breathing_aver_5to10) / length(breathing_aversive)) * 100;
pct_rew_5to10 = (length(breathing_rew_5to10) / length(breathing_reward)) * 100;

% Subplot 1: Histogram zoomed to 5-10 Hz
subplot(1, 3, 1);
edges_narrow = 5:0.1:10;
hold on;

histogram(breathing_aver_5to10, edges_narrow, 'FaceColor', [1, 0.6, 0.6], ...
    'EdgeColor', [0.8, 0, 0], 'FaceAlpha', 0.6, 'Normalization', 'probability');
histogram(breathing_rew_5to10, edges_narrow, 'FaceColor', [0.6, 1, 0.6], ...
    'EdgeColor', [0, 0.6, 0], 'FaceAlpha', 0.6, 'Normalization', 'probability');

hold off;

xlabel('Breathing Rate (Hz)', 'FontSize', 11);
ylabel('Probability', 'FontSize', 11);
title('5-10 Hz Range Distribution', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Aversive', 'Reward'}, 'Location', 'northeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

% Subplot 2: Percentage bar chart
subplot(1, 3, 2);
bar([pct_aver_5to10, pct_rew_5to10], 'FaceColor', 'flat', ...
    'CData', [1, 0.6, 0.6; 0.6, 1, 0.6]);
xticklabels({'Aversive', 'Reward'});
ylabel('Percentage (%)', 'FontSize', 11);
title('% of Time in 5-10 Hz Range', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 10);

% Add percentage values on bars
hold on;
text(1, pct_aver_5to10 + 2, sprintf('%.1f%%', pct_aver_5to10), ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
text(2, pct_rew_5to10 + 2, sprintf('%.1f%%', pct_rew_5to10), ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
hold off;

% Subplot 3: Statistics table
subplot(1, 3, 3);
axis off;

% Create text display
stats_text = {
    'BREATHING RATE STATISTICS';
    '';
    'Overall Distribution:';
    sprintf('  Aversive: %.2f ± %.2f Hz (n=%d)', stats.aversive.mean, stats.aversive.std, stats.aversive.n);
    sprintf('  Reward:   %.2f ± %.2f Hz (n=%d)', stats.reward.mean, stats.reward.std, stats.reward.n);
    sprintf('  Difference: %.2f Hz', stats.aversive.mean - stats.reward.mean);
    '';
    sprintf('  p-value: %.4f', p_value);
    sprintf('  Cohen''s d: %.3f', stats.cohens_d);
    '';
    '5-10 Hz Range:';
    sprintf('  Aversive: %.1f%% of samples', pct_aver_5to10);
    sprintf('  Reward:   %.1f%% of samples', pct_rew_5to10);
    sprintf('  Difference: %.1f%%', pct_aver_5to10 - pct_rew_5to10);
};

text(0.1, 0.9, stats_text, 'VerticalAlignment', 'top', ...
    'FontSize', 10, 'FontName', 'FixedWidth');

sgtitle('Breathing Rate: Focus on 5-10 Hz (Coherence-Relevant Band)', ...
    'FontSize', 13, 'FontWeight', 'bold');

fprintf('✓ Narrow-band focus created\n');

%% ========================================================================
%  SECTION 7: PRINT SUMMARY
%  ========================================================================

fprintf('\n=== BREATHING RATE STATISTICS ===\n\n');

fprintf('AVERSIVE SESSIONS:\n');
fprintf('  Mean:   %.3f ± %.3f Hz\n', stats.aversive.mean, stats.aversive.std);
fprintf('  Median: %.3f Hz\n', stats.aversive.median);
fprintf('  Range:  [%.3f, %.3f] Hz\n', stats.aversive.min, stats.aversive.max);
fprintf('  Q1-Q3:  [%.3f, %.3f] Hz\n', stats.aversive.q25, stats.aversive.q75);
fprintf('  N:      %d samples\n\n', stats.aversive.n);

fprintf('REWARD SESSIONS:\n');
fprintf('  Mean:   %.3f ± %.3f Hz\n', stats.reward.mean, stats.reward.std);
fprintf('  Median: %.3f Hz\n', stats.reward.median);
fprintf('  Range:  [%.3f, %.3f] Hz\n', stats.reward.min, stats.reward.max);
fprintf('  Q1-Q3:  [%.3f, %.3f] Hz\n', stats.reward.q25, stats.reward.q75);
fprintf('  N:      %d samples\n\n', stats.reward.n);

fprintf('COMPARISON:\n');
fprintf('  Mean difference:  %.3f Hz (Aversive - Reward)\n', ...
    stats.aversive.mean - stats.reward.mean);
fprintf('  Median difference: %.3f Hz\n', ...
    stats.aversive.median - stats.reward.median);
fprintf('  Cohen''s d:        %.3f\n', stats.cohens_d);
fprintf('  Wilcoxon p-value: %.6f', p_value);
if p_value < 0.001
    fprintf(' ***\n');
elseif p_value < 0.01
    fprintf(' **\n');
elseif p_value < 0.05
    fprintf(' *\n');
else
    fprintf(' (n.s.)\n');
end

fprintf('\n5-10 Hz BAND (COHERENCE-RELEVANT):\n');
fprintf('  Aversive: %.1f%% of samples\n', pct_aver_5to10);
fprintf('  Reward:   %.1f%% of samples\n', pct_rew_5to10);
fprintf('  Difference: %.1f%%\n', pct_aver_5to10 - pct_rew_5to10);

fprintf('\n========================================\n');
fprintf('ANALYSIS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Figures created:\n');
fprintf('  1. Histogram comparison with cumulative distributions\n');
fprintf('  2. Box plot and kernel density estimation\n');
fprintf('  3. Narrow-band focus on 5-10 Hz range\n');
fprintf('========================================\n');
