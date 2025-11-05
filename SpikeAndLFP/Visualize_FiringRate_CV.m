%% ========================================================================
%  VISUALIZE FIRING RATE & CV RESULTS
%  Loads and visualizes FR/CV analysis results
%  Compares Aversive vs Reward across Periods and Behaviors
%  ========================================================================

clear all
close all

%% ========================================================================
%  SECTION 1: LOAD RESULTS
%  ========================================================================

fprintf('Loading FR/CV analysis results...\n');

% Find most recent results file
files = dir('FR_CV_analysis_results_*.mat');
if isempty(files)
    error('No FR_CV_analysis_results files found!');
end

% Sort by date and load most recent
[~, idx] = sort([files.datenum], 'descend');
load_filename = files(idx(1)).name;
fprintf('  Loading: %s\n', load_filename);

load(load_filename);

% Extract data
tbl = results.tbl_aggregated;  % Use aggregated data for visualization
config = results.config;

fprintf('✓ Data loaded\n');
fprintf('  Aggregated data points: %d\n', height(tbl));
fprintf('  Sessions: %d aversive, %d reward\n\n', results.n_aversive_sessions, results.n_reward_sessions);

%% ========================================================================
%  SECTION 2: STATISTICAL ANALYSIS - FR
%  ========================================================================

fprintf('=== STATISTICAL ANALYSIS: FIRING RATE ===\n');

% Remove NaN values for FR
tbl_FR = tbl(~isnan(tbl.FR), :);
fprintf('  Valid FR data points: %d\n', height(tbl_FR));

% Post-hoc comparisons for FR (without LME - just pairwise tests)
fprintf('Computing pairwise comparisons for FR...\n');

FR_comparison = struct();
FR_comparison.behavior = [];
FR_comparison.period = [];
FR_comparison.aversive_mean = [];
FR_comparison.reward_mean = [];
FR_comparison.difference = [];
FR_comparison.pvalue = [];

for b = 1:config.n_behaviors
    for p = 1:4
        % Get observed means
        aver_mask = tbl_FR.SessionType == 'Aversive' & ...
                    tbl_FR.Period == categorical(p) & ...
                    tbl_FR.Behavior == config.behavior_names{b};
        rew_mask = tbl_FR.SessionType == 'Reward' & ...
                   tbl_FR.Period == categorical(p) & ...
                   tbl_FR.Behavior == config.behavior_names{b};

        aver_mean = mean(tbl_FR.FR(aver_mask), 'omitnan');
        rew_mean = mean(tbl_FR.FR(rew_mask), 'omitnan');

        % Statistical test
        if sum(aver_mask) > 0 && sum(rew_mask) > 0
            try
                p_val = ranksum(tbl_FR.FR(aver_mask), tbl_FR.FR(rew_mask));
            catch
                p_val = NaN;
            end
        else
            p_val = NaN;
        end

        FR_comparison.behavior(end+1) = b;
        FR_comparison.period(end+1) = p;
        FR_comparison.aversive_mean(end+1) = aver_mean;
        FR_comparison.reward_mean(end+1) = rew_mean;
        FR_comparison.difference(end+1) = aver_mean - rew_mean;
        FR_comparison.pvalue(end+1) = p_val;
    end
end

% FDR correction
try
    FR_comparison.pvalue_fdr = mafdr(FR_comparison.pvalue, 'BHFDR', true);
catch
    FR_comparison.pvalue_fdr = min(FR_comparison.pvalue * length(FR_comparison.pvalue), 1);
end

% Reshape for plotting
FR_pvals_matrix = reshape(FR_comparison.pvalue_fdr, [4, config.n_behaviors])';

n_sig_FR = sum(FR_comparison.pvalue_fdr < 0.05, 'omitnan');
fprintf('  Significant comparisons (FDR q<0.05): %d/%d\n\n', n_sig_FR, sum(~isnan(FR_comparison.pvalue_fdr)));

%% ========================================================================
%  SECTION 3: STATISTICAL ANALYSIS - CV
%  ========================================================================

fprintf('=== STATISTICAL ANALYSIS: COEFFICIENT OF VARIATION ===\n');

% Remove NaN values for CV
tbl_CV = tbl(~isnan(tbl.CV), :);
fprintf('  Valid CV data points: %d\n', height(tbl_CV));

% Post-hoc comparisons for CV
fprintf('Computing pairwise comparisons for CV...\n');

CV_comparison = struct();
CV_comparison.behavior = [];
CV_comparison.period = [];
CV_comparison.aversive_mean = [];
CV_comparison.reward_mean = [];
CV_comparison.difference = [];
CV_comparison.pvalue = [];

for b = 1:config.n_behaviors
    for p = 1:4
        % Get observed means
        aver_mask = tbl_CV.SessionType == 'Aversive' & ...
                    tbl_CV.Period == categorical(p) & ...
                    tbl_CV.Behavior == config.behavior_names{b};
        rew_mask = tbl_CV.SessionType == 'Reward' & ...
                   tbl_CV.Period == categorical(p) & ...
                   tbl_CV.Behavior == config.behavior_names{b};

        aver_mean = mean(tbl_CV.CV(aver_mask), 'omitnan');
        rew_mean = mean(tbl_CV.CV(rew_mask), 'omitnan');

        % Statistical test
        if sum(aver_mask) > 0 && sum(rew_mask) > 0
            try
                p_val = ranksum(tbl_CV.CV(aver_mask), tbl_CV.CV(rew_mask));
            catch
                p_val = NaN;
            end
        else
            p_val = NaN;
        end

        CV_comparison.behavior(end+1) = b;
        CV_comparison.period(end+1) = p;
        CV_comparison.aversive_mean(end+1) = aver_mean;
        CV_comparison.reward_mean(end+1) = rew_mean;
        CV_comparison.difference(end+1) = aver_mean - rew_mean;
        CV_comparison.pvalue(end+1) = p_val;
    end
end

% FDR correction
try
    CV_comparison.pvalue_fdr = mafdr(CV_comparison.pvalue, 'BHFDR', true);
catch
    CV_comparison.pvalue_fdr = min(CV_comparison.pvalue * length(CV_comparison.pvalue), 1);
end

% Reshape for plotting
CV_pvals_matrix = reshape(CV_comparison.pvalue_fdr, [4, config.n_behaviors])';

n_sig_CV = sum(CV_comparison.pvalue_fdr < 0.05, 'omitnan');
fprintf('  Significant comparisons (FDR q<0.05): %d/%d\n\n', n_sig_CV, sum(~isnan(CV_comparison.pvalue_fdr)));

%% ========================================================================
%  SECTION 4: FIGURE 1 - FR ACROSS TIME BY BEHAVIOR
%  ========================================================================

fprintf('Creating Figure 1: FR across time by behavior...\n');

figure('Position', [50, 50, 1800, 1000], 'Name', 'Firing Rate Across Time');
set(gcf, 'Color', 'white');

color_aversive = [1, 0.6, 0.6];
color_reward = [0.6, 1, 0.6];

for b = 1:config.n_behaviors
    subplot(3, 3, b);
    hold on;

    % Extract data for this behavior
    behavior_mask = tbl_FR.Behavior == config.behavior_names{b};
    behavior_data = tbl_FR(behavior_mask, :);

    if isempty(behavior_data)
        title(config.behavior_names{b}, 'FontWeight', 'bold');
        text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
        continue;
    end

    % Plot individual units (thin transparent lines)
    aversive_mask = behavior_data.SessionType == 'Aversive';
    aversive_units = unique(behavior_data.Unit(aversive_mask));

    for u = 1:min(length(aversive_units), 50)  % Limit to 50 for visibility
        unit_mask = behavior_data.Unit == aversive_units(u) & aversive_mask;
        unit_data = behavior_data(unit_mask, :);

        if height(unit_data) >= 2
            [~, sort_idx] = sort(double(unit_data.Period));
            plot(double(unit_data.Period(sort_idx)), unit_data.FR(sort_idx), ...
                 '-', 'Color', color_aversive, 'LineWidth', 0.5, 'HandleVisibility', 'off');
        end
    end

    reward_mask = behavior_data.SessionType == 'Reward';
    reward_units = unique(behavior_data.Unit(reward_mask));

    for u = 1:min(length(reward_units), 50)
        unit_mask = behavior_data.Unit == reward_units(u) & reward_mask;
        unit_data = behavior_data(unit_mask, :);

        if height(unit_data) >= 2
            [~, sort_idx] = sort(double(unit_data.Period));
            plot(double(unit_data.Period(sort_idx)), unit_data.FR(sort_idx), ...
                 '-', 'Color', color_reward, 'LineWidth', 0.5, 'HandleVisibility', 'off');
        end
    end

    % Calculate and plot means
    mean_aver = zeros(4, 1);
    sem_aver = zeros(4, 1);
    mean_rew = zeros(4, 1);
    sem_rew = zeros(4, 1);

    for p = 1:4
        period_mask = double(behavior_data.Period) == p;

        aver_data = behavior_data.FR(period_mask & aversive_mask);
        if ~isempty(aver_data)
            mean_aver(p) = mean(aver_data, 'omitnan');
            sem_aver(p) = std(aver_data, 'omitnan') / sqrt(sum(~isnan(aver_data)));
        else
            mean_aver(p) = NaN;
            sem_aver(p) = NaN;
        end

        rew_data = behavior_data.FR(period_mask & reward_mask);
        if ~isempty(rew_data)
            mean_rew(p) = mean(rew_data, 'omitnan');
            sem_rew(p) = std(rew_data, 'omitnan') / sqrt(sum(~isnan(rew_data)));
        else
            mean_rew(p) = NaN;
            sem_rew(p) = NaN;
        end
    end

    % Plot mean ± SEM
    errorbar(1:4, mean_aver, sem_aver, 'o-', 'LineWidth', 3, 'MarkerSize', 10, ...
             'Color', [1,0,0], 'MarkerFaceColor', [1,0,0], 'DisplayName', 'Aversive');
    errorbar(1:4, mean_rew, sem_rew, 's-', 'LineWidth', 3, 'MarkerSize', 10, ...
             'Color', [0,0.6,0], 'MarkerFaceColor', [0,0.6,0], 'DisplayName', 'Reward');

    % Add significance stars
    y_max = max([mean_aver + sem_aver; mean_rew + sem_rew], [], 'omitnan');
    star_y = y_max * 1.15;

    for p = 1:4
        p_val = FR_pvals_matrix(b, p);
        if ~isnan(p_val) && p_val < 0.05
            if p_val < 0.001
                star_text = '***';
            elseif p_val < 0.01
                star_text = '**';
            else
                star_text = '*';
            end
            text(p, star_y, star_text, 'FontSize', 14, 'FontWeight', 'bold', ...
                 'Color', [0.8, 0, 0], 'HorizontalAlignment', 'center');
        end
    end

    % Title - red if any period is significant
    title_color = 'k';
    if any(FR_pvals_matrix(b, :) < 0.05)
        title_color = 'r';
    end
    title(config.behavior_names{b}, 'FontWeight', 'bold', 'Color', title_color);

    xlabel('Period');
    ylabel('Firing Rate (Hz)');
    xticks(1:4);
    xticklabels({'P1', 'P2', 'P3', 'P4'});
    grid on;

    if b == 1
        legend('Location', 'best');
    end
end

sgtitle({'Firing Rate Across Time: Aversive vs Reward', ...
         'Thin lines = individual units; Thick lines = mean ± SEM', ...
         'Stars: * q<0.05, ** q<0.01, *** q<0.001 (FDR-corrected)'}, ...
        'FontSize', 14, 'FontWeight', 'bold');

fprintf('✓ Figure 1 complete\n');

%% ========================================================================
%  SECTION 5: FIGURE 2 - CV ACROSS TIME BY BEHAVIOR
%  ========================================================================

fprintf('Creating Figure 2: CV across time by behavior...\n');

figure('Position', [100, 100, 1800, 1000], 'Name', 'CV Across Time');
set(gcf, 'Color', 'white');

for b = 1:config.n_behaviors
    subplot(3, 3, b);
    hold on;

    % Extract data for this behavior
    behavior_mask = tbl_CV.Behavior == config.behavior_names{b};
    behavior_data = tbl_CV(behavior_mask, :);

    if isempty(behavior_data)
        title(config.behavior_names{b}, 'FontWeight', 'bold');
        text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
        continue;
    end

    % Plot individual units (thin transparent lines)
    aversive_mask = behavior_data.SessionType == 'Aversive';
    aversive_units = unique(behavior_data.Unit(aversive_mask));

    for u = 1:min(length(aversive_units), 50)
        unit_mask = behavior_data.Unit == aversive_units(u) & aversive_mask;
        unit_data = behavior_data(unit_mask, :);

        if height(unit_data) >= 2
            [~, sort_idx] = sort(double(unit_data.Period));
            plot(double(unit_data.Period(sort_idx)), unit_data.CV(sort_idx), ...
                 '-', 'Color', color_aversive, 'LineWidth', 0.5, 'HandleVisibility', 'off');
        end
    end

    reward_mask = behavior_data.SessionType == 'Reward';
    reward_units = unique(behavior_data.Unit(reward_mask));

    for u = 1:min(length(reward_units), 50)
        unit_mask = behavior_data.Unit == reward_units(u) & reward_mask;
        unit_data = behavior_data(unit_mask, :);

        if height(unit_data) >= 2
            [~, sort_idx] = sort(double(unit_data.Period));
            plot(double(unit_data.Period(sort_idx)), unit_data.CV(sort_idx), ...
                 '-', 'Color', color_reward, 'LineWidth', 0.5, 'HandleVisibility', 'off');
        end
    end

    % Calculate and plot means
    mean_aver = zeros(4, 1);
    sem_aver = zeros(4, 1);
    mean_rew = zeros(4, 1);
    sem_rew = zeros(4, 1);

    for p = 1:4
        period_mask = double(behavior_data.Period) == p;

        aver_data = behavior_data.CV(period_mask & aversive_mask);
        if ~isempty(aver_data)
            mean_aver(p) = mean(aver_data, 'omitnan');
            sem_aver(p) = std(aver_data, 'omitnan') / sqrt(sum(~isnan(aver_data)));
        else
            mean_aver(p) = NaN;
            sem_aver(p) = NaN;
        end

        rew_data = behavior_data.CV(period_mask & reward_mask);
        if ~isempty(rew_data)
            mean_rew(p) = mean(rew_data, 'omitnan');
            sem_rew(p) = std(rew_data, 'omitnan') / sqrt(sum(~isnan(rew_data)));
        else
            mean_rew(p) = NaN;
            sem_rew(p) = NaN;
        end
    end

    % Plot mean ± SEM
    errorbar(1:4, mean_aver, sem_aver, 'o-', 'LineWidth', 3, 'MarkerSize', 10, ...
             'Color', [1,0,0], 'MarkerFaceColor', [1,0,0], 'DisplayName', 'Aversive');
    errorbar(1:4, mean_rew, sem_rew, 's-', 'LineWidth', 3, 'MarkerSize', 10, ...
             'Color', [0,0.6,0], 'MarkerFaceColor', [0,0.6,0], 'DisplayName', 'Reward');

    % Add significance stars
    y_max = max([mean_aver + sem_aver; mean_rew + sem_rew], [], 'omitnan');
    star_y = y_max * 1.15;

    for p = 1:4
        p_val = CV_pvals_matrix(b, p);
        if ~isnan(p_val) && p_val < 0.05
            if p_val < 0.001
                star_text = '***';
            elseif p_val < 0.01
                star_text = '**';
            else
                star_text = '*';
            end
            text(p, star_y, star_text, 'FontSize', 14, 'FontWeight', 'bold', ...
                 'Color', [0.8, 0, 0], 'HorizontalAlignment', 'center');
        end
    end

    % Title - red if any period is significant
    title_color = 'k';
    if any(CV_pvals_matrix(b, :) < 0.05)
        title_color = 'r';
    end
    title(config.behavior_names{b}, 'FontWeight', 'bold', 'Color', title_color);

    xlabel('Period');
    ylabel('Coefficient of Variation');
    xticks(1:4);
    xticklabels({'P1', 'P2', 'P3', 'P4'});
    grid on;

    if b == 1
        legend('Location', 'best');
    end
end

sgtitle({'Coefficient of Variation Across Time: Aversive vs Reward', ...
         'Thin lines = individual units; Thick lines = mean ± SEM', ...
         'Stars: * q<0.05, ** q<0.01, *** q<0.001 (FDR-corrected)'}, ...
        'FontSize', 14, 'FontWeight', 'bold');

fprintf('✓ Figure 2 complete\n');

%% ========================================================================
%  SECTION 6: FIGURE 3 - STATISTICAL SUMMARY HEATMAPS
%  ========================================================================

fprintf('Creating Figure 3: Statistical summary heatmaps...\n');

figure('Position', [150, 150, 1400, 600], 'Name', 'Statistical Summary');
set(gcf, 'Color', 'white');

% FR p-value heatmap
subplot(1, 2, 1);
% Convert p-values to -log10(p) for better visualization
log_pvals = -log10(FR_pvals_matrix);
log_pvals(isinf(log_pvals)) = 10;  % Cap at 10

imagesc(log_pvals);
colormap(hot);
colorbar;
set(gca, 'CLim', [0, 3]);  % 0 = p=1, 1.3 = p=0.05, 2 = p=0.01, 3 = p=0.001

set(gca, 'XTick', 1:4, 'XTickLabel', {'P1', 'P2', 'P3', 'P4'});
set(gca, 'YTick', 1:7, 'YTickLabel', config.behavior_names);
xlabel('Period');
ylabel('Behavior');
title('Firing Rate: -log_{10}(q-value)', 'FontWeight', 'bold');

% Add reference lines
hold on;
for b = 1:7
    for p = 1:4
        if FR_pvals_matrix(b, p) < 0.05
            text(p, b, '*', 'FontSize', 20, 'FontWeight', 'bold', ...
                 'Color', 'cyan', 'HorizontalAlignment', 'center');
        end
    end
end

% CV p-value heatmap
subplot(1, 2, 2);
log_pvals = -log10(CV_pvals_matrix);
log_pvals(isinf(log_pvals)) = 10;

imagesc(log_pvals);
colormap(hot);
colorbar;
set(gca, 'CLim', [0, 3]);

set(gca, 'XTick', 1:4, 'XTickLabel', {'P1', 'P2', 'P3', 'P4'});
set(gca, 'YTick', 1:7, 'YTickLabel', config.behavior_names);
xlabel('Period');
ylabel('Behavior');
title('Coefficient of Variation: -log_{10}(q-value)', 'FontWeight', 'bold');

% Add reference lines
hold on;
for b = 1:7
    for p = 1:4
        if CV_pvals_matrix(b, p) < 0.05
            text(p, b, '*', 'FontSize', 20, 'FontWeight', 'bold', ...
                 'Color', 'cyan', 'HorizontalAlignment', 'center');
        end
    end
end

sgtitle({'Statistical Significance: Aversive vs Reward', ...
         'Cyan * indicates q < 0.05 (FDR-corrected)', ...
         'Brighter colors = more significant'}, ...
        'FontSize', 14, 'FontWeight', 'bold');

fprintf('✓ Figure 3 complete\n');

%% ========================================================================
%  SECTION 7: PRINT SUMMARY STATISTICS
%  ========================================================================

fprintf('\n=== SUMMARY STATISTICS ===\n');

fprintf('\nFIRING RATE:\n');
fprintf('  Total comparisons: %d\n', length(FR_comparison.pvalue));
fprintf('  Significant (FDR q<0.05): %d (%.1f%%)\n', ...
        sum(FR_comparison.pvalue_fdr < 0.05, 'omitnan'), ...
        100 * sum(FR_comparison.pvalue_fdr < 0.05, 'omitnan') / sum(~isnan(FR_comparison.pvalue_fdr)));

% List significant comparisons
sig_idx = find(FR_comparison.pvalue_fdr < 0.05);
if ~isempty(sig_idx)
    fprintf('\n  Significant comparisons:\n');
    for i = 1:length(sig_idx)
        idx = sig_idx(i);
        fprintf('    %s P%d: Aver=%.2f Hz, Rew=%.2f Hz, Diff=%.2f Hz, q=%.4f\n', ...
                config.behavior_names{FR_comparison.behavior(idx)}, ...
                FR_comparison.period(idx), ...
                FR_comparison.aversive_mean(idx), ...
                FR_comparison.reward_mean(idx), ...
                FR_comparison.difference(idx), ...
                FR_comparison.pvalue_fdr(idx));
    end
end

fprintf('\nCOEFFICIENT OF VARIATION:\n');
fprintf('  Total comparisons: %d\n', length(CV_comparison.pvalue));
fprintf('  Significant (FDR q<0.05): %d (%.1f%%)\n', ...
        sum(CV_comparison.pvalue_fdr < 0.05, 'omitnan'), ...
        100 * sum(CV_comparison.pvalue_fdr < 0.05, 'omitnan') / sum(~isnan(CV_comparison.pvalue_fdr)));

% List significant comparisons
sig_idx = find(CV_comparison.pvalue_fdr < 0.05);
if ~isempty(sig_idx)
    fprintf('\n  Significant comparisons:\n');
    for i = 1:length(sig_idx)
        idx = sig_idx(i);
        fprintf('    %s P%d: Aver=%.2f, Rew=%.2f, Diff=%.2f, q=%.4f\n', ...
                config.behavior_names{CV_comparison.behavior(idx)}, ...
                CV_comparison.period(idx), ...
                CV_comparison.aversive_mean(idx), ...
                CV_comparison.reward_mean(idx), ...
                CV_comparison.difference(idx), ...
                CV_comparison.pvalue_fdr(idx));
    end
end

fprintf('\n=== VISUALIZATION COMPLETE ===\n');
