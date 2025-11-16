%% Session-Level Variance Analysis: Before vs After First Noise
% This script properly accounts for inter-session variance and implements
% rigorous CDF comparison methods with confidence bands

clear all

%% Configuration
fprintf('=== SESSION-LEVEL VARIANCE ANALYSIS: BEFORE vs AFTER FIRST NOISE ===\n');

% Define analysis parameters
confidence_threshold = 0.8;
confidence_threshold_dominant = 0.3;
alpha_here = 0.05; % For 95% confidence bands
n_bootstrap = 1000;

%% Load data
fprintf('Loading data for before and after conditions...\n');

% Configuration
prediction_folder = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/BehaviorPrediction';
spike_folder = '/Volumes/ExpansionBackUp/Data/Struct_spike';

% Load sorting parameters
[T_sorted] = loadSortingParameters();

try
    % Select spike files
    [allfiles_aversive, ~, ~, ~] = selectFilesWithAnimalIDFiltering(spike_folder, 999, '2025*RewardAversive*.mat');

    % Calculate behavioral matrices from spike files
    sessions_before = loadSessionMetricsFromSpikeFiles(allfiles_aversive, T_sorted);

    % Load predictions
    prediction_sessions = loadBehaviorPredictionsFromSpikeFiles(allfiles_aversive, prediction_folder);

    fprintf('✓ Loaded data: %d sessions\n', length(sessions_before));
catch ME
    fprintf('❌ Failed to load data: %s\n', ME.message);
    return;
end

%% Extract data for before and after periods
fprintf('Extracting Before and After Aversive data...\n');
[data_before, ~] = extract_period_data(sessions_before, prediction_sessions, 'before_indices');
[data_after, ~] = extract_period_data(sessions_before, prediction_sessions, 'after_indices');

% Apply filtering
% data_before_filtered = apply_filters(data_before);
% data_after_filtered = apply_filters(data_after);

data_before_filtered = data_before;
data_after_filtered = data_after;

fprintf('Data points: Before=%d, After=%d\n', length(data_before_filtered), length(data_after_filtered));

%% Extract variables and organize by session
[speed_before, coupling_before, breathing_before, predictions_before] = extract_variables(data_before_filtered);
[speed_after, coupling_after, breathing_after, predictions_after] = extract_variables(data_after_filtered);

% Get session organization
session_ids_before = [data_before_filtered.session_id];
session_ids_after = [data_after_filtered.session_id];
unique_sessions_before = unique(session_ids_before);
unique_sessions_after = unique(session_ids_after);

fprintf('Unique sessions: Before=%d, After=%d\n', length(unique_sessions_before), length(unique_sessions_after));

%% Define variables
behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', 'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
n_behaviors = min(7, size(predictions_before, 2));
colors = struct('before', [0.1 0.4 0.7], 'after', [0.7 0.2 0.1]);
condition_names = {'Before First Noise', 'After First Noise'};

%% FIGURE 1: Session-Level Dominant Behavior Percentage Analysis (Single Plot)
fprintf('Creating session-level dominant behavior analysis...\n');

% Calculate session-level behavioral percentages
session_behavior_before = zeros(length(unique_sessions_before), n_behaviors);
session_behavior_after = zeros(length(unique_sessions_after), n_behaviors);

for i = 1:length(unique_sessions_before)
    sess_idx = session_ids_before == unique_sessions_before(i);
    sess_predictions = predictions_before(sess_idx, :);
    
    if ~isempty(sess_predictions)
        [max_confidence, dominant_behavior] = max(sess_predictions, [], 2);
        valid_predictions = max_confidence > confidence_threshold_dominant;
        
        if sum(valid_predictions) > 0
            dominant_behavior_valid = dominant_behavior(valid_predictions);
            for beh_idx = 1:n_behaviors
                behavior_count = sum(dominant_behavior_valid == beh_idx);
                session_behavior_before(i, beh_idx) = behavior_count / sum(valid_predictions) * 100;
            end
        end
    end
end

for i = 1:length(unique_sessions_after)
    sess_idx = session_ids_after == unique_sessions_after(i);
    sess_predictions = predictions_after(sess_idx, :);
    
    if ~isempty(sess_predictions)
        [max_confidence, dominant_behavior] = max(sess_predictions, [], 2);
        valid_predictions = max_confidence > confidence_threshold_dominant;
        
        if sum(valid_predictions) > 0
            dominant_behavior_valid = dominant_behavior(valid_predictions);
            for beh_idx = 1:n_behaviors
                behavior_count = sum(dominant_behavior_valid == beh_idx);
                session_behavior_after(i, beh_idx) = behavior_count / sum(valid_predictions) * 100;
            end
        end
    end
end

% Statistical analysis using proper session-level variance
fig = figure('Position', [100, 100, 1200, 600]);

% Calculate statistics
session_means_before = mean(session_behavior_before, 1, 'omitnan');
session_means_after = mean(session_behavior_after, 1, 'omitnan');
session_sems_before = std(session_behavior_before, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(session_behavior_before), 1));
session_sems_after = std(session_behavior_after, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(session_behavior_after), 1));

session_p_values = zeros(1, n_behaviors);
session_effect_sizes = zeros(1, n_behaviors);

% for beh_idx = 1:n_behaviors
%     before_vals = session_behavior_before(:, beh_idx);
%     after_vals = session_behavior_after(:, beh_idx);
%     
%     before_vals_clean = before_vals(~isnan(before_vals));
%     after_vals_clean = after_vals(~isnan(after_vals));
%     
%     if length(before_vals_clean) >= 3 && length(after_vals_clean) >= 3
%         [~, session_p_values(beh_idx)] = ttest2(after_vals_clean, before_vals_clean);
%         
%         pooled_std = sqrt(((length(before_vals_clean)-1)*var(before_vals_clean) + ...
%                           (length(after_vals_clean)-1)*var(after_vals_clean)) / ...
%                          (length(before_vals_clean) + length(after_vals_clean) - 2));
%         session_effect_sizes(beh_idx) = (mean(after_vals_clean) - mean(before_vals_clean)) / pooled_std;
%     else
%         session_p_values(beh_idx) = NaN;
%         session_effect_sizes(beh_idx) = NaN;
%     end
% end

for beh_idx = 1:n_behaviors
    before_vals = session_behavior_before(:, beh_idx);
    after_vals = session_behavior_after(:, beh_idx);
    
    % Find paired sessions (both before and after have valid data)
    valid_pairs = ~isnan(before_vals) & ~isnan(after_vals);
    
    before_vals_paired = before_vals(valid_pairs);
    after_vals_paired = after_vals(valid_pairs);
    
    if length(before_vals_paired) >= 3
        % Wilcoxon Signed-Rank Test (paired non-parametric test)
        [session_p_values(beh_idx), ~, stats] = signrank(before_vals_paired, after_vals_paired);
        
        % Calculate matched-pairs rank-biserial correlation as effect size
        n_pairs = length(before_vals_paired);
        
        % Calculate differences
        differences = after_vals_paired - before_vals_paired;
        abs_diffs = abs(differences);
        ranks = tiedrank(abs_diffs);
        
        % Sum of positive ranks (where after > before)
        W_plus = sum(ranks(differences > 0));
        % Total possible sum of ranks
        W_max = n_pairs * (n_pairs + 1) / 2;
        
        % Matched-pairs rank-biserial correlation
        % r = (W+ / W_max) * 2 - 1
        % Range: -1 (all negative) to +1 (all positive)
        session_effect_sizes(beh_idx) = (W_plus / W_max) * 2 - 1;
        
    else
        session_p_values(beh_idx) = NaN;
        session_effect_sizes(beh_idx) = NaN;
    end
end

% Main plot: Session-level behavioral percentages
x_pos = 1:n_behaviors;
bar_width = 0.35;

% Plot error bars with mean ± SEM
errorbar(x_pos - bar_width/2, session_means_before, session_sems_before, ...
    'o', 'MarkerSize', 8, 'MarkerFaceColor', colors.before, 'Color', colors.before, 'LineWidth', 2);
hold on;
errorbar(x_pos + bar_width/2, session_means_after, session_sems_after, ...
    'o', 'MarkerSize', 8, 'MarkerFaceColor', colors.after, 'Color', colors.after, 'LineWidth', 2);

% Add individual session points with jitter
for beh_idx = 1:n_behaviors
    before_vals = session_behavior_before(:, beh_idx);
    after_vals = session_behavior_after(:, beh_idx);
    
    before_vals_clean = before_vals(~isnan(before_vals));
    after_vals_clean = after_vals(~isnan(after_vals));
    
    jitter_before = (rand(size(before_vals_clean)) - 0.5) * 0.2;
    jitter_after = (rand(size(after_vals_clean)) - 0.5) * 0.2;
    
    scatter(beh_idx - bar_width/2 + jitter_before, before_vals_clean, 25, colors.before, 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(beh_idx + bar_width/2 + jitter_after, after_vals_clean, 25, colors.after, 'filled', 'MarkerFaceAlpha', 0.6);
end

% Add connecting lines between paired sessions
for beh_idx = 1:n_behaviors
    before_vals = session_behavior_before(:, beh_idx);
    after_vals = session_behavior_after(:, beh_idx);
    
    % Only connect sessions with data in both conditions
    valid_sessions = ~isnan(before_vals) & ~isnan(after_vals);
    
    if sum(valid_sessions) > 0
        before_vals_paired = before_vals(valid_sessions);
        after_vals_paired = after_vals(valid_sessions);
        
        for sess_i = 1:length(before_vals_paired)
            plot([beh_idx - bar_width/2, beh_idx + bar_width/2], ...
                 [before_vals_paired(sess_i), after_vals_paired(sess_i)], ...
                 'k-', 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5 0.7]);
        end
    end
end

% Add significance stars
y_max = max([session_means_before + session_sems_before; session_means_after + session_sems_after]);
for beh_idx = 1:n_behaviors
    if ~isnan(session_p_values(beh_idx))
        % Determine star marker based on p-value
        if session_p_values(beh_idx) < 0.001
            star_text = '***';
        elseif session_p_values(beh_idx) < 0.01
            star_text = '**';
        elseif session_p_values(beh_idx) < 0.05
            star_text = '*';
        else
            star_text = '';  % No star for non-significant
        end
        
        if ~isempty(star_text)
            % Position star above the higher error bar
            star_y = y_max(beh_idx) + 2;
            
            % Add horizontal line connecting the two conditions
            line_y = y_max(beh_idx) + 1;
            plot([beh_idx - bar_width/2, beh_idx + bar_width/2], [line_y, line_y], 'k-', 'LineWidth', 1);
            
            % Add star above the line
            text(beh_idx, star_y, star_text, 'HorizontalAlignment', 'center', ...
                'FontSize', 14, 'FontWeight', 'bold', 'Color', 'red');
        end
    end
end

ylabel('Dominant Behavior Percentage (%)');
xlabel('Behavior');
title('Session-Level Behavioral Percentages: Before vs After First Noise');
set(gca, 'XTick', x_pos, 'XTickLabel', behavior_names, 'XTickLabelRotation', 45);
legend(condition_names, 'Location', 'best', 'FontSize', 10);
grid on;

% Adjust y-axis to accommodate stars
ylim([0, max(y_max) + 20]);

%%
% output = '/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/APCCN2025/BehChageAfterNoise.svg';
% print(fig, output,'-painters','-dsvg')

%% FIGURE 2: Speed During High-Confidence Behaviors (PDF/CDF with confidence bands)
fprintf('Creating speed analysis during high-confidence behaviors...\n');

fig = figure('Position', [150, 150, 1800, 1400]);

% Organize speed data by session and behavior
speed_session_data_before = cell(length(unique_sessions_before), n_behaviors);
speed_session_data_after = cell(length(unique_sessions_after), n_behaviors);

for i = 1:length(unique_sessions_before)
    sess_idx = session_ids_before == unique_sessions_before(i);
    for beh_idx = 1:n_behaviors
        high_conf_idx = predictions_before(sess_idx, beh_idx) > confidence_threshold;
        if sum(high_conf_idx) > 0
            sess_speed = speed_before(sess_idx);
            speed_session_data_before{i, beh_idx} = sess_speed(high_conf_idx);
            % Remove outliers
            speed_session_data_before{i, beh_idx} = speed_session_data_before{i, beh_idx}(...
                speed_session_data_before{i, beh_idx} >= 0 & speed_session_data_before{i, beh_idx} <= 500);
        end
    end
end

for i = 1:length(unique_sessions_after)
    sess_idx = session_ids_after == unique_sessions_after(i);
    for beh_idx = 1:n_behaviors
        high_conf_idx = predictions_after(sess_idx, beh_idx) > confidence_threshold;
        if sum(high_conf_idx) > 0
            sess_speed = speed_after(sess_idx);
            speed_session_data_after{i, beh_idx} = sess_speed(high_conf_idx);
            % Remove outliers
            speed_session_data_after{i, beh_idx} = speed_session_data_after{i, beh_idx}(...
                speed_session_data_after{i, beh_idx} >= 0 & speed_session_data_after{i, beh_idx} <= 500);
        end
    end
end

% Create PDF/CDF plots for each behavior
for beh_idx = 1:n_behaviors
    % PDF subplot (top row)
    subplot(2, n_behaviors, beh_idx);
    
    % Pool data for PDFs
    all_speed_before = [];
    all_speed_after = [];
    for i = 1:size(speed_session_data_before, 1)
        if ~isempty(speed_session_data_before{i, beh_idx})
            all_speed_before = [all_speed_before; speed_session_data_before{i, beh_idx}];
        end
    end
    for i = 1:size(speed_session_data_after, 1)
        if ~isempty(speed_session_data_after{i, beh_idx})
            all_speed_after = [all_speed_after; speed_session_data_after{i, beh_idx}];
        end
    end
    
    if length(all_speed_before) >= 10 || length(all_speed_after) >= 10
        hold on;
        if length(all_speed_before) >= 10
            histogram(all_speed_before, 20, 'Normalization', 'pdf', 'FaceColor', colors.before, ...
                'EdgeColor', 'none', 'FaceAlpha', 0.6);
        end
        if length(all_speed_after) >= 10
            histogram(all_speed_after, 20, 'Normalization', 'pdf', 'FaceColor', colors.after, ...
                'EdgeColor', 'none', 'FaceAlpha', 0.6);
        end
        
        if length(all_speed_before) >= 10 && length(all_speed_after) >= 10
            y_lim = ylim;
            line([median(all_speed_before) median(all_speed_before)], y_lim, 'Color', colors.before, 'LineWidth', 2, 'LineStyle', '--');
            line([median(all_speed_after) median(all_speed_after)], y_lim, 'Color', colors.after, 'LineWidth', 2, 'LineStyle', '--');
        end
    else
        text(0.5, 0.5, 'Insufficient Data', 'HorizontalAlignment', 'center', 'Units', 'normalized');
    end
    
    xlabel('Speed (mm/s)');
    ylabel('Probability Density');
    title(sprintf('%s PDF', behavior_names{beh_idx}));
    grid on;
    
    if beh_idx == 1
        legend(condition_names, 'Location', 'best', 'FontSize', 8);
    end
    
    % CDF subplot with confidence bands (bottom row)
    subplot(2, n_behaviors, beh_idx + n_behaviors);
    
    % Calculate CDFs with confidence bands
    [x_before, cdf_before, cdf_before_lower, cdf_before_upper] = calculate_cdf_with_confidence(...
        speed_session_data_before(:, beh_idx), alpha_here, n_bootstrap);
    [x_after, cdf_after, cdf_after_lower, cdf_after_upper] = calculate_cdf_with_confidence(...
        speed_session_data_after(:, beh_idx), alpha_here, n_bootstrap);
    
    hold on;
    if ~isempty(x_before)
        % Plot CDF with confidence band
        fill([x_before, fliplr(x_before)], [cdf_before_lower, fliplr(cdf_before_upper)], ...
            colors.before, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(x_before, cdf_before, 'Color', colors.before, 'LineWidth', 3);
    end
    if ~isempty(x_after)
        % Plot CDF with confidence band
        fill([x_after, fliplr(x_after)], [cdf_after_lower, fliplr(cdf_after_upper)], ...
            colors.after, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(x_after, cdf_after, 'Color', colors.after, 'LineWidth', 3);
    end
    
    xlabel('Speed (mm/s)');
    ylabel('Cumulative Probability');
    title(sprintf('%s CDF', behavior_names{beh_idx}));
    ylim([0, 1]);
    grid on;
    
    % Statistical test if both conditions have data
    if ~isempty(all_speed_before) && ~isempty(all_speed_after) && ...
       length(all_speed_before) >= 10 && length(all_speed_after) >= 10
        [~, p_ks] = kstest2(all_speed_before, all_speed_after);
        
        % Effect size
        cohens_d = (median(all_speed_after) - median(all_speed_before)) / ...
                   sqrt((var(all_speed_before) + var(all_speed_after)) / 2);
        
        stats_text = sprintf('KS p=%.3f\nCohen''s d=%.2f\nn=%d,%d', ...
            p_ks, cohens_d, length(all_speed_before), length(all_speed_after));
        text(0.02, 0.98, stats_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'BackgroundColor', 'white', 'EdgeColor', 'black', 'FontSize', 7);
    end
end

sgtitle('Speed During High-Confidence Behaviors: PDF (Top) and CDF with 95% Confidence Bands (Bottom)', ...
    'FontSize', 14, 'FontWeight', 'bold');

%%
% output = '/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/APCCN2025/BehSpeedBeforeAfterNoise.svg';
% print(fig, output,'-painters','-dsvg')

%% Figur 3 Speed CDF Summary Statistics (Session-Level Between-Group Analysis)
fprintf('Creating speed CDF summary statistics with session-level between-group analysis...\n');

figure('Position', [200, 200, 1200, 600]);

speed_session_medians_before = zeros(length(unique_sessions_before), n_behaviors);
speed_session_medians_after = zeros(length(unique_sessions_after), n_behaviors);
speed_between_p_values = zeros(1, n_behaviors);
speed_between_effect_sizes = zeros(1, n_behaviors);
speed_means_before = zeros(1, n_behaviors);
speed_means_after = zeros(1, n_behaviors);
speed_sems_before = zeros(1, n_behaviors);
speed_sems_after = zeros(1, n_behaviors);

% Calculate session-level medians
for sess_idx = 1:length(unique_sessions_before)
    for beh_idx = 1:n_behaviors
        if ~isempty(speed_session_data_before{sess_idx, beh_idx})
            speed_session_medians_before(sess_idx, beh_idx) = median(speed_session_data_before{sess_idx, beh_idx});
        else
            speed_session_medians_before(sess_idx, beh_idx) = NaN;
        end
    end
end

for sess_idx = 1:length(unique_sessions_after)
    for beh_idx = 1:n_behaviors
        if ~isempty(speed_session_data_after{sess_idx, beh_idx})
            speed_session_medians_after(sess_idx, beh_idx) = median(speed_session_data_after{sess_idx, beh_idx});
        else
            speed_session_medians_after(sess_idx, beh_idx) = NaN;
        end
    end
end

% Between-group analysis for each behavior
for beh_idx = 1:n_behaviors
    % Get session medians for this behavior
    before_medians = speed_session_medians_before(:, beh_idx);
    after_medians = speed_session_medians_after(:, beh_idx);
    
    % Remove NaN values
    before_medians_clean = before_medians(~isnan(before_medians));
    after_medians_clean = after_medians(~isnan(after_medians));
    
    if length(before_medians_clean) >= 3 && length(after_medians_clean) >= 3
        % Between-group t-test (unpaired)
        [~, speed_between_p_values(beh_idx)] = ttest2(after_medians_clean, before_medians_clean);
        
        % Calculate group means and SEMs
        speed_means_before(beh_idx) = mean(before_medians_clean);
        speed_means_after(beh_idx) = mean(after_medians_clean);
        speed_sems_before(beh_idx) = std(before_medians_clean) / sqrt(length(before_medians_clean));
        speed_sems_after(beh_idx) = std(after_medians_clean) / sqrt(length(after_medians_clean));
        
        % Effect size (Cohen's d for between groups)
        pooled_std = sqrt(((length(before_medians_clean)-1)*var(before_medians_clean) + ...
                          (length(after_medians_clean)-1)*var(after_medians_clean)) / ...
                         (length(before_medians_clean) + length(after_medians_clean) - 2));
        speed_between_effect_sizes(beh_idx) = (speed_means_after(beh_idx) - speed_means_before(beh_idx)) / pooled_std;
    else
        speed_between_p_values(beh_idx) = NaN;
        speed_between_effect_sizes(beh_idx) = NaN;
        speed_means_before(beh_idx) = NaN;
        speed_means_after(beh_idx) = NaN;
        speed_sems_before(beh_idx) = NaN;
        speed_sems_after(beh_idx) = NaN;
    end
end

% Main plot: Session-level means with error bars and individual points
x_pos = 1:n_behaviors;
bar_width = 0.35;

% Plot error bars with mean ± SEM
errorbar(x_pos - bar_width/2, speed_means_before, speed_sems_before, ...
    'o', 'MarkerSize', 8, 'MarkerFaceColor', colors.before, 'Color', colors.before, 'LineWidth', 2);
hold on;
errorbar(x_pos + bar_width/2, speed_means_after, speed_sems_after, ...
    'o', 'MarkerSize', 8, 'MarkerFaceColor', colors.after, 'Color', colors.after, 'LineWidth', 2);

% Add individual session points with jitter
for beh_idx = 1:n_behaviors
    before_medians = speed_session_medians_before(:, beh_idx);
    after_medians = speed_session_medians_after(:, beh_idx);
    
    before_medians_clean = before_medians(~isnan(before_medians));
    after_medians_clean = after_medians(~isnan(after_medians));
    
    jitter_before = (rand(size(before_medians_clean)) - 0.5) * 0.2;
    jitter_after = (rand(size(after_medians_clean)) - 0.5) * 0.2;
    
    scatter(beh_idx - bar_width/2 + jitter_before, before_medians_clean, 25, colors.before, 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(beh_idx + bar_width/2 + jitter_after, after_medians_clean, 25, colors.after, 'filled', 'MarkerFaceAlpha', 0.6);
end

% Add connecting lines between paired sessions
for beh_idx = 1:n_behaviors
    before_vals = speed_session_medians_before(:, beh_idx);
    after_vals = speed_session_medians_after(:, beh_idx);
    
    % Only connect sessions with data in both conditions
    valid_sessions = ~isnan(before_vals) & ~isnan(after_vals);
    
    if sum(valid_sessions) > 0
        before_vals_paired = before_vals(valid_sessions);
        after_vals_paired = after_vals(valid_sessions);
        
        for sess_i = 1:length(before_vals_paired)
            plot([beh_idx - bar_width/2, beh_idx + bar_width/2], ...
                 [before_vals_paired(sess_i), after_vals_paired(sess_i)], ...
                 'k-', 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5 0.7]);
        end
    end
end

% Add significance stars
y_max = max([speed_means_before + speed_sems_before; speed_means_after + speed_sems_after]);
for beh_idx = 1:n_behaviors
    if ~isnan(speed_between_p_values(beh_idx))
        % Determine star marker based on p-value
        if speed_between_p_values(beh_idx) < 0.001
            star_text = '***';
        elseif speed_between_p_values(beh_idx) < 0.01
            star_text = '**';
        elseif speed_between_p_values(beh_idx) < 0.05
            star_text = '*';
        else
            star_text = '';  % No star for non-significant
        end
        
        if ~isempty(star_text)
            % Position star above the higher error bar
            star_y = y_max(beh_idx) + 2;
            
            % Add horizontal line connecting the two conditions
            line_y = y_max(beh_idx) + 1;
            plot([beh_idx - bar_width/2, beh_idx + bar_width/2], [line_y, line_y], 'k-', 'LineWidth', 1);
            
            % Add star above the line
            text(beh_idx, star_y, star_text, 'HorizontalAlignment', 'center', ...
                'FontSize', 14, 'FontWeight', 'bold', 'Color', 'red');
        end
    end
end
ylabel('Speed (mm/s)');
xlabel('Behavior');
title('Speed During High-Confidence Behaviors: Session-Level Analysis');
set(gca, 'XTick', x_pos, 'XTickLabel', behavior_names, 'XTickLabelRotation', 45);
legend(condition_names, 'Location', 'best', 'FontSize', 10);
grid on;

% Adjust y-axis to accommodate stars
ylim([0, max(y_max) + (max(y_max) * 0.15)]);

%% FIGURE 4: Breathing Rate During High-Confidence Behaviors (PDF/CDF)
fprintf('Creating breathing rate analysis during high-confidence behaviors...\n');

fig = figure('Position', [250, 250, 1800, 1400]);

% Organize breathing data by session and behavior
breathing_session_data_before = cell(length(unique_sessions_before), n_behaviors);
breathing_session_data_after = cell(length(unique_sessions_after), n_behaviors);

for i = 1:length(unique_sessions_before)
    sess_idx = session_ids_before == unique_sessions_before(i);
    for beh_idx = 1:n_behaviors
        high_conf_idx = predictions_before(sess_idx, beh_idx) > confidence_threshold;
        if sum(high_conf_idx) > 0
            sess_breathing = breathing_before(sess_idx);
            breathing_session_data_before{i, beh_idx} = sess_breathing(high_conf_idx);
            % Remove outliers
            breathing_session_data_before{i, beh_idx} = breathing_session_data_before{i, beh_idx}(...
                breathing_session_data_before{i, beh_idx} >= 1 & breathing_session_data_before{i, beh_idx} <= 20);
        end
    end
end

for i = 1:length(unique_sessions_after)
    sess_idx = session_ids_after == unique_sessions_after(i);
    for beh_idx = 1:n_behaviors
        high_conf_idx = predictions_after(sess_idx, beh_idx) > confidence_threshold;
        if sum(high_conf_idx) > 0
            sess_breathing = breathing_after(sess_idx);
            breathing_session_data_after{i, beh_idx} = sess_breathing(high_conf_idx);
            % Remove outliers
            breathing_session_data_after{i, beh_idx} = breathing_session_data_after{i, beh_idx}(...
                breathing_session_data_after{i, beh_idx} >= 1 & breathing_session_data_after{i, beh_idx} <= 20);
        end
    end
end

% Create PDF/CDF plots for each behavior
for beh_idx = 1:n_behaviors
    % PDF subplot (top row)
    subplot(2, n_behaviors, beh_idx);
    
    % Pool data for PDFs
    all_breathing_before = [];
    all_breathing_after = [];
    for i = 1:size(breathing_session_data_before, 1)
        if ~isempty(breathing_session_data_before{i, beh_idx})
            all_breathing_before = [all_breathing_before; breathing_session_data_before{i, beh_idx}];
        end
    end
    for i = 1:size(breathing_session_data_after, 1)
        if ~isempty(breathing_session_data_after{i, beh_idx})
            all_breathing_after = [all_breathing_after; breathing_session_data_after{i, beh_idx}];
        end
    end
    
    if length(all_breathing_before) >= 10 || length(all_breathing_after) >= 10
        hold on;
        if length(all_breathing_before) >= 10
            histogram(all_breathing_before, 15, 'Normalization', 'pdf', 'FaceColor', colors.before, ...
                'EdgeColor', 'none', 'FaceAlpha', 0.6);
        end
        if length(all_breathing_after) >= 10
            histogram(all_breathing_after, 15, 'Normalization', 'pdf', 'FaceColor', colors.after, ...
                'EdgeColor', 'none', 'FaceAlpha', 0.6);
        end
        
        if length(all_breathing_before) >= 10 && length(all_breathing_after) >= 10
            y_lim = ylim;
            line([median(all_breathing_before) median(all_breathing_before)], y_lim, 'Color', colors.before, 'LineWidth', 2, 'LineStyle', '--');
            line([median(all_breathing_after) median(all_breathing_after)], y_lim, 'Color', colors.after, 'LineWidth', 2, 'LineStyle', '--');
        end
    else
        text(0.5, 0.5, 'Insufficient Data', 'HorizontalAlignment', 'center', 'Units', 'normalized');
    end
    
    xlabel('Breathing Rate (Hz)');
    ylabel('Probability Density');
    title(sprintf('%s PDF', behavior_names{beh_idx}));
    grid on;
    
    if beh_idx == 1
        legend(condition_names, 'Location', 'best', 'FontSize', 8);
    end
    
    % CDF subplot with confidence bands (bottom row)
    subplot(2, n_behaviors, beh_idx + n_behaviors);
    
    % Calculate CDFs with confidence bands
    [x_before, cdf_before, cdf_before_lower, cdf_before_upper] = calculate_cdf_with_confidence(...
        breathing_session_data_before(:, beh_idx), alpha_here, n_bootstrap);
    [x_after, cdf_after, cdf_after_lower, cdf_after_upper] = calculate_cdf_with_confidence(...
        breathing_session_data_after(:, beh_idx), alpha_here, n_bootstrap);
    
    hold on;
    if ~isempty(x_before)
        fill([x_before, fliplr(x_before)], [cdf_before_lower, fliplr(cdf_before_upper)], ...
            colors.before, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(x_before, cdf_before, 'Color', colors.before, 'LineWidth', 3);
    end
    if ~isempty(x_after)
        fill([x_after, fliplr(x_after)], [cdf_after_lower, fliplr(cdf_after_upper)], ...
            colors.after, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(x_after, cdf_after, 'Color', colors.after, 'LineWidth', 3);
    end
    
    xlabel('Breathing Rate (Hz)');
    ylabel('Cumulative Probability');
    title(sprintf('%s CDF', behavior_names{beh_idx}));
    ylim([0, 1]);
    grid on;
    
    % Statistical test if both conditions have data
    if ~isempty(all_breathing_before) && ~isempty(all_breathing_after) && ...
       length(all_breathing_before) >= 10 && length(all_breathing_after) >= 10
        [~, p_ks] = kstest2(all_breathing_before, all_breathing_after);
        
        cohens_d = (median(all_breathing_after) - median(all_breathing_before)) / ...
                   sqrt((var(all_breathing_before) + var(all_breathing_after)) / 2);
        
        stats_text = sprintf('KS p=%.3f\nCohen''s d=%.2f\nn=%d,%d', ...
            p_ks, cohens_d, length(all_breathing_before), length(all_breathing_after));
        text(0.02, 0.98, stats_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'BackgroundColor', 'white', 'EdgeColor', 'black', 'FontSize', 7);
    end
end

sgtitle('Breathing Rate During High-Confidence Behaviors: PDF (Top) and CDF with 95% Confidence Bands (Bottom)', ...
    'FontSize', 14, 'FontWeight', 'bold');

%%
% output = '/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/APCCN2025/BehBreathingBeforeAfterNoise.svg';
% print(fig, output,'-painters','-dsvg')

%% Breathing Rate CDF Summary Statistics (KS Test Analysis)
fprintf('Creating breathing rate >5Hz CDF summary statistics with KS test analysis...\n');

figure('Position', [300, 300, 1200, 600]);

breathing_ks_stats = zeros(1, n_behaviors);
breathing_ks_p_values = zeros(1, n_behaviors);
breathing_median_differences = zeros(1, n_behaviors);
breathing_cohens_d = zeros(1, n_behaviors);
breathing_medians_before = zeros(1, n_behaviors);
breathing_medians_after = zeros(1, n_behaviors);
breathing_n_before = zeros(1, n_behaviors);
breathing_n_after = zeros(1, n_behaviors);

% Calculate KS statistics for each behavior
for beh_idx = 1:n_behaviors
    % Pool all data for this behavior
    all_breathing_before = [];
    all_breathing_after = [];
    
    for i = 1:size(breathing_session_data_before, 1)
        if ~isempty(breathing_session_data_before{i, beh_idx})
            all_breathing_before = [all_breathing_before; breathing_session_data_before{i, beh_idx}];
        end
    end
    for i = 1:size(breathing_session_data_after, 1)
        if ~isempty(breathing_session_data_after{i, beh_idx})
            all_breathing_after = [all_breathing_after; breathing_session_data_after{i, beh_idx}];
        end
    end
    
    if length(all_breathing_before) >= 10 && length(all_breathing_after) >= 10
        % Kolmogorov-Smirnov test
        [~, breathing_ks_p_values(beh_idx), breathing_ks_stats(beh_idx)] = kstest2(all_breathing_before, all_breathing_after);
        
        % Calculate descriptive statistics
        breathing_medians_before(beh_idx) = median(all_breathing_before);
        breathing_medians_after(beh_idx) = median(all_breathing_after);
        breathing_median_differences(beh_idx) = breathing_medians_after(beh_idx) - breathing_medians_before(beh_idx);
        breathing_n_before(beh_idx) = length(all_breathing_before);
        breathing_n_after(beh_idx) = length(all_breathing_after);
        
        % Effect size (Cohen's d using medians and pooled variance)
        breathing_cohens_d(beh_idx) = breathing_median_differences(beh_idx) / ...
            sqrt((var(all_breathing_before) + var(all_breathing_after)) / 2);
    else
        breathing_ks_stats(beh_idx) = NaN;
        breathing_ks_p_values(beh_idx) = NaN;
        breathing_median_differences(beh_idx) = NaN;
        breathing_cohens_d(beh_idx) = NaN;
        breathing_medians_before(beh_idx) = NaN;
        breathing_medians_after(beh_idx) = NaN;
        breathing_n_before(beh_idx) = NaN;
        breathing_n_after(beh_idx) = NaN;
    end
end

% Main plot: KS statistics with effect size visualization
x_pos = 1:n_behaviors;
bar_width = 0.6;

% Create bars colored by significance and effect size
bars = bar(x_pos, breathing_ks_stats, bar_width);

% Add text annotations with effect sizes and sample sizes
for i = 1:n_behaviors
    if ~isnan(breathing_ks_stats(i))
        % Add Cohen's d on top of bars
        y_pos = breathing_ks_stats(i) + max(breathing_ks_stats) * 0.02;
        text(i, y_pos, sprintf('d=%.2f', breathing_cohens_d(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
        
        % Add median difference as second line
        text(i, y_pos + max(breathing_ks_stats) * 0.03, sprintf('Δ=%.2f', breathing_median_differences(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', [0.3 0.3 0.3]);
    end
end

ylabel('KS Statistic (Distribution Difference)');
xlabel('Behavior');
title('Breathing Rate Distribution Differences: Kolmogorov-Smirnov Analysis');
set(gca, 'XTick', x_pos, 'XTickLabel', behavior_names, 'XTickLabelRotation', 45);
grid on;

% Adjust y-axis to accommodate annotations
ylim([min(breathing_ks_stats) - max(breathing_ks_stats) * 0.1, max(breathing_ks_stats) + max(breathing_ks_stats) * 0.15]);
%% FIGURE 6: Coupling During High-Confidence Behaviors (PDF/CDF)
fprintf('Creating coupling analysis during high-confidence behaviors...\n');

fig = figure('Position', [350, 350, 1800, 1400]);

% Organize coupling data by session and behavior
coupling_session_data_before = cell(length(unique_sessions_before), n_behaviors);
coupling_session_data_after = cell(length(unique_sessions_after), n_behaviors);

for i = 1:length(unique_sessions_before)
    sess_idx = session_ids_before == unique_sessions_before(i);
    for beh_idx = 1:n_behaviors
        high_conf_idx = predictions_before(sess_idx, beh_idx) > confidence_threshold;
        if sum(high_conf_idx) > 0
            sess_coupling = coupling_before(sess_idx);
            coupling_session_data_before{i, beh_idx} = sess_coupling(high_conf_idx);
            % Remove extreme outliers
            coupling_session_data_before{i, beh_idx} = coupling_session_data_before{i, beh_idx}(...
                abs(coupling_session_data_before{i, beh_idx}) < 5);
        end
    end
end

for i = 1:length(unique_sessions_after)
    sess_idx = session_ids_after == unique_sessions_after(i);
    for beh_idx = 1:n_behaviors
        high_conf_idx = predictions_after(sess_idx, beh_idx) > confidence_threshold;
        if sum(high_conf_idx) > 0
            sess_coupling = coupling_after(sess_idx);
            coupling_session_data_after{i, beh_idx} = sess_coupling(high_conf_idx);
            % Remove extreme outliers
            coupling_session_data_after{i, beh_idx} = coupling_session_data_after{i, beh_idx}(...
                abs(coupling_session_data_after{i, beh_idx}) < 5);
        end
    end
end

% Create PDF/CDF plots for each behavior
for beh_idx = 1:n_behaviors
    % PDF subplot (top row)
    subplot(2, n_behaviors, beh_idx);
    
    % Pool data for PDFs
    all_coupling_before = [];
    all_coupling_after = [];
    for i = 1:size(coupling_session_data_before, 1)
        if ~isempty(coupling_session_data_before{i, beh_idx})
            all_coupling_before = [all_coupling_before; coupling_session_data_before{i, beh_idx}];
        end
    end
    for i = 1:size(coupling_session_data_after, 1)
        if ~isempty(coupling_session_data_after{i, beh_idx})
            all_coupling_after = [all_coupling_after; coupling_session_data_after{i, beh_idx}];
        end
    end
    
    if length(all_coupling_before) >= 10 || length(all_coupling_after) >= 10
        hold on;
        if length(all_coupling_before) >= 10
            histogram(all_coupling_before, 20, 'Normalization', 'pdf', 'FaceColor', colors.before, ...
                'EdgeColor', 'none', 'FaceAlpha', 0.6);
        end
        if length(all_coupling_after) >= 10
            histogram(all_coupling_after, 20, 'Normalization', 'pdf', 'FaceColor', colors.after, ...
                'EdgeColor', 'none', 'FaceAlpha', 0.6);
        end
        
        % Add baseline reference
        y_lim = ylim;
        line([0 0], y_lim, 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'LineStyle', ':');
        
        if length(all_coupling_before) >= 10 && length(all_coupling_after) >= 10
            line([median(all_coupling_before) median(all_coupling_before)], y_lim, 'Color', colors.before, 'LineWidth', 2, 'LineStyle', '--');
            line([median(all_coupling_after) median(all_coupling_after)], y_lim, 'Color', colors.after, 'LineWidth', 2, 'LineStyle', '--');
        end
    else
        text(0.5, 0.5, 'Insufficient Data', 'HorizontalAlignment', 'center', 'Units', 'normalized');
    end
    
    xlabel('Coupling Strength (Z-score)');
    ylabel('Probability Density');
    title(sprintf('%s PDF', behavior_names{beh_idx}));
    grid on;
    
    if beh_idx == 1
        legend_items = [condition_names, {'Reward Baseline'}];
        legend(legend_items, 'Location', 'best', 'FontSize', 8);
    end
    
    % CDF subplot with confidence bands (bottom row)
    subplot(2, n_behaviors, beh_idx + n_behaviors);
    
    % Calculate CDFs with confidence bands
    [x_before, cdf_before, cdf_before_lower, cdf_before_upper] = calculate_cdf_with_confidence(...
        coupling_session_data_before(:, beh_idx), alpha_here, n_bootstrap);
    [x_after, cdf_after, cdf_after_lower, cdf_after_upper] = calculate_cdf_with_confidence(...
        coupling_session_data_after(:, beh_idx), alpha_here, n_bootstrap);
    
    hold on;
    if ~isempty(x_before)
        fill([x_before, fliplr(x_before)], [cdf_before_lower, fliplr(cdf_before_upper)], ...
            colors.before, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(x_before, cdf_before, 'Color', colors.before, 'LineWidth', 3);
    end
    if ~isempty(x_after)
        fill([x_after, fliplr(x_after)], [cdf_after_lower, fliplr(cdf_after_upper)], ...
            colors.after, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(x_after, cdf_after, 'Color', colors.after, 'LineWidth', 3);
    end
    
    % Add baseline reference
    line([0 0], [0, 1], 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'LineStyle', ':');
    
    xlabel('Coupling Strength (Z-score)');
    ylabel('Cumulative Probability');
    title(sprintf('%s CDF', behavior_names{beh_idx}));
    ylim([0, 1]);
    grid on;
    
    % Statistical test if both conditions have data
    if ~isempty(all_coupling_before) && ~isempty(all_coupling_after) && ...
       length(all_coupling_before) >= 10 && length(all_coupling_after) >= 10
        [~, p_ks] = kstest2(all_coupling_before, all_coupling_after);
        
        cohens_d = (median(all_coupling_after) - median(all_coupling_before)) / ...
                   sqrt((var(all_coupling_before) + var(all_coupling_after)) / 2);
        
        stats_text = sprintf('KS p=%.3f\nCohen''s d=%.2f\nn=%d,%d', ...
            p_ks, cohens_d, length(all_coupling_before), length(all_coupling_after));
        text(0.02, 0.98, stats_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'BackgroundColor', 'white', 'EdgeColor', 'black', 'FontSize', 7);
    end
end

sgtitle('Coupling Strength During High-Confidence Behaviors: PDF (Top) and CDF with 95% Confidence Bands (Bottom)', ...
    'FontSize', 14, 'FontWeight', 'bold');%% FIGURE 7: Coupling CDF Summary Statistics

fprintf('\n✓ Session-level variance analysis complete!\n');
fprintf('Figures created:\n');
fprintf('  1. Dominant behavior percentage analysis (session-level)\n');
fprintf('  2. Speed during behaviors PDF/CDF with confidence bands\n');
fprintf('  3. Speed CDF summary statistics\n');
fprintf('  [Additional figures for breathing and coupling to be added...]\n');

%%
% output = '/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/APCCN2025/BehCouplingBeforeAfterNoise.svg';
% print(fig, output,'-painters','-dsvg')

%% Coupling CDF Summary Statistics (KS Test Analysis)
fprintf('Creating coupling CDF summary statistics with KS test analysis...\n');

fig = figure('Position', [400, 400, 1200, 600]);

coupling_ks_stats = zeros(1, n_behaviors);
coupling_ks_p_values = zeros(1, n_behaviors);
coupling_median_differences = zeros(1, n_behaviors);
coupling_cohens_d = zeros(1, n_behaviors);
coupling_medians_before = zeros(1, n_behaviors);
coupling_medians_after = zeros(1, n_behaviors);
coupling_n_before = zeros(1, n_behaviors);
coupling_n_after = zeros(1, n_behaviors);

% Calculate KS statistics for each behavior
for beh_idx = 1:n_behaviors
    % Pool all data for this behavior
    all_coupling_before = [];
    all_coupling_after = [];
    
    for i = 1:size(coupling_session_data_before, 1)
        if ~isempty(coupling_session_data_before{i, beh_idx})
            all_coupling_before = [all_coupling_before; coupling_session_data_before{i, beh_idx}];
        end
    end
    for i = 1:size(coupling_session_data_after, 1)
        if ~isempty(coupling_session_data_after{i, beh_idx})
            all_coupling_after = [all_coupling_after; coupling_session_data_after{i, beh_idx}];
        end
    end
    
    if length(all_coupling_before) >= 10 && length(all_coupling_after) >= 10
        % Kolmogorov-Smirnov test
        [~, coupling_ks_p_values(beh_idx), coupling_ks_stats(beh_idx)] = kstest2(all_coupling_before, all_coupling_after);
        
        % Calculate descriptive statistics
        coupling_medians_before(beh_idx) = median(all_coupling_before);
        coupling_medians_after(beh_idx) = median(all_coupling_after);
        coupling_median_differences(beh_idx) = coupling_medians_after(beh_idx) - coupling_medians_before(beh_idx);
        coupling_n_before(beh_idx) = length(all_coupling_before);
        coupling_n_after(beh_idx) = length(all_coupling_after);
        
        % Effect size (Cohen's d using medians and pooled variance)
        coupling_cohens_d(beh_idx) = coupling_median_differences(beh_idx) / ...
            sqrt((var(all_coupling_before) + var(all_coupling_after)) / 2);
    else
        coupling_ks_stats(beh_idx) = NaN;
        coupling_ks_p_values(beh_idx) = NaN;
        coupling_median_differences(beh_idx) = NaN;
        coupling_cohens_d(beh_idx) = NaN;
        coupling_medians_before(beh_idx) = NaN;
        coupling_medians_after(beh_idx) = NaN;
        coupling_n_before(beh_idx) = NaN;
        coupling_n_after(beh_idx) = NaN;
    end
end

% Main plot: KS statistics with effect size visualization
x_pos = 1:n_behaviors;
bar_width = 0.6;

% Create bars colored by significance and effect size
bars = bar(x_pos, coupling_ks_stats, bar_width);

% Add text annotations with effect sizes and sample sizes
for i = 1:n_behaviors
    if ~isnan(coupling_ks_stats(i))
        % Add Cohen's d on top of bars
        y_pos = coupling_ks_stats(i) + max(coupling_ks_stats) * 0.02;
        text(i, y_pos, sprintf('d=%.2f', coupling_cohens_d(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
        % Add median difference as second line
        text(i, y_pos + max(coupling_ks_stats) * 0.03, sprintf('Δ=%.2f', coupling_median_differences(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', [0.3 0.3 0.3]);
    end
end

ylabel('KS Statistic (Distribution Difference)');
xlabel('Behavior');
title('Coupling Distribution Differences: Kolmogorov-Smirnov Analysis');
set(gca, 'XTick', x_pos, 'XTickLabel', behavior_names, 'XTickLabelRotation', 45);
grid on;

% Adjust y-axis to accommodate annotations
ylim([min(coupling_ks_stats) - max(coupling_ks_stats) * 0.1, max(coupling_ks_stats) + max(coupling_ks_stats) * 0.15]);

%%
% output = '/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/APCCN2025/BehCouplingBeforeAfterNoiseKSstats.svg';
% print(fig, output,'-painters','-dsvg')
%% FIGURE 6 B: Coupling Confidence Comparison (Low vs High Confidence)
fprintf('Creating coupling analysis: Low vs High confidence comparison...\n');

fig = figure('Position', [400, 400, 1800, 1400]);

% Pool both phases together
all_session_ids = [session_ids_before, session_ids_after];
all_predictions = [predictions_before; predictions_after];
all_coupling = [coupling_before; coupling_after];

% Get unique sessions from pooled data
unique_sessions_pooled = unique(all_session_ids);

% Organize coupling data by confidence level (low vs high) for each behavior
coupling_low_conf_data = cell(length(unique_sessions_pooled), n_behaviors);
coupling_high_conf_data = cell(length(unique_sessions_pooled), n_behaviors);

for i = 1:length(unique_sessions_pooled)
    sess_idx = all_session_ids == unique_sessions_pooled(i);
    for beh_idx = 1:n_behaviors
        % Get predictions for this session and behavior
        sess_predictions = all_predictions(sess_idx, beh_idx);
        sess_coupling = all_coupling(sess_idx);
        
        if ~isempty(sess_predictions) && ~isempty(sess_coupling)
            % Low confidence: predictions <= confidence_threshold
            low_conf_idx = sess_predictions <= confidence_threshold;
            % High confidence: predictions > confidence_threshold
            high_conf_idx = sess_predictions > confidence_threshold;
            
            if sum(low_conf_idx) > 0
                coupling_low_conf_data{i, beh_idx} = sess_coupling(low_conf_idx);
                % Remove extreme outliers
                coupling_low_conf_data{i, beh_idx} = coupling_low_conf_data{i, beh_idx}(...
                    abs(coupling_low_conf_data{i, beh_idx}) < 5);
            end
            
            if sum(high_conf_idx) > 0
                coupling_high_conf_data{i, beh_idx} = sess_coupling(high_conf_idx);
                % Remove extreme outliers
                coupling_high_conf_data{i, beh_idx} = coupling_high_conf_data{i, beh_idx}(...
                    abs(coupling_high_conf_data{i, beh_idx}) < 5);
            end
        end
    end
end

% Define colors for low vs high confidence
colors_conf.low = [0.3 0.3 0.8];    % Blue for low confidence
colors_conf.high = [0.8 0.3 0.3];   % Red for high confidence
condition_names_conf = {'Low Confidence', 'High Confidence'};

%% Create PDF/CDF plots for each behavior
for beh_idx = 1:n_behaviors
    % PDF subplot (top row)
    subplot(2, n_behaviors, beh_idx);
    
    % Pool data for PDFs
    all_coupling_low = [];
    all_coupling_high = [];
    for i = 1:size(coupling_low_conf_data, 1)
        if ~isempty(coupling_low_conf_data{i, beh_idx})
            all_coupling_low = [all_coupling_low; coupling_low_conf_data{i, beh_idx}];
        end
    end
    for i = 1:size(coupling_high_conf_data, 1)
        if ~isempty(coupling_high_conf_data{i, beh_idx})
            all_coupling_high = [all_coupling_high; coupling_high_conf_data{i, beh_idx}];
        end
    end
    
    if length(all_coupling_low) >= 10 || length(all_coupling_high) >= 10
        hold on;
        if length(all_coupling_low) >= 10
            histogram(all_coupling_low, 20, 'Normalization', 'pdf', 'FaceColor', colors_conf.low, ...
                'EdgeColor', 'none', 'FaceAlpha', 0.6);
        end
        if length(all_coupling_high) >= 10
            histogram(all_coupling_high, 20, 'Normalization', 'pdf', 'FaceColor', colors_conf.high, ...
                'EdgeColor', 'none', 'FaceAlpha', 0.6);
        end
        
        % Add baseline reference
        y_lim = ylim;
        line([0 0], y_lim, 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'LineStyle', ':');
        
        if length(all_coupling_low) >= 10 && length(all_coupling_high) >= 10
            line([median(all_coupling_low) median(all_coupling_low)], y_lim, 'Color', colors_conf.low, 'LineWidth', 2, 'LineStyle', '--');
            line([median(all_coupling_high) median(all_coupling_high)], y_lim, 'Color', colors_conf.high, 'LineWidth', 2, 'LineStyle', '--');
        end
    else
        text(0.5, 0.5, 'Insufficient Data', 'HorizontalAlignment', 'center', 'Units', 'normalized');
    end
    
    xlabel('Coupling Strength (Z-score)');
    ylabel('Probability Density');
    title(sprintf('%s PDF', behavior_names{beh_idx}));
    grid on;
    
    if beh_idx == 1
        legend_items = [condition_names_conf, {'Reward Baseline'}];
        legend(legend_items, 'Location', 'best', 'FontSize', 8);
    end
    
    % CDF subplot with confidence bands (bottom row)
    subplot(2, n_behaviors, beh_idx + n_behaviors);
    
    % Calculate CDFs with confidence bands
    [x_low, cdf_low, cdf_low_lower, cdf_low_upper] = calculate_cdf_with_confidence(...
        coupling_low_conf_data(:, beh_idx), alpha_here, n_bootstrap);
    [x_high, cdf_high, cdf_high_lower, cdf_high_upper] = calculate_cdf_with_confidence(...
        coupling_high_conf_data(:, beh_idx), alpha_here, n_bootstrap);
    
    hold on;
    if ~isempty(x_low)
        fill([x_low, fliplr(x_low)], [cdf_low_lower, fliplr(cdf_low_upper)], ...
            colors_conf.low, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(x_low, cdf_low, 'Color', colors_conf.low, 'LineWidth', 3);
    end
    if ~isempty(x_high)
        fill([x_high, fliplr(x_high)], [cdf_high_lower, fliplr(cdf_high_upper)], ...
            colors_conf.high, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(x_high, cdf_high, 'Color', colors_conf.high, 'LineWidth', 3);
    end
    
    % Add baseline reference
    line([0 0], [0, 1], 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'LineStyle', ':');
    
    xlabel('Coupling Strength (Z-score)');
    ylabel('Cumulative Probability');
    title(sprintf('%s CDF', behavior_names{beh_idx}));
    ylim([0, 1]);
    grid on;
    
    % Statistical test if both conditions have data
    if ~isempty(all_coupling_low) && ~isempty(all_coupling_high) && ...
       length(all_coupling_low) >= 10 && length(all_coupling_high) >= 10
        [~, p_ks] = kstest2(all_coupling_low, all_coupling_high);
        
        cohens_d = (median(all_coupling_high) - median(all_coupling_low)) / ...
                   sqrt((var(all_coupling_low) + var(all_coupling_high)) / 2);
        
        stats_text = sprintf('KS p=%.3f\nCohen''s d=%.2f\nn=%d,%d', ...
            p_ks, cohens_d, length(all_coupling_low), length(all_coupling_high));
        text(0.02, 0.98, stats_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'BackgroundColor', 'white', 'EdgeColor', 'black', 'FontSize', 7);
    end
end

sgtitle('Coupling Strength: Low vs High Confidence Comparison (Pooled Phases) - PDF (Top) and CDF with 95% Confidence Bands (Bottom)', ...
    'FontSize', 14, 'FontWeight', 'bold');

%%
% output = '/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/APCCN2025/BehCouplingConfidence.svg';
% print(fig, output,'-painters','-dsvg');

%% Helper Functions
function filtered_data = apply_filters(data)
    if isempty(data)
        filtered_data = [];
        return;
    end
    
    speed = [data.speed_median]';
    coupling_normalized = [data.coupling_median_normalized]';
    breathing = [data.breathing_median]';
    prediction_matrix = vertcat(data.prediction_scores);
    is_in_overlap = [data.is_in_temporal_overlap]';
    
    basic_valid = ~isnan(speed) & ~isnan(coupling_normalized) & ~isnan(breathing) & ...
                  all(~isnan(prediction_matrix), 2);
    overlap_valid = is_in_overlap;
    extreme_valid = coupling_normalized > -10 & coupling_normalized < 10 & ...
                    speed >= 0 & speed <= 1000 & ...
                    breathing >= 1 & breathing <= 20;
    
    valid_idx = basic_valid & overlap_valid & extreme_valid;
    filtered_data = data(valid_idx);
end

function [speed, coupling, breathing, predictions] = extract_variables(data)
    if isempty(data)
        speed = []; coupling = []; breathing = []; predictions = [];
        return;
    end
    speed = [data.speed_median]';
    coupling = [data.coupling_median_normalized]';
    breathing = [data.breathing_median]';
    predictions = vertcat(data.prediction_scores);
end

% function [extracted_data, baseline_info] = extract_period_data(sessions, prediction_sessions, period_field)
%     reward_baselines = struct();
%     
%     for i = 1:length(sessions)
%         session = sessions{i};
%         if ~isfield(session, 'behavioral_matrix_full') || ~isfield(session, 'coupling_results_multiband')
%             reward_baselines(i).valid = false;
%             continue;
%         end
%         
%         behavioral_matrix = session.behavioral_matrix_full;
%         coupling_data_session = session.coupling_results_multiband;
%         
%         if isempty(coupling_data_session) || isempty(coupling_data_session.summary)
%             reward_baselines(i).valid = false;
%             continue;
%         end
%         
%         coupling_time = coupling_data_session.summary.window_times;
%         coupling_MI = coupling_data_session.summary.all_MI_values(4, :); % 7Hz band
%         neural_time = session.NeuralTime;
%         
%         if size(behavioral_matrix, 2) >= 1
%             [reward_periods, ~, ~] = group_behavioral_states(behavioral_matrix);
%         else
%             reward_periods = true(size(neural_time));
%         end
%         
%         try
%             coupling_neural = interp1(coupling_time, coupling_MI, neural_time, 'linear', 'extrap');
%             reward_coupling = coupling_neural(reward_periods);
%             reward_coupling_clean = reward_coupling(abs(reward_coupling - mean(reward_coupling)) < 3 * std(reward_coupling));
%             
%             if length(reward_coupling_clean) > 10
%                 baseline_mean = mean(reward_coupling_clean);
%                 baseline_std = std(reward_coupling_clean);
%                 if baseline_std == 0, baseline_std = 0.001; end
%                 
%                 reward_baselines(i).mean = baseline_mean;
%                 reward_baselines(i).std = baseline_std;
%                 reward_baselines(i).valid = true;
%             else
%                 reward_baselines(i).valid = false;
%             end
%         catch
%             reward_baselines(i).valid = false;
%         end
%     end
%     
%     all_data = [];
%     
%     for i = 1:length(sessions)
%         session = sessions{i};
%         if ~isfield(session, 'behavioral_matrix_full') || ~isfield(session, 'coupling_results_multiband') || ...
%            ~isfield(session, 'TriggerMid') || ~reward_baselines(i).valid
%             continue;
%         end
%         
%         if ~isempty(period_field) && isfield(session, period_field)
%             period_indices = session.(period_field);
%             behavioral_matrix = session.behavioral_matrix_full(period_indices, :);
%             neural_time = session.NeuralTime(period_indices);
%             speed = session.Speed(period_indices);
%         else
%             behavioral_matrix = session.behavioral_matrix_full;
%             neural_time = session.NeuralTime;
%             speed = session.Speed;
%         end
%         
%         camera_time = session.TriggerMid;
%         coupling_data_session = session.coupling_results_multiband;
%         
%         if i <= length(prediction_sessions) && isfield(prediction_sessions(1), 'prediction_scores')
%             prediction_scores = prediction_sessions(i).prediction_scores;
%         else
%             continue;
%         end
%         
%         coupling_time = coupling_data_session.summary.window_times;
%         coupling_MI = coupling_data_session.summary.all_MI_values(4, :);
%         
%         try
%             speed_camera = interp1(neural_time, speed, camera_time, 'linear', 'extrap');
%             
%             if size(behavioral_matrix, 2) >= 8
%                 breathing_neural = behavioral_matrix(:, 8);
%                 valid_breathing = ~isnan(breathing_neural) & breathing_neural > 0 & breathing_neural <= 20;
%                 if sum(valid_breathing) > 10
%                     breathing_camera = interp1(neural_time(valid_breathing), breathing_neural(valid_breathing), ...
%                                               camera_time, 'linear', 'extrap');
%                     breathing_camera = max(1, min(20, breathing_camera));
%                 else
%                     continue;
%                 end
%             else
%                 continue;
%             end
%             
%             coupling_camera_raw = interp1(coupling_time, coupling_MI, camera_time, 'linear', 'extrap');
%             baseline_mean = reward_baselines(i).mean;
%             baseline_std = reward_baselines(i).std;
%             coupling_camera_normalized = (coupling_camera_raw - baseline_mean) / baseline_std;
%             
%         catch
%             continue;
%         end
%         
%         prediction_indices = 1:20:length(camera_time)+1;
%         n_predictions = min(length(prediction_indices), size(prediction_scores, 1));
%         
%         for p = 1:n_predictions-1
%             window_start = prediction_indices(p);
%             window_end = prediction_indices(p+1);
%             
%             if window_end <= window_start, continue; end
%             
%             data_point = struct();
%             data_point.session_id = i;
%             data_point.prediction_id = p;
%             data_point.camera_time = camera_time(prediction_indices(p));
%             
%             data_point.speed_median = median(speed_camera(window_start:window_end), 'omitnan');
%             data_point.breathing_median = median(breathing_camera(window_start:window_end), 'omitnan');
%             data_point.coupling_median_normalized = median(coupling_camera_normalized(window_start:window_end), 'omitnan');
%             data_point.coupling_median_raw = median(coupling_camera_raw(window_start:window_end), 'omitnan');
%             data_point.prediction_scores = prediction_scores(p, :);
%             data_point.reward_baseline_mean = baseline_mean;
%             data_point.reward_baseline_std = baseline_std;
%             
%             overlap_start = max([neural_time(1), camera_time(1), coupling_time(1)]);
%             overlap_end = min([neural_time(end), camera_time(end), coupling_time(end)]);
%             data_point.is_in_temporal_overlap = (data_point.camera_time >= overlap_start) && ...
%                                                (data_point.camera_time <= overlap_end);
%             
%             all_data = [all_data; data_point];
%         end
%     end
%     
%     extracted_data = all_data;
%     baseline_info = reward_baselines;
% end

function [extracted_data, baseline_info] = extract_period_data(sessions, prediction_sessions, period_field)
    n_sessions = length(sessions);
    reward_baselines = struct('mean', cell(n_sessions, 1), 'std', cell(n_sessions, 1), 'valid', cell(n_sessions, 1));
    
    %% First pass: Calculate baselines
    for i = 1:n_sessions
        fprintf("%d Session \n", i)
        session = sessions{i};
        reward_baselines(i).valid = false;
        
        if ~isfield(session, 'behavioral_matrix_full') || ~isfield(session, 'coupling_results_multiband')
            continue;
        end
        
        behavioral_matrix = session.behavioral_matrix_full;
        coupling_data_session = session.coupling_results_multiband;
        
        if isempty(coupling_data_session) || isempty(coupling_data_session.summary)
            continue;
        end
        
        coupling_time = coupling_data_session.summary.window_times;
%         coupling_MI = coupling_data_session.summary.all_MI_values(4, :); % 7Hz band
        coupling_MI = coupling_data_session.summary.all_MI_values; % 7Hz band
        neural_time = session.NeuralTime;
        
        if size(behavioral_matrix, 2) >= 1
            [reward_periods, ~, ~] = group_behavioral_states(behavioral_matrix);
        else
            reward_periods = true(size(neural_time));
        end
        
        try
            coupling_neural = interp1(coupling_time, coupling_MI, neural_time, 'linear', 'extrap');
            reward_coupling = coupling_neural(reward_periods);
            
            % Vectorized outlier removal
            mu = mean(reward_coupling);
            sigma = std(reward_coupling);
            reward_coupling_clean = reward_coupling(abs(reward_coupling - mu) < 3 * sigma);
            
            if length(reward_coupling_clean) > 10
                baseline_mean = mean(reward_coupling_clean);
                baseline_std = std(reward_coupling_clean);
                if baseline_std == 0, baseline_std = 0.001; end
                
                reward_baselines(i).mean = baseline_mean;
                reward_baselines(i).std = baseline_std;
                reward_baselines(i).valid = true;
            end
        catch
            % Already set to invalid
        end
    end
    
    %% Second pass: Extract data
    % Pre-allocate cell array for speed
    all_data_cell = cell(n_sessions, 1);
    
    for i = 1:n_sessions
        fprintf("%d Session \n", i)
        session = sessions{i};
        
        if ~reward_baselines(i).valid || ...
           ~isfield(session, 'behavioral_matrix_full') || ...
           ~isfield(session, 'coupling_results_multiband') || ...
           ~isfield(session, 'TriggerMid')
            continue;
        end
        
        % Extract period data
        if ~isempty(period_field) && isfield(session, period_field)
            period_indices = session.(period_field);
        end

        behavioral_matrix = session.behavioral_matrix_full;
        neural_time = session.NeuralTime;
        speed = session.Speed;
        camera_time = session.TriggerMid;
        coupling_data_session = session.coupling_results_multiband;
        
        % Get prediction scores
        if i <= length(prediction_sessions) && isfield(prediction_sessions(1), 'prediction_scores')
            prediction_scores = prediction_sessions(i).prediction_scores;
        else
            continue;
        end
        
        coupling_time = coupling_data_session.summary.window_times;
%         coupling_MI = coupling_data_session.summary.all_MI_values(4, :); % 7Hz band
        coupling_MI = coupling_data_session.summary.all_MI_values; % 7Hz band
        
        try
            % Interpolate all data at once
            speed_camera = interp1(neural_time, speed, camera_time, 'linear', 'extrap');
            
            % Handle breathing data
            if size(behavioral_matrix, 2) >= 8
                breathing_neural = behavioral_matrix(:, 8);
                breathing_camera = interp1(neural_time, breathing_neural, camera_time, 'linear', 'extrap');
            else
                continue;
            end
            
            % Interpolate coupling data
            coupling_camera_raw = interp1(coupling_time, coupling_MI, camera_time, 'linear', 'extrap');
            baseline_mean = reward_baselines(i).mean;
            baseline_std = reward_baselines(i).std;
            coupling_camera_normalized = (coupling_camera_raw - baseline_mean) / baseline_std;
            
        catch
            continue;
        end

        % Pre-calculate prediction windows
        prediction_indices = 1:20:length(camera_time)+1;
        
        % Pre-allocate struct array for this session
        session_data = struct(...
            'session_id', [], ...
            'prediction_id', [], ...
            'camera_time', [], ...
            'speed_median', [], ...
            'breathing_median', [], ...
            'coupling_median_normalized', [], ...
            'coupling_median_raw', [], ...
            'prediction_scores', [], ...
            'reward_baseline_mean', [], ...
            'reward_baseline_std', [], ...
            'is_in_temporal_overlap', []);
        
        session_data = repmat(session_data, size(prediction_scores, 1), 1);
        
        % Vectorized computation where possible
        valid_count = 0;
        for p = 1:length(prediction_indices)-1
            window_start = prediction_indices(p);
            window_end = prediction_indices(p+1)-1;
            
            if or(camera_time(window_start) <= neural_time(period_indices(1)),camera_time(window_start) >= neural_time(period_indices(end))), continue; end
            
            valid_count = valid_count + 1;
            
            % Extract window data
            speed_window = speed_camera(window_start:window_end);
            breathing_window = breathing_camera(window_start:window_end);
            coupling_norm_window = coupling_camera_normalized(window_start:window_end);
            coupling_raw_window = coupling_camera_raw(window_start:window_end);
            
            session_data(valid_count).session_id = i;
            session_data(valid_count).prediction_id = p;
            session_data(valid_count).camera_time = camera_time(prediction_indices(p));
            session_data(valid_count).speed_median = median(speed_window, 'omitnan');
            session_data(valid_count).breathing_median = median(breathing_window, 'omitnan');
            session_data(valid_count).coupling_median_normalized = median(coupling_norm_window, 'omitnan');
            session_data(valid_count).coupling_median_raw = median(coupling_raw_window, 'omitnan');
            session_data(valid_count).prediction_scores = prediction_scores(p, :);
            session_data(valid_count).reward_baseline_mean = baseline_mean;
            session_data(valid_count).reward_baseline_std = baseline_std;
        end
        
        % Trim to valid entries
        if valid_count > 0
            all_data_cell{i} = session_data(1:valid_count);
        end
    end
    
    %% Concatenate all data efficiently
    % Remove empty cells
    all_data_cell = all_data_cell(~cellfun(@isempty, all_data_cell));
    
    % Concatenate all at once
    if ~isempty(all_data_cell)
        extracted_data = vertcat(all_data_cell{:});
    else
        extracted_data = [];
    end
    
    baseline_info = reward_baselines;
end

function [x_eval, cdf_est, cdf_lower, cdf_upper] = calculate_cdf_with_confidence(session_data, alpha_here, n_bootstrap)
    % Pool all data from sessions
    all_data = [];
    for i = 1:length(session_data)
        if ~isempty(session_data{i})
            all_data = [all_data; session_data{i}(:)];
        end
    end
    
    if length(all_data) < 10
        x_eval = []; cdf_est = []; cdf_lower = []; cdf_upper = [];
        return;
    end
    
    % Remove NaN and outliers
    all_data = all_data(~isnan(all_data));
    
    % Create evaluation points
    x_eval = linspace(min(all_data), max(all_data), 100);
    
    % Calculate empirical CDF
    cdf_est = zeros(size(x_eval));
    for i = 1:length(x_eval)
        cdf_est(i) = sum(all_data <= x_eval(i)) / length(all_data);
    end
    
    % Bootstrap confidence bands - session-level resampling
    n_sessions = length(session_data);
    cdf_bootstrap = zeros(n_bootstrap, length(x_eval));
    
    for boot_i = 1:n_bootstrap
        % Resample sessions with replacement
        boot_sessions = randsample(n_sessions, n_sessions, true);
        boot_data = [];
        
        for sess_i = boot_sessions'
            if ~isempty(session_data{sess_i})
                boot_data = [boot_data; session_data{sess_i}(:)];
            end
        end
        
        boot_data = boot_data(~isnan(boot_data));
        
        if length(boot_data) >= 10
            for i = 1:length(x_eval)
                cdf_bootstrap(boot_i, i) = sum(boot_data <= x_eval(i)) / length(boot_data);
            end
        end
    end
    
    % Calculate confidence bands
    cdf_lower = prctile(cdf_bootstrap, 100 * alpha_here/2, 1);
    cdf_upper = prctile(cdf_bootstrap, 100 * (1 - alpha_here/2), 1);
end