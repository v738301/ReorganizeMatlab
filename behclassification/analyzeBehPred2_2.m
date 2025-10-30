%% Simplified Behavioral Analysis: Reward Port, Rearing, and Goal-Directed Movement
% Comparing 7 periods divided by 6 aversive noises + Before vs After first noise

clear all

%% Configuration
fprintf('=== SIMPLIFIED BEHAVIORAL ANALYSIS: 3 BEHAVIORS ACROSS 7 PERIODS ===\n');

%% Load data
fprintf('Loading data...\n');

try
%     coupling_data = load('RewardAversive_session_metrics_breathing_LFPCcouple(10-1).mat');
%     sessions = coupling_data.all_session_metrics;
%     fprintf('✓ Loaded data: %d sessions\n', length(sessions));

    coupling_data = load('27-Oct-2025_RewardAversive_session_metrics_breathing_LFPCcouple(10-1)');
    sessions = coupling_data.all_session_metrics;
    fprintf('✓ Loaded data: %d sessions\n', length(sessions));
catch ME
    fprintf('❌ Failed to load data: %s\n', ME.message);
    return;
end

%% Define animal colors (14 sessions = 7 animals, 2 sessions each)
n_animals = 12;
animal_colors = lines(n_animals);
session_to_animal = ceil((1:24)'/2);

%% Define behaviors to analyze
behavior_names = {'At Reward Port', 'Rearing', 'Goal-Directed Movement'};
behavior_columns = [4, 6, 7];  % Columns in behavioral_matrix
n_behaviors = 3;

%% Extract data for all 7 periods
fprintf('Extracting behavioral data for 7 periods...\n');
session_behavior_freq = cell(7, 1);  % Frequency (percentage of time)
session_behavior_dur = cell(7, 1);   % Average bout duration

for period = 1:7
    fprintf('Processing period %d...\n', period);
    
    n_sessions = length(sessions);
    freq_data = nan(n_sessions, n_behaviors);
    dur_data = nan(n_sessions, n_behaviors);
    
    for sess_idx = 1:n_sessions
        session = sessions{sess_idx};
        
        % Check required fields
        if ~isfield(session, 'behavioral_matrix_full') || ...
           ~isfield(session, 'all_aversive_time') || ...
           ~isfield(session, 'NeuralTime')
            continue;
        end
        
        behavioral_matrix = session.behavioral_matrix_full;
        aversive_times = session.all_aversive_time;
        neural_time = session.NeuralTime;
        
        % Determine period boundaries
        if period == 1
            % Period 1: From start to first aversive noise
            period_start = neural_time(1);
            period_end = aversive_times(1);
        elseif period <= length(aversive_times)
            % Periods 2-6: Between consecutive noises
            period_start = aversive_times(period - 1);
            period_end = aversive_times(period);
        else
            % Period 7: After last noise
            period_start = aversive_times(end);
            period_end = neural_time(end);
        end
        
        % Find indices for this period
        period_indices = find(neural_time >= period_start & neural_time <= period_end);
        
        if isempty(period_indices)
            continue;
        end
        
        % Extract behavioral data for this period
        period_behavior = behavioral_matrix(period_indices, behavior_columns);
        
        % Calculate frequency and duration for each behavior
        for beh_idx = 1:n_behaviors
            behavior_binary = period_behavior(:, beh_idx);
            
            % Frequency: percentage of time in this behavior
            freq_data(sess_idx, beh_idx) = (sum(behavior_binary) / length(behavior_binary)) * 100;
            
            % Duration: average bout length
            onset = find(diff([0; behavior_binary(:)]) == 1);
            offset = find(diff([behavior_binary(:); 0]) == -1);
            
            if ~isempty(onset) && ~isempty(offset)
                bout_durations = offset - onset + 1;  % in samples (1 Hz)
                dur_data(sess_idx, beh_idx) = mean(bout_durations);
            else
                dur_data(sess_idx, beh_idx) = 0;
            end
        end
    end
    
    session_behavior_freq{period} = freq_data;
    session_behavior_dur{period} = dur_data;
end

%% Combine periods 2-7 (all after first aversive noise)
fprintf('Combining periods 2-7 for Before vs After comparison...\n');

n_sessions = length(sessions);
session_behavior_freq_before = session_behavior_freq{1};
session_behavior_freq_after = nan(n_sessions, n_behaviors);
session_behavior_dur_before = session_behavior_dur{1};
session_behavior_dur_after = nan(n_sessions, n_behaviors);

for sess_idx = 1:n_sessions
    session = sessions{sess_idx};
    
    % Check required fields
    if ~isfield(session, 'behavioral_matrix_full') || ...
       ~isfield(session, 'all_aversive_time') || ...
       ~isfield(session, 'NeuralTime')
        continue;
    end
    
    behavioral_matrix = session.behavioral_matrix_full;
    aversive_times = session.all_aversive_time;
    neural_time = session.NeuralTime;
    
    % All after first aversive noise (periods 2-7 combined)
    period_start = aversive_times(1);
    period_end = neural_time(end);
    
    % Find indices
    period_indices = find(neural_time >= period_start & neural_time <= period_end);
    
    if isempty(period_indices)
        continue;
    end
    
    % Extract behavioral data
    period_behavior = behavioral_matrix(period_indices, behavior_columns);
    
    % Calculate frequency and duration for each behavior
    for beh_idx = 1:n_behaviors
        behavior_binary = period_behavior(:, beh_idx);
        
        % Frequency
        session_behavior_freq_after(sess_idx, beh_idx) = (sum(behavior_binary) / length(behavior_binary)) * 100;
        
        % Duration
        onset = find(diff([0; behavior_binary(:)]) == 1);
        offset = find(diff([behavior_binary(:); 0]) == -1);
        
        if ~isempty(onset) && ~isempty(offset)
            bout_durations = offset - onset + 1;
            session_behavior_dur_after(sess_idx, beh_idx) = mean(bout_durations);
        else
            session_behavior_dur_after(sess_idx, beh_idx) = 0;
        end
    end
end

%% Define colors
period_colors = [0.1 0.4 0.7;   % Period 1 - Blue
                 0.9 0.3 0.2;   % Period 2 - Red
                 0.9 0.6 0.1;   % Period 3 - Orange
                 0.5 0.8 0.2;   % Period 4 - Green
                 0.7 0.2 0.7;   % Period 5 - Purple
                 0.2 0.7 0.7;   % Period 6 - Cyan
                 0.9 0.5 0.6];  % Period 7 - Pink

before_after_colors = struct('before', [0.1 0.4 0.7], 'after', [0.9 0.3 0.2]);

%% FIGURE 1: Behavioral Frequency (Percentage) - 7 Periods
fprintf('Creating frequency comparison plot...\n');

fig1 = figure('Position', [100, 100, 1400, 500]);

for beh_idx = 1:n_behaviors
    subplot(1, 3, beh_idx);
    hold on;
    
    % Collect statistics
    all_medians = zeros(7, 1);
    all_q25 = zeros(7, 1);
    all_q75 = zeros(7, 1);
    
    for period = 1:7
        data = session_behavior_freq{period}(:, beh_idx);
        all_medians(period) = median(data, 'omitnan');
        all_q25(period) = prctile(data, 25);
        all_q75(period) = prctile(data, 75);
    end
    
    % Plot median with IQR
    err_low = all_medians - all_q25;
    err_high = all_q75 - all_medians;
    
    for period = 1:7
        errorbar(period, all_medians(period), err_low(period), err_high(period), ...
            'o', 'MarkerSize', 10, 'MarkerFaceColor', period_colors(period, :), ...
            'Color', period_colors(period, :), 'LineWidth', 2.5);
    end
    
    % Add individual session points with animal colors
    for period = 1:7
        data = session_behavior_freq{period}(:, beh_idx);
        
        for sess_idx = 1:length(data)
            if ~isnan(data(sess_idx))
                animal_id = session_to_animal(sess_idx);
                jitter = (rand - 0.5) * 0.15;
                
                scatter(period + jitter, data(sess_idx), 50, ...
                    animal_colors(animal_id, :), 'filled', 'MarkerFaceAlpha', 0.7);
            end
        end
    end
    
    % Add connecting lines for same animals
    for animal_id = 1:n_animals
        animal_sessions = find(session_to_animal == animal_id);
        
        for sess = animal_sessions'
            period_values = nan(7, 1);
            for period = 1:7
                period_values(period) = session_behavior_freq{period}(sess, beh_idx);
            end
            
            valid_periods = ~isnan(period_values);
            if sum(valid_periods) > 1
                plot(find(valid_periods), period_values(valid_periods), ...
                    '-', 'Color', [animal_colors(animal_id, :), 0.4], 'LineWidth', 1.5);
            end
        end
    end
    
    % Statistical tests: Period 1 vs others
    y_max = max(all_q75) + 2;
    for period = 2:7
        data_p1 = session_behavior_freq{1}(:, beh_idx);
        data_p = session_behavior_freq{period}(:, beh_idx);
        
        valid_pairs = ~isnan(data_p1) & ~isnan(data_p);
        if sum(valid_pairs) >= 3
            p_val = signrank(data_p1(valid_pairs), data_p(valid_pairs));
            
            if p_val < 0.001
                star_text = '***';
            elseif p_val < 0.01
                star_text = '**';
            elseif p_val < 0.05
                star_text = '*';
            else
                star_text = '';
            end
            
            if ~isempty(star_text)
                text(period, y_max, star_text, 'HorizontalAlignment', 'center', ...
                    'FontSize', 14, 'FontWeight', 'bold', 'Color', 'red');
            end
        end
    end
    
    ylabel('Frequency (%)', 'FontSize', 12);
    xlabel('Period', 'FontSize', 12);
    title(behavior_names{beh_idx}, 'FontSize', 13, 'FontWeight', 'bold');
    set(gca, 'XTick', 1:7, 'XTickLabel', {'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'});
    grid on;
    ylim([0, y_max + 3]);
    set(gca, 'FontSize', 11);
end

sgtitle('Behavioral Frequency Across 7 Periods (Median ± IQR)', ...
    'FontSize', 15, 'FontWeight', 'bold');

%% FIGURE 2: Behavioral Duration (seconds) - 7 Periods
fprintf('Creating duration comparison plot...\n');

fig2 = figure('Position', [100, 100, 1400, 500]);

for beh_idx = 1:n_behaviors
    subplot(1, 3, beh_idx);
    hold on;
    
    % Collect statistics
    all_medians = zeros(7, 1);
    all_q25 = zeros(7, 1);
    all_q75 = zeros(7, 1);
    
    for period = 1:7
        data = session_behavior_dur{period}(:, beh_idx);
        all_medians(period) = median(data, 'omitnan');
        all_q25(period) = prctile(data, 25);
        all_q75(period) = prctile(data, 75);
    end
    
    % Plot median with IQR
    err_low = all_medians - all_q25;
    err_high = all_q75 - all_medians;
    
    for period = 1:7
        errorbar(period, all_medians(period), err_low(period), err_high(period), ...
            'o', 'MarkerSize', 10, 'MarkerFaceColor', period_colors(period, :), ...
            'Color', period_colors(period, :), 'LineWidth', 2.5);
    end
    
    % Add individual session points with animal colors
    for period = 1:7
        data = session_behavior_dur{period}(:, beh_idx);
        
        for sess_idx = 1:length(data)
            if ~isnan(data(sess_idx))
                animal_id = session_to_animal(sess_idx);
                jitter = (rand - 0.5) * 0.15;
                
                scatter(period + jitter, data(sess_idx), 50, ...
                    animal_colors(animal_id, :), 'filled', 'MarkerFaceAlpha', 0.7);
            end
        end
    end
    
    % Add connecting lines for same animals
    for animal_id = 1:n_animals
        animal_sessions = find(session_to_animal == animal_id);
        
        for sess = animal_sessions'
            period_values = nan(7, 1);
            for period = 1:7
                period_values(period) = session_behavior_dur{period}(sess, beh_idx);
            end
            
            valid_periods = ~isnan(period_values);
            if sum(valid_periods) > 1
                plot(find(valid_periods), period_values(valid_periods), ...
                    '-', 'Color', [animal_colors(animal_id, :), 0.4], 'LineWidth', 1.5);
            end
        end
    end
    
    % Statistical tests: Period 1 vs others
    y_max = max(all_q75) + 0.5;
    for period = 2:7
        data_p1 = session_behavior_dur{1}(:, beh_idx);
        data_p = session_behavior_dur{period}(:, beh_idx);
        
        valid_pairs = ~isnan(data_p1) & ~isnan(data_p);
        if sum(valid_pairs) >= 3
            p_val = signrank(data_p1(valid_pairs), data_p(valid_pairs));
            
            if p_val < 0.001
                star_text = '***';
            elseif p_val < 0.01
                star_text = '**';
            elseif p_val < 0.05
                star_text = '*';
            else
                star_text = '';
            end
            
            if ~isempty(star_text)
                text(period, y_max, star_text, 'HorizontalAlignment', 'center', ...
                    'FontSize', 14, 'FontWeight', 'bold', 'Color', 'red');
            end
        end
    end
    
    ylabel('Duration (s)', 'FontSize', 12);
    xlabel('Period', 'FontSize', 12);
    title(behavior_names{beh_idx}, 'FontSize', 13, 'FontWeight', 'bold');
    set(gca, 'XTick', 1:7, 'XTickLabel', {'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'});
    grid on;
    ylim([0, y_max + 1]);
    set(gca, 'FontSize', 11);
end

sgtitle('Average Bout Duration Across 7 Periods (Median ± IQR)', ...
    'FontSize', 15, 'FontWeight', 'bold');

%% FIGURE 3: Before vs After First Aversive Noise - FREQUENCY
fprintf('Creating Before vs After comparison - FREQUENCY...\n');

fig3 = figure('Position', [100, 100, 1400, 500]);

for beh_idx = 1:n_behaviors
    subplot(1, 3, beh_idx);
    hold on;
    
    % Calculate medians and IQR
    med_before = median(session_behavior_freq_before(:, beh_idx), 'omitnan');
    med_after = median(session_behavior_freq_after(:, beh_idx), 'omitnan');
    
    q25_before = prctile(session_behavior_freq_before(:, beh_idx), 25);
    q75_before = prctile(session_behavior_freq_before(:, beh_idx), 75);
    q25_after = prctile(session_behavior_freq_after(:, beh_idx), 25);
    q75_after = prctile(session_behavior_freq_after(:, beh_idx), 75);
    
    err_low_before = med_before - q25_before;
    err_high_before = q75_before - med_before;
    err_low_after = med_after - q25_after;
    err_high_after = q75_after - med_after;
    
    bar_width = 0.35;
    x_pos = [1, 2];
    
    % Plot error bars
    errorbar(x_pos(1), med_before, err_low_before, err_high_before, ...
        'o', 'MarkerSize', 12, 'MarkerFaceColor', before_after_colors.before, ...
        'Color', before_after_colors.before, 'LineWidth', 3);
    errorbar(x_pos(2), med_after, err_low_after, err_high_after, ...
        'o', 'MarkerSize', 12, 'MarkerFaceColor', before_after_colors.after, ...
        'Color', before_after_colors.after, 'LineWidth', 3);
    
    % Add individual session points with animal colors
    for sess_idx = 1:length(session_behavior_freq_before(:, beh_idx))
        if ~isnan(session_behavior_freq_before(sess_idx, beh_idx))
            animal_id = session_to_animal(sess_idx);
            jitter = (rand - 0.5) * 0.1;
            
            scatter(x_pos(1) + jitter, session_behavior_freq_before(sess_idx, beh_idx), 60, ...
                animal_colors(animal_id, :), 'filled', 'MarkerFaceAlpha', 0.7, 'LineWidth', 1.5);
        end
    end
    
    for sess_idx = 1:length(session_behavior_freq_after(:, beh_idx))
        if ~isnan(session_behavior_freq_after(sess_idx, beh_idx))
            animal_id = session_to_animal(sess_idx);
            jitter = (rand - 0.5) * 0.1;
            
            scatter(x_pos(2) + jitter, session_behavior_freq_after(sess_idx, beh_idx), 60, ...
                animal_colors(animal_id, :), 'filled', 'MarkerFaceAlpha', 0.7, 'LineWidth', 1.5);
        end
    end
    
    % Add connecting lines for paired sessions
    for sess_idx = 1:n_sessions
        if ~isnan(session_behavior_freq_before(sess_idx, beh_idx)) && ...
           ~isnan(session_behavior_freq_after(sess_idx, beh_idx))
            animal_id = session_to_animal(sess_idx);
            plot([x_pos(1), x_pos(2)], ...
                [session_behavior_freq_before(sess_idx, beh_idx), ...
                 session_behavior_freq_after(sess_idx, beh_idx)], ...
                '-', 'Color', [animal_colors(animal_id, :), 0.4], 'LineWidth', 1.5);
        end
    end
    
    % Statistical test
    valid_pairs = ~isnan(session_behavior_freq_before(:, beh_idx)) & ...
                  ~isnan(session_behavior_freq_after(:, beh_idx));
    if sum(valid_pairs) >= 3
        p_val = signrank(session_behavior_freq_before(valid_pairs, beh_idx), ...
                        session_behavior_freq_after(valid_pairs, beh_idx));
        
        if p_val < 0.001
            star_text = '***';
        elseif p_val < 0.01
            star_text = '**';
        elseif p_val < 0.05
            star_text = '*';
        else
            star_text = 'ns';
        end
        
        % Add significance
        y_max = max([q75_before, q75_after]) + 2;
        line_y = y_max - 1;
        plot([x_pos(1), x_pos(2)], [line_y, line_y], 'k-', 'LineWidth', 1.5);
        text(mean(x_pos), y_max, star_text, 'HorizontalAlignment', 'center', ...
            'FontSize', 16, 'FontWeight', 'bold', 'Color', 'red');
    end
    
    ylabel('Frequency (%)', 'FontSize', 12);
    xlabel('Condition', 'FontSize', 12);
    title(behavior_names{beh_idx}, 'FontSize', 13, 'FontWeight', 'bold');
    set(gca, 'XTick', x_pos, 'XTickLabel', {'Before', 'After'});
    xlim([0.5, 2.5]);
    grid on;
    set(gca, 'FontSize', 11);
end

sgtitle('Behavioral Frequency: Before vs After First Aversive Noise (Median ± IQR)', ...
    'FontSize', 15, 'FontWeight', 'bold');

%% FIGURE 4: Before vs After First Aversive Noise - DURATION
fprintf('Creating Before vs After comparison - DURATION...\n');

fig4 = figure('Position', [100, 100, 1400, 500]);

for beh_idx = 1:n_behaviors
    subplot(1, 3, beh_idx);
    hold on;
    
    % Calculate medians and IQR
    med_before = median(session_behavior_dur_before(:, beh_idx), 'omitnan');
    med_after = median(session_behavior_dur_after(:, beh_idx), 'omitnan');
    
    q25_before = prctile(session_behavior_dur_before(:, beh_idx), 25);
    q75_before = prctile(session_behavior_dur_before(:, beh_idx), 75);
    q25_after = prctile(session_behavior_dur_after(:, beh_idx), 25);
    q75_after = prctile(session_behavior_dur_after(:, beh_idx), 75);
    
    err_low_before = med_before - q25_before;
    err_high_before = q75_before - med_before;
    err_low_after = med_after - q25_after;
    err_high_after = q75_after - med_after;
    
    bar_width = 0.35;
    x_pos = [1, 2];
    
    % Plot error bars
    errorbar(x_pos(1), med_before, err_low_before, err_high_before, ...
        'o', 'MarkerSize', 12, 'MarkerFaceColor', before_after_colors.before, ...
        'Color', before_after_colors.before, 'LineWidth', 3);
    errorbar(x_pos(2), med_after, err_low_after, err_high_after, ...
        'o', 'MarkerSize', 12, 'MarkerFaceColor', before_after_colors.after, ...
        'Color', before_after_colors.after, 'LineWidth', 3);
    
    % Add individual session points with animal colors
    for sess_idx = 1:length(session_behavior_dur_before(:, beh_idx))
        if ~isnan(session_behavior_dur_before(sess_idx, beh_idx))
            animal_id = session_to_animal(sess_idx);
            jitter = (rand - 0.5) * 0.1;
            
            scatter(x_pos(1) + jitter, session_behavior_dur_before(sess_idx, beh_idx), 60, ...
                animal_colors(animal_id, :), 'filled', 'MarkerFaceAlpha', 0.7, 'LineWidth', 1.5);
        end
    end
    
    for sess_idx = 1:length(session_behavior_dur_after(:, beh_idx))
        if ~isnan(session_behavior_dur_after(sess_idx, beh_idx))
            animal_id = session_to_animal(sess_idx);
            jitter = (rand - 0.5) * 0.1;
            
            scatter(x_pos(2) + jitter, session_behavior_dur_after(sess_idx, beh_idx), 60, ...
                animal_colors(animal_id, :), 'filled', 'MarkerFaceAlpha', 0.7, 'LineWidth', 1.5);
        end
    end
    
    % Add connecting lines for paired sessions
    for sess_idx = 1:n_sessions
        if ~isnan(session_behavior_dur_before(sess_idx, beh_idx)) && ...
           ~isnan(session_behavior_dur_after(sess_idx, beh_idx))
            animal_id = session_to_animal(sess_idx);
            plot([x_pos(1), x_pos(2)], ...
                [session_behavior_dur_before(sess_idx, beh_idx), ...
                 session_behavior_dur_after(sess_idx, beh_idx)], ...
                '-', 'Color', [animal_colors(animal_id, :), 0.4], 'LineWidth', 1.5);
        end
    end
    
    % Statistical test
    valid_pairs = ~isnan(session_behavior_dur_before(:, beh_idx)) & ...
                  ~isnan(session_behavior_dur_after(:, beh_idx));
    if sum(valid_pairs) >= 3
        p_val = signrank(session_behavior_dur_before(valid_pairs, beh_idx), ...
                        session_behavior_dur_after(valid_pairs, beh_idx));
        
        if p_val < 0.001
            star_text = '***';
        elseif p_val < 0.01
            star_text = '**';
        elseif p_val < 0.05
            star_text = '*';
        else
            star_text = 'ns';
        end
        
        % Add significance
        y_max = max([q75_before, q75_after]) + 0.5;
        line_y = y_max - 0.3;
        plot([x_pos(1), x_pos(2)], [line_y, line_y], 'k-', 'LineWidth', 1.5);
        text(mean(x_pos), y_max, star_text, 'HorizontalAlignment', 'center', ...
            'FontSize', 16, 'FontWeight', 'bold', 'Color', 'red');
    end
    
    ylabel('Duration (s)', 'FontSize', 12);
    xlabel('Condition', 'FontSize', 12);
    title(behavior_names{beh_idx}, 'FontSize', 13, 'FontWeight', 'bold');
    set(gca, 'XTick', x_pos, 'XTickLabel', {'Before', 'After'});
    xlim([0.5, 2.5]);
    grid on;
    set(gca, 'FontSize', 11);
end

sgtitle('Average Bout Duration: Before vs After First Aversive Noise (Median ± IQR)', ...
    'FontSize', 15, 'FontWeight', 'bold');

%% Print Summary Statistics
fprintf('\n=== FREQUENCY STATISTICS (7 Periods) ===\n');
for beh_idx = 1:n_behaviors
    fprintf('\n%s:\n', behavior_names{beh_idx});
    fprintf('Period\tMedian (%%)\tp-value (vs P1)\n');
    
    for period = 1:7
        med = median(session_behavior_freq{period}(:, beh_idx), 'omitnan');
        if period == 1
            fprintf('P%d\t%.2f\t\t-\n', period, med);
        else
            data_p1 = session_behavior_freq{1}(:, beh_idx);
            data_p = session_behavior_freq{period}(:, beh_idx);
            valid_pairs = ~isnan(data_p1) & ~isnan(data_p);
            if sum(valid_pairs) >= 3
                p_val = signrank(data_p1(valid_pairs), data_p(valid_pairs));
                fprintf('P%d\t%.2f\t\t%.4f\n', period, med, p_val);
            else
                fprintf('P%d\t%.2f\t\tN/A\n', period, med);
            end
        end
    end
end

fprintf('\n=== DURATION STATISTICS (7 Periods) ===\n');
for beh_idx = 1:n_behaviors
    fprintf('\n%s:\n', behavior_names{beh_idx});
    fprintf('Period\tMedian (s)\tp-value (vs P1)\n');
    
    for period = 1:7
        med = median(session_behavior_dur{period}(:, beh_idx), 'omitnan');
        if period == 1
            fprintf('P%d\t%.2f\t\t-\n', period, med);
        else
            data_p1 = session_behavior_dur{1}(:, beh_idx);
            data_p = session_behavior_dur{period}(:, beh_idx);
            valid_pairs = ~isnan(data_p1) & ~isnan(data_p);
            if sum(valid_pairs) >= 3
                p_val = signrank(data_p1(valid_pairs), data_p(valid_pairs));
                fprintf('P%d\t%.2f\t\t%.4f\n', period, med, p_val);
            else
                fprintf('P%d\t%.2f\t\tN/A\n', period, med);
            end
        end
    end
end

fprintf('\n=== BEFORE vs AFTER FIRST AVERSIVE NOISE ===\n');
fprintf('\nFREQUENCY:\n');
fprintf('Behavior\t\t\tBefore\tAfter\tp-value\n');
fprintf('--------------------------------------------------------\n');
for beh_idx = 1:n_behaviors
    med_before = median(session_behavior_freq_before(:, beh_idx), 'omitnan');
    med_after = median(session_behavior_freq_after(:, beh_idx), 'omitnan');
    
    valid_pairs = ~isnan(session_behavior_freq_before(:, beh_idx)) & ...
                  ~isnan(session_behavior_freq_after(:, beh_idx));
    if sum(valid_pairs) >= 3
        p_val = signrank(session_behavior_freq_before(valid_pairs, beh_idx), ...
                        session_behavior_freq_after(valid_pairs, beh_idx));
        fprintf('%s\t\t%.2f\t%.2f\t%.4f\n', behavior_names{beh_idx}, med_before, med_after, p_val);
    else
        fprintf('%s\t\t%.2f\t%.2f\tN/A\n', behavior_names{beh_idx}, med_before, med_after);
    end
end

fprintf('\nDURATION:\n');
fprintf('Behavior\t\t\tBefore\tAfter\tp-value\n');
fprintf('--------------------------------------------------------\n');
for beh_idx = 1:n_behaviors
    med_before = median(session_behavior_dur_before(:, beh_idx), 'omitnan');
    med_after = median(session_behavior_dur_after(:, beh_idx), 'omitnan');
    
    valid_pairs = ~isnan(session_behavior_dur_before(:, beh_idx)) & ...
                  ~isnan(session_behavior_dur_after(:, beh_idx));
    if sum(valid_pairs) >= 3
        p_val = signrank(session_behavior_dur_before(valid_pairs, beh_idx), ...
                        session_behavior_dur_after(valid_pairs, beh_idx));
        fprintf('%s\t\t%.2f\t%.2f\t%.4f\n', behavior_names{beh_idx}, med_before, med_after, p_val);
    else
        fprintf('%s\t\t%.2f\t%.2f\tN/A\n', behavior_names{beh_idx}, med_before, med_after);
    end
end

fprintf('\n✓ Analysis complete!\n');