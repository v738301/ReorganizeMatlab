%% Coupling Strength Analysis: Time Periods × Behavior Types
% Analyzes breathing-LFP coupling for 7 behaviors across:
% - 7 time periods in aversive sessions (defined by 6 aversive noises)
% - 4 time periods in reward sessions (defined by 3 matched time points)
% Modified to create LME-ready tables

clear all
% close all

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('=== COUPLING STRENGTH BY PERIOD AND BEHAVIOR ===\n\n');

config = struct();
config.behavior_names = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', ...
                         'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
config.n_behaviors = 7;
config.confidence_threshold = 0.3;  % Minimum confidence for behavior classification

%% ========================================================================
%  SECTION 2: ANALYZE AVERSIVE SESSIONS - MODIFIED FOR LME
%  ========================================================================

fprintf('PART 1: AVERSIVE SESSIONS (7 periods)\n');
fprintf('======================================\n\n');

% Load aversive data
fprintf('Loading aversive session data...\n');
coupling_data = load('27-Oct-2025_RewardAversive_session_metrics_breathing_LFPCcouple(10-1)');
sessions_aversive = coupling_data.all_session_metrics;
pred_data = load('lstm_prediction_results_aversive_27-Oct-2025');
prediction_sessions_aversive = pred_data.final_results.session_predictions;
fprintf('✓ Loaded data: %d sessions\n\n', length(sessions_aversive));

%%
% Initialize storage for LME - MODIFIED
aversive_long_data = struct();
aversive_long_data.session_id = [];
aversive_long_data.period = [];
aversive_long_data.behavior = [];
aversive_long_data.coupling = [];
aversive_long_data.change = [];

% Also keep original format for visualization
aversive_results = struct();
aversive_results.coupling_values = cell(config.n_behaviors, 7);
for beh = 1:config.n_behaviors
    for period = 1:7
        aversive_results.coupling_values{beh, period} = [];
    end
end

% Process each aversive session
fprintf('Processing aversive sessions...\n');
n_valid_sessions_aversive = 0;
session_data_aversive = {};

for sess_idx = 1:length(sessions_aversive)
    session = sessions_aversive{sess_idx};
    
    % Check required fields
    if ~isfield(session, 'all_aversive_time') || ...
       ~isfield(session, 'coupling_results_multiband') || ...
       sess_idx > length(prediction_sessions_aversive)
        continue;
    end
    
    aversive_times = session.all_aversive_time;
    if length(aversive_times) < 6
        continue;
    end
    
    n_valid_sessions_aversive = n_valid_sessions_aversive + 1;
    session_data_aversive{n_valid_sessions_aversive} = nan(config.n_behaviors, 7);
    
    coupling = session.coupling_results_multiband.band_results{1}.MI_values;
    coupling_time = session.coupling_results_multiband.band_results{1}.window_times;
    prediction_scores = prediction_sessions_aversive(sess_idx).prediction_scores;
    prediction_ind = 1:20:length(session.TriggerMid);
    prediction_ind = prediction_ind+10;
    prediction_time = session.TriggerMid(prediction_ind);
    coupling_intp = interp1(coupling_time,coupling,prediction_time,'nearest');
    behavioral_matrix = session.behavioral_matrix_full;
    neural_time = session.NeuralTime;

    % Period 1: Before first aversive noise
    period1_idx = neural_time < aversive_times(1);

    % Period 2: After first aversive noise
    period2_idx = and(neural_time >= aversive_times(1), neural_time < aversive_times(4));

    % Calculate Goal-Directed Movement frequency
    goal_movement = behavioral_matrix(:, 7);
    freq_before = (sum(goal_movement(period1_idx)) / sum(period1_idx)) * 100;
    freq_after = (sum(goal_movement(period2_idx)) / sum(period2_idx)) * 100;
    goal_movement_change = freq_after - freq_before;
    
    % Define 7 time periods based on 6 aversive noises
    period_boundaries = [session.TriggerMid(1), aversive_times(1:6)'+session.TriggerMid(1), session.TriggerMid(end)];
    
    baselineMean = nan(1, config.n_behaviors);
    baselinesStd = nan(1, config.n_behaviors);
    
    % Process each time period
    for period = 1:7
        period_start = period_boundaries(period);
        period_end = period_boundaries(period + 1);

        % Get neural indices in this period
        prediction_idx = prediction_time >= period_start & prediction_time < period_end;
        if sum(prediction_idx) < 10
            continue;
        end
        prediction_index = find(prediction_idx);
        allcoupling_val = [];
        alldominant_beh = [];
        for pred_idx = 1:length(prediction_index)
            [max_conf, dominant_beh] = max(prediction_scores(prediction_index(pred_idx), :));
            if max_conf > config.confidence_threshold
                coupling_val = coupling_intp(prediction_index(pred_idx));
                allcoupling_val = [allcoupling_val, coupling_val];
                alldominant_beh = [alldominant_beh,dominant_beh];
            end
        end
        
        for behID = 1:config.n_behaviors
            if period == 1
                baselineMean(behID) = nanmean(allcoupling_val(alldominant_beh==behID));
                baselinesStd(behID) = nanstd(allcoupling_val(alldominant_beh==behID));
            end
            normed_beh_coupling = (allcoupling_val(alldominant_beh==behID) - baselineMean(behID))./baselinesStd(behID);
            
            % Store in aggregated results (original)
            aversive_results.coupling_values{behID, period} = ...
                [aversive_results.coupling_values{behID, period}; normed_beh_coupling'];
            
            % Store individual session mean (original)
            session_data_aversive{n_valid_sessions_aversive}(behID, period) = nanmean(normed_beh_coupling);
            
            % NEW: Store in long format for LME
            n_points = length(normed_beh_coupling);
            if n_points > 0
                aversive_long_data.session_id = [aversive_long_data.session_id; repmat(n_valid_sessions_aversive, n_points, 1)];
                aversive_long_data.period = [aversive_long_data.period; repmat(period, n_points, 1)];
                aversive_long_data.behavior = [aversive_long_data.behavior; repmat(behID, n_points, 1)];
                aversive_long_data.coupling = [aversive_long_data.coupling; normed_beh_coupling(:)];
                aversive_long_data.change = [aversive_long_data.change; repmat(goal_movement_change, n_points, 1)];
            end
        end
    end
end

fprintf('✓ Processed %d valid aversive sessions\n', n_valid_sessions_aversive);
fprintf('✓ Long-format data: %d rows\n\n', length(aversive_long_data.coupling));

% Calculate statistics for aversive sessions (original)
fprintf('Calculating aversive session statistics...\n');
aversive_results.mean_coupling = nan(config.n_behaviors, 7);
aversive_results.sem_coupling = nan(config.n_behaviors, 7);
aversive_results.n_samples = zeros(config.n_behaviors, 7);

for beh = 1:config.n_behaviors
    for period = 1:7
        values = aversive_results.coupling_values{beh, period};
        if ~isempty(values)
            aversive_results.mean_coupling(beh, period) = nanmean(values);
            aversive_results.sem_coupling(beh, period) = nanstd(values) / sqrt(length(values));
            aversive_results.n_samples(beh, period) = length(values);
        end
    end
end

fprintf('✓ Statistics calculated\n\n');

%% ========================================================================
%  SECTION 3: ANALYZE REWARD SESSIONS - MODIFIED FOR LME
%  ========================================================================

fprintf('PART 2: REWARD SESSIONS (4 periods)\n');
fprintf('====================================\n\n');

% Load reward data
fprintf('Loading reward session data...\n');
coupling_data = load('27-Oct-2025_RewardSeeking_session_metrics_breathing_LFPCcouple(10-1)');
sessions_reward = coupling_data.all_session_metrics;
pred_data = load('lstm_prediction_results_reward_27-Oct-2025');
prediction_sessions_reward = pred_data.final_results.session_predictions;
fprintf('✓ Loaded data: %d sessions\n\n', length(sessions_reward));

%%
% Initialize storage for LME - MODIFIED
reward_long_data = struct();
reward_long_data.session_id = [];
reward_long_data.period = [];
reward_long_data.behavior = [];
reward_long_data.coupling = [];

% Also keep original format
reward_results = struct();
reward_results.coupling_values = cell(config.n_behaviors, 4);
for beh = 1:config.n_behaviors
    for period = 1:4
        reward_results.coupling_values{beh, period} = [];
    end
end

% Process each reward session
fprintf('Processing reward sessions...\n');
n_valid_sessions_reward = 0;
session_data_reward = {};

for sess_idx = 1:length(sessions_reward)
    session = sessions_reward{sess_idx};
    
    % Check required fields
    if ~isfield(session, 'coupling_results_multiband') || ...
       sess_idx > length(prediction_sessions_reward)
        continue;
    end
    
    n_valid_sessions_reward = n_valid_sessions_reward + 1;
    session_data_reward{n_valid_sessions_reward} = nan(config.n_behaviors, 4);
    
    coupling = session.coupling_results_multiband.band_results{1}.MI_values;
    coupling_time = session.coupling_results_multiband.band_results{1}.window_times;
    prediction_scores = prediction_sessions_reward(sess_idx).prediction_scores;
    prediction_ind = 1:20:length(session.TriggerMid);
    prediction_ind = prediction_ind+10;
    prediction_time = session.TriggerMid(prediction_ind);
    coupling_intp = interp1(coupling_time,coupling,prediction_time,'nearest');
    
    time_boundaries = [8*60, 16*60, 24*60, 30*60];
    period_boundaries = [session.TriggerMid(1), time_boundaries + session.TriggerMid(1)];

    baselineMean = nan(1, config.n_behaviors);
    baselinesStd = nan(1, config.n_behaviors);
    
    % Process each time period
    for period = 1:4
        period_start = period_boundaries(period);
        period_end = period_boundaries(period + 1);

        prediction_idx = prediction_time >= period_start & prediction_time < period_end;
        if sum(prediction_idx) < 10
            continue;
        end
        prediction_index = find(prediction_idx);
        allcoupling_val = [];
        alldominant_beh = [];
        for pred_idx = 1:length(prediction_index)
            [max_conf, dominant_beh] = max(prediction_scores(prediction_index(pred_idx), :));
            if max_conf > config.confidence_threshold
                coupling_val = coupling_intp(prediction_index(pred_idx));
                allcoupling_val = [allcoupling_val, coupling_val];
                alldominant_beh = [alldominant_beh,dominant_beh];
            end
        end
        
        for behID = 1:config.n_behaviors
            if period == 1
                baselineMean(behID) = nanmean(allcoupling_val(alldominant_beh==behID));
                baselinesStd(behID) = nanstd(allcoupling_val(alldominant_beh==behID));
            end
            normed_beh_coupling = (allcoupling_val(alldominant_beh==behID) - baselineMean(behID))./baselinesStd(behID);
            
            % Store in aggregated results (original)
            reward_results.coupling_values{behID, period} = ...
                [reward_results.coupling_values{behID, period}; normed_beh_coupling'];
            
            % Store individual session mean (original)
            session_data_reward{n_valid_sessions_reward}(behID, period) = nanmean(normed_beh_coupling);
            
            % NEW: Store in long format for LME
            n_points = length(normed_beh_coupling);
            if n_points > 0
                reward_long_data.session_id = [reward_long_data.session_id; repmat(n_valid_sessions_reward, n_points, 1)];
                reward_long_data.period = [reward_long_data.period; repmat(period, n_points, 1)];
                reward_long_data.behavior = [reward_long_data.behavior; repmat(behID, n_points, 1)];
                reward_long_data.coupling = [reward_long_data.coupling; normed_beh_coupling(:)];
            end
        end
    end
end

fprintf('✓ Processed %d valid reward sessions\n', n_valid_sessions_reward);
fprintf('✓ Long-format data: %d rows\n\n', length(reward_long_data.coupling));

% Calculate statistics for reward sessions (original)
fprintf('Calculating reward session statistics...\n');
reward_results.mean_coupling = nan(config.n_behaviors, 4);
reward_results.sem_coupling = nan(config.n_behaviors, 4);
reward_results.n_samples = zeros(config.n_behaviors, 4);

for beh = 1:config.n_behaviors
    for period = 1:4
        values = reward_results.coupling_values{beh, period};
        if ~isempty(values)
            reward_results.mean_coupling(beh, period) = nanmean(values);
            reward_results.sem_coupling(beh, period) = nanstd(values) / sqrt(length(values));
            reward_results.n_samples(beh, period) = length(values);
        end
    end
end

fprintf('✓ Statistics calculated\n\n');

%% ========================================================================
%  SECTION 3.5: CREATE LME TABLES AND FIT MODELS - NEW
%  ========================================================================

fprintf('=== LINEAR MIXED-EFFECTS MODEL ANALYSIS ===\n\n');

%% Aversive Sessions LME
fprintf('AVERSIVE SESSIONS:\n');
fprintf('------------------\n\n');

% Create table for LME
tbl_aversive = table(aversive_long_data.session_id, ...
                     aversive_long_data.period, ...
                     aversive_long_data.behavior, ...
                     aversive_long_data.coupling, ...
                     'VariableNames', {'Session', 'Period', 'Behavior', 'Coupling'});

% Convert to categorical
tbl_aversive.Session = categorical(tbl_aversive.Session);
tbl_aversive.Period = categorical(tbl_aversive.Period);
tbl_aversive.Behavior = categorical(tbl_aversive.Behavior, 1:7, config.behavior_names);

fprintf('Dataset size: %d rows\n', height(tbl_aversive));
fprintf('Sessions: %d\n', length(unique(tbl_aversive.Session)));
fprintf('Periods: %d\n', length(unique(tbl_aversive.Period)));
fprintf('Behaviors: %d\n\n', length(unique(tbl_aversive.Behavior)));

% Display first 20 rows
fprintf('First 20 rows of data:\n');
disp(tbl_aversive(1:min(20, height(tbl_aversive)), :));

% Fit LME model
fprintf('\nFitting LME model: Coupling ~ Period * Behavior + (1|Session)\n');
fprintf('This may take a few minutes...\n\n');

try
    lme_aversive = fitlme(tbl_aversive, 'Coupling ~ Period * Behavior + (1|Session)');
    fprintf('✓ Aversive model fitted successfully\n\n');
    disp(lme_aversive);
catch ME
    fprintf('ERROR fitting aversive model: %s\n', ME.message);
    fprintf('Trying simpler model without interaction...\n');
    try
        lme_aversive = fitlme(tbl_aversive, 'Coupling ~ Period + Behavior + (1|Session)');
        fprintf('✓ Simplified aversive model fitted\n\n');
        disp(lme_aversive);
    catch ME2
        fprintf('ERROR: %s\n', ME2.message);
        lme_aversive = [];
    end
end

%% Reward Sessions LME
fprintf('\n\nREWARD SESSIONS:\n');
fprintf('----------------\n\n');

% Create table for LME
tbl_reward = table(reward_long_data.session_id, ...
                   reward_long_data.period, ...
                   reward_long_data.behavior, ...
                   reward_long_data.coupling, ...
                   'VariableNames', {'Session', 'Period', 'Behavior', 'Coupling'});

% Convert to categorical
tbl_reward.Session = categorical(tbl_reward.Session);
tbl_reward.Period = categorical(tbl_reward.Period);
tbl_reward.Behavior = categorical(tbl_reward.Behavior, 1:7, config.behavior_names);

fprintf('Dataset size: %d rows\n', height(tbl_reward));
fprintf('Sessions: %d\n', length(unique(tbl_reward.Session)));
fprintf('Periods: %d\n', length(unique(tbl_reward.Period)));
fprintf('Behaviors: %d\n\n', length(unique(tbl_reward.Behavior)));

% Fit LME model
fprintf('Fitting LME model: Coupling ~ Period * Behavior + (1|Session)\n');
fprintf('This may take a few minutes...\n\n');

try
    lme_reward = fitlme(tbl_reward, 'Coupling ~ Period * Behavior + (1|Session)');
    fprintf('✓ Reward model fitted successfully\n\n');
    disp(lme_reward);
catch ME
    fprintf('ERROR fitting reward model: %s\n', ME.message);
    fprintf('Trying simpler model without interaction...\n');
    try
        lme_reward = fitlme(tbl_reward, 'Coupling ~ Period + Behavior + (1|Session)');
        fprintf('✓ Simplified reward model fitted\n\n');
        disp(lme_reward);
    catch ME2
        fprintf('ERROR: %s\n', ME2.message);
        lme_reward = [];
    end
end


%% ========================================================================
%  COMBINE AVERSIVE AND REWARD DATA FOR THREE-WAY INTERACTION
%  ========================================================================

fprintf('=== PREPARING DATA FOR THREE-WAY INTERACTION MODEL ===\n\n');

% 1. Add SessionType to existing long-format data

% Aversive data
aversive_long_data.session_type = repmat({'Aversive'}, length(aversive_long_data.coupling), 1);

% Reward data  
reward_long_data.session_type = repmat({'Reward'}, length(reward_long_data.coupling), 1);

% 2. Align periods (use only first 4 periods from both)
% Since Reward has 4 periods and Aversive has 7, we need to decide:
% Option A: Use only first 4 periods from Aversive (matched timing)
% Option B: Keep all 7 Aversive periods, Reward periods 1-4 (unbalanced)

% OPTION A (RECOMMENDED): Match first 4 periods
fprintf('Matching periods: Using P1-P4 from both session types\n');

% Filter aversive to only first 4 periods
aversive_matched_idx = aversive_long_data.period <= 4;
aversive_matched = struct();
aversive_matched.session_id = aversive_long_data.session_id(aversive_matched_idx);
aversive_matched.period = aversive_long_data.period(aversive_matched_idx);
aversive_matched.behavior = aversive_long_data.behavior(aversive_matched_idx);
aversive_matched.coupling = aversive_long_data.coupling(aversive_matched_idx);
aversive_matched.session_type = aversive_long_data.session_type(aversive_matched_idx);

fprintf('  Aversive (P1-P4): %d data points\n', length(aversive_matched.coupling));
fprintf('  Reward (P1-P4): %d data points\n\n', length(reward_long_data.coupling));

% 3. Combine datasets
% Need to make session IDs unique across session types
max_aversive_session = max(aversive_matched.session_id);
reward_matched = reward_long_data;
reward_matched.session_id = reward_long_data.session_id + max_aversive_session;

% Concatenate
combined_data = struct();
combined_data.session_id = [aversive_matched.session_id; reward_matched.session_id];
combined_data.period = [aversive_matched.period; reward_matched.period];
combined_data.behavior = [aversive_matched.behavior; reward_matched.behavior];
combined_data.coupling = [aversive_matched.coupling; reward_matched.coupling];
combined_data.session_type = [aversive_matched.session_type; reward_matched.session_type];

fprintf('Combined dataset: %d total data points\n', length(combined_data.coupling));
fprintf('  Unique sessions: %d\n', length(unique(combined_data.session_id)));
fprintf('  Aversive sessions: %d\n', n_valid_sessions_aversive);
fprintf('  Reward sessions: %d\n\n', n_valid_sessions_reward);

% 4. Create table for LME
tbl_combined = table(combined_data.session_id, ...
                     combined_data.period, ...
                     combined_data.behavior, ...
                     combined_data.coupling, ...
                     combined_data.session_type, ...
                     'VariableNames', {'Session', 'Period', 'Behavior', 'Coupling', 'SessionType'});

% Convert to categorical
tbl_combined.Session = categorical(tbl_combined.Session);
tbl_combined.Period = categorical(tbl_combined.Period);
tbl_combined.Behavior = categorical(tbl_combined.Behavior, 1:7, config.behavior_names);
tbl_combined.SessionType = categorical(tbl_combined.SessionType);

fprintf('Table created:\n');
fprintf('  Rows: %d\n', height(tbl_combined));
fprintf('  Variables: %s\n\n', strjoin(tbl_combined.Properties.VariableNames, ', '));

% Display structure
disp(head(tbl_combined, 20));

% 5. Check data distribution
fprintf('\nData distribution check:\n');
fprintf('------------------------\n');
summary_table = groupsummary(tbl_combined, {'SessionType', 'Behavior', 'Period'}, 'mean', 'Coupling');
disp(summary_table(1:20, :));

%% ========================================================================
%  FIT THREE-WAY INTERACTION MODEL
%  ========================================================================

fprintf('\n=== FITTING THREE-WAY INTERACTION MODEL ===\n\n');

% Model 1: Full three-way interaction
fprintf('Model 1: Full three-way interaction\n');
fprintf('Formula: Coupling ~ Period * Behavior * SessionType + (1|Session)\n');
fprintf('This may take several minutes...\n\n');

try
    lme_threeway = fitlme(tbl_combined, ...
                         'Coupling ~ Period * Behavior * SessionType + (1|Session)', ...
                         'FitMethod', 'REML');
    
    fprintf('✓ Three-way interaction model fitted successfully\n\n');
    disp(lme_threeway);
    
    % Save model
    results.lme_threeway = lme_threeway;
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    fprintf('Model may be too complex. Trying reduced models...\n\n');
    lme_threeway = [];
end

% Extract coefficients
fprintf('\n=== KEY COEFFICIENTS ===\n\n');

if ~isempty(lme_threeway)
    coef_table = lme_threeway.Coefficients;
    
    % Find three-way interaction terms for Rearing
    rearing_threeway = contains(coef_table.Name, 'Rearing') & ...
                       contains(coef_table.Name, 'Period') & ...
                       contains(coef_table.Name, 'SessionType');
    
    if any(rearing_threeway)
        fprintf('THREE-WAY INTERACTION: Period × Rearing × SessionType\n');
        fprintf('-----------------------------------------------------\n');
        disp(coef_table(rearing_threeway, :));
        fprintf('\n');
    end
    
    % Find two-way: Period × Rearing (main effect across session types)
    rearing_period = contains(coef_table.Name, 'Rearing') & ...
                     contains(coef_table.Name, 'Period') & ...
                     ~contains(coef_table.Name, 'SessionType');
    
    if any(rearing_period)
        fprintf('TWO-WAY INTERACTION: Period × Rearing (averaged)\n');
        fprintf('------------------------------------------------\n');
        disp(coef_table(rearing_period, :));
        fprintf('\n');
    end
end

%% ========================================================================
%  COMPLETE VISUALIZATION: ALL BEHAVIORS × SESSION TYPES - PERIOD-SPECIFIC SIGNIFICANCE
%  ========================================================================
if ~isempty(lme_threeway)
    figure('Position', [50, 50, 1800, 1000]);
    
    % Create full prediction grid - CORRECTED
    n_pred_total = 2 * config.n_behaviors * 4;  % 2 session types × 7 behaviors × 4 periods
    pred_session = ones(n_pred_total, 1);
    pred_period = zeros(n_pred_total, 1);
    pred_behavior = zeros(n_pred_total, 1);
    pred_sessiontype = cell(n_pred_total, 1);
    session_type_names = {'Aversive', 'Reward'};
    
    idx = 1;
    for st = 1:2
        for b = 1:config.n_behaviors
            for p = 1:4
                pred_period(idx) = p;
                pred_behavior(idx) = b;
                pred_sessiontype{idx} = session_type_names{st};
                idx = idx + 1;
            end
        end
    end
    
    % Create table
    pred_grid_full = table(categorical(pred_session), ...
                          categorical(pred_period), ...
                          categorical(pred_behavior, 1:7, config.behavior_names), ...
                          categorical(pred_sessiontype), ...
                          'VariableNames', {'Session', 'Period', 'Behavior', 'SessionType'});
    
    % Get predictions
    [pred_full, ~] = predict(lme_threeway, pred_grid_full, 'Conditional', false);
    
    % Reshape: [period × behavior × session_type]
    pred_array = reshape(pred_full, 4, config.n_behaviors, 2);
    
    % TEST SIGNIFICANCE OF PERIOD × SESSIONTYPE INTERACTION FOR EACH BEHAVIOR × PERIOD
    fprintf('\n=== TESTING PERIOD × SESSIONTYPE INTERACTION BY BEHAVIOR AND PERIOD ===\n\n');
    
    coef_table = lme_threeway.Coefficients;
    
    % Store p-values for each behavior × period combination
    % Rows = behaviors, Columns = periods (2, 3, 4)
    % Period 1 is reference, so no test needed
    period_pvals = nan(config.n_behaviors, 3);  % Periods 2, 3, 4
    
    % For each behavior, find its three-way interaction terms
    for b = 1:config.n_behaviors
        beh_name = config.behavior_names{b};
        
        fprintf('%s:\n', beh_name);
        
        % Find all three-way interaction terms for this behavior
        interaction_idx = contains(coef_table.Name, ['Behavior_', beh_name]) & ...
                         contains(coef_table.Name, 'Period') & ...
                         contains(coef_table.Name, 'SessionType');
        
        if any(interaction_idx)
            beh_interactions = coef_table(interaction_idx, :);
            fprintf('  Three-way interaction terms found: %d\n', height(beh_interactions));
            
            % Parse each interaction term to extract which period it refers to
            for i = 1:height(beh_interactions)
                term_name = beh_interactions.Name{i};
                
                % Extract period number from term name
                % Term format: "Period_X:Behavior_Y:SessionType_Z"
                period_match = regexp(term_name, 'Period_(\d+)', 'tokens');
                
                if ~isempty(period_match)
                    period_num = str2double(period_match{1}{1});
                    
                    % Store p-value (Period 2 -> column 1, Period 3 -> column 2, etc.)
                    if period_num >= 2 && period_num <= 4
                        period_pvals(b, period_num - 1) = beh_interactions.pValue(i);
                        
                        fprintf('    Period %d: β=%.4f, p=%.4f', ...
                               period_num, ...
                               beh_interactions.Estimate(i), ...
                               beh_interactions.pValue(i));
                        
                        if beh_interactions.pValue(i) < 0.05
                            fprintf(' ***');
                        end
                        fprintf('\n');
                    end
                end
            end
        else
            fprintf('  No interaction terms found (reference category)\n');
        end
        fprintf('\n');
    end
    
    % PLOT WITH PERIOD-SPECIFIC SIGNIFICANCE STARS
    ax = [];
    for b = 1:config.n_behaviors
        ax(end+1) = subplot(3, 3, b);
        hold on;
        
        % Aversive
        h_av = plot(1:4, pred_array(:, b, 1), 'o-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
             'Color', [0.8, 0.2, 0.2], 'MarkerFaceColor', [0.8, 0.2, 0.2], ...
             'DisplayName', 'Aversive');
        
        % Reward
        h_rw = plot(1:4, pred_array(:, b, 2), 's-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
             'Color', [0.2, 0.6, 0.2], 'MarkerFaceColor', [0.2, 0.6, 0.2], ...
             'DisplayName', 'Reward');
        
        % Get y-axis limits for star placement
        y_vals = pred_array(:, b, :);
        y_min = min(y_vals(:));
        y_max = max(y_vals(:));
        y_range = y_max - y_min;
        
        % Add significance stars for each period (P2, P3, P4)
        star_y = y_max + 0.15 * y_range;  % Position stars above data
        
        for p = 2:4  % Periods 2, 3, 4 (Period 1 is reference)
            p_val = period_pvals(b, p - 1);
            
            if ~isnan(p_val)
                if p_val < 0.001
                    star_text = '***';
                    star_size = 14;
                    star_color = [0.8, 0, 0];
                elseif p_val < 0.01
                    star_text = '**';
                    star_size = 13;
                    star_color = [0.8, 0.2, 0];
                elseif p_val < 0.05
                    star_text = '*';
                    star_size = 12;
                    star_color = [0.8, 0.4, 0];
                else
                    star_text = '';
                    star_size = 0;
                end
                
                % Plot star if significant
                if ~isempty(star_text)
                    text(p, star_y, star_text, ...
                         'FontSize', star_size, ...
                         'FontWeight', 'bold', ...
                         'Color', star_color, ...
                         'HorizontalAlignment', 'center', ...
                         'VerticalAlignment', 'bottom');
                end
            end
        end
        
        % Title
        title_str = config.behavior_names{b};
        title_color = 'k';
        
        % Check if ANY period is significant
        if any(period_pvals(b, :) < 0.05)
            title_color = 'r';
        end
        
        title(title_str, 'FontSize', 13, 'FontWeight', 'bold', 'Color', title_color);
        
        xlabel('Period', 'FontSize', 11);
        ylabel('Coupling (Z-score)', 'FontSize', 11);
        xticks(1:4);
        xticklabels({'P1', 'P2', 'P3', 'P4'});
        
        % Adjust y-limits to accommodate stars
        ylim([y_min - 0.1 * y_range, y_max + 0.25 * y_range]);
        
        if b == 1
            legend([h_av, h_rw], 'Location', 'northwest', 'FontSize', 10);
        end
        grid on;
        set(gca, 'FontSize', 10);
        hold off;
    end
    linkaxes([ax], 'xy');
    
    % Add overall title with legend
    sgtitle({'Three-Way Interaction: Period × Behavior × SessionType', ...
            'Stars above each period indicate significant Period×SessionType interaction for that behavior', ...
            '* p<0.05, ** p<0.01, *** p<0.001'}, ...
            'FontSize', 15, 'FontWeight', 'bold');
    
end
%% ========================================================================
%  LME MODEL WITH CONTINUOUS BEHAVIORAL CHANGE VARIABLE
%  ========================================================================

fprintf('\n=== LME WITH CONTINUOUS CHANGE VARIABLE ===\n\n');

% 1. Create table with change variable
tbl_aversive_change = table(aversive_long_data.session_id, ...
                            aversive_long_data.period, ...
                            aversive_long_data.behavior, ...
                            aversive_long_data.coupling, ...
                            aversive_long_data.change, ...
                            'VariableNames', {'Session', 'Period', 'Behavior', 'Coupling', 'Change'});

% Convert to categorical (except Change - keep continuous)
tbl_aversive_change.Session = categorical(tbl_aversive_change.Session);
tbl_aversive_change.Period = categorical(tbl_aversive_change.Period);
tbl_aversive_change.Behavior = categorical(tbl_aversive_change.Behavior, 1:7, config.behavior_names);
% Change stays as numeric (continuous)

fprintf('Dataset created:\n');
fprintf('  Rows: %d\n', height(tbl_aversive_change));
fprintf('  Sessions: %d\n', length(unique(tbl_aversive_change.Session)));
fprintf('  Periods: %d\n', length(unique(tbl_aversive_change.Period)));
fprintf('  Behaviors: %d\n', length(unique(tbl_aversive_change.Behavior)));
fprintf('  Change range: [%.2f, %.2f]\n\n', ...
       min(tbl_aversive_change.Change), max(tbl_aversive_change.Change));

% Display first 20 rows
fprintf('First 20 rows:\n');
disp(tbl_aversive_change(1:20, :));

% 2. Standardize the Change variable (recommended for interpretation)
fprintf('\nStandardizing Change variable for better interpretation...\n');
tbl_aversive_change.Change_z = (tbl_aversive_change.Change - mean(tbl_aversive_change.Change)) / ...
                                std(tbl_aversive_change.Change);

fprintf('  Mean Change: %.3f\n', mean(tbl_aversive_change.Change));
fprintf('  SD Change: %.3f\n', std(tbl_aversive_change.Change));
fprintf('  After z-scoring: Mean=%.3f, SD=%.3f\n\n', ...
       mean(tbl_aversive_change.Change_z), std(tbl_aversive_change.Change_z));

% 3. Fit LME models

fprintf('=== FITTING LME MODELS ===\n\n');

% Model 1: Full three-way interaction with continuous change
fprintf('Model 1: Full interaction with continuous Change\n');
fprintf('Formula: Coupling ~ Period * Behavior * Change_z + (1|Session)\n');
fprintf('This may take several minutes...\n\n');

try
    lme_change_full = fitlme(tbl_aversive_change, ...
                            'Coupling ~ Period * Behavior * Change_z + (1|Session)', ...
                            'FitMethod', 'REML');
    
    fprintf('✓ Full model fitted successfully\n\n');
    disp(lme_change_full);
    
    % Save model
    results.lme_change_full = lme_change_full;
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    fprintf('Model too complex. Trying simpler model...\n\n');
    lme_change_full = [];
end

% 4. Extract and interpret key coefficients
if ~isempty(lme_change_full)
    fprintf('\n=== KEY FINDINGS FROM FULL MODEL ===\n\n');
    
    coef_table = lme_change_full.Coefficients;
    
    % Main effect of Change
    fprintf('1. MAIN EFFECT OF CHANGE:\n');
    fprintf('   Does goal-directed change predict coupling overall?\n');
    change_main_idx = strcmp(coef_table.Name, 'Change_z');
    if any(change_main_idx)
        fprintf('   β = %.4f, p = %.4f', ...
               coef_table.Estimate(change_main_idx), ...
               coef_table.pValue(change_main_idx));
        if coef_table.pValue(change_main_idx) < 0.05
            fprintf(' ***\n');
            fprintf('   → Significant! Sessions with greater goal-directed change show ');
            if coef_table.Estimate(change_main_idx) > 0
                fprintf('higher coupling.\n');
            else
                fprintf('lower coupling.\n');
            end
        else
            fprintf('\n   → Not significant\n');
        end
    end
    fprintf('\n');
    
    % Period × Change interaction
    fprintf('2. PERIOD × CHANGE INTERACTION:\n');
    fprintf('   Does the effect of change vary across periods?\n');
    period_change_idx = contains(coef_table.Name, 'Period') & ...
                       contains(coef_table.Name, 'Change_z') & ...
                       ~contains(coef_table.Name, 'Behavior');
    if any(period_change_idx)
        period_change_coefs = coef_table(period_change_idx, :);
        fprintf('   Found %d interaction terms:\n', height(period_change_coefs));
        for i = 1:height(period_change_coefs)
            fprintf('     %s: β=%.4f, p=%.4f', ...
                   period_change_coefs.Name{i}, ...
                   period_change_coefs.Estimate(i), ...
                   period_change_coefs.pValue(i));
            if period_change_coefs.pValue(i) < 0.05
                fprintf(' *\n');
            else
                fprintf('\n');
            end
        end
    else
        fprintf('   No significant interactions found\n');
    end
    fprintf('\n');
    
    % Behavior × Change interaction
    fprintf('3. BEHAVIOR × CHANGE INTERACTION:\n');
    fprintf('   Does the effect of change differ by behavior?\n');
    behavior_change_idx = contains(coef_table.Name, 'Behavior') & ...
                         contains(coef_table.Name, 'Change_z') & ...
                         ~contains(coef_table.Name, 'Period');
    if any(behavior_change_idx)
        behavior_change_coefs = coef_table(behavior_change_idx, :);
        fprintf('   Found %d interaction terms:\n', height(behavior_change_coefs));
        
        % Highlight significant ones
        sig_idx = behavior_change_coefs.pValue < 0.05;
        if any(sig_idx)
            fprintf('   SIGNIFICANT interactions:\n');
            sig_coefs = behavior_change_coefs(sig_idx, :);
            for i = 1:height(sig_coefs)
                fprintf('     %s: β=%.4f, p=%.4f ***\n', ...
                       sig_coefs.Name{i}, ...
                       sig_coefs.Estimate(i), ...
                       sig_coefs.pValue(i));
            end
        else
            fprintf('   No significant interactions\n');
        end
    end
    fprintf('\n');
    
    % Three-way interaction: Period × Behavior × Change
    fprintf('4. THREE-WAY INTERACTION: Period × Behavior × Change\n');
    fprintf('   Does the Period × Behavior effect depend on goal-directed change?\n');
    fprintf('   (e.g., Does Rearing increase more in sessions with greater change?)\n');
    threeway_idx = contains(coef_table.Name, 'Period') & ...
                   contains(coef_table.Name, 'Behavior') & ...
                   contains(coef_table.Name, 'Change_z');
    
    if any(threeway_idx)
        threeway_coefs = coef_table(threeway_idx, :);
        fprintf('   Found %d three-way interaction terms\n', height(threeway_coefs));
        
        % Look specifically for Rearing
        rearing_threeway_idx = contains(threeway_coefs.Name, 'Rearing');
        if any(rearing_threeway_idx)
            fprintf('\n   REARING-specific interactions:\n');
            rearing_threeway = threeway_coefs(rearing_threeway_idx, :);
            for i = 1:height(rearing_threeway)
                fprintf('     %s:\n', rearing_threeway.Name{i});
                fprintf('       β=%.4f, p=%.4f', ...
                       rearing_threeway.Estimate(i), ...
                       rearing_threeway.pValue(i));
                if rearing_threeway.pValue(i) < 0.05
                    fprintf(' ***\n');
                    fprintf('       → Rearing period effect modulated by goal-directed change!\n');
                else
                    fprintf('\n');
                end
            end
        end
        
        % Show all significant three-way interactions
        sig_threeway_idx = threeway_coefs.pValue < 0.05;
        if any(sig_threeway_idx)
            fprintf('\n   ALL SIGNIFICANT three-way interactions:\n');
            sig_threeway = threeway_coefs(sig_threeway_idx, :);
            disp(sig_threeway);
        end
    else
        fprintf('   No three-way interaction terms found\n');
    end
    fprintf('\n');
end

% 5. Visualize: Coupling vs Change by Behavior

if ~isempty(lme_change_full) || ~isempty(lme_change_main)
    
    % Use whichever model is available
    model_to_plot = lme_change_full;
    if isempty(model_to_plot)
        model_to_plot = lme_change_main;
    end
    
    figure('Position', [100, 100, 1600, 900]);
    
    % Create prediction grid: Change values × Behaviors × Periods
    change_range = linspace(min(tbl_aversive_change.Change_z), ...
                           max(tbl_aversive_change.Change_z), 50);
    
    colors = [0.8 0.2 0.2; 0.2 0.4 0.8; 0.2 0.8 0.4; 0.8 0.6 0.2;
              0.6 0.2 0.8; 0.8 0.4 0.6; 0.4 0.4 0.4];
    
    % Plot each behavior
    for b = 1:config.n_behaviors
        subplot(3, 3, b);
        hold on;
        
        % Plot for each period
        for p = [1, 4]  % Show Period 1 and Period 4 for clarity
            
            % Create prediction grid
            pred_grid = table();
            for c = 1:length(change_range)
                pred_grid = [pred_grid; table(categorical(1), ...  % Reference session
                                             categorical(p), ...
                                             categorical(b, 1:7, config.behavior_names), ...
                                             change_range(c), ...
                                             'VariableNames', {'Session', 'Period', 'Behavior', 'Change_z'})];
            end
            
            % Get predictions
            [pred_coupling, ~] = predict(model_to_plot, pred_grid, 'Conditional', false);
            
            % Plot line
            if p == 1
                linestyle = '-';
                label_str = 'Period 1';
                line_color = colors(b, :) * 0.6;  % Darker
            else
                linestyle = '--';
                label_str = 'Period 4';
                line_color = colors(b, :);  % Lighter
            end
            
            plot(change_range, pred_coupling, linestyle, ...
                 'LineWidth', 2.5, 'Color', line_color, ...
                 'DisplayName', label_str);
        end
        
        % Formatting
        xlabel('Goal-Directed Change (z-score)', 'FontSize', 11);
        ylabel('Predicted Coupling', 'FontSize', 11);
        title(config.behavior_names{b}, 'FontSize', 12, 'FontWeight', 'bold');
        
        if b == 1
            legend('Location', 'best', 'FontSize', 10);
        end
        grid on;
        set(gca, 'FontSize', 10);
        hold off;
    end
    
    sgtitle('Coupling vs Goal-Directed Change: Period 1 (solid) vs Period 4 (dashed)', ...
            'FontSize', 15, 'FontWeight', 'bold');
end

% 6. Visualize: Effect of Change on Period slopes

if ~isempty(lme_change_full)
    
    figure('Position', [100, 100, 1400, 600]);
    
    % Calculate predicted slopes at different levels of Change
    change_levels = [-1, 0, 1];  % Low, Medium, High change (in z-scores)
    change_labels = {'Low Change', 'Medium Change', 'High Change'};
    
    for change_level = 1:3
        subplot(1, 3, change_level);
        hold on;
        
        % For each behavior, get predictions across periods at this change level
        for b = 1:config.n_behaviors
            pred_periods = nan(1, 4);
            
            for p = 1:4
                pred_grid = table(categorical(1), ...
                                 categorical(p), ...
                                 categorical(b, 1:7, config.behavior_names), ...
                                 change_levels(change_level), ...
                                 'VariableNames', {'Session', 'Period', 'Behavior', 'Change_z'});
                
                pred_periods(p) = predict(lme_change_full, pred_grid, 'Conditional', false);
            end
            
            % Plot
            plot(1:4, pred_periods, 'o-', 'LineWidth', 2, 'MarkerSize', 6, ...
                 'Color', colors(b, :), 'MarkerFaceColor', colors(b, :), ...
                 'DisplayName', config.behavior_names{b});
        end
        
        xlabel('Period', 'FontSize', 12);
        ylabel('Predicted Coupling', 'FontSize', 12);
        title(change_labels{change_level}, 'FontSize', 13, 'FontWeight', 'bold');
        xticks(1:4);
        xticklabels({'P1', 'P2', 'P3', 'P4'});
        
        if change_level == 1
            legend('Location', 'best', 'FontSize', 9);
        end
        grid on;
        hold off;
    end
    
    sgtitle('How Goal-Directed Change Modulates Period Effects by Behavior', ...
            'FontSize', 16, 'FontWeight', 'bold');
end

% 7. Summary statistics

fprintf('\n=== SUMMARY: CHANGE VARIABLE STATISTICS ===\n\n');
fprintf('Goal-Directed Change by Session:\n');
fprintf('  Mean: %.3f%%\n', mean(tbl_aversive_change.Change));
fprintf('  SD: %.3f%%\n', std(tbl_aversive_change.Change));
fprintf('  Range: [%.3f%%, %.3f%%]\n', ...
       min(tbl_aversive_change.Change), max(tbl_aversive_change.Change));
fprintf('  Median: %.3f%%\n\n', median(tbl_aversive_change.Change));

% Correlation between Change and mean coupling by session
session_means = groupsummary(tbl_aversive_change, 'Session', 'mean', {'Coupling', 'Change'});
[r, p] = corr(session_means.mean_Coupling, session_means.mean_Change, 'Type', 'Pearson');
fprintf('Correlation: Coupling vs Change (across sessions)\n');
fprintf('  r = %.3f, p = %.4f', r, p);
if p < 0.05
    fprintf(' ***\n');
else
    fprintf('\n');
end
%% ========================================================================
%  SECTION 4: VISUALIZATION (ORIGINAL)
%  ========================================================================

fprintf('\nCreating visualizations...\n\n');

%% Figure 1: Aversive Sessions - Individual Session Lines
fig1 = figure('Position', [50, 50, 1600, 900]);
ax = [];

for beh = 1:config.n_behaviors
    ax(end+1) = subplot(3, 3, beh);
    hold on;
    
    % Plot each individual session as thin line
    for sess = 1:n_valid_sessions_aversive
        plot(1:7, session_data_aversive{sess}(beh, :), '-', ...
             'LineWidth', 1.5, 'Color', [0.2 0.4 0.8 0.25]);
    end
    
    % Plot mean on top as thick line
    mean_vals = aversive_results.mean_coupling(beh, :);
    plot(1:7, mean_vals, 'o-', 'LineWidth', 3, ...
         'MarkerSize', 10, 'MarkerFaceColor', [0.8 0.2 0.2], ...
         'Color', [0.8 0.2 0.2], 'MarkerEdgeColor', 'k');
    
    xlabel('Time Period', 'FontSize', 11);
    ylabel('Coupling Strength (Z-score)', 'FontSize', 11);
    title(config.behavior_names{beh}, 'FontSize', 12, 'FontWeight', 'bold');
    xlim([0.5, 7.5]);
    xticks(1:7);
    xticklabels({'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'});
    grid on;
    set(gca, 'FontSize', 10);
    hold off;
end

linkaxes([ax],'xy')
ylim([-1,4])
sgtitle(sprintf('Aversive Sessions: Individual Lines (n=%d) + Mean', n_valid_sessions_aversive), ...
        'FontSize', 15, 'FontWeight', 'bold');

%% Figure 2: Reward Sessions - Individual Session Lines
fig2 = figure('Position', [100, 100, 1600, 900]);
ax = [];

for beh = 1:config.n_behaviors
    ax(end+1) = subplot(3, 3, beh);
    hold on;
    
    % Plot each individual session as thin line
    for sess = 1:n_valid_sessions_reward
        plot(1:4, session_data_reward{sess}(beh, :), '-', ...
             'LineWidth', 1.5, 'Color', [0.8 0.4 0.2 0.25]);
    end
    
    % Plot mean on top as thick line
    mean_vals = reward_results.mean_coupling(beh, :);
    plot(1:4, mean_vals, 'o-', 'LineWidth', 3, ...
         'MarkerSize', 10, 'MarkerFaceColor', [0.2 0.6 0.2], ...
         'Color', [0.2 0.6 0.2], 'MarkerEdgeColor', 'k');
    
    xlabel('Time Period', 'FontSize', 11);
    ylabel('Coupling Strength (Z-score)', 'FontSize', 11);
    title(config.behavior_names{beh}, 'FontSize', 12, 'FontWeight', 'bold');
    xlim([0.5, 4.5]);
    xticks(1:4);
    xticklabels({'P1', 'P2', 'P3', 'P4'});
    grid on;
    set(gca, 'FontSize', 10);
    hold off;
end

linkaxes([ax],'xy')
ylim([-1,4])
sgtitle(sprintf('Reward Sessions: Individual Lines (n=%d) + Mean', n_valid_sessions_reward), ...
        'FontSize', 15, 'FontWeight', 'bold');

%% Figure 3: Heatmap Comparison
fig3 = figure('Position', [150, 150, 1400, 600]);

% Aversive heatmap
subplot(1, 2, 1);
imagesc(aversive_results.mean_coupling');
colorbar;
xlabel('Behavior Type', 'FontSize', 12);
ylabel('Time Period', 'FontSize', 12);
title('Aversive Sessions', 'FontSize', 13, 'FontWeight', 'bold');
xticks(1:7);
xticklabels(cellfun(@(x) x(1:min(3,length(x))), config.behavior_names, 'UniformOutput', false));
yticks(1:7);
yticklabels({'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'});
set(gca, 'FontSize', 11);
axis tight;

% Reward heatmap
subplot(1, 2, 2);
imagesc(reward_results.mean_coupling');
colorbar;
xlabel('Behavior Type', 'FontSize', 12);
ylabel('Time Period', 'FontSize', 12);
title('Reward Sessions', 'FontSize', 13, 'FontWeight', 'bold');
xticks(1:7);
xticklabels(cellfun(@(x) x(1:min(3,length(x))), config.behavior_names, 'UniformOutput', false));
yticks(1:4);
yticklabels({'P1', 'P2', 'P3', 'P4'});
set(gca, 'FontSize', 11);
axis tight;

sgtitle('Coupling Strength Heatmaps: Behavior × Time Period', ...
        'FontSize', 15, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 5: SUMMARY STATISTICS (ORIGINAL)
%  ========================================================================

fprintf('=== SUMMARY STATISTICS ===\n\n');

fprintf('AVERSIVE SESSIONS (n=%d):\n', n_valid_sessions_aversive);
fprintf('------------------\n');
for beh = 1:config.n_behaviors
    fprintf('%s:\n', config.behavior_names{beh});
    for period = 1:7
        if ~isnan(aversive_results.mean_coupling(beh, period))
            fprintf('  Period %d: %.4f ± %.4f (n=%d)\n', ...
                    period, ...
                    aversive_results.mean_coupling(beh, period), ...
                    aversive_results.sem_coupling(beh, period), ...
                    aversive_results.n_samples(beh, period));
        end
    end
    fprintf('\n');
end

fprintf('REWARD SESSIONS (n=%d):\n', n_valid_sessions_reward);
fprintf('----------------\n');
for beh = 1:config.n_behaviors
    fprintf('%s:\n', config.behavior_names{beh});
    for period = 1:4
        if ~isnan(reward_results.mean_coupling(beh, period))
            fprintf('  Period %d: %.4f ± %.4f (n=%d)\n', ...
                    period, ...
                    reward_results.mean_coupling(beh, period), ...
                    reward_results.sem_coupling(beh, period), ...
                    reward_results.n_samples(beh, period));
        end
    end
    fprintf('\n');
end


%% INTERACTION PLOT: Separate line for each behavior

figure('Position', [100, 100, 1200, 800]);

% Extract predicted values for each Period × Behavior combination
behaviors = categories(tbl_aversive.Behavior);
% periods = categories(tbl_aversive.Period);
periods = categories(tbl_reward.Period);
n_behaviors = length(behaviors);
n_periods = length(periods);

% Create prediction grid
pred_data = table();
for b = 1:n_behaviors
    for p = 1:n_periods
        % Create one row for each combination (using a reference session)
        pred_data = [pred_data; table(categorical(1), ...  % Reference session
                                     categorical(p), ...
                                     categorical(b, 1:7, config.behavior_names), ...
                                     'VariableNames', {'Session', 'Period', 'Behavior'})];
    end
end

% Get predictions from the model
% [pred_coupling, pred_ci] = predict(lme_aversive, pred_data, 'Conditional', false);
[pred_coupling, pred_ci] = predict(lme_reward, pred_data, 'Conditional', false);

% Reshape for plotting
pred_matrix = reshape(pred_coupling, n_periods, n_behaviors);
pred_ci_lower = reshape(pred_ci(:,1), n_periods, n_behaviors);
pred_ci_upper = reshape(pred_ci(:,2), n_periods, n_behaviors);

% Define colors for each behavior
colors = [0.8 0.2 0.2;    % Reward - red
          0.2 0.4 0.8;    % Walking - blue
          0.2 0.8 0.4;    % Rearing - green
          0.8 0.6 0.2;    % Scanning - orange
          0.6 0.2 0.8;    % Ground-Sniff - purple
          0.8 0.4 0.6;    % Grooming - pink
          0.4 0.4 0.4];   % Standing - gray

% Plot lines for each behavior
hold on;
for b = 1:n_behaviors
    % Plot prediction line
    plot(1:n_periods, pred_matrix(:, b), 'o-', ...
         'LineWidth', 2.5, 'MarkerSize', 8, ...
         'Color', colors(b, :), 'MarkerFaceColor', colors(b, :), ...
         'DisplayName', config.behavior_names{b});
    
    % Add confidence interval (optional)
    fill([1:n_periods, fliplr(1:n_periods)], ...
         [pred_ci_lower(:, b)', fliplr(pred_ci_upper(:, b)')], ...
         colors(b, :), 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
         'HandleVisibility', 'off');
end

% Formatting
xlabel('Time Period', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Predicted Coupling Strength (Z-score)', 'FontSize', 14, 'FontWeight', 'bold');
title('Interaction Plot: Period × Behavior Effect on Coupling', ...
      'FontSize', 16, 'FontWeight', 'bold');
xticks(1:n_periods);
xticklabels({'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'});
legend('Location', 'best', 'FontSize', 11);
grid on;
set(gca, 'FontSize', 12);
hold off;

% Add interpretation guide
annotation('textbox', [0.15, 0.02, 0.7, 0.05], ...
          'String', 'Non-parallel lines indicate interaction: behaviors respond differently to periods', ...
          'EdgeColor', 'none', 'FontSize', 10, 'FontWeight', 'bold', ...
          'HorizontalAlignment', 'center');

%% COEFFICIENT PLOT: Visualize interaction coefficients

% Extract fixed effects coefficients
coef_table = lme_aversive.Coefficients;

% Find interaction terms (contain both Period and Behavior names)
is_interaction = contains(coef_table.Name, 'Period') & ...
                 contains(coef_table.Name, 'Behavior');
interaction_coefs = coef_table(is_interaction, :);

if height(interaction_coefs) > 0
    figure('Position', [100, 100, 1000, 800]);
    
    % Extract period and behavior from coefficient names
    n_interactions = height(interaction_coefs);
    
    % Create horizontal coefficient plot with confidence intervals
    subplot(1, 1, 1);
    hold on;
    
    y_positions = 1:n_interactions;
    
    for i = 1:n_interactions
        est = interaction_coefs.Estimate(i);
        se = interaction_coefs.SE(i);
        ci_lower = interaction_coefs.Lower(i);
        ci_upper = interaction_coefs.Upper(i);
        
        % Color by significance
        if interaction_coefs.pValue(i) < 0.05
            color = [0.8, 0.2, 0.2];  % Red for significant
            marker_face = [0.8, 0.2, 0.2];
        else
            color = [0.5, 0.5, 0.5];  % Gray for non-significant
            marker_face = 'none';
        end
        
        % Plot CI line
        plot([ci_lower, ci_upper], [i, i], '-', ...
             'LineWidth', 2, 'Color', color);
        
        % Plot point estimate
        plot(est, i, 'o', 'MarkerSize', 8, ...
             'MarkerFaceColor', marker_face, ...
             'MarkerEdgeColor', color, 'LineWidth', 2);
    end
    
    % Add zero reference line
    plot([0, 0], [0, n_interactions+1], 'k--', 'LineWidth', 1.5);
    
    % Formatting
    xlabel('Interaction Coefficient Estimate', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Interaction Term', 'FontSize', 13, 'FontWeight', 'bold');
    title('Period × Behavior Interaction Coefficients', ...
          'FontSize', 15, 'FontWeight', 'bold');
    yticks(y_positions);
    yticklabels(interaction_coefs.Name);
    ylim([0, n_interactions+1]);
    grid on;
    set(gca, 'FontSize', 10);
    hold off;
    
    % Add legend
    legend({'Non-significant', 'Significant (p<0.05)'}, ...
           'Location', 'best', 'FontSize', 11);
end


%% ========================================================================
%  SECTION 6: SAVE RESULTS - MODIFIED
%  ========================================================================

fprintf('Saving results...\n');

results = struct();
results.config = config;
results.aversive = aversive_results;
results.reward = reward_results;
results.session_data_aversive = session_data_aversive;
results.session_data_reward = session_data_reward;
results.n_sessions_aversive = n_valid_sessions_aversive;
results.n_sessions_reward = n_valid_sessions_reward;

% NEW: Save LME data and models
results.aversive_long_data = aversive_long_data;
results.reward_long_data = reward_long_data;
results.tbl_aversive = tbl_aversive;
results.tbl_reward = tbl_reward;
if exist('lme_aversive', 'var') && ~isempty(lme_aversive)
    results.lme_aversive = lme_aversive;
end
if exist('lme_reward', 'var') && ~isempty(lme_reward)
    results.lme_reward = lme_reward;
end

save('coupling_by_period_and_behavior_results_LME.mat', 'results');

fprintf('✓ Results saved to: coupling_by_period_and_behavior_results_LME.mat\n');
fprintf('\n=== ANALYSIS COMPLETE ===\n');