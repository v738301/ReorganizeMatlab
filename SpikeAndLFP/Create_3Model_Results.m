% Create_3Model_Results.m
%
% Create 3-model nested results from existing 4-model results
%
% Extracts:
%   Model 1 (3-model) = Model 2 (4-model): History + Events
%   Model 2 (3-model) = Model 3 (4-model): History + Events + Speed
%   Model 3 (3-model) = Model 4 (4-model): History + Events + Speed + Breathing
%
% This skips the "History only" baseline from the 4-model approach

clear; clc;

fprintf('=== Creating 3-Model Results from 4-Model Results ===\n\n');

%% Load existing 4-model results
input_file = 'Unit_GLM_Nested_Results.mat';
output_file = 'Unit_GLM_Nested_Results_3models.mat';

fprintf('Loading 4-model results: %s\n', input_file);
load(input_file, 'all_results', 'config');

n_units = length(all_results);
fprintf('  Found %d units\n\n', n_units);

%% Transform results
fprintf('Transforming to 3-model structure...\n');

% Create new structure array
all_results_3models = struct();

for i = 1:n_units
    % Copy metadata fields
    all_results_3models(i).session_name = all_results(i).session_name;
    all_results_3models(i).session_type = all_results(i).session_type;
    all_results_3models(i).unit_idx = all_results(i).unit_idx;
    all_results_3models(i).mean_firing_rate = all_results(i).mean_firing_rate;
    all_results_3models(i).predictor_info = all_results(i).predictor_info;

    % Map 4-model results to 3-model structure
    % Old model2 (History+Events) → New model1
    all_results_3models(i).model1 = all_results(i).model2;

    % Old model3 (History+Events+Speed) → New model2
    all_results_3models(i).model2 = all_results(i).model3;

    % Old model4 (History+Events+Speed+Breathing) → New model3
    all_results_3models(i).model3 = all_results(i).model4;

    % Update LRT comparisons
    % Model 1 has no previous model
    all_results_3models(i).model1.LRT_vs_previous = NaN;
    all_results_3models(i).model1.LRT_df = NaN;
    all_results_3models(i).model1.LRT_p_value = NaN;

    % Model 2 compares to Model 1 (was comparing model3 to model2)
    % LRT statistic is already correct (comparing same models)

    % Model 3 compares to Model 2 (was comparing model4 to model3)
    % LRT statistic is already correct (comparing same models)
end

fprintf('  ✓ Transformed %d units\n\n', n_units);

%% Update config
config_3models = config;
config_3models.n_models = 3;
config_3models.model_names = {
    '1. History + Events (baseline)'
    '2. + Speed'
    '3. + Breathing'
};

%% Save results
fprintf('Saving 3-model results: %s\n', output_file);
all_results = all_results_3models;
config = config_3models;
save(output_file, 'all_results', 'config', '-v7.3');
fprintf('  ✓ Saved\n\n');

%% Summary statistics
fprintf('=== Summary Statistics ===\n');

% Extract deviance explained for each model
dev_exp_1 = arrayfun(@(x) x.model1.deviance_explained, all_results);
dev_exp_2 = arrayfun(@(x) x.model2.deviance_explained, all_results);
dev_exp_3 = arrayfun(@(x) x.model3.deviance_explained, all_results);

fprintf('Deviance Explained (across %d units):\n', length(all_results));
fprintf('  Model 1 (History+Events):     %.2f ± %.2f%%\n', mean(dev_exp_1), std(dev_exp_1));
fprintf('  Model 2 (+ Speed):            %.2f ± %.2f%%\n', mean(dev_exp_2), std(dev_exp_2));
fprintf('  Model 3 (+ Breathing):        %.2f ± %.2f%%\n', mean(dev_exp_3), std(dev_exp_3));

% Additional variance explained
add_var_speed = dev_exp_2 - dev_exp_1;
add_var_breathing = dev_exp_3 - dev_exp_2;

fprintf('\nAdditional Variance Explained:\n');
fprintf('  Speed:     %.2f ± %.2f%% (range: %.2f to %.2f%%)\n', ...
    mean(add_var_speed), std(add_var_speed), min(add_var_speed), max(add_var_speed));
fprintf('  Breathing: %.2f ± %.2f%% (range: %.2f to %.2f%%)\n', ...
    mean(add_var_breathing), std(add_var_breathing), min(add_var_breathing), max(add_var_breathing));

% Count significant LRTs (check if field exists)
if isfield(all_results(1).model2, 'LRT_p_value')
    sig_speed = sum(arrayfun(@(x) x.model2.LRT_p_value < 0.05, all_results));
    sig_breathing = sum(arrayfun(@(x) x.model3.LRT_p_value < 0.05, all_results));

    fprintf('\nSignificant LRTs (p < 0.05):\n');
    fprintf('  Speed:     %d/%d (%.1f%%)\n', sig_speed, length(all_results), ...
        100*sig_speed/length(all_results));
    fprintf('  Breathing: %d/%d (%.1f%%)\n', sig_breathing, length(all_results), ...
        100*sig_breathing/length(all_results));
else
    fprintf('\nWarning: LRT p-values not found in original models\n');
end

% Model selection by AIC/BIC (check if fields exist)
if isfield(all_results(1).model1, 'AIC') && isfield(all_results(1).model1, 'BIC')
    aic_vals = arrayfun(@(x) [x.model1.AIC, x.model2.AIC, x.model3.AIC], all_results, 'UniformOutput', false);
    bic_vals = arrayfun(@(x) [x.model1.BIC, x.model2.BIC, x.model3.BIC], all_results, 'UniformOutput', false);

    aic_best = cellfun(@(x) find(x == min(x), 1), aic_vals);
    bic_best = cellfun(@(x) find(x == min(x), 1), bic_vals);

    fprintf('\nBest Model by AIC:\n');
    for m = 1:3
        fprintf('  Model %d: %d/%d (%.1f%%)\n', m, sum(aic_best == m), length(all_results), ...
            100*sum(aic_best == m)/length(all_results));
    end

    fprintf('\nBest Model by BIC:\n');
    for m = 1:3
        fprintf('  Model %d: %d/%d (%.1f%%)\n', m, sum(bic_best == m), length(all_results), ...
            100*sum(bic_best == m)/length(all_results));
    end
else
    fprintf('\nWarning: AIC/BIC not found in original models\n');
end

fprintf('\n✓ Conversion complete!\n');
