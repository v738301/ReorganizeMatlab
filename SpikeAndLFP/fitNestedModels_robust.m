function [model1, model2, model3, model4, model5] = fitNestedModels_robust(spike_counts, DM1, DM2, DM3, DM4, config)
% Fit 5 nested Poisson GLMs using robust optimization with analytical gradients
% Adapted from Pillow Lab's GLMspiketools
%
% NESTED MODEL STRUCTURE:
%   Model 1: Intercept + Spike History (autoregressive baseline)
%   Model 2: Intercept + Spike History + Events (reward/aversion)
%   Model 3: Intercept + Spike History + Events + Coordinates + Spatial
%   Model 4: Intercept + Spike History + Events + Coordinates + Spatial + Speeds
%   Model 5: Intercept + Spike History + Events + Coordinates + Spatial + Speeds + Breathing (8Hz + 1.5Hz)
%
% INTERCEPT (CONSTANT TERM):
%   All models include an intercept β₀ representing the baseline log firing rate:
%   log(λ) = β₀ + X*w
%
%   Without intercept: model predicts log(λ)=0 (i.e., λ=1 spike/bin) when all
%   predictors are zero. This forces unrealistic constraints on other coefficients.
%
%   With intercept: β₀ absorbs the overall firing rate level, allowing other
%   coefficients to represent deviations from baseline.
%
% RATIONALE FOR THIS ORDER:
%   Spike history is often the strongest predictor of neural activity due to:
%   - Refractory period (can't spike immediately after a spike)
%   - Bursting dynamics (spikes come in bursts)
%   - Adaptation (firing rate decreases over time)
%
%   By starting with spike history, we establish a baseline autoregressive
%   model, then sequentially add task-related predictors to see what
%   additional variance they explain beyond intrinsic dynamics.
%
% Inputs:
%   spike_counts: [n_bins × 1] spike counts per bin
%   DM1: [n_bins × n_event_predictors] Events only (IR1ON, IR2ON, WP1ON, WP2ON, Aversive)
%   DM2: [n_bins × n_predictors] Events + Coordinates + Spatial
%   DM3: [n_bins × n_predictors] Events + Coordinates + Spatial + Speeds
%   DM4: [n_bins × n_predictors] Events + Coordinates + Spatial + Speeds + Breathing (8Hz + 1.5Hz)
%   config: Configuration struct with:
%     - history_lags: number of spike history lags
%     - use_regularization: whether to use ridge regularization
%     - lambda_grid: grid of lambda values for CV
%     - cv_folds: number of cross-validation folds
%     - max_iter: maximum iterations for optimization
%     - display_fitting: display level ('off', 'iter', etc.)
%
% Outputs:
%   model1, model2, model3, model4, model5: Model structs with:
%     - coefficients: ML/ridge weight estimates
%     - standard_errors: Standard errors from Hessian
%     - z_scores: Coefficient z-scores (coefficient / SE)
%     - p_values: Two-tailed p-values
%     - deviance_explained: Percent deviance explained
%     - log_likelihood: Model log-likelihood
%     - n_predictors: Number of predictors
%     - lambda: Regularization parameter (if using ridge)

%% Setup optimization options
opts = optimoptions('fminunc', ...
    'Algorithm', 'trust-region', ...
    'SpecifyObjectiveGradient', true, ...
    'Display', config.display_fitting, ...
    'MaxIterations', config.max_iter, ...
    'OptimalityTolerance', 1e-6, ...
    'StepTolerance', 1e-10);

%% Compute null model log-likelihood (for deviance explained)
% Null model: constant firing rate (no predictors)
lambda_null = mean(spike_counts);
log_lambda_null = log(lambda_null + 1e-10);
null_ll = sum(spike_counts * log_lambda_null - lambda_null);

%% Create spike history design matrix (used in all models)
% This captures the autoregressive structure of neural spiking
fprintf('\n--- Creating spike history design matrix ---\n');

n_bins = length(spike_counts);
n_lags = config.history_lags;
history_matrix = zeros(n_bins, n_lags);

% Build lagged spike counts
% history_matrix(:,1) = spike_counts shifted by 1 bin
% history_matrix(:,2) = spike_counts shifted by 2 bins, etc.
for lag = 1:n_lags
    history_matrix(lag+1:end, lag) = spike_counts(1:end-lag);
end

% Z-score each lag for numerical stability
% This ensures all predictors are on the same scale
for lag = 1:n_lags
    if std(history_matrix(:, lag)) > 0
        history_matrix(:, lag) = zscore(history_matrix(:, lag));
    end
end

fprintf('  Spike history: %d lags (%.0f ms)\n', n_lags, n_lags * config.bin_size * 1000);

%% Fit Model 1: Spike History only (autoregressive baseline)
fprintf('\n--- Fitting Model 1: Spike History only ---\n');

% Design matrix: intercept + spike history
% Intercept (constant term) represents baseline log firing rate
DM_model1 = [ones(n_bins, 1), history_matrix];

if config.use_regularization
    [w1, neglogli1, H1, lambda1, cv1] = fitPoissonGLM_ridge(DM_model1, spike_counts, ...
        config.lambda_grid, config.cv_folds, opts);
    model1.lambda = lambda1;
    model1.cv_results = cv1;
else
    [w1, neglogli1, H1] = fitPoissonGLM_ML(DM_model1, spike_counts, [], opts);
    model1.lambda = 0;
end

% Store results
model1.coefficients = w1;
model1.log_likelihood = -neglogli1;
model1.deviance_explained = 100 * (1 - (-neglogli1 / null_ll));
model1.n_predictors = length(w1);
model1.hessian = H1;
model1.standard_errors = sqrt(diag(inv(H1)));
model1.z_scores = w1 ./ model1.standard_errors;
model1.p_values = 2 * normcdf(-abs(model1.z_scores));

% Model selection criteria
model1.AIC = 2 * neglogli1 + 2 * model1.n_predictors;
model1.BIC = 2 * neglogli1 + model1.n_predictors * log(n_bins);

% No comparison for first model (it's the baseline)
model1.LRT_vs_previous = NaN;
model1.LRT_df = NaN;
model1.LRT_p_value = NaN;

fprintf('  Deviance explained: %.2f%%\n', model1.deviance_explained);
fprintf('  Log-likelihood: %.2f\n', model1.log_likelihood);
if config.use_regularization
    fprintf('  Lambda: %.2e\n', model1.lambda);
end

%% Fit Model 2: Spike History + Events
fprintf('\n--- Fitting Model 2: Spike History + Events ---\n');

% Design matrix: intercept + spike history + event predictors
% DM1 contains event predictors (IR1ON, IR2ON, Aversive with basis functions)
DM_model2 = [ones(n_bins, 1), history_matrix, DM1];

if config.use_regularization
    [w2, neglogli2, H2, lambda2, cv2] = fitPoissonGLM_ridge(DM_model2, spike_counts, ...
        config.lambda_grid, config.cv_folds, opts);
    model2.lambda = lambda2;
    model2.cv_results = cv2;
else
    [w2, neglogli2, H2] = fitPoissonGLM_ML(DM_model2, spike_counts, [], opts);
    model2.lambda = 0;
end

model2.coefficients = w2;
model2.log_likelihood = -neglogli2;
model2.deviance_explained = 100 * (1 - (-neglogli2 / null_ll));
model2.n_predictors = length(w2);
model2.hessian = H2;
model2.standard_errors = sqrt(diag(inv(H2)));
model2.z_scores = w2 ./ model2.standard_errors;
model2.p_values = 2 * normcdf(-abs(model2.z_scores));

% Model selection criteria
model2.AIC = 2 * neglogli2 + 2 * model2.n_predictors;
model2.BIC = 2 * neglogli2 + model2.n_predictors * log(n_bins);

% Likelihood Ratio Test vs Model 1
model2.LRT_vs_previous = -2 * (model1.log_likelihood - model2.log_likelihood);
model2.LRT_df = model2.n_predictors - model1.n_predictors;
model2.LRT_p_value = 1 - chi2cdf(model2.LRT_vs_previous, model2.LRT_df);

fprintf('  Deviance explained: %.2f%%\n', model2.deviance_explained);
fprintf('  Additional variance from Events: %.2f%%\n', ...
    model2.deviance_explained - model1.deviance_explained);
fprintf('  Log-likelihood: %.2f\n', model2.log_likelihood);
if config.use_regularization
    fprintf('  Lambda: %.2e\n', model2.lambda);
end

%% Fit Model 3: Spike History + Events + Coordinates + Spatial
fprintf('\n--- Fitting Model 3: Spike History + Events + Coordinates + Spatial ---\n');

% Design matrix: intercept + spike history + events + coordinates + spatial
% DM2 = [DM1, Coordinates, Spatial] (events + coordinates + spatial kernels)
DM_model3 = [ones(n_bins, 1), history_matrix, DM2];

if config.use_regularization
    [w3, neglogli3, H3, lambda3, cv3] = fitPoissonGLM_ridge(DM_model3, spike_counts, ...
        config.lambda_grid, config.cv_folds, opts);
    model3.lambda = lambda3;
    model3.cv_results = cv3;
else
    [w3, neglogli3, H3] = fitPoissonGLM_ML(DM_model3, spike_counts, [], opts);
    model3.lambda = 0;
end

model3.coefficients = w3;
model3.log_likelihood = -neglogli3;
model3.deviance_explained = 100 * (1 - (-neglogli3 / null_ll));
model3.n_predictors = length(w3);
model3.hessian = H3;
model3.standard_errors = sqrt(diag(inv(H3)));
model3.z_scores = w3 ./ model3.standard_errors;
model3.p_values = 2 * normcdf(-abs(model3.z_scores));

% Model selection criteria
model3.AIC = 2 * neglogli3 + 2 * model3.n_predictors;
model3.BIC = 2 * neglogli3 + model3.n_predictors * log(n_bins);

% Likelihood Ratio Test vs Model 2
model3.LRT_vs_previous = -2 * (model2.log_likelihood - model3.log_likelihood);
model3.LRT_df = model3.n_predictors - model2.n_predictors;
model3.LRT_p_value = 1 - chi2cdf(model3.LRT_vs_previous, model3.LRT_df);

fprintf('  Deviance explained: %.2f%%\n', model3.deviance_explained);
fprintf('  Additional variance from Coordinates + Spatial: %.2f%%\n', ...
    model3.deviance_explained - model2.deviance_explained);
fprintf('  Log-likelihood: %.2f\n', model3.log_likelihood);
if config.use_regularization
    fprintf('  Lambda: %.2e\n', model3.lambda);
end

%% Fit Model 4: Spike History + Events + Coordinates + Spatial + Speeds
fprintf('\n--- Fitting Model 4: + Speeds ---\n');

% Design matrix: intercept + spike history + events + coordinates + spatial + speeds
% DM3 = [DM2, Speeds] (events + coordinates + spatial + speeds)
DM_model4 = [ones(n_bins, 1), history_matrix, DM3];

if config.use_regularization
    [w4, neglogli4, H4, lambda4, cv4] = fitPoissonGLM_ridge(DM_model4, spike_counts, ...
        config.lambda_grid, config.cv_folds, opts);
    model4.lambda = lambda4;
    model4.cv_results = cv4;
else
    [w4, neglogli4, H4] = fitPoissonGLM_ML(DM_model4, spike_counts, [], opts);
    model4.lambda = 0;
end

model4.coefficients = w4;
model4.log_likelihood = -neglogli4;
model4.deviance_explained = 100 * (1 - (-neglogli4 / null_ll));
model4.n_predictors = length(w4);
model4.hessian = H4;
model4.standard_errors = sqrt(diag(inv(H4)));
model4.z_scores = w4 ./ model4.standard_errors;
model4.p_values = 2 * normcdf(-abs(model4.z_scores));

% Model selection criteria
model4.AIC = 2 * neglogli4 + 2 * model4.n_predictors;
model4.BIC = 2 * neglogli4 + model4.n_predictors * log(n_bins);

% Likelihood Ratio Test vs Model 3
model4.LRT_vs_previous = -2 * (model3.log_likelihood - model4.log_likelihood);
model4.LRT_df = model4.n_predictors - model3.n_predictors;
model4.LRT_p_value = 1 - chi2cdf(model4.LRT_vs_previous, model4.LRT_df);

fprintf('  Deviance explained: %.2f%%\n', model4.deviance_explained);
fprintf('  Additional variance from Speeds: %.2f%%\n', ...
    model4.deviance_explained - model3.deviance_explained);
fprintf('  Log-likelihood: %.2f\n', model4.log_likelihood);
if config.use_regularization
    fprintf('  Lambda: %.2e\n', model4.lambda);
end

%% Fit Model 5: Full model (+ Breathing)
fprintf('\n--- Fitting Model 5: Full model (+ Breathing) ---\n');

% Design matrix: intercept + spike history + events + coordinates + spatial + speeds + breathing
% DM4 = [DM3, Breathing_8Hz, Breathing_1.5Hz]
DM_model5 = [ones(n_bins, 1), history_matrix, DM4];

if config.use_regularization
    [w5, neglogli5, H5, lambda5, cv5] = fitPoissonGLM_ridge(DM_model5, spike_counts, ...
        config.lambda_grid, config.cv_folds, opts);
    model5.lambda = lambda5;
    model5.cv_results = cv5;
else
    [w5, neglogli5, H5] = fitPoissonGLM_ML(DM_model5, spike_counts, [], opts);
    model5.lambda = 0;
end

model5.coefficients = w5;
model5.log_likelihood = -neglogli5;
model5.deviance_explained = 100 * (1 - (-neglogli5 / null_ll));
model5.n_predictors = length(w5);
model5.hessian = H5;
model5.standard_errors = sqrt(diag(inv(H5)));
model5.z_scores = w5 ./ model5.standard_errors;
model5.p_values = 2 * normcdf(-abs(model5.z_scores));

% Model selection criteria
model5.AIC = 2 * neglogli5 + 2 * model5.n_predictors;
model5.BIC = 2 * neglogli5 + model5.n_predictors * log(n_bins);

% Likelihood Ratio Test vs Model 4
model5.LRT_vs_previous = -2 * (model4.log_likelihood - model5.log_likelihood);
model5.LRT_df = model5.n_predictors - model4.n_predictors;
model5.LRT_p_value = 1 - chi2cdf(model5.LRT_vs_previous, model5.LRT_df);

fprintf('  Deviance explained: %.2f%%\n', model5.deviance_explained);
fprintf('  Additional variance from Breathing: %.2f%%\n', ...
    model5.deviance_explained - model4.deviance_explained);
fprintf('  Log-likelihood: %.2f\n', model5.log_likelihood);
if config.use_regularization
    fprintf('  Lambda: %.2e\n', model5.lambda);
end

%% Summary
fprintf('\n✓ All 5 nested models fitted successfully\n');
fprintf('\n=== NESTED MODEL COMPARISON ===\n');
fprintf('%-30s %8s %8s %8s %10s %8s\n', 'Model', 'Dev.Exp', 'AIC', 'BIC', 'LRT (χ²)', 'p-value');
fprintf('%s\n', repmat('-', 1, 90));
fprintf('%-30s %7.2f%% %8.1f %8.1f %10s %8s\n', ...
    '1. History', model1.deviance_explained, model1.AIC, model1.BIC, '-', '-');
fprintf('%-30s %7.2f%% %8.1f %8.1f %10.2f %8.4f %s\n', ...
    '2. + Events', model2.deviance_explained, model2.AIC, model2.BIC, ...
    model2.LRT_vs_previous, model2.LRT_p_value, ...
    getSignificanceStr(model2.LRT_p_value));
fprintf('%-30s %7.2f%% %8.1f %8.1f %10.2f %8.4f %s\n', ...
    '3. + Coord + Spatial', model3.deviance_explained, model3.AIC, model3.BIC, ...
    model3.LRT_vs_previous, model3.LRT_p_value, ...
    getSignificanceStr(model3.LRT_p_value));
fprintf('%-30s %7.2f%% %8.1f %8.1f %10.2f %8.4f %s\n', ...
    '4. + Speeds', model4.deviance_explained, model4.AIC, model4.BIC, ...
    model4.LRT_vs_previous, model4.LRT_p_value, ...
    getSignificanceStr(model4.LRT_p_value));
fprintf('%-30s %7.2f%% %8.1f %8.1f %10.2f %8.4f %s\n', ...
    '5. + Breathing', model5.deviance_explained, model5.AIC, model5.BIC, ...
    model5.LRT_vs_previous, model5.LRT_p_value, ...
    getSignificanceStr(model5.LRT_p_value));
fprintf('%s\n', repmat('=', 1, 90));
fprintf('\nModel Selection:\n');
[~, aic_best] = min([model1.AIC, model2.AIC, model3.AIC, model4.AIC, model5.AIC]);
[~, bic_best] = min([model1.BIC, model2.BIC, model3.BIC, model4.BIC, model5.BIC]);
fprintf('  Best by AIC: Model %d\n', aic_best);
fprintf('  Best by BIC: Model %d (more conservative)\n', bic_best);

end

function sig_str = getSignificanceStr(p_value)
    % Convert p-value to significance string
    if isnan(p_value)
        sig_str = '';
    elseif p_value < 0.001
        sig_str = '***';
    elseif p_value < 0.01
        sig_str = '**';
    elseif p_value < 0.05
        sig_str = '*';
    else
        sig_str = 'n.s.';
    end

end
