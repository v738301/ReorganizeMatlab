function [w, neglogli, H, lambda_opt, cv_results] = fitPoissonGLM_ridge(X, y, lambda_grid, cv_folds, opts)
% [w, neglogli, H, lambda_opt, cv_results] = fitPoissonGLM_ridge(X, y, lambda_grid, cv_folds, opts)
%
% Ridge regression for Poisson GLM with cross-validated regularization.
% Adapted from Pillow Lab's GLMspiketools.
%
% Inputs:
%   X           - [n_bins × n_predictors] design matrix
%   y           - [n_bins × 1] spike counts
%   lambda_grid - [1 × n_lambda] vector of regularization values to test
%                 (default: 2.^(-5:15))
%   cv_folds    - Number of cross-validation folds (default: 5)
%   opts        - Optimization options (optional)
%
% Outputs:
%   w           - [n_predictors × 1] ridge weight estimates
%   neglogli    - Negative log-likelihood at ridge estimate
%   H           - [n_predictors × n_predictors] Hessian matrix
%   lambda_opt  - Optimal regularization parameter selected by CV
%   cv_results  - Structure with CV results for all lambda values
%
% Example:
%   [w, neglogli, H, lambda] = fitPoissonGLM_ridge(X, spike_counts);
%   fprintf('Optimal lambda: %.2e\n', lambda);
%   SE = sqrt(diag(inv(H)));
%
% See also: fitPoissonGLM_ML, negLogLikelihood_Poisson_grad

% Set defaults
if nargin < 3 || isempty(lambda_grid)
    lambda_grid = 2.^(-5:15);  % Test values from 2^-5 to 2^15
end

if nargin < 4 || isempty(cv_folds)
    cv_folds = 5;
end

if nargin < 5 || isempty(opts)
    opts = optimoptions('fminunc', ...
        'Algorithm', 'trust-region', ...
        'SpecifyObjectiveGradient', true, ...
        'Display', 'off', ...
        'MaxIterations', 500, ...
        'OptimalityTolerance', 1e-6);
end

% Number of samples and predictors
[n_bins, n_predictors] = size(X);
n_lambda = length(lambda_grid);

% Check if cross-validation should be skipped
if cv_folds == 1
    % Skip cross-validation: use first lambda value
    lambda_opt = lambda_grid(1);
    fprintf('Skipping cross-validation (cv_folds = 1)\n');
    fprintf('Using lambda = %.2e\n', lambda_opt);

    % Initialize empty CV results
    cv_logli = [];
    cv_logli_mean = [];
    cv_logli_se = [];
    best_idx = 1;
else
    % Initialize CV results
    cv_logli = zeros(n_lambda, cv_folds);
    cv_logli_mean = zeros(n_lambda, 1);
    cv_logli_se = zeros(n_lambda, 1);

    % Create CV partition indices
    cv_indices = crossvalind('Kfold', n_bins, cv_folds);

    % Cross-validation loop
    fprintf('Running %d-fold cross-validation for %d lambda values...\n', cv_folds, n_lambda);

    for i = 1:n_lambda
        lambda = lambda_grid(i);

        for fold = 1:cv_folds
            % Split into train and test
            test_idx = (cv_indices == fold);
            train_idx = ~test_idx;

            X_train = X(train_idx, :);
            y_train = y(train_idx);
            X_test = X(test_idx, :);
            y_test = y(test_idx);

            % Fit on training data
            w0 = zeros(n_predictors, 1);
            lossfun = @(w) negLogLikelihood_Poisson_grad(w, X_train, y_train, lambda);
            w_train = fminunc(lossfun, w0, opts);

            % Evaluate on test data (no regularization penalty for test likelihood)
            Xw_test = X_test * w_train;
            rate_test = exp(Xw_test);
            test_logli = sum(y_test .* Xw_test - rate_test);  % Positive log-likelihood

            cv_logli(i, fold) = test_logli;
        end

        % Compute mean and SE across folds
        cv_logli_mean(i) = mean(cv_logli(i, :));
        cv_logli_se(i) = std(cv_logli(i, :)) / sqrt(cv_folds);

        if mod(i, 5) == 0 || i == n_lambda
            fprintf('  Lambda %d/%d: %.2e, CV log-likelihood: %.2f ± %.2f\n', ...
                i, n_lambda, lambda, cv_logli_mean(i), cv_logli_se(i));
        end
    end

    % Select optimal lambda (maximum CV log-likelihood)
    [max_logli, best_idx] = max(cv_logli_mean);
    lambda_opt = lambda_grid(best_idx);

    fprintf('\n✓ Optimal lambda: %.2e (CV log-likelihood: %.2f ± %.2f)\n', ...
        lambda_opt, cv_logli_mean(best_idx), cv_logli_se(best_idx));
end

% Fit final model on all data with optimal lambda
fprintf('Fitting final model with optimal lambda...\n');
w0 = zeros(n_predictors, 1);
lossfun = @(w) negLogLikelihood_Poisson_grad(w, X, y, lambda_opt);

% Use more verbose options for final fit
opts_final = opts;
[w, neglogli, ~, ~, ~, H] = fminunc(lossfun, w0, opts_final);

fprintf('✓ Final model fitted successfully\n\n');

% Store CV results
cv_results.lambda_grid = lambda_grid;
cv_results.cv_logli = cv_logli;
cv_results.cv_logli_mean = cv_logli_mean;
cv_results.cv_logli_se = cv_logli_se;
cv_results.lambda_opt = lambda_opt;
cv_results.best_idx = best_idx;

end
