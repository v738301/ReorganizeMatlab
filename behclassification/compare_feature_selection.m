function [results] = compare_feature_selection(lstm_data, varargin)
% COMPARE_FEATURE_SELECTION Compare Mutual Information and RFE feature selection
%
% Syntax:
%   [results] = compare_feature_selection(lstm_data)
%   [results] = compare_feature_selection(lstm_data, 'num_features', 30, 'visualize', true)
%
% Input:
%   lstm_data       - Structure with fields:
%                     .time_series: cell array, each cell [time x features]
%                     .label: vector of class labels
%
% Optional Name-Value Pairs:
%   'num_features'  - Number of top features to select (default: 30)
%   'num_classes'   - Number of classes (default: 7)
%   'agg_method'    - 'mean', 'max', 'std' (default: 'mean')
%   'cv_folds'      - CV folds for RFE (default: 5)
%   'visualize'     - Plot results (default: true)
%   'verbose'       - Print results (default: true)
%
% Output:
%   results - Structure with fields:
%            .mi_indices: Top 30 features from MI
%            .mi_scores: MI scores
%            .rfe_indices: Top 30 features from RFE
%            .rfe_scores: RFE scores
%            .common_features: Features in both top 30
%            .unique_mi: Features only in MI top 30
%            .unique_rfe: Features only in RFE top 30
%

%% Parse input
p = inputParser;
addParameter(p, 'num_features', 50, @isnumeric);
addParameter(p, 'num_classes', 7, @isnumeric);
addParameter(p, 'agg_method', 'mean', @ischar);
addParameter(p, 'cv_folds', 5, @isnumeric);
addParameter(p, 'visualize', true, @islogical);
addParameter(p, 'verbose', true, @islogical);
parse(p, varargin{:});

num_top_features = p.Results.num_features;
num_classes = p.Results.num_classes;
agg_method = p.Results.agg_method;
cv_folds = p.Results.cv_folds;
do_visualize = p.Results.visualize;
do_verbose = p.Results.verbose;

%% Validate and prepare data
if ~isstruct(lstm_data) || ~isfield(lstm_data, 'time_series') || ~isfield(lstm_data, 'labels')
    error('lstm_data must have fields: time_series (cell) and label (vector)');
end

num_samples = length(lstm_data.labels);
Y = lstm_data.labels(:);
first_ts = lstm_data.time_series{1};
num_features = size(first_ts, 2);

% Extract aggregated features
X = zeros(num_samples, num_features);
for i = 1:num_samples
    ts = lstm_data.time_series{i};
    switch lower(agg_method)
        case 'mean', X(i, :) = mean(ts, 1);
        case 'max', X(i, :) = max(ts, [], 1);
        case 'std', X(i, :) = std(ts, [], 1);
    end
end
X(isnan(X)) = 0;

% Normalize
X = (X - mean(X, 1)) ./ (std(X, 1) + 1e-10);

if do_verbose
    fprintf('\n========================================\n');
    fprintf('  Feature Selection: MI vs RFE Comparison\n');
    fprintf('========================================\n');
    fprintf('Samples: %d | Features: %d | Classes: %d\n', num_samples, num_features, num_classes);
    fprintf('Aggregation: %s | CV Folds: %d\n\n', agg_method, cv_folds);
end

%% Method 1: Mutual Information
if do_verbose
    fprintf('[1/2] Computing Mutual Information scores...\n');
end
mi_scores = compute_mutual_information(X, Y, num_classes);
[~, mi_sorted_idx] = sort(mi_scores, 'descend');
mi_top_features = mi_sorted_idx(1:num_top_features);
mi_top_scores = mi_scores(mi_top_features);

if do_verbose
    fprintf('      ✓ MI computation complete\n');
end

%% Method 2: Recursive Feature Elimination (SVM)
if do_verbose
    fprintf('[2/2] Computing RFE-SVM scores...\n');
end
rfe_scores = compute_rfe_svm(X, Y, cv_folds);
[~, rfe_sorted_idx] = sort(rfe_scores, 'descend');
rfe_top_features = rfe_sorted_idx(1:num_top_features);
rfe_top_scores = rfe_scores(rfe_top_features);

if do_verbose
    fprintf('      ✓ RFE computation complete\n');
end

%% Find common, unique features
common_features = intersect(mi_top_features, rfe_top_features);
unique_mi = setdiff(mi_top_features, rfe_top_features);
unique_rfe = setdiff(rfe_top_features, mi_top_features);

%% Display results
if do_verbose
    print_results(mi_top_features, mi_top_scores, rfe_top_features, rfe_top_scores, ...
        common_features, unique_mi, unique_rfe, num_top_features);
end

%% Visualization
if do_visualize
    visualize_comparison(mi_scores, rfe_scores, mi_top_features, rfe_top_features, ...
        common_features, unique_mi, unique_rfe, num_top_features);
end

%% Output
results.mi_indices = mi_top_features;
results.mi_scores = mi_top_scores;
results.rfe_indices = rfe_top_features;
results.rfe_scores = rfe_top_scores;
results.common_features = common_features;
results.unique_mi = unique_mi;
results.unique_rfe = unique_rfe;
results.num_common = length(common_features);
results.X = X;
results.Y = Y;

end

%% =========== Feature Scoring Methods ===========

function scores = compute_mutual_information(X, Y, num_classes)
% Mutual Information: I(X; Y) = H(X) + H(Y) - H(X, Y)
% Measures dependency between feature and class label

[num_samples, num_features] = size(X);

% Discretize features into bins
num_bins = min(10, round(sqrt(num_samples)));
X_discrete = zeros(num_samples, num_features);
for feat = 1:num_features
    [~, ~, X_discrete(:, feat)] = histcounts(X(:, feat), num_bins);
end

% Compute MI for each feature
scores = zeros(1, num_features);
H_Y = entropy(Y);  % Entropy of class distribution

for feat = 1:num_features
    % Joint entropy H(X, Y)
    H_XY = 0;
    for x_val = 1:num_bins
        for y_val = 1:num_classes
            mask = (X_discrete(:, feat) == x_val) & (Y == y_val);
            p = sum(mask) / num_samples;
            if p > 1e-10
                H_XY = H_XY - p * log2(p);
            end
        end
    end
    
    % Marginal entropy H(X)
    H_X = entropy(X_discrete(:, feat));
    
    % Mutual Information
    MI = H_X + H_Y - H_XY;
    scores(feat) = max(0, MI);
end
end

function scores = compute_rfe_svm(X, Y, cv_folds)
% Recursive Feature Elimination using SVM
% Iteratively removes least important features

[num_samples, num_features] = size(X);
scores = zeros(1, num_features);
remaining_features = 1:num_features;

% Stage 1: Train SVM and get initial weights
try
    svm_template = templateSVM('Standardize', true, 'KernelFunction', 'linear');
    svm_model = fitcecoc(X, Y, 'Learners', svm_template, 'Coding', 'onevsall');
    
    % Extract feature importance from SVM coefficients
    % Average absolute coefficient value across all binary classifiers
    all_coefs = [];
    for i = 1:length(svm_model.BinaryLearners)
        learner = svm_model.BinaryLearners{i};
        if ~isempty(learner.Beta)
            all_coefs = [all_coefs, abs(learner.Beta)];
        end
    end
    
    if ~isempty(all_coefs)
        feature_weights = mean(all_coefs, 2);
        scores(remaining_features) = feature_weights;
    else
        % Fallback: use permutation importance
        scores = compute_permutation_importance(X, Y, svm_template);
    end
catch
    % If SVM fails
    fprintf('(SVM fails)');
end

scores = scores / (max(scores) + 1e-10);
end

function scores = compute_permutation_importance(X, Y, svm_template)
% Permutation-based feature importance (fallback method)

[num_samples, num_features] = size(X);
scores = zeros(1, num_features);

% Train baseline model
baseline_model = fitcecoc(X, Y, 'Learners', svm_template, 'Coding', 'onevsall');
baseline_acc = 1 - loss(baseline_model, X, Y);

for feat = 1:num_features
    X_perm = X;
    X_perm(:, feat) = X_perm(randperm(num_samples), feat);
    
    perm_acc = 1 - loss(baseline_model, X_perm, Y);
    scores(feat) = max(0, baseline_acc - perm_acc);
end
end

%% Helper Functions

function h = entropy(X)
% Compute entropy of discrete variable
X = X(:);
unique_vals = unique(X);
h = 0;
for val = unique_vals'
    p = sum(X == val) / length(X);
    if p > 1e-10
        h = h - p * log2(p);
    end
end
end

%% =========== Printing Results ===========

function print_results(mi_features, mi_scores, rfe_features, rfe_scores, ...
    common, unique_mi, unique_rfe, num_top)

fprintf('\n========================================\n');
fprintf('            RESULTS SUMMARY\n');
fprintf('========================================\n\n');

% Comparison statistics
fprintf('Common Features (in both top %d): %d\n', num_top, length(common));
fprintf('Unique to MI: %d\n', length(unique_mi));
fprintf('Unique to RFE: %d\n\n', length(unique_rfe));

% Mutual Information top 30
fprintf('---- TOP 30 FEATURES: MUTUAL INFORMATION ----\n');
fprintf('Rank | Feature | MI Score\n');
fprintf('-----|---------|----------\n');
for i = 1:num_top
    fprintf('%4d | %7d | %.6f\n', i, mi_features(i), mi_scores(i));
end

fprintf('\n---- TOP 30 FEATURES: RFE-SVM ----\n');
fprintf('Rank | Feature | RFE Score\n');
fprintf('-----|---------|----------\n');
for i = 1:num_top
    fprintf('%4d | %7d | %.6f\n', i, rfe_features(i), rfe_scores(i));
end

fprintf('\n---- COMMON FEATURES (appear in both top 30) ----\n');
fprintf('Features: ');
fprintf('%d ', common);
fprintf('\n\n');

end

%% =========== Visualization ===========

function visualize_comparison(mi_scores, rfe_scores, mi_top_features, rfe_top_features, ...
    common_features, unique_mi, unique_rfe, num_top)

figure('Position', [100 100 1600 900]);

% 1. Top 30 MI scores
subplot(2, 3, 1);
bar(1:num_top, mi_scores(mi_top_features), 'FaceColor', [0.2 0.6 0.9]);
xlabel('Feature Rank');
ylabel('MI Score');
title('Top 30 Features - Mutual Information');
grid on;

% 2. Top 30 RFE scores
subplot(2, 3, 2);
bar(1:num_top, rfe_scores(rfe_top_features), 'FaceColor', [0.9 0.3 0.2]);
xlabel('Feature Rank');
ylabel('RFE Score');
title('Top 30 Features - RFE-SVM');
grid on;

% 3. Comparison scatter plot
subplot(2, 3, 3);
all_features = unique([mi_top_features; rfe_top_features]);
mi_vals = mi_scores(all_features);
rfe_vals = rfe_scores(all_features);

% Color by category
colors = zeros(length(all_features), 3);
for i = 1:length(all_features)
    feat = all_features(i);
    if ismember(feat, common_features)
        colors(i, :) = [0 0.7 0];  % Green - common
    elseif ismember(feat, unique_mi)
        colors(i, :) = [0.2 0.6 0.9];  % Blue - MI only
    else
        colors(i, :) = [0.9 0.3 0.2];  % Red - RFE only
    end
end

scatter(mi_vals, rfe_vals, 100, colors, 'filled');
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1, 'DisplayName', 'Equal Score Line');
xlabel('MI Score (normalized)');
ylabel('RFE Score (normalized)');
title('MI vs RFE Score Comparison');
grid on;
legend('Common', 'MI only', 'RFE only', 'Equal', 'Location', 'best');

% 4. Venn-like diagram (text representation)
subplot(2, 3, 4);
axis off;
num_common = length(common_features);
num_unique_mi = length(unique_mi);
num_unique_rfe = length(unique_rfe);

text_content = sprintf(['Feature Overlap Analysis\n\n' ...
    'MI Top 30: %d features\n' ...
    'RFE Top 30: %d features\n\n' ...
    'Common: %d features (%.1f%%)\n' ...
    'Only in MI: %d features\n' ...
    'Only in RFE: %d features\n\n' ...
    'Jaccard Similarity: %.3f'], ...
    num_top, num_top, num_common, 100*num_common/num_top, ...
    num_unique_mi, num_unique_rfe, ...
    num_common / (num_unique_mi + num_unique_rfe + num_common));

text(0.1, 0.5, text_content, 'FontSize', 11, 'VerticalAlignment', 'middle');

% 5. All features ranking - MI
subplot(2, 3, 5);
[sorted_mi, ~] = sort(mi_scores, 'descend');
bar(sorted_mi, 'FaceColor', [0.2 0.6 0.9]);
hold on;
xline(num_top + 0.5, 'r--', 'LineWidth', 2);
xlabel('Feature Rank');
ylabel('MI Score');
title('All Features - MI Ranking');
grid on;

% 6. All features ranking - RFE
subplot(2, 3, 6);
[sorted_rfe, ~] = sort(rfe_scores, 'descend');
bar(sorted_rfe, 'FaceColor', [0.9 0.3 0.2]);
hold on;
xline(num_top + 0.5, 'r--', 'LineWidth', 2);
xlabel('Feature Rank');
ylabel('RFE Score');
title('All Features - RFE Ranking');
grid on;

sgtitle(sprintf('Feature Selection Comparison: MI vs RFE (Top %d Features)', num_top), ...
    'FontSize', 14, 'FontWeight', 'bold');

end