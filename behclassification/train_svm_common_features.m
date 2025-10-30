function [model, accuracy, confusion_matrix] = train_svm_common_features(lstm_data, results, varargin)
% TRAIN_SVM_COMMON_FEATURES Train SVM using common features from MI and RFE
%
% Syntax:
%   [model, accuracy, cm] = train_svm_common_features(lstm_data, results)
%   [model, accuracy, cm] = train_svm_common_features(lstm_data, results, ...
%       'cv_folds', 5, 'kernel', 'rbf', 'visualize', true)
%
% Input:
%   lstm_data   - Original LSTM data structure
%   results     - Output from compare_feature_selection()
%
% Optional Name-Value Pairs:
%   'cv_folds'      - Cross-validation folds (default: 5)
%   'kernel'        - SVM kernel: 'linear', 'rbf', 'polynomial' (default: 'rbf')
%   'agg_method'    - 'mean', 'max', 'std' (default: 'mean')
%   'visualize'     - Plot confusion matrix (default: true)
%   'verbose'       - Print results (default: true)
%   'train_ratio'   - Train/test split (default: 0.8)
%
% Output:
%   model           - Trained SVM model
%   accuracy        - Classification accuracy
%   confusion_matrix - Confusion matrix [7 x 7]
%

%% Parse input
p = inputParser;
addParameter(p, 'cv_folds', 5, @isnumeric);
addParameter(p, 'kernel', 'rbf', @ischar);
addParameter(p, 'agg_method', 'mean', @ischar);
addParameter(p, 'visualize', true, @islogical);
addParameter(p, 'verbose', true, @islogical);
addParameter(p, 'train_ratio', 0.8, @isnumeric);
parse(p, varargin{:});

cv_folds = p.Results.cv_folds;
kernel_type = p.Results.kernel;
agg_method = p.Results.agg_method;
do_visualize = p.Results.visualize;
do_verbose = p.Results.verbose;
train_ratio = p.Results.train_ratio;

%% Validate input
if ~isfield(results, 'common_features') || isempty(results.common_features)
    error('No common features found. Results must have non-empty common_features field.');
end

if ~isstruct(lstm_data) || ~isfield(lstm_data, 'time_series') || ~isfield(lstm_data, 'labels')
    error('lstm_data must have fields: time_series and label');
end

%% Extract features
num_samples = length(lstm_data.labels);
Y = lstm_data.labels(:);
common_feat_indices = results.common_features;
num_common_features = length(common_feat_indices);

if do_verbose
    fprintf('\n========================================\n');
    fprintf('  SVM Classification with Common Features\n');
    fprintf('========================================\n');
    fprintf('Common features selected: %d\n', num_common_features);
    fprintf('Kernel: %s | CV Folds: %d\n\n', kernel_type, cv_folds);
end

% Extract aggregated features
X = zeros(num_samples, num_common_features);
for i = 1:num_samples
    ts = lstm_data.time_series{i};
    switch lower(agg_method)
        case 'mean', feature_vec = mean(ts, 1);
        case 'max', feature_vec = max(ts, [], 1);
        case 'std', feature_vec = std(ts, [], 1);
    end
    X(i, :) = feature_vec(common_feat_indices);
end

X(isnan(X)) = 0;

% Normalize
X = (X - mean(X, 1)) ./ (std(X, 1) + 1e-10);

%% Train/Test Split
if do_verbose
    fprintf('Creating train/test split: %.1f%% / %.1f%%\n', train_ratio*100, (1-train_ratio)*100);
end

rng(42);  % For reproducibility
n_train = floor(num_samples * train_ratio);
indices = randperm(num_samples);
train_idx = indices(1:n_train);
test_idx = indices(n_train+1:end);

X_train = X(train_idx, :);
Y_train = Y(train_idx);
X_test = X(test_idx, :);
Y_test = Y(test_idx);

%% Train SVM model
if do_verbose
    fprintf('Training SVM model (%d training samples)...\n', length(Y_train));
end

svm_template = templateSVM('Standardize', true, 'KernelFunction', kernel_type);
model = fitcecoc(X_train, Y_train, 'Learners', svm_template, 'Coding', 'onevsall');

if do_verbose
    fprintf('✓ Model trained successfully\n\n');
end

%% Predictions
Y_pred_train = predict(model, X_train);
Y_pred_test = predict(model, X_test);

%% Calculate accuracies
train_accuracy = sum(Y_pred_train == Y_train) / length(Y_train) * 100;
test_accuracy = sum(Y_pred_test == Y_test) / length(Y_test) * 100;

if do_verbose
    fprintf('Train Accuracy: %.2f%%\n', train_accuracy);
    fprintf('Test Accuracy:  %.2f%%\n\n', test_accuracy);
end

%% Cross-validation
if do_verbose
    fprintf('Running %d-fold cross-validation...\n', cv_folds);
end

cv_model = fitcecoc(X, Y, 'Learners', svm_template, 'Coding', 'onevsall');
cv_accuracy = kfoldLoss(crossval(cv_model, 'KFold', cv_folds));
cv_accuracy = (1 - cv_accuracy) * 100;

if do_verbose
    fprintf('%d-Fold CV Accuracy: %.2f%%\n\n', cv_folds, cv_accuracy);
end

%% Compute confusion matrix
confusion_matrix = confusionmat(Y_test, Y_pred_test);
accuracy = test_accuracy;

%% Print per-class metrics
if do_verbose
    fprintf('========================================\n');
    fprintf('  Per-Class Performance (Test Set)\n');
    fprintf('========================================\n');
    fprintf('Class | Precision | Recall | F1-Score\n');
    fprintf('------|-----------|--------|----------\n');
    
    for class = 1:7
        tp = confusion_matrix(class, class);
        fp = sum(confusion_matrix(:, class)) - tp;
        fn = sum(confusion_matrix(class, :)) - tp;
        
        precision = tp / (tp + fp + 1e-10) * 100;
        recall = tp / (tp + fn + 1e-10) * 100;
        f1 = 2 * precision * recall / (precision + recall + 1e-10);
        
        fprintf('%5d | %9.2f%% | %6.2f%% | %8.2f%%\n', ...
            class, precision, recall, f1);
    end
    fprintf('\n');
end

%% Visualization
if do_visualize
    visualize_results(confusion_matrix, Y_test, Y_pred_test, train_accuracy, ...
        test_accuracy, cv_accuracy, common_feat_indices);
end

end

%% =========== Visualization ===========

function visualize_results(confusion_matrix, Y_test, Y_pred_test, train_acc, test_acc, cv_acc, common_features)

figure('Position', [100 100 1400 800]);

% 1. Confusion Matrix (raw counts)
subplot(1, 2, 1);
imagesc(confusion_matrix);
colorbar;
colormap(gca, 'hot');
% caxis([0 max(confusion_matrix(:))]);
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix (Raw Counts)');
set(gca, 'XTick', 1:7, 'YTick', 1:7);

% Add text annotations
for i = 1:7
    for j = 1:7
        text(j, i, sprintf('%d', confusion_matrix(i, j)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');
    end
end

% 2. Normalized Confusion Matrix (percentages)
subplot(1, 2, 2);
confusion_norm = confusion_matrix ./ (sum(confusion_matrix, 2) + 1e-10);
imagesc(confusion_norm);
colorbar;
colormap(gca, 'cool');
% caxis([0 1]);
xlabel('Predicted Class');
ylabel('True Class');
title('Normalized Confusion Matrix (%)');
set(gca, 'XTick', 1:7, 'YTick', 1:7);

% Add percentage annotations
for i = 1:7
    for j = 1:7
        text(j, i, sprintf('%.1f%%', confusion_norm(i, j)*100), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', 'black', 'FontSize', 9, 'FontWeight', 'bold');
    end
end

sgtitle(sprintf('SVM Classification - Common Features (%d features)', length(common_features)), ...
    'FontSize', 13, 'FontWeight', 'bold');

% 3. Accuracy comparison
figure('Position', [100 950 600 400]);
accuracies = [train_acc, test_acc, cv_acc];
bar(1:3, accuracies, 'FaceColor', [0.2 0.6 0.9], 'EdgeColor', 'black', 'LineWidth', 1.5);
set(gca, 'XTickLabel', {'Train', 'Test', 'Cross-Val'});
ylabel('Accuracy (%)');
title('SVM Performance Metrics');
ylim([0 100]);
grid on;
hold on;

% Add value labels on bars
for i = 1:3
    text(i, accuracies(i) + 2, sprintf('%.2f%%', accuracies(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% 4. Per-class accuracy
figure('Position', [800 950 600 400]);
per_class_acc = zeros(1, 7);
for class = 1:7
    per_class_acc(class) = confusion_matrix(class, class) / sum(confusion_matrix(class, :)) * 100;
end
bar(1:7, per_class_acc, 'FaceColor', [0.9 0.3 0.2], 'EdgeColor', 'black', 'LineWidth', 1.5);
set(gca, 'XTickLabel', arrayfun(@num2str, 1:7, 'UniformOutput', false));
xlabel('Class');
ylabel('Recall (%)');
title('Per-Class Accuracy (Recall)');
ylim([0 100]);
grid on;
hold on;

% Add value labels
for i = 1:7
    text(i, per_class_acc(i) + 2, sprintf('%.1f%%', per_class_acc(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
end

end