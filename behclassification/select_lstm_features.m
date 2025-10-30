function [selected_features] = select_lstm_features(lstm_data, varargin)
% SELECT_LSTM_FEATURES Selects top discriminative features from LSTM time series data
%
% Syntax:
%   [selected_features] = select_lstm_features(lstm_data)
%   [selected_features] = select_lstm_features(lstm_data, 'num_features', 30, 'kl_weight', 0.5)
%
% Input:
%   lstm_data       - Structure with fields:
%                     .time_series: cell array, each cell contains matrix [time x features]
%                     .labels: vector of class labels (1-7)
%
% Optional Name-Value Pairs:
%   'num_features'  - Number of top features to select (default: 30)
%   'num_classes'   - Number of classes (default: 7)
%   'kl_weight'     - Weight for KL divergence score (default: 0.5)
%   'roc_weight'    - Weight for ROC score (default: 0.5)
%   'agg_method'    - Aggregation method: 'mean', 'max', 'std' (default: 'mean')
%   'visualize'     - Plot results (default: true)
%   'verbose'       - Print results table (default: true)
%
% Output:
%   selected_features - Structure with fields:
%                      .indices: feature indices (1 x num_features)
%                      .combined_scores: combined scores
%                      .kl_scores: KL divergence scores (normalized)
%                      .roc_scores: ROC scores (normalized)
%                      .X: normalized feature matrix [samples x features]
%                      .Y: class labels
%

%% Parse input arguments
p = inputParser;
addParameter(p, 'num_features', 30, @isnumeric);
addParameter(p, 'num_classes', 7, @isnumeric);
addParameter(p, 'kl_weight', 0.5, @isnumeric);
addParameter(p, 'roc_weight', 0.5, @isnumeric);
addParameter(p, 'agg_method', 'mean', @ischar);
addParameter(p, 'visualize', true, @islogical);
addParameter(p, 'verbose', true, @islogical);
parse(p, varargin{:});

num_top_features = p.Results.num_features;
num_classes = p.Results.num_classes;
kl_weight = p.Results.kl_weight;
roc_weight = p.Results.roc_weight;
agg_method = p.Results.agg_method;
do_visualize = p.Results.visualize;
do_verbose = p.Results.verbose;

%% Validate input
if ~isstruct(lstm_data) || ~isfield(lstm_data, 'time_series') || ~isfield(lstm_data, 'labels')
    error('lstm_data must be a structure with fields: time_series (cell array) and label (vector)');
end

if ~iscell(lstm_data.time_series)
    error('lstm_data.time_series must be a cell array');
end

%% Prepare data: Extract features from all time steps
num_samples = length(lstm_data.labels);
Y = lstm_data.labels(:);

% Infer number of features from first sample
first_ts = lstm_data.time_series{1};
if ~ismatrix(first_ts)
    error('Each element in lstm_data.time_series must be a matrix [time x features]');
end
num_features = size(first_ts, 2);

% Extract features using aggregation method
X = zeros(num_samples, num_features);
for i = 1:num_samples
    ts = lstm_data.time_series{i};  % shape [time x features]
    
    switch lower(agg_method)
        case 'mean'
            X(i, :) = mean(ts, 1);
        case 'max'
            X(i, :) = max(ts, [], 1);
        case 'std'
            X(i, :) = std(ts, [], 1);
        otherwise
            error('Unknown aggregation method: %s', agg_method);
    end
end

% Handle NaN values
X(isnan(X)) = 0;

% Normalize features
X_mean = mean(X, 1);
X_std = std(X, 1);
X_std(X_std == 0) = 1;  % Avoid division by zero
X = (X - X_mean) ./ X_std;

if do_verbose
    fprintf('\n=== LSTM Feature Selection ===\n');
    fprintf('Number of samples: %d\n', num_samples);
    fprintf('Number of features: %d\n', num_features);
    fprintf('Number of classes: %d\n', num_classes);
    fprintf('Aggregation method: %s\n', agg_method);
    fprintf('Computing feature importance scores...\n\n');
end

%% Calculate feature importance scores

% 1. KL Divergence Score
if do_verbose
    fprintf('Computing KL divergence scores...\n');
end
kl_scores = zeros(1, num_features);
for feat = 1:num_features
    kl_sum = 0;
    valid_classes = 0;
    
    for class = 1:num_classes
        class_mask = Y == class;
        if sum(class_mask) < 2
            continue;  % Skip classes with fewer than 2 samples
        end
        
        class_data = X(class_mask, feat);
        
        % Create distribution (histogram)
        [counts, ~] = histcounts(class_data, 10);
        p = (counts + 1) / (sum(counts) + 10);  % Add smoothing
        
        % KL divergence from uniform distribution
        q = ones(1, 10) / 10;
        kl_div = sum(p .* log(p ./ (q + 1e-10)));
        kl_sum = kl_sum + max(0, kl_div);  % Ensure non-negative
        valid_classes = valid_classes + 1;
    end
    
    if valid_classes > 0
        kl_scores(feat) = kl_sum / valid_classes;
    end
end

% 2. ROC Score (One-vs-Rest, averaged across classes)
if do_verbose
    fprintf('Computing ROC scores...\n');
end
roc_scores = zeros(1, num_features);
for feat = 1:num_features
    auc_scores = [];
    
    for class = 1:num_classes
        class_mask = Y == class;
        if sum(class_mask) < 2 || sum(~class_mask) < 2
            continue;  % Need at least 2 samples in each class
        end
        
        % One-vs-Rest
        y_binary = double(class_mask);
        x_feat = X(:, feat);
        
        % Calculate AUC
        try
            [~, ~, ~, auc] = perfcurve(y_binary, x_feat, 1);
            auc_scores = [auc_scores; auc];
        catch
            % Skip if perfcurve fails
        end
    end
    
    if ~isempty(auc_scores)
        % Use deviation from 0.5 as score (0.5 = no discrimination)
        roc_scores(feat) = mean(abs(auc_scores - 0.5)) * 2;
    end
end

%% Normalize scores
kl_norm = (kl_scores - min(kl_scores)) / (max(kl_scores) - min(kl_scores) + 1e-10);
roc_norm = (roc_scores - min(roc_scores)) / (max(roc_scores) - min(roc_scores) + 1e-10);

% Ensure weights sum to 1
total_weight = kl_weight + roc_weight;
kl_weight = kl_weight / total_weight;
roc_weight = roc_weight / total_weight;

combined_score = kl_weight * kl_norm + roc_weight * roc_norm;

%% Select top N features
num_top_features = min(num_top_features, num_features);
[~, top_indices] = sort(combined_score, 'descend');
top_features = top_indices(1:num_top_features);
top_scores = combined_score(top_features);

%% Display results
if do_verbose
    fprintf('\n=== Top %d Most Discriminative Features ===\n', num_top_features);
    fprintf('Feature Index | Combined Score | KL Score | ROC Score\n');
    fprintf('%-13s | %-14s | %-8s | %-8s\n', '---', '---', '---', '---');
    for i = 1:num_top_features
        feat = top_features(i);
        fprintf('%-13d | %-14.4f | %-8.4f | %-8.4f\n', feat, combined_score(feat), kl_norm(feat), roc_norm(feat));
    end
    fprintf('\n');
end

%% Visualization
if do_visualize
    visualize_feature_selection(top_features, combined_score, kl_norm, roc_norm, X, Y, num_classes);
end

%% Output
selected_features.indices = top_features;
selected_features.combined_scores = top_scores;
selected_features.kl_scores = kl_norm(top_features);
selected_features.roc_scores = roc_norm(top_features);
selected_features.all_combined_scores = combined_score;
selected_features.all_kl_scores = kl_norm;
selected_features.all_roc_scores = roc_norm;
selected_features.X = X;
selected_features.Y = Y;
selected_features.agg_method = agg_method;

end

%% Helper function for visualization
function visualize_feature_selection(top_features, combined_score, kl_norm, roc_norm, X, Y, num_classes)

figure('Position', [100 100 1400 900]);

num_top_features = length(top_features);

% 1. Top N Features Score Bar Plot
subplot(2, 3, 1);
top_scores = combined_score(top_features);
bar(1:num_top_features, top_scores, 'FaceColor', [0.2 0.6 0.9]);
xlabel('Feature Rank');
ylabel('Combined Score');
title(sprintf('Top %d Features - Combined Score', num_top_features));
grid on;

% 2. KL Divergence vs ROC Score Scatter
subplot(2, 3, 2);
scatter(kl_norm(top_features), roc_norm(top_features), 100, top_scores, 'filled', 'o');
colorbar;
xlabel('KL Divergence Score (normalized)');
ylabel('ROC Score (normalized)');
title('Feature Score Distribution');
grid on;
hold on;
for i = 1:min(10, num_top_features)
    text(kl_norm(top_features(i)), roc_norm(top_features(i)), num2str(top_features(i)), ...
        'FontSize', 7, 'HorizontalAlignment', 'center');
end

% 3. All Features Ranking
subplot(2, 3, 3);
[sorted_scores, ~] = sort(combined_score, 'descend');
bar(1:length(combined_score), sorted_scores, 'FaceColor', [0.3 0.3 0.3]);
hold on;
xline(num_top_features + 0.5, 'r--', 'LineWidth', 2);
xlabel('Feature Rank');
ylabel('Combined Score');
title('All Features Ranking');
legend('All features', sprintf('Top %d cutoff', num_top_features));
grid on;

% 4. Box plot of top features across classes
subplot(2, 3, 4);
data_for_box = [];
labels_for_box = [];
for i = 1:min(10, num_top_features)
    feat_idx = top_features(i);
    for class = 1:num_classes
        class_mask = Y == class;
        if sum(class_mask) > 0
            class_data = X(class_mask, feat_idx);
            data_for_box = [data_for_box; class_data];
            labels_for_box = [labels_for_box; ones(length(class_data), 1) * (i + (class-1)*0.08)];
        end
    end
end
boxplot(data_for_box, labels_for_box);
xlabel('Feature Index');
ylabel('Normalized Value');
title('Top 10 Features Distribution Across Classes');
grid on;

% 5. ROC Curves for top 3 features (One-vs-Rest for class 1)
subplot(2, 3, 5);
colors = lines(3);
legend_entries = {};
for i = 1:min(3, num_top_features)
    feat_idx = top_features(i);
    y_binary = (Y == 1);
    if sum(y_binary) > 1 && sum(~y_binary) > 1
        x_feat = X(:, feat_idx);
        [fpr, tpr, ~, auc] = perfcurve(y_binary, x_feat, 1);
        plot(fpr, tpr, 'LineWidth', 2, 'Color', colors(i, :), ...
            'DisplayName', sprintf('Feature %d (AUC=%.3f)', feat_idx, auc));
        hold on;
    end
end
plot([0 1], [0 1], 'k--', 'LineWidth', 1, 'DisplayName', 'Random');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves - Top 3 Features (Class 1 vs Rest)');
legend('Location', 'best');
grid on;
xlim([0 1]); ylim([0 1]);

% 6. Heatmap of top features for each class
subplot(2, 3, 6);
heatmap_data = [];
for class = 1:num_classes
    class_mask = Y == class;
    if sum(class_mask) > 0
        class_features = X(class_mask, top_features);
        heatmap_data = [heatmap_data; mean(class_features, 1)];
    end
end
imagesc(heatmap_data);
colorbar;
set(gca, 'YTick', 1:num_classes);
set(gca, 'XTick', 1:num_top_features);
xlabel('Feature Index');
ylabel('Class');
title('Mean Feature Values per Class');

sgtitle(sprintf('LSTM Feature Selection Analysis - Top %d Features', num_top_features), 'FontSize', 14, 'FontWeight', 'bold');

end