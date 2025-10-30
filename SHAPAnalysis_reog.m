clear all
close all

%% SHAP analysis 
% Configuration
datapath = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10ClustersRegularized4/*reward_*Ypred_full*';
dirpath = dir(datapath);
output_dir = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SHAPAnalysis';
struct_path = '/Users/hsiehkunlin/Desktop/Data/Struct';

% SNR thresholds
snr_threshold_1 = 1;  % photometry SNR threshold
snr_threshold_2 = 1;  % breathing SNR threshold

% Feature names for better interpretation
feature_names = {'Dominant Freq', 'X Position', 'Y Position', 'Z Position', ...
                'Speed', 'tSNE-X', 'tSNE-Y', 'IR1', 'IR2', 'WP1', 'WP2', 'Shock', 'Sound'};

% Data Collection
fprintf('Loading data from %d sessions...\n', length(dirpath));

allshap = [];
allsnr = [];
allname = [];
allcoef = [];
allerror = [];

for SessionID = 1:length(dirpath)
    fprintf('Processing session %d/%d\n', SessionID, length(dirpath));
    
    file1 = dirpath(SessionID).name;
    file2 = dirpath(SessionID).name;
    file2(end-14:end) = [];
    file = file2;
    file2 = strcat(file2, "_SHAP_full.csv");
    path = dirpath(SessionID).folder;
    allname = [allname; file2];

    % Load prediction data
    y_pred = readtable(fullfile(path, file1));
    y_pred = table2array(y_pred);

    % Load SHAP values
    shap_values = readtable(fullfile(path, file2));
    shap_values = table2array(shap_values);
    allshap = [allshap; mean(abs(shap_values))];

    % Calculate correlation and error
    Y = y_pred(:,1)';
    B = y_pred(:,2)';
    C = corr(Y(:), B(:));
    allcoef = [allcoef, C];
    
    absolute_errors = abs(Y(:) - y_pred(:,2));
    MedAE = median(absolute_errors);
    allerror = [allerror, MedAE];
    
    % Load SNR data
    ALLStructFile_temp = load(fullfile(struct_path, file));
    SNR = ALLStructFile_temp.SNR;
    allsnr = [allsnr; SNR];
end

% Data Categorization
ind1 = allsnr(:,1) > snr_threshold_1 & allsnr(:,3) > snr_threshold_2;  % both good
ind2 = allsnr(:,1) > snr_threshold_1 & allsnr(:,3) <= snr_threshold_2; % bad breathing
ind3 = allsnr(:,1) <= snr_threshold_1 & allsnr(:,3) > snr_threshold_2;  % bad photometry
ind4 = allsnr(:,1) <= snr_threshold_1 & allsnr(:,3) <= snr_threshold_2; % both bad

groups = ind1 + ind2*2 + ind3*3 + ind4*4;
group_names = {'Both Good', 'Bad Breathing', 'Bad Photometry', 'Both Bad'};
group_colors = {'#2E8B57', '#4169E1', '#FF6347', '#DC143C'}; % Modern color palette

fprintf('Data summary:\n');
fprintf('Both good: %d sessions\n', sum(ind1));
fprintf('Bad breathing: %d sessions\n', sum(ind2));
fprintf('Bad photometry: %d sessions\n', sum(ind3));
fprintf('Both bad: %d sessions\n', sum(ind4));

% Main SHAP Values Visualization with Multiple Comparison Correction
fig1 = figure('Position', [100, 100, 1200, 800]);
set(fig1, 'Color', 'white');

% Calculate statistics for plotting
mean_shap_1 = mean(allshap(ind1,:), 1);
mean_shap_2 = mean(allshap(ind2,:), 1);
std_shap_1 = std(allshap(ind1,:), 1);
std_shap_2 = std(allshap(ind2,:), 1);

% Main plot with error bars
hold on;
h1 = errorbar(1:size(allshap,2), mean_shap_1, std_shap_1, ...
    'Color', group_colors{1}, 'LineWidth', 3, 'Marker', 'o', ...
    'MarkerSize', 8, 'MarkerFaceColor', group_colors{1}, ...
    'DisplayName', group_names{1});

h2 = errorbar(1:size(allshap,2), mean_shap_2, std_shap_2, ...
    'Color', group_colors{2}, 'LineWidth', 3, 'Marker', 's', ...
    'MarkerSize', 8, 'MarkerFaceColor', group_colors{2}, ...
    'DisplayName', group_names{2});

% Statistical significance with multiple comparison correction
[~, p_values_raw] = ttest2(allshap(ind1,:), allshap(ind2,:));

% Apply multiple comparison corrections
% 1. Bonferroni correction
p_bonferroni = p_values_raw * length(p_values_raw);
p_bonferroni(p_bonferroni > 1) = 1; % Cap at 1

% 2. Benjamini-Hochberg (FDR) correction
[p_fdr] = mafdr(p_values_raw, 'BHFDR', true);

% 3. Holm-Bonferroni correction (more powerful than Bonferroni)
[p_sorted, sort_idx] = sort(p_values_raw);
n_tests = length(p_values_raw);
p_holm = zeros(size(p_values_raw));

for i = 1:n_tests
    p_holm(sort_idx(i)) = min(1, p_sorted(i) * (n_tests - i + 1));
end

% Find significant features for different corrections
sig_features_raw = find(p_values_raw < 0.05);
sig_features_bonferroni = find(p_bonferroni < 0.05);
sig_features_fdr = find(p_fdr < 0.05);
sig_features_holm = find(p_holm < 0.05);

% Plot significance markers at different heights
max_y = max([mean_shap_1 + std_shap_1, mean_shap_2 + std_shap_2]);
y_offset = max_y * 0.05;

% Raw p-values (uncorrected)
if ~isempty(sig_features_raw)
    plot(sig_features_raw, repmat(max_y * 1.05, length(sig_features_raw)), ...
        'k*', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Uncorrected p < 0.05');
end

% FDR corrected (most commonly used for SHAP)
if ~isempty(sig_features_fdr)
    plot(sig_features_fdr, repmat(max_y * 1.15, length(sig_features_fdr)), ...
        'r*', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'FDR p < 0.05');
end

% Bonferroni corrected (most conservative)
if ~isempty(sig_features_bonferroni)
    plot(sig_features_bonferroni, repmat(max_y * 1.25, length(sig_features_bonferroni)), ...
        'b*', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Bonferroni p < 0.05');
end

% Holm-Bonferroni corrected (good balance)
if ~isempty(sig_features_holm)
    plot(sig_features_holm, repmat(max_y * 1.35, length(sig_features_holm)), ...
        'g*', 'MarkerSize', 11, 'LineWidth', 2, 'DisplayName', 'Holm p < 0.05');
end

% Formatting
set(gca, 'XTick', 1:length(feature_names), 'XTickLabel', feature_names, ...
    'FontSize', 12, 'FontWeight', 'bold');
xtickangle(45);
xlabel('Features', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Mean Absolute SHAP Values', 'FontSize', 14, 'FontWeight', 'bold');
title('Feature Importance: High vs Low SNR Sessions (Multiple Comparison Corrected)', ...
    'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 12);
grid on;
grid('minor');
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.1);

% Adjust y-axis to accommodate significance markers
ylim([0, max_y * 1.5]);

% Add text summary of corrections
text_str = sprintf('Multiple Comparison Summary:\n');
text_str = [text_str sprintf('Raw significant: %d/%d features\n', length(sig_features_raw), length(p_values_raw))];
text_str = [text_str sprintf('FDR significant: %d/%d features\n', length(sig_features_fdr), length(p_values_raw))];
text_str = [text_str sprintf('Bonferroni significant: %d/%d features\n', length(sig_features_bonferroni), length(p_values_raw))];
text_str = [text_str sprintf('Holm significant: %d/%d features', length(sig_features_holm), length(p_values_raw))];

text(0.02, 0.98, text_str, 'Units', 'normalized', 'FontSize', 10, ...
    'VerticalAlignment', 'top', 'BackgroundColor', 'white', ...
    'EdgeColor', 'black', 'FontWeight', 'bold');

% Print detailed results to console
fprintf('\n=== MULTIPLE COMPARISON CORRECTION RESULTS ===\n');
fprintf('Feature\t\tRaw p\t\tFDR p\t\tBonferroni p\tHolm p\n');
fprintf('-------\t\t-----\t\t-----\t\t-----------\t------\n');

for i = 1:length(feature_names)
    fprintf('%-15s\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f', ...
        feature_names{i}, p_values_raw(i), p_fdr(i), p_bonferroni(i), p_holm(i));
    
    % Mark significant results
    sig_markers = '';
    if p_values_raw(i) < 0.05, sig_markers = [sig_markers '*']; end
    if p_fdr(i) < 0.05, sig_markers = [sig_markers 'F']; end
    if p_bonferroni(i) < 0.05, sig_markers = [sig_markers 'B']; end
    if p_holm(i) < 0.05, sig_markers = [sig_markers 'H']; end
    
    if ~isempty(sig_markers)
        fprintf('\t(%s)', sig_markers);
    end
    fprintf('\n');
end

fprintf('\nLegend: * = Raw significant, F = FDR significant, B = Bonferroni significant, H = Holm significant\n');

% Create a summary table for export
summary_table = table();
summary_table.Feature = feature_names';
summary_table.Raw_p_value = p_values_raw';
summary_table.FDR_corrected_p = p_fdr';
summary_table.Bonferroni_corrected_p = p_bonferroni';
summary_table.Holm_corrected_p = p_holm';
summary_table.Raw_significant = p_values_raw' < 0.05;
summary_table.FDR_significant = p_fdr' < 0.05;
summary_table.Bonferroni_significant = p_bonferroni' < 0.05;
summary_table.Holm_significant = p_holm' < 0.05;

% Save the table
writetable(summary_table, fullfile(output_dir, 'SHAP_Multiple_Comparison_Results.csv'));

fprintf('\nResults saved to: SHAP_Multiple_Comparison_Results.csv\n');
saveas(fig1, fullfile(output_dir, 'SHAP_Pvalue_Comparison_Controled.png'), 'png');
saveas(fig1, fullfile(output_dir, 'SHAP_Pvalue_Comparison_Controled.svg'), 'svg');

% Alternative visualization: P-value comparison plot
fig_pval = figure('Position', [200, 200, 1000, 600]);
set(fig_pval, 'Color', 'white');

subplot(2,1,1);
hold on;
plot(1:length(p_values_raw), -log10(p_values_raw), 'ko-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Raw p-values');
plot(1:length(p_fdr), -log10(p_fdr), 'ro-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'FDR corrected');
plot(1:length(p_bonferroni), -log10(p_bonferroni), 'bo-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Bonferroni');
plot(1:length(p_holm), -log10(p_holm), 'go-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Holm');

% Add significance threshold line
yline(-log10(0.05), '--k', 'p = 0.05', 'LineWidth', 2);

set(gca, 'XTick', 1:length(feature_names), 'XTickLabel', feature_names);
xtickangle(45);
xlabel('Features', 'FontWeight', 'bold');
ylabel('-log₁₀(p-value)', 'FontWeight', 'bold');
title('Multiple Comparison Correction Effects', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

subplot(2,1,2);
% Effect sizes (mean differences)
effect_sizes = abs(mean_shap_1 - mean_shap_2);
bar(1:length(effect_sizes), effect_sizes, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'black');
hold on;

% Highlight significant features with different corrections
if ~isempty(sig_features_fdr)
    bar(sig_features_fdr, effect_sizes(sig_features_fdr), 'FaceColor', 'red', 'EdgeColor', 'black');
end

set(gca, 'XTick', 1:length(feature_names), 'XTickLabel', feature_names);
xtickangle(45);
xlabel('Features', 'FontWeight', 'bold');
ylabel('Effect Size (|Mean Difference|)', 'FontWeight', 'bold');
title('Effect Sizes with FDR Significant Features Highlighted', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

sgtitle('Statistical Significance Analysis with Multiple Comparison Correction', 'FontSize', 16, 'FontWeight', 'bold');

% Save p-value comparison figure
saveas(fig_pval, fullfile(output_dir, 'SHAP_Pvalue_Comparison.png'), 'png');
saveas(fig_pval, fullfile(output_dir, 'SHAP_Pvalue_Comparison.svg'), 'svg');

% Comprehensive Analysis Figure
fig2 = figure('Position', [150, 150, 1400, 1000]);
set(fig2, 'Color', 'white');

% Subplot 1: SHAP values for all groups
subplot(3,1,1);
hold on;
colors = cellfun(@(x) sscanf(x(2:end),'%2x%2x%2x',[1 3])/255, group_colors, 'UniformOutput', false);

plot(1:size(allshap,2), mean(allshap(ind1,:),1), 'Color', colors{1}, ...
    'LineWidth', 3, 'Marker', 'o', 'DisplayName', group_names{1});
plot(1:size(allshap,2), mean(allshap(ind2,:),1), 'Color', colors{2}, ...
    'LineWidth', 3, 'Marker', 's', 'DisplayName', group_names{2});
if sum(ind3) > 0
    plot(1:size(allshap,2), mean(allshap(ind3,:),1), 'Color', colors{3}, ...
        'LineWidth', 3, 'Marker', '^', 'DisplayName', group_names{3});
end
if sum(ind4) > 0
    plot(1:size(allshap,2), mean(allshap(ind4,:),1), 'Color', colors{4}, ...
        'LineWidth', 3, 'Marker', 'd', 'DisplayName', group_names{4});
end

set(gca, 'XTick', 1:length(feature_names), 'XTickLabel', feature_names, 'FontSize', 10);
xtickangle(45);
xlabel('Features', 'FontWeight', 'bold');
ylabel('Mean Abs SHAP Values', 'FontWeight', 'bold');
title('SHAP Values by Signal Quality', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

% Subplot 4: SNR distribution
subplot(3,1,2);
scatter(allsnr(ind1,1), allsnr(ind1,3), 100, colors{1}, 'filled', 'DisplayName', group_names{1});
hold on;
scatter(allsnr(ind2,1), allsnr(ind2,3), 100, colors{2}, 'filled', 'DisplayName', group_names{2});
if sum(ind3) > 0
    scatter(allsnr(ind3,1), allsnr(ind3,3), 100, colors{3}, 'filled', 'DisplayName', group_names{3});
end
if sum(ind4) > 0
    scatter(allsnr(ind4,1), allsnr(ind4,3), 100, colors{4}, 'filled', 'DisplayName', group_names{4});
end

% Add threshold lines
xline(snr_threshold_1, '--k', 'LineWidth', 2);
yline(snr_threshold_2, '--k', 'LineWidth', 2);

xlabel('Photometry SNR', 'FontWeight', 'bold');
ylabel('Breathing SNR', 'FontWeight', 'bold');
title('SNR Distribution', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

% Subplot 5: Feature importance heatmap
subplot(3,1,3);
shap_matrix = [mean(allshap(ind1,:),1); mean(allshap(ind2,:),1)];
if sum(ind3) > 0
    shap_matrix = [shap_matrix; mean(allshap(ind3,:),1)];
end
if sum(ind4) > 0
    shap_matrix = [shap_matrix; mean(allshap(ind4,:),1)];
end

imagesc(shap_matrix);
colormap('hot');
colorbar;
set(gca, 'XTick', 1:length(feature_names), 'XTickLabel', feature_names, ...
    'YTick', 1:sum(valid_groups), 'YTickLabel', group_labels(valid_groups));
xtickangle(45);
title('Feature Importance Heatmap', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Features', 'FontWeight', 'bold');
ylabel('Signal Quality Groups', 'FontWeight', 'bold');

% Save comprehensive figure
saveas(fig2, fullfile(output_dir, 'SHAP_Comprehensive_Analysis.png'), 'png');
saveas(fig2, fullfile(output_dir, 'SHAP_Comprehensive_Analysis.svg'), 'svg');

%% drop analysis
base_path = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10ClustersRegularized4/drop_feature_analysis/';
base_path2 = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10ClustersRegularized4/';
struct_path = '/Users/hsiehkunlin/Desktop/Data/Struct';
output_dir = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SHAPAnalysis';

% Create output directory if it doesn't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Feature names (based on your original code comments)
feature_names = {'Dominant Freq', 'X Position', 'Y Position', 'Z Position', ...
                'Speed', 'tSNE-X', 'tSNE-Y', 'IR1', 'IR2', 'WP1', 'WP2', 'Shock', 'Sound'};

% SNR thresholds for categorization
snr_threshold_1 = 1;  % photometry SNR threshold
snr_threshold_2 = 1;  % breathing SNR threshold

% Find baseline (full model) and dropped feature files
fprintf('Scanning for feature drop analysis files...\n');

% Get all relevant files
all_files = dir(fullfile(base_path, '*reward_AllStruct_CameraTime_dropped_Feature_*_Ypred.csv'));
baseline_files = dir(fullfile(base_path2, '*reward_AllStruct_CameraTime_Ypred_test.csv'));

if isempty(all_files)
    error('No dropped feature files found. Check the path: %s', base_path);
end

fprintf('Found %d dropped feature files\n', length(all_files));
fprintf('Found %d baseline files\n', length(baseline_files));

% Extract session information and organize data
session_data = struct();
session_id_mapping = containers.Map(); % Map from cleaned names to original names
dropped_features = [];

% Process dropped feature files
for i = 1:length(all_files)
    filename = all_files(i).name;
    
    % Extract session ID and dropped feature number
    tokens = regexp(filename, '(.+)_dropped_Feature_(\d+)_Ypred\.csv', 'tokens');
    if ~isempty(tokens)
        session_id_raw = tokens{1}{1};
        % Clean session ID for valid MATLAB field name
        session_id = matlab.lang.makeValidName(session_id_raw);
        % Store mapping from cleaned to original name
        session_id_mapping(session_id) = session_id_raw;
        dropped_feature = str2double(tokens{1}{2});
        
        if ~isfield(session_data, session_id)
            session_data.(session_id) = struct();
        end
        
        % Load predictions for dropped feature model
        pred_data = readtable(fullfile(base_path, filename));
        pred_array = table2array(pred_data);
        
        Y_true = pred_array(:,1);
        Y_pred = pred_array(:,2);
        
        % Calculate R²
        r_squared = corr(Y_true, Y_pred)^2;
        
        session_data.(session_id).(['dropped_' num2str(dropped_feature)]) = r_squared;
        dropped_features = unique([dropped_features, dropped_feature]);
    end
end


% Process baseline files (full model)
baseline_r2 = struct();
for i = 1:length(baseline_files)
    filename = baseline_files(i).name;
    
    % Extract session ID
    tokens = regexp(filename, '(.+)_Ypred_test\.csv', 'tokens');
    if ~isempty(tokens)
        session_id_raw = tokens{1}{1};
        % Clean session ID for valid MATLAB field name
        session_id = matlab.lang.makeValidName(session_id_raw);
        
        % Load baseline predictions
        pred_data = readtable(fullfile(base_path2, filename));
        pred_array = table2array(pred_data);
        
        Y_true = pred_array(:,1);
        Y_pred = pred_array(:,2);
        
        % Calculate baseline R²
        baseline_r2.(session_id) = corr(Y_true, Y_pred)^2;
    end
end

% Load SNR data for categorization
session_names = fieldnames(session_data);
all_snr = [];
session_indices = [];

fprintf('Loading SNR data for %d sessions...\n', length(session_names));

for i = 1:length(session_names)
    session_id = session_names{i};
    
    % Get original session name for file lookup
    if isKey(session_id_mapping, session_id)
        original_session_id = session_id_mapping(session_id);
    else
        original_session_id = session_id; % fallback
    end
    
    % Try to load SNR data using original session name
    struct_file = fullfile(struct_path, [original_session_id '.mat']);
    if exist(struct_file, 'file')
        try
            data = load(struct_file);
            if isfield(data, 'SNR')
                all_snr = [all_snr; data.SNR];
                session_indices = [session_indices; i];
            else
                fprintf('Warning: SNR field not found in %s\n', struct_file);
            end
        catch
            fprintf('Warning: Could not load %s\n', struct_file);
        end
    else
        fprintf('Warning: Struct file not found: %s\n', struct_file);
    end
end

% Categorize sessions by SNR quality
if ~isempty(all_snr)
    ind1 = all_snr(:,1) > snr_threshold_1 & all_snr(:,3) > snr_threshold_2;  % both good
    ind2 = all_snr(:,1) > snr_threshold_1 & all_snr(:,3) <= snr_threshold_2; % bad breathing
    ind3 = all_snr(:,1) <= snr_threshold_1 & all_snr(:,3) > snr_threshold_2;  % bad photometry
    ind4 = all_snr(:,1) <= snr_threshold_1 & all_snr(:,3) <= snr_threshold_2; % both bad
    
    group_names = {'Both Good', 'Bad Breathing', 'Bad Photometry', 'Both Bad'};
    group_colors = {[0.18 0.55 0.34], [0.26 0.41 0.88], [1.0 0.39 0.28], [0.86 0.08 0.24]};
else
    fprintf('Warning: No SNR data available for categorization\n');
    ind1 = true(length(session_names), 1);
    ind2 = false(length(session_names), 1);
    ind3 = false(length(session_names), 1);
    ind4 = false(length(session_names), 1);
end

% Calculate R² changes for each feature
fprintf('Calculating R² changes...\n');

% Sort dropped features for consistent ordering
dropped_features = sort(dropped_features);
num_features = length(dropped_features);

% Initialize matrices
r2_baseline_all = [];
r2_dropped_all = zeros(length(session_names), num_features);
r2_change_all = zeros(length(session_names), num_features);

valid_sessions = [];

for i = 1:length(session_names)
    session_id = session_names{i};
    
    % Check if baseline exists
    if isfield(baseline_r2, session_id)
        baseline = baseline_r2.(session_id);
        r2_baseline_all = [r2_baseline_all; baseline];
        
        session_valid = true;
        for j = 1:num_features
            feature_num = dropped_features(j);
            field_name = ['dropped_' num2str(feature_num)];
            
            if isfield(session_data.(session_id), field_name)
                dropped_r2 = session_data.(session_id).(field_name);
                r2_dropped_all(i, j) = dropped_r2;
                r2_change_all(i, j) = baseline - dropped_r2;  % Positive = performance drop
            else
                session_valid = false;
                break;
            end
        end
        
        if session_valid
            valid_sessions = [valid_sessions; i];
        end
    end
end

% Filter to valid sessions only
r2_baseline_all = r2_baseline_all(valid_sessions);
r2_dropped_all = r2_dropped_all(valid_sessions, :);
r2_change_all = r2_change_all(valid_sessions, :);

if ~isempty(all_snr)
    % Update indices for valid sessions only
    valid_session_indices = session_indices(valid_sessions);
    ind1_valid = ind1(valid_session_indices);
    ind2_valid = ind2(valid_session_indices);
    ind3_valid = ind3(valid_session_indices);
    ind4_valid = ind4(valid_session_indices);
else
    ind1_valid = true(length(valid_sessions), 1);
    ind2_valid = false(length(valid_sessions), 1);
    ind3_valid = false(length(valid_sessions), 1);
    ind4_valid = false(length(valid_sessions), 1);
end

fprintf('Analysis completed for %d valid sessions\n', length(valid_sessions));

% Visualization 1: Overall Feature Importance (R² Drop)
fig1 = figure('Position', [100, 100, 1200, 800]);
set(fig1, 'Color', 'white');

% Calculate mean and SEM for all sessions
mean_r2_change = mean(r2_change_all, 1);
sem_r2_change = std(r2_change_all, 1) / sqrt(size(r2_change_all, 1));

% Create bar plot with error bars
bar_handle = bar(1:num_features, mean_r2_change, 'FaceColor', [0.3 0.6 0.8], ...
    'EdgeColor', 'black', 'LineWidth', 1.5);
hold on;
errorbar(1:num_features, mean_r2_change, sem_r2_change, 'k', ...
    'LineStyle', 'none', 'LineWidth', 2, 'CapSize', 10);

% Format plot
feature_labels = feature_names(dropped_features+1);
set(gca, 'XTick', 1:num_features, 'XTickLabel', feature_labels, ...
    'FontSize', 12, 'FontWeight', 'bold');
xtickangle(45);
xlabel('Dropped Feature', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('R² Decrease (Baseline - Dropped)', 'FontSize', 14, 'FontWeight', 'bold');
title('Feature Importance: R² Drop When Each Feature is Removed', ...
    'FontSize', 16, 'FontWeight', 'bold');

% Add horizontal line at zero
yline(0, '--k', 'LineWidth', 1, 'Alpha', 0.5);

% Add value labels on bars
for i = 1:num_features
    if mean_r2_change(i) > 0
        text(i, mean_r2_change(i) + sem_r2_change(i) + 0.001, ...
            sprintf('%.3f', mean_r2_change(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
    end
end

grid on;
set(gca, 'GridAlpha', 0.3);

% Save figure
saveas(fig1, fullfile(output_dir, 'Feature_Importance_R2_Drop.png'), 'png');
saveas(fig1, fullfile(output_dir, 'Feature_Importance_R2_Drop.svg'), 'svg');

% Visualization 2: SNR-stratified Analysis
fig2 = figure('Position', [150, 150, 1400, 1000]);
set(fig2, 'Color', 'white');

% Subplot 1: R² changes by SNR group
subplot(2,2,1);
hold on;

if sum(ind1_valid) > 0
    mean_change_good = mean(r2_change_all(ind1_valid, :), 1);
    sem_change_good = std(r2_change_all(ind1_valid, :), 1) / sqrt(sum(ind1_valid));
    errorbar(1:num_features, mean_change_good, sem_change_good, ...
        'Color', group_colors{1}, 'LineWidth', 3, 'Marker', 'o', ...
        'MarkerSize', 8, 'MarkerFaceColor', group_colors{1}, ...
        'DisplayName', sprintf('%s (n=%d)', group_names{1}, sum(ind1_valid)));
end

if sum(ind2_valid) > 0
    mean_change_bad = mean(r2_change_all(ind2_valid, :), 1);
    sem_change_bad = std(r2_change_all(ind2_valid, :), 1) / sqrt(sum(ind2_valid));
    errorbar(1:num_features, mean_change_bad, sem_change_bad, ...
        'Color', group_colors{2}, 'LineWidth', 3, 'Marker', 's', ...
        'MarkerSize', 8, 'MarkerFaceColor', group_colors{2}, ...
        'DisplayName', sprintf('%s (n=%d)', group_names{2}, sum(ind2_valid)));
end

set(gca, 'XTick', 1:num_features, 'XTickLabel', feature_labels, 'FontSize', 10);
xtickangle(45);
xlabel('Dropped Feature', 'FontWeight', 'bold');
ylabel('Mean R² Decrease', 'FontWeight', 'bold');
title('Feature Importance by Signal Quality', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
yline(0, '--k', 'LineWidth', 1, 'Alpha', 0.5);

% Subplot 2: Baseline R² distribution
subplot(2,2,2);
hold on;

if sum(ind1_valid) > 0
    histogram(r2_baseline_all(ind1_valid), 'FaceColor', group_colors{1}, ...
        'FaceAlpha', 0.7, 'EdgeColor', 'black', 'DisplayName', group_names{1});
end
if sum(ind2_valid) > 0
    histogram(r2_baseline_all(ind2_valid), 'FaceColor', group_colors{2}, ...
        'FaceAlpha', 0.7, 'EdgeColor', 'black', 'DisplayName', group_names{2});
end

xlabel('Baseline R²', 'FontWeight', 'bold');
ylabel('Frequency', 'FontWeight', 'bold');
title('Baseline Model Performance Distribution', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

% Subplot 3: Feature importance heatmap
subplot(2,2,3);
if sum(ind1_valid) > 0 && sum(ind2_valid) > 0
    heatmap_data = [mean(r2_change_all(ind1_valid, :), 1); 
                    mean(r2_change_all(ind2_valid, :), 1)];
    
    imagesc(heatmap_data);
    colormap('hot');
    colorbar;
    set(gca, 'XTick', 1:num_features, 'XTickLabel', feature_labels, ...
        'YTick', 1:2, 'YTickLabel', {group_names{1}, group_names{2}});
    xtickangle(45);
    title('R² Drop Heatmap', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Dropped Feature', 'FontWeight', 'bold');
    ylabel('Signal Quality Group', 'FontWeight', 'bold');
end

% Subplot 4: Statistical significance
subplot(2,2,4);
if sum(ind1_valid) > 0 && sum(ind2_valid) > 0
    p_values = zeros(1, num_features);
    
    for i = 1:num_features
        [~, p_values(i)] = ttest2(r2_change_all(ind1_valid, i), r2_change_all(ind2_valid, i));
    end
    
    % Bar plot of p-values
    bar(1:num_features, -log10(p_values), 'FaceColor', [0.8 0.4 0.4], 'EdgeColor', 'black');
    hold on;
    yline(-log10(0.05), '--r', 'p = 0.05', 'LineWidth', 2);
    yline(-log10(0.01), '--r', 'p = 0.01', 'LineWidth', 2);
    
    set(gca, 'XTick', 1:num_features, 'XTickLabel', feature_labels);
    xtickangle(45);
    xlabel('Feature', 'FontWeight', 'bold');
    ylabel('-log₁₀(p-value)', 'FontWeight', 'bold');
    title('Statistical Significance of Group Differences', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
end

sgtitle('Feature Drop Analysis: Impact on Model Performance', ...
    'FontSize', 16, 'FontWeight', 'bold');

% Save figure
saveas(fig2, fullfile(output_dir, 'Feature_Drop_SNR_Analysis.png'), 'png');
saveas(fig2, fullfile(output_dir, 'Feature_Drop_SNR_Analysis.svg'), 'svg');

% Summary Statistics Table
fprintf('\n=== FEATURE IMPORTANCE ANALYSIS SUMMARY ===\n');
fprintf('Total sessions analyzed: %d\n', length(valid_sessions));

if ~isempty(all_snr)
    fprintf('Good SNR sessions: %d\n', sum(ind1_valid));
    fprintf('Bad breathing sessions: %d\n', sum(ind2_valid));
    fprintf('Bad photometry sessions: %d\n', sum(ind3_valid));
    fprintf('Both bad SNR sessions: %d\n', sum(ind4_valid));
end

fprintf('\nMean baseline R²: %.4f ± %.4f\n', mean(r2_baseline_all), std(r2_baseline_all));

fprintf('\n--- Feature Importance Ranking (by R² drop) ---\n');
[sorted_importance, sort_idx] = sort(mean_r2_change, 'descend');

for i = 1:num_features
    feature_idx = sort_idx(i);
    fprintf('%d. %s: %.4f ± %.4f\n', i, feature_labels{feature_idx}, ...
        sorted_importance(i), sem_r2_change(feature_idx));
end

% Create summary table
summary_table = table();
summary_table.Feature = feature_labels';
summary_table.Mean_R2_Drop = mean_r2_change';
summary_table.SEM_R2_Drop = sem_r2_change';

if sum(ind1_valid) > 0
    summary_table.GoodSNR_R2_Drop = mean(r2_change_all(ind1_valid, :), 1)';
end
if sum(ind2_valid) > 0
    summary_table.BadBreathing_R2_Drop = mean(r2_change_all(ind2_valid, :), 1)';
end

% Save summary table
writetable(summary_table, fullfile(output_dir, 'Feature_Drop_Summary.csv'));

fprintf('\nAnalysis complete! Results saved to: %s\n', output_dir);
fprintf('Summary table saved as: Feature_Drop_Summary.csv\n');

%% Feature Importance Analysis with Statistical Testing and FDR Correction
% Tests whether R² drop is significantly different from zero for each feature
% Statistical Testing for Feature Importance
fprintf('Running statistical tests for feature importance...\n');

% One-sample t-tests against zero (is R² drop significantly > 0?)
p_values_ttest = zeros(num_features, 1);
t_statistics = zeros(num_features, 1);
effect_sizes = zeros(num_features, 1);  % Cohen's d
ci_lower = zeros(num_features, 1);
ci_upper = zeros(num_features, 1);

for i = 1:num_features
    r2_drops = r2_change_all(:, i);  % R² drops for this feature
    
    % One-sample t-test against zero
    [~, p, ci, stats] = ttest(r2_drops, 0);
    
    p_values_ttest(i) = p;
    t_statistics(i) = stats.tstat;
    ci_lower(i) = ci(1);
    ci_upper(i) = ci(2);
    
    % Effect size (Cohen's d)
    effect_sizes(i) = mean(r2_drops) / std(r2_drops);
end

% Apply FDR correction
p_fdr = mafdr(p_values_ttest, 'BHFDR', true);

% Find significant features
sig_features_idx = find(p_fdr < 0.05);
sig_features_names = feature_names(dropped_features(sig_features_idx)+1);

fprintf('\n=== FEATURE IMPORTANCE STATISTICAL RESULTS ===\n');
fprintf('Testing: H0: R² drop = 0 vs H1: R² drop > 0\n');
fprintf('Multiple comparison correction: FDR (Benjamini-Hochberg)\n\n');

fprintf('Significant features (FDR p < 0.05): %d out of %d\n', ...
    length(sig_features_idx), num_features);

if ~isempty(sig_features_idx)
    fprintf('\nSignificant features:\n');
    fprintf('Feature\t\t\tMean R² Drop\tSEM\t\tt-stat\t\tRaw p\t\tFDR p\t\tCohen''s d\n');
    fprintf('-------\t\t\t------------\t---\t\t------\t\t-----\t\t-----\t\t---------\n');
    
    for i = sig_features_idx'
        feature_idx = dropped_features(i);
        fprintf('%-15s\t%.4f\t\t%.4f\t\t%.3f\t\t%.6f\t%.6f\t%.3f\n', ...
            feature_names{feature_idx+1}, mean_r2_change(i), sem_r2_change(i), ...
            t_statistics(i), p_values_ttest(i), p_fdr(i), effect_sizes(i));
    end
end

fprintf('\nAll features results:\n');
fprintf('Feature\t\t\tMean R² Drop\tt-stat\t\tRaw p\t\tFDR p\t\tSignificant\n');
fprintf('-------\t\t\t------------\t------\t\t-----\t\t-----\t\t-----------\n');

for i = 1:num_features
    feature_idx = dropped_features(i);
    is_sig = p_fdr(i) < 0.05;
    sig_marker = '';
    if is_sig
        sig_marker = '*';
    end
    
    fprintf('%-15s\t%.4f\t\t%.3f\t\t%.6f\t%.6f\t%s\n', ...
        feature_names{feature_idx+1}, mean_r2_change(i), t_statistics(i), ...
        p_values_ttest(i), p_fdr(i), sig_marker);
end

% Enhanced Visualization with Statistical Results
fig1 = figure('Position', [100, 100, 1400, 900]);
set(fig1, 'Color', 'white');

% Calculate mean and SEM for all sessions
mean_r2_change = mean(r2_change_all, 1);
sem_r2_change = std(r2_change_all, 1) / sqrt(size(r2_change_all, 1));

% Create colors based on significance
bar_colors = repmat([0.7 0.7 0.7], num_features, 1);  % Default gray
bar_colors(sig_features_idx, :) = repmat([0.2 0.6 0.2], length(sig_features_idx), 1);  % Significant = green

% Create bar plot with different colors
hold on;
for i = 1:num_features
    bar_handle = bar(i, mean_r2_change(i), 'FaceColor', bar_colors(i, :), ...
        'EdgeColor', 'black', 'LineWidth', 1.5);
end

% Add error bars
errorbar(1:num_features, mean_r2_change, sem_r2_change, 'k', ...
    'LineStyle', 'none', 'LineWidth', 2, 'CapSize', 10);

% Format plot
feature_labels = feature_names(dropped_features+1);
set(gca, 'XTick', 1:num_features, 'XTickLabel', feature_labels, ...
    'FontSize', 12, 'FontWeight', 'bold');
xtickangle(45);
xlabel('Dropped Feature', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('R² Decrease (Baseline - Dropped)', 'FontSize', 14, 'FontWeight', 'bold');
title('Feature Importance: R² Drop When Each Feature is Removed (FDR Corrected)', ...
    'FontSize', 16, 'FontWeight', 'bold');

% Add horizontal line at zero
yline(0, '--k', 'LineWidth', 1, 'Alpha', 0.5);

% Add significance stars above bars
for i = 1:num_features
    if p_fdr(i) < 0.001
        sig_text = '***';
    elseif p_fdr(i) < 0.01
        sig_text = '**';
    elseif p_fdr(i) < 0.05
        sig_text = '*';
    else
        sig_text = '';
    end
    
    if ~isempty(sig_text)
        text(i, mean_r2_change(i) + sem_r2_change(i) + 0.002, sig_text, ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
            'FontSize', 12, 'Color', 'red');
    end
end

% Add value labels on bars (only for significant ones)
for i = sig_features_idx'
    text(i, mean_r2_change(i) + sem_r2_change(i) + 0.005, ...
        sprintf('%.3f', mean_r2_change(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
end

grid on;
set(gca, 'GridAlpha', 0.3);

% Add legend
legend_handles = [];
legend_labels = {};

% Create dummy bars for legend
h1 = bar(NaN, NaN, 'FaceColor', [0.2 0.6 0.2], 'EdgeColor', 'black');
h2 = bar(NaN, NaN, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'black');
legend_handles = [h1, h2];
legend_labels = {'Significant (FDR p < 0.05)', 'Not Significant'};

legend(legend_handles, legend_labels, 'Location', 'best', 'FontSize', 12);

% Add statistical summary text box
summary_text = sprintf('Statistical Summary:\n');
summary_text = [summary_text sprintf('Significant features: %d/%d\n', length(sig_features_idx), num_features)];
summary_text = [summary_text sprintf('FDR correction applied\n')];
summary_text = [summary_text sprintf('* p < 0.05, ** p < 0.01, *** p < 0.001')];

annotation('textbox', [0.02, 0.98, 0.25, 0.15], 'String', summary_text, ...
    'Units', 'normalized', 'FontSize', 10, 'FontWeight', 'bold', ...
    'BackgroundColor', 'white', 'EdgeColor', 'black', ...
    'VerticalAlignment', 'top');

% Save Results
% Create comprehensive results table
results_table = table();
results_table.Feature = feature_names(dropped_features+1)';
results_table.Mean_R2_Drop = mean_r2_change';
results_table.SEM = sem_r2_change';
results_table.CI_Lower = ci_lower;
results_table.CI_Upper = ci_upper;
results_table.t_statistic = t_statistics;
results_table.Raw_p_value = p_values_ttest;
results_table.FDR_corrected_p = p_fdr;
results_table.Cohens_d = effect_sizes;
results_table.Significant_Raw = p_values_ttest < 0.05;
results_table.Significant_FDR = p_fdr < 0.05;

% Sort by FDR p-value
results_table = sortrows(results_table, 'FDR_corrected_p');

% Save table
writetable(results_table, fullfile(output_dir, 'Feature_Drop_Statistical_Results.csv'));

% Save figures
saveas(fig1, fullfile(output_dir, 'Feature_Importance_R2_Drop_Statistical.png'), 'png');
saveas(fig1, fullfile(output_dir, 'Feature_Importance_R2_Drop_Statistical.svg'), 'svg');

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Results saved to:\n');
fprintf('- Feature_Drop_Statistical_Results.csv\n');
fprintf('- Statistical visualization figures\n');

if ~isempty(sig_features_idx)
    fprintf('\nMost important features (FDR significant):\n');
    top_results = results_table(results_table.Significant_FDR, :);
    for i = 1:height(top_results)
        fprintf('%d. %s: R² drop = %.4f, FDR p = %.6f\n', ...
            i, top_results.Feature{i}, top_results.Mean_R2_Drop(i), top_results.FDR_corrected_p(i));
    end
else
    fprintf('\nNo features showed statistically significant importance after FDR correction.\n');
    fprintf('Consider examining the top features by effect size instead.\n');
end

%% SHAP Values vs Dominant Frequency Scatter Plot
% Configuration
datapath = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10ClustersRegularized4/*reward_*Ypred_full*';
datapath2 = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10Clusters4';
dirpath = dir(datapath);
output_dir = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SHAPAnalysis';
struct_path = '/Users/hsiehkunlin/Desktop/Data/Struct';

if ~exist(output_dir, 'dir'), mkdir(output_dir); end

snr_threshold_1 = 1; % photometry SNR threshold
snr_threshold_2 = 1; % breathing SNR threshold

feature_names = {'Dominant Freq', 'X Position', 'Y Position', 'Z Position', ...
    'Speed', 'tSNE-X', 'tSNE-Y', 'IR1', 'IR2', 'WP1', 'WP2', 'Shock', 'Sound'};

dom_freq_idx = 1; % Dominant Freq is first feature
fprintf('Loading data from %d sessions...\n', length(dirpath));

% Data Collection
all_shap_values = [];  % All SHAP values for dominant freq feature (time points x sessions)
allsnr = [];
all_dominant_freq = [];  % All dominant freq values (time points x sessions)
session_info = [];  % Track which session each time point belongs to

for SessionID = 1:length(dirpath)
    fprintf('Processing session %d/%d\n', SessionID, length(dirpath));
    
    file1 = dirpath(SessionID).name;
    file2 = dirpath(SessionID).name;
    file2(end-14:end) = [];
    file = file2;
    file2 = strcat(file2, "_SHAP_full.csv");
    path = dirpath(SessionID).folder;
    
    % Load SHAP values (time points x features)
    shap_values = readtable(fullfile(path, file2));
    shap_values = table2array(shap_values);
    
    % Extract SHAP values for dominant frequency feature only
    shap_dom_freq = shap_values(:, dom_freq_idx);  % Keep original sign
    all_shap_values = [all_shap_values; shap_dom_freq];
    
    % Load SNR data
    ALLStructFile_temp = load(fullfile(struct_path, file));
    SNR = ALLStructFile_temp.SNR;
    allsnr = [allsnr; SNR];
    
    % Extract dominant frequency values (time points x 1)
    dom_freq_val_temp = readtable(fullfile(datapath2, file));
    dom_freq_val_temp = table2array(dom_freq_val_temp);
    dom_freq_val = dom_freq_val_temp(:,1);  % First column is dominant freq
    all_dominant_freq = [all_dominant_freq; dom_freq_val];
    
    % Track session ID for each time point
    session_info = [session_info; repmat(SessionID, length(dom_freq_val), 1)];
end

fprintf('Total time points across all sessions: %d\n', length(all_dominant_freq));

% Categorize sessions by SNR quality
ind1 = allsnr(:,1) > snr_threshold_1 & allsnr(:,3) > snr_threshold_2;  % both good
ind2 = allsnr(:,1) > snr_threshold_1 & allsnr(:,3) <= snr_threshold_2; % bad breathing
ind3 = allsnr(:,1) <= snr_threshold_1 & allsnr(:,3) > snr_threshold_2;  % bad photometry
ind4 = allsnr(:,1) <= snr_threshold_1 & allsnr(:,3) <= snr_threshold_2; % both bad

group_names = {'Both Good', 'Bad Breathing', 'Bad Photometry', 'Both Bad'};
group_colors = {[0.2 0.6 0.2], [0.4 0.4 0.8], [1.0 0.6 0.0], [0.8 0.2 0.2]};
markers = {'o', 's', '^', 'd'};

fprintf('Session distribution:\n');
fprintf('Both good SNR: %d sessions\n', sum(ind1));
fprintf('Bad breathing: %d sessions\n', sum(ind2));
fprintf('Bad photometry: %d sessions\n', sum(ind3));
fprintf('Both bad SNR: %d sessions\n', sum(ind4));

% Create group labels for each time point based on session SNR quality
time_point_groups = zeros(size(session_info));
for i = 1:length(session_info)
    session_idx = session_info(i);
    if ind1(session_idx)
        time_point_groups(i) = 1;
    elseif ind2(session_idx)
        time_point_groups(i) = 2;
    elseif ind3(session_idx)
        time_point_groups(i) = 3;
    elseif ind4(session_idx)
        time_point_groups(i) = 4;
    end
end

% Remove invalid time points
valid_points = time_point_groups > 0 & ~isnan(all_dominant_freq) & ~isnan(all_shap_values);
all_dominant_freq = all_dominant_freq(valid_points);
all_shap_values = all_shap_values(valid_points);
time_point_groups = time_point_groups(valid_points);
session_info = session_info(valid_points);

fprintf('Valid time points for analysis: %d\n', sum(valid_points));

% Define breathing rate categories
breath_rate_low = all_dominant_freq < 2;      % Low: < 2 Hz
breath_rate_mid = all_dominant_freq >= 2 & all_dominant_freq <= 5;  % Mid: 2-5 Hz
breath_rate_high = all_dominant_freq > 5;     % High: > 5 Hz

fprintf('Breathing rate distribution:\n');
fprintf('Low (< 2 Hz): %d points\n', sum(breath_rate_low));
fprintf('Mid (2-5 Hz): %d points\n', sum(breath_rate_mid));
fprintf('High (> 5 Hz): %d points\n', sum(breath_rate_high));

% Create four subplots for four SNR groups
fig = figure('Position', [100, 100, 1600, 1200]);
set(fig, 'Color', 'white');

% Regression colors for breathing rate categories
reg_colors = {[0.8 0.2 0.8], [0.2 0.8 0.8], [0.8 0.8 0.2]};  % Purple, Cyan, Yellow
reg_names = {'Low (< 2 Hz)', 'Mid (2-5 Hz)', 'High (> 5 Hz)'};

for group = 1:4
    subplot(2, 2, group);
    hold on;
    
    % Get data for this SNR group
    group_idx = time_point_groups == group;
    
    if sum(group_idx) > 0
        freq_data = all_dominant_freq(group_idx);
        shap_data = all_shap_values(group_idx);
        
        fprintf('\n=== %s Group ===\n', group_names{group});
        fprintf('Total points: %d\n', sum(group_idx));
        
        % Subsample for visualization if too many points
        max_points = 5000;
        if length(freq_data) > max_points
            sample_idx = randsample(length(freq_data), max_points);
            freq_plot = freq_data(sample_idx);
            shap_plot = shap_data(sample_idx);
            fprintf('Subsampled to %d points for visualization\n', max_points);
        else
            freq_plot = freq_data;
            shap_plot = shap_data;
        end
        
        % Plot scatter points
        scatter(freq_plot, shap_plot, 15, group_colors{group}, 'filled', ...
            'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'none');
        
        % Three regression analyses for different breathing rate ranges
        legend_handles = [];
        legend_labels = {};
        
        % Low breathing rate regression (< 2 Hz)
        low_idx = freq_data < 2;
        if sum(low_idx) > 10
            freq_low = freq_data(low_idx);
            shap_low = shap_data(low_idx);
            
            p_low = polyfit(freq_low, shap_low, 1);
            x_low = linspace(min(freq_low), max(freq_low), 100);
            y_low = polyval(p_low, x_low);
            
            h1 = plot(x_low, y_low, '-', 'Color', reg_colors{1}, 'LineWidth', 3);
            
            [r_low, p_val_low] = corr(freq_low, shap_low);
            
            fprintf('Low rate (< 2 Hz): n=%d, r=%.3f, p=%.6f\n', sum(low_idx), r_low, p_val_low);
            
            legend_handles = [legend_handles, h1];
            legend_labels = [legend_labels, {sprintf('%s (r=%.2f)', reg_names{1}, r_low)}];
        end
        
        % Mid breathing rate regression (2-5 Hz)
        mid_idx = freq_data >= 2 & freq_data <= 5;
        if sum(mid_idx) > 10
            freq_mid = freq_data(mid_idx);
            shap_mid = shap_data(mid_idx);
            
            p_mid = polyfit(freq_mid, shap_mid, 1);
            x_mid = linspace(min(freq_mid), max(freq_mid), 100);
            y_mid = polyval(p_mid, x_mid);
            
            h2 = plot(x_mid, y_mid, '-', 'Color', reg_colors{2}, 'LineWidth', 3);
            
            [r_mid, p_val_mid] = corr(freq_mid, shap_mid);
            
            fprintf('Mid rate (2-5 Hz): n=%d, r=%.3f, p=%.6f\n', sum(mid_idx), r_mid, p_val_mid);
            
            legend_handles = [legend_handles, h2];
            legend_labels = [legend_labels, {sprintf('%s (r=%.2f)', reg_names{2}, r_mid)}];
        end
        
        % High breathing rate regression (> 5 Hz)
        high_idx = freq_data > 5;
        if sum(high_idx) > 10
            freq_high = freq_data(high_idx);
            shap_high = shap_data(high_idx);
            
            p_high = polyfit(freq_high, shap_high, 1);
            x_high = linspace(min(freq_high), max(freq_high), 100);
            y_high = polyval(p_high, x_high);
            
            h3 = plot(x_high, y_high, '-', 'Color', reg_colors{3}, 'LineWidth', 3);
            
            [r_high, p_val_high] = corr(freq_high, shap_high);
            
            fprintf('High rate (> 5 Hz): n=%d, r=%.3f, p=%.6f\n', sum(high_idx), r_high, p_val_high);
            
            legend_handles = [legend_handles, h3];
            legend_labels = [legend_labels, {sprintf('%s (r=%.2f)', reg_names{3}, r_high)}];
        end
        
        % Add vertical lines to show breathing rate boundaries
        y_limits = ylim;
        line([2, 2], y_limits, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 1);
        line([5, 5], y_limits, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 1);
        
        % Formatting
        xlabel('Dominant Breathing Frequency (Hz)', 'FontWeight', 'bold');
        ylabel('SHAP Value', 'FontWeight', 'bold');
        title(sprintf('%s (n=%d)', group_names{group}, sum(group_idx)), ...
            'FontSize', 14, 'FontWeight', 'bold', 'Color', group_colors{group});
        
        if ~isempty(legend_handles)
            legend(legend_handles, legend_labels, 'Location', 'best', 'FontSize', 9);
        end
        
        grid on;
        set(gca, 'GridAlpha', 0.3);
        
        % Set consistent axis limits across subplots
        xlim([min(all_dominant_freq) - 0.1, max(all_dominant_freq) + 0.1]);
        ylim([min(all_shap_values) - 0.01, max(all_shap_values) + 0.01]);
    else
        % Empty subplot for groups with no data
        text(0.5, 0.5, sprintf('No data for %s', group_names{group}), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', 14, 'Units', 'normalized');
        title(group_names{group}, 'FontSize', 14, 'FontWeight', 'bold');
    end
end

sgtitle('SHAP vs Dominant Breathing Frequency: SNR Groups with Rate-Specific Regression', ...
    'FontSize', 16, 'FontWeight', 'bold');

% Create summary statistics table
summary_stats = table();
row_idx = 1;

for group = 1:4
    group_idx = time_point_groups == group;
    
    if sum(group_idx) > 0
        freq_data = all_dominant_freq(group_idx);
        shap_data = all_shap_values(group_idx);
        
        % Overall group statistics
        summary_stats.Group{row_idx} = group_names{group};
        summary_stats.Rate_Category{row_idx} = 'Overall';
        summary_stats.N_Points(row_idx) = sum(group_idx);
        [r_overall, p_overall] = corr(freq_data, shap_data);
        summary_stats.Correlation(row_idx) = r_overall;
        summary_stats.P_Value(row_idx) = p_overall;
        summary_stats.Mean_Freq(row_idx) = mean(freq_data);
        summary_stats.Mean_SHAP(row_idx) = mean(shap_data);
        row_idx = row_idx + 1;
        
        % Low rate statistics
        low_idx = freq_data < 2;
        if sum(low_idx) > 10
            summary_stats.Group{row_idx} = group_names{group};
            summary_stats.Rate_Category{row_idx} = 'Low (< 2 Hz)';
            summary_stats.N_Points(row_idx) = sum(low_idx);
            [r_low, p_low] = corr(freq_data(low_idx), shap_data(low_idx));
            summary_stats.Correlation(row_idx) = r_low;
            summary_stats.P_Value(row_idx) = p_low;
            summary_stats.Mean_Freq(row_idx) = mean(freq_data(low_idx));
            summary_stats.Mean_SHAP(row_idx) = mean(shap_data(low_idx));
            row_idx = row_idx + 1;
        end
        
        % Mid rate statistics
        mid_idx = freq_data >= 2 & freq_data <= 5;
        if sum(mid_idx) > 10
            summary_stats.Group{row_idx} = group_names{group};
            summary_stats.Rate_Category{row_idx} = 'Mid (2-5 Hz)';
            summary_stats.N_Points(row_idx) = sum(mid_idx);
            [r_mid, p_mid] = corr(freq_data(mid_idx), shap_data(mid_idx));
            summary_stats.Correlation(row_idx) = r_mid;
            summary_stats.P_Value(row_idx) = p_mid;
            summary_stats.Mean_Freq(row_idx) = mean(freq_data(mid_idx));
            summary_stats.Mean_SHAP(row_idx) = mean(shap_data(mid_idx));
            row_idx = row_idx + 1;
        end
        
        % High rate statistics
        high_idx = freq_data > 5;
        if sum(high_idx) > 10
            summary_stats.Group{row_idx} = group_names{group};
            summary_stats.Rate_Category{row_idx} = 'High (> 5 Hz)';
            summary_stats.N_Points(row_idx) = sum(high_idx);
            [r_high, p_high] = corr(freq_data(high_idx), shap_data(high_idx));
            summary_stats.Correlation(row_idx) = r_high;
            summary_stats.P_Value(row_idx) = p_high;
            summary_stats.Mean_Freq(row_idx) = mean(freq_data(high_idx));
            summary_stats.Mean_SHAP(row_idx) = mean(shap_data(high_idx));
            row_idx = row_idx + 1;
        end
    end
end

% Save results
scatter_data = table();
scatter_data.Session_ID = session_info;
scatter_data.Time_Point = (1:length(all_dominant_freq))';
scatter_data.SNR_Group = categorical(time_point_groups, 1:4, group_names);
scatter_data.Dominant_Frequency = all_dominant_freq;
scatter_data.SHAP_Value = all_shap_values;
scatter_data.Rate_Category = categorical(zeros(length(all_dominant_freq), 1), 1:3, reg_names);

% Assign rate categories
scatter_data.Rate_Category(breath_rate_low) = categorical(1, 1:3, reg_names);
scatter_data.Rate_Category(breath_rate_mid) = categorical(2, 1:3, reg_names);
scatter_data.Rate_Category(breath_rate_high) = categorical(3, 1:3, reg_names);

writetable(scatter_data, fullfile(output_dir, 'SHAP_vs_DominantFreq_FourGroups.csv'));
writetable(summary_stats, fullfile(output_dir, 'SHAP_Regression_Summary_Stats.csv'));
saveas(fig, fullfile(output_dir, 'SHAP_vs_DominantFreq_FourGroups.png'), 'png');
saveas(fig, fullfile(output_dir, 'SHAP_vs_DominantFreq_FourGroups.svg'), 'svg');

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Results saved to: %s\n', output_dir);
fprintf('Files created:\n');
fprintf('- SHAP_vs_DominantFreq_FourGroups.csv (all data points)\n');
fprintf('- SHAP_Regression_Summary_Stats.csv (correlation statistics)\n');
fprintf('- Figure files (.png and .svg)\n');


%% R² Comparison Analysis: SNR Groups vs Breathing Rates
% Simple R² Line Chart and ANOVA for 4 Groups × 3 Breathing Rates

% Calculate R² for each group and rate combination
group_names = {'Both Good', 'Bad Breathing', 'Bad Photometry', 'Both Bad'};
rate_names = {'Low (<2Hz)', 'Mid (2-5Hz)', 'High (>5Hz)'};
group_colors = {[0.2 0.6 0.2], [0.4 0.4 0.8], [1.0 0.6 0.0], [0.8 0.2 0.2]};

r2_matrix = zeros(4, 3);  % 4 groups × 3 rates
n_matrix = zeros(4, 3);   % sample sizes

% Calculate R² for each combination
for group = 1:4
    group_idx = time_point_groups == group;
    
    if sum(group_idx) > 10
        freq_data = all_dominant_freq(group_idx);
        shap_data = all_shap_values(group_idx);
        
        % Three rate categories
        rate_indices = {freq_data < 2, freq_data >= 2 & freq_data <= 5, freq_data > 5};
        
        for rate = 1:3
            rate_idx = rate_indices{rate};
            if sum(rate_idx) > 10
                r = corr(freq_data(rate_idx), shap_data(rate_idx));
                r2_matrix(group, rate) = r^2;
                n_matrix(group, rate) = sum(rate_idx);
            end
        end
    end
end

% Create line chart
fig = figure('Position', [100, 100, 1000, 600]);
set(fig, 'Color', 'white');

hold on;
legend_handles = [];

for group = 1:4
    valid_rates = r2_matrix(group, :) > 0;
    if sum(valid_rates) > 0
        x_vals = find(valid_rates);
        y_vals = r2_matrix(group, valid_rates);
        
        h = plot(x_vals, y_vals, 'o-', 'Color', group_colors{group}, ...
            'LineWidth', 3, 'MarkerSize', 8, 'MarkerFaceColor', group_colors{group}, ...
            'DisplayName', group_names{group});
        legend_handles = [legend_handles, h];
        
        % Add sample size labels
        for i = 1:length(x_vals)
            text(x_vals(i), y_vals(i) + 0.01, sprintf('n=%d', n_matrix(group, x_vals(i))), ...
                'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', group_colors{group});
        end
    end
end

set(gca, 'XTick', 1:3, 'XTickLabel', rate_names);
xlabel('Breathing Rate Category', 'FontWeight', 'bold');
ylabel('R² (Correlation Strength)', 'FontWeight', 'bold');
title('R² by SNR Group and Breathing Rate', 'FontSize', 14, 'FontWeight', 'bold');
legend(legend_handles, 'Location', 'best');
grid on;
ylim([0, max(r2_matrix(:)) * 1.2]);

% Save results
saveas(fig, fullfile(output_dir, 'R2_Line_Chart.png'), 'png');
saveas(fig, fullfile(output_dir, 'R2_Line_Chart.svg'), 'svg');

% Save data table
results_table = table();
row = 1;
for group = 1:4
    for rate = 1:3
        if r2_matrix(group, rate) > 0
            results_table.SNR_Group{row} = group_names{group};
            results_table.Breathing_Rate{row} = rate_names{rate};
            results_table.R_squared(row) = r2_matrix(group, rate);
            results_table.Sample_Size(row) = n_matrix(group, rate);
            row = row + 1;
        end
    end
end

writetable(results_table, fullfile(output_dir, 'R2_Line_Chart_Data.csv'));

fprintf('\nR² Summary:\n');
fprintf('Group\t\tLow\tMid\tHigh\n');
for group = 1:4
    fprintf('%-12s\t%.2f\t%.2f\t%.2f\n', group_names{group}, r2_matrix(group, :));
end

fprintf('\nFiles saved: R2_Line_Chart_ANOVA.png/.svg and R2_Line_Chart_Data.csv\n');