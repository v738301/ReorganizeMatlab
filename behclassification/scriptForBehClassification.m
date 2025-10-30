%%
% Load your data
structure_data = load('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/RewardAversive_session_metrics_breathing_LFPCcouple(10-1).mat');

% Check how many sessions you have
num_sessions = length(structure_data.all_session_metrics);
fprintf('Total sessions available: %d\n', num_sessions);

% Start with a subset of sessions (you can add more later)
sessions_to_process = [1:num_sessions];  % Adjust based on what you have

% Step 2: Do labeling sessions (saves to cache, no memory accumulation)
fprintf('Starting labeling sessions...\n');
analysis_type = 3;  % AfterAversive

% lstm_data = collectCachedLabelsForTraining(structure_data, sessions_to_process, analysis_type);
% save('lstm_training_data.mat', 'lstm_data', '-v7.3');

%% normalized the data
lstm_data = load('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/lstm_training_data.mat');
lstm_data_reduced = lstm_data.lstm_data;
lstm_data_reduced = rmfield(lstm_data_reduced,{'num_features','session_info','subpatch_ids','video_frame_indices','dataset_info','sequence_stats','class_distribution'});

% normalize by feature 70
time_series = {};
for i = 1:length(lstm_data_reduced.time_series)
    time_series{i} = lstm_data_reduced.time_series{i}./lstm_data_reduced.time_series{i}(:,70);
end
normalize_data = struct();
normalize_data.time_series = time_series;
normalize_data.labels = lstm_data_reduced.labels;
normalize_data.behavior_classes = lstm_data_reduced.behavior_classes;
normalize_data.sequence_session_map = lstm_data_reduced.sequence_session_map;

% relabel 4 and 7 classes
options = struct();
options.energy_metric = 'variance_based';  % This is now the default
options.threshold_method = 'manual';  % 'optimal', 'median', 'manual'
options.manual_threshold = 1;
[new_data_structure, relabeling_report] = relabelByTotalEnergy(normalize_data, options);

%% feature selection
% selected_features = [5, 6, 7, 9, 13, 14, 15, 16, 17, 20, 23, 30, 33, 35, 36, 37, 38, 44, 45, 46, 47, 48, 52, 74, 75, 89, 125, 135, 138, 143];
% common_features = [5, 6, 7, 9, 13, 16, 20, 27, 28, 39, 40, 42, 43, 44, 48, 79, 82, 90, 103, 141, 142, 145, 147]
[results] = compare_feature_selection(new_data_structure);
[model, accuracy, confusion_matrix] = train_svm_common_features(new_data_structure, results);

%% reduce feature number and add new feature
for i = 1:length(new_data_structure.time_series)
    new_data_structure.time_series{i} = new_data_structure.time_series{i}(:,results.common_features);
end

% add temporal variance feature
options = struct();
options.variance_feature_types = {'total_temporal_variance','total_variance'};
options.normalize_features = false;
enhanced_data = addVarianceFeature(new_data_structure, options);

%% training 
fast_options = struct();
fast_options.skip_cross_validation = true;    % 🔥 SKIP CV FOR SPEED
[trained_net, validation_report] = trainDeepLSTMClassifier(enhanced_data,fast_options);
% save('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/trained_net_new.mat','trained_net','validation_report')
%% LSTM Model Evaluation and Confusion Matrix
% This code evaluates the trained LSTM model on test data and creates a confusion matrix
load('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/trained_net_new.mat')
% Extract test data and labels
test_data = enhanced_data.time_series;
true_labels = enhanced_data.labels;
test_data_in = {};
for i = 1:length(test_data)
    test_data_in{i} = test_data{i}';
end

% Make predictions using the trained network
predicted_labels = predict(trained_net, test_data_in);

% Convert categorical arrays to numeric if needed
if iscategorical(true_labels)
    true_labels_num = double(true_labels);
else
    true_labels_num = true_labels;
end

if iscategorical(predicted_labels)
    [~,ind] = max(predicted_labels,[],2);
    predicted_labels_num = double(ind);
else
    [~,ind] = max(predicted_labels,[],2);
    predicted_labels_num = ind;
end

% Calculate overall accuracy
accuracy = sum(predicted_labels_num(:) == true_labels_num(:)) / length(true_labels_num);
fprintf('Overall Accuracy: %.2f%%\n', accuracy * 100);

% Ensure vectors are column vectors and same length
true_labels_vec = true_labels_num(:);
predicted_labels_vec = predicted_labels_num(:);

% Check if lengths match
if length(true_labels_vec) ~= length(predicted_labels_vec)
    error('True labels and predicted labels have different lengths: %d vs %d', ...
          length(true_labels_vec), length(predicted_labels_vec));
end

% Define behavior classes
behavior_classes = {'Reward', 'Walking', 'Rearing', 'Scanning/Air-Sniff', 'Ground-Sniff', 'Grooming', 'Standing/Immobility'};
num_classes = length(behavior_classes);
confusion_matrix = zeros(num_classes, num_classes);

% Fill confusion matrix
for i = 1:length(true_labels_vec)
    true_class = true_labels_vec(i);
    pred_class = predicted_labels_vec(i);
    confusion_matrix(true_class, pred_class) = confusion_matrix(true_class, pred_class) + 1;
end

% Calculate percentages (row-normalized)
confusion_percentages = confusion_matrix ./ sum(confusion_matrix, 2) * 100;

% Create the plot
fig = figure('Position', [100, 100, 800, 600]);
imagesc(confusion_percentages);
colormap('hot');
colorbar;

% Add text annotations with percentages
for i = 1:num_classes
    for j = 1:num_classes
        text(j, i, sprintf('%.1f%%\n(%d)', confusion_percentages(i,j), confusion_matrix(i,j)), ...
             'HorizontalAlignment', 'center', ...
             'FontSize', 10, 'FontWeight', 'bold');
    end
end

% Customize plot
title('LSTM Behavior Classification Confusion Matrix', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Predicted Behavior Class', 'FontSize', 12);
ylabel('True Behavior Class', 'FontSize', 12);
set(gca, 'XTick', 1:num_classes, 'YTick', 1:num_classes);
set(gca, 'XTickLabel', behavior_classes, 'YTickLabel', behavior_classes);
xtickangle(45);  % Rotate x-axis labels for better readability
axis equal tight;

% Calculate per-class accuracy from our confusion matrix
class_accuracies = diag(confusion_matrix) ./ sum(confusion_matrix, 2);

for i = 1:num_classes
    fprintf('Class %d (%s) Accuracy: %.2f%%\n', i, behavior_classes{i}, class_accuracies(i) * 100);
end

% Display summary statistics
fprintf('\nSummary Statistics:\n');
fprintf('Mean per-class accuracy: %.2f%%\n', mean(class_accuracies) * 100);
fprintf('Standard deviation: %.2f%%\n', std(class_accuracies) * 100);
fprintf('Total samples: %d\n', length(true_labels_num));

%% visulize predicted label and video
% Find a specific behavior sequence
reward_seqs = findSequencesByBehavior(lstm_data.lstm_data, 'Rearing', 10);
info = getVideoReferenceInfo(lstm_data.lstm_data, reward_seqs(10));

% Extract the exact video clip
fprintf('📹 Video clip location:\n');
fprintf('   Session: %d\n', info.session_index);
fprintf('   Animal: %s\n', info.animal_id);
fprintf('   Frames: %d to %d (%d frames)\n', info.start_frame, info.end_frame, info.duration_frames);
fprintf('   Behavior: %s\n', info.behavior_class);

% Get paths using existing logic
single_session = structure_data.all_session_metrics{info.session_index};
[animal_id, session_date, video_path, tracking_path] = parseFilenameAndGetPaths(single_session.filename);

% Extract and play video clip
if exist(video_path, 'file')
    fprintf('🎬 Playing video clip: %s, frames %d-%d\n', info.behavior_class, info.start_frame, info.end_frame);
    
    % Read video frames
    video_reader = VideoReader(video_path);
    frames = read(video_reader, [info.start_frame, info.end_frame]);
    
    % Play clip
    figure('Name', sprintf('Behavior: %s (Session %d)', info.behavior_class, info.session_index));
    for i = 1:size(frames, 4)
        imshow(frames(:,:,:,i));
        title(sprintf('%s - Frame %d/%d', info.behavior_class, i, size(frames, 4)));
        pause(0.1);  % Adjust speed
    end
else
    fprintf('❌ Video file not found: %s\n', video_path);
end

%% predict all session data
savepath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification';
savename = fullfile(savepath,'lstm_prediction_results_aversive_27-Oct-2025.mat');
[~, ~] = runCompleteLSTMPrediction('27-Oct-2025_RewardAversive_session_metrics_breathing_LFPCcouple(10-1).mat',...    % Your data file
    'trained_net_new.mat',...            % Your trained model  
    [1:24],...                               % Sessions to process
    3,...                                     % Analysis type
    savename ...
);

savename = fullfile(savepath,'lstm_prediction_results_reward_27-Oct-2025.mat');
[~, ~] = runCompleteLSTMPrediction('27-Oct-2025_RewardSeeking_session_metrics_breathing_LFPCcouple(10-1).mat',...    % Your data file
    'trained_net_new.mat',...            % Your trained model  
    [1:18],...                               % Sessions to process
    1,...                                     % Analysis type
    savename ...
);
%% Launch validation GUI
structure_data = load('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/RewardAversive_session_metrics_breathing_LFPCcouple(10-1).mat');
load('lstm_prediction_results_aversive.mat', 'final_results', 'processing_summary')
gui = createPredictionValidationGUI(final_results, structure_data);

% 3. Use the GUI to:
%    - Watch video and see predictions in real-time
%    - Navigate between sessions and time points
%    - Mark predictions as correct/incorrect/uncertai
%    - Review prediction confidence scores
