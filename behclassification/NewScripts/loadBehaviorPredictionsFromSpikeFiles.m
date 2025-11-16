function [prediction_sessions] = loadBehaviorPredictionsFromSpikeFiles(allfiles, prediction_folder)
% loadBehaviorPredictionsFromSpikeFiles - Load predictions matched to spike structure files
%
% This function takes the output from selectFilesWithAnimalIDFiltering and loads
% the corresponding behavior prediction files from the prediction folder.
%
% Input:
%   allfiles - struct array from selectFilesWithAnimalIDFiltering
%              Each element has .name field with the spike structure filename
%   prediction_folder - folder containing *_BehaviorPrediction.mat files
%
% Output:
%   prediction_sessions - array of prediction structures
%                        Each element contains:
%                          .filename
%                          .animal_id
%                          .session_date
%                          .predictions
%                          .prediction_scores
%                          .frame_info
%                          .num_patches
%                          .num_subpatches
%                          .selected_features
%                          .behavior_classes
%
% Example:
%   [allfiles, ~, ~, ~] = selectFilesWithAnimalIDFiltering(...
%       '/Volumes/Expansion/Data/Struct_spike', 2, '2025*RewardAversive*.mat');
%   prediction_folder = './BehaviorPrediction';
%   prediction_sessions = loadBehaviorPredictionsFromSpikeFiles(allfiles, prediction_folder);

    fprintf('Loading behavior predictions matched to spike structure files...\n');
    fprintf('  Spike files: %d\n', length(allfiles));
    fprintf('  Prediction folder: %s\n', prediction_folder);

    % Check if prediction folder exists
    if ~exist(prediction_folder, 'dir')
        error('Prediction folder does not exist: %s', prediction_folder);
    end

    % Initialize output array
    prediction_sessions = [];

    loaded_count = 0;
    missing_count = 0;

    % Process each spike file
    for i = 1:length(allfiles)
        spike_filename = allfiles(i).name;

        % Construct prediction filename from spike filename
        % Remove .mat extension, add _BehaviorPrediction.mat
        [~, base_name, ~] = fileparts(spike_filename);
        pred_filename = sprintf('%s_BehaviorPrediction.mat', base_name);
        pred_filepath = fullfile(prediction_folder, pred_filename);

        % Try to load the prediction file
        if exist(pred_filepath, 'file')
            try
                % Load the prediction file
                pred_data = load(pred_filepath);

                % Extract session_prediction structure
                if isfield(pred_data, 'session_prediction')
                    session_pred = pred_data.session_prediction;

                    % Add to array
                    if i == 1
                        prediction_sessions = session_pred;
                    else
                        prediction_sessions(i) = session_pred;
                    end

                    loaded_count = loaded_count + 1;
                    fprintf('  ✅ [%d/%d] Loaded: %s\n', i, length(allfiles), pred_filename);

                else
                    fprintf('  ⚠️  [%d/%d] %s: No session_prediction field\n', ...
                        i, length(allfiles), pred_filename);
                    missing_count = missing_count + 1;
                    prediction_sessions(i) = createEmptyPrediction(spike_filename);
                end

            catch ME
                fprintf('  ❌ [%d/%d] %s: Load failed - %s\n', ...
                    i, length(allfiles), pred_filename, ME.message);
                missing_count = missing_count + 1;
                prediction_sessions(i) = createEmptyPrediction(spike_filename);
            end

        else
            fprintf('  ⚠️  [%d/%d] %s: File not found\n', ...
                i, length(allfiles), pred_filename);
            missing_count = missing_count + 1;
            prediction_sessions(i) = createEmptyPrediction(spike_filename);
        end
    end

    fprintf('\n✓ Summary: Loaded %d/%d predictions (%d missing)\n\n', ...
        loaded_count, length(allfiles), missing_count);
end

function empty_pred = createEmptyPrediction(filename)
% Create an empty prediction structure for missing sessions

    empty_pred = struct();
    empty_pred.filename = filename;
    empty_pred.animal_id = 'unknown';
    empty_pred.session_date = 'unknown';
    empty_pred.predictions = [];
    empty_pred.prediction_scores = [];
    empty_pred.frame_info = [];
    empty_pred.num_patches = 0;
    empty_pred.num_subpatches = 0;
    empty_pred.selected_features = [];
    empty_pred.behavior_classes = {};
end
