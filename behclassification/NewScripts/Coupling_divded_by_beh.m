clear all
close all

%%
numofsession = 2;
folderpath = "/Volumes/Expansion/Data/Struct_spike";
% [allfiles, folderpath, num_sessions] = selectFilesWithAnimalIDFiltering(folderpath,numofsession,'2025*RewardSeeking*.mat');
[allfiles, folderpath, num_sessions] = selectFilesWithAnimalIDFiltering(folderpath,numofsession,'2025*RewardAversive*.mat');

% load('lstm_prediction_results_aversive.mat', 'final_results', 'processing_summary')
% load('lstm_prediction_results_reward.mat', 'final_results', 'processing_summary')
load('lstm_prediction_results_aversive_27-Oct-2025.mat', 'final_results', 'processing_summary')
% load('lstm_prediction_results_reward_27-Oct-2025.mat', 'final_results', 'processing_summary')
%% Configuration Parameters
% Analysis parameters
config.bp_range = [1 300];     % Bandpass filter range
config.min_duration = 2;       % Minimum duration in seconds for breathing events

% Breathing frequency bands
config.f_slow = [0.5, 3];      % slow breathing (0.5-3 Hz)
config.f_mid = [3, 6];         % medium breathing (3-6 Hz)
config.f_fast = [6, 15];       % fast breathing (6-15 Hz)

%% Load Sorting Parameters
[T_sorted] = loadSortingParameters();

%% Initialize results structure
% Create output directory if it doesn't exist
output_dir = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/corss_modu_beh';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Define behavior labels
beh_labels = {'1.Reward', '2.Walking', '3.Rearing', '4.Scanning/Air-Sniff', ...
              '5.Ground-Sniff', '6.Grooming', '7.Standing/Immobility'};

%% Process Each Session with Data Collection
for sessionID = 1:num_sessions
    fprintf('\n==== Processing session %d/%d: %s ====\n', sessionID, num_sessions, allfiles(sessionID).name);
   
    %% Load and preprocess session data
    Timelimits = 'No';
    [NeuralTime, ~, ~, Signal, ~, ~, ~, ~, AversiveSound, ~, ~, Fs, TriggerMid] = loadAndPrepareSessionData(allfiles(sessionID), T_sorted, Timelimits);
    
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);
    bestChannel = findBestLFPChannel(filtered_data, Fs);
    LFP = filtered_data(:, bestChannel);
    breathing = filtered_data(:, 32);

    %% Breathing-to-LFP Cross-Frequency Coupling
    fprintf('  Computing Breathing-to-LFP cross-frequency coupling...\n');
    
    % Signal preparation
    gammarange = 1:1:120;
    thetarange = 0.5:0.25:15;
    
    % Extract features
    [~, gammaamps] = multiphasevec(gammarange, LFP, Fs, 8);
    thetaangles = extractThetaPhase(breathing, Fs, 'wavelet', thetarange);
    
    %% Get behavior labels
    labels = interp1(TriggerMid, repelem(final_results.session_predictions(sessionID).predictions, 20), NeuralTime, 'nearest');
    
    % Create figure for this session with all behaviors
    fig = figure('Position', [100, 100, 1800, 900]);
    sgtitle(sprintf('Session %d: %s - Cross-frequency coupling for all behaviors', sessionID, allfiles(sessionID).name), ...
            'Interpreter', 'none', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Setup parallel pool with 5 workers (do this once per session)
    if isempty(gcp('nocreate'))
        pool = parpool(5);
    else
        pool = gcp;
        if pool.NumWorkers ~= 5
            delete(pool);
            pool = parpool(5);
        end
    end
    
    % Store z-scored MI for colormap scaling
    all_z_scored_MI = [];
    
    %% Process each behavior
    for beh_id = 1:7
        fprintf('    Processing behavior %d: %s\n', beh_id, beh_labels{beh_id});
        
        % Select data for specific behavior
        beh_ind = labels == beh_id;
        
        if sum(beh_ind) < 1000  % Skip if too few samples
            fprintf('      Warning: Too few samples for behavior %d (only %d samples)\n', beh_id, sum(beh_ind));
            subplot(2, 4, beh_id);
            text(0.5, 0.5, sprintf('Insufficient data\n(%d samples)', sum(beh_ind)), ...
                 'HorizontalAlignment', 'center', 'FontSize', 10);
            axis off;
            continue;
        end
        
        % Subsample if necessary
        if sum(beh_ind) > 500000
            rnd_ind = randperm(length(find(beh_ind)), 500000);
            beh_ind_full = find(beh_ind);
            selected_ind = beh_ind_full(rnd_ind);
            gammaamps_teim = gammaamps(:, selected_ind);
            thetaangles_teim = thetaangles(:, selected_ind);
        else
            gammaamps_teim = gammaamps(:, beh_ind);
            thetaangles_teim = thetaangles(:, beh_ind);
        end
        
        % Compute MI Comodulation
        nbins = 36;
        [modindex, meanamps, bincenters] = computeModIndex(gammaamps_teim, thetaangles_teim, nbins);
        
        % Perform shuffling analysis
        shuffles = 10;
        fprintf('      Running %d shuffles for behavior %d...\n', shuffles, beh_id);
        
        if shuffles > 0
            rng('shuffle');
            offset = randi((size(thetaangles_teim,2)-2), 1, shuffles) + 1;
            
            % Pre-allocate futures
            futures(shuffles) = parallel.FevalFuture;
            
            % Submit jobs
            for i = 1:shuffles
                ta = [thetaangles_teim(:, offset(1,i)+1:end), thetaangles_teim(:, 1:offset(1,i))];
                futures(i) = parfeval(pool, @computeModIndex, 1, gammaamps_teim, ta, nbins);
            end
            
            % Collect results
            modindex_out = zeros(size(gammaamps_teim,1), size(thetaangles_teim,1), shuffles);
            for i = 1:shuffles
                [completedIdx, value] = fetchNext(futures);
                modindex_out(:,:,completedIdx) = value;
            end
            
            modindex_shuffled = nanmean(modindex_out, 3);
            modindex_std = nanstd(modindex_out, [], 3);
        else
            modindex_shuffled = NaN(size(modindex));
            modindex_std = NaN(size(modindex));
        end
        
        % Calculate z-scored MI
        z_scored_MI = (modindex - modindex_shuffled) ./ modindex_std;
        all_z_scored_MI = cat(3, all_z_scored_MI, z_scored_MI);
        
        % Plot in subplot
        subplot(2, 4, beh_id);
        imagesc(thetarange, gammarange, z_scored_MI);
        axis xy;
        colorbar;
        ylim([gammarange(1), gammarange(end)]);
        xlim([thetarange(1), thetarange(end)]);
        title(sprintf('%s (n=%d)', beh_labels{beh_id}, sum(beh_ind)));
        xlabel('Breathing Freq (Hz)');
        ylabel('LFP Freq (Hz)');
        
        % Add grid for better visualization
        grid on;
        set(gca, 'GridAlpha', 0.3);
    end
    
    % Add overall behavior distribution pie chart in the 8th subplot
    subplot(2, 4, 8);
    counts = accumarray(labels(~isnan(labels)), 1);
    pie(counts, beh_labels);
    title('Behavior Distribution');
    
    % Set consistent colormap scale across all subplots
    if ~isempty(all_z_scored_MI)
        z_min = min(all_z_scored_MI(:));
        z_max = max(all_z_scored_MI(:));
        for subplot_idx = 1:7
            subplot(2, 4, subplot_idx);
            caxis([z_min, z_max]);
        end
    end
    
    % Save figure
    filename = sprintf('Session_%02d_%s_CrossFreqCoupling_AllBehaviors.png', sessionID, allfiles(sessionID).name(1:end-4));
    filename = fullfile(output_dir, filename);
    saveas(fig, filename);
    fprintf('    Figure saved: %s\n', filename);
    
    % Also save as .fig for later editing
    fig_filename = strrep(filename, '.png', '.fig');
    savefig(fig, fig_filename);
    
    % Close figure to save memory
    close(fig);
end

% Clean up parallel pool
if ~isempty(gcp('nocreate'))
    delete(gcp);
end

fprintf('\n==== Analysis complete! ====\n');
fprintf('All figures saved to: %s\n', output_dir);