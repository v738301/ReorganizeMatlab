clear all
close all

%% Load Allanalysisstruct and datapath
paraPath = "/Users/hsiehkunlin/Desktop/Data/DannceData/parameters";
Allanalysisstructfile = fullfile(paraPath, "Allanalysisstructfile.mat");
load(Allanalysisstructfile,"Allanalysisstruct")
sorted_clust_ind = Allanalysisstruct.sorted_clust_ind;
T = Allanalysisstruct.T;
% combine this two T, sorted_clust_ind can used to turn cluster mode 2 -->  3
% T_sorted, convert cluster mode 2 --> 3
T_sorted = T(sorted_clust_ind);
reducedOn = 1;

%% Load structure
datapath = '/Users/hsiehkunlin/Desktop/Data/Struct/*creRD*.mat';
dirpath = dir(datapath);

%% identify last reward and first aversive
% Extract all sessions and identify last reward before first aversive
% filenames = {dirpath.name};
% dates = cellfun(@(x) regexp(x, '\d{4}-\d{2}-\d{2}', 'match', 'once'), filenames, 'UniformOutput', false);
% animalIDs = cellfun(@(x) regexp(x, 'cre[A-Z]{2}\d+', 'match', 'once'), filenames, 'UniformOutput', false);
% expNames = cellfun(@(x) regexp(x, 'cre[A-Z]{2}\d+_(.+?)_AllStruct_CameraTime\.mat', 'tokens', 'once'), filenames, 'UniformOutput', false);
% 
% validIdx = ~cellfun(@isempty, dates) & ~cellfun(@isempty, animalIDs) & ~cellfun(@isempty, expNames);
% dates = datenum(dates(validIdx), 'yyyy-mm-dd');
% animalIDs = animalIDs(validIdx);
% expNames = cellfun(@(x) x{1}, expNames(validIdx), 'UniformOutput', false);
% 
% % Find sessions of interest for each animal
% aversiveIdx = contains(expNames, 'Aversive', 'IgnoreCase', true);
% rewardIdx = contains(expNames, 'reward', 'IgnoreCase', true) & ~aversiveIdx;
% uniqueAnimals = unique(animalIDs);
% firstAversiveIdx = false(size(dates));
% lastRewardIdx = false(size(dates));
% 
% for i = 1:length(uniqueAnimals)
%     animalIdx = strcmp(animalIDs, uniqueAnimals{i});
%     
%     % Find first aversive session
%     animalAversiveIdx = animalIdx(:) & aversiveIdx(:);
%     if any(animalAversiveIdx)
%         animalAversiveDates = dates(animalAversiveIdx);
%         [firstAversiveDate, minIdx] = min(animalAversiveDates);
%         animalAversiveIndices = find(animalAversiveIdx);
%         firstAversiveIdx(animalAversiveIndices(minIdx)) = true;
%         
%         % Find last reward session before first aversive
%         animalRewardIdx = animalIdx(:) & rewardIdx(:);
%         if any(animalRewardIdx)
%             animalRewardDates = dates(animalRewardIdx);
%             beforeAversiveRewards = animalRewardDates < firstAversiveDate;
%             if any(beforeAversiveRewards)
%                 [~, maxIdx] = max(animalRewardDates(beforeAversiveRewards));
%                 animalRewardIndices = find(animalRewardIdx);
%                 beforeAversiveIndices = animalRewardIndices(beforeAversiveRewards);
%                 lastRewardIdx(beforeAversiveIndices(maxIdx)) = true;
%             end
%         end
%     end
% end
% 
% % Filter to keep only sessions of interest
% keepIdx = firstAversiveIdx | lastRewardIdx;
% dates = dates(keepIdx);
% animalIDs = animalIDs(keepIdx);
% expNames = expNames(keepIdx);
% isFirstAversive = firstAversiveIdx(keepIdx);
% isLastReward = lastRewardIdx(keepIdx);
% 
% % Create visualization
% figure('Position', [100, 100, 1000, 600]);
% uniqueAnimals = unique(animalIDs);
% 
% for i = 1:length(uniqueAnimals)
%     animalIdx = strcmp(animalIDs, uniqueAnimals{i});
%     
%     % Plot last reward sessions (blue circles)
%     rewardIdx = animalIdx(:) & isLastReward(:);
%     if any(rewardIdx)
%         scatter(dates(rewardIdx), repmat(i, sum(rewardIdx), 1), 100, 'b', 'o', 'filled', 'MarkerEdgeColor', 'k');
%         hold on;
%     end
%     
%     % Plot first aversive sessions (red squares)
%     aversiveIdx = animalIdx(:) & isFirstAversive(:);
%     if any(aversiveIdx)
%         scatter(dates(aversiveIdx), repmat(i, sum(aversiveIdx), 1), 100, 'r', 's', 'filled', 'MarkerEdgeColor', 'k');
%         hold on;
%     end
% end
% 
% yticks(1:length(uniqueAnimals));
% yticklabels(uniqueAnimals);
% xlabel('Date');
% ylabel('Animal ID');
% title('Last Reward (Blue Circles) Before First Aversive (Red Squares)');
% datetick('x', 'yyyy-mm-dd');
% xtickangle(45);
% grid on;
% 
% % keep only last reward and first rewardAversive_onlyNoise sessions
% dirpath_filtered = dirpath(keepIdx);

%% collect all data from sessions
for SessionID = 1:length(dirpath)
    SessionID
    file = dirpath(SessionID).name;
    path = dirpath(SessionID).folder;
    ALLStructFile_temp = load(fullfile(path,file));
    CameraTime = ALLStructFile_temp.CameraTime;
    FS = ceil(1/mean(diff(CameraTime))); %ALLStructFile_temp.FS;
    if FS~=50
        warning('MyComponent:incorrectType','Error. \nFS ~= 50')
    end
    xtimeResolution = round(mean(diff(CameraTime)),2);
    com_denoised = ALLStructFile_temp.Com_denoised;
    Features = ALLStructFile_temp.appendage_lengths_high_intp;
    DeltaFF = ALLStructFile_temp.DeltaFF;
    tsnexy = double(ALLStructFile_temp.TsneZval);  

    if isfield(ALLStructFile_temp, 'Breath_sig_demedain_deoutliers_low_smo')
        breathing = ALLStructFile_temp.Breath_sig_demedain_deoutliers_low_smo;

        frequency_range = [0.1 20]; % You can adjust the lower bound if you want
        [cwt_coeffs, frequencies] = cwt(breathing, FS, 'FrequencyLimits', frequency_range, 'VoicesPerOctave', 40);
        power = abs(cwt_coeffs);

        freq_idx = frequencies >= 1 & frequencies <= 15;
        S_limited = cwt_coeffs(freq_idx, :);
        F_limited = frequencies(freq_idx);
        [~, idx_max] = max(abs(S_limited), [], 1);
        dominant_freq = F_limited(idx_max);
    end

    %% event active
    % Discrete Events
    featureName = {'IR1ON','IR2ON','WP1ON','WP2ON','ShockON','Sound3ON'};
    OneHotFeature = zeros(size(featureName,2),size(CameraTime,1));
    % collect features
    for EventsID = 1:size(featureName,2)
        EventsName = featureName{EventsID};
        eventsON = ALLStructFile_temp.(EventsName);
        eventsOFF = ALLStructFile_temp.([EventsName(1:end-2),'OFF']);
        if ~isempty(find(eventsON))
            % find cluster start
            activeON = convertONOFFtoActive(eventsON,eventsOFF,[]);
            OneHotFeature(EventsID,:) = activeON;
        end
    end

    % explonential decay kernel
    ind = 0:1:10*FS; % Time step of 0.1 seconds
    time = ind./FS; % Time step of 0.1 seconds
    decay_constant = 0.6; % You can adjust this value as needed
    decay_curve = exp(-decay_constant * time);
    decay_curve = decay_curve./sum(decay_curve);
    % figure; plot(decay_curve)

    temp = []; lags = 0;
    for k = 1:size(OneHotFeature,1)
        temp1 = conv2(OneHotFeature(k,:),decay_curve,'full');
        temp1 = temp1(:,1+lags:length(DeltaFF)+lags);
        temp(k,:) = temp1;
    end
    ConvOneHotFeature = temp;

    %% realign the xyz coordinate and speed
    [AdjustedXYZ, AdjustedXYZ_speed, processingInfo] = process3DCOM(com_denoised, FS);

    %% move too much in XY when have high Z
%     % 1. Identify high-Z axis frames.
%     % We get a logical array where 'true' means the Z-coordinate is above
%     % the defined threshold.
%     z_threshold = 50;
%     high_z_frames_logical = AdjustedXYZ(:, 3) > z_threshold;
%     
%     % We will analyze frames from the 2nd frame onwards to calculate velocity.
%     high_z_frames_logical = high_z_frames_logical(2:end);
%     
%     % 3. Analyze the patches of high-Z frames.
%     % This part identifies continuous "patches" of high-Z frames.
%     % We find the start and end indices of these patches.
%     high_z_indices = find(high_z_frames_logical);
%     
%     % We find gaps between consecutive high-Z frames to identify patches.
%     patch_breaks = find(diff(high_z_indices) > 1);
%     
%     patch_starts = [high_z_indices(1); high_z_indices(patch_breaks + 1)];
%     patch_ends = [high_z_indices(patch_breaks); high_z_indices(end)];
%     all_max_displace = [];
%     for i = 1:length(patch_starts)
%         start_idx = patch_starts(i);
%         end_idx = patch_ends(i);
%         % A valid patch must have at least 2 frames to calculate displacement.
%         if end_idx > start_idx
%             patch_xy_displacements = sqrt(sum((AdjustedXYZ(start_idx,1:2)-AdjustedXYZ(start_idx:end_idx,1:2)).^2,2));
%             max_xy_displace_in_patch = max(patch_xy_displacements);
%             all_max_displace(i) = max_xy_displace_in_patch;
%         end
%     end
%     move_too_much_ind = all_max_displace > 100;
%     move_too_much_frames_ind = cell2mat(arrayfun(@(i) (patch_starts(i):patch_ends(i)), find(move_too_much_ind), 'UniformOutput', false))';
% 
%     AdjustedXYZ_fill = AdjustedXYZ;
%     AdjustedXYZ_fill(move_too_much_frames_ind,:) = NaN;
%     AdjustedXYZ_fill = fillmissing(AdjustedXYZ_fill, 'linear');
%     AdjustedXYZ_speed = [0; sqrt(sum(diff(AdjustedXYZ).^2, 2)) * FS];

    %% save all figures
    saveDir = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10Clusters4/savedFigure';
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    % Save all open figures
    figHandles = findall(0, 'Type', 'figure');
    for i = 1:length(figHandles)
        figName = get(figHandles(i), 'Name');
        if isempty(figName)
            figName = sprintf('Figure_%d', figHandles(i).Number);
        end
        savefig(figHandles(i), fullfile(saveDir, [figName '.fig']));
        saveas(figHandles(i), fullfile(saveDir, [figName '.png']));
    end

    % Close all figure
    close all;
    fprintf('Saved %d figures to %s\n', length(figHandles), saveDir);
    %% collect all relevant data
    % matlab
    % 1 = mainFr_peaks
    % 2~4 = medfilt1(com_denoised, FS)
    % 5 = SpeedSmooth
    % 6~7 = tsnexy
    % 8~13 = BehEvents [IR1,IR2,WP1,WP2,Shock,Sound]
    X = [dominant_freq, AdjustedXYZ, AdjustedXYZ_speed, tsnexy, OneHotFeature'];
    Y = DeltaFF;

    %% downsample to 10Hz
    r = 5; %10hz
    X_sub = zeros(ceil(size(X,1)/r),size(X,2));
    for i = 1:size(X,2)
        x = X(:,i);
        X_sub(:,i) = decimate(x,r);
    end
    Y_sub = decimate(Y,r);

    filename = [file(1:end-4),'.csv'];
    writematrix([X_sub,Y_sub], fullfile('/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10Clusters4',filename));

end


