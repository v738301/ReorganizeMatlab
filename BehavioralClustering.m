%% BehaviroalAnalysisA001
clear all
close all

%% Load Allanalysisstruct and datapath
paraPath = "/Users/hsiehkunlin/Desktop/Data/DannceData/parameters";
Allanalysisstructfile = fullfile(paraPath, "Allanalysisstructfile.mat");
load(Allanalysisstructfile,"Allanalysisstruct")
%% Load structure
datapath = '/Users/hsiehkunlin/Desktop/Data/Struct/*creRD*OnlyNoise*.mat';
dirpath = dir(datapath);

%% reward aproximaty, breathing rate and the SHAP value
datapathSHAP = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10ClustersRegularized4/*OnlyNoise*Ypred_full*';
% datapathSHAP = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10ClustersRegularized4/*reward_*Ypred_full*';
datapathSHAP2 = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10Clusters4';
dirpathSHAP = dir(datapathSHAP);

for SessionID = 1:length(dirpathSHAP)
    file1 = dirpathSHAP(SessionID).name;
    file2 = dirpathSHAP(SessionID).name;
    file2(end-14:end) = [];
    file = file2;
    file2 = strcat(file2,"_SHAP_full.csv");
    path = dirpathSHAP(SessionID).folder;
    FS = 10;

    shap_values = readtable(fullfile(path,file2));
    shap_values = table2array(shap_values);

    Data_set = readtable(fullfile(datapathSHAP2,[file,'.csv']));
    Data_set = table2array(Data_set);

    X = Data_set(:,1:end-1);
    Y = Data_set(:,end);

    path2 = '/Users/hsiehkunlin/Desktop/Data/Struct';
    ALLStructFile_temp = load(fullfile(path2,file));
    CameraTime = ALLStructFile_temp.CameraTime;
    FS = round(1/mean(diff(CameraTime))); %ALLStructFile_temp.FS;
    tsneval = ALLStructFile_temp.TsneZval;
    breathing = ALLStructFile_temp.Breath_sig_demedain_deoutliers_low_smo;
    FS_breath = ALLStructFile_temp.FS;
    dominant_freq = EstimateBreathingRate(breathing, FS_breath, 1);
    r = 5;
    CameraIndex = 1:length(CameraTime);
    CameraDeciIndex = fliplr(length(CameraTime):-r:1);
    speed = ALLStructFile_temp.Com_speed;

    thershold = FS;
    all_OneHotFeature=[];
    % Discrete Events
    featureName = {'IR1ON','IR2ON'};
    OneHotFeature = zeros(size(featureName,2),size(CameraTime,1));
    % collect features
    for EventsID = 1:size(featureName,2)
        EventsName = featureName{EventsID};
        onset = ALLStructFile_temp.(EventsName);
        if ~isempty(find(onset))
            offset = ALLStructFile_temp.([EventsName(1:end-2),'OFF']);
            eventDiff = onset - offset;
            activeEvents = cumsum(eventDiff);
            activeEvents(activeEvents < 0) = 0;
            onsetPeriod = activeEvents > 0;
            onsetPeriod_merge = mergeCloseEvents(onsetPeriod, thershold);
            OneHotFeature(EventsID,:) = onsetPeriod_merge;
        end
    end
    OneHotFeature(sum(OneHotFeature,2)==0,:) = [];
    all_OneHotFeature = [all_OneHotFeature;sum(OneHotFeature,1)>0];

    featureName = {'WP1ON','WP2ON'};
    OneHotFeature = zeros(size(featureName,2),size(CameraTime,1));
    % collect features
    for EventsID = 1:size(featureName,2)
        EventsName = featureName{EventsID};
        onset = ALLStructFile_temp.(EventsName);
        if ~isempty(find(onset))
            offset = ALLStructFile_temp.([EventsName(1:end-2),'OFF']);
            eventDiff = onset - offset;
            activeEvents = cumsum(eventDiff);
            activeEvents(activeEvents < 0) = 0;
            onsetPeriod = activeEvents > 0;
            onsetPeriod_merge = mergeCloseEvents(onsetPeriod, thershold);
            OneHotFeature(EventsID,:) = onsetPeriod_merge;
        end
    end
    OneHotFeature(sum(OneHotFeature,2)==0,:) = [];

    all_OneHotFeature = [all_OneHotFeature;sum(OneHotFeature,1)>0];

    featureName = {'ShockON','Sound3ON'};
    OneHotFeature = zeros(size(featureName,2),size(CameraTime,1));
    % collect features
    for EventsID = 1:size(featureName,2)
        EventsName = featureName{EventsID};
        onset = ALLStructFile_temp.(EventsName);
        if ~isempty(find(onset))
            offset = ALLStructFile_temp.([EventsName(1:end-2),'OFF']);
            eventDiff = onset - offset;
            activeEvents = cumsum(eventDiff);
            activeEvents(activeEvents < 0) = 0;
            onsetPeriod = activeEvents > 0;
            onsetPeriod_merge = mergeCloseEvents(onsetPeriod, thershold);
            OneHotFeature(EventsID,:) = onsetPeriod_merge;
        end
    end
    OneHotFeature(sum(OneHotFeature,2)==0,:) = [];
    all_OneHotFeature = [all_OneHotFeature;sum(OneHotFeature,1)>0];

    alpha = 1e-2;
    alpha2 = 1e-5;
    temp = [];
    temp2 = [];
    t = 1:size(all_OneHotFeature,2);
    for k = 1:size(all_OneHotFeature,1)
        diff_event = diff([0 all_OneHotFeature(k,:)]);
        event_ends = find(diff_event == -1);
        last_event_end = NaN(1, size(all_OneHotFeature,2));
        last_event_end(event_ends) = event_ends;
        last_event_end = fillmissing(last_event_end, 'previous');
        last_event_end(isnan(last_event_end)) = 0;
        distance = t - last_event_end;
        index = exp(-alpha * distance);
        index(all_OneHotFeature(k,:)==1) = 1;
        temp(k,:) = index;

        index2 = exp(alpha2 * distance);
        index2(all_OneHotFeature(k,:)==1) = 1;
        temp2(k,:) = index2;
    end
    ConvOneHotFeature = temp;
    ConvOneHotFeature2 = temp2;

    % figure; hold on;
    % plot(ConvOneHotFeature')
    %
    % figure;
    % plot(ConvOneHotFeature2')

    % matlab
    % 1 = mainFr_peaks
    % 2~4 = medfilt1(com_denoised, FS)
    % 5 = SpeedSmooth
    % 6~7 = tsnexy
    % 8~13 = BehEvents [IR1,IR2,WP1,WP2,Shock,Sound]

    % different decay constant for discrete event - shorter time constant
    figure('OuterPosition', [-1919,1000,1920,800]);
    subplot(2,5,1)
    scatter(dominant_freq(CameraDeciIndex), ConvOneHotFeature(1,CameraDeciIndex), 15, shap_values(:,1), 'fill')
    caxis([-0.5 0.5]);
    xlabel('breathing rate')
    ylabel('reward port proximity')
    title('color by breath SHAP')
    subplot(2,5,2)
    scatter(dominant_freq(CameraDeciIndex), ConvOneHotFeature(2,CameraDeciIndex), 15, shap_values(:,1), 'fill')
    caxis([-0.5 0.5]);
    xlabel('breathing rate')
    ylabel('reward delivary proximity')
    title('color by breath SHAP')
    subplot(2,5,3)
    scatter(dominant_freq(CameraDeciIndex), ConvOneHotFeature(2,CameraDeciIndex), 15, speed(CameraDeciIndex), 'fill')
    caxis([0 0.1]);
    xlabel('breathing rate')
    ylabel('reward delivary proximity')
    title('color by Speed')
    subplot(2,5,4)
    scatter(dominant_freq(CameraDeciIndex), ConvOneHotFeature(2,CameraDeciIndex), 15, rescale(X(:,3)), 'fill')
    caxis([0 1]);
    xlabel('breathing rate')
    ylabel('reward delivary proximity')
    title('color by Y axis')
    subplot(2,5,5)
    scatter(dominant_freq(CameraDeciIndex), ConvOneHotFeature(2,CameraDeciIndex), 15, rescale(X(:,4)), 'fill')
    caxis([0 1]);
    xlabel('breathing rate')
    ylabel('reward delivary proximity')
    title('color by Z axis')

    % different decay constant for discrete event - longer time constant
    subplot(2,5,6)
    scatter(dominant_freq(CameraDeciIndex), ConvOneHotFeature2(1,CameraDeciIndex), 15, shap_values(:,1), 'fill')
    caxis([-0.5 0.5]);
    xlabel('breathing rate')
    ylabel('reward port distance')
    title('color by breath SHAP')
    subplot(2,5,7)
    scatter(dominant_freq(CameraDeciIndex), ConvOneHotFeature2(2,CameraDeciIndex), 15, shap_values(:,1), 'fill')
    caxis([-0.5 0.5]);
    xlabel('breathing rate')
    ylabel('reward delivary distance')
    title('color by breath SHAP')
    subplot(2,5,8)
    scatter(dominant_freq(CameraDeciIndex), ConvOneHotFeature2(2,CameraDeciIndex), 15, speed(CameraDeciIndex), 'fill')
    caxis([0 0.1]);
    xlabel('breathing rate')
    ylabel('reward delivary distance')
    title('color by Speed')
    subplot(2,5,9)
    scatter(dominant_freq(CameraDeciIndex), ConvOneHotFeature2(2,CameraDeciIndex), 15, rescale(X(:,3)), 'fill')
    caxis([0 1]);
    xlabel('breathing rate')
    ylabel('reward delivary distance')
    title('color by Y axis')
    subplot(2,5,10)
    scatter(dominant_freq(CameraDeciIndex), ConvOneHotFeature2(2,CameraDeciIndex), 15, rescale(X(:,4)), 'fill')
    caxis([0 1]);
    xlabel('breathing rate')
    ylabel('reward delivary distance')
    title('color by Z axis')

    keyboard;
    saveas(gcf,fullfile('/Users/hsiehkunlin/Desktop/savedimage',[file(1:end-4),'_Breath_Proximity_ColorByOthers.png']))
    close all;
end


%% find the foraging pattern, and the relationship between them and rearing type
% we found rat relocation are triggered by high breathing rate and rearing;
datapathSHAP = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10ClustersRegularized4/*OnlyNoise*Ypred_full*';
% datapathSHAP = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10ClustersRegularized4/*reward_*Ypred_full*';
datapathSHAP2 = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10Clusters4';
dirpathSHAP = dir(datapathSHAP);

for SessionID = 1:length(dirpathSHAP)
    file1 = dirpathSHAP(SessionID).name;
    file2 = dirpathSHAP(SessionID).name;
    file2(end-14:end) = [];
    file = file2;
    file2 = strcat(file2,"_SHAP_full.csv");
    path = dirpathSHAP(SessionID).folder;
    FS = 10;

    shap_values = readtable(fullfile(path,file2));
    shap_values = table2array(shap_values);

    Data_set = readtable(fullfile(datapathSHAP2,[file,'.csv']));
    Data_set = table2array(Data_set);

    X = Data_set(:,1:end-1);
    Y = Data_set(:,end);

    %% Top Edge Detection using Density + Hough Transform
    % Create density image from data
    x_data = X(:,2);
    y_data = X(:,3);
    img_size = 200;
    x_range = linspace(min(x_data), max(x_data), img_size);
    y_range = linspace(min(y_data), max(y_data), img_size);

    % Create density image
    density_img = hist3([y_data, x_data], {y_range, x_range});

    % Focus on TOP portion of the image only
    top_portion = 1:round(img_size * 0.2);  % Top 20% of image
    top_density = density_img(top_portion, :);

    % Create edge image from top portion only
    edge_img = top_density > prctile(top_density(:), 1);

    % Hough transform on top portion
    [H, T, R] = hough(edge_img);
    P = houghpeaks(H, 1);  % Get top 3 lines
    lines = houghlines(edge_img, T, R, P);

    if ~isempty(lines)
        line_angles = [];
        line_scores = [];  % Score based on how horizontal and how high

        for k = 1:length(lines)
            xy = [lines(k).point1; lines(k).point2];
            dx = xy(2,1) - xy(1,1);
            dy = xy(2,2) - xy(1,2);

            % Calculate angle (0 = horizontal, pi/2 = vertical)
            angle = atan2(abs(dy), abs(dx));  % Always positive
            line_angles = [line_angles, angle];

            % Score: prefer horizontal lines (small angle) in top region
            avg_y = mean([xy(1,2), xy(2,2)]);
            horizontal_score = 1 - (angle / (pi/2));  % 1 = horizontal, 0 = vertical
            height_score = avg_y / size(edge_img, 1);  % Higher = better

            line_scores = [line_scores, horizontal_score * height_score];
        end

        % Select the most horizontal line in the top region
        [~, best_idx] = max(line_scores);
        best_angle = line_angles(best_idx);

        % Calculate rotation needed to make this line horizontal
        rotation_needed = best_angle;  % Negative because we want to reduce the angle

        fprintf('Detected %d lines in top region\n', length(lines));
        fprintf('Best line angle: %.2f° from horizontal\n', rad2deg(best_angle));
        fprintf('Rotation needed: %.2f°\n', rad2deg(rotation_needed));

        % Apply rotation
        cos_theta = cos(rotation_needed);
        sin_theta = sin(rotation_needed);
        x_corrected = x_data * cos_theta - y_data * sin_theta;
        y_corrected = x_data * sin_theta + y_data * cos_theta;

    else
        fprintf('No lines detected in top region\n');
        x_corrected = x_data;
        y_corrected = y_data;
        rotation_needed = 0;
    end

    % Visualization
    figure('Position', [100, 100, 1200, 400]);

    % Show top density region and detected lines
    subplot(1,3,1);
    imagesc(edge_img);
    colormap(gca, 'gray');
    hold on;

    if exist('lines', 'var') && ~isempty(lines)
        for k = 1:length(lines)
            xy = [lines(k).point1; lines(k).point2];
            if k == best_idx
                plot(xy(:,1), xy(:,2), 'r-', 'LineWidth', 3);  % Best line in red
            else
                plot(xy(:,1), xy(:,2), 'g-', 'LineWidth', 2);  % Others in green
            end
        end
    end
    title('Top Region Edge Detection');
    axis equal; axis tight;

    % Original data
    subplot(1,3,2);
    scatter(x_data, y_data, 2, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
    axis equal; grid on;
    title('Original Data');

    % Corrected data
    subplot(1,3,3);
    scatter(x_corrected, y_corrected, 2, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
    axis equal; grid on;
    title(sprintf('Corrected (%.1f° rotation)', rad2deg(rotation_needed)));

    fprintf('Top edge detection complete.\n');


    %%
    cluster_idx = zeros(1,length(Y));
    cluster_idx(and(x_corrected>=0,y_corrected>=0)) = 1;
    cluster_idx(and(x_corrected>=0,y_corrected<0)) = 2;
    cluster_idx(and(x_corrected<0,y_corrected<0)) = 3;
    cluster_idx(and(x_corrected<0,y_corrected<0)) = 4;

    SHAPRearClass = X(:,4) > prctile(X(:,4), 95);

    % rearing induce region transition
    RegionTransitions = diff([0,cluster_idx]) ~=0;
    input = double(RegionTransitions);
    [clusterStarts, clusterEnds] = findEventCluster(find(SHAPRearClass), 1, 2, 'plotON', 1);
    EventsTimes = clusterEnds;
    xtimeind = -FS_SHAP*120:1:FS_SHAP*120;
    xtime = xtimeind*1/FS_SHAP;
    EventOnsetRange = bsxfun(@plus,EventsTimes(:),xtimeind);
    tempout = [];
    for i = 1:size(EventOnsetRange,1)
        ind = EventOnsetRange(i,:);
        outOfRangeInd = find(or(ind<1,ind>length(input)));
        ind(outOfRangeInd) = 1;
        temp = input(ind);
        temp(outOfRangeInd) = NaN;
        tempout(i,:) = temp;
    end
    % permutation
    alltempout = [];
    for i = 1:100
        EventsTimes = sort(randi(length(RegionTransitions),length(EventsTimes),1));
        input = double(RegionTransitions);
        xtimeind = -FS_SHAP*120:1:FS_SHAP*120;
        xtime = xtimeind*1/FS_SHAP;
        EventOnsetRange = bsxfun(@plus,EventsTimes(:),xtimeind);
        tempout1 = [];
        for k = 1:size(EventOnsetRange,1)
            ind = EventOnsetRange(k,:);
            outOfRangeInd = find(or(ind<1,ind>length(input)));
            ind(outOfRangeInd) = 1;
            temp = input(ind);
            temp(outOfRangeInd) = NaN;
            tempout1(k,:) = temp;
        end
        alltempout(i,:) = nanmean(tempout1);
    end
    figure; hold on;
    plot(xtime,nanmean(tempout),'b','linewidth',2)
    shadedErrorBar(xtime,nanmean(alltempout),1.96*nanstd(alltempout)./sqrt(size(alltempout,1)))

    % region transition induce rearing
    RegionTransitions = diff([0,cluster_idx]) ~=0;
    EventsTimes = find(RegionTransitions);
    input = double(SHAPRearClass);
    xtimeind = -FS_SHAP*120:1:FS_SHAP*120;
    xtime = xtimeind*1/FS_SHAP;
    EventOnsetRange = bsxfun(@plus,EventsTimes(:),xtimeind);
    tempout = [];
    for i = 1:size(EventOnsetRange,1)
        ind = EventOnsetRange(i,:);
        outOfRangeInd = find(or(ind<1,ind>length(input)));
        ind(outOfRangeInd) = 1;
        temp = input(ind);
        temp(outOfRangeInd) = NaN;
        tempout(i,:) = temp;
    end
    figure; hold on;
    plot(xtime,nanmean(tempout),'b','linewidth',2)

    keyboard;
    close all
end

%% reclustering using Tsne and breathing features
datapathSHAP = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10ClustersRegularized4/*OnlyNoise*Ypred_full*';
% datapathSHAP = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10ClustersRegularized4/*reward_*Ypred_full*';
datapathSHAP2 = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10Clusters4';
dirpathSHAP = dir(datapathSHAP);

allX = [];
allSHAP = [];
allGoodSNR = [];
allStat = [];
allY = [];
for SessionID = 1:length(dirpathSHAP)
    SessionID
    file1 = dirpathSHAP(SessionID).name;
    file2 = dirpathSHAP(SessionID).name;
    file2(end-14:end) = [];
    file = file2;
    file2 = strcat(file2,"_SHAP_full.csv");
    path = dirpathSHAP(SessionID).folder;
    FS = 10;

    shap_values = readtable(fullfile(path,file2));
    shap_values = table2array(shap_values);
    Data_set = readtable(fullfile(datapathSHAP2,[file,'.csv']));
    Data_set = table2array(Data_set);
    X = Data_set(:,1:end-1);
    Y = Data_set(:,end);

    if SessionID == 2
        com_denoised = [X(:,2),X(:,3),X(:,4)];
        FS = 10;
        [AdjustedXYZ, AdjustedXYZ_speed, processingInfo] = process3DCOM(com_denoised, FS, 'showPlots', false);
        X(:,2) = AdjustedXYZ(:,1);
        X(:,3) = AdjustedXYZ(:,2);
        X(:,4) = AdjustedXYZ(:,3);
        X(:,5) = AdjustedXYZ_speed;
    end

    path2 = '/Users/hsiehkunlin/Desktop/Data/Struct';
    ALLStructFile_temp = load(fullfile(path2,file));
    FS_breath = ALLStructFile_temp.FS;
    SNR = ALLStructFile_temp.SNR;
    if SNR(3) >= 1
        allX = [allX;X];
        allY = [allY;Y];
        allSHAP = [allSHAP;shap_values];
        if SNR(1) >= 1
            allGoodSNR = [allGoodSNR;1];
        else
            allGoodSNR = [allGoodSNR;0];
        end
        stat_temp = [];
        for i = 1:size(X,2)
            stat_temp = [stat_temp,range(X(:,i))];
        end
        for i = 1:size(X,2)
            stat_temp = [stat_temp,min(X(:,i))];
        end
        for i = 1:size(X,2)
            stat_temp = [stat_temp,max(X(:,i))];
        end
        for i = 1:size(X,2)
            stat_temp = [stat_temp,mean(X(:,i))];
        end
        allStat = [allStat;stat_temp];
    end
end

%%
% Short script to compare feature distributions across sessions
% Assumes: allX (Time x Features), session_durations vector

[~, n_features] = size(allX);
n_sessions = sum(allGoodSNR);
session_durations = size(X,1);

% Split data into sessions
sessions = {};
start_idx = 1;
for s = 1:n_sessions
    end_idx = start_idx + session_durations - 1;
    sessions{s} = allX(start_idx:end_idx, :);
    start_idx = end_idx + 1;
end


% Create subplot grid
n_cols = ceil(sqrt(n_features));
n_rows = ceil(n_features / n_cols);

figure('Position', [100, 100, 300*n_cols, 200*n_rows]);
colors = lines(n_sessions);

for f = 1:n_features
    subplot(n_rows, n_cols, f);
    hold on;
    
    % Plot histogram for each session
    for s = 1:n_sessions
        data = sessions{s}(:, f);
        histogram(data, 20, 'FaceColor', colors(s,:), 'FaceAlpha', 0.6, ...
                 'EdgeColor', 'none', 'Normalization', 'probability');
    end
    
    title(sprintf('Feature %d', f));
    xlabel('Value');
    ylabel('Probability');
    grid on;
    
    if f == 1  % Add legend to first subplot
        legend_labels = arrayfun(@(s) sprintf('Session %d', s), 1:n_sessions, 'UniformOutput', false);
        legend(legend_labels, 'Location', 'best');
    end
end

sgtitle('Feature Distributions Across All Sessions', 'FontSize', 16);
%%
% matlab
% 1 = mainFr_peaks
% 2~4 = medfilt1(com_denoised, FS)
% 5 = SpeedSmooth
% 6~7 = tsnexy
% 8~13 = BehEvents [IR1,IR2,WP1,WP2,Shock,Sound]

figure; scatter(allX(:,2),allX(:,3),5,sum(allX(:,8:9),2),'filled')

norm_allX = normalize(allX);
norm_allX(isnan(norm_allX)) = 0;
input = norm_allX;
input = input([1:10:end],:);%% down sampled to 1Hz
input(:,[12,13]) = []; %% reomove Shock and Sound
allmedian = median(input);
% input = allX;
opts = statset('OutputFcn',@(optimValues,state) KLLogging(optimValues,state,input(:,1)));
% set different initial position for different points
InitialYIN = randn(size(input,1),2);
% Aversive right top
InitialYIN(input(:,1)>allmedian(1),:) = randn(sum(input(:,1)>allmedian(1)),2)+10; % breathing
InitialYIN(input(:,4)>allmedian(4),:) = randn(sum(input(:,4)>allmedian(4)),2)+10; % Rearing
% Reward seeking left bottom
InitialYIN(input(:,8)>allmedian(8),:) = randn(sum(input(:,8)>allmedian(8)),2)-10; % IR1
InitialYIN(input(:,9)>allmedian(9),:) = randn(sum(input(:,9)>allmedian(9)),2)-10; % IR2
InitialYIN(input(:,10)>allmedian(10),:) = randn(sum(input(:,10)>allmedian(10)),2)-10; % WP1
InitialYIN(input(:,11)>allmedian(11),:) = randn(sum(input(:,11)>allmedian(11)),2)-10; % WP2
figure; scatter(InitialYIN(:,1), InitialYIN(:,2))

% % set different initial position for different points
% InitialYIN = randn(size(input,1),2);
% % Aversive right top
% InitialYIN(input(:,1)>allmedian(1),:) = randn(sum(input(:,1)>allmedian(1)),2) + [15,-5]; % breathing
% InitialYIN(input(:,4)>allmedian(4),:) = randn(sum(input(:,4)>allmedian(4)),2) + [-5,+15]; % Rearing
% % Reward seeking left bottom
% InitialYIN(input(:,8)>allmedian(8),:) = randn(sum(input(:,8)>allmedian(8)),2)-10; % IR1
% InitialYIN(input(:,9)>allmedian(9),:) = randn(sum(input(:,9)>allmedian(9)),2)-10; % IR2
% InitialYIN(input(:,10)>allmedian(10),:) = randn(sum(input(:,10)>allmedian(10)),2)-10; % WP1
% InitialYIN(input(:,11)>allmedian(11),:) = randn(sum(input(:,11)>allmedian(11)),2)-10; % WP2
% figure; scatter(InitialYIN(:,1), InitialYIN(:,2))


Y = tsne(input,'Options',opts,Perplexity=300,LearnRate=5000,InitialY=InitialYIN);
figure; scatter(Y(:,1),Y(:,2),5,input(:,1))

input = norm_allX;
input = input([1:10:end],:);%% down sampled to 1Hz
input(:,[12,13]) = []; %% reomove Shock and Sound
figure('OuterPosition', [-1919,1000,1920,1500]);
for i = 1:size(input,2)
    h = subplot(4,4,i);
    x = Y(:,1);
    y = Y(:,2);
    intensity = input(:,i);
    scatter(x,y,5,intensity)
    clim([-1,1])
    xlim([-100,100])
    ylim([-100,100])
    title(['Feature ',int2str(i)])
end

input = allSHAP([1:10:end],:);%% down sampled to 1Hz
input(:,[12,13]) = []; %% reomove Shock and Sound
figure('OuterPosition', [-1919,1000,1920,1500]);
for i = 1:size(input,2)
    h = subplot(4,4,i);
    x = Y(:,1);
    y = Y(:,2);
    intensity = input(:,i);
    scatter(x,y,5,intensity)
    clim([-0.01,0.01])
    xlim([-100,100])
    ylim([-100,100])
    title(['Feature ',int2str(i)])
end

input = norm_allX;
input = input([1:10:end],:);
input(:,[12,13]) = [];
save('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/Embeding/tsneEmbed.mat', 'input', 'Y');

%% validate by 3D Tsne without intitial points
% matlab
% 1 = mainFr_peaks
% 2~4 = medfilt1(com_denoised, FS)
% 5 = SpeedSmooth
% 6~7 = tsnexy
% 8~13 = BehEvents [IR1,IR2,WP1,WP2,Shock,Sound]

norm_allX = normalize(allX);
norm_allX(isnan(norm_allX)) = 0;
input = norm_allX;
input = input([1:10:end],:);%% down sampled to 1Hz
input(:,[12,13]) = []; %% reomove Shock and Sound

opts = statset('OutputFcn',@(optimValues,state) KLLogging(optimValues,state,input(:,1)));
Y = tsne(input,'Options',opts,Perplexity=300,LearnRate=5000,NumDimensions=3);

figure; scatter(Y(:,1),Y(:,2),5,input(:,1))
input = norm_allX;
input = input([1:10:end],:);%% down sampled to 1Hz
input(:,[12,13]) = []; %% reomove Shock and Sound
figure('OuterPosition', [-1919,1000,1920,1500]);
for i = 1:size(input,2)
    h = subplot(4,4,i);
    x = Y(:,1);
    y = Y(:,2);
    intensity = input(:,i);
    scatter(x,y,5,intensity)
    clim([-1,1])
    xlim([-100,100])
    ylim([-100,100])
    title(['Feature ',int2str(i)])
end

%% embeding plot
input = norm_allX;
input = input([1:10:end],:);
input(:,[12,13]) = []; %% reomove Shock and Sound
figure('OuterPosition', [-1919,1000,1920,1500]);
for i = 1:size(input,2)
    h = subplot(4,4,i);
    x = Y(:,1);
    y = Y(:,2);
    intensity = input(:,i);
    scatter(x,y,5,intensity)
    clim([-1,1])
    xlim([-100,100])
    ylim([-100,100])
    title(['Feature ',int2str(i)])
end

saveas(gcf,fullfile('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/Embeding/NewEmbedingWithFeatures.png'))
close all;

input = allSHAP([1:10:end],:);
input(:,[12,13]) = []; %% reomove Shock and Sound
figure('OuterPosition', [-1919,1000,1920,1500]);
for i = 1:size(input,2)
    h = subplot(4,4,i);
    x = Y(:,1);
    y = Y(:,2);
    intensity = input(:,i);
    scatter(x,y,5,intensity)
    clim([-0.01,0.01])
    xlim([-100,100])
    ylim([-100,100])
    title(['Feature ',int2str(i)])
end

saveas(gcf,fullfile('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/Embeding/NewEmbedingWithSHAP.png'))
close all;

%% MLP training
input = norm_allX;
input = input([1:10:end],:);
input(:,[12,13]) = [];

X = input;
Y = Y;
net = MLPregressionFeaturesToTsne(X,Y,0.1);
save('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/Embeding/MLP.mat','net')

%% predict
input = norm_allX;
input(:,[12,13]) = [];
ALLzvals = predict(net,input);

input = norm_allX;
figure('OuterPosition', [-1919,1000,1920,1500]);
for i = 1:size(input,2)
    h = subplot(4,4,i);
    x = ALLzvals(:,1);
    y = ALLzvals(:,2);
    intensity = input(:,i);
    scatter(x,y,5,intensity)
    clim([-1,1])
    xlim([-100,100])
    ylim([-100,100])
    title(['Feature ',int2str(i)])
end
saveas(gcf,fullfile('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/Embeding/ALLNewEmbedingWithFeatures.png'))
close all;


input = allSHAP;
figure('OuterPosition', [-1919,1000,1920,1500]);
for i = 1:size(input,2)
    h = subplot(4,4,i);
    x = ALLzvals(:,1);
    y = ALLzvals(:,2);
    intensity = input(:,i);
    scatter(x,y,5,intensity)
    clim([-0.01,0.01])
    xlim([-100,100])
    ylim([-100,100])
    title(['Feature ',int2str(i)])
end
saveas(gcf,fullfile('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/Embeding/ALLNewEmbedingWithSHAP.png'))
close all;

save('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/Embeding/ALLPredictedtsneEmbed.mat', 'allX', 'allSHAP', 'ALLzvals', 'allY', 'allGoodSNR');