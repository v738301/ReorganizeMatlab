%% BehaviroalAnalysisA001
clear all
close all

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

    Ymedian = median(Y);
    Ymad = median(abs(Y-Ymedian), 1);

    path2 = '/Users/hsiehkunlin/Desktop/Data/Struct';
    ALLStructFile_temp = load(fullfile(path2,file));
    FS_breath = ALLStructFile_temp.FS;
    SNR = ALLStructFile_temp.SNR;
    if SNR(3) >= 1
        allX = [allX;X];
        allY = [allY;(Y-Ymedian)./Ymad];
        allSHAP = [allSHAP;(shap_values-Ymedian)./Ymad];
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

%% remove outlier 
allY(36230:36238) = mean(allY);     

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
    sessions{s} = [allX(start_idx:end_idx, :),allY(start_idx:end_idx, :)];
    start_idx = end_idx + 1;
end
n_features = n_features + 1;

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
input = input([1:100:end],[1:5]);%% down sampled to 1Hz
% input = input([1:100:end],:);%% down sampled to 1Hz
% input(:,[8:13]) = []; %% reomove Shock and Sound

norm_allY = normalize(allY);
norm_allY(isnan(norm_allY)) = 0;
input2 = norm_allY;
input2 = input2([1:100:end],:);%% down sampled to 1Hz

%% try Isomap
D = pdist([input,input2], 'euclidean');     % Compute pairwise distances
D_square = squareform(D);               % Converts to full [N x N] square matrix
% save('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/Embeding/Distance_matrix2.mat', 'D', 'D_square', '-v7.3');

options.dims = [2,3];
% Run ISOMAP
[Y, R, E] = IsoMap(D_square, 'k', 10, options);  % 10-NN graph, reduce to 2D


Yall = Y.coords{1};
% Plot
figure; hold on;
scatter(Yall(1,:),Yall(2,:),5,input(:,1));

Yall = Y.coords{2};
% Plot
figure; hold on;
scatter3(Yall(1,:),Yall(2,:),Yall(3,:),5,input(:,1));

input = [norm_allX,norm_allY];
input = input([1:100:end],:);%% down sampled to 1Hz
figure('OuterPosition', [-1919,1000,1920,1500]);
for i = 1:size(input,2)
    h = subplot(4,4,i);
    x = Yall(1,:);
    y = Yall(2,:);
    z = Yall(3,:);
    intensity = input(:,i);
    scatter3(x,y,z,5,intensity)
    clim([-1,1])
    xlim([-10,10])
    ylim([-10,10])
    title(['Feature ',int2str(i)])
end


input = [allSHAP];
input = input([1:100:end],:);%% down sampled to 1Hz
figure('OuterPosition', [-1919,1000,1920,1500]);
for i = 1:size(input,2)
    h = subplot(4,4,i);
    x = Yall(1,:);
    y = Yall(2,:);
    z = Yall(3,:);
    intensity = input(:,i);
    scatter3(x,y,z,5,intensity)
    clim([-1,1])
    xlim([-10,10])
    ylim([-10,10])
    title(['Feature ',int2str(i)])
end
