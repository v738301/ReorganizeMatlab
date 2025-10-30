% matlab
% 1 = mainFr_peaks
% 2~4 = medfilt1(com_denoised, FS)
% 5 = SpeedSmooth
% 6~7 = tsnexy
% 8~13 = BehEvents [IR1,IR2,WP1,WP2,Shock,Sound]
%%
ALLPredictedtsneEmbed = load('/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/Embeding/ALLPredictedtsneEmbed.mat');
ALLzvals = ALLPredictedtsneEmbed.ALLzvals;
allSHAP = ALLPredictedtsneEmbed.allSHAP;
allX = ALLPredictedtsneEmbed.allX;
allY = ALLPredictedtsneEmbed.allY;
FS_SHAP = 10;
savebase = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/Embeding';

%%
EventsTimes = allX(:,13)>0.5; %% allX(:,13) --> shock
EventsTimes = diff([0;EventsTimes])==1;
EventsTimes = find(EventsTimes);
xtimeind = -FS_SHAP*60:1:FS_SHAP*300;
xtime = xtimeind*1/FS_SHAP;
input = sum(allX(:,[8:9]),2); %% allX(:,8:9) --> IR1 or IR2
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
fig = figure;
% plot(xtime,conv(nanmean(tempout))
shadedErrorBar(xtime,conv(nanmean(tempout),ones(1,FS_SHAP*5),'same'),conv(nanstd(tempout),ones(1,FS_SHAP*5),'same')./sqrt(size(tempout,1)),{'k-','markerfacecolor','k'},1)
% xticklabels([])
% yticklabels([])
xline(0,'r-','Linewidth',3)
output = fullfile(savebase,'RewardSeekingAfterNoise.svg');
print(fig, output,'-painters','-dsvg')

%%
EventsTimes = allX(:,13)>0.5;
EventsTimes = diff([0;EventsTimes])==1;
EventsTimes = find(EventsTimes);
xtimeind = -FS_SHAP*60:1:FS_SHAP*300;
xtime = xtimeind*1/FS_SHAP;
input = allX(:,5);
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
fig = figure;
% plot(xtime,conv(nanmean(tempout))
shadedErrorBar(xtime,conv(nanmean(tempout),ones(1,FS_SHAP*5),'same'),conv(nanstd(tempout),ones(1,FS_SHAP*5),'same')./sqrt(size(tempout,1)),{'k-','markerfacecolor','k'},1)
% xticklabels([])
% yticklabels([])
xline(0,'r-','Linewidth',3)
output = fullfile(savebase,'SpeedAfterNoise.svg');
print(fig, output,'-painters','-dsvg')

%%
EventsTimes = allX(:,13)>0.5;
EventsTimes = diff([0;EventsTimes])==1;
EventsTimes = find(EventsTimes);
xtimeind = -FS_SHAP*60:1:FS_SHAP*300;
xtime = xtimeind*1/FS_SHAP;
input = allX(:,1);
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
fig = figure;
% plot(xtime,conv(nanmean(tempout))
shadedErrorBar(xtime,conv(nanmean(tempout),ones(1,FS_SHAP*5)./(FS_SHAP*5),'same'),conv(nanstd(tempout),ones(1,FS_SHAP*5)./(FS_SHAP*5),'same')./sqrt(size(tempout,1)),{'k-','markerfacecolor','k'},1)
% xticklabels([])
% yticklabels([])
xline(0,'r-','Linewidth',3)
output = fullfile(savebase,'BreathingAfterNoise.svg');
print(fig, output,'-painters','-dsvg')

%%
datapathSHAP = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10ClustersRegularized4/*OnlyNoise*Ypred_full*';
% datapathSHAP = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10ClustersRegularized4/*reward_*Ypred_full*';
datapathSHAP2 = '/Users/hsiehkunlin/Desktop/SHAP/allshapvaluesXBRBIG10Clusters4';
dirpathSHAP = dir(datapathSHAP);

%
for SessionID = 1:length(dirpathSHAP)
    SessionID
    file1 = dirpathSHAP(SessionID).name;
    file2 = dirpathSHAP(SessionID).name;
    file2(end-14:end) = [];
    file = file2;
    file2 = strcat(file2,"_SHAP_full.csv");
    path = dirpathSHAP(SessionID).folder;
    FS = 10;

    path2 = '/Users/hsiehkunlin/Desktop/Data/Struct';
    ALLStructFile_temp = load(fullfile(path2,file));
    SNR = ALLStructFile_temp.SNR;
    clear ALLStructFile_temp

    %% analysis
    shap_values = readtable(fullfile(path,file2));
    shap_values = table2array(shap_values);
    r = 5;
    CameraIndex = [0:r:size(shap_values,1)*r];
    CameraIndex(1) = [];
    CameraTime = CameraIndex*(1/50);

    y_pred = readtable(fullfile(path,file1));
    y_pred = table2array(y_pred);
    y_pred(1,:) = [];

    %%
    Data_set = readtable(fullfile(datapathSHAP2,[file,'.csv']));
    Data_set = table2array(Data_set);

    X = Data_set(:,1:end-1);
    Y = Data_set(:,end);

    fig = figure('OuterPosition', [-1919,1000,800,600]);
    h = [];
    subplot(2,1,1); hold on;
    h(end+1) = plot(CameraTime,X(:,8), 'DisplayName', 'IR');
    h(end+1) = plot(CameraTime,X(:,9), 'DisplayName', 'WP');
    h(end+1) = plot(CameraTime,X(:,13), 'DisplayName', 'AVESIVE');
    xlim([0,3600])
    subplot(2,1,2); hold on;
    h(end+1) = plot(CameraTime,conv(zscore(X(:,1)),ones(1,FS*60),'same'), 'DisplayName', 'IR');
    h(end+1) = plot(CameraTime,conv(zscore(X(:,8)),ones(1,FS*60),'same'), 'DisplayName', 'IR');
    xline(CameraTime(X(:,13)>1),'r-', 'DisplayName', 'Aversive');
    % h(end+1) = plot(CameraTime,conv(X(:,9),ones(1,FS*60),'same'), 'DisplayName', 'WP');
    % h(end+1) = plot(CameraTime,conv(X(:,10),ones(1,FS*60),'same'), 'DisplayName', 'AVESIVE');
    xlim([0,3600])
    keyboard;
    output = fullfile(savebase,'ExampleRewardAversiveSession.svg');
    print(fig, output,'-painters','-dsvg')
end 

%%
data = allX(:,1);

% === Fit GMM with 3 Components ===
gm = fitgmdist(data, 3);

% Extract GMM parameters
mu = gm.mu;                     % Means of the Gaussians (1x3)
sigma = sqrt(gm.Sigma(:))';   % Standard deviations (1x3)
pi_weights = gm.ComponentProportion; % Mixing proportions (1x3)

% Sort the Gaussians by their means for consistency
[mu_sorted, sortIdx] = sort(mu);
sigma_sorted = sigma(sortIdx);
pi_sorted = pi_weights(sortIdx);

% Assign sorted parameters
mu1 = mu_sorted(1); sigma1 = sigma_sorted(1); pi1 = pi_sorted(1);
mu2 = mu_sorted(2); sigma2 = sigma_sorted(2); pi2 = pi_sorted(2);
mu3 = mu_sorted(3); sigma3 = sigma_sorted(3); pi3 = pi_sorted(3);

pdf1 = @(x) pi1 * normpdf(x, mu1, sigma1);
pdf2 = @(x) pi2 * normpdf(x, mu2, sigma2);
pdf3 = @(x) pi3 * normpdf(x, mu3, sigma3);

misclassError = @(t1, t2) ...
    pi1 * (1 - normcdf(t1, mu1, sigma1)) + ...  % False Positives for Group 1
    pi2 * (normcdf(t1, mu2, sigma2) - normcdf(t2, mu2, sigma2)) + ... % Misclassification between Group 2
    pi3 * normcdf(t2, mu3, sigma3);

% Define the objective function for optimization
objective = @(t) misclassError(t(1), t(2));

% Perform optimization using fminsearch or fmincon
% Here, we use fminsearch for simplicity
initial_t1 = 2;
initial_t2 = 5;
optThreshold = fminsearch(objective, [initial_t1, initial_t2]);

% Assign optimized thresholds
t1_opt = optThreshold(1);
t2_opt = optThreshold(2);

% === Display the Optimal Thresholds ===
fprintf('Optimal Thresholds Minimizing Misclassification Error:\n');
fprintf('Threshold 1: %.4f\n', t1_opt);
fprintf('Threshold 2: %.4f\n', t2_opt);

% === Visualization ===
fig = figure; hold on;

% Plot Histogram
histogram(data, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'BinWidth', (max(data)-min(data))/100);

% Plot Fitted Gaussians
h = [];
xFit = linspace(min(data)-1, max(data)+1, 1000);
h(1) = plot(xFit, pdf1(xFit), 'b-', 'LineWidth', 2, 'DisplayName', 'Gaussian 1');
h(2) = plot(xFit, pdf2(xFit), 'r-', 'LineWidth', 2, 'DisplayName', 'Gaussian 2');
h(3) = plot(xFit, pdf3(xFit), 'g-', 'LineWidth', 2, 'DisplayName', 'Gaussian 3');
xline(2,'r','LineWidth',2)
xline(5,'g','LineWidth',2)
% Plot Optimized Thresholds
% plot([t1_opt, t1_opt], [0, max([pdf1(xFit), pdf2(xFit), pdf3(xFit)])], 'k--', 'LineWidth', 2, 'DisplayName', 'Threshold 1');
% plot([t2_opt, t2_opt], [0, max([pdf1(xFit), pdf2(xFit), pdf3(xFit)])], 'k--', 'LineWidth', 2, 'DisplayName', 'Threshold 2');

% Customize Plot
xlabel('Data Values');
ylabel('Probability Density');
title('Optimal Thresholds for breathing frequencies');
% legend(h(:));
% grid on;
hold off;

output = fullfile(savebase,'BreathingCluster.svg');
print(fig, output,'-painters','-dsvg')
close all

%%
data = allX(:,4);

% === Fit GMM with 2 Components ===
gm = fitgmdist(data, 2);
mu = gm.mu;                     % Means of the Gaussians (1x2)
sigma = sqrt(gm.Sigma(:))';   % Standard deviations (1x2)
pi_weights = gm.ComponentProportion; % Mixing proportions (1x2)
[mu_sorted, sortIdx] = sort(mu);
sigma_sorted = sigma(sortIdx);
pi_sorted = pi_weights(sortIdx);

mu1 = mu_sorted(1); sigma1 = sigma_sorted(1); pi1 = pi_sorted(1);
mu2 = mu_sorted(2); sigma2 = sigma_sorted(2); pi2 = pi_sorted(2);
pdf1 = @(x) pi1 * normpdf(x, mu1, sigma1);
pdf2 = @(x) pi2 * normpdf(x, mu2, sigma2);

misclassError = @(t1) ...
    pi1 * (1 - normcdf(t1, mu1, sigma1)) + ...  % False Positives for Group 1
    pi2 * normcdf(t1, mu2, sigma2);

% Define the objective function for optimization
objective = @(t) misclassError(t(1));

% Perform optimization using fminsearch or fmincon
% Here, we use fminsearch for simplicity
initial_t1 = (mu1 + mu2) / 2;
% Add bounds to prevent unreasonable thresholds
lower_bound = min(mu1 - 3*sigma1, mu2 - 3*sigma2);
upper_bound = max(mu1 + 3*sigma1, mu2 + 3*sigma2);
optThreshold = fminbnd(misclassError, lower_bound, upper_bound);

fig = figure; hold on;
histogram(data, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'BinWidth', (350)/100);
h = [];
xFit = linspace(min(data)-1, max(data)+1, 1000);
h(1) = plot(xFit, pdf1(xFit), 'b-', 'LineWidth', 2, 'DisplayName', 'Gaussian 1');
h(2) = plot(xFit, pdf2(xFit), 'r-', 'LineWidth', 2, 'DisplayName', 'Gaussian 2');
xlabel('Data Values');
ylabel('Probability Density');
xlim([-50,300])
plot([optThreshold, optThreshold], [0, max([pdf1(xFit), pdf2(xFit)])], 'k--', 'LineWidth', 2, 'DisplayName', 'Threshold 1');

output = fullfile(savebase,'ZaxisCluster.svg');
print(fig, output,'-painters','-dsvg')
close all


%%
norm_allX = normalize(allX);
norm_allX(isnan(norm_allX)) = 0;
input = norm_allX;

x = ALLzvals(:,1);
y = -ALLzvals(:,2);
xx_big = linspace(-100,100,301);
yy_big = linspace(-100,100,301);
indmatrix = [discretize(x, xx_big), discretize(y, yy_big)];
nanINDEX = isnan(sum(indmatrix,2));
indmatrix(nanINDEX,:) = [];
centersX = (xx_big(1:end-1) + xx_big(2:end)) / 2;
centersY = (yy_big(1:end-1) + yy_big(2:end)) / 2;

fig = figure('OuterPosition', [-1919,1000,800,600]);
subplot(2,3,1)
intensity = sum(input(:,1),2);
intensity(nanINDEX) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(xx_big), length(xx_big)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[-2,2]);
xlim([-100,100])
ylim([-100,100])
title('Breathing Rate')

subplot(2,3,2)
intensity = sum(input(:,5),2);
intensity(nanINDEX) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(xx_big), length(xx_big)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[-2,2]);
xlim([-100,100])
ylim([-100,100])
title('Speed')

subplot(2,3,3)
intensity = sum(input(:,[8:11]),2);
intensity(nanINDEX) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(xx_big), length(xx_big)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[-2,2]);
xlim([-100,100])
ylim([-100,100])
title('Reward Seeking')

subplot(2,3,4)
intensity = sum(input(:,2),2);
intensity(nanINDEX) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(xx_big), length(xx_big)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[-2,2]);
xlim([-100,100])
ylim([-100,100])
title('X')

subplot(2,3,5)
intensity = sum(input(:,3),2);
intensity(nanINDEX) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(xx_big), length(xx_big)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[-2,2]);
xlim([-100,100])
ylim([-100,100])
title('Y')

subplot(2,3,6)
intensity = sum(input(:,4),2);
intensity(nanINDEX) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(xx_big), length(xx_big)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[-2,2]);
xlim([-100,100])
ylim([-100,100])
title('Z')

output = fullfile(savebase,'NewClusterFeatures.svg');
print(fig, output,'-painters','-dsvg')
close all


%%
session_durations = 36000;
n_sessions = size(allY,1)./session_durations;
norm_allY = [];

% Split data into sessions
start_idx = 1;
for s = 1:n_sessions
    end_idx = start_idx + session_durations - 1;
    norm_allY(start_idx:end_idx, :) = normalize(allY(start_idx:end_idx, :));
    start_idx = end_idx + 1;
end

input = norm_allY;
x = ALLzvals(:,1);
y = -ALLzvals(:,2);
xx_big = linspace(-100,100,301);
yy_big = linspace(-100,100,301);
indmatrix = [discretize(x, xx_big), discretize(y, yy_big)];
nanINDEX = isnan(sum(indmatrix,2));
indmatrix(nanINDEX,:) = [];
centersX = (xx_big(1:end-1) + xx_big(2:end)) / 2;
centersY = (yy_big(1:end-1) + yy_big(2:end)) / 2;

fig = figure('OuterPosition', [-1919,1000,800,600]);
intensity = sum(input(:,1),2);
intensity(nanINDEX) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(xx_big), length(xx_big)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[-2,2]);
xlim([-100,100])
ylim([-100,100])

%%
norm_allX = normalize(allX);
norm_allX(isnan(norm_allX)) = 0;
input = norm_allX;

comXY = allX(:,2:3);
initial_centroids = [
    min(comXY(:,1)), min(comXY(:,2));  % Bottom-left corner
    max(comXY(:,1)), min(comXY(:,2));  % Bottom-right corner
    min(comXY(:,1)), max(comXY(:,2));  % Top-left corner
    max(comXY(:,1)), max(comXY(:,2));   % Top-right corner
    %mean([min(comXY(:,1)),max(comXY(:,1))]),mean([min(comXY(:,2)),max(comXY(:,2))])
    ];

% Perform k-means clustering with the predefined cluster centers
k = 4;
[cluster_idx, cluster_centroids] = kmeans(comXY(:,1:2), k, 'Start', initial_centroids);
fig = figure; gscatter(comXY(1:10:end,1),comXY(1:10:end,2),cluster_idx(1:10:end),'filled')
legend('off')
output = fullfile(savebase,'Cluster4Corner.svg');
print(fig, output,'-painters','-dsvg')
close all

fig = figure;
gscatter(comXY(1:10:end,1),comXY(1:10:end,2),cluster_idx(1:10:end))
output = fullfile(savebase,'Cluster4Corner3.svg');
print(fig, output,'-painters','-dsvg')

intensity = cluster_idx;
intensity(nanINDEX) = [];
[uniqueXY, ~, group] = unique(indmatrix, 'rows');
modeFunc = @(classes) mode(classes);
dominantClasses = splitapply(modeFunc, intensity, group);
dominantData = [uniqueXY, dominantClasses];

uniqueX = unique(uniqueXY(:,1));
uniqueY = unique(uniqueXY(:,2));

dominantMatrix = NaN(max(uniqueX), max(uniqueY));
linearIndices = sub2ind(size(dominantMatrix), uniqueXY(:,1), uniqueXY(:,2));
dominantMatrix(linearIndices) = dominantClasses;
dominantMatrix(isnan(dominantMatrix)) = 0;
mymap = [1,1,1;
    0.00,0.45,0.74;
    0.85,0.33,0.10;
    0.93,0.69,0.13;
    0.49,0.18,0.56];


fig = figure;
imagesc(centersX, centersY, dominantMatrix');
colormap(mymap)
xlim([-120,120])
ylim([-120,120])

output = fullfile(savebase,'Cluster4CornerOnTsne.svg');
print(fig, output,'-painters','-dsvg')
close all

fig = figure;
gscatter(ALLzvals(1:10:end,1),ALLzvals(1:10:end,2),cluster_idx(1:10:end))
output = fullfile(savebase,'Cluster4Corner2.svg');
print(fig, output,'-painters','-dsvg')

%%
figure;
x = ALLzvals(:,1);
y = -ALLzvals(:,2);
intensity = sum(input(:,[8:11]),2);
xx_big = linspace(-100,100,301);
yy_big = linspace(-100,100,301);
[counts, edgesX, edgesY] = histcounts2(x, y, xx_big, yy_big);
centersX = (edgesX(1:end-1) + edgesX(2:end)) / 2;
centersY = (edgesY(1:end-1) + edgesY(2:end)) / 2;
indmatrix = [discretize(x, edgesX), discretize(y, edgesY)];
intensity(isnan(sum(indmatrix,2))) = [];
indmatrix(isnan(sum(indmatrix,2)),:) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(edgesX), length(edgesX)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[-2,2]);
xlim([-100,100])
ylim([-100,100])

figure;
x = ALLzvals(:,1);
y = -ALLzvals(:,2);
intensity = sum(input(:,[1]),2);
xx_big = linspace(-100,100,301);
yy_big = linspace(-100,100,301);
[counts, edgesX, edgesY] = histcounts2(x, y, xx_big, yy_big);
centersX = (edgesX(1:end-1) + edgesX(2:end)) / 2;
centersY = (edgesY(1:end-1) + edgesY(2:end)) / 2;
indmatrix = [discretize(x, edgesX), discretize(y, edgesY)];
intensity(isnan(sum(indmatrix,2))) = [];
indmatrix(isnan(sum(indmatrix,2)),:) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(edgesX), length(edgesX)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[-2,2]);
xlim([-100,100])
ylim([-100,100])

figure;
x = ALLzvals(:,1);
y = -ALLzvals(:,2);
intensity = sum(input(:,[4]),2);
xx_big = linspace(-100,100,301);
yy_big = linspace(-100,100,301);
[counts, edgesX, edgesY] = histcounts2(x, y, xx_big, yy_big);
centersX = (edgesX(1:end-1) + edgesX(2:end)) / 2;
centersY = (edgesY(1:end-1) + edgesY(2:end)) / 2;
indmatrix = [discretize(x, edgesX), discretize(y, edgesY)];
intensity(isnan(sum(indmatrix,2))) = [];
indmatrix(isnan(sum(indmatrix,2)),:) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(edgesX), length(edgesX)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[-2,2]);
xlim([-100,100])
ylim([-100,100])

midPoint = [mean([min(comXY(:,1)),max(comXY(:,1))]),mean([min(comXY(:,2)),max(comXY(:,2))])];
PortA = [min(comXY(:,1)),min(comXY(:,2))];
PortB = [max(comXY(:,1)),max(comXY(:,2))];
distanceToRewardPortA = sqrt((comXY(:,1) - PortA(1)).^2 + (comXY(:,2) - PortA(2)).^2);
distanceToRewardPortB = sqrt((comXY(:,1) - PortB(1)).^2 + (comXY(:,2) - PortB(2)).^2);

farFromReward = 1./(distanceToRewardPortA.^2) + 1./(distanceToRewardPortB.^2);
farFromReward = 1./farFromReward;
figure;
scatter(comXY(:,1),comXY(:,2),5,farFromReward)

figure;
x = ALLzvals(:,1);
y = -ALLzvals(:,2);
intensity = farFromReward;
xx_big = linspace(-100,100,301);
yy_big = linspace(-100,100,301);
[counts, edgesX, edgesY] = histcounts2(x, y, xx_big, yy_big);
centersX = (edgesX(1:end-1) + edgesX(2:end)) / 2;
centersY = (edgesY(1:end-1) + edgesY(2:end)) / 2;
indmatrix = [discretize(x, edgesX), discretize(y, edgesY)];
intensity(isnan(sum(indmatrix,2))) = [];
indmatrix(isnan(sum(indmatrix,2)),:) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(edgesX), length(edgesX)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[0,0.5e7]);
xlim([-100,100])
ylim([-100,100])

figure;
x = ALLzvals(:,1);
y = -ALLzvals(:,2);
intensity = allX(:,4);
xx_big = linspace(-100,100,301);
yy_big = linspace(-100,100,301);
[counts, edgesX, edgesY] = histcounts2(x, y, xx_big, yy_big);
centersX = (edgesX(1:end-1) + edgesX(2:end)) / 2;
centersY = (edgesY(1:end-1) + edgesY(2:end)) / 2;
indmatrix = [discretize(x, edgesX), discretize(y, edgesY)];
intensity(isnan(sum(indmatrix,2))) = [];
indmatrix(isnan(sum(indmatrix,2)),:) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(edgesX), length(edgesX)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[0,1e3]);
xlim([-100,100])
ylim([-100,100])


%% Breathing toward reward port, rearing, runing, aversive noise
norm_allX = normalize(allX);
norm_allX(isnan(norm_allX)) = 0;
input = norm_allX;

EventsTimes = sum(allX(:,[10:11]),2)>0.5;
EventsTimes = diff([0;EventsTimes])==1;
EventsTimes = find(EventsTimes);

epsilon = FS_SHAP; % Maximum distance between points in a cluster
minPts = 5; % Minimum number of points to form a cluster
timestamps = EventsTimes(:);
idx = dbscan(timestamps, epsilon, minPts);
firstTimestamps = [];
lastTimestamps = [];
uniqueClusters = unique(idx(idx > 0)); % Exclude noise points (idx <= 0)
for i = 1:length(uniqueClusters)
    clusterTimestamps = timestamps(idx == uniqueClusters(i));
    firstTimestamps = [firstTimestamps; min(clusterTimestamps)];
    lastTimestamps = [lastTimestamps; max(clusterTimestamps)];
end

EventsTimes = firstTimestamps(:);
xtimeind = -FS_SHAP*30:1:FS_SHAP*60;
xtime = xtimeind*1/FS_SHAP;
input = sum(allX(:,[1]),2);
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
averageKwngth = mean(lastTimestamps - firstTimestamps)./FS_SHAP;

fig = figure; hold on;
% x = [0,0,averageKwngth,averageKwngth];
% y = [0,10,10,0];
% patch(x,y,[0,0,0.5],'FaceAlpha',0.1,'EdgeColor','none')
xline(averageKwngth,'--','Color',[0,0,0.5])
shadedErrorBar(xtime,conv(nanmean(tempout),ones(1,FS_SHAP*5)./(FS_SHAP*5),'same'),conv(nanstd(tempout),ones(1,FS_SHAP*5)./(FS_SHAP*5),'same')./sqrt(size(tempout,1)),{'-','markerfacecolor',[0,0,0.5],'Color',[0,0,0.5]},1)

EventsTimes = allX(:,5)>prctile(allX(:,5),90);
% EventsTimes = diff([0;EventsTimes])==1;
EventsTimes = find(EventsTimes);

epsilon = FS_SHAP; % Maximum distance between points in a cluster
minPts = FS_SHAP*2; % Minimum number of points to form a cluster
timestamps = EventsTimes(:);
idx = dbscan(timestamps, epsilon, minPts);
firstTimestamps = [];
lastTimestamps = [];
uniqueClusters = unique(idx(idx > 0)); % Exclude noise points (idx <= 0)
for i = 1:length(uniqueClusters)
    clusterTimestamps = timestamps(idx == uniqueClusters(i));
    firstTimestamps = [firstTimestamps; min(clusterTimestamps)];
    lastTimestamps = [lastTimestamps; max(clusterTimestamps)];
end

EventsTimes = firstTimestamps(:);
xtimeind = -FS_SHAP*30:1:FS_SHAP*60;
xtime = xtimeind*1/FS_SHAP;
input = sum(allX(:,[1]),2);
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
averageKwngth = mean(lastTimestamps - firstTimestamps)./FS_SHAP;

% x = [0,0,averageKwngth,averageKwngth];
% y = [0,10,10,0];
% patch(x,y,[0.5,0,0],'FaceAlpha',0.1,'EdgeColor','none')
xline(averageKwngth,'--','Color',[0.5,0,0])
shadedErrorBar(xtime,conv(nanmean(tempout),ones(1,FS_SHAP*5)./(FS_SHAP*5),'same'),conv(nanstd(tempout),ones(1,FS_SHAP*5)./(FS_SHAP*5),'same')./sqrt(size(tempout,1)),{'-','markerfacecolor',[0.5,0,0],'Color',[0.5,0,0]},1)


EventsTimes = allX(:,4)>prctile(allX(:,4),90);
% EventsTimes = diff([0;EventsTimes])==1;
EventsTimes = find(EventsTimes);

epsilon = FS_SHAP; % Maximum distance between points in a cluster
minPts = FS_SHAP*2; % Minimum number of points to form a cluster
timestamps = EventsTimes(:);
idx = dbscan(timestamps, epsilon, minPts);
firstTimestamps = [];
lastTimestamps = [];
uniqueClusters = unique(idx(idx > 0)); % Exclude noise points (idx <= 0)
for i = 1:length(uniqueClusters)
    clusterTimestamps = timestamps(idx == uniqueClusters(i));
    firstTimestamps = [firstTimestamps; min(clusterTimestamps)];
    lastTimestamps = [lastTimestamps; max(clusterTimestamps)];
end

EventsTimes = firstTimestamps(:);
xtimeind = -FS_SHAP*30:1:FS_SHAP*60;
xtime = xtimeind*1/FS_SHAP;
input = sum(allX(:,[1]),2);
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
averageKwngth = mean(lastTimestamps - firstTimestamps)./FS_SHAP;

% x = [0,0,averageKwngth,averageKwngth];
% y = [0,10,10,0];
% patch(x,y,[0,0.5,0.5],'FaceAlpha',0.1,'EdgeColor','none')
xline(averageKwngth,'--','Color',[0,0.5,0.5])
shadedErrorBar(xtime,conv(nanmean(tempout),ones(1,FS_SHAP*5)./(FS_SHAP*5),'same'),conv(nanstd(tempout),ones(1,FS_SHAP*5)./(FS_SHAP*5),'same')./sqrt(size(tempout,1)),{'-','markerfacecolor',[0,0.5,0.5],'Color',[0,0.5,0.5]},1)
xlim([-10,20])
ylim([0,10])
xline(0,'k-')

output = fullfile(savebase,'BreathingToDiffBeh.svg');
print(fig, output,'-painters','-dsvg')

%%
% matlab
% 1 = mainFr_peaks
% 2~4 = medfilt1(com_denoised, FS)
% 5 = SpeedSmooth
% 6~7 = tsnexy
% 8~13 = BehEvents [IR1,IR2,WP1,WP2,Shock,Sound]

intensity = cluster_idx;
intensity(nanINDEX) = [];
[uniqueXY, ~, group] = unique(indmatrix, 'rows');
modeFunc = @(classes) mode(classes);
dominantClasses = splitapply(modeFunc, intensity, group);
dominantData = [uniqueXY, dominantClasses];

uniqueX = unique(uniqueXY(:,1));
uniqueY = unique(uniqueXY(:,2));

dominantMatrix = NaN(max(uniqueX), max(uniqueY));
linearIndices = sub2ind(size(dominantMatrix), uniqueXY(:,1), uniqueXY(:,2));
dominantMatrix(linearIndices) = dominantClasses;
dominantMatrix(isnan(dominantMatrix)) = 0;
mymap = [1,1,1;
    0.00,0.45,0.74;
    0.85,0.33,0.10;
    0.93,0.69,0.13;
    0.49,0.18,0.56];

fig = figure; hold on;
imagesc(centersX, centersY, dominantMatrix');
colormap(mymap)
xlim([-120,120])
ylim([-120,120])
set(gca,'YDir','reverse')

RewardClassed = sum(allX(:,[8:11]),2) > 0.5;
HighBreathingClassed = allX(:,1)> 5;
labels = zeros(1,size(ALLzvals,1));
labels(HighBreathingClassed) = 2;
labels(RewardClassed) = 1;
nonIndex = labels == 0;
X = ALLzvals;
labels(nonIndex) = [];
X(nonIndex,:) = [];

uniqueClasses = unique(labels);
numClasses = length(uniqueClasses);
meanVectors = zeros(numClasses, size(X, 2));

for i = 1:numClasses
    classData = X(labels == uniqueClasses(i), :);
    meanVectors(i, :) = mean(classData);
end

figure; hold on;
gscatter(X(:,1),X(:,2),labels)
scatter(meanVectors(:,1),meanVectors(:,2),'go')
a = mean(X(labels==1,:));
b = mean(X(labels==2,:));
plot(a(1),a(2),'ro');
plot(b(1),b(2),'ro');
xlim([-120,120])
ylim([-120,120])

LDA_Model = fitcdiscr(X, labels, 'DiscrimType', 'linear');
coefficients = LDA_Model.Coeffs(1,2).Linear;
w_builtin = coefficients / norm(coefficients);
disp('LDA Projection Vector from fitcdiscr:');
disp(w_builtin);

scale = 5;
origin = meanVectors(1, :)';  % Starting point at Class 1 mean
point = origin + w_builtin * scale;
quiver(origin(1), origin(2), w_builtin(1), w_builtin(2), scale, 'k', 'LineWidth', 2, 'MaxHeadSize', 2, 'DisplayName', 'LDA Direction');

x_min = min(X(:,1)) - 1;
x_max = max(X(:,1)) + 1;
slope = w_builtin(2) / w_builtin(1);
overall_mean = mean(X, 1)';
y_intercept = overall_mean(2) - slope * overall_mean(1);
x_line = [x_min, x_max];
y_line = slope * x_line + y_intercept;
plot(x_line, y_line, 'k-', 'LineWidth', 2, 'DisplayName', 'LDA Axis');

fig = figure; hold on;
imagesc(centersX, centersY, dominantMatrix');
scatter(meanVectors(:,1),-meanVectors(:,2),'go')
scatter(mean(meanVectors(:,1)),-mean(meanVectors(:,2)),'ro')
plot(x_line, -y_line, 'k-', 'LineWidth', 2, 'DisplayName', 'LDA Axis');
colormap(mymap)
xlim([-120,120])
ylim([-120,120])
set(gca,'YDir','reverse')
output = fullfile(savebase,'LDA2.svg');
print(fig, output,'-painters','-dsvg')

% project all data onto LDA axis
SubProjected = X * -LDA_Model.Coeffs(1,2).Linear;
AllProjected = ALLzvals * -LDA_Model.Coeffs(1,2).Linear;

class1_mean = mean(SubProjected(labels == 1));
class2_mean = mean(SubProjected(labels == 2));
decision_point = (class1_mean + class2_mean) / 2;

fig=figure; hold on;
histogram(AllProjected,[-30:1:30])
plot([class1_mean,class2_mean],[0,0],'go')
% figure; histogram(X,[-30:1:30])
output = fullfile(savebase,'LDA1.svg');
print(fig, output,'-painters','-dsvg')

fig=figure; hold on;
plot(sum(allX(:,[8:9]),2))
plot(sum(allX(:,[10:11]),2))
plot(sum(allX(:,[13]),2))
scales = 0.01;
plot(AllProjected.*scales)
yline(decision_point*scales,'r--')


[pks,EventsTimes] = findpeaks(allX(:,13),MinPeakProminence=0.5);
xtimeind = -FS_SHAP*100:1:FS_SHAP*800;
xtime = xtimeind*1/FS_SHAP;
input = AllProjected;
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
fig = figure; hold on;
shadedErrorBar(xtime,conv(nanmean(tempout),ones(1,FS_SHAP*5),'same'),conv(nanstd(tempout),ones(1,FS_SHAP*5),'same')./sqrt(size(tempout,1)),{'k-','markerfacecolor','k'},1)
% xticklabels([])
% yticklabels([])
xline(0,'r-','Linewidth',3)

% permute
[pks,EventsTimes] = findpeaks(allX(:,13),MinPeakProminence=0.5);
xtimeind = -FS_SHAP*100:1:FS_SHAP*800;
xtime = xtimeind*1/FS_SHAP;
input = AllProjected;
permNum = length(EventsTimes);
firstShock = EventsTimes(1);
alltempout = [];
for i = 1:100
    EventsTimes = sort(randi([firstShock,length(AllProjected)],permNum,1));
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
% figure; hold on;
plot(xtime,nanmean(alltempout),'b','linewidth',2)
shadedErrorBar(xtime,nanmean(alltempout),1.96*nanstd(alltempout)./sqrt(size(alltempout,1)))
xline(4646./FS_SHAP,'r','Linewidth',3)
output = fullfile(savebase,'LDAAfterNoise.svg');
print(fig, output,'-painters','-dsvg')

%% LDA scores explained by variables
% You already have AllProjected from t-SNE LDA
% AllProjected = ALLzvals * -LDA_Model.Coeffs(1,2).Linear;
% Correlate each original variable with the LDA projection
correlations = corr(allX, AllProjected);
variance_explained = correlations.^2;  % R-squared = variance explained

% Variable names
varNames = {'mainFr_peaks', 'com_X', 'com_Y', 'com_Z', 'Speed', ...
            'tsne_X', 'tsne_Y', 'IR1', 'IR2', 'WP1', 'WP2', 'Shock', 'Sound'};

% Display results
disp('Variance explained by each original variable:');
for i = 1:length(varNames)
    fprintf('%s: R=%.3f, R²=%.4f (%.2f%%)\n', ...
            varNames{i}, correlations(i), variance_explained(i), variance_explained(i)*100);
end

% Plot
fig = figure;
bar(variance_explained);
xlabel('Variable');
ylabel('Variance Explained (R²)');
title('How Much Each Original Variable Explains LDA Score');
xticks(1:length(varNames));
xticklabels(varNames);
xtickangle(45);
grid on;
ylim([0, max(variance_explained)*1.1]);

% Add percentage labels on bars
for i = 1:length(variance_explained)
    text(i, variance_explained(i), sprintf('%.1f%%', variance_explained(i)*100), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

output = fullfile(savebase,'VarianceExplained.svg');
print(fig, output,'-painters','-dsvg');

%% short time after noise
intensity = cluster_idx;
intensity(nanINDEX) = [];
[uniqueXY, ~, group] = unique(indmatrix, 'rows');
modeFunc = @(classes) mode(classes);
dominantClasses = splitapply(modeFunc, intensity, group);
dominantData = [uniqueXY, dominantClasses];

uniqueX = unique(uniqueXY(:,1));
uniqueY = unique(uniqueXY(:,2));

dominantMatrix = NaN(max(uniqueX), max(uniqueY));
linearIndices = sub2ind(size(dominantMatrix), uniqueXY(:,1), uniqueXY(:,2));
dominantMatrix(linearIndices) = dominantClasses;
dominantMatrix(isnan(dominantMatrix)) = 0;
mymap = [1,1,1;
    0.00,0.45,0.74;
    0.85,0.33,0.10;
    0.93,0.69,0.13;
    0.49,0.18,0.56];

fig = figure; hold on;
imagesc(centersX, centersY, dominantMatrix');
colormap(mymap)
xlim([-120,120])
ylim([-120,120])
set(gca,'YDir','reverse')

fig=figure; hold on;
plot(sum(allX(:,[8:9]),2))
plot(sum(allX(:,[10:11]),2))
plot(sum(allX(:,[13]),2))
scales = 0.01;
plot(AllProjected.*scales)
yline(decision_point*scales,'r--')



EventsTimes = allX(:,13)>0.5;
EventsTimes = diff([0;EventsTimes])==1;
EventsTimes = find(EventsTimes);
xtimeind = -FS_SHAP*0:1:FS_SHAP*5;
xtime = xtimeind*1/FS_SHAP;
input = ALLzvals(:,1);
EventOnsetRange = bsxfun(@plus,EventsTimes(:),xtimeind);
tempoutX = [];
for i = 1:size(EventOnsetRange,1)
    ind = EventOnsetRange(i,:);
    outOfRangeInd = find(or(ind<1,ind>length(input)));
    ind(outOfRangeInd) = 1;
    temp = input(ind);
    temp(outOfRangeInd) = NaN;
    tempoutX(i,:) = temp;
end
input = ALLzvals(:,2);
EventOnsetRange = bsxfun(@plus,EventsTimes(:),xtimeind);
tempoutY = [];
for i = 1:size(EventOnsetRange,1)
    ind = EventOnsetRange(i,:);
    outOfRangeInd = find(or(ind<1,ind>length(input)));
    ind(outOfRangeInd) = 1;
    temp = input(ind);
    temp(outOfRangeInd) = NaN;
    tempoutY(i,:) = temp;
end

input = AllProjected;
EventOnsetRange = bsxfun(@plus,EventsTimes(:),xtimeind);
tempoutZ = [];
for i = 1:size(EventOnsetRange,1)
    ind = EventOnsetRange(i,:);
    outOfRangeInd = find(or(ind<1,ind>length(input)));
    ind(outOfRangeInd) = 1;
    temp = input(ind);
    temp(outOfRangeInd) = NaN;
    tempoutZ(i,:) = temp;
end

fig = figure;
ax1 = axes('Position',[0.1 0.1 0.8 0.8]);  % [left bottom width height]
imagesc(ax1,centersX, centersY, dominantMatrix');
colormap(mymap)
xlim([-120,120])
ylim([-120,120])
set(gca,'YDir','reverse')
hold on;

% for i = 1:size(tempoutY,1)
%     plot(tempoutX(i,:),-tempoutY(i,:),'k-')
% end
n = 256;
n1 = round(n/2);  % Green to Yellow
n2 = n - n1;       % Yellow to Red
green_to_yellow = [linspace(0,1,n1)', linspace(1,1,n1)', zeros(n1,1)];
yellow_to_red = [linspace(1,1,n2)', linspace(1,0,n2)', zeros(n2,1)];
green_to_red = [green_to_yellow; yellow_to_red];

ax2 = axes('Position', get(ax1, 'Position'));  % Same position as ax1
% colormap(ax2,green_to_red)
colormap(ax2,cool)
set(ax2, 'Color', 'none', 'XTick', [], 'YTick', [], 'ZTick', []);
xlim([-120,120])
ylim([-120,120])
Z = 1:size(tempoutY,2);
for i = 1:size(tempoutY,1)
    if AllProjected(EventsTimes(i))< -5
        surface(ax2,[tempoutX(i,:); tempoutX(i,:)], [tempoutY(i,:); tempoutY(i,:)], [zeros(size(Z));zeros(size(Z))], [Z; Z], ...
            'FaceColor', 'none', ...          % No face coloring
            'EdgeColor', 'interp', ...        % Interpolate edge colors based on z
            'LineWidth', 2);
        keyboard
    end
end

output = '/Users/hsiehkunlin/Desktop/Allreport/20241104USJAPAN/AversiveStimInduceCESomeExp.svg';
print(fig, output,'-painters','-dsvg')

% scatter(tempoutX(1,:),-tempoutY(1,:),10,[1:size(tempoutY,2)],'fill')
% colormap turbo
% colorbar
% xlim([-120,120])
% ylim([-120,120])


% figure; plot(ALLzvals)
%
% fig=figure; hold on;
% plot(sum(allX(:,[8])*100,2))
% plot(sum(allX(:,[9])*100,2))
% % plot(sum(allX(:,[10:11])*100,2))
% plot(sum(allX(:,[13])*100,2))
% plot(ALLzvals)
% scales = 0.01;
% plot(AllProjected.*scales)
% yline(decision_point*scales,'r--')
%
%
%
% figure;
% for i = 1:30
%     subplot(6,5,i)
%     ind = [(((i-1)*33500)+1):(((i)*33500))];
%     gscatter(ALLzvals(ind,1),ALLzvals(ind,2),cluster_idx(ind))
% end

%%

figure; hold on;
plot_regression(AllProjected,allSHAP(:,1),0.8,0.8)

x = ALLzvals(:,1);
y = -ALLzvals(:,2);
xx_big = linspace(-100,100,301);
yy_big = linspace(-100,100,301);
indmatrix = [discretize(x, xx_big), discretize(y, yy_big)];
nanINDEX = isnan(sum(indmatrix,2));
indmatrix(nanINDEX,:) = [];
centersX = (xx_big(1:end-1) + xx_big(2:end)) / 2;
centersY = (yy_big(1:end-1) + yy_big(2:end)) / 2;

norm_allX = normalize(allX);
norm_allX(isnan(norm_allX)) = 0;
input = norm_allX;

fig = figure('OuterPosition', [-1919,1000,800,600]);
subplot(1,3,1)
intensity = sum(input(:,1),2);
intensity(nanINDEX) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(edgesX), length(edgesX)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[-10,10]);
xlim([-100,100])
ylim([-100,100])
subplot(1,3,2)
intensity = double(sum(AllProjected,2));
intensity(nanINDEX) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(edgesX), length(edgesX)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[-200,200]);
xlim([-100,100])
ylim([-100,100])
subplot(1,3,3)
intensity = sum(allSHAP(:,1),2);
intensity(nanINDEX) = [];
sumIntensity = accumarray(indmatrix, intensity, [length(edgesX), length(edgesX)], @sum, 0);
imagesc(centersX, centersY, sumIntensity',[-0.2,0.2]);
xlim([-100,100])
ylim([-100,100])

