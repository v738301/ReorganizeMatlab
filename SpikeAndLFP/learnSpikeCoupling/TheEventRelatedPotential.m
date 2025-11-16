clear all
close all

%%
path = '/Users/hsiehkunlin/Downloads/Case-Studies-Python-master/matfiles';
data = load(fullfile(path,"02_EEG-1.mat"));

%%
EEGa = data.EEGa;
t = data.t;
nTrials = length(EEGa);

mSignal = mean(EEGa);
sdSignal = std(EEGa);
semSignal = sdSignal./sqrt(nTrials);

figure; hold on;
plot(t, mSignal, 'k-')
plot(t, mSignal + 2*semSignal, 'k--');
plot(t, mSignal - 2*semSignal, 'k--');
xlabel('Time [s]') 
ylabel('Voltage [$\mu$ V]')
title('ERP of condition A') 

%%
EEGa = data.EEGa;
EEGb = data.EEGb;
t = data.t;
nTrials = length(EEGa);

dt = t(2) - t(1);

%%
figure; hold on;
plot(t, EEGa(1,:))
xline(0.25,'r--')

xlabel('Time [s]') 
ylabel('Voltage [$\mu$ V]') 
title('EEG data from condition A, Trial 1')

%%
figure;
imagesc(t, 1:nTrials, EEGa);        % Image the data from condition A
set(gca, 'YDir', 'normal');         % Put origin in the lower-left corner
xlabel('Time [s]');                 % Label the x-axis
ylabel('Trial #');                  % Label the y-axis
colorbar;                           % Show voltage-to-color mapping
hold on;
xline(0.25, 'k', 'LineWidth', 2);   % Indicate stimulus onset with a line
hold off;

%%
figure; hold on;

mnA = mean(EEGa);
sdmnA = std(EEGa)./ sqrt(nTrials);

mnB = mean(EEGb);
sdmnB = std(EEGb)./ sqrt(nTrials);

mnD = mnA - mnB;
sdmnD = sqrt(sdmnA .^ 2 + sdmnB .^ 2);

figure; hold on;
plot(t, mnD, 'k');
plot(t, mnD + 2*sdmnD, 'k--');
plot(t, mnD - 2*sdmnD, 'k--');
yline(0, 'r--')


%% permutation test
allERP = [];
for k = 1:3000
    i = randi(nTrials, nTrials, 1);
    EEG0 = EEGa(i,:);
    ERP0 = mean(EEG0);
    allERP = [allERP; ERP0];
end

allERP = sort(allERP,1);
N = size(allERP,1);
ciL = allERP(round(0.025*N),:);
ciU = allERP(round(0.975*N),:);

%%

figure; hold on;
plot(t, mnA, 'k');
plot(t, ciU, 'k--');
plot(t, ciL, 'k--');

plot(t, mnA + 2*sdmnA, 'r--');
plot(t, mnA - 2*sdmnA, 'r--');

%%

max(abs(mnD))

%%
allD = [];
allEEG = [EEGa; EEGb];
for k = 1:3000
    i = randi(nTrials*2, nTrials, 1);
    EEGa = allEEG(i,:);
    ERPa = mean(EEGa);

    i = randi(nTrials*2, nTrials, 1);
    EEGb = allEEG(i,:);
    ERPb = mean(EEGb);
        
    mnDper = max(abs(ERPa - ERPb));
    allD = [allD; mnDper];
end

%%

figure; hold on;
hist(allD)
xline(prctile(allD,95))
xline(max(abs(mnD)),'r')

sum(max(abs(mnD)) <= allD)./length(allD)