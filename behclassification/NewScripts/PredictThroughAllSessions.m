clear all
close all

savepath = '/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/behclassification/BehaviorPrediction';
model_path = 'trained_net_new.mat';

%%
[allfiles, folderpath, num_sessions, unique_animals] = selectFilesWithAnimalIDFiltering('/Volumes/ExpansionBackUp/Data/Struct_spike', 2, '2025*RewardAversive*.mat');
[~, processing_summary] = runStreamlinedLSTMPrediction(allfiles, model_path, savepath);

%%
[allfiles, folderpath, num_sessions, unique_animals] = selectFilesWithAnimalIDFiltering('/Volumes/ExpansionBackUp/Data/Struct_spike', 2, '2025*RewardSeeking*.mat');
[~, processing_summary] = runStreamlinedLSTMPrediction(allfiles, model_path, savepath);
