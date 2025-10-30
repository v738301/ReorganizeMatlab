clear all
close all

%%
numofsession = 2;
folderpath = "/Volumes/Expansion/Data/Struct_spike";
% [allfiles, folderpath, num_sessions] = selectFilesWithAnimalIDFiltering(folderpath,numofsession,'2025*RewardSeeking*.mat');
[allfiles, folderpath, num_sessions] = selectFilesWithAnimalIDFiltering(folderpath,numofsession,'2025*RewardAversive*.mat');

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
all_results = struct();
all_results.metadata = struct();
all_results.metadata.analysis_date = datestr(now);
all_results.metadata.config = config;
all_results.metadata.num_sessions = num_sessions;

% Preallocate session data
session_data = struct();

%% Process Each Session with Data Collection
for sessionID = 1:num_sessions
    fprintf('\n==== Processing session %d/%d: %s ====\n', sessionID, num_sessions, allfiles(sessionID).name);
    
    % Store session info
    session_data(sessionID).filename = allfiles(sessionID).name;
    session_data(sessionID).session_id = sessionID;
    
    %% Load and preprocess session data
    Timelimits = 'No';
    [NeuralTime, ~, ~, Signal, ~, ~, ~, ~, AversiveSound, ~, ~, Fs, ~] = loadAndPrepareSessionData(allfiles(sessionID), T_sorted, Timelimits);
    
    [filtered_data] = preprocessSignals(Signal, Fs, config.bp_range);
    bestChannel = findBestLFPChannel(filtered_data, Fs);
    LFP = filtered_data(:, bestChannel);
    breathing = filtered_data(:, 32);
    
    % Store basic info
    session_data(sessionID).Fs = Fs;
    session_data(sessionID).best_channel = bestChannel;
    session_data(sessionID).recording_duration = length(NeuralTime)/Fs;
    session_data(sessionID).AversiveSound = AversiveSound;
    
    %% 1. Breathing Frequency Distribution
    [dominant_freq_full, ~, ~, ~, ~, ~, ~, ~, ~, ~] = ...
        identifyBreathingFrequencyBands(breathing, NeuralTime, Fs, ...
        config.f_slow, config.f_mid, config.f_fast, config.min_duration);
    
    % Calculate KDE
    [f, xi] = ksdensity(dominant_freq_full, [0:0.2:15]);
    bin_width = 0.2;
    f_scaled = f * bin_width;
    
    % Store breathing frequency data
    session_data(sessionID).breathing = struct();
    session_data(sessionID).breathing.dominant_freq_full = dominant_freq_full;
    session_data(sessionID).breathing.kde_x = xi;
    session_data(sessionID).breathing.kde_y = f_scaled;
    session_data(sessionID).breathing.mean_freq = mean(dominant_freq_full);
    session_data(sessionID).breathing.median_freq = median(dominant_freq_full);
    session_data(sessionID).breathing.std_freq = std(dominant_freq_full);
    [~, peak_idx] = max(f_scaled);
    session_data(sessionID).breathing.peak_freq = xi(peak_idx);
    
    fprintf('  Breathing: Mean=%.2f Hz, Peak=%.2f Hz\n', ...
        session_data(sessionID).breathing.mean_freq, ...
        session_data(sessionID).breathing.peak_freq);
    
    %% 2. LFP Frequency Distribution (Normalized - FOOOF)
    data = [];
    rpt = 1;
    data.trial{1,rpt} = LFP';
    data.time{1,rpt} = NeuralTime';
    data.label{1} = 'chan';
    data.trialinfo(rpt,1) = rpt;
    
    cfg = [];
    cfg.length = 2;
    cfg.overlap = 0.5;
    data = ft_redefinetrial(cfg, data);
    
    cfg = [];
    cfg.foilim = [1 150];
    cfg.pad = 4;
    cfg.tapsmofrq = 2;
    cfg.method = 'mtmfft';
    cfg.output = 'fooof_aperiodic';
    fractal = ft_freqanalysis(cfg, data);
    cfg.output = 'pow';
    original = ft_freqanalysis(cfg, data);
    
    cfg = [];
    cfg.parameter = 'powspctrm';
    cfg.operation = 'x2./x1';
    oscillatory_alt = ft_math(cfg, fractal, original);
    
    % Store LFP power data
    session_data(sessionID).lfp = struct();
    session_data(sessionID).lfp.freq = oscillatory_alt.freq;
    session_data(sessionID).lfp.powspctrm_normalized = oscillatory_alt.powspctrm;
    session_data(sessionID).lfp.powspctrm_original = original.powspctrm;
    session_data(sessionID).lfp.powspctrm_fractal = fractal.powspctrm;
    
    % Find dominant frequency in different bands
    bands = struct('delta', [1 4], 'theta', [4 8], 'alpha', [8 13], ...
                   'beta', [13 30], 'gamma', [30 80], 'high_gamma', [80 150]);
    band_names = fieldnames(bands);
    for b = 1:length(band_names)
        band_range = bands.(band_names{b});
        freq_idx = oscillatory_alt.freq >= band_range(1) & oscillatory_alt.freq <= band_range(2);
        [max_power, max_idx] = max(oscillatory_alt.powspctrm(freq_idx));
        freq_in_band = oscillatory_alt.freq(freq_idx);
        session_data(sessionID).lfp.(['peak_' band_names{b}]) = freq_in_band(max_idx);
        session_data(sessionID).lfp.(['power_' band_names{b}]) = max_power;
    end
    
    fprintf('  LFP: Theta peak=%.2f Hz, Gamma peak=%.2f Hz\n', ...
        session_data(sessionID).lfp.peak_theta, ...
        session_data(sessionID).lfp.peak_gamma);
    
    %% 3. Coherence between Breathing and LFP
    window = hamming(10*Fs);
    noverlap = round(0.5*length(window));
    nfft = length(window);
    n_perms = 100;
    
    % Original coherence
    [Cxy, f_coh] = mscohere(LFP, breathing, window, noverlap, nfft, Fs);
    
    % Permutation test
    perm_coh = zeros(length(f_coh), n_perms);
    win_size = 10*Fs;
    n_segments = floor((length(LFP) - win_size) / (win_size/2)) + 1;
    
    fprintf('  Running coherence permutation test...\n');
    for p = 1:n_perms
        shuffled_sig = zeros(size(LFP));
        seg_indices = randperm(n_segments);
        
        for i = 1:n_segments
            start_idx = (i-1)*(win_size/2) + 1;
            end_idx = min(start_idx + win_size - 1, length(LFP));
            orig_start = (seg_indices(i)-1)*(win_size/2) + 1;
            orig_end = min(orig_start + win_size - 1, length(LFP));
            shuffled_sig(start_idx:end_idx) = breathing(orig_start:orig_end);
        end
        
        [perm_coh(:,p), ~] = mscohere(LFP, shuffled_sig, window, noverlap, nfft, Fs);
        
        if mod(p, 20) == 0
            fprintf('    Completed %d/%d permutations\n', p, n_perms);
        end
    end
    
    mean_permute = mean(perm_coh, 2);
    std_permute = std(perm_coh, [], 2);
    
    % Store coherence data
    session_data(sessionID).coherence = struct();
    session_data(sessionID).coherence.freq = f_coh;
    session_data(sessionID).coherence.Cxy = Cxy;
    session_data(sessionID).coherence.mean_permute = mean_permute;
    session_data(sessionID).coherence.std_permute = std_permute;
    session_data(sessionID).coherence.normalized = (Cxy - mean_permute) ./ std_permute;
    
    % Find significant coherence peaks
    sig_idx = session_data(sessionID).coherence.normalized > 2;
    session_data(sessionID).coherence.significant_freqs = f_coh(sig_idx);
    session_data(sessionID).coherence.significant_values = Cxy(sig_idx);
    
    fprintf('  Coherence: %d significant frequencies found\n', sum(sig_idx));
    
    %% 4. Breathing-to-LFP Cross-Frequency Coupling
    fprintf('  Computing Breathing-to-LFP cross-frequency coupling...\n');
    
    % Signal preparation
    gammarange = 1:1:120;
    thetarange = 0.5:0.25:15;
    
    % Extract features
    [~, gammaamps] = multiphasevec(gammarange, LFP, Fs, 8);
    thetaangles = extractThetaPhase(breathing, Fs, 'wavelet', thetarange);
    
    % Downsample for computation efficiency
    gammaamps_teim = gammaamps(:, 1:10:end);
    thetaangles_teim = thetaangles(:, 1:10:end);
    
    % Compute MI Comodulation
    nbins = 36;
    [modindex, meanamps, bincenters] = computeModIndex(gammaamps_teim, thetaangles_teim, nbins);
    maxMI = max(modindex(:));
    
    % Perform shuffling analysis
    shuffles = 10;
    fprintf('    Running %d shuffles...\n', shuffles);
    
    if shuffles > 0
        rng('shuffle');
        offset = randi((size(thetaangles_teim,2)-2), 1, shuffles) + 1;
        modindex_out = zeros(size(gammaamps_teim,1), size(thetaangles_teim,1), shuffles);
        
        for i = 1:shuffles
            ta = [thetaangles_teim(:, offset(1,i)+1:end), thetaangles_teim(:, 1:offset(1,i))];
            modindex_out(:,:,i) = computeModIndex(gammaamps_teim, ta, nbins);
            if mod(i, 5) == 0
                fprintf('      Completed shuffle %d/%d\n', i, shuffles);
            end
        end
        
        modindex_shuffled = nanmean(modindex_out, 3);
        modindex_std = nanstd(modindex_out, [], 3);
    else
        modindex_shuffled = NaN(size(modindex));
        modindex_std = NaN(size(modindex));
    end
    
    % Store cross-frequency coupling data
    session_data(sessionID).coupling_breathing_lfp = struct();
    session_data(sessionID).coupling_breathing_lfp.gammarange = gammarange;
    session_data(sessionID).coupling_breathing_lfp.thetarange = thetarange;
    session_data(sessionID).coupling_breathing_lfp.modindex = modindex;
    session_data(sessionID).coupling_breathing_lfp.modindex_shuffled = modindex_shuffled;
    session_data(sessionID).coupling_breathing_lfp.modindex_std = modindex_std;
    session_data(sessionID).coupling_breathing_lfp.normalized_modindex = ...
        (modindex - modindex_shuffled) ./ modindex_std;
    session_data(sessionID).coupling_breathing_lfp.max_modindex = max(modindex(:));
    session_data(sessionID).coupling_breathing_lfp.max_normalized = ...
        max(session_data(sessionID).coupling_breathing_lfp.normalized_modindex(:));
    
    % Calculate mean phase
    [nFreq, nTheta, nBins] = size(meanamps);
    phaseValues = linspace(-pi, pi, nBins);
    meanPhase = zeros(nFreq, nTheta);
    modStrength = zeros(nFreq, nTheta);
    
    for iFreq = 1:nFreq
        for iTheta = 1:nTheta
            ampDist = squeeze(meanamps(iFreq, iTheta, :));
            x = sum(ampDist .* cos(phaseValues)');
            y = sum(ampDist .* sin(phaseValues)');
            meanPhase(iFreq, iTheta) = atan2(y, x);
            modStrength(iFreq, iTheta) = sqrt(x^2 + y^2) / sum(ampDist);
        end
    end
    
    session_data(sessionID).coupling_breathing_lfp.meanPhase = meanPhase;
    session_data(sessionID).coupling_breathing_lfp.meanPhaseDeg = rad2deg(meanPhase);
    session_data(sessionID).coupling_breathing_lfp.modStrength = modStrength;
    
    %% 5. LFP-to-LFP Cross-Frequency Coupling
    fprintf('  Computing LFP-to-LFP cross-frequency coupling...\n');
    
    % Signal preparation
    gammarange_lfp = 30:1:200;
    thetarange_lfp = 2:0.25:15;
    
    % Extract features
    [~, gammaamps_lfp] = multiphasevec(gammarange_lfp, LFP, Fs, 8);
    thetaangles_lfp = extractThetaPhase(LFP, Fs, 'wavelet', thetarange_lfp);
    
    % Downsample
    gammaamps_lfp_teim = gammaamps_lfp(:, 1:10:end);
    thetaangles_lfp_teim = thetaangles_lfp(:, 1:10:end);
    
    % Compute MI
    [modindex_lfp, meanamps_lfp, ~] = computeModIndex(gammaamps_lfp_teim, thetaangles_lfp_teim, nbins);
    
    % Shuffling
    fprintf('    Running %d shuffles...\n', shuffles);
    if shuffles > 0
        offset_lfp = randi((size(thetaangles_lfp_teim,2)-2), 1, shuffles) + 1;
        modindex_lfp_out = zeros(size(gammaamps_lfp_teim,1), size(thetaangles_lfp_teim,1), shuffles);
        
        for i = 1:shuffles
            ta = [thetaangles_lfp_teim(:, offset_lfp(1,i)+1:end), thetaangles_lfp_teim(:, 1:offset_lfp(1,i))];
            modindex_lfp_out(:,:,i) = computeModIndex(gammaamps_lfp_teim, ta, nbins);
            if mod(i, 5) == 0
                fprintf('      Completed shuffle %d/%d\n', i, shuffles);
            end
        end
        
        modindex_lfp_shuffled = nanmean(modindex_lfp_out, 3);
        modindex_lfp_std = nanstd(modindex_lfp_out, [], 3);
    else
        modindex_lfp_shuffled = NaN(size(modindex_lfp));
        modindex_lfp_std = NaN(size(modindex_lfp));
    end
    
    % Store LFP-LFP coupling data
    session_data(sessionID).coupling_lfp_lfp = struct();
    session_data(sessionID).coupling_lfp_lfp.gammarange = gammarange_lfp;
    session_data(sessionID).coupling_lfp_lfp.thetarange = thetarange_lfp;
    session_data(sessionID).coupling_lfp_lfp.modindex = modindex_lfp;
    session_data(sessionID).coupling_lfp_lfp.modindex_shuffled = modindex_lfp_shuffled;
    session_data(sessionID).coupling_lfp_lfp.modindex_std = modindex_lfp_std;
    session_data(sessionID).coupling_lfp_lfp.normalized_modindex = ...
        (modindex_lfp - modindex_lfp_shuffled) ./ modindex_lfp_std;
    session_data(sessionID).coupling_lfp_lfp.max_modindex = max(modindex_lfp(:));
    session_data(sessionID).coupling_lfp_lfp.max_normalized = ...
        max(session_data(sessionID).coupling_lfp_lfp.normalized_modindex(:));
   
    fprintf('  Session %d data collection complete.\n', sessionID);
end

%% Compile all results
all_results.sessions = session_data;
all_results.metadata.collection_date = datestr(now);

%% Save compiled results
output_filename = fullfile('/Volumes/My980Pro/reorganize/SpikeAndLFP/DataSet', sprintf('AllSessions_CompiledResults_%s.mat', ...
    datestr(now, 'yyyymmdd_HHMMSS')));

try
    save(output_filename, 'all_results', '-v7.3');
    fprintf('\n=== Successfully saved all results to: %s ===\n', output_filename);
    fprintf('Total sessions processed: %d\n', num_sessions);
catch ME
    warning('Failed to save compiled results: %s\n', ME.message);
end
