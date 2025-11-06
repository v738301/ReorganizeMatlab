%% ========================================================================
%  EXTRACT WAVEFORMS FROM .NEX FILES
%  Template for waveform extraction - ADAPT based on your .nex reading method
%  ========================================================================
%
%  This template shows the general structure for extracting waveforms.
%  YOU MUST ADAPT THIS based on:
%  1. Your .nex file reading function (readNexFile, nex_info, etc.)
%  2. Your specific .nex file structure
%  3. How waveforms are stored in your .nex files
%
%  Reference: TryToIdentifyWaveForm.m on your local machine
%
%% ========================================================================

function [waveform_data] = Extract_Waveforms_From_Nex(nex_path, unit_list, config)
%
% Inputs:
%   nex_path: Path to .nex file or directory containing .nex files
%   unit_list: Struct array with fields:
%              - global_unit_id
%              - session_id
%              - unit_id
%              - session_filename (to match with .nex file)
%   config: Configuration struct with:
%           - sampling_rate: Waveform sampling rate (Hz)
%           - waveform_window_ms: Time window for waveform (ms)
%
% Output:
%   waveform_data: Struct array with:
%                  - global_unit_id
%                  - session_id
%                  - unit_id
%                  - waveform: Average waveform
%                  - waveform_std: Standard deviation
%                  - n_waveforms: Number of spikes averaged
%                  - sampling_rate
%                  - nex_filename

fprintf('=== EXTRACTING WAVEFORMS FROM .NEX FILES ===\n');

% Initialize output
waveform_data = struct();
wf_counter = 0;

% ========================================================================
% METHOD 1: Using NeuroExplorer MATLAB SDK
% ========================================================================
% If you have the NeuroExplorer SDK installed, use nex_info and nex_wf

% Example:
% [nvar, names, types] = nex_info(nex_filepath);
% for i = 1:nvar
%     if types(i) == 3  % Waveform variable
%         [n, ts, nf, w] = nex_wf(nex_filepath, names{i});
%         % n: number of waveforms
%         % ts: timestamps
%         % nf: number of data points in each waveform
%         % w: waveform matrix (n x nf)
%     end
% end

% ========================================================================
% METHOD 2: Using custom readNexFile function
% ========================================================================
% If you have a custom readNexFile function:

% Example:
% nexData = readNexFile(nex_filepath);
% waveforms = nexData.waves{unit_id}.waveforms;
% timestamps = nexData.waves{unit_id}.timestamps;

% ========================================================================
% METHOD 3: Using FieldTrip toolbox
% ========================================================================
% If you use FieldTrip:

% cfg = [];
% cfg.dataset = nex_filepath;
% data = ft_preprocessing(cfg);

% ========================================================================
% TEMPLATE EXTRACTION LOOP
% ========================================================================

for u_idx = 1:length(unit_list)
    unit = unit_list(u_idx);

    fprintf('[%d/%d] Processing Unit %d (Session %d, Unit %d)\n', ...
            u_idx, length(unit_list), unit.global_unit_id, ...
            unit.session_id, unit.unit_id);

    % ====================================================================
    % STEP 1: Find corresponding .nex file
    % ====================================================================

    % Convert .mat filename to .nex filename
    % This depends on your naming convention
    % Example: '2025-01-15_Animal01_RewardAversive_sorted.mat'
    %       -> '2025-01-15_Animal01.nex'

    nex_filename = convertMatToNexFilename(unit.session_filename);
    nex_filepath = fullfile(nex_path, nex_filename);

    if ~exist(nex_filepath, 'file')
        fprintf('  WARNING: .nex file not found: %s\n', nex_filepath);
        continue;
    end

    % ====================================================================
    % STEP 2: Read waveform data from .nex file
    % ====================================================================

    try
        % TODO: REPLACE THIS WITH YOUR ACTUAL .NEX READING CODE
        % Example using hypothetical function:

        % [waveforms, timestamps, sampling_rate] = readWaveformsFromNex(...
        %     nex_filepath, unit.unit_id);

        % For now, create placeholder
        waveforms = [];  % Should be [n_spikes x n_samples] matrix
        timestamps = [];
        sampling_rate = config.sampling_rate;

        % PLACEHOLDER WARNING
        warning('Using placeholder waveform extraction - REPLACE WITH ACTUAL CODE');

    catch ME
        fprintf('  ERROR reading waveform: %s\n', ME.message);
        continue;
    end

    % ====================================================================
    % STEP 3: Process waveforms
    % ====================================================================

    if isempty(waveforms)
        fprintf('  WARNING: No waveforms found for unit %d\n', unit.unit_id);
        continue;
    end

    % Quality control: Remove outlier waveforms
    % Compute median waveform for outlier detection
    median_wf = median(waveforms, 1);

    % Compute correlation of each waveform with median
    correlations = zeros(size(waveforms, 1), 1);
    for i = 1:size(waveforms, 1)
        correlations(i) = corr(waveforms(i, :)', median_wf');
    end

    % Keep only waveforms with correlation > 0.8
    good_waveforms = waveforms(correlations > 0.8, :);

    fprintf('  Found %d waveforms, kept %d after QC\n', ...
            size(waveforms, 1), size(good_waveforms, 1));

    if size(good_waveforms, 1) < 10
        fprintf('  WARNING: Too few waveforms after QC, skipping\n');
        continue;
    end

    % ====================================================================
    % STEP 4: Compute average waveform and features
    % ====================================================================

    mean_waveform = mean(good_waveforms, 1);
    std_waveform = std(good_waveforms, 0, 1);

    % Normalize waveform (optional - useful for comparison)
    % mean_waveform = mean_waveform / max(abs(mean_waveform));

    % ====================================================================
    % STEP 5: Store results
    % ====================================================================

    wf_counter = wf_counter + 1;

    waveform_data(wf_counter).global_unit_id = unit.global_unit_id;
    waveform_data(wf_counter).session_id = unit.session_id;
    waveform_data(wf_counter).unit_id = unit.unit_id;
    waveform_data(wf_counter).waveform = mean_waveform;
    waveform_data(wf_counter).waveform_std = std_waveform;
    waveform_data(wf_counter).n_waveforms = size(good_waveforms, 1);
    waveform_data(wf_counter).sampling_rate = sampling_rate;
    waveform_data(wf_counter).nex_filename = nex_filename;
    waveform_data(wf_counter).timestamps = timestamps;
end

fprintf('✓ Extracted waveforms for %d/%d units\n', wf_counter, length(unit_list));

end

%% ========================================================================
%  HELPER FUNCTION: Convert .mat filename to .nex filename
%  ========================================================================

function nex_filename = convertMatToNexFilename(mat_filename)
% Convert sorted spike data filename to original .nex filename
%
% This depends on your specific naming convention
% MODIFY THIS FUNCTION based on how your files are named

% Example 1: Simple case
% '2025-01-15_Animal01_sorted.mat' -> '2025-01-15_Animal01.nex'
base_name = extractBefore(mat_filename, '_sorted');
nex_filename = [base_name, '.nex'];

% Example 2: More complex case
% '2025-01-15_RewardAversive_Animal01_sorted.mat'
% -> '2025-01-15_Animal01.nex'
% tokens = strsplit(mat_filename, '_');
% nex_filename = sprintf('%s_%s.nex', tokens{1}, tokens{3});

% Example 3: Use regex
% pattern = '(\d{4}-\d{2}-\d{2})_.*_(Animal\d+)';
% tokens = regexp(mat_filename, pattern, 'tokens');
% nex_filename = sprintf('%s_%s.nex', tokens{1}{1}, tokens{1}{2});

end

%% ========================================================================
%  HELPER FUNCTION: Read waveforms from .nex file
%  ========================================================================

function [waveforms, timestamps, sampling_rate] = readWaveformsFromNex(nex_filepath, unit_id)
% Read waveform data for a specific unit from .nex file
%
% YOU MUST IMPLEMENT THIS based on your .nex reading tools
%
% Inputs:
%   nex_filepath: Full path to .nex file
%   unit_id: Unit index to extract
%
% Outputs:
%   waveforms: [n_spikes x n_samples] matrix
%   timestamps: [n_spikes x 1] spike times (seconds)
%   sampling_rate: Waveform sampling rate (Hz)

% ========================================================================
% OPTION 1: NeuroExplorer SDK
% ========================================================================
% [nvar, names, types] = nex_info(nex_filepath);
%
% % Find waveform variable for this unit
% wf_var_idx = unit_id;  % Adjust based on your indexing
%
% if types(wf_var_idx) == 3  % Waveform variable
%     [n, ts, nf, w] = nex_wf(nex_filepath, names{wf_var_idx});
%
%     waveforms = w;  % [n_spikes x n_samples]
%     timestamps = ts;
%
%     % Get sampling rate from header
%     [nexFile] = readNexHeader(nex_filepath);
%     sampling_rate = nexFile.freq;
% else
%     error('Variable %d is not a waveform type', wf_var_idx);
% end

% ========================================================================
% OPTION 2: Custom readNexFile
% ========================================================================
% nexData = readNexFile(nex_filepath);
%
% if unit_id <= length(nexData.waves)
%     waveforms = nexData.waves{unit_id}.waveforms;
%     timestamps = nexData.waves{unit_id}.timestamps;
%     sampling_rate = nexData.waves{unit_id}.ADFrequency;
% else
%     error('Unit %d not found in .nex file', unit_id);
% end

% ========================================================================
% PLACEHOLDER
% ========================================================================
error('readWaveformsFromNex not implemented - see comments for examples');

waveforms = [];
timestamps = [];
sampling_rate = 30000;

end
