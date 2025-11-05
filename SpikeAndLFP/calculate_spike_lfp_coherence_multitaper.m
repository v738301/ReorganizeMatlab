function [coherence, phase, freq, S_spike, S_lfp] = calculate_spike_lfp_coherence_multitaper(spike_times, LFP, NeuralTime, Fs, params)
%% ========================================================================
%  CALCULATE SPIKE-LFP COHERENCE USING CHRONUX MULTITAPER METHOD
%  ========================================================================
%
%  Wrapper function for Chronux coherencysegcpt() to compute coherence
%  between spike times and continuous LFP signal using windowed approach.
%
%  INPUTS:
%    spike_times - Vector of spike times (in seconds)
%    LFP         - LFP signal vector
%    NeuralTime  - Time vector for LFP (in seconds)
%    Fs          - Sampling frequency (Hz)
%    params      - Structure with fields:
%                    .freq_range    - [fmin, fmax] frequency range
%                    .tapers        - [TW, K] time-bandwidth and number of tapers
%                    .pad           - FFT padding (-1, 0, 1, ...)
%                    .Fs            - Sampling frequency (Hz)
%                    .window_size   - Window size in seconds (default: 10 sec)
%
%  OUTPUTS:
%    coherence  - Coherence magnitude (averaged over segments)
%    phase      - Phase relationship (radians)
%    freq       - Frequency vector (Hz)
%    S_spike    - Power spectrum of spikes
%    S_lfp      - Power spectrum of LFP
%
%  REQUIRES: Chronux toolbox (coherencysegcpt function)
%
%% ========================================================================

%% Validate inputs
if isempty(spike_times)
    error('No spikes provided');
end

if length(LFP) ~= length(NeuralTime)
    error('LFP and NeuralTime must have same length');
end

%% Window size for segmentation (to avoid memory issues)
if isfield(params, 'window_size')
    window_size = params.window_size;
else
    window_size = 10;  % Default: 10 seconds per window
end

%% Prepare Chronux params structure
chronux_params = struct();
chronux_params.tapers = params.tapers;  % [TW K]
chronux_params.pad = params.pad;  % Padding factor
chronux_params.Fs = Fs;  % Sampling frequency
chronux_params.fpass = params.freq_range;  % [fmin fmax]
chronux_params.err = 0;  % No error bars

%% Adjust spike times relative to NeuralTime start
% Chronux expects spike times relative to data start
spike_times_adjusted = spike_times - NeuralTime(1);

% Remove any negative spike times
spike_times_adjusted = spike_times_adjusted(spike_times_adjusted >= 0);

% Remove spike times beyond LFP duration
duration = (length(LFP) - 1) / Fs;
spike_times_adjusted = spike_times_adjusted(spike_times_adjusted <= duration);

if isempty(spike_times_adjusted)
    error('No valid spikes within LFP time range');
end

%% Call Chronux coherencysegcpt (windowed coherence)
% data1: continuous LFP (column vector)
% data2: spike times (vector)
% win: window size in seconds
% params: Chronux params structure
% segave: 1 = average over segments (default)
% fscorr: 0 = no finite size correction (default)
try
    segave = 1;  % Average over segments
    fscorr = 0;  % No finite size correction
    % Must request 7 outputs (coherencysegcpt only handles 7, 9, or 10 outputs)
    [coherence, phase, ~, S_lfp, S_spike, freq, zerosp] = coherencysegcpt(LFP(:), spike_times_adjusted, window_size, chronux_params, segave, fscorr);
catch ME
    error('Chronux coherencysegcpt failed: %s', ME.message);
end

%% Ensure outputs are column vectors
coherence = coherence(:);
phase = phase(:);
freq = freq(:);
S_spike = S_spike(:);
S_lfp = S_lfp(:);

end
