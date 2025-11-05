function session_results = calculate_spike_phase_coupling(valid_spikes, LFP, NeuralTime, Fs, config)
% Calculate spike-phase coupling for all units across multiple frequency bands
%
% Inputs:
%   valid_spikes - Cell array of spike times for each unit
%   LFP          - LFP signal vector [n_samples x 1]
%   NeuralTime   - Time vector corresponding to LFP samples
%   Fs           - Sampling frequency
%   config       - Configuration structure with frequency_bands, n_phase_bins, alpha
%
% Outputs:
%   session_results - Structure containing coupling results for all units and bands

n_units = length(valid_spikes);
n_bands = size(config.frequency_bands, 1);

% Initialize results structure
session_results = struct();
session_results.n_units = n_units;
session_results.n_bands = n_bands;
session_results.band_names = config.frequency_bands(:, 1)';
session_results.band_ranges = cell2mat(config.frequency_bands(:, 2));
session_results.units = cell(n_units, 1);

fprintf('  Computing phase coupling for %d units across %d frequency bands...\n', n_units, n_bands);

% Pre-compute phase signals for all bands (EFFICIENCY: done once per band, not per unit)
phase_signals = cell(n_bands, 1);
fprintf('  Pre-computing phase signals for all bands...\n');
for band_idx = 1:n_bands
    band_name = config.frequency_bands{band_idx, 1};
    band_range = config.frequency_bands{band_idx, 2};

    % Bandpass filter for this frequency band
    LFP_filtered = bandpass(LFP, band_range, Fs);

    % Extract instantaneous phase using Hilbert transform
    analytic_signal = hilbert(LFP_filtered);
    phase_signals{band_idx} = angle(analytic_signal);  % Phase in radians [-pi, pi]

    fprintf('    %s (%.1f-%.1f Hz): Complete\n', band_name, band_range(1), band_range(2));
end

% Process each unit
fprintf('  Processing units: ');
for unit_idx = 1:n_units
    if mod(unit_idx, 10) == 0
        fprintf('%d...', unit_idx);
    end

    % Get spike times for this unit
    spike_times = valid_spikes{unit_idx};
    n_spikes = length(spike_times);

    % Convert spike times to sample indices
    spike_indices = round((spike_times - NeuralTime(1)) * Fs) + 1;

    % Remove spikes outside valid range
    valid_spike_mask = (spike_indices > 0) & (spike_indices <= length(LFP));
    spike_indices = spike_indices(valid_spike_mask);
    spike_times = spike_times(valid_spike_mask);
    n_spikes = length(spike_times);

    % Initialize unit results
    unit_results = struct();
    unit_results.unit_id = unit_idx;
    unit_results.n_spikes = n_spikes;
    unit_results.spike_times = spike_times;  % Store spike times for time-resolved analysis
    unit_results.band_coupling = struct();

    % Skip units with too few spikes
    if n_spikes < 50
        % Still store empty results for this unit
        for band_idx = 1:n_bands
            unit_results.band_coupling(band_idx).band_name = config.frequency_bands{band_idx, 1};
            unit_results.band_coupling(band_idx).band_range = config.frequency_bands{band_idx, 2};
            unit_results.band_coupling(band_idx).spike_phases = [];
            unit_results.band_coupling(band_idx).MRL = NaN;
            unit_results.band_coupling(band_idx).rayleigh_z = NaN;
            unit_results.band_coupling(band_idx).rayleigh_p = NaN;
            unit_results.band_coupling(band_idx).preferred_phase = NaN;
            unit_results.band_coupling(band_idx).is_significant = false;
            unit_results.band_coupling(band_idx).phase_hist = zeros(config.n_phase_bins, 1);
        end
        session_results.units{unit_idx} = unit_results;
        continue;
    end

    % Process each frequency band for this unit
    for band_idx = 1:n_bands
        band_name = config.frequency_bands{band_idx, 1};
        band_range = config.frequency_bands{band_idx, 2};

        % Extract phases at spike times (FAST: just indexing into pre-computed phase)
        spike_phases = phase_signals{band_idx}(spike_indices);

        % Calculate circular statistics
        [MRL, preferred_phase, rayleigh_z, rayleigh_p] = calculate_circular_statistics(spike_phases);

        % Calculate phase histogram
        phase_edges = linspace(-pi, pi, config.n_phase_bins + 1);
        phase_hist = histcounts(spike_phases, phase_edges)';

        % Store results for this band
        unit_results.band_coupling(band_idx).band_name = band_name;
        unit_results.band_coupling(band_idx).band_range = band_range;
        unit_results.band_coupling(band_idx).spike_phases = spike_phases;  % Store for time-resolved analysis
        unit_results.band_coupling(band_idx).MRL = MRL;
        unit_results.band_coupling(band_idx).rayleigh_z = rayleigh_z;
        unit_results.band_coupling(band_idx).rayleigh_p = rayleigh_p;
        unit_results.band_coupling(band_idx).preferred_phase = preferred_phase;
        unit_results.band_coupling(band_idx).is_significant = (rayleigh_p < config.alpha);
        unit_results.band_coupling(band_idx).phase_hist = phase_hist;
    end

    % Store unit results
    session_results.units{unit_idx} = unit_results;
end

fprintf(' Done!\n');

end


%% Helper function: Calculate circular statistics
function [MRL, preferred_phase, rayleigh_z, rayleigh_p] = calculate_circular_statistics(phases)
% Calculate circular statistics for phase-locking analysis
%
% Inputs:
%   phases - Vector of phases in radians [-pi, pi]
%
% Outputs:
%   MRL             - Mean Resultant Length (0-1, strength of phase-locking)
%   preferred_phase - Mean phase angle in radians
%   rayleigh_z      - Rayleigh Z-statistic
%   rayleigh_p      - P-value from Rayleigh test

n = length(phases);

if n == 0
    MRL = NaN;
    preferred_phase = NaN;
    rayleigh_z = NaN;
    rayleigh_p = NaN;
    return;
end

% Convert phases to unit vectors
x = cos(phases);
y = sin(phases);

% Mean resultant vector
mean_x = mean(x);
mean_y = mean(y);

% Mean Resultant Length (MRL)
MRL = sqrt(mean_x^2 + mean_y^2);

% Preferred phase (mean angle)
preferred_phase = atan2(mean_y, mean_x);

% Rayleigh test for non-uniformity
rayleigh_z = n * MRL^2;

% P-value approximation (valid for n > 50)
if n > 50
    rayleigh_p = exp(-rayleigh_z) * (1 + (2*rayleigh_z - rayleigh_z^2) / (4*n) - ...
                 (24*rayleigh_z - 132*rayleigh_z^2 + 76*rayleigh_z^3 - 9*rayleigh_z^4) / (288*n^2));
else
    % For small n, use exact formula (more conservative)
    rayleigh_p = exp(-rayleigh_z);
end

% Ensure p-value is in valid range [0, 1]
rayleigh_p = max(0, min(1, rayleigh_p));

end
