function [results] = calculate_spike_phase_coupling_by_behavior(valid_spikes, LFP, NeuralTime, Fs, BehaviorClass, config)
%% ========================================================================
%  CALCULATE SPIKE-PHASE COUPLING BY BEHAVIOR
%  ========================================================================
%
%  Calculates spike-phase coupling separately for each behavioral class.
%  Includes statistical reliability measures for all units.
%
%  INPUTS:
%    valid_spikes   - Cell array of spike times (one cell per unit)
%    LFP            - LFP signal vector (filtered, single channel)
%    NeuralTime     - Time vector for LFP sampling
%    Fs             - Sampling frequency
%    BehaviorClass  - Behavior label for each NeuralTime point (1-7)
%    config         - Configuration struct with:
%                       .frequency_bands (cell array: {name, [low, high]})
%                       .behavior_names (cell array of behavior names)
%                       .n_phase_bins (number of phase histogram bins)
%                       .alpha (significance level for Rayleigh test)
%                       .bootstrap_samples (number of bootstrap iterations)
%                       .ci_level (confidence interval level)
%                       .reliability_thresholds (spike count thresholds)
%
%  OUTPUTS:
%    results        - Struct containing:
%                       .unit_results (cell array, one per unit)
%                       .n_units (total number of units)
%                       .config (copy of input config)
%
%% ========================================================================

fprintf('  Calculating behavior-specific spike-phase coupling...\n');

n_units = length(valid_spikes);
n_bands = size(config.frequency_bands, 1);
n_behaviors = length(config.behavior_names);

%% ========================================================================
%  SECTION 1: PRE-COMPUTE PHASE SIGNALS FOR ALL FREQUENCY BANDS
%  ========================================================================

fprintf('  Pre-computing phase signals for %d frequency bands...\n', n_bands);

phase_signals = cell(n_bands, 1);

for band_idx = 1:n_bands
    band_name = config.frequency_bands{band_idx, 1};
    band_range = config.frequency_bands{band_idx, 2};

    % Bandpass filter LFP
    LFP_filtered = bandpass(LFP, band_range, Fs, 'ImpulseResponse', 'iir', 'Steepness', 0.95);

    % Hilbert transform to get instantaneous phase
    analytic_signal = hilbert(LFP_filtered);
    phase_signals{band_idx} = angle(analytic_signal);  % Phase in radians [-π, π]
end

fprintf('  ✓ Phase signals computed\n');

%% ========================================================================
%  SECTION 2: PROCESS EACH UNIT
%  ========================================================================

fprintf('  Processing %d units...\n', n_units);

unit_results = cell(n_units, 1);

for unit_idx = 1:n_units

    % Get spike times for this unit
    spike_times = valid_spikes{unit_idx};
    n_spikes_total = length(spike_times);

    if n_spikes_total == 0
        fprintf('    Unit %d: No spikes - skipping\n', unit_idx);
        continue;
    end

    % Convert spike times to indices in NeuralTime (optimized vectorized approach)
    spike_indices = interp1(NeuralTime, 1:length(NeuralTime), spike_times, 'nearest', 'extrap');

    % Remove out-of-bounds indices
    valid_mask = spike_indices > 0 & spike_indices <= length(BehaviorClass);
    spike_indices = spike_indices(valid_mask);
    n_spikes_valid = length(spike_indices);

    % Get behavior class for each spike
    spike_behaviors = BehaviorClass(spike_indices);

    % Remove spikes with no behavior assignment (BehaviorClass == 0)
    behavior_mask = spike_behaviors > 0;
    spike_indices = spike_indices(behavior_mask);
    spike_behaviors = spike_behaviors(behavior_mask);
    n_spikes_with_behavior = length(spike_indices);

    if mod(unit_idx, 10) == 0 || unit_idx == 1
        fprintf('    Unit %d/%d: %d spikes total, %d with behavior\n', ...
            unit_idx, n_units, n_spikes_total, n_spikes_with_behavior);
    end

    %% Process each frequency band
    band_results = cell(n_bands, 1);

    for band_idx = 1:n_bands
        band_name = config.frequency_bands{band_idx, 1};

        % Extract spike phases for this band
        spike_phases = phase_signals{band_idx}(spike_indices);

        % Process each behavior class
        behavior_results = cell(n_behaviors, 1);

        for beh_idx = 1:n_behaviors

            % Select spikes during this behavior
            beh_mask = spike_behaviors == beh_idx;
            beh_spike_phases = spike_phases(beh_mask);
            n_spikes_beh = length(beh_spike_phases);

            % Initialize result structure
            beh_result = struct();
            beh_result.behavior_name = config.behavior_names{beh_idx};
            beh_result.n_spikes = n_spikes_beh;

            if n_spikes_beh == 0
                % No spikes for this behavior - fill with NaN
                beh_result.MRL = NaN;
                beh_result.preferred_phase = NaN;
                beh_result.rayleigh_z = NaN;
                beh_result.rayleigh_p = NaN;
                beh_result.is_significant = false;
                beh_result.MRL_CI_lower = NaN;
                beh_result.MRL_CI_upper = NaN;
                beh_result.phase_CI_lower = NaN;
                beh_result.phase_CI_upper = NaN;
                beh_result.phase_CI_width_deg = NaN;
                beh_result.reliability_class = 'no_data';
                beh_result.reliability_score = 0;
                beh_result.spike_phases = [];

            else
                % Calculate circular statistics
                [MRL, preferred_phase, rayleigh_z, rayleigh_p] = ...
                    calculate_circular_statistics(beh_spike_phases);

                beh_result.MRL = MRL;
                beh_result.preferred_phase = preferred_phase;
                beh_result.rayleigh_z = rayleigh_z;
                beh_result.rayleigh_p = rayleigh_p;
                beh_result.is_significant = rayleigh_p < config.alpha;

                % Bootstrap confidence interval for MRL
                if n_spikes_beh >= 3  % Need at least 3 spikes for bootstrap
                    [MRL_CI_lower, MRL_CI_upper] = ...
                        bootstrap_MRL_CI(beh_spike_phases, config.bootstrap_samples, config.ci_level);
                    beh_result.MRL_CI_lower = MRL_CI_lower;
                    beh_result.MRL_CI_upper = MRL_CI_upper;
                else
                    beh_result.MRL_CI_lower = NaN;
                    beh_result.MRL_CI_upper = NaN;
                end

                % Circular confidence interval for preferred phase
                if n_spikes_beh >= 3 && MRL > 0
                    [phase_CI_lower, phase_CI_upper, phase_CI_width_deg] = ...
                        circular_phase_CI(preferred_phase, MRL, n_spikes_beh, config.ci_level);
                    beh_result.phase_CI_lower = phase_CI_lower;
                    beh_result.phase_CI_upper = phase_CI_upper;
                    beh_result.phase_CI_width_deg = phase_CI_width_deg;
                else
                    beh_result.phase_CI_lower = NaN;
                    beh_result.phase_CI_upper = NaN;
                    beh_result.phase_CI_width_deg = NaN;
                end

                % Determine reliability classification
                [reliability_class, reliability_score] = ...
                    classify_reliability(n_spikes_beh, beh_result.phase_CI_width_deg, config.reliability_thresholds);
                beh_result.reliability_class = reliability_class;
                beh_result.reliability_score = reliability_score;

                % Store raw spike phases for potential further analysis
                beh_result.spike_phases = beh_spike_phases;
            end

            behavior_results{beh_idx} = beh_result;
        end

        band_results{band_idx} = struct(...
            'band_name', band_name, ...
            'behavior_results', {behavior_results});
    end

    % Store results for this unit
    unit_results{unit_idx} = struct(...
        'unit_id', unit_idx, ...
        'n_spikes_total', n_spikes_total, ...
        'n_spikes_with_behavior', n_spikes_with_behavior, ...
        'band_results', {band_results});
end

fprintf('  ✓ Unit processing complete\n');

%% ========================================================================
%  SECTION 3: PACKAGE RESULTS
%  ========================================================================

results = struct();
results.unit_results = unit_results;
results.n_units = n_units;
results.config = config;

fprintf('  ✓ Analysis complete\n');

end

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function [MRL, preferred_phase, rayleigh_z, rayleigh_p] = calculate_circular_statistics(phases)
    % Calculate circular statistics for a set of phase angles
    %
    % INPUTS:
    %   phases - Vector of phase angles in radians
    %
    % OUTPUTS:
    %   MRL            - Mean Resultant Length (0-1)
    %   preferred_phase - Preferred phase angle in radians [-π, π]
    %   rayleigh_z     - Rayleigh test statistic
    %   rayleigh_p     - Rayleigh test p-value

    n = length(phases);

    % Convert to unit vectors
    x = cos(phases);
    y = sin(phases);

    % Mean direction
    mean_x = mean(x);
    mean_y = mean(y);

    % Mean Resultant Length
    MRL = sqrt(mean_x^2 + mean_y^2);

    % Preferred phase (mean direction)
    preferred_phase = atan2(mean_y, mean_x);

    % Rayleigh test for uniformity
    rayleigh_z = n * MRL^2;

    % P-value approximation (valid for n > 50, conservative for smaller n)
    if n >= 50
        rayleigh_p = exp(-rayleigh_z);
    else
        % Use more accurate formula for small n (Zar, 1999)
        rayleigh_p = exp(-rayleigh_z) * (1 + (2*rayleigh_z - rayleigh_z^2) / (4*n) - ...
                     (24*rayleigh_z - 132*rayleigh_z^2 + 76*rayleigh_z^3 - 9*rayleigh_z^4) / (288*n^2));
    end

    % Ensure p-value is in valid range
    rayleigh_p = max(0, min(1, rayleigh_p));
end

function [CI_lower, CI_upper] = bootstrap_MRL_CI(phases, n_bootstrap, ci_level)
    % Bootstrap confidence interval for Mean Resultant Length
    %
    % INPUTS:
    %   phases      - Vector of phase angles in radians
    %   n_bootstrap - Number of bootstrap samples
    %   ci_level    - Confidence level (e.g., 0.95 for 95% CI)
    %
    % OUTPUTS:
    %   CI_lower - Lower bound of confidence interval
    %   CI_upper - Upper bound of confidence interval

    n = length(phases);
    bootstrap_MRLs = zeros(n_bootstrap, 1);

    for i = 1:n_bootstrap
        % Resample with replacement
        resampled_phases = phases(randi(n, n, 1));

        % Calculate MRL for resampled data
        x = cos(resampled_phases);
        y = sin(resampled_phases);
        bootstrap_MRLs(i) = sqrt(mean(x)^2 + mean(y)^2);
    end

    % Calculate percentile-based confidence interval
    alpha = 1 - ci_level;
    CI_lower = prctile(bootstrap_MRLs, 100 * alpha / 2);
    CI_upper = prctile(bootstrap_MRLs, 100 * (1 - alpha / 2));
end

function [CI_lower, CI_upper, CI_width_deg] = circular_phase_CI(mean_phase, MRL, n, ci_level)
    % Circular confidence interval for preferred phase
    %
    % INPUTS:
    %   mean_phase - Preferred phase in radians
    %   MRL        - Mean Resultant Length
    %   n          - Number of samples
    %   ci_level   - Confidence level (e.g., 0.95)
    %
    % OUTPUTS:
    %   CI_lower      - Lower bound in radians [-π, π]
    %   CI_upper      - Upper bound in radians [-π, π]
    %   CI_width_deg  - Width of CI in degrees

    % Estimate concentration parameter κ from MRL
    % Using approximation: κ ≈ MRL * (2 - MRL^2) / (1 - MRL^2) for MRL < 0.9
    if MRL < 0.9
        kappa = MRL * (2 - MRL^2) / (1 - MRL^2);
    else
        % For high MRL, use alternative approximation
        kappa = 1 / (2 * (1 - MRL));
    end

    % Confidence interval half-width (Fisher, 1993)
    % For von Mises distribution: d = acos(1 - (1-ci_level)*(1-exp(-2*n*kappa))/(exp(2*n*kappa)))
    % Simplified approximation for moderate to large n:
    z = norminv((1 + ci_level) / 2);  % z-score for confidence level
    d = z / sqrt(n * kappa);  % Half-width in radians

    % Apply bounds
    CI_lower = mean_phase - d;
    CI_upper = mean_phase + d;

    % Convert width to degrees
    CI_width_deg = rad2deg(2 * d);

    % Wrap to [-π, π]
    CI_lower = angle(exp(1i * CI_lower));
    CI_upper = angle(exp(1i * CI_upper));
end

function [reliability_class, reliability_score] = classify_reliability(n_spikes, phase_CI_width_deg, thresholds)
    % Classify reliability based on spike count and CI width
    %
    % INPUTS:
    %   n_spikes           - Number of spikes
    %   phase_CI_width_deg - Width of phase CI in degrees (NaN if not computed)
    %   thresholds         - Struct with spike count thresholds
    %
    % OUTPUTS:
    %   reliability_class - String: 'very_low', 'low', 'moderate', 'good', 'excellent'
    %   reliability_score - Numeric score: 1 (very_low) to 5 (excellent)

    % Initial classification based on spike count
    if n_spikes < thresholds.very_low
        reliability_class = 'very_low';
        reliability_score = 1;
    elseif n_spikes < thresholds.low
        reliability_class = 'low';
        reliability_score = 2;
    elseif n_spikes < thresholds.moderate
        reliability_class = 'moderate';
        reliability_score = 3;
    elseif n_spikes < thresholds.good
        reliability_class = 'good';
        reliability_score = 4;
    else
        reliability_class = 'excellent';
        reliability_score = 5;
    end

    % Downgrade if CI is very wide (>90 degrees), indicating high uncertainty
    if ~isnan(phase_CI_width_deg) && phase_CI_width_deg > 90 && reliability_score > 2
        reliability_score = max(2, reliability_score - 1);

        % Update class name
        score_to_class = {'very_low', 'low', 'moderate', 'good', 'excellent'};
        reliability_class = score_to_class{reliability_score};
    end
end
