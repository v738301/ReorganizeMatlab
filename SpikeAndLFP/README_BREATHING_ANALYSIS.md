# Spike-Breathing Coupling Analysis Scripts

This directory contains analysis scripts for investigating spike-breathing signal coupling using channel 32 from the ephys data.

## Overview

The breathing analysis scripts are parallel versions of the LFP coupling analyses, adapted to use the breathing signal (channel 32) instead of selecting the best LFP channel.

## Analysis Scripts

### 1. Unit_PPC_Analysis_Breathing.m
**Purpose**: Compute Pairwise Phase Consistency (PPC) for spike-breathing coupling

**Key Features**:
- Analyzes phase coupling between spikes and breathing signal
- Frequency bands: 0.1-20 Hz (1 Hz bins)
- Periods: 7 for aversive, 4 for reward
- **Channel**: Uses channel 32 (breathing signal)

**Output**:
- `DataSet/RewardSeeking_UnitPPC_Breathing/`
- `DataSet/RewardAversive_UnitPPC_Breathing/`
- Files: `*_unit_ppc_breathing.mat`

**Metrics**:
- PPC (Pairwise Phase Consistency): 0-1, unbiased by spike count
- Preferred phase: circular mean of spike phases
- Reliability: based on spike count

### 2. Spike_Triggered_LFP_Analysis_Breathing.m
**Purpose**: Compute spike-triggered breathing signal averages

**Key Features**:
- Uses broadband breathing signal (0.1-300 Hz)
- STA window: ±500 ms around spikes
- Minimum spikes: 50
- **Channel**: Uses channel 32 (breathing signal)

**Output**:
- `DataSet/RewardSeeking_STA_Breathing/`
- `DataSet/RewardAversive_STA_Breathing/`
- Files: `*_sta_breathing.mat`

**Metrics**:
- STA waveform: mean breathing signal around spikes
- STA peak amplitude
- STA consistency: lower = more phase-locked

### 3. Amplitude_Correlation_Analysis_Breathing.m
**Purpose**: Test spike rate vs breathing amplitude correlation

**Key Features**:
- Extracts breathing amplitude envelope (Hilbert transform)
- Bins spike rate over time (100 ms bins)
- Computes Pearson correlation (linear coupling)
- Computes mutual information (nonlinear coupling)
- Frequency bands: 0.1-20 Hz (1 Hz bins)
- **Channel**: Uses channel 32 (breathing signal)

**Output**:
- `DataSet/RewardSeeking_AmpCorr_Breathing/`
- `DataSet/RewardAversive_AmpCorr_Breathing/`
- Files: `*_ampcorr_breathing.mat`

**Metrics**:
- Pearson R: linear correlation
- Mutual information: nonlinear coupling
- Interpretation: High MI + Low R → context-dependent coupling

### 4. Spike_LFP_Coherence_Overall_Breathing.m
**Purpose**: Compute coherence between spikes and breathing signal

**Key Features**:
- Multitaper spectral estimation (TW=3, K=5)
- Frequency range: 1-150 Hz
- Window size: 10 sec (for memory efficiency)
- Minimum spikes: 50
- **Channel**: Uses channel 32 (breathing signal)

**Output**:
- `DataSet/RewardSeeking_SpikeBreathingCoherence_Overall/`
- `DataSet/RewardAversive_SpikeBreathingCoherence_Overall/`
- Files: `*_spike_breathing_coherence_overall.mat`

**Metrics**:
- Coherence spectrum: 1-150 Hz
- Phase spectrum
- Power spectra (spike & breathing)
- Band-specific mean coherence (Delta, Theta, Beta, Gamma)

## Visualization Scripts

### 1. Visualize_Unit_PPC_Breathing.m
Visualizes PPC results from Unit_PPC_Analysis_Breathing.m

**Figures**:
- PPC Spectrograms (Frequency × Period)
- Frequency Band Summaries
- Preferred Phase Analysis
- Session-Level PPC Spectra
- Unit-Level PPC Distributions
- Individual Unit PPC Spectra
- Session-by-Session Heatmaps

**Output Directory**: `Unit_PPC_Breathing_Figures/`

### 2. Visualize_Spike_Triggered_LFP_Breathing.m
Visualizes STA results

**Note**: To create this script, copy `Visualize_Spike_Triggered_LFP.m` and update:
- Paths: `RewardSeeking_STA_Breathing`, `RewardAversive_STA_Breathing`
- File pattern: `*_sta_breathing.mat`
- Output directory: `STA_Breathing_Figures`
- Titles: Add "Breathing Signal" references

### 3. Visualize_Amplitude_Correlation_Breathing.m
Visualizes amplitude correlation results

**Note**: To create this script, copy `Visualize_Amplitude_Correlation.m` and update:
- Paths: `RewardSeeking_AmpCorr_Breathing`, `RewardAversive_AmpCorr_Breathing`
- File pattern: `*_ampcorr_breathing.mat`
- Output directory: `AmpCorr_Breathing_Figures`
- Titles: Add "Breathing Signal" references

### 4. Visualize_Spike_LFP_Coherence_Overall_Breathing.m
Visualizes coherence results

**Note**: To create this script, copy `Visualize_Spike_LFP_Coherence_Overall.m` and update:
- Paths: `RewardSeeking_SpikeBreathingCoherence_Overall`, `RewardAversive_SpikeBreathingCoherence_Overall`
- File pattern: `*_spike_breathing_coherence_overall.mat`
- Output directory: `SpikeBreathingCoherence_Figures`
- Titles: Add "Breathing Signal" references

## Key Differences from LFP Analysis

| Aspect | LFP Analysis | Breathing Analysis |
|--------|--------------|-------------------|
| Signal Source | `findBestLFPChannel()` | Channel 32 (fixed) |
| Variable Name | `LFP` | `Breathing` |
| Output Paths | `*_UnitPPC/` | `*_UnitPPC_Breathing/` |
| File Names | `*_unit_ppc.mat` | `*_unit_ppc_breathing.mat` |

## Expected Breathing Frequencies

Breathing in rodents typically occurs at:
- **Resting**: 0.5-2 Hz (~30-120 breaths/min)
- **Active**: 2-4 Hz (~120-240 breaths/min)
- **Sniffing**: 4-12 Hz (~240-720 Hz, high-frequency exploratory sniffing)

Therefore, the most relevant frequency bands for breathing are:
- **0.1-1 Hz**: Very slow breathing
- **1-4 Hz**: Normal breathing range
- **4-12 Hz**: Sniffing/exploratory breathing

## Running the Analysis

### Step 1: Run Analysis Scripts

```matlab
% 1. PPC Analysis
run Unit_PPC_Analysis_Breathing.m

% 2. Spike-Triggered Average
run Spike_Triggered_LFP_Analysis_Breathing.m

% 3. Amplitude Correlation
run Amplitude_Correlation_Analysis_Breathing.m

% 4. Coherence Analysis
run Spike_LFP_Coherence_Overall_Breathing.m
```

### Step 2: Run Visualization Scripts

```matlab
% After completing analysis scripts above:

% 1. Visualize PPC
run Visualize_Unit_PPC_Breathing.m

% 2-4. Create and run other visualization scripts
% (Copy and modify as described above)
```

## Interpreting Results

### PPC (Pairwise Phase Consistency)
- **High PPC (>0.2)**: Strong phase-locking to breathing rhythm
- **Low PPC (<0.1)**: Weak/no phase-locking
- **Expected**: Higher PPC at breathing frequencies (1-4 Hz)

### Spike-Triggered Average
- **Sharp, rhythmic waveform**: Phase-locked to breathing
- **Flat/noisy waveform**: No phase-locking (could still show amplitude coupling)
- **Low consistency**: Amplitude-modulated, not phase-locked

### Amplitude Correlation
- **High Pearson R**: Linear relationship between spike rate and breathing amplitude
- **High MI, Low R**: Nonlinear/context-dependent coupling
- **Both low**: No amplitude coupling

### Coherence
- **High coherence**: Strong rhythmic coupling
- **Peaks at specific frequencies**: Identify dominant coupling frequencies
- **Compare with PPC**: High coherence + Low PPC = amplitude coupling only

## Comparison with LFP

To understand differences between LFP and breathing coupling:

1. **Run both analyses** (LFP and breathing versions)
2. **Compare PPC values** across frequency bands
3. **Look for dissociations**:
   - Units coupled to LFP but not breathing
   - Units coupled to breathing but not LFP
   - Units coupled to both (cortico-respiratory coupling)

## Data Locations

All analysis outputs are saved to:
```
/Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP/DataSet/
├── RewardSeeking_UnitPPC_Breathing/
├── RewardAversive_UnitPPC_Breathing/
├── RewardSeeking_STA_Breathing/
├── RewardAversive_STA_Breathing/
├── RewardSeeking_AmpCorr_Breathing/
├── RewardAversive_AmpCorr_Breathing/
├── RewardSeeking_SpikeBreathingCoherence_Overall/
└── RewardAversive_SpikeBreathingCoherence_Overall/
```

## Notes

1. **Channel 32 must contain valid breathing signal** in your data files
2. **Preprocessing** uses the same `preprocessSignals()` function as LFP
3. **Frequency range**: Scripts test 0.1-20 Hz in 1 Hz bins (matching LFP analysis)
4. **Performance**: Pre-allocated arrays and optimized for large datasets

## References

- **PPC**: Vinck et al., NeuroImage 51(1):112-122, 2010
- **Multitaper coherence**: Chronux toolbox (Mitra & Bokil, 2008)
- **Breathing-neuron coupling**: Tort et al., Neuron 2018; Lockmann et al., Nature Communications 2016

## Contact

For questions about these scripts, refer to the main analysis pipeline documentation or contact the lab.

---
**Created**: 2025
**Last Updated**: 2025-11-17
