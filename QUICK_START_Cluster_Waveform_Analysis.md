# Quick Start: Cluster-Waveform Analysis

## What We're Investigating

**Research Question:** Do the 5 cluster types you identified (oscillation coupling, high FR, aversive/reward modulated units) correspond to different **cell types** (interneurons vs pyramidal cells)?

**Why This Matters:** Understanding if functional properties are driven by cell-type differences vs. cell-type-independent mechanisms.

---

## What I've Created For You

### 1. Main Analysis Script
**File:** `SpikeAndLFP/Cluster_Waveform_Analysis.m`

**What it does:**
- Loads your simplified clustering results
- Extracts waveforms from .nex files (YOU need to implement the .nex reading part)
- Computes waveform features (trough-to-peak time, width, etc.)
- Classifies units as putative interneurons (<0.4ms) or pyramidal cells (>0.4ms)
- Computes temporal FR & CV across the session
- Integrates all data and runs correlation analyses

### 2. Visualization Script
**File:** `SpikeAndLFP/Cluster_Waveform_Visualization.m`

**Creates 4 figures:**
1. Waveform properties by cluster
2. FR/CV properties by cluster
3. Waveform-functional feature correlations (KEY FIGURE)
4. Temporal FR/CV dynamics

### 3. Waveform Extraction Template
**File:** `SpikeAndLFP/Extract_Waveforms_From_Nex.m`

**YOUR ACTION REQUIRED:**
- Adapt this based on your .nex reading method
- Look at `/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/TryToIdentifyWaveForm.m`
- Implement two functions:
  1. `readWaveformsFromNex()` - reads waveforms from .nex file
  2. `convertMatToNexFilename()` - converts your .mat filenames to .nex filenames

### 4. Comprehensive README
**File:** `SpikeAndLFP/README_Cluster_Waveform_Analysis.md`

Full documentation of the pipeline, hypotheses, expected results, and troubleshooting.

---

## What You Need To Do

### STEP 1: Implement Waveform Extraction (REQUIRED)

1. **Open:** `/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/TryToIdentifyWaveForm.m`

2. **Identify** which .nex reading function you use:
   - NeuroExplorer SDK? (`nex_info`, `nex_wf`)
   - Custom function? (`readNexFile`)
   - FieldTrip? (`ft_preprocessing`)

3. **Copy** the relevant code into `Extract_Waveforms_From_Nex.m`

4. **Implement** these functions in that file:
   ```matlab
   function [waveforms, timestamps, sampling_rate] = readWaveformsFromNex(nex_filepath, unit_id)
   % YOUR CODE HERE
   end

   function nex_filename = convertMatToNexFilename(mat_filename)
   % Example: '2025_Animal01_RewardAversive_sorted.mat' -> '2025_Animal01.nex'
   % YOUR NAMING CONVENTION HERE
   end
   ```

5. **Test** on a few units to make sure it works

### STEP 2: Update Paths in Main Script

**Edit:** `Cluster_Waveform_Analysis.m` Section 1

```matlab
config.nex_data_path = '/Volumes/ExpansionBackup/Data/Ephy';  % ✓ Correct
config.spike_data_path = '/Volumes/ExpansionBackup/Data/Struct_spike';  % ✓ Correct
```

### STEP 3: Uncomment Waveform Extraction Code

**In:** `Cluster_Waveform_Analysis.m` Section 3

Once you've implemented the extraction functions, uncomment lines ~66-91:
```matlab
% TODO: Uncomment and implement
for unit_idx = 1:length(results.units)
    unit = results.units(unit_idx);

    % Find corresponding .nex file
    nex_filename = convertMatFilenameToNex(unit.session_filename);
    nex_filepath = fullfile(config.nex_data_path, nex_filename);

    % ... rest of code
end
```

### STEP 4: Run Analysis

```matlab
% In MATLAB:
cd /Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP

% Run main analysis
Cluster_Waveform_Analysis
% This will prompt you to select your clustering results file

% Run visualization
Cluster_Waveform_Visualization
% This will prompt you to select the analysis results
```

### STEP 5: Interpret Results

**Key Questions to Answer:**

1. **Cell Type Distribution:**
   - Are certain clusters dominated by interneurons or pyramidal cells?
   - Check Figure 1, Panel C (cell type composition)

2. **Waveform-Function Relationship:**
   - Do waveform features correlate with functional features?
   - Check Figure 3 (correlation heatmap)
   - Look for significant correlations (p < 0.05, marked with *)

3. **FR/CV by Cell Type:**
   - Do interneurons have higher FR and lower CV?
   - Check Figure 2, Panels E and F

4. **Temporal Dynamics:**
   - Do different clusters show different temporal patterns?
   - Check Figure 4

---

## Expected Outcomes

### Scenario A: Cell Type Predicts Cluster
**Observation:** Clusters are dominated by one cell type
**Interpretation:** Functional roles are cell-type specific
- Interneuron clusters → Oscillation coupling
- Pyramidal clusters → Task modulation

### Scenario B: Cell Type Doesn't Predict Cluster
**Observation:** Each cluster has mixed cell types
**Interpretation:** Functional specialization is independent of cell type
- Both cell types can be task-modulated
- Both can couple to oscillations
- Functional diversity within cell types

### Scenario C: Partial Relationship
**Observation:** Some clusters are cell-type specific, others are mixed
**Interpretation:** Multiple mechanisms
- Some functions require specific cell types
- Other functions are cell-type independent

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| ".nex file not found" | Check `convertMatToNexFilename()` logic |
| "Waveform extraction fails" | Test `readWaveformsFromNex()` on one file manually |
| "No waveform data" | Make sure `waveform_data_available = true` is set |
| "Analysis is slow" | Reduce `config.fr_bin_size` or test on fewer sessions |
| "No correlations found" | This might be real! Report null result |

---

## File Structure Created

```
SpikeAndLFP/
├── Cluster_Waveform_Analysis.m              # Main analysis script
├── Cluster_Waveform_Visualization.m         # Visualization script
├── Extract_Waveforms_From_Nex.m             # Template for waveform extraction
├── README_Cluster_Waveform_Analysis.md      # Full documentation
└── QUICK_START_Cluster_Waveform_Analysis.md # This file
```

**After running analysis:**
```
SpikeAndLFP/
├── cluster_waveform_analysis_2025-*.mat     # Analysis results
└── Cluster_Waveform_Figures/                # Saved figures
    ├── Fig1_Waveform_Properties_*.png
    ├── Fig2_Firing_Properties_*.png
    ├── Fig3_Feature_Correlations_*.png
    └── Fig4_Temporal_Dynamics_*.png
```

---

## Next Steps After Analysis

1. **Examine Figure 3** correlation heatmap - this answers your main question
2. **Quantify** cell type enrichment in clusters (chi-square test)
3. **Compare** your results to literature on mPFC cell types
4. **Consider** re-running clustering separately for each cell type
5. **Investigate** units that don't fit expected patterns

---

## Important Notes

⚠️ **YOU MUST implement waveform extraction** - I provided templates but the actual .nex reading code depends on your system

✓ **FR/CV analysis code is complete** - it uses the existing helper functions

✓ **Visualization is ready to run** - once you have waveform data

✓ **Statistical testing is included** - correlation analysis with significance

---

## Quick Reference: What Each Script Does

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| `Extract_Waveforms_From_Nex.m` | .nex files | waveform_data struct | Extract mean waveforms |
| `Cluster_Waveform_Analysis.m` | Clustering results<br>.nex files<br>Spike data | cluster_waveform_analysis_*.mat | Integrate all data |
| `Cluster_Waveform_Visualization.m` | Analysis results | 4 figures | Visualize relationships |

---

**Ready to start?**
1. Implement waveform extraction (Step 1)
2. Run analysis (Step 4)
3. Interpret results (Step 5)

**Questions?** See `README_Cluster_Waveform_Analysis.md` for full details.
