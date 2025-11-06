# Cluster-Waveform-Property Analysis Pipeline

## Overview

This analysis pipeline investigates the **critical neuroscience question**: How do intrinsic cellular properties (waveform shape, firing rate, CV) relate to functional properties (oscillation coupling, task modulation)?

### Research Question

You've identified 5 cluster types from simplified functional clustering:
1. **Strong local oscillation coupling units**
2. **High firing rate units**
3. **Aversive modulated** (inhibited or activated)
4. **Reward modulated** (inhibited or activated)
5. **Aversive/Reward mix modulated units**

**Now we ask**: Are these functional differences driven by different cell types (interneurons vs pyramidal cells)?

---

## Analysis Components

### 1. Waveform Properties (Cell Type Indicators)

**Key Features Extracted:**
- **Trough-to-peak time** (ms) - Primary cell type classifier
  - Interneurons: < 0.4 ms (narrow)
  - Pyramidal cells: > 0.4 ms (broad)
- **Full-width at half-maximum (FWHM)** - Waveform duration
- **Amplitude** - Peak-to-trough voltage
- **Asymmetry index** - Waveform shape
- **Repolarization rate** - Recovery speed

**Biological Significance:**
- Narrow waveforms (fast-spiking) typically indicate GABAergic interneurons
- Broad waveforms typically indicate excitatory pyramidal cells
- These cell types have different roles in circuit function

### 2. Firing Rate (FR) & Coefficient of Variation (CV)

**Temporal Dynamics:**
- **Mean firing rate** - Overall spiking activity (Hz)
- **FR timecourse** - Changes across session (60-sec bins)
- **FR variability** - Standard deviation of FR
- **Coefficient of Variation** - ISI regularity (CV = std(ISI)/mean(ISI))
  - Low CV (~0.5-0.7): Regular firing
  - High CV (>1.0): Irregular/bursty firing

**Biological Significance:**
- Interneurons often have higher, more regular firing (high FR, low CV)
- Pyramidal cells often have lower, irregular firing (low FR, high CV)
- CV can indicate network state and functional role

### 3. Functional Features (From Clustering)

**Already computed features:**
- Spike-LFP coherence (oscillation coupling)
- Phase locking strength (MRL)
- PSTH responses to task events
- Task modulation profiles

---

## Pipeline Workflow

### Step 1: Prepare Data

**Prerequisites:**
1. Simplified clustering results: `simplified_clustering_*.mat`
2. .nex files with waveform data: `/Volumes/ExpansionBackup/Data/Ephy/`
3. Sorted spike data: `/Volumes/ExpansionBackup/Data/Struct_spike/`
4. Helper functions on MATLAB path:
   - `loadSortingParameters()`
   - `loadAndPrepareSessionData()`
   - `selectFilesWithAnimalIDFiltering()`

### Step 2: Extract Waveforms from .nex Files

**Action Required:**
You need to implement waveform extraction based on your system.

1. **Locate reference script:** `/Users/hsiehkunlin/Desktop/Matlab_scripts/SpikePipeline/TryToIdentifyWaveForm.m`

2. **Adapt template:** `Extract_Waveforms_From_Nex.m`
   - Identify which .nex reading function you use:
     - NeuroExplorer SDK: `nex_info()`, `nex_wf()`
     - Custom function: `readNexFile()`
     - FieldTrip: `ft_preprocessing()`
   - Implement `readWaveformsFromNex()` function
   - Implement `convertMatToNexFilename()` based on your naming convention

3. **Test extraction** on a few units first

**Expected output:**
```matlab
waveform_data(i).global_unit_id
waveform_data(i).waveform        % Average waveform [1 x N_samples]
waveform_data(i).waveform_std    % Standard deviation
waveform_data(i).n_waveforms     % Number of spikes averaged
waveform_data(i).sampling_rate   % Hz (typically 30000)
```

### Step 3: Run Main Analysis

**Script:** `Cluster_Waveform_Analysis.m`

This script will:
1. Load simplified clustering results
2. Extract waveforms from .nex files (using your implementation)
3. Compute waveform features (trough-to-peak, FWHM, etc.)
4. Compute temporal FR & CV from spike data
5. Integrate all data sources
6. Analyze relationships:
   - Cell type distribution across clusters
   - FR/CV properties across clusters
   - Correlations between waveform and functional features

**Key sections to customize:**
- Section 3: Waveform extraction (uncomment after implementing)
- Paths in Section 1 configuration

**Output:** `cluster_waveform_analysis_YYYY-MM-DD_HHMMSS.mat`

### Step 4: Generate Visualizations

**Script:** `Cluster_Waveform_Visualization.m`

This creates 4 comprehensive figures:

**Figure 1: Waveform Properties by Cluster**
- Trough-to-peak time distribution
- FWHM distribution
- Cell type composition (pie/bar charts)
- Waveform feature space (scatter plots)
- Example waveforms per cluster
- Asymmetry analysis

**Figure 2: Firing Properties by Cluster**
- Mean firing rate by cluster
- CV distribution by cluster
- FR vs CV relationships
- FR variability analysis
- Properties by putative cell type

**Figure 3: Waveform-Functional Feature Relationships**
- Correlation heatmap (waveform/FR/CV × functional features)
- Significance testing
- Top correlations highlighted
- Scatter plots of key relationships

**Figure 4: Temporal FR/CV Dynamics**
- FR timecourse per cluster (mean ± SEM)
- CV timecourse across clusters
- Temporal stability analysis

**Output:** Saved to `Cluster_Waveform_Figures/` directory

---

## Expected Results & Interpretations

### Hypothesis 1: Cell Type Predicts Functional Cluster

**If TRUE:**
- Interneuron-dominated clusters should show:
  - Strong oscillation coupling (high coherence)
  - High, regular firing (high FR, low CV)
  - Fast responses to stimuli

- Pyramidal-dominated clusters should show:
  - Task modulation (reward/aversive selective)
  - Lower, irregular firing (low FR, high CV)
  - Broader PSTH responses

**Test:** Chi-square test on cell type × cluster contingency table

### Hypothesis 2: FR/CV Profile Distinguishes Functional Roles

**If TRUE:**
- High FR units = oscillation generators
- High CV units = task-responsive units
- FR/CV should correlate with functional features

**Test:** Spearman correlations in Figure 3

### Hypothesis 3: Mixed Cell Types Serve Different Roles

**If TRUE:**
- Each cluster contains both cell types
- Waveform features weakly correlate with functional features
- Suggests cell-type-independent functional specialization

**Test:** Within-cluster variance in waveform properties

---

## Data Files Structure

### Input Files

```
simplified_clustering_Aversive_*.mat
├── clustering                 % Hierarchical clustering results
├── features                   % Feature matrix & names
├── units                      % Per-unit metadata & cluster assignments
├── cluster_lookup             % Per-cluster information
└── metadata                   % Analysis configuration

/Volumes/ExpansionBackup/Data/Ephy/*.nex
└── Waveform data for each recording

/Volumes/ExpansionBackup/Data/Struct_spike/*.mat
└── Sorted spike times & session data
```

### Output Files

```
cluster_waveform_analysis_*.mat
├── config                     % Analysis configuration
├── clustering_results         % Original clustering data
├── waveform_features          % Extracted waveform properties
├── temporal_fr_cv             % Temporal FR/CV dynamics
└── integrated_data            % All data combined per unit
    ├── global_unit_id
    ├── cluster_id
    ├── functional_features    % From clustering
    ├── wf_trough_to_peak     % Cell type indicator
    ├── wf_fwhm
    ├── wf_asymmetry
    ├── putative_cell_type    % "Interneuron" or "Pyramidal"
    ├── mean_fr
    ├── mean_cv
    ├── fr_timecourse
    └── cv_timecourse
```

---

## Implementation Checklist

- [x] Create analysis script framework (`Cluster_Waveform_Analysis.m`)
- [x] Create visualization script (`Cluster_Waveform_Visualization.m`)
- [x] Create waveform extraction template (`Extract_Waveforms_From_Nex.m`)
- [ ] **USER ACTION:** Implement waveform extraction from .nex files
  - [ ] Identify .nex reading function in `TryToIdentifyWaveForm.m`
  - [ ] Implement `readWaveformsFromNex()`
  - [ ] Implement `convertMatToNexFilename()`
  - [ ] Test on a few units
- [ ] **USER ACTION:** Run `Cluster_Waveform_Analysis.m`
- [ ] **USER ACTION:** Run `Cluster_Waveform_Visualization.m`
- [ ] **USER ACTION:** Interpret results and answer research question

---

## Troubleshooting

### Issue: .nex files not found

**Solution:** Check file naming convention in `convertMatToNexFilename()`. Print out expected vs actual filenames to debug.

### Issue: Waveform extraction fails

**Solution:**
1. Check which .nex reading tools you have installed
2. Refer to `TryToIdentifyWaveForm.m` on your local machine
3. Test with `nex_info(filepath)` to see file structure

### Issue: FR/CV computation is slow

**Solution:**
- Reduce `config.fr_bin_size` (currently 60 sec)
- Process fewer sessions for testing
- Add session filtering in config

### Issue: No significant correlations found

**Interpretation:** This could be meaningful! It might suggest:
- Functional roles are independent of cell type
- Need more units for statistical power
- Features need refinement

---

## Statistical Considerations

### Multiple Comparisons

When testing many correlations, use correction:
- Bonferroni: α = 0.05 / n_tests
- FDR (False Discovery Rate)
- Report uncorrected p-values with asterisks as initial exploration

### Sample Size

Minimum recommended units per cluster for reliable analysis:
- Cell type analysis: ≥ 10 units
- Correlation analysis: ≥ 15 units
- Temporal dynamics: ≥ 5 units with full timecourse

### Effect Sizes

Report both statistical significance AND effect sizes:
- Spearman r > 0.3: Small effect
- Spearman r > 0.5: Medium effect
- Spearman r > 0.7: Large effect

---

## Next Steps After Analysis

### If cell type predicts cluster:
1. Re-run clustering separately for interneurons and pyramidal cells
2. Ask: Do interneurons have functional subtypes?
3. Investigate interneuron-pyramidal interactions

### If cell type doesn't predict cluster:
1. Investigate other waveform features (e.g., AHP depth)
2. Consider that functional roles transcend cell type
3. Look for within-cell-type functional diversity

### Publication/Presentation:
1. Figure 1: Cell type composition across clusters
2. Figure 3: Waveform-functional correlations (key finding)
3. Figure 4: Temporal dynamics by cluster type
4. Statistical table: Cluster × Cell Type × Features

---

## References

**Cell Type Classification:**
- Barthó et al. (2004) - Characterization of neocortical principal cells and interneurons
- Mitchell et al. (2007) - Interneurons are narrow-spiking
- Sirota et al. (2008) - Spike shape analysis

**Firing Statistics:**
- Shinomoto et al. (2009) - CV interpretation
- Softky & Koch (1993) - Irregular firing in cortex

**Functional Clustering:**
- Your own: `README_Simplified_Clustering_Results.md`

---

## Contact & Support

For questions about:
- Analysis pipeline: See this README
- Clustering results: See `README_Simplified_Clustering_Results.md`
- Waveform extraction: Refer to your `TryToIdentifyWaveForm.m`

---

**Last Updated:** 2025-01-06
**Author:** Analysis Pipeline
**Version:** 1.0
