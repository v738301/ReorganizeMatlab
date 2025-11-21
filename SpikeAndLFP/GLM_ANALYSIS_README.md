# Poisson GLM Analysis for Neural Data

## Overview
This analysis pipeline fits Generalized Linear Models to disentangle the contributions of multiple behavioral and task variables to neural firing rates.

## Files Created

### 1. **Unit_Poisson_GLM_Analysis.m**
Main analysis script that:
- Loads spike data and behavioral variables from your existing pipeline
- Creates design matrices with raised cosine basis functions (5 basis per event)
- Fits Poisson GLM for each unit using maximum likelihood
- Computes model performance (deviance explained)
- Calculates feature importance via leave-one-out analysis

### 2. **Visualize_GLM_Results.m**
Visualization script that creates:
- Coefficient heatmaps across all units
- Feature importance matrices and summaries
- Temporal filters for each event type
- Model performance distributions
- Cluster-specific analyses (if clustering data available)

### 3. **neuroGLM/** (cloned from GitHub)
Established toolbox from Pillow Lab with utilities for:
- Raised cosine basis function creation
- GLM design matrix construction
- Optimization and model fitting

## Predictors Included

### Event-Based Predictors (with raised cosine basis functions)
1. **IR1ON** - Reward port 1 onsets (10 basis functions, [-1s, +2s] window)
2. **IR2ON** - Reward port 2 onsets (10 basis functions, [-1s, +2s] window)
3. **WP1ON** - Water port 1 onsets (10 basis functions, [-1s, +2s] window)
4. **WP2ON** - Water port 2 onsets (10 basis functions, [-1s, +2s] window)
5. **Aversive** - Aversive sound onsets (10 basis functions, [-1s, +2s] window)

**Note:** The [-1s, +2s] window captures both preparatory activity before events and extended responses after events, matching what you observed in your PSTH analysis.

### Continuous Predictors
6. **Speed** - Locomotion speed (z-scored, smoothed 200ms)
7. **Breathing** - Breathing rate (z-scored, smoothed 200ms)

## How to Run

### Step 1: Fit GLM Models
```matlab
% Navigate to SpikeAndLFP directory
cd /Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP

% Run GLM analysis (this will take some time!)
Unit_Poisson_GLM_Analysis
```

**Expected output:**
- `Unit_GLM_Results.mat` - Contains coefficients and statistics for all units
- Console output showing progress and fit quality for each unit

### Step 2: Visualize Results
```matlab
% After GLM analysis completes, visualize results
Visualize_GLM_Results
```

**Expected output:**
- `GLM_Figures/` directory with 5-6 figures:
  - Fig1: Coefficient heatmap
  - Fig2: Feature importance matrix
  - Fig3: Feature importance summary
  - Fig4: Model performance
  - Fig5: Temporal filters
  - Fig6: Cluster-specific analysis (if available)
- `GLM_Summary_Statistics.txt` - Text summary of results

## Key Configuration Parameters

Located in `Unit_Poisson_GLM_Analysis.m`:

```matlab
config.bin_size = 0.05;              % 50 ms time bins
config.n_basis_funcs = 10;           % 10 raised cosine basis functions
config.event_window_pre = 1.0;       % 1000 ms pre-event window (preparatory)
config.event_window_post = 2.0;      % 2000 ms post-event window (response)
config.smooth_window = 0.2;          % 200 ms smoothing for continuous vars
```

**Why extended windows?**
- Your PSTH analysis revealed preparatory activity starting ~1 sec before events
- Responses extend up to 2 seconds after events
- The GLM now captures both anticipatory and sustained dynamics

## Understanding the Output

### Deviance Explained
- Pseudo-R² metric: 1 - (model_nll / null_model_nll)
- Values > 0 indicate model better than baseline
- Typical range: 5-40% for well-modulated units

### Feature Importance
- **% Deviance Contribution**: How much worse the model gets when excluding that predictor
- Higher values = more important for explaining firing
- Allows ranking: Which variable drives each unit?

### Temporal Filters
- Shows how firing rate changes after each event type
- Reconstructed from weighted basis functions
- Positive values = excitation, negative = suppression

## Integration with Clustering

If you have PSTH-based clustering results (from `Visualize_PSTH_by_Cluster.m`), the visualization script will automatically:
1. Load `unit_cluster_assignments.mat`
2. Match units to clusters
3. Create cluster-specific feature importance plots
4. Identify which predictors dominate each cluster

## Troubleshooting

### "Too few spikes" warnings
- Units with < 100 spikes are skipped (insufficient data for GLM)
- This is normal - focus on well-isolated, active units

### Negative deviance explained
- Model is worse than baseline (rare, ~5-10% of units)
- Usually indicates poor unit isolation or non-Poisson firing

### Memory issues
- Analysis processes all sessions - can be memory intensive
- If needed, reduce `config.numofsession` to analyze fewer sessions
- Results are saved incrementally, safe to restart

### Optimization warnings
- "Local minimum possible" - Usually okay, GLM still fitted
- "Maximum iterations reached" - Increase `config.max_iter`

## Next Steps

### 1. Identify Dominant Encoding
```matlab
% Load results
load('Unit_GLM_Results.mat');

% Find units most modulated by each predictor
for i = 1:length(results.glm_results)
    fi = results.glm_results(i).feature_importance;
    % Check fi.IR1ON.percent_deviance, fi.Speed.percent_deviance, etc.
end
```

### 2. Compare to Clustering
- Do reward-modulated clusters (from PSTH) have high IR/WP feature importance?
- Do speed-modulated clusters have high Speed importance?
- GLM can separate confounded variables (e.g., reward + speed)

### 3. Cross-Validation (Advanced)
- Modify `fitPoissonGLM()` to implement k-fold CV
- Evaluate generalization to held-out data
- More robust model comparison

### 4. Add More Predictors
- LSTM behavioral states (7 classes)
- Behavioral matrix features (8 features)
- Spike history (auto-correlation)
- LFP phase or power

## References

- **neuroGLM toolbox**: https://github.com/pillowlab/neuroGLM
- **Pillow et al. (2008)**: "Spatio-temporal correlations and visual signalling in a complete neuronal population"
- **Park et al. (2014)**: "Encoding and decoding in parietal cortex during sensorimotor decision-making"

## Contact

For issues or questions about this analysis pipeline, check:
1. Console error messages - usually informative
2. MATLAB documentation: `help fitglm`
3. neuroGLM documentation in `neuroGLM/docs/`
