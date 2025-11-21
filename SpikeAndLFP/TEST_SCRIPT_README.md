# Quick GLM Test Script

## Purpose
Test the Poisson GLM analysis on a **small subset of data** (~6-10 units) before running the full analysis on all units.

## File
`Test_GLM_Quick.m`

## What It Does

1. **Loads limited data:**
   - 2 reward sessions only
   - 3 units per session
   - Total: ~6 units

2. **Fits GLM models:**
   - Same design matrix as full analysis
   - Same optimization procedure
   - Full feature importance calculation

3. **Creates 4 visualization figures:**
   - **Figure 1**: Model performance summary (deviance, firing rates, coefficients)
   - **Figure 2**: Actual vs predicted firing rate (example unit)
   - **Figure 3**: Temporal filters for all event types
   - **Figure 4**: Feature importance across units

## How to Run

```matlab
% Navigate to directory
cd /Users/hsiehkunlin/Desktop/Matlab_scripts/reorganize/SpikeAndLFP

% Run test script
Test_GLM_Quick
```

**Expected runtime:** ~2-5 minutes (vs hours for full analysis)

## Output

### Console Output
```
=== QUICK GLM TEST ===

Test Configuration:
  Max sessions: 2
  Max units per session: 3
  Bin size: 50 ms
  Event window: [-1000, +2000] ms

=== Session 1/2: 2025...RewardSeeking...mat ===
Loaded 30 units
Building design matrix...
  Design matrix: 36000 bins × 52 predictors
  Unit 1: Fitting GLM (450 spikes)...
    ✓ Dev explained: 15.2%, Mean FR: 2.5 spikes/s
  Unit 2: Fitting GLM (820 spikes)...
    ✓ Dev explained: 22.3%, Mean FR: 4.1 spikes/s
  ...

=== TEST COMPLETE ===
Total units fitted: 6

Summary:
  Units fitted: 6
  Mean deviance explained: 18.5% (±5.2%)
  Mean firing rate: 3.2 spikes/s (±1.5)
```

### Figures
All saved to `GLM_Test_Figures/`:
- `Test_1_Model_Performance.png`
- `Test_2_Actual_vs_Predicted.png`
- `Test_3_Temporal_Filters.png`
- `Test_4_Feature_Importance.png`

## What to Check

### ✅ Good Results
- **Deviance explained**: 10-40% (positive values)
- **Temporal filters**: Show clear event responses (not flat)
- **Actual vs Predicted**: Scatter points follow diagonal line
- **Feature importance**: Some features have >5% contribution
- **No errors**: Script completes without crashes

### ⚠️ Potential Issues
- **Negative deviance**: Model worse than baseline (check for bugs)
- **Flat temporal filters**: No event responses detected
- **Poor prediction**: Scatter plot shows no correlation
- **All features ~0%**: Model not learning anything
- **Optimization warnings**: May need more iterations

## Common Issues

### Issue: "Too few spikes, skipping"
**Solution**: Normal - script skips low-firing units (<100 spikes)

### Issue: "GLM fitting failed"
**Possible causes:**
1. Design matrix has NaN/Inf values (check data loading)
2. Too few events (session too short)
3. Optimization failed to converge (increase max_iter)

### Issue: Very low deviance explained (<5%)
**Possible causes:**
1. Unit not modulated by any predictor
2. Wrong event timing (check IR1ON, IR2ON, etc.)
3. Model configuration issue (check basis functions)

### Issue: Negative deviance explained
**Possible causes:**
1. Overfitting (too many predictors for available data)
2. Bug in null model calculation
3. Optimization didn't converge

## Interpretation

### Temporal Filters (Figure 3)

**Good example:**
```
IR1ON Filter:
         ╱‾‾╲
        ╱    ╲___
    ___╱         ╲___
  -1s   0s        +2s
        ↑ Event

→ Pre-event preparation, peak at event, sustained response
```

**Flat/No response:**
```
IR1ON Filter:
  ___________________
  -1s   0s        +2s

→ Unit not modulated by IR1ON events
```

### Feature Importance (Figure 4)

**Example:**
```
IR1ON:     15%  ← Strongly modulated by IR1 port
IR2ON:      5%  ← Weakly modulated
Speed:     20%  ← Strongly modulated by locomotion
Breathing:  3%  ← Weakly modulated
```

**Interpretation**: This unit encodes both reward (IR1) and movement (speed).

## Next Steps

### If Test Looks Good:
```matlab
% Run full analysis on all units
Unit_Poisson_GLM_Analysis
```

### If Test Shows Issues:
1. Check console for specific error messages
2. Verify data paths are correct
3. Inspect figures for anomalies
4. Try different units (edit `config.max_units_per_session`)
5. Check design matrix construction

## Customization

To test different units or sessions, edit the script:

```matlab
% Line 39-40: Change number of sessions/units
config.max_sessions = 3;           % Test 3 sessions
config.max_units_per_session = 5;  % 5 units per session

% Line 113: Change session type
session_pattern = '2025*RewardAversive*.mat';  % Test aversive sessions
```

## Performance Benchmark

| Dataset Size | Expected Time |
|--------------|---------------|
| 2 sessions, 3 units each | ~2-3 min |
| 3 sessions, 5 units each | ~5-7 min |
| 5 sessions, 10 units each | ~15-20 min |

**Full analysis** (100 sessions, all units): ~1-2 hours

## Troubleshooting Checklist

- [ ] neuroGLM toolbox cloned and in current directory
- [ ] Sorting parameters loaded successfully
- [ ] Session files found in spike_folder
- [ ] Behavioral data loaded without errors
- [ ] Design matrix created (52 predictors)
- [ ] At least 1 unit fitted successfully
- [ ] Figures saved to GLM_Test_Figures/
- [ ] Deviance explained is positive (>0%)
- [ ] Temporal filters show event responses

If all checked, proceed with full analysis!

## Contact

For issues, check:
1. Console error messages
2. MATLAB version (requires R2019b or later)
3. Path settings (make sure helper functions are accessible)
4. Data file formats (check session files load correctly)
