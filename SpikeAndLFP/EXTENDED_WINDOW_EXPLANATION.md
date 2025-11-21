# Extended Window GLM: Capturing Preparatory and Post-Event Activity

## Problem: Your PSTH Shows Pre-Event Activity

Based on your observation, neural responses to events (IR1ON, IR2ON, WP1ON, WP2ON, Aversive) span:
- **-1 second before event**: Preparatory activity (anticipation, approach, planning)
- **0 to +2 seconds after event**: Response and extended effects

## Solution: Acausal Basis Functions

The updated GLM now uses **10 raised cosine basis functions** spanning **[-1, +2] seconds** around each event.

### Visual Example

```
Timeline around reward event:
                    ↓ IR1ON Event Occurs
    ════════════════╬════════════════════════════
    -1s             0                        +2s
    ← Preparation → ← Response & Extended Effect →
```

### Basis Functions Layout

The 10 basis functions are distributed across the 3-second window:

```
Basis  1:  ╱╲___________________________   (peaks at -900ms, anticipatory)
Basis  2:  ___╱╲________________________   (peaks at -600ms)
Basis  3:  ______╱╲_____________________   (peaks at -300ms)
Basis  4:  _________╱╲__________________   (peaks at 0ms, event onset)
Basis  5:  ____________╱╲_______________   (peaks at +300ms)
Basis  6:  _______________╱╲____________   (peaks at +600ms)
Basis  7:  __________________╱╲_________   (peaks at +900ms)
Basis  8:  _____________________╱╲______   (peaks at +1200ms)
Basis  9:  ________________________╱╲___   (peaks at +1500ms)
Basis 10:  ___________________________╱╲   (peaks at +1900ms, late response)

         -1000ms      0ms              +2000ms
                      ↑ Event
```

## How It Works: Timeline Example

### Scenario: IR1ON event at t = 10.0 seconds

**Traditional PSTH approach:**
1. Align spikes to event
2. Average spike counts in bins around event
3. Result: One curve showing average response

**GLM approach with extended basis:**
1. Event occurs at t = 10.0s
2. Create 10 predictors, each "active" at different times relative to event:
   - Basis 1 predictor: High at t = 9.1s (900ms before event)
   - Basis 2 predictor: High at t = 9.4s (600ms before event)
   - Basis 3 predictor: High at t = 9.7s (300ms before event)
   - Basis 4 predictor: High at t = 10.0s (event onset)
   - Basis 5 predictor: High at t = 10.3s (300ms after event)
   - ... and so on ...
3. GLM finds weights w₁, w₂, ..., w₁₀ that best predict firing rate
4. Temporal filter = weighted sum of basis functions

## Design Matrix Construction

For each 50ms time bin, the design matrix contains:

```matlab
% At time t = 9.7s (300ms before IR1ON event at t=10.0s):
X[bin] = [
    1,              % Bias
    0.0,            % IR1ON_basis1 (not active yet, peaks at -900ms)
    0.0,            % IR1ON_basis2 (not active yet, peaks at -600ms)
    0.8,            % IR1ON_basis3 (ACTIVE! peaks at -300ms = NOW)
    0.2,            % IR1ON_basis4 (slightly active, peaks at 0ms)
    0.0,            % IR1ON_basis5 (not active, peaks at +300ms)
    ...
    0.0,            % IR2ON_basis1 (no IR2 events nearby)
    ...
    0.5,            % Speed (continuous)
    4.2             % Breathing (continuous)
]
```

## What the GLM Learns

After fitting, you can ask:

### 1. **Are there preparatory dynamics?**
```matlab
% If w₁, w₂, w₃ are large → unit shows anticipatory activity
% Example: Approach-related neurons increase firing before reward arrival
```

### 2. **What's the post-event response profile?**
```matlab
% If w₅, w₆, w₇ are large → sustained response
% If only w₄, w₅ large → brief phasic response
```

### 3. **When does the response peak?**
```matlab
% Reconstruct temporal filter:
temporal_filter = w₁*basis₁ + w₂*basis₂ + ... + w₁₀*basis₁₀

% Find peak:
[peak_value, peak_bin] = max(temporal_filter)
peak_time = (peak_bin - 20) * 50ms  % 20 bins = 1 sec pre-event
% e.g., peak_time = +200ms → response peaks 200ms after event
```

## Interpretation: Coefficient Patterns

### Example 1: Reward Anticipation Neuron
```
w = [0.8, 1.2, 1.5, 0.9, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]
     ^^^^^^^^^^^^^^^^^^^                                  High pre-event weights

Temporal filter:
         ╱‾‾‾╲____
        ╱      ╲___
       ╱          ╲___
    -1s    0s         +2s
           ↑ Event

→ Firing increases before reward (anticipatory)
```

### Example 2: Phasic Response Neuron
```
w = [0.0, 0.0, 0.2, 1.5, 1.2, 0.3, 0.0, 0.0, 0.0, 0.0]
                        ^^^^^^^^^^^                      High peri-event weights

Temporal filter:
              ╱╲
             ╱  ╲___
         ___╱    ╲______
    -1s    0s         +2s
           ↑ Event

→ Brief response at event onset
```

### Example 3: Sustained Response Neuron
```
w = [0.0, 0.0, 0.3, 0.8, 1.2, 1.5, 1.3, 0.9, 0.5, 0.2]
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^       High throughout post-event

Temporal filter:
              ╱‾‾‾‾‾‾‾╲
             ╱          ╲___
         ___╱               ╲___
    -1s    0s         +2s
           ↑ Event

→ Sustained elevation for 2 seconds
```

## Comparison: PSTH vs GLM

| Aspect | PSTH | GLM with Extended Basis |
|--------|------|------------------------|
| **Window** | [-1, +2] sec | [-1, +2] sec |
| **Confounds** | Cannot separate speed from reward | Separates all predictors |
| **Overlapping events** | Averages out | Handles via superposition |
| **Statistical test** | z-score > threshold | Coefficient significance |
| **Temporal resolution** | Fixed bins (50ms) | Smooth via basis functions |
| **Pre-event** | Shows average | Explains via anticipation |

## Why 10 Basis Functions?

- **3-second window** (60 bins at 50ms) is long
- **Need enough basis functions** to capture complex temporal dynamics
- **10 basis functions** gives ~6 bins per function (good resolution)
- **Too few** (e.g., 5) → can't capture detailed dynamics
- **Too many** (e.g., 20) → overfitting, redundant

## Implementation Changes

### Before (Causal Only):
```matlab
config.event_window = 0.5;  % 500ms after event
n_basis = 5

Timeline:    Event
              ↓
              [════]
              0   +0.5s
```

### After (Extended Window):
```matlab
config.event_window_pre = 1.0;   % 1s before event
config.event_window_post = 2.0;  % 2s after event
n_basis = 10

Timeline:      Event
                ↓
        [══════╬════════════]
        -1s    0         +2s
```

## Expected Results

When you run the updated analysis, **Figure 5 (Temporal Filters)** will show:

1. **Pre-event region** (shaded gray): -1000ms to 0ms
   - If weights are non-zero here → preparatory activity
   - Example: Speed often increases before reward seeking

2. **Event onset** (dashed line): 0ms
   - Sharp changes here → event-locked response

3. **Post-event region** (white): 0ms to +2000ms
   - Shape reveals response profile
   - Phasic (sharp peak) vs sustained (plateau)

## Validation: Compare to PSTH

After GLM fitting, you can validate by comparing:

```matlab
% 1. Load GLM results
load('Unit_GLM_Results.mat');

% 2. For a specific unit, reconstruct predicted PSTH
unit_idx = 1;
coeffs = results.glm_results(unit_idx).coefficients;

% 3. Extract IR1ON temporal filter
ir1_coefs = coeffs(2:11);  % IR1ON has basis 1-10
basis_funcs = createRaisedCosineBasis(10, 60);
ir1_filter = basis_funcs * ir1_coefs;

% 4. Compare to actual PSTH
% (Load from PSTH_Survey_Results.mat)
load('PSTH_Survey_Results.mat');
actual_psth = results.unit_data(unit_idx).IR1ON_psth;

% Should match!
```

## Summary

✅ **Now captures full [-1, +2] sec window** observed in your PSTHs
✅ **10 basis functions** provide smooth, flexible temporal profiles
✅ **Separates preparatory from response activity**
✅ **Disentangles confounded variables** (e.g., speed increases before reward)
✅ **Handles overlapping events** via linear superposition

The GLM will tell you: **"Is this neuron truly reward-responsive, or just speed-modulated during reward approach?"**
