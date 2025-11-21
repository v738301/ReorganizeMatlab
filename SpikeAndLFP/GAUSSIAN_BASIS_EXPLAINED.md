# Gaussian Basis Functions for GLM

## What Changed

### Before: Raised Cosine Basis
- **10 basis functions** logarithmically spaced
- Complex to interpret (log-scaled peaks)
- Non-uniform temporal resolution

### After: Gaussian Basis
- **5 symmetric Gaussian functions**
- **Centers**: -0.5s, 0s, 0.5s, 1.0s, 1.5s
- **Width (FWHM)**: 1.0 second
- Easy to interpret (each basis = time period)

---

## Visual Representation

Run `visualize_basis_functions.m` to see the basis functions:

```
Basis 1 (center = -0.5s):  Preparatory activity (~500ms before event)
Basis 2 (center = 0s):     Event-locked response
Basis 3 (center = 0.5s):   Early post-event (~500ms after)
Basis 4 (center = 1.0s):   Mid post-event (~1s after)
Basis 5 (center = 1.5s):   Late post-event (~1.5s after)
```

### Shape:
```
           Basis 2 (centered at event)
                 ╱‾‾╲
                ╱    ╲
            ___╱      ╲___
          -1s   0s    +1s   +2s
                ↑
              Event
```

Each Gaussian has:
- **FWHM**: 1.0 second (full width at half maximum)
- **Sigma (σ)**: ~0.425 seconds
- **Coverage**: ±1.5σ ≈ ±0.64 seconds from center

---

## Why Gaussian Basis Functions?

### 1. **Easier Interpretation**
```matlab
% GLM coefficients for IR1ON:
w_IR1ON = [0.2, 1.5, 0.8, 0.3, 0.1];
           ↑    ↑    ↑    ↑    ↑
         -0.5s  0s  0.5s  1s  1.5s

Interpretation: "Firing increases strongly at event onset (w=1.5),
                 with moderate increase 500ms after (w=0.8)"
```

### 2. **Symmetric & Smooth**
- Gaussian is the "smoothest" function (maximum entropy)
- Symmetric around center (no temporal bias)
- Differentiable everywhere (good for optimization)

### 3. **Better for Sparse Events**
- 5 basis functions instead of 10
- Less overfitting with limited data
- Each basis covers a clear time period

### 4. **Easier Model Comparison**
- Centers align with PSTH time points
- Can directly compare GLM filters to PSTH
- Weights have clear temporal meaning

---

## Mathematical Details

### Gaussian Formula
```
B_i(t) = exp(-((t - c_i)^2) / (2σ^2))
```
Where:
- `t` = time from event
- `c_i` = center of basis i
- `σ` = width parameter (related to FWHM)

### FWHM to Sigma Conversion
```
FWHM = 2 * sqrt(2 * ln(2)) * σ ≈ 2.355 * σ

For FWHM = 1.0 second:
σ = 1.0 / 2.355 ≈ 0.425 seconds
```

### Normalization
Each basis function is normalized so that:
```
sum(B_i) = 1.0
```
This makes weights interpretable as "average contribution" over the covered time period.

---

## Coverage Analysis

### Temporal Coverage
```
Time:        -1.0s  -0.5s   0s   0.5s   1.0s   1.5s   2.0s
Basis 1:      ╱‾╲_________________________________
Basis 2:      ___________╱‾╲______________________
Basis 3:      __________________╱‾╲_______________
Basis 4:      __________________________╱‾╲_______
Basis 5:      __________________________________╱‾╲

Combined:     ═══════════════════════════════════  (full coverage!)
```

**Key insight**: The 5 Gaussians with 1-second FWHM provide complete, overlapping coverage of the [-1, +2] second window.

### Overlap
Adjacent basis functions overlap by ~60% at their half-maximum points:
```
Basis 2 (center=0s) and Basis 3 (center=0.5s) both have significant weight at t=0.25s
→ Model can flexibly represent responses that peak between basis centers
```

---

## Comparison to Raised Cosine

| Aspect | Raised Cosine | Gaussian |
|--------|---------------|----------|
| **Number of basis** | 10 | 5 |
| **Spacing** | Logarithmic | Uniform (0.5s) |
| **Shape** | Cosine bumps | Smooth bell curves |
| **Symmetry** | Yes | Yes |
| **Early resolution** | High (many early basis) | Moderate |
| **Late resolution** | Low (few late basis) | Moderate |
| **Interpretation** | "Basis 7 peaks at..." | "0.5s after event" |
| **Overfitting risk** | Higher (more params) | Lower |

### When to Use Each:

**Raised Cosine:**
- Need high temporal resolution early (e.g., sensory responses)
- Long-duration responses (>3 seconds)
- Many events (>100 per session)

**Gaussian (current choice):**
- Want easy interpretation
- Uniform temporal interest
- Moderate event counts
- Comparing to PSTH results

---

## Example GLM Interpretation

### Scenario: Reward-responsive unit

**GLM Coefficients:**
```
IR1ON_basis1:  0.3   (-0.5s, preparatory)
IR1ON_basis2:  1.8   (0s, event onset) ←  Strong!
IR1ON_basis3:  1.2   (0.5s, early response)
IR1ON_basis4:  0.4   (1.0s, sustained)
IR1ON_basis5:  0.1   (1.5s, return to baseline)

Speed:         0.5   (moderate speed modulation)
Breathing:     0.1   (minimal breathing effect)
```

**Interpretation:**
1. **Preparatory activity** (basis 1 = 0.3): Slight increase 500ms before reward
2. **Strong phasic response** (basis 2 = 1.8): Large firing rate increase at reward delivery
3. **Sustained response** (basis 3-4): Elevated firing for ~1 second post-reward
4. **Also speed-modulated** (Speed = 0.5): Firing increases during locomotion
5. **Not breathing-modulated** (Breathing ≈ 0)

**Conclusion**: This is a reward-responsive unit with anticipatory activity and sustained post-reward firing, also modulated by movement.

---

## Reconstructing Temporal Filters

To visualize the temporal response profile:

```matlab
% Load GLM results
w_IR1ON = [0.3, 1.8, 1.2, 0.4, 0.1];  % Coefficients for IR1ON

% Reconstruct filter
basis = createGaussianBasis([-0.5, 0, 0.5, 1.0, 1.5], 1.0, 60, 0.05);
temporal_filter = basis * w_IR1ON';

% Plot
time_ms = ((-20:39) * 50) - 1000;  % -1000ms to +2000ms
plot(time_ms, temporal_filter);
xlabel('Time from IR1ON (ms)');
ylabel('Relative Firing Rate Change');
```

This gives you the **actual temporal response profile** learned by the GLM!

---

## Configuration

In `Test_GLM_Quick.m`:

```matlab
config.basis_type = 'gaussian';
config.basis_centers = [-0.5, 0, 0.5, 1.0, 1.5];  % 5 centers
config.basis_width = 1.0;                         % 1 second FWHM
```

**To modify:**
- **More resolution**: Add more centers (e.g., every 0.25s)
- **Different window**: Change centers (e.g., [0, 0.5, 1, 1.5, 2] for causal only)
- **Narrower basis**: Decrease FWHM (e.g., 0.5s for sharper peaks)
- **Wider basis**: Increase FWHM (e.g., 1.5s for smoother filters)

---

## Validation

To verify the Gaussian basis functions work correctly:

1. **Run visualization:**
   ```matlab
   visualize_basis_functions
   ```
   Check that basis functions:
   - Are centered at correct times
   - Have appropriate width
   - Cover the full [-1, +2] second window

2. **Run test GLM:**
   ```matlab
   Test_GLM_Quick
   ```
   Check that:
   - Models fit successfully
   - Temporal filters show clear event responses
   - Deviance explained is positive (>0%)

3. **Compare to PSTH:**
   - GLM temporal filters should resemble PSTH shapes
   - Peaks should occur at similar times
   - Overall response profiles should match

---

## Summary

✅ **5 symmetric Gaussian basis functions**
✅ **Centers at: -0.5s, 0s, 0.5s, 1.0s, 1.5s**
✅ **FWHM: 1.0 second (smooth, overlapping coverage)**
✅ **Easy interpretation**: Each basis = specific time period
✅ **Direct comparison to PSTH** time windows
✅ **Reduced overfitting** (5 params vs 10)

The Gaussian basis provides an optimal balance between temporal resolution and interpretability for your neural data analysis!
