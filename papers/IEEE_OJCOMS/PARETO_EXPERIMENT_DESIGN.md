# Pareto Frontier Experimental Design
## Parameter-Efficiency vs Performance Analysis for PEFT Methods

**Date**: 2025-11-02
**Status**: Planned - Ready for Execution
**Purpose**: Scientific comparison of PEFT methods for channel estimation

---

## 1. Overview

### 1.1 Motivation

**Problem Identified**:
Initial experimental design compared PEFT methods with vastly different parameter counts:
- Adapter: 524,288 params (27.87%)
- LoRA: 98,304 params (7.18%)
- Hybrid: 61,952 params (4.53%)
- Prompt: 12,800 params (0.94%)

**Issue**: Cannot fairly compare performance when parameter budgets differ by 40×.
- If Adapter > LoRA, is it because the method is better, or just because it has 5× more parameters?
- If Prompt < LoRA, is it inherently worse, or just more parameter-efficient?

**Solution**: **Pareto Frontier Analysis** - Test each method at multiple parameter scales to create parameter-performance curves.

---

## 2. Experimental Design

### 2.1 Pareto Frontier Approach

Rather than single point comparison, we create **parameter-performance curves** for each method:

```
Performance (NMSE)
      ↑
      │        Full Fine-tuning (1.35M params)
      │       /
      │      /  Adapter curves
      │     /  /
      │    /  /  LoRA curves
      │   /  /  /
      │  /  /  /  Hybrid curves
      │ /  /  /  /
      │/  /  /  /  Prompt curves
      └──────────────────────→ Trainable Parameters
```

**Key Insight**: The method that achieves the **highest performance per parameter** wins.

### 2.2 Configuration Matrix

| Method | Config Name | Hyperparameter | Trainable Params | % of Full Model |
|--------|-------------|----------------|------------------|-----------------|
| **Adapter** | dim8 | bottleneck=8 | ~65,536 | 4.83% |
| | dim16 | bottleneck=16 | ~131,072 | 9.66% |
| | dim32 | bottleneck=32 | ~262,144 | 19.33% |
| | dim64 | bottleneck=64 | ~524,288 | 38.66% |
| **LoRA** | r4 | rank=4 | ~19,660 | 1.45% |
| | r8 | rank=8 | ~39,320 | 2.90% |
| | r16 | rank=16 | ~78,640 | 5.80% |
| | r20 | rank=20 | ~98,304 | 7.25% |
| **Prompt** | len50 | length=50 | 6,400 | 0.47% |
| | len100 | length=100 | 12,800 | 0.94% |
| | len200 | length=200 | 25,600 | 1.89% |
| **Hybrid** | p50_r5 | (prompt=50, rank=5) | ~30,230 | 2.23% |
| | p100_r10 | (prompt=100, rank=10) | ~61,952 | 4.57% |
| | p200_r20 | (prompt=200, rank=20) | ~123,904 | 9.14% |

**Total Configurations**: 4 + 4 + 3 + 3 = **14 configs**

### 2.3 Scenarios

All methods tested on **5 scenarios**:
1. **InH** (Indoor Hotspot): Dense indoor, LoS/NLoS
2. **InF** (Indoor Factory): 5 sub-scenarios (SL, SH, DL, DH, HH)
3. **UMi** (Urban Micro): Small cell urban, LoS/NLoS
4. **UMa** (Urban Macro): Large cell urban, LoS/NLoS
5. **RMa** (Rural Macro): Extended range rural, LoS/NLoS

### 2.4 Total Experimental Load

```
14 configs × 5 scenarios = 70 runs

Breakdown:
- Adapter:  4 configs × 5 scenarios = 20 runs (~40 hours)
- LoRA:     4 configs × 5 scenarios = 20 runs (~40 hours)
- Prompt:   3 configs × 5 scenarios = 15 runs (~60 hours)
- Hybrid:   3 configs × 5 scenarios = 15 runs (~30 hours)

Total time estimate: 140-280 hours (6-12 days)
```

---

## 3. Key Design Decisions

### 3.1 Why Pareto, Not Fixed Budget?

**Alternative Considered**: Match all methods to ~100K params
- Adapter-12, LoRA-20, Hybrid-controlled, Prompt-781

**Why Rejected**:
1. Some methods (Prompt) are designed to work with **very few parameters** - forcing 100K may not be realistic
2. Misses the **parameter efficiency story** - Prompt achieving 95% performance with 13K vs LoRA needing 98K
3. Doesn't show **scaling behavior** - does Adapter benefit from more params? Does LoRA saturate?

**Pareto Advantage**:
- Shows **full picture** of parameter-performance tradeoff
- Can identify **optimal operating points** for each method
- Can extract **multiple insights**:
  - Which method is most parameter-efficient at low budgets?
  - Which method saturates earliest?
  - Which method benefits most from additional parameters?

### 3.2 Hybrid Configuration: Diagonal vs Grid

**Current Design (Diagonal)**:
```
(Prompt, LoRA): (50,5), (100,10), (200,20)
→ Balanced scaling: both increase together
```

**Why Start Diagonal**:
1. **Efficiency**: 3 configs vs 9 for full grid
2. **Clear interpretation**: Tests "balanced hybrid" scaling
3. **Baseline establishment**: Need to prove hybrid works at all before exploring ratios

**Future Extension (Full Grid)**:
```
        LoRA rank
         5     10    20
Prompt
50      ×     ×     ×     ← LoRA-dominant (weight adaptation)
100     ×     ×     ×     ← Balanced
200     ×     ×     ×     ← Prompt-dominant (input adaptation)
```

**Research Questions for Grid**:
- Is input-level (Prompt) or weight-level (LoRA) adaptation more important?
- What is the optimal allocation ratio for a given parameter budget?
- Do the two adaptations have multiplicative or additive effects?

**Decision**: Start with diagonal, expand to grid if:
1. Hybrid shows clear advantage over single methods
2. Reviewer requests more thorough analysis
3. Interesting behavior observed that warrants deeper investigation

### 3.3 Training Configuration

All experiments use **identical training settings**:
- **Iterations**: 100,000 (100K)
- **Checkpoints**: Every 20,000 (20K, 40K, 60K, 80K, 100K)
- **Batch Size**: 32
- **Learning Rate**: 1e-4 with cosine annealing + warmup
- **Gradient Clipping**: Max norm 1.0
- **Base Model**: `Large_estimator_v4_base_final` (1M iterations)

**Rationale**:
- 100K iterations: Balance between convergence and time
- 20K checkpoints: 5 points per curve for smooth plotting
- Identical LR: Eliminates hyperparameter tuning as confound

---

## 4. Expected Results and Analysis

### 4.1 Hypotheses

**H1 (Prompt Efficiency)**:
Prompt Learning will achieve competitive NMSE with **10-50× fewer parameters** than other methods.

**H2 (LoRA Effectiveness)**:
LoRA will provide the best **performance-per-parameter** in the mid-range (50-100K params).

**H3 (Adapter Saturation)**:
Adapter will saturate at dim=32-64, showing diminishing returns beyond ~260K params.

**H4 (Hybrid Synergy)**:
Hybrid will **dominate the Pareto frontier** in its parameter range, outperforming both Prompt-only and LoRA-only at matched parameter counts.

**H5 (Scaling Laws)**:
Each method will follow a **log-linear scaling law**: NMSE ∝ log(params), but with different slopes.

### 4.2 Key Metrics

For each (method, config, scenario) tuple:

**Primary Metric**:
- **Final NMSE** (at 100K iterations)

**Secondary Metrics**:
- **Convergence Speed** (iterations to 90% of final performance)
- **Parameter Efficiency** (NMSE / trainable_params)
- **Relative Performance** (% of full fine-tuning performance)

**Visualization**:
1. **Pareto Curves**: NMSE vs Params for each method (5 curves, 1 per scenario)
2. **Efficiency Heatmap**: Performance/Param across methods and scenarios
3. **Convergence Plot**: Training curves for representative configs
4. **Scaling Laws**: Log-log plot showing power-law fits

### 4.3 Statistical Analysis

**Significance Testing**:
- Paired t-test for configs with similar parameter counts
- Example: Compare Hybrid (62K) vs LoRA (78K) vs Adapter (66K)

**Effect Size**:
- Report Cohen's d for meaningful differences
- Example: "Hybrid achieves 0.5 std better NMSE than LoRA (d=0.8, p<0.01)"

**Confidence Intervals**:
- Bootstrap 95% CI for each Pareto point (using checkpoints as pseudo-replicates)

---

## 5. Implementation Details

### 5.1 File Structure

**Configuration Files**:
```
config/
├── config_pareto_adapter.yaml    # 4 adapter configs
├── config_pareto_lora.yaml       # 4 LoRA configs
├── config_pareto_prompt.yaml     # 3 Prompt configs
└── config_pareto_hybrid.yaml     # 3 Hybrid configs
```

**Execution Scripts**:
```
Transfer_Pareto_Adapter.py   # 20 runs (4×5)
Transfer_Pareto_LoRA.py      # 20 runs (4×5)
Transfer_Pareto_Prompt.py    # 15 runs (3×5)
Transfer_Pareto_Hybrid.py    # 15 runs (3×5)
```

**Output Structure**:
```
saved_model/pareto/
├── Large_estimator_v3_to_InH_adapter_dim8.pt
├── Large_estimator_v3_to_InH_adapter_dim8_iter_20000.pt
├── ... (70 final models + 350 checkpoints = 420 files)
```

### 5.2 Execution Order (Recommended)

**Phase 1: Quick Methods First** (Get early results)
1. ✅ Prompt (15 runs, ~30 hours)
2. ✅ LoRA (20 runs, ~40 hours)
3. ✅ Hybrid (15 runs, ~30 hours)

**Phase 2: Slow Methods** (Run in parallel if GPUs available)
4. ✅ Adapter (20 runs, ~40 hours)

**Rationale**:
- Early results from fast methods allow preliminary analysis
- Can adjust strategy if unexpected patterns emerge
- Can start writing paper sections while Adapter runs

### 5.3 Monitoring and Checkpointing

**WandB Projects** (separate for each method):
```
DNN_channel_estimation_InH_Adapter_Pareto
DNN_channel_estimation_InH_LoRA_Pareto
DNN_channel_estimation_InH_Prompt_Pareto
DNN_channel_estimation_InH_Hybrid_Pareto
... (20 projects total)
```

**Key Logged Metrics**:
- `ch_nmse`: Primary performance metric
- `ch_loss`: Training objective
- `learning_rate`: LR schedule verification
- `trainable_params`: Confirm parameter count
- `iteration`: Training progress

**Resume Capability**:
- Each script saves checkpoints every 20K
- Can restart from last checkpoint if interrupted
- Independent runs allow parallel execution on multiple GPUs

---

## 6. Analysis Plan

### 6.1 Data Collection

After all runs complete, collect into structured format:

```python
results_df = pd.DataFrame({
    'method': [...],
    'config': [...],
    'scenario': [...],
    'params': [...],
    'nmse': [...],
    'mse': [...],
    'convergence_iter': [...]
})
```

### 6.2 Visualization Script

Create `plot_pareto_curves.py`:

```python
# 1. Pareto Frontier (5 plots, 1 per scenario)
for scenario in scenarios:
    plt.figure()
    for method in methods:
        plot_curve(params, nmse, method, scenario)
    plt.xlabel('Trainable Parameters')
    plt.ylabel('NMSE (dB)')
    plt.legend()

# 2. Normalized Efficiency Heatmap
efficiency = nmse / params  # Lower is better
sns.heatmap(pivot_table(scenario, method, efficiency))

# 3. Scaling Laws (log-log)
for method in methods:
    fit_power_law(log(params), log(nmse))
    plot_fit_and_data()
```

### 6.3 Paper Sections

**Section IV: Experimental Design**
- Subsection: Pareto Frontier Methodology
- Table 3: Configuration Matrix
- Justification for parameter ranges

**Section V: Results**
- Subsection A: Pareto Curves by Scenario
  - Figure 5: InH/InF Pareto Frontiers
  - Figure 6: UMi/UMa/RMa Pareto Frontiers
- Subsection B: Parameter Efficiency Analysis
  - Table 4: Best Configuration per Budget
  - Figure 7: Efficiency Heatmap
- Subsection C: Hybrid Synergy
  - Figure 8: Iso-parameter comparison
  - Statistical test: Hybrid vs single methods at matched params

**Section VI: Discussion**
- Optimal operating points for deployment scenarios
- Parameter budget recommendations
- Scaling law interpretation

---

## 7. Risk Mitigation

### 7.1 Potential Issues

**Issue 1: Non-convergence at extreme configs**
- Example: Adapter dim=8 may be too small, Prompt len=200 may be too large
- **Mitigation**: Monitor first few scenarios, adjust ranges if needed

**Issue 2: Inconsistent trends across scenarios**
- Example: Prompt best for InH but worst for RMa
- **Mitigation**: This is actually interesting! Report scenario-specific recommendations

**Issue 3: Hybrid shows no synergy**
- Example: Hybrid(50,5) ≈ average of Prompt(50) and LoRA(5)
- **Mitigation**: Honest reporting, discuss when hybrid NOT beneficial (still a contribution)

**Issue 4: Time/resource constraints**
- Example: Lab GPU goes down mid-experiment
- **Mitigation**:
  - Independent runs can be executed separately
  - Prioritize 1-2 scenarios first, extend to all 5 if promising
  - Can present partial results (3/5 scenarios) if deadline approaches

### 7.2 Contingency Plans

**Plan A (Ideal)**: Complete all 70 runs → Full Pareto analysis

**Plan B (Time-constrained)**: Complete 2-3 scenarios → Partial Pareto, extrapolate

**Plan C (Minimal)**: Complete 1 scenario (RMa) with all configs → Case study analysis

**Plan D (Ultimate fallback)**: Use existing results + 1-2 new controlled experiments

---

## 8. Timeline

### 8.1 Execution Schedule

**Week 1**: Prompt + LoRA (35 runs, ~70 hours)
- Day 1-3: Prompt (15 runs)
- Day 4-6: LoRA (20 runs)
- Day 7: Analysis of preliminary results

**Week 2**: Hybrid + Adapter (35 runs, ~70 hours)
- Day 1-3: Hybrid (15 runs)
- Day 4-7: Adapter (20 runs)

**Week 3**: Analysis and Writing
- Day 1-2: Generate all Pareto plots
- Day 3-4: Statistical analysis
- Day 5-7: Write results section

### 8.2 Milestones

- [ ] **M1**: Complete Prompt experiments (first insights into parameter efficiency)
- [ ] **M2**: Complete LoRA experiments (establish mid-range baseline)
- [ ] **M3**: Complete Hybrid experiments (test synergy hypothesis)
- [ ] **M4**: Complete Adapter experiments (fill high-parameter regime)
- [ ] **M5**: Generate all Pareto curves
- [ ] **M6**: Draft results section with figures
- [ ] **M7**: Complete discussion and conclusions

---

## 9. Success Criteria

**Minimum Success**:
- Pareto curves show clear differentiation between methods
- At least one method shows strong parameter efficiency (NMSE < threshold with <50K params)
- Statistical significance for key comparisons

**Target Success**:
- Hybrid demonstrates synergy (outperforms singles at matched budgets)
- Clear recommendations for parameter budgets (e.g., "use Prompt for <20K, LoRA for 20-100K, Adapter for >100K")
- Scaling laws fit well (R²>0.9) for all methods

**Stretch Success**:
- Hybrid grid exploration reveals optimal Prompt:LoRA ratio
- Generalization analysis across scenarios (identify when each method excels)
- Theoretical explanation for observed scaling behaviors

---

## 10. Open Questions (Future Work)

### 10.1 Hybrid Grid Exploration

After diagonal results, explore full 3×3 grid to answer:
- What is the optimal Prompt:LoRA ratio?
- Does the optimal ratio depend on total parameter budget?
- Are Prompt and LoRA contributions independent (additive) or multiplicative?

### 10.2 Multi-objective Optimization

Beyond NMSE, consider:
- **Inference Latency**: Does Prompt add overhead? Does LoRA?
- **Memory Footprint**: Runtime memory vs parameter count
- **Adaptation Speed**: Iterations to convergence (important for online learning)

### 10.3 Scaling Beyond Tested Range

- **Ultra-low regime**: Prompt len=10-20, LoRA r=1-2
- **High regime**: Adapter dim=128, LoRA r=64
- Do power laws extrapolate? Where do they break?

### 10.4 Scenario-Specific Tuning

- Can we **learn** optimal configs per scenario?
- Meta-learning: Predict best (method, config) from scenario features

---

## 11. Conclusion

This Pareto Frontier design provides:
1. ✅ **Fair comparison** across methods despite parameter differences
2. ✅ **Scientific rigor** through systematic parameter scaling
3. ✅ **Practical insights** for deployment (which method for which budget?)
4. ✅ **Extensibility** to future work (hybrid grid, multi-objective)
5. ✅ **Publication strength** (addresses reviewer concerns about fairness)

The **70 experimental runs** represent a comprehensive investigation that will provide definitive answers about PEFT method selection for channel estimation tasks.

---

**Document Status**: ✅ Complete and Ready for Execution
**Next Step**: Begin execution with `python Transfer_Pareto_Prompt.py`
**Contact**: See project README for questions or issues
