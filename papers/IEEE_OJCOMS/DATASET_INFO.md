# Dataset Information - IEEE OJCOMS Paper

## PDP Dataset Structure

### Available Files
Located in: `dataset/PDP_processed/`

```
PDP_InH_Los_10000.mat     # Indoor Hotspot - Line of Sight
PDP_InH_Nlos_10000.mat    # Indoor Hotspot - Non Line of Sight
PDP_InF_Los_10000.mat     # Indoor Factory - LoS
PDP_InF_Nlos_10000.mat    # Indoor Factory - NLoS
PDP_UMi_Los_10000.mat     # Urban Micro - LoS
PDP_UMi_Nlos_10000.mat    # Urban Micro - NLoS
PDP_RMa_Los_10000.mat     # Rural Macro - LoS
PDP_RMa_Nlos_10000.mat    # Rural Macro - NLoS
PDP_UMa_Los_10000.mat     # Urban Macro - LoS
PDP_UMa_Nlos_10000.mat    # Urban Macro - NLoS
```

**Total**: 10 channel condition files × 10,000 samples each = 100,000 PDP samples

### Scenario Characteristics

#### 1. InH (Indoor Hotspot)
- **Environment**: Dense indoor spaces (shopping malls, airports, train stations)
- **Frequency**: 28 GHz (mmWave)
- **Distance Range**: 5-100m
- **Key Features**:
  - High user density
  - Rich scattering environment
  - Short propagation distances

#### 2. InF (Indoor Factory)
- **Environment**: Industrial facilities, large warehouses
- **Frequency**: 28 GHz (mmWave)
- **Distance Range**: 10-100m (realistic indoor range)
- **Key Features**:
  - Metal structures causing multipath
  - Moderate to large coverage area
  - Industrial IoT applications

#### 3. UMi (Urban Micro)
- **Environment**: Small cell deployments in urban areas
- **Frequency**: 28 GHz (mmWave)
- **Distance Range**: 10-500m
- **Key Features**:
  - Street-level coverage
  - Building reflections
  - Pedestrian-level communications

#### 4. RMa (Rural Macro)
- **Environment**: Rural areas, highways, farmland
- **Frequency**: 28 GHz (mmWave)
- **Distance Range**: 10-10,000m (extended range)
- **Key Features**:
  - Long-distance propagation
  - Limited obstacles
  - Sparse deployment
  - Extended coverage requirement

#### 5. UMa (Urban Macro)
- **Environment**: Urban macro cell deployments
- **Frequency**: 28 GHz (mmWave)
- **Distance Range**: 10-10,000m (extended range)
- **Key Features**:
  - High-rise buildings
  - Complex propagation environment
  - Wide area coverage

### Distance Range Justification

| Scenario | Range | Justification |
|----------|-------|---------------|
| InH | 5-100m | Indoor spaces have physical boundaries; typical coverage 50-100m |
| InF | 10-100m | Factory floors typically 50-200m; realistic indoor factory coverage |
| UMi | 10-500m | Street-level small cells; typical ISD (Inter-Site Distance) 200-500m |
| RMa | 10-10,000m | Rural macro cells require long-range coverage; ISD up to 5-10km |
| UMa | 10-10,000m | Urban macro cells need wide coverage; typical ISD 200m-2km, max up to 5km |

**Key Innovation**: Unlike previous approaches using uniform distance ranges, we implement **scenario-specific distance ranges** that reflect real-world deployment constraints.

### PDP Generation

**Simulator**: NYUSim (New York University Wireless Simulator)
- Industry-standard channel simulator
- Validated against measurement campaigns
- Supports 3GPP channel models

**Configuration**:
- **Carrier Frequency**: 28 GHz (5G mmWave)
- **Bandwidth**: 400 MHz
- **Environment**: Scenario-specific settings
- **Extended Distance**: RMa and UMa configured for up to 10km propagation

### Data Usage

#### Base Model Training
- **All scenarios**: InH, InF, UMi, RMa, UMa (Los + Nlos)
- **Purpose**: Learn general channel estimation capabilities across diverse environments
- **Benefit**: Universal feature extraction

#### Transfer Learning
- **Target scenarios**: InF (Indoor) and RMa (Outdoor)
- **Purpose**: Adapt base model to specific environments using PEFT methods
- **Datasets**:
  - InF: `InF_Los_10000 + InF_Nlos_10000` (20,000 samples)
  - RMa: `RMa_Los_10000 + RMa_Nlos_10000` (20,000 samples)
- **PEFT Methods**:
  - **Adapter v3**: 524,288 trainable params, 60K iterations
  - **LoRA v4**: 98,304 trainable params, 60K iterations
  - **Prompt v4**: 12,800 trainable params, 150K iterations (requires longer training)

### Technical Implementation

**Scenario-Specific Distance Sampling** (in `dataset.py`):

```python
# Each PDP is tagged with its scenario (e.g., "InH_Los", "RMa_Nlos")
# During training, distance is sampled from scenario-specific range:

for pdp_idx in batch:
    scenario = pdp_idx_to_scenario[pdp_idx]  # e.g., "InH_Los"
    dist_range = distance_ranges[scenario]    # e.g., [5.0, 100.0]
    distance = uniform(dist_range[0], dist_range[1])

    # Combine PDP characteristics + appropriate distance
    channel = generate_channel(pdp, distance)
```

**Benefits**:
1. **Physical Realism**: InH PDP never paired with 5km distance
2. **Better Learning**: Model learns valid scenario-distance relationships
3. **Improved Generalization**: Training distribution matches deployment reality

### Comparison to ICTC 2025 Paper

| Aspect | ICTC 2025 | IEEE OJCOMS (This Work) |
|--------|-----------|-------------------------|
| Scenarios | 3 (InF, RMa, UMa) | 5 (+ InH, UMi) |
| Distance Handling | Global range | Scenario-specific ranges |
| Transfer Methods | Adapter vs LoRA | Adapter vs LoRA vs Prompt |
| Dataset Size | Mixed | Uniform 10k per condition |
| Distance Ranges | Not optimized | Physically realistic |
| Parameter Efficiency | 5.24% ~ 7.18% | 0.94% ~ 7.18% |

**Key Enhancement**: Addition of Prompt Learning (0.94% trainable params) enables exploration of extreme parameter efficiency, 7.7× more efficient than LoRA while maintaining competitive performance.

---

## Dataset Usage for Prompt Learning

### Why Prompt Learning Requires More Training Data Iterations?

Unlike Adapter and LoRA which modify the model's internal processing, Prompt Learning only adds learnable input tokens:

**Training Dynamics Comparison**:
```
Adapter (524K params):
- Modifies each layer's processing
- Rich parameter space for learning
- Converges in ~40K iterations

LoRA (98K params):
- Adds low-rank matrices to key projections
- Moderate parameter space
- Converges in ~40K iterations

Prompt (12.8K params):
- Only modifies input context
- Limited parameter space
- Requires ~100K iterations for convergence
→ Solution: Train for 150K iterations to ensure full convergence
```

**Dataset Efficiency**:
```python
# Same 20,000 training samples used for all methods
# Iterations required vary based on parameter count

Adapter: 60K iter × 32 batch = 1.92M training instances
LoRA:    60K iter × 32 batch = 1.92M training instances
Prompt:  150K iter × 32 batch = 4.8M training instances

# Prompt sees 2.5× more data examples to compensate for fewer params
# This is computationally acceptable (still converges in ~4 hours)
```

**Practical Benefit**:
- Same dataset (no additional data collection needed)
- Only requires longer training time (~2h extra)
- Achieves competitive performance with 7.7× fewer parameters

---

**Summary**: This dataset provides comprehensive coverage of 5G/6G deployment scenarios with physically realistic distance ranges, enabling fair comparison of PEFT methods across diverse wireless environments. The uniform 10k-sample-per-condition structure ensures all three PEFT methods (Adapter, LoRA, Prompt) train on identical data distributions, with only iteration count varying based on parameter efficiency.