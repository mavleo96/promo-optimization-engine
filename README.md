# Promotion Optimization Engine
### ABInBev Global Analytics Hackathon: Team - Simpson's Paradox

This project implements a hierarchical regression model for promotion optimization, developed during the **ABInBev Global Analytics Hackathon**. The solution focuses on optimizing promotional spend allocation across a multi-level product hierarchy (brand, price segment, pack type, size, and SKU) while respecting business constraints. **Our team placed 7th globally out of 150+ teams in the competition.**

## Model Architecture

The model uses a hierarchical parameter structure where each parameter consists of two components:
1. A global parameter that represents the shared behavior across all SKUs
2. A hierarchical parameter that captures specific variations for each SKU

This structure is implemented through the `BaseModuleClass`, which manages both global and hierarchical parameters. The `ActivatedParameter` class is a wrapper around PyTorch parameters that enables dynamic application of activation functions (tanh, sigmoid, or relu) to the parameters.

We enforce the hierarchical structure through hierarchical regularization (instead of KL divergence) to ensure parameters for each SKU come from the same distribution:

```math
L_{\text{hier}} = \lambda_h \sum(w^h)^2 + \lambda_r \sum w^2
```

where:
- $w^h$ are hierarchical weights
- $w$ are global weights
- $\lambda_h$ is the hierarchical regularization weight
- $\lambda_r$ is the L2 regularization weight

### Model Components

1. **Baseline Layer** - Predicts base sales using time and lagged sales with hierarchical trend and seasonality effects
   ```math
   \text{baseline} = \sigma(\text{intercept}) + (\tanh(w_1) + \tanh(w_1^h)) \cdot \alpha t + (\tanh(w_2) + \tanh(w_2^h)) \cdot \text{sales\_lag}
   ```

2. **Mixed Effect Layer** - Models macroeconomic impacts and ROI multipliers with hierarchical sensitivity
   ```math
   \text{mixed\_effect} = 1 + \tanh((w_{me} + w_{me}^h) \cdot \text{macro})
   ```
   ```math
   \text{roi\_mult} = 1 + \tanh((w_{roi} + w_{roi}^h) \cdot \text{mixed\_effect})
   ```

3. **Discount Layer** - Models linear impact of promotional spend with hierarchical sensitivity to different discount types
   ```math
   \text{uplift} = (\sigma(w_d) + \sigma(w_d^h)) \cdot \text{discount}
   ```

4. **Volume Conversion Layer** - Converts sales predictions to volume using independent conversion parameters
   ```math
   \text{volume} = \sigma(w_v) \cdot \text{sales} + \tanh(w_i)
   ```

### Final Prediction

The final sales prediction combines all components:
```math
\text{sales\_pred} = \text{baseline} \cdot \prod(\text{mixed\_effect}) + \sum(\text{uplift}) \cdot \prod(\text{roi\_mult})
```

### Optimization Engine

The optimization engine maximizes ROI while respecting business constraints:

```math
L_{\text{opt}} = -(\text{nr\_increase}) - \lambda_{roi} \cdot \text{roi} + \sum \lambda_c \cdot L_c
```

where:
- $\text{nr\_increase}$ is the increase in net revenue
- $\text{roi}$ is the return on investment
- $L_c$ are constraint losses for:
  - Brand-level discount limits
  - Pack-type discount limits
  - Price segment constraints
  - Volume variation bounds
  - Non-negative discounts

The engine uses gradient-based optimization to find the optimal promotional spend allocation that maximizes ROI while satisfying all business constraints.
