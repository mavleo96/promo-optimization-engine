# Promotion Optimization Engine
### AB InBev Global Analytics Hackathon: Team - Simpson's Paradox

This project implements a hierarchical regression model for promotion optimization, developed during the **AB InBev Global Analytics Hackathon**. The solution focuses on optimizing promotional spend allocation across a multi-level product hierarchy (brand, price segment, pack type, size, and SKU) while respecting business constraints. **Our team placed 7th globally out of 150+ teams in the competition.**

---

## Model Architecture

The model uses a hierarchical parameter structure where each parameter consists of two components:
1. A global parameter $w$ that represents the shared behavior across all SKUs
2. A hierarchical parameter $w^h$ that captures SKU-specific deviations from the global

This structure is implemented through the `BaseModuleClass`, which manages both global and hierarchical parameters. The `ActivatedParameter` class wraps PyTorch parameters with configurable activation functions (`tanh`, `sigmoid`, or `relu`).

We enforce the hierarchical structure through regularization (instead of KL divergence) to ensure SKU-level parameters stay close to the global distribution:

$$\mathcal{L}_{\text{reg}} = \lambda_h \sum_k (w_k^h)^2 + \lambda_r \sum_k w_k^2$$

where $\lambda_h$ penalizes SKU-level deviation and $\lambda_r$ is standard L2 regularization on global weights.

---

## Layers

**Baseline:** Predicts base sales from time trend and lagged sales with hierarchical seasonality effects per SKU

$$\hat{b} = \sigma(w_0) + \bigl(\tanh(w_1) + \tanh(w_1^h)\bigr) \alpha t + \bigl(\tanh(w_2) + \tanh(w_2^h)\bigr) y_{t-1}$$

**Mixed Effect:** Computes a macroeconomic scaling factor $m$ and an ROI multiplier $r$ that scales promotional uplift based on macro conditions

$$m = 1 + \tanh\left((w_m + w_m^h)^\top x_{\text{macro}}\right)$$

$$r = 1 + \tanh\left((w_r + w_r^h) \cdot m\right)$$

**Discount:** Computes promotional uplift $u$ as a hierarchically-weighted linear function of discount spend $d$

$$u = \bigl(\sigma(w_d) + \sigma(w_d^h)\bigr) \cdot d$$

**Volume Conversion:** Converts sales predictions to volume using independent conversion parameters

$$v = \sigma(w_v) \hat{y} + \tanh(w_i)$$

---

## Final Prediction

The final sales prediction combines all components:

$$\hat{y} = \hat{b} \cdot \prod_j m_j + \sum_j u_j \cdot \prod_j r_j$$

---

## Optimization Engine

Given a trained model, the optimizer backpropagates through the frozen network to find the discount allocation $d^*$ that minimizes:

$$\mathcal{L}_{\text{opt}} = -\Delta_{\text{NR}} - \lambda_\rho \cdot \rho + \sum_c \lambda_c \mathcal{L}_c$$

where $\Delta_{\text{NR}}$ is the increase in net revenue over baseline, and $\rho$ is defined as:

$$\rho = \frac{\sum \hat{y}_{\text{opt}} - \sum \hat{y}_{\text{init}}}{\sum d + \epsilon}$$

Penalty terms $\mathcal{L}_c$ enforce the following constraints: brand-level discount caps, pack-type limits, price segment bounds, volume variation limits, and non-negativity of discounts.
