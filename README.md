# Startup Health Scoring Model

## Project Overview

This is a comprehensive machine learning project that develops a sophisticated startup evaluation and ranking system. The project implements multiple approaches to assess startup potential using various ML algorithms and neural networks, ultimately creating a composite scoring mechanism to rank startups based on key performance indicators.

## Project Structure

```
├── Startup_Scoring_Dataset.csv          # Main dataset with startup metrics
├── notebooks/                           # Jupyter notebooks for different approaches
│   ├── ML_model.ipynb                  # Traditional ML models and baseline scoring
│   ├── finetune.ipynb                  # Hyperparameter optimization using Optuna
│   ├── neural_net_pytorch.ipynb       # PyTorch neural network implementation
│   ├── neural_net_tensorflow.ipynb    # TensorFlow neural network implementation
└── output/                             # Generated visualizations and results
    ├── composite_scores_bar_chart.png
    ├── composite_score_distribution.png
    ├── correlation_heatmap.png
    └── feature_importances_*.png
├── requirements.txt                    # Python package dependencies
└── README.md                           # Project documentation
```

## Dataset Description

The dataset contains 101 startups with the following key metrics:

| Feature | Description | Range/Type |
|---------|-------------|------------|
| `startup_id` | Unique identifier | S001-S101 |
| `team_experience` | Years of team experience | 1-10 years |
| `market_size_million_usd` | Total addressable market | $14M-$996M |
| `monthly_active_users` | User engagement metric | 954-98,606 users |
| `monthly_burn_rate_inr` | Monthly cash expenditure | ₹1.66L-₹99L |
| `funds_raised_inr` | Total funding raised | ₹1.1L-₹4.96Cr |
| `valuation_inr` | Current company valuation | ₹1Cr-₹49Cr |

## Methodology

### 1. Data Preprocessing (`ML_model.ipynb`)

#### Handling Negatively Correlated Metrics
One of the key challenges was handling the `monthly_burn_rate_inr` feature, which is negatively correlated with startup success (higher burn rate = worse performance). 

**Solution Implemented:**
```python
df['monthly_burn_rate_inr'] = df['monthly_burn_rate_inr'].max() - df['monthly_burn_rate_inr']
```

**Rationale:** This transformation inverts the burn rate so that:
- High burn rate (bad) → Low transformed value
- Low burn rate (good) → High transformed value

This ensures all features are positively correlated with startup success, making the composite scoring more intuitive and mathematically sound.

#### Normalization Strategy
Applied MinMaxScaler to normalize all features to [0,1] range:
```python
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
```

This prevents features with larger numerical ranges (like valuation) from dominating the composite score.

### 2. Composite Scoring System

#### Initial Weight Selection Rationale

The baseline weights were carefully chosen based on startup evaluation best practices:

```python
weights = {
    'team_experience': 15,          # 15% - Foundational but not everything
    'market_size_million_usd': 20,  # 20% - Large markets enable scale
    'monthly_active_users': 25,     # 25% - Highest weight for user traction
    'monthly_burn_rate_inr': 10,    # 10% - Efficiency matters but less critical
    'funds_raised_inr': 15,         # 15% - Capital access is important
    'valuation_inr': 15            # 15% - Market perception indicator
}
```

**Why These Weights?**

1. **Monthly Active Users (25%)**: Highest weight because user traction is the strongest predictor of startup success. It demonstrates product-market fit and growth potential.

2. **Market Size (20%)**: Second highest because large markets provide the ceiling for potential returns. A great product in a small market has limited upside.

3. **Team Experience, Funds Raised, Valuation (15% each)**: Equal importance as supporting factors that enable execution, sustainability, and market confidence.

4. **Monthly Burn Rate (10%)**: Lowest weight because while efficiency matters, growth often requires higher spending in early stages.

### 3. Model Implementations

#### Traditional ML Models (`ML_model.ipynb`)
Implemented four different algorithms to predict composite scores:
- **Random Forest**: Ensemble method for robust predictions
- **XGBoost**: Gradient boosting for high performance
- **Gradient Boosting**: Traditional boosting approach

#### Neural Networks
- **PyTorch Implementation** (`neural_net_pytorch.ipynb`): Custom 3-layer neural network
- **TensorFlow Implementation** (`neural_net_tensorflow.ipynb`): Similar architecture with different framework

#### Hyperparameter Optimization (`finetune.ipynb`)
Used Optuna framework to simultaneously optimize:
- **Feature weights**: Dynamic weight assignment for composite scoring
- **Model hyperparameters**: Learning rate, depth, regularization, etc.

**Key Innovation**: The optimization process treats feature weights as hyperparameters, allowing the model to learn optimal importance values rather than using fixed weights.

## Key Findings and Insights

### 1. Top Performing Startup Analysis

**Highest Ranked: S006 (Score: ~81.03)**
- **Team Experience**: 10/10 (Exceptional leadership)
- **Market Size**: $416M (Large addressable market)
- **Monthly Active Users**: 95,939 (Strong user traction)
- **Low Burn Rate**: Efficient capital utilization
- **High Valuation**: ₹30Cr (Market confidence)
- **Significant Funding**: ₹4.8Cr (Well-capitalized)

**Analysis**: S006 represents the "ideal startup" with balanced excellence across all metrics. Strong team + large market + proven traction + efficient operations = high success probability.

### 2. Lowest Performing Startup Analysis

**Lowest Ranked: S055 (Score: ~19.09)**
- **Team Experience**: Below average
- **Market Size**: Small addressable market
- **Monthly Active Users**: Low user traction
- **High Burn Rate**: Inefficient spending
- **Low Funding**: Limited capital access
- **Low Valuation**: Market skepticism

**Analysis**: S055 demonstrates the compound effect of poor fundamentals across multiple dimensions.

### 3. Feature Importance Insights

Across all models, the consistent ranking emerged:

1. **Monthly Active Users**: Universally most important feature
2. **Market Size**: Second most critical factor
3. **Valuation**: Strong predictor of success
4. **Team Experience**: Moderately important
5. **Funds Raised**: Supporting factor
6. **Burn Rate**: Least predictive (after transformation)

## Hyperparameter Optimization Results

The Optuna optimization revealed optimal feature weights that differed from initial assumptions:

```python
# Optimized average weights across models
optimized_weights = {
    'monthly_active_users': 0.320,     # 32.0% - Highest predictor
    'funds_raised_inr': 0.221,         # 22.1% - Second highest
    'market_size_million_usd': 0.150,  # 15.0% - Third highest
    'valuation_inr': 0.112,            # 11.2% - Fourth
    'monthly_burn_rate_inr': 0.105,    # 10.5% - Fifth
    'team_experience': 0.093           # 9.3% - Lowest
}
```

### Key Learnings
- **Monthly Active Users Dominates (32.0%)**: As expected, user traction emerged as the top predictor. This indicates:

  - User engagement is the strongest signal of product-market fit
  - Active user base directly correlates with revenue potential
  - Sustainable growth requires genuine user adoption

- **Funds Raised Second Most Important (22.1%)**: Funding amount ranked second, suggesting:

  - Investor confidence (reflected in funding) is a strong success indicator
  - Well-funded startups have higher survival and growth rates
  - Capital availability enables better execution and market capture

- **Market Size Third Most Important (15.0%)**: Market opportunity ranked third, indicating:

  - Large addressable markets provide growth ceiling
  - Market timing and size does matter 
  - TAM enables scalability and attracts further investment

- **Valuation Moderately Predictive (11.2%)**: Market perception through valuation provides meaningful signal:

  - External validation correlates with internal fundamentals
  - Higher valuations reflect investor confidence in potential
  - Market sentiment can be a leading indicator

- **Burn Rate Efficiency Important (10.5%)**: Capital discipline ranked fifth, showing:

  - Efficient capital utilization correlates with success
  - Lower burn rates provide longer runway for growth
  - The inversion transformation was effective in capturing efficiency

- **Team Experience Least Predictive (9.3%)**: Surprisingly, experienced teams had lowest weight:

  - Execution and results matter more than past experience
  - Fresh perspectives might be valuable in rapidly evolving markets
  - Other factors compensate for experience gaps

#### Model-Specific Insights:

- **XGBoost**: Heavily weighted user traction (32%) and funding (34%)
- **Random Forest**: Emphasized user traction (32%) and funding (28%)
- **GradientBoost**: Prioritized market size (33%) and user traction (31%)

This data-driven approach reveals that **user traction combined with strong funding and market opportunity** are the strongest predictors of startup success, while team experience has surprisingly minimal predictive power.

## Model Performance

### Traditional ML Models
- **Gradient Boosting**: Best overall performance (R² ≈ 0.90+)
- **Random Forest**: Robust baseline performance
- **XGBoost**: Good performance with interpretability

### Neural Networks
- **PyTorch**: R² ≈ 0.95+ with good feature importance extraction
- **TensorFlow**: Second best performance (R² ≈ 0.90) with different implementation approach

### Hyperparameter Optimization
- **Optuna**: Successfully optimized both feature weights and model hyperparameters, achieving improved performance across all models with R² scores exceeding 0.93.

## Future Enhancements

1. **Real-time Data Integration**: Connect to live APIs for dynamic scoring
2. **Industry-specific Weights**: Different weight schemes for different sectors
3. **Temporal Analysis**: Track startup progression over time
4. **External Factors**: Incorporate market conditions, competition analysis
5. **Risk Assessment**: Add probability distributions for uncertainty quantification

## Technical Implementation

### Requirements
```bash
pip install -r requirements.txt
```

### Usage
1. **Basic Scoring**: Run `ML_model.ipynb` for traditional approach
2. **Optimization**: Use `finetune.ipynb` for parameter tuning
3. **Neural Networks**: Experiment with PyTorch/TensorFlow implementations
4. **Visualization**: Check `output/` folder for generated charts

## Conclusion

This project successfully demonstrates that startup success can be quantitatively modeled using fundamental business metrics. However, the hyperparameter optimization revealed that user-centric metrics matter more than traditional wisdom suggests.

**Major Discovery:** The fine-tuning process showed that **monthly active users (32.0%) and funds raised (22.1%)** are the dominant predictors, while **team experience dropped to just 9.3%** and **burn rate efficiency ranked 10.5%** in actual predictive importance.

**Key Revelation:** Optimization revealed that **user traction (32.0%) combined with financial backing (22.1%) and market opportunity (15.0%)** account for 69.1% of predictive power, validating that product-market fit and capital access are fundamental to startup success.

**Most Surprising Finding:** Team experience had minimal predictive power (9.3%), despite initial assumptions of high importance. This suggests that execution, user adoption, and market dynamics matter more than founder background in determining startup outcomes.

The fine-tuned models achieved exceptional performance (R² > 0.98), validating that **sustainable startup success is built on strong user engagement, adequate funding, and significant market opportunity** rather than just experienced leadership. This data-driven framework provides investors, founders, and evaluators with a rigorously tested foundation for making informed decisions in the entrepreneurial ecosystem.

---

