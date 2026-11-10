# <span style="font-size: 20px;">Regression Metrics</span>

## <span style="font-size: 18px;">What Are Regression Metrics?</span>

<span style="font-size: 14px;">Regression metrics quantify how well a model's predicted values match the true values in a continuous output setting. Three metrics are fundamental: MSE, MAE, and R².</span>

## <span style="font-size: 18px;">Mean Squared Error (MSE)</span>

<span style="font-size: 14px;">MSE averages the squared differences between predictions and true values:</span>

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

<span style="font-size: 14px;">Key properties:</span>

* <span style="font-size: 14px;">Always non-negative; equals zero only for perfect predictions</span>
* <span style="font-size: 14px;">Squaring amplifies large errors, making MSE sensitive to outliers</span>
* <span style="font-size: 14px;">Units are squared relative to the target variable</span>
* <span style="font-size: 14px;">Differentiable everywhere, useful for gradient-based optimization</span>

## <span style="font-size: 18px;">Mean Absolute Error (MAE)</span>

<span style="font-size: 14px;">MAE averages the absolute differences between predictions and true values:</span>

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

<span style="font-size: 14px;">Key properties:</span>

* <span style="font-size: 14px;">Always non-negative; equals zero only for perfect predictions</span>
* <span style="font-size: 14px;">Linear penalty makes it more robust to outliers than MSE</span>
* <span style="font-size: 14px;">Units match the target variable directly</span>
* <span style="font-size: 14px;">Not differentiable at zero, but subgradients exist</span>

## <span style="font-size: 18px;">Coefficient of Determination (R²)</span>

<span style="font-size: 14px;">R² measures the fraction of variance in the true values explained by the model:</span>

$$
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
$$

<span style="font-size: 14px;">where the residual sum of squares and total sum of squares are:</span>

$$
\text{SS}_{\text{res}} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

$$
\text{SS}_{\text{tot}} = \sum_{i=1}^{n}(y_i - \bar{y})^2
$$

<span style="font-size: 14px;">Key properties:</span>

* <span style="font-size: 14px;">R² = 1.0: perfect predictions, model explains all variance</span>
* <span style="font-size: 14px;">R² = 0.0: model does no better than predicting the mean every time</span>
* <span style="font-size: 14px;">R² < 0: model is worse than the mean baseline (can occur with bad models)</span>
* <span style="font-size: 14px;">When all true values are identical, SS_tot = 0 and R² is undefined; by convention return 0.0</span>

## <span style="font-size: 18px;">Choosing Between Metrics</span>

* <span style="font-size: 14px;">Use MSE when large errors should be penalized more heavily (e.g., safety-critical predictions)</span>
* <span style="font-size: 14px;">Use MAE when outliers should not dominate (e.g., noisy real-world data)</span>
* <span style="font-size: 14px;">Use R² to communicate model quality in a scale-free, interpretable way</span>
* <span style="font-size: 14px;">Reporting all three gives a complete picture of model performance</span>