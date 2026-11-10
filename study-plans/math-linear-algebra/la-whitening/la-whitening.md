## Whitening

<span style="font-size: 14px;">Whitening (or sphering) transforms data so that its covariance matrix is the identity. This means each feature has unit variance and all features are uncorrelated.</span>

---

## Algorithm: PCA Whitening

<span style="font-size: 14px;">1. **Center**: $\tilde{X} = X - \bar{X}$</span>

<span style="font-size: 14px;">2. **Covariance**: $\Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X}$</span>

<span style="font-size: 14px;">3. **Eigendecompose**: $\Sigma = V D V^T$ where $D = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)$</span>

<span style="font-size: 14px;">4. **Whiten**: $X_w = \tilde{X} V D^{-1/2}$</span>

<span style="font-size: 14px;">The key insight: projecting onto eigenvectors ($\tilde{X} V$) decorrelates the data, and dividing by $\sqrt{\lambda_i}$ normalizes each component to unit variance. The whitening matrix is $W = V D^{-1/2}$, so the transformation is $X_w = \tilde{X} W$.</span>

---

## Verification

<span style="font-size: 14px;">The sample covariance of the whitened data is:</span>

$$
\frac{1}{n-1} X_w^T X_w = D^{-1/2} V^T \Sigma V D^{-1/2} = D^{-1/2} D D^{-1/2} = I
$$

---

## PCA Whitening vs ZCA Whitening

<span style="font-size: 14px;">**PCA whitening**: $X_w = \tilde{X} V D^{-1/2}$ - data is rotated to PCA coordinates and scaled. The whitened data lives in a different coordinate frame from the original.</span>

<span style="font-size: 14px;">**ZCA whitening**: $X_w = \tilde{X} V D^{-1/2} V^T$ - an extra rotation back to the original coordinate frame by multiplying with $V^T$. The ZCA whitening matrix is $W_{ZCA} = V D^{-1/2} V^T = \Sigma^{-1/2}$, which is the symmetric matrix square root of $\Sigma^{-1}$.</span>

<span style="font-size: 14px;">The key difference: ZCA whitened data stays close to the original data. Formally, ZCA minimizes $\|X_w - \tilde{X}\|_F$ among all whitening transforms. This makes ZCA preferred when spatial structure matters - for example, whitened images under ZCA still look like images, while PCA whitening scrambles the pixel ordering.</span>

---

## Connection to Decorrelation

<span style="font-size: 14px;">Whitening is a two-step process: first decorrelate, then normalize. The decorrelation step ($\tilde{X} V$) rotates data to the principal axes where the covariance is diagonal. The normalization step (dividing by $\sqrt{\lambda_i}$) rescales each axis to unit variance. Decorrelation alone (without normalization) is sometimes sufficient - for example, when you want uncorrelated features but want to preserve the relative magnitudes of variance.</span>

---

## Handling Zero Eigenvalues

<span style="font-size: 14px;">If the data lies in a lower-dimensional subspace, some eigenvalues will be zero. We cannot divide by zero, so eigenvalues below a threshold $\epsilon$ are treated as zero and their corresponding dimensions are left unscaled (or zeroed out). This is equivalent to projecting onto the non-degenerate subspace first.</span>

---

## Why Whitening Helps Training

<span style="font-size: 14px;">Neural networks converge faster with whitened inputs because the loss landscape becomes more isotropic. Without whitening, the Hessian of the loss can have a large condition number $\kappa = \lambda_{\max}/\lambda_{\min}$, creating elongated valleys where gradient descent oscillates. Whitening makes the Hessian closer to a scalar multiple of identity, so gradient descent moves directly toward the minimum. The convergence rate of gradient descent is bounded by $(\kappa - 1)/(\kappa + 1)$ per step, so reducing $\kappa$ directly accelerates training.</span>

---

## Applications in ML

* <span style="font-size: 14px;">**Preprocessing for ICA**: Independent Component Analysis requires whitened input. Whitening reduces the problem from learning a general mixing matrix to learning an orthogonal one, cutting the number of parameters from $d^2$ to $d(d-1)/2$.</span>
* <span style="font-size: 14px;">**Neural network training**: whitened inputs can accelerate convergence because the loss surface becomes more isotropic, avoiding elongated valleys that slow gradient descent.</span>
* <span style="font-size: 14px;">**Batch normalization**: while not full whitening, batch norm normalizes each feature independently per mini-batch, approximating the diagonal part of whitening. It was inspired by the idea that internal covariate shift slows training; whitening each layer's inputs addresses this. Full whitening across features (decorrelated batch normalization) has been proposed but is more expensive.</span>
* <span style="font-size: 14px;">**Natural gradient**: the natural gradient method whitens the gradient using the Fisher information matrix, adapting to the local geometry of parameter space. This is equivalent to preconditioning with the inverse of the expected Hessian.</span>