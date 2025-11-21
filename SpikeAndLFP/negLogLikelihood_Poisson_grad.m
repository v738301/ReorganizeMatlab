function [neglogli, grad, H] = negLogLikelihood_Poisson_grad(w, X, y, lambda)
% [neglogli, grad, H] = negLogLikelihood_Poisson_grad(w, X, y, lambda)
%
% =========================================================================
% POISSON GENERALIZED LINEAR MODEL (GLM) - LOSS FUNCTION
% =========================================================================
%
% Computes the negative log-likelihood, gradient, and Hessian for a
% Poisson GLM with exponential link function. This is the core loss
% function used for fitting neural spike train models.
%
% Adapted from Pillow Lab's GLMspiketools:
% https://github.com/pillowlab/GLMspiketools
%
% -------------------------------------------------------------------------
% MATHEMATICAL BACKGROUND: POISSON GLM
% -------------------------------------------------------------------------
%
% The Poisson GLM models spike counts as a function of predictors:
%
% 1. LINEAR PREDICTOR:
%    u(t) = X(t,:) * w = sum_j X(t,j) * w(j)
%
%    where:
%    - X(t,:) is the vector of predictors at time bin t
%    - w is the weight vector (parameters to estimate)
%    - u(t) is the "log firing rate" (in log space)
%
% 2. EXPONENTIAL LINK FUNCTION (nonlinearity):
%    λ(t) = exp(u(t)) = exp(X(t,:) * w)
%
%    where:
%    - λ(t) is the conditional intensity (expected spike count)
%    - exp() ensures λ(t) > 0 (firing rates must be positive)
%
% 3. POISSON LIKELIHOOD:
%    P(y(t) | X(t,:)) = Poisson(y(t); λ(t))
%                     = (λ(t)^y(t) * exp(-λ(t))) / y(t)!
%
%    where:
%    - y(t) is the observed spike count at time t
%    - We assume spike counts are conditionally independent across time
%
% 4. LOG-LIKELIHOOD:
%    For a single time bin:
%    log P(y(t) | X(t,:)) = y(t)*log(λ(t)) - λ(t) - log(y(t)!)
%
%    Substituting λ(t) = exp(X(t,:)*w):
%    log P(y(t) | X(t,:)) = y(t)*X(t,:)*w - exp(X(t,:)*w) - log(y(t)!)
%                           ^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^   ^^^^^^^^^^^
%                           linear term     exp nonlinearity  constant
%
%    For all T time bins (summing log-likelihoods):
%    L(w) = sum_{t=1}^T [y(t)*X(t,:)*w - exp(X(t,:)*w) - log(y(t)!)]
%
%    Dropping the constant term log(y(t)!) (doesn't depend on w):
%    L(w) = sum_{t=1}^T [y(t)*X(t,:)*w - exp(X(t,:)*w)]
%         = y'*X*w - sum(exp(X*w))
%
% 5. NEGATIVE LOG-LIKELIHOOD (what we minimize):
%    NLL(w) = -L(w) = sum(exp(X*w)) - y'*X*w
%
%    In element-wise notation:
%    NLL(w) = sum_{t=1}^T [exp(X(t,:)*w) - y(t)*X(t,:)*w]
%           = sum_{t=1}^T [λ(t) - y(t)*u(t)]
%
% -------------------------------------------------------------------------
% L2 REGULARIZATION (Ridge Penalty)
% -------------------------------------------------------------------------
%
% To prevent overfitting, we add an L2 penalty (ridge regularization):
%
%    NLL_regularized(w) = NLL(w) + (λ/2) * ||w||²
%                       = NLL(w) + (λ/2) * sum(w²)
%
% where:
% - λ (lambda) controls the strength of regularization
% - ||w||² = w'*w is the squared L2 norm of weights
% - Larger λ → stronger regularization → smaller weights
% - λ = 0 → no regularization (maximum likelihood)
%
% Physical interpretation:
% - Ridge penalty shrinks weights toward zero
% - Prevents any single predictor from having too large an effect
% - Bayesian interpretation: Gaussian prior on weights w ~ N(0, 1/λ)
%
% -------------------------------------------------------------------------
% GRADIENT (First Derivative)
% -------------------------------------------------------------------------
%
% To minimize NLL using gradient descent, we need ∂NLL/∂w:
%
% Starting with NLL(w) = sum_t [λ(t) - y(t)*u(t)]
%
% Step 1: Derivative with respect to u(t) = X(t,:)*w
%    ∂NLL/∂u(t) = ∂λ(t)/∂u(t) - y(t)
%               = exp(u(t)) - y(t)
%               = λ(t) - y(t)
%
% Step 2: Chain rule to get derivative with respect to w_j
%    ∂NLL/∂w_j = sum_t [∂NLL/∂u(t) * ∂u(t)/∂w_j]
%              = sum_t [(λ(t) - y(t)) * X(t,j)]
%
% Step 3: Vectorized form
%    ∂NLL/∂w = X' * (λ - y)
%
% where:
% - X' is the transpose of the design matrix [n_predictors × n_bins]
% - (λ - y) is the residual: (predicted rate - observed counts)
% - This is the SAME form as linear regression gradient!
%
% With L2 regularization:
%    ∂NLL_reg/∂w = X' * (λ - y) + λ * w
%
% Intuition:
% - If λ(t) > y(t): predicted rate too high → decrease weights
% - If λ(t) < y(t): predicted rate too low → increase weights
% - X(t,j) determines how much predictor j contributes to the update
%
% -------------------------------------------------------------------------
% HESSIAN (Second Derivative / Curvature)
% -------------------------------------------------------------------------
%
% The Hessian is the matrix of second derivatives ∂²NLL/∂w_i∂w_j
%
% Step 1: Second derivative with respect to u(t)
%    ∂²NLL/∂u(t)² = ∂λ(t)/∂u(t) = exp(u(t)) = λ(t)
%
% Step 2: Chain rule for second derivative
%    ∂²NLL/∂w_i∂w_j = sum_t [λ(t) * X(t,i) * X(t,j)]
%
% Step 3: Vectorized form
%    H = X' * diag(λ) * X
%
% where diag(λ) is a diagonal matrix with λ(t) on the diagonal.
%
% More efficient computation (avoids creating large diagonal matrix):
%    H = (sqrt(λ) .* X)' * (sqrt(λ) .* X)
%
% With L2 regularization:
%    H = X' * diag(λ) * X + λ * I
%
% Interpretation:
% - H is the "Fisher Information Matrix" (expected curvature)
% - H is always positive definite (for Poisson GLM)
% - inv(H) gives the covariance matrix of parameter estimates
% - sqrt(diag(inv(H))) gives standard errors of each weight
%
% Why this matters:
% - Newton's method uses H to find better search directions: Δw = -H^(-1) * grad
% - Uncertainty quantification: SE(w_i) = sqrt(H^(-1)_ii)
% - Statistical testing: z-score = w_i / SE(w_i)
%
% -------------------------------------------------------------------------
% INPUTS
% -------------------------------------------------------------------------
%   w      - [n_predictors × 1] Weight vector (parameters to optimize)
%            Example: [w_event1_basis1; w_event1_basis2; ...; w_speed; w_breathing]
%
%   X      - [n_bins × n_predictors] Design matrix
%            Each row is a time bin, each column is a predictor
%            Example: [event_basis_1, event_basis_2, ..., speed, breathing]
%
%   y      - [n_bins × 1] Observed spike counts
%            Integer counts (0, 1, 2, 3, ...) for each time bin
%
%   lambda - Scalar L2 regularization parameter (optional, default = 0)
%            lambda = 0: Maximum likelihood (no regularization)
%            lambda > 0: Ridge regression (shrinks weights)
%            Typical range: 2^(-5) to 2^15 (selected by cross-validation)
%
% -------------------------------------------------------------------------
% OUTPUTS
% -------------------------------------------------------------------------
%   neglogli - Scalar negative log-likelihood (the loss to minimize)
%              Lower values = better fit to data
%
%   grad     - [n_predictors × 1] Gradient vector ∂NLL/∂w
%              Direction of steepest ascent in NLL
%              Optimization uses -grad (steepest descent)
%
%   H        - [n_predictors × n_predictors] Hessian matrix ∂²NLL/∂w²
%              Curvature of the loss function
%              Used for: Newton's method, uncertainty quantification
%
% -------------------------------------------------------------------------
% EXAMPLE USAGE
% -------------------------------------------------------------------------
%   % Setup
%   n_bins = 1000; n_predictors = 10;
%   X = randn(n_bins, n_predictors);     % Random predictors
%   w_true = randn(n_predictors, 1);     % True weights
%   lambda = exp(X * w_true);            % True rates
%   y = poissrnd(lambda);                % Poisson spike counts
%
%   % Compute loss, gradient, and Hessian
%   w_init = zeros(n_predictors, 1);     % Initial guess
%   [nll, grad, H] = negLogLikelihood_Poisson_grad(w_init, X, y, 0);
%
%   % Use with optimization
%   lossfun = @(w) negLogLikelihood_Poisson_grad(w, X, y, 0);
%   w_ml = fminunc(lossfun, w_init, options);
%
%   % Uncertainty quantification
%   [~, ~, H] = negLogLikelihood_Poisson_grad(w_ml, X, y, 0);
%   SE = sqrt(diag(inv(H)));             % Standard errors
%   z_scores = w_ml ./ SE;                % Statistical significance
%
% -------------------------------------------------------------------------
% REFERENCES
% -------------------------------------------------------------------------
% [1] Pillow et al. (2008). "Spatio-temporal correlations and visual
%     signalling in a complete neuronal population." Nature, 456(7219).
%
% [2] Truccolo et al. (2005). "A point process framework for relating
%     neural spiking activity to spiking history, neural ensemble, and
%     extrinsic covariate effects." Journal of Neurophysiology, 93(2).
%
% [3] Paninski (2004). "Maximum likelihood estimation of cascade
%     point-process neural encoding models." Network, 15(4).
%
% =========================================================================

%% Input validation and setup
if nargin < 4
    lambda = 0;  % No regularization by default
end

% Ensure w is a column vector [n_predictors × 1]
w = w(:);

%% STEP 1: Compute linear predictor u(t) = X(t,:) * w
% This is the "log firing rate" before applying the exponential nonlinearity
Xw = X * w;  % [n_bins × 1]

%% STEP 2: Apply exponential link function λ(t) = exp(u(t))
% The conditional intensity (expected spike count at each time bin)
% Exponential ensures λ(t) > 0 for all t (firing rates must be positive)
rate = exp(Xw);  % [n_bins × 1]

%% STEP 3: Compute negative log-likelihood (Poisson loss)
% NLL = sum_t [λ(t) - y(t)*u(t)]
%     = sum(rate - y .* Xw)
%
% Mathematical derivation:
%   log P(y|X,w) = sum_t [y(t)*log(λ(t)) - λ(t)] + const
%                = sum_t [y(t)*u(t) - exp(u(t))] + const
%   NLL = -log P(y|X,w) = sum_t [exp(u(t)) - y(t)*u(t)]
%
neglogli = sum(rate - y .* Xw);

% Add L2 regularization penalty: (lambda/2) * ||w||²
% This encourages smaller weights (less overfitting)
if lambda > 0
    neglogli = neglogli + 0.5 * lambda * (w' * w);
end

%% STEP 4: Compute gradient ∂NLL/∂w (if requested)
if nargout > 1
    % Gradient: ∂NLL/∂w = X' * (λ - y)
    %
    % Derivation:
    %   ∂NLL/∂u(t) = ∂λ(t)/∂u(t) - y(t) = exp(u(t)) - y(t) = λ(t) - y(t)
    %   ∂NLL/∂w_j = sum_t [∂NLL/∂u(t) * ∂u(t)/∂w_j]
    %             = sum_t [(λ(t) - y(t)) * X(t,j)]
    %             = X(:,j)' * (λ - y)
    %   ∂NLL/∂w = X' * (λ - y)
    %
    % Interpretation:
    % - (rate - y) is the "residual" (prediction error)
    % - X' weights the residual by each predictor's contribution
    % - Positive gradient → increase w makes NLL higher → should decrease w
    %
    grad = X' * (rate - y);  % [n_predictors × 1]

    % Add L2 regularization gradient: λ * w
    % This pulls weights toward zero (shrinkage)
    if lambda > 0
        grad = grad + lambda * w;
    end
end

%% STEP 5: Compute Hessian ∂²NLL/∂w² (if requested)
if nargout > 2
    % Hessian: H = X' * diag(λ) * X
    %
    % Derivation:
    %   ∂²NLL/∂w_i∂w_j = sum_t [∂²NLL/∂u(t)² * ∂u(t)/∂w_i * ∂u(t)/∂w_j]
    %                  = sum_t [λ(t) * X(t,i) * X(t,j)]
    %                  = X(:,i)' * diag(λ) * X(:,j)
    %
    % This is the "Fisher Information Matrix" - the expected curvature
    % of the log-likelihood surface.
    %
    % For numerical stability and efficiency, we compute:
    %   H = (sqrt(λ) .* X)' * (sqrt(λ) .* X)
    % instead of explicitly forming diag(λ)
    %
    sqrt_rate = sqrt(rate);                          % [n_bins × 1]
    X_weighted = bsxfun(@times, X, sqrt_rate);       % [n_bins × n_predictors]
    H = X_weighted' * X_weighted;                    % [n_predictors × n_predictors]

    % Ensure H is full (not sparse) for compatibility with inv()
    % Some MATLAB functions return sparse matrices for efficiency,
    % but inv() works better with full matrices for small problems
    if issparse(H)
        H = full(H);
    end

    % Add L2 regularization Hessian: λ * I
    % This adds λ to the diagonal, making H better conditioned
    if lambda > 0
        H = H + lambda * eye(size(H));
    end

    % Properties of the Hessian:
    % 1. Symmetric: H_ij = H_ji (order of differentiation doesn't matter)
    % 2. Positive definite: x'*H*x > 0 for all x ≠ 0 (for Poisson GLM)
    % 3. Curvature: eigenvalues show how "steep" the loss surface is
    % 4. Covariance: inv(H) approximates Cov(w) at the maximum likelihood
    %
    % Uses of Hessian:
    % - Newton's method: w_new = w_old - H^(-1) * grad (faster convergence)
    % - Standard errors: SE(w_i) = sqrt(inv(H)_ii)
    % - Confidence intervals: w_i ± 1.96 * SE(w_i) for 95% CI
    % - Hypothesis testing: z = w_i / SE(w_i), p = 2*normcdf(-|z|)
end

end
