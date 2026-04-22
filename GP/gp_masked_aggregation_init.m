function [P, p_dim] = gp_masked_aggregation_init( ...
    LocalGP_set, AgentQuantity, ...
    NumInducingPoints, InducingPoints_Coordinates, method)
%% gp_masked_aggregation_init
%  Compute reference signal P at all inducing points for each agent.
%  P is used as the DAC reference (the "innovation" ηn) in consensus dynamics.
%
%  Method determines the structure of P:
%
%  -----------------------------------------------------------------------
%  method   | p_dim | P rows per output dim    | Aggregation target
%  ---------|-------|--------------------------|---------------------------
%  'poe'    |  4   | [N*mu/var; N/var]         | Σ(μ/σ²) / Σ(1/σ²)
%  'gpoe'   |  4   | [N*β*mu/var; N*β/var]     | Σ(β*μ/σ²) / Σ(β/σ²)
%  'moe'    |  4   | [N*π*mu; N*π]             | Σ(π*μ) / Σ(π)  = mean(μ)
%  'bcm'    |  6   | [N*mu/var; N/var; N]       | Σ(μ/σ²) / (Σ(1/σ²)+(1-K)/σ₀²)
%  'rbcm'   |  6   | [N*β*mu/var; N*β/var; N*β]| Σ(β*μ/σ²) / (Σ(β/σ²)+(1-Σβ)/σ₀²)
%  -----------------------------------------------------------------------
%
%  Note: rows are interleaved per output dimension, i.e.:
%    rows 1,2,[3]  → output dim 1
%    rows 2+1,2+2,[2+3] → output dim 2
%  For p_dim=4: rows [1,2] for dim1, [3,4] for dim2
%  For p_dim=6: rows [1,2,3] for dim1, [4,5,6] for dim2
%
%  Inputs:
%    LocalGP_set              : cell(AgentQuantity,1), each LocalGP_MultiOutput
%    AgentQuantity            : number of agents N
%    NumInducingPoints        : M
%    InducingPoints_Coordinates : x_dim x M
%    method                   : string, one of {'poe','gpoe','moe','bcm','rbcm'}
%
%  Outputs:
%    P      : p_dim x AgentQuantity x M
%    p_dim  : dimension of P's first axis (4 or 6)

method = lower(method);
M = NumInducingPoints;
prior_var = LocalGP_set{1}.SigmaF^2;  % σ₀²

switch method
    case {'poe', 'gpoe', 'moe'}
        p_dim = 4;   % 2 rows per output dim (numerator, denominator)
    case {'bcm', 'rbcm'}
        p_dim = 6;   % 3 rows per output dim (numerator, precision, beta_sum)
    otherwise
        error('Unknown aggregation method: %s. Choose from poe/gpoe/moe/bcm/rbcm.', method);
end

P = zeros(p_dim, AgentQuantity, M);

for AgentNr = 1:AgentQuantity
    for m = 1:M
        x_m = InducingPoints_Coordinates(:, m);  % x_dim x 1
        [mu_n, var_n] = LocalGP_set{AgentNr}.predict(x_m);  % 2x1, 2x1

        mu1  = mu_n(1);  var1 = var_n(1);
        mu2  = mu_n(2);  var2 = var_n(2);

        switch method

            case 'poe'
                % PoE (Hinton 2002):
                %   σ̃⁻² = Σ σₙ⁻²
                %   µ̃    = σ̃² · Σ (µₙ / σₙ²)
                % DAC reference signal: P(1) = N*µ/σ², P(2) = N/σ²
                % After consensus: ξ(1)/ξ(2) → average of µₙ/σₙ² / average of 1/σₙ²
                %                             = (Σµₙ/σₙ²) / (Σ1/σₙ²) = µ̃_PoE
                P(1, AgentNr, m) = AgentQuantity * mu1 / var1;
                P(2, AgentNr, m) = AgentQuantity / var1;
                P(3, AgentNr, m) = AgentQuantity * mu2 / var2;
                P(4, AgentNr, m) = AgentQuantity / var2;

            case 'gpoe'
                % gPoE (Cao & Fleet 2014):
                %   βₙ = 0.5 * (log σ₀² - log σₙ²)  [information gain weight]
                %   σ̃⁻² = Σ βₙ σₙ⁻²
                %   µ̃    = σ̃² · Σ βₙ µₙ / σₙ²
                % Same structure as PoE but numerator/denominator weighted by βₙ
                beta1 = 0.5 * (log(prior_var) - log(var1));
                beta2 = 0.5 * (log(prior_var) - log(var2));
                P(1, AgentNr, m) = AgentQuantity * beta1 * mu1 / var1;
                P(2, AgentNr, m) = AgentQuantity * beta1 / var1;
                P(3, AgentNr, m) = AgentQuantity * beta2 * mu2 / var2;
                P(4, AgentNr, m) = AgentQuantity * beta2 / var2;

            case 'moe'
                % MoE (Jacobs et al. 1991, Tresp 2001):
                %   Uniform gating: πₙ = 1/K
                %   µ̃ = Σ πₙ µₙ = (1/K) Σ µₙ
                % DAC reference: P(1) = N * πₙ * µₙ = µₙ  (since N*1/N = 1)
                %                P(2) = N * πₙ       = 1
                % After consensus: ξ(1)/ξ(2) → (1/K)Σµₙ / (1/K)Σ1 = mean(µₙ)
                % Note: variance aggregation (law of total variance) requires
                %       post-hoc computation, not tracked via DAC here.
                pi_n = 1.0 / AgentQuantity;  % uniform gating weight
                P(1, AgentNr, m) = AgentQuantity * pi_n * mu1;   % = mu1
                P(2, AgentNr, m) = AgentQuantity * pi_n;         % = 1
                P(3, AgentNr, m) = AgentQuantity * pi_n * mu2;   % = mu2
                P(4, AgentNr, m) = AgentQuantity * pi_n;         % = 1

            case 'bcm'
                % BCM (Tresp 2000):
                %   σ̃⁻² = Σ σₙ⁻² + (1 - K) σ₀⁻²
                %   µ̃    = σ̃² · [Σ µₙ/σₙ² + (1-K) µ₀/σ₀²]   (µ₀=0 here)
                % 3 rows per dim: [N*mu/var, N/var, N]
                %   After consensus: row1/row2 contributes Σµₙ/σₙ²
                %                    row3 carries Σ1 = K to compute (1-K)/σ₀²
                P(1, AgentNr, m) = AgentQuantity * mu1 / var1;
                P(2, AgentNr, m) = AgentQuantity / var1;
                P(3, AgentNr, m) = AgentQuantity;               % carries K count
                P(4, AgentNr, m) = AgentQuantity * mu2 / var2;
                P(5, AgentNr, m) = AgentQuantity / var2;
                P(6, AgentNr, m) = AgentQuantity;

            case 'rbcm'
                % rBCM (Deisenroth & Ng 2015):
                %   βₙ = 0.5 * (log σ₀² - log σₙ²)
                %   σ̃⁻² = Σ βₙ σₙ⁻² + (1 - Σβₙ) σ₀⁻²
                %   µ̃    = σ̃² · Σ βₙ µₙ / σₙ²   (µ₀=0)
                % 3 rows per dim: [N*β*mu/var, N*β/var, N*β]
                beta1 = 0.5 * (log(prior_var) - log(var1));
                beta2 = 0.5 * (log(prior_var) - log(var2));
                P(1, AgentNr, m) = AgentQuantity * beta1 * mu1 / var1;
                P(2, AgentNr, m) = AgentQuantity * beta1 / var1;
                P(3, AgentNr, m) = AgentQuantity * beta1;       % carries Σβ
                P(4, AgentNr, m) = AgentQuantity * beta2 * mu2 / var2;
                P(5, AgentNr, m) = AgentQuantity * beta2 / var2;
                P(6, AgentNr, m) = AgentQuantity * beta2;
        end
    end
end
end