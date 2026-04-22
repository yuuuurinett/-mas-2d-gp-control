function [MaskedGP, Zeta_vector] = gp_masked_aggregation_update( ...
    P, Zeta_vector, L, ...
    Kappa_P, AgentQuantity, NumInducingPoints, TimeStep, ...
    InducingPoints_Coordinates, SigmaF, SigmaL, x_dim, method, p_dim)
%% gp_masked_aggregation_update
%  Run DAC step on P, then recover fused GP predictions at inducing points,
%  and build a new LocalGP per agent from those predictions.
%
%  DAC dynamics (continuous-time ODE, integrated with ode45):
%    Ż = κ (P - Z) L'
%  At convergence: Z → P·(L'/L') consensus → each row of Z approaches
%  the network average of that row of P, scaled appropriately.
%
%  After DAC, the consensus error Xi = P - Zeta carries the fused statistics.
%  The fusion formula to recover µ̃ depends on the method (see table below).
%
%  -----------------------------------------------------------------------
%  method | fusion formula for µ̃ (per output dim d)
%  -------|---------------------------------------------------------------
%  poe    | µ̃ = Xi_num / Xi_den
%         |   Xi_num = Xi(1,:,:)  ← avg of N*µ/σ²  → (1/K)Σµₙ/σₙ²
%         |   Xi_den = Xi(2,:,:)  ← avg of N/σ²    → (1/K)Σ1/σₙ²
%         |   ratio = (Σµₙ/σₙ²) / (Σ1/σₙ²)   [precision-weighted mean]
%  -------|---------------------------------------------------------------
%  gpoe   | identical structure to PoE, but with β-weighted statistics
%         | µ̃ = Xi(1) / Xi(2)   [β-weighted precision mean]
%  -------|---------------------------------------------------------------
%  moe    | µ̃ = Xi(1) / Xi(2)   [uniform weighted mean = simple average]
%  -------|---------------------------------------------------------------
%  bcm    | µ̃ = Xi_num / (Xi_den + (1 - Xi_cnt/σ₀²) / σ₀²)  [prior correction]
%         | denominator = Σ(1/σₙ²) + (1 - K)/σ₀²
%         | Xi_cnt carries K (the expert count) after averaging
%  -------|---------------------------------------------------------------
%  rbcm   | µ̃ = Xi_num / (Xi_den + (1 - Xi_beta)/σ₀²)
%         | denominator = Σβₙ/σₙ² + (1 - Σβₙ)/σ₀²
%         | Xi_beta carries Σβₙ after averaging
%  -----------------------------------------------------------------------
%
%  Inputs:
%    P          : p_dim x AgentQuantity x M  (from gp_masked_aggregation_init)
%    Zeta_vector: p_dim x AgentQuantity x M  (DAC state, updated in-place)
%    L          : AgentQuantity x AgentQuantity Laplacian
%    Kappa_P    : DAC gain κ
%    AgentQuantity, NumInducingPoints, TimeStep
%    InducingPoints_Coordinates : x_dim x M
%    SigmaF, SigmaL             : GP hyperparameters for rebuilt GPs
%    x_dim                      : input dimension
%    method                     : string {'poe','gpoe','moe','bcm','rbcm'}
%    p_dim                      : 4 or 6 (from init)
%
%  Outputs:
%    MaskedGP    : cell(AgentQuantity,1), each a LocalGP_MultiOutput built
%                  from the fused inducing-point predictions
%    Zeta_vector : updated DAC state (p_dim x AgentQuantity x M)

method  = lower(method);
M       = NumInducingPoints;
y_dim   = 2;
prior_var = SigmaF^2;   % σ₀²

%% Step 1: DAC integration over [0, TimeStep]
if TimeStep > 0
    for m = 1:M
        P_m    = P(:, :, m);             % p_dim x AgentQuantity
        Zeta_m = Zeta_vector(:, :, m);   % p_dim x AgentQuantity

        dac_ode = @(~, z) dac_derivative(z, P_m, L, Kappa_P, AgentQuantity, p_dim);
        [~, z_out] = ode45(dac_ode, [0, TimeStep], Zeta_m(:));

        Zeta_vector(:, :, m) = reshape(z_out(end,:)', p_dim, AgentQuantity);
    end
end

%% Step 2: Recover fused mean µ̃ at each inducing point for each agent
%  Xi = P - Zeta  captures the consensus error, which at consensus ≈ local P_n - global_avg(P)
%  But actually we use Xi directly as the "local view of the global statistics":
%  After DAC converges: Zeta_n → (1/K) Σ P_k  for all n (all agents agree)
%  So Xi_n = P_n - Zeta_n → P_n - (1/K)Σ P_k
%  But P(1)/P(2) ratio after summing recovers the global fused mean:
%    mean(P_num) / mean(P_den) = (Σµₙ/σₙ²/K) / (Σ1/σₙ²/K) = µ̃_PoE  ✓
%  The ratio of Xi rows at convergence: Xi(1)/Xi(2) approaches the same.
Xi_all = P - Zeta_vector;   % p_dim x AgentQuantity x M

% Extract per-dim statistics (all agents, all inducing points)
switch method
    case {'poe', 'gpoe', 'moe'}
        % p_dim = 4: [num1, den1, num2, den2]
        num1 = squeeze(Xi_all(1, :, :));  % AgentQuantity x M
        den1 = squeeze(Xi_all(2, :, :));
        num2 = squeeze(Xi_all(3, :, :));
        den2 = squeeze(Xi_all(4, :, :));

        phi1 = num1 ./ den1;   % AgentQuantity x M, fused µ̃ for dim 1
        phi2 = num2 ./ den2;

    case 'bcm'
        % p_dim = 6: [num1, den1, cnt1, num2, den2, cnt2]
        % µ̃ = num / (den + (1 - cnt/K) / σ₀²)  ... but cnt converges to K
        % More precisely: den_eff = Σ(1/σₙ²) + (1-K)/σ₀²
        %   Xi(2) after consensus ≈ (1/K)Σ(N/σₙ²) = Σ(1/σₙ²)/K * N/K ... 
        %   Re-derive: each agent contributes P(2)=N/var, DAC averages → Zeta(2)→ N*avg(1/var)
        %   Xi(2) = P_n(2) - Zeta(2) ≈ local deviation, but we need global sum.
        %   Since Zeta → (1/K)*sum(P), sum(Xi) = sum(P) - K*(1/K)*sum(P) = 0 ... 
        %   Use Xi at agent 1 (all agents agree at convergence):
        %     Xi(2,agent,:) ≈ P_n(2) - avg_P(2) = N/var_n - (1/K)Σ(N/var_k)
        %   Actually the correct approach: the converged Zeta for any agent ≈ (1/K)Σ P_k
        %   So for agent n: Zeta_n(2) = (1/K)Σ(N/var_k) = (N/K)Σ(1/var_k)
        %   The global sum Σ(1/var_k) = K * Zeta_n(2) / N
        %   Agent n uses Zeta directly for reconstruction:
        num1 = squeeze(Zeta_vector(1, :, :));   % (N/K) Σ(µₙ/σₙ²)
        den1 = squeeze(Zeta_vector(2, :, :));   % (N/K) Σ(1/σₙ²)
        cnt1 = squeeze(Zeta_vector(3, :, :));   % (N/K) * K = N  (constant)
        num2 = squeeze(Zeta_vector(4, :, :));
        den2 = squeeze(Zeta_vector(5, :, :));
        cnt2 = squeeze(Zeta_vector(6, :, :));

        % Reconstruct global sums from Zeta (which ≈ (1/K)*global_sum * N):
        %   global_sum(µ/σ²)  = K * Zeta(1) / N  ... but ratio is invariant
        %   µ̃_BCM = Σ(µₙ/σₙ²) / [Σ(1/σₙ²) + (1-K)/σ₀²]
        %          = Zeta(1) / [Zeta(2) + (1-cnt/N)*N/σ₀²/K]
        %   Simplify: at convergence cnt = N (all agents contribute N to P(3))
        %   (1-K) factor: cnt converges to N (since each agent puts N in P(3))
        %   K agents each put N → sum = K*N, average = N. So cnt ≈ N always.
        %   prior correction: (1-K)/σ₀² = (1 - AgentQuantity)/prior_var
        prior_correction = (1 - AgentQuantity) / prior_var;
        denom_eff1 = den1 + prior_correction * (AgentQuantity / AgentQuantity);
        denom_eff2 = den2 + prior_correction * (AgentQuantity / AgentQuantity);
        % More directly using scaled Zeta:
        %   µ̃ = Zeta(1)/Zeta(2) but with BCM correction term
        %   denom = K * [Zeta(2)/N] + (1-K)/σ₀²
        %         = Zeta(2) + (1-K)*N/(K*σ₀²) * (K/N)... let's simplify:
        % Zeta(i) → (1/K) sum_n P_n(i), and P_n(i) = N * (quantity_n)
        % So Zeta(i) → (N/K) * sum_n(quantity_n) = N * global_avg(quantity_n)
        % We want global_sum(quantity_n) = Zeta(i) * K / N = Zeta(i) / (N/K)
        % For BCM denom: Σ(1/σₙ²) + (1-K)/σ₀²
        %   Σ(1/σₙ²) = (K/N) * Zeta(2)
        % BCM denom = (K/N)*Zeta(2) + (1-K)/σ₀²
        K = AgentQuantity; N_scale = AgentQuantity;
        global_prec1 = (K / N_scale) * den1;
        global_prec2 = (K / N_scale) * den2;
        global_num1  = (K / N_scale) * num1;
        global_num2  = (K / N_scale) * num2;

        bcm_denom1 = global_prec1 + (1 - K) / prior_var;
        bcm_denom2 = global_prec2 + (1 - K) / prior_var;

        phi1 = global_num1 ./ bcm_denom1;
        phi2 = global_num2 ./ bcm_denom2;

    case 'rbcm'
        % rBCM: µ̃ = Σ(βₙµₙ/σₙ²) / [Σ(βₙ/σₙ²) + (1-Σβₙ)/σ₀²]
        num1  = squeeze(Zeta_vector(1, :, :));
        den1  = squeeze(Zeta_vector(2, :, :));
        bsum1 = squeeze(Zeta_vector(3, :, :));
        num2  = squeeze(Zeta_vector(4, :, :));
        den2  = squeeze(Zeta_vector(5, :, :));
        bsum2 = squeeze(Zeta_vector(6, :, :));

        K = AgentQuantity; N_scale = AgentQuantity;
        global_num1  = (K / N_scale) * num1;
        global_den1  = (K / N_scale) * den1;
        global_beta1 = (K / N_scale) * bsum1;
        global_num2  = (K / N_scale) * num2;
        global_den2  = (K / N_scale) * den2;
        global_beta2 = (K / N_scale) * bsum2;

        rbcm_denom1 = global_den1  + (1 - global_beta1) / prior_var;
        rbcm_denom2 = global_den2  + (1 - global_beta2) / prior_var;

        phi1 = global_num1 ./ rbcm_denom1;
        phi2 = global_num2 ./ rbcm_denom2;
end

%% Step 3: Rebuild one LocalGP per agent from fused inducing-point predictions
MaskedGP = cell(AgentQuantity, 1);
for AgentNr = 1:AgentQuantity
    Y_agent = [phi1(AgentNr, :); phi2(AgentNr, :)];  % 2 x M
    MaskedGP{AgentNr} = LocalGP_MultiOutput(x_dim, y_dim, M, 1e-6, SigmaF, SigmaL);
    MaskedGP{AgentNr}.add_Alldata(InducingPoints_Coordinates, Y_agent);
end

end

%% Local helper: DAC ODE derivative
function dz = dac_derivative(z_vec, P_ref, L, Kappa, AgentQty, p_dim)
    Z = reshape(z_vec, p_dim, AgentQty);
    dz = Kappa * (P_ref - Z) * L';
    dz = dz(:);
end