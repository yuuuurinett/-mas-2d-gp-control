function [P, p_dim] = gp_masked_aggregation_init( ...
    LocalGP_set, AgentQuantity, ...
    NumInducingPoints, InducingPoints_Coordinates, method)
%    InducingPoints_Coordinates : x_dim x M
%    method : one of {'poe','gpoe','moe','bcm','rbcm'}

%    P      : p_dim x AgentQuantity x M
%    p_dim  : dimension of P's first axis (4 or 6)

method = lower(method);
M = NumInducingPoints;
prior_var = LocalGP_set{1}.SigmaF^2;  % sigma_0^²

switch method
    case {'poe', 'gpoe', 'moe','bcm'}
        p_dim = 4;   % 2 rows per output dim (numerator, denominator)
    case {'rbcm'}
        p_dim = 6;   % 3 rows per output dim (numerator, precision, beta_sum)
    otherwise
        error('Unknown aggregation method: %s. Choose from poe/gpoe/moe/bcm/rbcm.', method);
end

P = zeros(p_dim, AgentQuantity, M);

for AgentNr = 1:AgentQuantity
    for InducingPointIdx = 1:M
        x_m = InducingPoints_Coordinates(:, InducingPointIdx);  % x_dim x 1
        [mu_n, var_n] = LocalGP_set{AgentNr}.predict(x_m);  % 2x1, 2x1

        mu1  = mu_n(1); 
        var1 = var_n(1);
        mu2  = mu_n(2);  
        var2 = var_n(2);

        switch method

            case 'poe'                        
                P(1, AgentNr, InducingPointIdx) = AgentQuantity * mu1 / var1;
                P(2, AgentNr, InducingPointIdx) = AgentQuantity / var1;
                P(3, AgentNr, InducingPointIdx) = AgentQuantity * mu2 / var2;
                P(4, AgentNr, InducingPointIdx) = AgentQuantity / var2;

            case 'gpoe'        
                beta1 = max(eps, 0.5 * (log(prior_var) - log(var1)));
                beta2 = max(eps, 0.5 * (log(prior_var) - log(var2)));
                P(1, AgentNr, InducingPointIdx) = AgentQuantity * beta1 * mu1 / var1;
                P(2, AgentNr, InducingPointIdx) = AgentQuantity * beta1 / var1;
                P(3, AgentNr, InducingPointIdx) = AgentQuantity * beta2 * mu2 / var2;
                P(4, AgentNr, InducingPointIdx) = AgentQuantity * beta2 / var2;

            case 'moe'
                omega_n = 1.0 / AgentQuantity;  % uniform gating weight
                P(1, AgentNr, InducingPointIdx) = AgentQuantity * omega_n * mu1;   
                P(2, AgentNr, InducingPointIdx) = AgentQuantity * omega_n;         
                P(3, AgentNr, InducingPointIdx) = AgentQuantity * omega_n * mu2;   
                P(4, AgentNr, InducingPointIdx) = AgentQuantity * omega_n;         

            case 'bcm'
                P(1, AgentNr, InducingPointIdx) = AgentQuantity * mu1 / var1;
                P(2, AgentNr, InducingPointIdx) = AgentQuantity * mu2 / var2;
                P(3, AgentNr, InducingPointIdx) = AgentQuantity / var1;
                P(4, AgentNr, InducingPointIdx) = AgentQuantity / var2;
            case 'rbcm'           
                beta1 = max(eps, 0.5 * (log(prior_var) - log(var1)));
                beta2 = max(eps, 0.5 * (log(prior_var) - log(var2)));
                P(1, AgentNr, InducingPointIdx) = AgentQuantity * beta1 * mu1 / var1;
                P(2, AgentNr, InducingPointIdx) = AgentQuantity * beta1 / var1;
                P(3, AgentNr, InducingPointIdx) = AgentQuantity * beta1;     
                P(4, AgentNr, InducingPointIdx) = AgentQuantity * beta2 * mu2 / var2;
                P(5, AgentNr, InducingPointIdx) = AgentQuantity * beta2 / var2;
                P(6, AgentNr, InducingPointIdx) = AgentQuantity * beta2;
        end
    end
end
end