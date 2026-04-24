function MaskedGP = gp_masked_aggregation_ac( ...
    P, InducingPoints_Coordinates, SigmaF, SigmaL, ...
    x_dim, AgentQuantity, NumInducingPoints, method, p_dim)
%% gp_masked_aggregation_ac

method    = lower(method);
M         = NumInducingPoints;
y_dim     = 2;
prior_var = SigmaF^2;

%% AC: Xi = mean(P, 2) broadcast to all agents
Xi_mean = mean(P, 2);                          % p_dim x 1
Xi_all  = repmat(Xi_mean, 1, AgentQuantity);   % p_dim x AgentQuantity
%% Decode fused mean (same formulas as gp_masked_aggregation_update)
switch method
    case {'poe', 'gpoe', 'moe'}
        num1 = squeeze(Xi_all(1, :, :));  % AgentQuantity x M
        num2 = squeeze(Xi_all(2, :, :));
        den1 = squeeze(Xi_all(3, :, :));
        den2 = squeeze(Xi_all(4, :, :));

        phi1 = num1 ./ den1;
        phi2 = num2 ./ den2;

    case 'bcm'
        num1 = squeeze(Xi_all(1, :, :));
        num2 = squeeze(Xi_all(2, :, :));
        den1 = squeeze(Xi_all(3, :, :));
        den2 = squeeze(Xi_all(4, :, :));

        prior_correction = (1 - AgentQuantity) / prior_var;
        phi1 = num1 ./ (den1 + prior_correction);
        phi2 = num2 ./ (den2 + prior_correction);

    case 'rbcm'
        % P layout: [N*β1*mu1/var1, N*β1/var1, N*β1, N*β2*mu2/var2, N*β2/var2, N*β2]
        num1  = squeeze(Xi_all(1, :, :));
        den1  = squeeze(Xi_all(2, :, :));
        beta1 = squeeze(Xi_all(3, :, :));
        num2  = squeeze(Xi_all(4, :, :));
        den2  = squeeze(Xi_all(5, :, :));
        beta2 = squeeze(Xi_all(6, :, :));

        phi1 = num1 ./ (den1 + (1 - beta1) / prior_var);
        phi2 = num2 ./ (den2 + (1 - beta2) / prior_var);
end

%% Rebuild one LocalGP per agent
MaskedGP = cell(AgentQuantity, 1);
for AgentNr = 1:AgentQuantity
    Y_agent = [phi1(AgentNr, :); phi2(AgentNr, :)];
    MaskedGP{AgentNr} = LocalGP_MultiOutput(x_dim, y_dim, M, 1e-6, SigmaF, SigmaL);
    MaskedGP{AgentNr}.add_Alldata(InducingPoints_Coordinates, Y_agent);
end
end