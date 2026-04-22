function run_main_simulation_mode(CurrentMode, SaveFolderName, SaveFileName)  
    rng(0); 
    %% 1. Multi Agent System Topology
    AgentQuantity = 6;
    AgentConnectionConfigurationSet = {
        {1,2,'Directed',0.5}; {1,6,'Directed',0.5};
        {2,3,'Directed',0.5}; {2,4,'Directed',0.5};  
        {3,2,'Directed',0.5}; {3,1,'Directed',0.5};
        {4,3,'Directed',0.5}; {4,5,'Directed',0.5};
        {5,6,'Directed',0.5}; {5,4,'Directed',0.5};
        {6,5,'Directed',0.5}; {6,1,'Directed',0.5}
    };

    LeaderQuantity = 1;
    AgentLeaderConnectionConfigurationSet = {
        {'Leader',1,'Agent',1,1};
        {'Leader',1,'Agent',4,1}
    };

    MultiAgentSystem = MultiAgentSystem_Class(AgentQuantity,LeaderQuantity);
    MultiAgentSystem.Agent_Topology.addConnectionSet(AgentConnectionConfigurationSet);
    MultiAgentSystem.Agent_Topology.check_Connectivity(true);
    MultiAgentSystem.Agent_Leader_Topology.addConnectionSet(AgentLeaderConnectionConfigurationSet);
    MultiAgentSystem.get_ExtendedTopology;

    L = MultiAgentSystem.Agent_Topology.LaplacianMatrix;
    B = MultiAgentSystem.Agent_Leader_Topology.ConnectionMatrix;

    %% 2. Time Horizon & Parameters
    t_start = 0; t_end = 4; t_step = 0.01;
    t_set = t_start:t_step:t_end;
    
    SystemOrder = 2; q_dim = 2; x_dim = q_dim * SystemOrder;
    m1 = 1; m2 = 1; L1 = 1; L2 = 1; g = 9.8;

    %% 3. Set Reference Trajectory
    x_trajectory_set = nan(x_dim, numel(t_set));
    x_trajectory_set(1, :) = sin(0.5*t_set);     
    x_trajectory_set(2, :) = cos(0.5*t_set);      
    x_trajectory_set(3, :) = 0.5*cos(0.5*t_set);  
    x_trajectory_set(4, :) = -0.5*sin(0.5*t_set); 

    %% 4. Set Gaussian Processes
    SigmaF = 1; SigmaL = 0.5 * ones(x_dim,1);
    GP_tau = 1e-8; GP_delta = 0.1; y_dim = q_dim;
    LocalGP_Quantity = AgentQuantity;
    DomainScale = 1.5;
    X_min = DomainScale * [-1, -1, -1, -1];
    X_max = DomainScale * [ 1,  1,  1,  1];

    Local_X_min = DomainScale * [-1, -1, -1, -1; -1, -1,  0,  0; 0,  0,  0,  0; 0,  0, -1, -1; -1,  0, -1,  0; 0, -1,  0, -1];
    Local_X_max = DomainScale * [ 0,  0,  0,  0; 0,  0,  1,  1; 1,  1,  1,  1; 1,  1,  0,  0; 0,  1,  0,  1; 1,  0,  1,  0];

    MaxDataQuantity_set = 100 * ones(AgentQuantity,1);
    OfflineDataQuantity_set = 100 * ones(AgentQuantity,1);
    SigmaN_set = 0.1 * ones(AgentQuantity,1);

    LocalGP_set = cell(LocalGP_Quantity,1);
    for LocalGP_Nr = 1:LocalGP_Quantity
        MaxDataQuantity = MaxDataQuantity_set(LocalGP_Nr);
        OfflineDataQuantity = OfflineDataQuantity_set(LocalGP_Nr);
        SigmaN = SigmaN_set(LocalGP_Nr);

        LocalGP_set{LocalGP_Nr} = LocalGP_MultiOutput(x_dim, y_dim, MaxDataQuantity, SigmaN, SigmaF, SigmaL);
      
        Data_mu = (Local_X_max(LocalGP_Nr,:) + Local_X_min(LocalGP_Nr,:)) / 2;
        Data_Le = (Local_X_max(LocalGP_Nr,:) - Local_X_min(LocalGP_Nr,:)) / 2;
        X_in = 2 * (rand(x_dim,OfflineDataQuantity) - 0.5) .* repmat(Data_Le',[1,OfflineDataQuantity]) + Data_mu';

        Y_in = zeros(y_dim, OfflineDataQuantity);
        for d = 1:OfflineDataQuantity
            Y_in(:,d) = Manipulator_2D_2DoF_UnknownDynamics(X_in(:,d));
        end
        Y_in = Y_in + SigmaN * randn(size(Y_in));

        LocalGP_set{LocalGP_Nr}.add_Alldata(X_in, Y_in); 
        LocalGP_set{LocalGP_Nr}.tau = GP_tau;
        LocalGP_set{LocalGP_Nr}.delta = GP_delta;
        LocalGP_set{LocalGP_Nr}.xMax = max(X_in, [], 2);
        LocalGP_set{LocalGP_Nr}.xMin = min(X_in, [], 2);
    end

    %% 5. Simulation Setup
    rng(0); 
    AgentInitialState_matrix = rand(4,AgentQuantity) - 0.5;
    lambda_set = [7/4; 1];
    Kappa_C = 1000;
    AdjMatrix  = MultiAgentSystem.Agent_Topology.AdjacencyMatrix;
    NeighbourSet = MultiAgentSystem.Agent_Topology.NeighbourSet;

    NumInducingPoints = 100;
    InducingPoints_Coordinates = 2 * DomainScale * rand(x_dim, NumInducingPoints) - DomainScale;
    
    TrackingError_vector = zeros(1, numel(t_set));

    %% 6. Mode-specific initialization
    % Determine if this is one of the 5 masked aggregation modes
    masked_methods = {'poe', 'gpoe', 'moe', 'bcm', 'rbcm'};
    is_masked_mode = ismember(lower(CurrentMode), masked_methods);

    switch lower(CurrentMode)
        case masked_methods
            % All 5 masked aggregation modes share the same init structure
            Kappa_P = 10;
            agg_method = lower(CurrentMode);
            [P_inducing, p_dim] = gp_masked_aggregation_init( ...
                LocalGP_set, AgentQuantity, NumInducingPoints, ...
                InducingPoints_Coordinates, agg_method);
            Zeta_vector_inducing = zeros(p_dim, AgentQuantity, NumInducingPoints);
            % Initial DAC update at t=0 (TimeStep=0 → no ODE, just build GP)
            [MaskedGP, Zeta_vector_inducing] = gp_masked_aggregation_update( ...
                P_inducing, Zeta_vector_inducing, L, Kappa_P, AgentQuantity, ...
                NumInducingPoints, 0, InducingPoints_Coordinates, SigmaF, SigmaL, ...
                x_dim, agg_method, p_dim);

        case 'distributed_poe'
            Kappa_P = 1000;  
            Zeta_vector = zeros(4, AgentQuantity);

        case 'distributed_rbcm'
            Kappa_P = 1000;  
            Zeta_vector = zeros(6, AgentQuantity); 

        otherwise % 'local', 'exact', 'none'
            Kappa_P = 0;  
            Zeta_vector = zeros(4, AgentQuantity);
    end

    %% 7. Control Loop
    tic; t_gp = 0; t_ode = 0;
    AgentState_matrix = AgentInitialState_matrix;

    for k = 1:numel(t_set)
        CurrentTime = t_set(k);
        LeaderState_vector = x_trajectory_set(:, k);
        TrackingError_vector(1, k) = norm(AgentState_matrix(:) - repmat(LeaderState_vector, AgentQuantity, 1));
        
        if k == numel(t_set), break; end

        Phi_Xi_vector = zeros(q_dim, AgentQuantity);
        tic_gp = tic;

        switch lower(CurrentMode)
            case masked_methods
                % Step 1: Predict at current agent states using fused GP
                for AgentNr = 1:AgentQuantity
                    x_i = AgentState_matrix(:, AgentNr);
                    [mu_hat, ~] = MaskedGP{AgentNr}.predict(x_i);
                    Phi_Xi_vector(:, AgentNr) = mu_hat;
                end
                % Step 2: Update DAC and rebuild fused GP for next step
                [MaskedGP, Zeta_vector_inducing] = gp_masked_aggregation_update( ...
                    P_inducing, Zeta_vector_inducing, L, Kappa_P, AgentQuantity, ...
                    NumInducingPoints, t_step, InducingPoints_Coordinates, ...
                    SigmaF, SigmaL, x_dim, agg_method, p_dim);

            case 'distributed_poe'
                [Phi_Xi_vector, Zeta_vector, ~] = gp_poe( ...
                    AgentState_matrix, LocalGP_set, L, ...
                    Kappa_P, AgentQuantity, Zeta_vector, t_step);

            case 'distributed_rbcm'
                [Phi_Xi_vector, Zeta_vector] = gp_rbcm( ...
                    AgentState_matrix, LocalGP_set, L, ...
                    Kappa_P, AgentQuantity, Zeta_vector, t_step);

            case 'local'
                for AgentNr = 1:AgentQuantity
                    [mu_n, ~] = LocalGP_set{AgentNr}.predict(AgentState_matrix(:, AgentNr));
                    Phi_Xi_vector(:, AgentNr) = mu_n;
                end

            case 'none'
                Phi_Xi_vector = zeros(q_dim, AgentQuantity);

            case 'exact'
                for AgentNr = 1:AgentQuantity
                    Phi_Xi_vector(:, AgentNr) = Manipulator_2D_2DoF_UnknownDynamics(AgentState_matrix(:, AgentNr));
                end
        end

        t_gp = t_gp + toc(tic_gp);

        MultiAgent_Dynamics_Handle = @(t, x) multi_agent_dynamics( ...
            t, x, Phi_Xi_vector, AdjMatrix, NeighbourSet, B, lambda_set, ...
            Kappa_C, AgentQuantity, SystemOrder, q_dim, L1, L2, m1, m2);

        tic_ode = tic;
        opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);
        [~, x_output] = ode45(MultiAgent_Dynamics_Handle, ...
            [CurrentTime, CurrentTime + t_step], AgentState_matrix(:), opts);
        AgentState_matrix = reshape(x_output(end,:)', x_dim, AgentQuantity);
        t_ode = t_ode + toc(tic_ode);
        fprintf('t = %6.4f\n', CurrentTime);
    end
    fprintf('Mode: %s done, 总时间=%.2f s, GP=%.2f s, ODE=%.2f s\n', CurrentMode, toc, t_gp, t_ode);

    %% 8. Compute Ultimate Error Bound
    L_tilde = MultiAgentSystem.Extended_Topology.LaplacianMatrix(1:AgentQuantity, 1:AgentQuantity);
    sigma_L_tilde = min(svd(L_tilde));
    sigma_bar_L_tilde = max(svd(L_tilde));

    LyapunovQ_gain = 600; Q_val = LyapunovQ_gain;
    P_val = Q_val / (2 * lambda_set(1));
    Lambda_norm = lambda_set(1);

    iota = norm(eye(AgentQuantity) - L_tilde) * lambda_set(1);
    Upsilon = [Kappa_C * sigma_L_tilde - iota, -0.5*(1 + lambda_set(1)) - 0.5*sqrt(P_val);
              -0.5*(1 + lambda_set(1)) - 0.5*sqrt(P_val), 0.5*Q_val];
    sigma_Upsilon = min(eig(Upsilon));

    C = sqrt(AgentQuantity) * sigma_bar_L_tilde * sqrt(P_val) * (1 + Lambda_norm) / (sigma_Upsilon * sigma_L_tilde);
    Delta_hat = 1;
    GP_tau_bound = 0.01; r_Omega  = 2*sqrt(2);
    beta_val = 2*2*log(r_Omega*sqrt(2)) - log(2*GP_tau_bound) - 2*log(GP_delta/AgentQuantity);
    beta_val = max(beta_val, 0);

    bound_distributed = zeros(1, numel(t_set));
    bound_local = zeros(1, numel(t_set));
    bound_exact = zeros(1, numel(t_set));

    for k = 1:numel(t_set)
        LeaderState_k = x_trajectory_set(:, k);
        sigma_sum_dist = 0; sigma_sum_local = 0; precison_sum = 0;

        for AgentNr = 1:AgentQuantity
            [~, var_n] = LocalGP_set{AgentNr}.predict(LeaderState_k);
            var_n = var_n(1);
            sigma_sum_dist = sigma_sum_dist + (1/var_n) * sqrt(var_n);
            sigma_sum_local = sigma_sum_local + sqrt(var_n);
            precison_sum = precison_sum + 1/var_n;
        end

        eta_GP_dist  = 2*sqrt(beta_val) * sigma_sum_dist / precison_sum;
        eta_GP_local = 2*sqrt(beta_val) * sigma_sum_local / AgentQuantity;

        bound_distributed(k) = C * (eta_GP_dist + Delta_hat);
        bound_local(k)       = C * (eta_GP_local + Delta_hat);
        bound_exact(k)       = C * Delta_hat;
    end

    %% 9. Save results
    if ~exist(SaveFolderName, 'dir')
        mkdir(SaveFolderName);
    end
    save(fullfile(SaveFolderName, [SaveFileName, '.mat']), ...
        't_set', 'TrackingError_vector', 'CurrentMode', ...
        'bound_distributed', 'bound_local', 'bound_exact'); 
end