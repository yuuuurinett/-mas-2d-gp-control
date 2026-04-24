function run_simulation_test_point(CurrentMode, SaveFolderName, SaveFileName)
rng(0);
%% 1. Topology
AgentQuantity = 6;
AgentConnectionConfigurationSet = {
    {1,2,'Directed',0.5};
    {1,6,'Directed',0.5};
    {2,3,'Directed',0.5};
    {2,4,'Directed',0.5};
    {3,2,'Directed',0.5}; 
    {3,1,'Directed',0.5};
    {4,3,'Directed',0.5}; 
    {4,5,'Directed',0.5};
    {5,6,'Directed',0.5}; 
    {5,4,'Directed',0.5};
    {6,5,'Directed',0.5}; 
    {6,1,'Directed',0.5}
};
LeaderQuantity = 1;
AgentLeaderConnectionConfigurationSet = {
    {'Leader',1,'Agent',1,1}; {'Leader',1,'Agent',4,1}
};
MultiAgentSystem = MultiAgentSystem_Class(AgentQuantity, LeaderQuantity);
MultiAgentSystem.Agent_Topology.addConnectionSet(AgentConnectionConfigurationSet);
MultiAgentSystem.Agent_Topology.check_Connectivity(true);
MultiAgentSystem.Agent_Leader_Topology.addConnectionSet(AgentLeaderConnectionConfigurationSet);
MultiAgentSystem.get_ExtendedTopology;
L = MultiAgentSystem.Agent_Topology.LaplacianMatrix;
B = MultiAgentSystem.Agent_Leader_Topology.ConnectionMatrix;

%% 2. Time
t_start = 0; t_end = 4; t_step = 0.01;
t_set = t_start:t_step:t_end;
SystemOrder = 2; q_dim = 2; x_dim = q_dim * SystemOrder;
m1 = 1; m2 = 1; L1 = 1; L2 = 1; g = 9.8;

%% 3. Reference Trajectory
x_trajectory_set = nan(x_dim, numel(t_set));
x_trajectory_set(1,:) = sin(0.5*t_set);
x_trajectory_set(2,:) = cos(0.5*t_set);
x_trajectory_set(3,:) = 0.5*cos(0.5*t_set);
x_trajectory_set(4,:) = -0.5*sin(0.5*t_set);

%% 4. Local GPs
SigmaF = 1; SigmaL = 0.5*ones(x_dim,1);
GP_tau = 1e-8; GP_delta = 0.1; y_dim = q_dim;
DomainScale = 1.5;
Local_X_min = DomainScale*[-1,-1,-1,-1;-1,-1,0,0;0,0,0,0;0,0,-1,-1;-1,0,-1,0;0,-1,0,-1];
Local_X_max = DomainScale*[0,0,0,0;0,0,1,1;1,1,1,1;1,1,0,0;0,1,0,1;1,0,1,0];
OfflineDataQuantity = 100;
SigmaN = 0.1;

LocalGP_set = cell(AgentQuantity,1);
for n = 1:AgentQuantity
    LocalGP_set{n} = LocalGP_MultiOutput(x_dim, y_dim, 100, SigmaN, SigmaF, SigmaL);
    Data_mu = (Local_X_max(n,:) + Local_X_min(n,:)) / 2;
    Data_Le = (Local_X_max(n,:) - Local_X_min(n,:)) / 2;
    X_in = 2*(rand(x_dim,OfflineDataQuantity)-0.5) .* repmat(Data_Le',[1,OfflineDataQuantity]) + Data_mu';
    Y_in = zeros(y_dim, OfflineDataQuantity);
    for d = 1:OfflineDataQuantity
        Y_in(:,d) = Manipulator_2D_2DoF_UnknownDynamics(X_in(:,d));
    end
    Y_in = Y_in + SigmaN * randn(size(Y_in));
    LocalGP_set{n}.add_Alldata(X_in, Y_in);
    LocalGP_set{n}.tau = GP_tau; LocalGP_set{n}.delta = GP_delta;
    LocalGP_set{n}.xMax = max(X_in,[],2); LocalGP_set{n}.xMin = min(X_in,[],2);
end

%% 5. Setup
rng(0);
AgentInitialState_matrix = rand(4, AgentQuantity) - 0.5;
lambda_set = [7/4; 1];
Kappa_C = 100;
AdjMatrix  = MultiAgentSystem.Agent_Topology.AdjacencyMatrix;
NeighbourSet = MultiAgentSystem.Agent_Topology.NeighbourSet;
TrackingError_vector = zeros(1, numel(t_set));

%% 6. Mode init
Kappa_P = 1000;
switch lower(CurrentMode)
    case {'poe','gpoe','moe','bcm'}
        Zeta_vector = zeros(4, AgentQuantity);
    case 'rbcm'
        Zeta_vector = zeros(6, AgentQuantity);
    otherwise
        Zeta_vector = zeros(4, AgentQuantity);
end

%% 7. Control Loop
tic; 
t_gp = 0; 
t_ode = 0;
AgentState_matrix = AgentInitialState_matrix;

for k = 1:numel(t_set)
    CurrentTime = t_set(k);
    LeaderState_vector = x_trajectory_set(:, k);
    TrackingError_vector(1,k) = norm(AgentState_matrix(:) - repmat(LeaderState_vector, AgentQuantity, 1));
    if k == numel(t_set), break; end

    Phi_Xi_vector = zeros(q_dim, AgentQuantity);
    tic_gp = tic;

    switch lower(CurrentMode)
        case 'poe'
            [Phi_Xi_vector, Zeta_vector] = gp_test_poe( ...
                AgentState_matrix, LocalGP_set, L, Kappa_P, AgentQuantity, Zeta_vector, t_step);
        case 'gpoe'
            [Phi_Xi_vector, Zeta_vector] = gp_test_gpoe( ...
                AgentState_matrix, LocalGP_set, L, Kappa_P, AgentQuantity, Zeta_vector, t_step);
        case 'moe'
            [Phi_Xi_vector, Zeta_vector] = gp_test_moe( ...
                AgentState_matrix, LocalGP_set, L, Kappa_P, AgentQuantity, Zeta_vector, t_step);
        case 'bcm'
            [Phi_Xi_vector, Zeta_vector] = gp_test_bcm( ...
                AgentState_matrix, LocalGP_set, L, Kappa_P, AgentQuantity, Zeta_vector, t_step);
        case 'rbcm'
            [Phi_Xi_vector, Zeta_vector] = gp_test_rbcm( ...
                AgentState_matrix, LocalGP_set, L, Kappa_P, AgentQuantity, Zeta_vector, t_step);
        case 'local'
            for n = 1:AgentQuantity
                [mu_n,~] = LocalGP_set{n}.predict(AgentState_matrix(:,n));
                Phi_Xi_vector(:,n) = mu_n;
            end
        case 'exact'
            for n = 1:AgentQuantity
                Phi_Xi_vector(:,n) = Manipulator_2D_2DoF_UnknownDynamics(AgentState_matrix(:,n));
            end
    end
    t_gp = t_gp + toc(tic_gp);

    MultiAgent_Dynamics_Handle = @(t,x) multi_agent_dynamics( ...
        t, x, Phi_Xi_vector, AdjMatrix, NeighbourSet, B, lambda_set, ...
        Kappa_C, AgentQuantity, SystemOrder, q_dim, L1, L2, m1, m2);

    tic_ode = tic;
    opts = odeset('RelTol',1e-6,'AbsTol',1e-6);
    [~, x_output] = ode45(MultiAgent_Dynamics_Handle, ...
        [CurrentTime, CurrentTime+t_step], AgentState_matrix(:), opts);
    AgentState_matrix = reshape(x_output(end,:)', x_dim, AgentQuantity);
    t_ode = t_ode + toc(tic_ode);
    %fprintf('t = %6.4f\n', CurrentTime);
end
fprintf('Mode: %s done, total=%.2fs, GP=%.2fs, ODE=%.2fs\n', CurrentMode, toc, t_gp, t_ode);

%% 8. Save
if ~exist(SaveFolderName,'dir'), mkdir(SaveFolderName); end
save(fullfile(SaveFolderName,[SaveFileName,'.mat']), ...
    't_set','TrackingError_vector','CurrentMode');
end