function run_kin40k_comparison()
rng(0);

%% 1. 加载数据
X_train = load('kin40k_train_data.asc');
Y_train = load('kin40k_train_labels.asc');
X_test  = load('kin40k_test_data.asc');
Y_test  = load('kin40k_test_labels.asc');

X_train = X_train'; Y_train = Y_train';
X_test  = X_test';  Y_test  = Y_test';

x_dim   = size(X_train, 1);
y_dim   = 1;
N_train = size(X_train, 2);

fprintf('Training: %d x %d, Test: %d x %d\n', x_dim, N_train, x_dim, size(X_test,2));

%% 2. 划分数据
AgentQuantity   = 6;
SigmaF          = 1;
SigmaL          = 0.5 * ones(x_dim, 1);
SigmaN          = 0.1;
MaxDataQuantity = ceil(N_train / AgentQuantity);

idx     = randperm(N_train);
X_train = X_train(:, idx);
Y_train = Y_train(:, idx);

LocalGP_set = cell(AgentQuantity, 1);
for n = 1:AgentQuantity
    start_idx = (n-1) * MaxDataQuantity + 1;
    end_idx   = min(n * MaxDataQuantity, N_train);
    X_n = X_train(:, start_idx:end_idx);
    Y_n = Y_train(:, start_idx:end_idx);
    LocalGP_set{n} = LocalGP_MultiOutput(x_dim, y_dim, MaxDataQuantity, SigmaN, SigmaF, SigmaL);
    LocalGP_set{n}.add_Alldata(X_n, Y_n);
    LocalGP_set{n}.xMax = max(X_n, [], 2);
    LocalGP_set{n}.xMin = min(X_n, [], 2);
    fprintf('Agent %d: %d samples\n', n, end_idx - start_idx + 1);
end

%% 3. 评估测试点
N_eval    = 10000;  
X_eval    = X_test(:, 1:N_eval);
Y_eval    = Y_test(:, 1:N_eval);
prior_var = SigmaF^2;

agg_methods = {'poe','gpoe','moe','bcm','rbcm'};
N_methods   = numel(agg_methods);

mu_agg  = zeros(N_methods, N_eval);
var_agg = zeros(N_methods, N_eval);

% local: 每个agent单独预测
mu_local_all  = zeros(AgentQuantity, N_eval);
var_local_all = zeros(AgentQuantity, N_eval);

for pt = 1:N_eval
    x_star = X_eval(:, pt);

    mu_n  = zeros(AgentQuantity, 1);
    var_n = zeros(AgentQuantity, 1);
    for n = 1:AgentQuantity
        [mu_i, var_i]      = LocalGP_set{n}.predict(x_star);
        mu_n(n)            = mu_i(1);
        var_n(n)           = var_i(1);
        mu_local_all(n,pt)  = mu_i(1);
        var_local_all(n,pt) = var_i(1);
    end

    for mi = 1:N_methods
        switch agg_methods{mi}
            case 'poe'
                prec = sum(1 ./ var_n);
                mu_agg(mi,pt)  = sum(mu_n ./ var_n) / prec;
                var_agg(mi,pt) = 1 / prec;
            case 'gpoe'
                beta = max(eps, 0.5*(log(prior_var) - log(var_n)));
                prec = sum(beta ./ var_n);
                mu_agg(mi,pt)  = sum(beta .* mu_n ./ var_n) / prec;
                var_agg(mi,pt) = 1 / prec;
            case 'moe'
                mu_agg(mi,pt)  = mean(mu_n);
                var_agg(mi,pt) = mean(var_n + mu_n.^2) - mean(mu_n)^2;
            case 'bcm'
                prec = sum(1./var_n) + (1-AgentQuantity)/prior_var;
                mu_agg(mi,pt)  = sum(mu_n./var_n) / prec;
                var_agg(mi,pt) = 1 / prec;
            case 'rbcm'
                beta = max(eps, 0.5*(log(prior_var) - log(var_n)));
                prec = max(sum(beta./var_n) + (1-sum(beta))/prior_var, 1e-6);
                mu_agg(mi,pt)  = sum(beta.*mu_n./var_n) / prec;
                var_agg(mi,pt) = 1 / prec;
        end
    end

    if mod(pt,100) == 0
        fprintf('Evaluated %d/%d\n', pt, N_eval);
    end
end

%% 4. 计算指标
Y_var_baseline = var(Y_eval);

fprintf('\n%s\n', repmat('=',1,52));
fprintf('  %-10s  %16s  %16s\n','Method','SMSE','MNLP');
fprintf('  %-10s  %16s  %16s\n','------','----','----');

smse_all = zeros(N_methods+1, 1);
mnlp_all = zeros(N_methods+1, 1);

% 聚合方法
for mi = 1:N_methods
    err  = Y_eval - mu_agg(mi,:);
    smse = mean(err.^2) / Y_var_baseline;
    mnlp = 0.5 * mean(log(2*pi*var_agg(mi,:)) + err.^2./var_agg(mi,:));
    smse_all(mi) = smse;
    mnlp_all(mi) = mnlp;
    fprintf('  %-10s  %16.4f  %16.4f\n', agg_methods{mi}, smse, mnlp);
end

% Local baseline: mean ± std across agents
smse_agents = zeros(AgentQuantity, 1);
mnlp_agents = zeros(AgentQuantity, 1);
for n = 1:AgentQuantity
    err  = Y_eval - mu_local_all(n,:);
    smse_agents(n) = mean(err.^2) / Y_var_baseline;
    mnlp_agents(n) = 0.5 * mean(log(2*pi*var_local_all(n,:)) + err.^2./var_local_all(n,:));
end
smse_all(end) = mean(smse_agents);
mnlp_all(end) = mean(mnlp_agents);

fprintf('  %-10s  %12.4f±%.4f  %12.4f±%.4f\n', 'local', ...
    mean(smse_agents), std(smse_agents), ...
    mean(mnlp_agents), std(mnlp_agents));
fprintf('%s\n\n', repmat('=',1,52));

%% 5. 绘图
if ~exist('Result', 'dir'), mkdir('Result'); end
all_labels = [agg_methods, {'local'}];

figure('Color','w');
bar(smse_all);
set(gca,'XTickLabel',all_labels,'FontSize',11,'FontName','Times New Roman');
ylabel('SMSE','FontSize',13);
title('KIN40K: SMSE Comparison','FontSize',12,'FontName','Times New Roman');
grid on;
saveas(gcf, fullfile('Result','KIN40K_SMSE.png'));

figure('Color','w');
bar(mnlp_all);
set(gca,'XTickLabel',all_labels,'FontSize',11,'FontName','Times New Roman');
ylabel('MNLP','FontSize',13);
title('KIN40K: MNLP Comparison','FontSize',12,'FontName','Times New Roman');
grid on;
saveas(gcf, fullfile('Result','KIN40K_MNLP.png'));

fprintf('完成。\n');
end