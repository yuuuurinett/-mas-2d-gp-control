clc; clear; close all;

%% 1. 配置仿真模式
% 5种masked GP聚合方法 + 原有对比基线
SimulationModes = {
    'poe',               % Masked PoE (Product of Experts)
    'gpoe',              % Masked gPoE (Generalized Product of Experts)
    'moe',               % Masked MoE (Mixture of Experts)
    'bcm',               % Masked BCM (Bayesian Committee Machine)
    'rbcm',              % Masked rBCM (Robust Bayesian Committee Machine)
    'local',             % Local GP (no cooperation, baseline)
    'exact'              % Exact dynamics (oracle upper bound)
};

SaveFolderName = fullfile('Result', 'Main_Simulation_Results');
if ~exist(SaveFolderName, 'dir'), mkdir(SaveFolderName); end

%% 2. 运行仿真
do_simulation = true; 
if do_simulation
    for ModeNr = 1:numel(SimulationModes)
        CurrentMode = SimulationModes{ModeNr};
        fprintf('\n========================================\n');
        fprintf('运行模式 [%d/%d]: %s\n', ModeNr, numel(SimulationModes), CurrentMode);
        fprintf('========================================\n');
        run_main_simulation_mode(CurrentMode, SaveFolderName, CurrentMode);
    end
end

%% 3. 加载所有结果
NumModes = numel(SimulationModes);

% 从第一个文件读取公共时间轴和误差上界
FirstFile = fullfile(SaveFolderName, [SimulationModes{1}, '.mat']);
temp_data = load(FirstFile, 't_set', 'bound_local', 'bound_distributed', 'bound_exact');
t_set          = temp_data.t_set;
bound_local    = temp_data.bound_local;
bound_distributed = temp_data.bound_distributed;
bound_exact    = temp_data.bound_exact;

% 读取所有模式的跟踪误差
TrackingError_matrix = zeros(NumModes, numel(t_set));
for ModeNr = 1:NumModes
    DataFile = fullfile(SaveFolderName, [SimulationModes{ModeNr}, '.mat']);
    d = load(DataFile, 'TrackingError_vector');
    TrackingError_matrix(ModeNr, :) = d.TrackingError_vector;
end

%% 4. 汇总表格：终端误差 & 平均误差
fprintf('\n');
fprintf('============================================================\n');
fprintf('  GP Aggregation Method Comparison (masked inducing-point)\n');
fprintf('============================================================\n');
fprintf('  %-20s  %12s\n', 'Method', 'Final ||e||');
fprintf('  %-20s  %12s\n', '------', '-----------');

% 只统计前5种masked方法
for ModeNr = 1:5
    err = TrackingError_matrix(ModeNr, :);
    fprintf('  %-20s  %12.4f\n', SimulationModes{ModeNr}, err(end));
end
fprintf('  %-20s  %12s\n', '------', '-----------');
for ModeNr = 6:NumModes
    err = TrackingError_matrix(ModeNr, :);
    fprintf('  %-20s  %12.4f\n', SimulationModes{ModeNr}, err(end));
end
fprintf('============================================================\n\n');

%% 5. 绘图
figure('Color','w','Position',[100 100 750 580]);
t_start = t_set(1);
t_end   = t_set(end);

% 线型/颜色配置：5种masked方法用实线，基线用虚线
colors = {[0.0  0.45 0.74],  ... % poe    - blue
          [0.85 0.33 0.10],  ... % gpoe   - orange
          [0.47 0.67 0.19],  ... % moe    - green
          [0.63 0.08 0.18],  ... % bcm    - dark red
          [0.93 0.69 0.13],  ... % rbcm   - yellow
          [0.5  0.5  0.5 ],  ... % local  - gray
          [0.0  0.0  0.0 ]};    % exact  - black
styles  = {'-', '-', '-', '-', '-', '--', ':'};
lw      = [1.8, 1.8, 1.8, 1.8, 1.8, 1.5, 1.5];

LegendNames = {'PoE (masked)', 'gPoE (masked)', 'MoE (masked)', ...
               'BCM (masked)', 'rBCM (masked)', 'Local', 'Exact'};

% 上图: ||e|| (log scale)
subplot(2,1,1);
hold on; grid on; box on;
set(gca, 'YScale','log', 'FontSize',11, 'FontName','Times New Roman');
for ModeNr = 1:NumModes
    plot(t_set, TrackingError_matrix(ModeNr,:), ...
        'Color', colors{ModeNr}, 'LineStyle', styles{ModeNr}, ...
        'LineWidth', lw(ModeNr));
end
ylabel('$\|e\|$', 'Interpreter','latex', 'FontSize',13);
xlim([t_start, t_end]);
legend(LegendNames, 'Location','northeast', 'FontSize',9, ...
       'NumColumns', 2);
title('Tracking Error: 5 Masked GP Aggregation Methods', ...
      'FontSize',12, 'FontName','Times New Roman');
set(gca, 'XTickLabel', []);

% 下图: 误差上界 v(t)
subplot(2,1,2);
hold on; grid on; box on;
set(gca, 'FontSize',11, 'FontName','Times New Roman');
plot(t_set, bound_local,       'k-',  'LineWidth',1.5, 'DisplayName','v(t) local bound');
plot(t_set, bound_distributed, 'r-',  'LineWidth',1.5, 'DisplayName','v(t) distributed bound');
plot(t_set, bound_exact,       'b--', 'LineWidth',1.5, 'DisplayName','v(t) exact bound');
ylabel('$v(t)$', 'Interpreter','latex', 'FontSize',13);
xlabel('$t$',    'Interpreter','latex', 'FontSize',13);
xlim([t_start, t_end]);
legend('Location','northeast', 'FontSize',10);

saveas(gcf, fullfile(SaveFolderName, 'Aggregation_Comparison.fig'));
saveas(gcf, fullfile(SaveFolderName, 'Aggregation_Comparison.png'));
fprintf('Plot saved to %s\n', SaveFolderName);