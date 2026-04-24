function main_BatchTest(mode, framework)

Modes_dac      = {'poe','gpoe','moe','bcm','rbcm'};
Modes_ac       = {'poe_ac','gpoe_ac','moe_ac','bcm_ac','rbcm_ac'};
Modes_baseline = {'local','exact'};

SaveFolder_Test     = fullfile('Result','Test_Point');
SaveFolder_Inducing = fullfile('Result','Inducing_Point');

if nargin == 0
    AllModes_inducing = [Modes_dac, Modes_ac, Modes_baseline];
    AllModes_test     = [Modes_dac, Modes_baseline];
    framework = 'both';
else
    AllModes_inducing = {mode};
    AllModes_test     = {mode};
end
framework = lower(framework);

%% 1. 运行仿真
do_simulation = true;
if do_simulation
    if ismember(framework, {'inducing','both'})
        fprintf('\n======== 诱导点聚合 ========\n');
        for m = 1:numel(AllModes_inducing)
            fprintf('[%d/%d] %s\n', m, numel(AllModes_inducing), ...
                AllModes_inducing{m});
            run_simulation_inducing_point(AllModes_inducing{m}, ...
                SaveFolder_Inducing, AllModes_inducing{m});
        end
    end

    if ismember(framework, {'test','both'})
        fprintf('\n======== 测试点聚合 ========\n');
        for m = 1:numel(AllModes_test)
            fprintf('[%d/%d] %s\n', m, numel(AllModes_test), ...
                AllModes_test{m});
            run_simulation_test_point(AllModes_test{m}, SaveFolder_Test, ...
                AllModes_test{m});
        end
    end
end

%% 2. 加载结果
% 诱导点
has_inducing = exist(fullfile(SaveFolder_Inducing, ...
    [AllModes_inducing{1},'.mat']),'file');
if has_inducing
    temp = load(fullfile(SaveFolder_Inducing, ...
        [AllModes_inducing{1},'.mat']),'t_set');
    t_set = temp.t_set;
    N_ind = numel(AllModes_inducing);
    Err_Inducing = zeros(N_ind, numel(t_set));
    for m = 1:N_ind
        d = load(fullfile(SaveFolder_Inducing, ...
            [AllModes_inducing{m},'.mat']),'TrackingError_vector');
        Err_Inducing(m,:) = d.TrackingError_vector;
    end
end

% 测试点
has_test = exist(fullfile(SaveFolder_Test, ...
    [AllModes_test{1},'.mat']),'file');
if has_test
    temp = load(fullfile(SaveFolder_Test, ...
        [AllModes_test{1},'.mat']),'t_set');
    t_set = temp.t_set;
    N_test = numel(AllModes_test);
    Err_Test = zeros(N_test, numel(t_set));
    for m = 1:N_test
        d = load(fullfile(SaveFolder_Test, ...
            [AllModes_test{m},'.mat']),'TrackingError_vector');
        Err_Test(m,:) = d.TrackingError_vector;
    end
end

%% 3. 打印表格
fprintf('\n%s\n', repmat('=',1,44));
if has_inducing
    fprintf('Inducing-Point:\n');
    fprintf('  %-12s  %16s\n','Method','Final ||e||');
    fprintf('  %-12s  %16s\n','------','-----------');
    for m = 1:N_ind
        fprintf('  %-12s  %16.4f\n', AllModes_inducing{m}, Err_Inducing(m,end));
    end
    fprintf('\n');
end
if has_test
    fprintf('Test-Point:\n');
    fprintf('  %-12s  %16s\n','Method','Final ||e||');
    fprintf('  %-12s  %16s\n','------','-----------');
    for m = 1:N_test
        fprintf('  %-12s  %16.4f\n', AllModes_test{m}, Err_Test(m,end));
    end
end
fprintf('%s\n\n', repmat('=',1,44));

%% 4. 绘图
t_start = t_set(1); t_end = t_set(end);

styles_dac      = {'b-','r-','g-','m-','y-'};
styles_ac       = {'b--','r--','g--','m--','y--'};
styles_baseline = {'k-','k:'};

N_dac  = numel(Modes_dac);
N_ac   = numel(Modes_ac);
N_base = numel(Modes_baseline);

% Figure 1: 诱导点 DAC
if has_inducing
    figure(1); clf;
    hold on; grid on; box on;
    set(gca,'YScale','log','FontSize',11,'FontName','Times New Roman');
    for m = 1:N_dac
        plot(t_set, Err_Inducing(m,:), styles_dac{m}, 'LineWidth',1.5, ...
            'DisplayName', Modes_dac{m});
    end
    for m = 1:N_base
        plot(t_set, Err_Inducing(N_dac+N_ac+m,:), styles_baseline{m}, ...
            'LineWidth',1.5, 'DisplayName', Modes_baseline{m});
    end
    ylabel('$\|e\|$','Interpreter','latex','FontSize',13);
    xlabel('$t$','Interpreter','latex','FontSize',13);
    title('Inducing-Point Aggregation (DAC)','FontSize',12, ...
        'FontName','Times New Roman');
    legend('Location','northeast','FontSize',9,'NumColumns',2);
    xlim([t_start, t_end]);
    saveas(gcf, fullfile('Result','Inducing_DAC_Comparison.png'));
 
end

% Figure 2: 诱导点 AC
if has_inducing
    figure(2); clf;
    hold on; grid on; box on;
    set(gca,'YScale','log','FontSize',11,'FontName','Times New Roman');
    for m = 1:N_ac
        plot(t_set, Err_Inducing(N_dac+m,:), styles_ac{m}, ...
            'LineWidth',1.5, 'DisplayName', Modes_ac{m});
    end
    for m = 1:N_base
        plot(t_set, Err_Inducing(N_dac+N_ac+m,:), styles_baseline{m}, ...
            'LineWidth',1.5, ...
            'DisplayName', Modes_baseline{m});
    end
    ylabel('$\|e\|$','Interpreter','latex','FontSize',13);
    xlabel('$t$','Interpreter','latex','FontSize',13);
    title('Inducing-Point Aggregation (AC)','FontSize',12, ...
        'FontName','Times New Roman');
    legend('Location','northeast','FontSize',9,'NumColumns',2);
    xlim([t_start, t_end]);
     saveas(gcf, fullfile('Result','Inducing_AC_Comparison.png'));
  
end

% Figure 3: 测试点 DAC
if has_test
    figure(3); clf;
    hold on; grid on; box on;
    set(gca,'YScale','log','FontSize',11,'FontName','Times New Roman');
    all_styles = [styles_dac, styles_baseline];
    for m = 1:N_test
        plot(t_set, Err_Test(m,:), all_styles{m}, 'LineWidth',1.5, ...
            'DisplayName', AllModes_test{m});
    end
    ylabel('$\|e\|$','Interpreter','latex','FontSize',13);
    xlabel('$t$','Interpreter','latex','FontSize',13);
    title('Test-Point Aggregation (DAC)','FontSize',12,'FontName','Times New Roman');
    legend('Location','northeast','FontSize',9,'NumColumns',2);
    xlim([t_start, t_end]);
   saveas(gcf, fullfile('Result','Test_Point_Comparison.png'));
 
end

fprintf('完成。\n');
end
