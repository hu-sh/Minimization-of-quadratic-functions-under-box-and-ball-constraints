global rank_factor 
rank_factor = 0.2;

rng(5);
ns = [500, 1000, 3000];
results = cell(length(ns), 10);
opts = get_standard_opts();

for i = 1:length(ns)
    n = ns(i);
    fprintf('Testing n = %d\n', n);

    % data generation
    A = randn(n, rank_factor*n);
    Q = A*A'; Q = 0.5*(Q+Q');
    q = randn(n, 1);
    l = -1.5*ones(n, 1); u = 1.5*ones(n, 1);
    c = 0.2*randn(n, 1); r = 1.0;

    k_Q = cond(full(Q));

    % proposed method
    t_start = tic;
    opts.mu_init_factor = opts.mu_init_factor*log(k_Q);

    [x_dual, ~, info] = solve_qp_box_ball(Q, q, l, u, c, r, opts);
    t_dual = toc(t_start);

    % fmincon
    [x_fmc, f_fmc, t_fmc] = run_fmincon(Q, q, l, u, c, r);


    f_dual = x_dual'*(Q*x_dual) + q'*x_dual;
    diff_f= f_dual - f_fmc;
    rel_diff_x = norm(x_dual - x_fmc) / max(1, norm(x_fmc));

    % Violations
    v_box_d = max([0; l - x_dual; x_dual - u], [], 'all');
    v_ball_d = max(0, norm(x_dual - c) - r);
    v_box_f  = max([0; l - x_fmc; x_fmc - u], [], 'all');
    v_ball_f = max(0, norm(x_fmc - c) - r);

    results(i, :) = {n, k_Q, t_dual, t_fmc, diff_f, rel_diff_x, v_box_d, v_ball_d, v_box_f, v_ball_f};

    generate_plots(info, n);

    % Reset mu factor
    opts.mu_init_factor = opts.mu_init_factor/log(k_Q);
end
print_summary_table(results);


function opts = get_standard_opts()
    opts = struct();
    opts.eps = 1e-15;
    opts.tol_pg = 1e-15;
    opts.tol_inner_phi = 1e-15;
    opts.tol_inner_step = 1e-15;
    opts.rho_bisect_it = 1e6;
    opts.verbose = false;

    opts.mu_init_factor = 1/opts.eps;
    opts.mu_decay = 10;
    opts.min_iter_per_mu = 10;

    opts.optimality_factor = 0.5;

    opts.after_it = 1000;
    opts.stag_tol = 0.001;
    opts.window_size = 30;
    opts.patience = 30;
end

function [x, f, t] = run_fmincon(Q, q, l, u, c, r)
    x0 = min(max(c, l), u); 

    t_limit = 600; 
    t_start = tic;

    stop = @(x, optimValues, state) check_timeout(optimValues, state, t_start, t_limit);

    fmc_opts = optimoptions('fmincon', ...
    'Algorithm', 'interior-point', ...
    'SpecifyObjectiveGradient', true, ...
    'SpecifyConstraintGradient', true, ... 
    'Display', 'off', ... 
    'ConstraintTolerance', 1e-12, ...
    'OptimalityTolerance', 1e-15, ...
    'StepTolerance', 1e-10, ...
    'MaxIterations', 5000, ...
    'OutputFcn', stop, ...
    'MaxFunctionEvaluations', 2e6);            
    
    obj = @(x) deal(x'*(Q*x) + q'*x, 2*Q*x + q);
    nonlcon = @(x) ball_constraint(x, c, r);
    
    tic;
    [x, f, ~] = fmincon(obj, x0, [], [], [], [], l, u, nonlcon, fmc_opts);
    t = toc(t_start);
end

function stop = check_timeout(~, ~, startTime, limit)
    stop = false;
    if toc(startTime) > limit
        stop = true;
    end
end

function generate_plots(info, n)
    fig = figure('Name', sprintf('Analysis n=%d', n), 'Color', 'w');
    
    % safety eps
    safe_eps = 1e-16;
    
    mu_plot = info.mu_trace + safe_eps;
    pg_plot = info.pg_norm + safe_eps;
    
    % F_mu 
    F_val = info.Fmu;
    F_min = min(F_val);
    F_plot = (F_val - F_min) + safe_eps;
    
    % |x_k+1 - x_k|
    dist_plot = [];
    iters = length(pg_plot);
    
    X = info.x;
            
    if ~isempty(X)
        dx = diff(X, 1, 1);
        dist_vals = sqrt(sum(dx.^2, 2));
        dist_plot = dist_vals + safe_eps; 
    end

    
    % identification of start of stages
    stage_starts = [];
    if isfield(info, 'stage_id')
        stage_starts = find(diff(info.stage_id) > 0) + 1;
    end

    % mu steps
    ax1 = subplot(2,2,1);
    semilogy(1:length(mu_plot), mu_plot, 'LineWidth', 1.2);
    hold on;
    add_markers(stage_starts, mu_plot);
    title('\mu Steps'); grid on; ylabel('\mu'); xlabel('Iter');
    set(ax1, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
    dezoom(mu_plot);

    % projected gradient norm
    ax2 = subplot(2,2,2);
    semilogy(1:length(pg_plot), pg_plot, 'LineWidth', 1.2);
    hold on;
    add_markers(stage_starts, pg_plot);
    title('PG Norm'); grid on; ylabel('||pg||'); xlabel('Iter');
    set(ax2, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
    dezoom(pg_plot);

    % |F_mu - F_mu_min + 1e16|
    ax3 = subplot(2,2,3);
    semilogy(1:length(F_plot), F_plot, 'LineWidth', 1.2);
    hold on;
    add_markers(stage_starts, F_plot);
    title('Smoothed Dual Gap (F_\mu - F_{min})'); grid on; ylabel('F_\mu Gap'); xlabel('Iter');
    set(ax3, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
    dezoom(F_plot);
    
    % |x_k+1 - x_k| +1e16
    ax4 = subplot(2,2,4);
    if ~isempty(dist_plot)
        semilogy(2:iters, dist_plot, 'LineWidth', 1.2);
        hold on;
        
        valid_starts = stage_starts(stage_starts > 1);
        if ~isempty(valid_starts)
            vals = dist_plot(valid_starts - 1);
            plot(valid_starts, vals, 'ro', 'MarkerSize', 3, 'MarkerFaceColor', 'r');
        end
        
        title('Step ||x_k - x_{k-1}||'); grid on; ylabel('||dx||'); xlabel('Iter');
        dezoom(dist_plot);
    end
    set(ax4, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');

    global rank_factor;
    if rank_factor < 1
        filename_base = sprintf('%d_low_rank', n);
    else 
        filename_base = sprintf('%d_full_rank', n);
    end

    savefig(fig, fullfile(pwd, [filename_base '.fig']));
    try
        exportgraphics(fig, fullfile(pwd, [filename_base '.pdf']), 'ContentType', 'vector');
    catch
        print(fig, fullfile(pwd, [filename_base '.pdf']), '-dpdf', '-BestFit');
    end
end

function add_markers(idxs, y_data)
    if isempty(idxs), return; end
    
    idxs = idxs(idxs <= length(y_data));
    if isempty(idxs), return; end
    
    vals = y_data(idxs);
    plot(idxs, vals, 'ro', 'MarkerSize', 3, 'MarkerFaceColor', 'r');
end

function dezoom(data)
    min_val = min(data);
    max_val = max(data);
    
    if min_val >= max_val
        min_val = min_val * 0.1;
        max_val = max(max_val * 10, realmin);
    end

    ylim([min_val * 0.5, max_val * 5]);
    xlim([1, length(data) * 1.05]);
end


function print_summary_table(res)
    fprintf('%-5s | %-8s | %-7s | %-7s | %-9s | %-7s | %-7s | %-7s | %-7s | %-7s\n', ...
        'n', 'k', 't_dual', 't_fmin', 'f_d-f_f', 'Rel diff x', 'VBoxD', 'VBallD', 'VBoxF', 'VBallF');
    fprintf('%s\n', repmat('-', 1, 102));
    
    for i = 1:size(res,1)
        fprintf('%-5d | %-8.1e | %-7.3f | %-7.3f | %-+9.1e | %-7.1e | %-7.1e | %-7.1e | %-7.1e | %-7.1e\n', ...
            res{i,1}, res{i,2}, res{i,3}, res{i,4}, res{i,5}, res{i,6}, res{i,7}, res{i,8}, res{i,9}, res{i,10});
    end
end

function [cineq, ceq, gineq, geq] = ball_constraint(x, c, r)
    dist = x - c;
    cineq = dist' * dist - r^2;
    ceq = [];
    if nargout > 2
        gineq = 2 * dist; 
        geq = [];
    end
end