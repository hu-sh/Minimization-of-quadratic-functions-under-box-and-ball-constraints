function [x, lambda, info] = solve_qp_box_ball(Q, q, l, u, c, r, opts)
    n = length(q);

    % compute rho and an interior point x_tilde 
    [rho, x_tilde, rho_info] = find_margin_rho_(l,u,c,r, opts.rho_bisect_it);
    if rho == 0
        x = min(max(c, l), u);
        if norm(x - c) > r
            fprintf("Domain is empty");
        else
            fprintf("The only feasible point is %g", x);
        end
        lambda = -1;
        info = struct();
        return
    end
        

    % compute f_lb computing a lower bound on the ball only using convexity
    [f_lb, flb_info] = lower_bound_ball_only_(Q, q, c, r);

    % compute K
    f_tilde = x_tilde'*(Q*x_tilde) + q'*x_tilde;
    K = max(0, (f_tilde - f_lb) / rho);
    if K == 0, K = 1; end

    % target mu and complexity budget
    eps_target = opts.eps;
    mu_min = eps_target / (r^2);
    max_iter_total = min(1e+5, ceil(2 * K * r * sqrt(n) / eps_target)); 
    
    % precompute eigen-decomposition of 2Q once 
    Q2 = 2*Q;
    [V, D2] = eig(full(Q2));
    lam = diag(D2);

    % Nesterov initialization on Q1=[0,K]^{2n} 
    alpha0 = 0.5*K*ones(n,1);
    beta0  = 0.5*K*ones(n,1);
    xk = [alpha0; beta0]; 
    
    lo = zeros(2*n,1);
    hi = K*ones(2*n,1);

    % Continuation schedule for mu
    mu = max(mu_min, opts.mu_init_factor * mu_min);
    Lmu = 2 / mu;
    stage_id = 1;
    
    lambda0_stage = xk;
    Sk = zeros(2*n,1);
    k_stage = 0;

    % Warm-start for inner Newton on gamma
    gamma_ws = 0;

    % ==== traces ====
    info = struct();
    info.rho = rho;
    info.K = K;
    info.eps = eps_target;
    info.mu_min = mu_min;
    info.max_iter = max_iter_total;
    info.f_lb = f_lb;
    info.x_tilde = x_tilde;
    info.rho_info = rho_info;
    info.flb_info = flb_info;

    info.pg_norm = zeros(max_iter_total,1);
    info.x = zeros(max_iter_total,n);
    info.Fmu     = zeros(max_iter_total,1);
    info.mu_trace = zeros(max_iter_total,1);
    info.stage_id = zeros(max_iter_total,1);
    info.inner_newton_iters = zeros(max_iter_total,1);

    if opts.verbose
        fprintf('n=%d, eps=%g, mu_min=%g, K=%g, rho=%g, max_iter=%d\n', n, eps_target, mu_min, K, rho, max_iter_total);
        fprintf('mu-continuation: init_factor=%g, decay=%g, switch_ratio=%g, min_iter_per_mu=%d\n', ...
            opts.mu_init_factor, opts.mu_decay, opts.mu_switch_ratio, opts.min_iter_per_mu);
    end

    % outer loop 
    stag_counter = 0;
    win = opts.window_size;
    xmu_k = c;
    
    for k = 1:max_iter_total

        alpha = xk(1:n);
        beta  = xk(n+1:end);
        w = alpha - beta;

        % inner loop
        [xmu_k, inner_info] = inner_solve_ball_qp(V, lam, q, c, r, w, mu, opts.tol_inner_phi, opts.tol_inner_step, gamma_ws);
        info.x(k, :) = xmu_k;

        if isfield(inner_info,'gamma') && isfinite(inner_info.gamma)
            gamma_ws = max(0, inner_info.gamma);
        else
            gamma_ws = 0;
        end

        % gradient of F_mu
        g = [-l + xmu_k;  u - xmu_k];

        % projected gradient norm
        x_proj = proj_box_(xk - g, lo, hi);
        pg = norm(xk - x_proj);

        % info
        Fmu_val = (-l')*alpha + (u')*beta + gmu_val_(Q, q, c, w, mu, xmu_k);

        info.pg_norm(k) = pg;
        info.Fmu(k)     = Fmu_val;
        info.mu_trace(k) = mu;
        info.stage_id(k) = stage_id;
        info.inner_newton_iters(k) = inner_info.newton_iters;
        

        % Nesterov update
        yk = proj_box_(xk - (1/Lmu)*g, lo, hi);

        ai = (k_stage+1)/2;
        Sk = Sk + ai * g;
        zk = proj_box_(lambda0_stage - (1/Lmu)*Sk, lo, hi);

        tau = 2/(k_stage+3);
        xk = tau*zk + (1-tau)*yk;
        k_stage = k_stage + 1;

        if opts.verbose && (k==1 || mod(k, max(1, floor(max_iter_total/10)))==0)
            fprintf('iter %5d | stage=%d | mu=%9.3e | pg=%9.3e | Fmu=%9.3e | inner it=%d\n', ...
                k, stage_id, mu, pg, Fmu_val, inner_info.newton_iters);
        end

        % early stop check
        if (mu <= mu_min*(1+1e-12)) && (pg < opts.tol_pg)
            info.stop_reason = 'early_stop_projected_gradient';
            info.iters = k;
            break;
        end

        stage_target = opts.optimality_factor * mu; 
        switch_ = false;
        if pg <= stage_target
            switch_ = true;
        end

        % stagnation check
        if ~switch_ && k_stage > opts.after_it && k_stage > 2*win
            curr_idx = (k - win + 1) : k;
            mean_curr = mean(info.pg_norm(curr_idx));
            prev_idx = (k - 2*win + 1) : (k - win);
            mean_prev = mean(info.pg_norm(prev_idx));
            rel_improv = (mean_prev - mean_curr) / (mean_prev + 1e-16);
            if rel_improv < opts.stag_tol
                stag_counter = stag_counter + 1; 
            else
                stag_counter = 0; 
            end
            if stag_counter >= opts.patience
                switch_ = true;
            end
        else
             stag_counter = 0; 
        end

        % change of stage
        if (mu > mu_min*(1+1e-12)) && (k_stage >= opts.min_iter_per_mu) && switch_
            mu_new = max(mu_min, mu/opts.mu_decay);
            if mu_new < mu*(1-1e-12)
                mu = mu_new;
                Lmu = 2/mu;
                stage_id = stage_id + 1;
                lambda0_stage = xk;
                Sk = zeros(2*n,1);
                k_stage = 0;
                stag_counter = 0;
                if opts.verbose
                    fprintf(': mu update: stage=%d, mu=%g\n', stage_id, mu);
                end
            else
                mu = mu_min;
                Lmu = 2/mu;
            end
        end
    end

    if ~isfield(info,'iters')
        info.stop_reason = 'reached_max_iter_complexity_budget';
        info.iters = max_iter_total;
    end

    % trim vector info
    fields = {'pg_norm','Fmu','mu_trace','stage_id','inner_newton_iters'};
    for fi = 1:numel(fields)
        f = fields{fi};
        info.(f) = info.(f)(1:info.iters);
    end
    info.x = info.x(1:info.iters, :);

    % return final solution
    alpha = xk(1:n); beta = xk(n+1:end);
    w = alpha - beta;
    [x, inner_info_final] = inner_solve_ball_qp(V, lam, q, c, r, w, mu, opts.tol_inner_phi, opts.tol_inner_step, gamma_ws);
    info.inner_gamma_final = inner_info_final.gamma;
    info.mu_final = mu;
    info.stage_final = stage_id;

    lambda = struct('alpha',alpha,'beta',beta);
end

function val = gmu_val_(Q, q, c, w, mu, x)
% evaluation of the smoothed term g_mu
    phib = x'*(Q*x) + q'*x;
    val = w'*x - phib - (mu/2)*norm(x-c)^2;
end