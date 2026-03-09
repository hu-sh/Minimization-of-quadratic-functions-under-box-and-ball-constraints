function [x, info] = inner_solve_ball_qp(V, lam, q, c, r, w, mu, tol_phi, tol_step, gamma0)

    qtilde = q - w - mu*c;

    % eigenvalues of Qtilde
    lam = lam + mu;
    lam = max(lam, 1e-15);

    % candidate interior point (gamma=0)
    x0 =  V * ((V' * (-qtilde)) ./ lam); 
    if norm(x0 - c) <= r
        x = x0;
        info = struct('case','interior','gamma',0,'newton_iters',0);
        return;
    end

    % uhat = V'*(Qtilde*c + qtilde)
    Vtc = V' * c;
    Vtq = V' * qtilde;
    uhat = lam .* Vtc + Vtq;
    
    gamma = max(0, gamma0); 
    newton_iters = 0;
    max_iter = 200;

    for it = 1:max_iter
        newton_iters = it;
        
        denom = lam + 2*gamma;
        phi = sum((uhat.^2) ./ (denom.^2)) - r^2;
        
        if abs(phi) <= tol_phi
            break;
        end

        % phi derivative
        dphi = -4*sum((uhat.^2) ./ (denom.^3));
        
        % Newton step
        step = -phi/dphi;
        gamma_new = gamma + step;

        if abs(gamma_new - gamma) <= tol_step*(1+gamma)
            gamma = gamma_new;
            break;
        end
        
        gamma = max(0, gamma_new);
    end

    
    % reconstruction of x
    denom = lam + 2*gamma;
    x = c - V*(uhat ./ denom);

    info = struct('case','boundary','gamma',gamma,'newton_iters',newton_iters);
end
