function [rho, x_rho, info] = find_margin_rho_(l,u,c,r, maxit)
% Find rho>0 such that projection of c onto [l+rho,u-rho] lies in ball
    n = length(c);
    rho_max = 0.5*min(u-l);
    if rho_max <= 0
        rho = 0; x_rho = min(max(c,l),u);
        info = struct('rho_max',rho_max,'iters',0,'status','rho_max_nonpositive');
        return;
    end

    % quick test at rho=0
    x0 = min(max(c,l),u);
    if norm(x0-c) > r
        % if rho=0 infeasible then intersection is empty or just one point
        rho = 0; x_rho = x0;
        info = struct('rho_max',rho_max,'iters',0,'status','no_feasible_margin_found');
        return;
    end

    
    lo = 0;
    hi = rho_max;

    rho = 0;
    x_rho = x0;

    for it=1:maxit
        mid = 0.5*(lo+hi);
        xmid = proj_box_(c, l + mid*ones(n,1), u - mid*ones(n,1));
        feas = (norm(xmid - c) <= r);

        if feas
            rho = mid;
            x_rho = xmid;
            lo = mid;
        else
            hi = mid;
        end

        % early stop once we have a nonzero feasible margin
        if rho > 0 && (hi-lo) <= 1e-12*rho_max
            break;
        end
    end

    info = struct('rho_max',rho_max,'iters',it,'status','ok');
end
