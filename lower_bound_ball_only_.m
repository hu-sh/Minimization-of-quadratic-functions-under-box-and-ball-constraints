function [f_lb, info] = lower_bound_ball_only_(Q, q, c, r)
% Lower bound for a convex quadratic function on a sphere
Qc   = Q*c;
fc   = c' * Qc + q' * c;
g    = 2*Qc + q;
gn   = norm(g, 2);

f_lb = fc - r * gn;

info = struct();
info.fc = fc;
info.grad_norm = gn;
end
