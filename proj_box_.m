function y = proj_box_(x, l, u)
    y = min(max(x, l), u);
end
