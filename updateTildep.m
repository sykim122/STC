function [tilde_p cluster_p] = updateTildep(p, Cx, Cz)

% joint probability between row and column clustering function, p(tilde_x, tilde_y)
[i j] = meshgrid(Cx, Cz);
subs = [i(:) j(:)];
val = p';
cluster_p = accumarray(subs, val(:));

% joint probability of X and Z with respect to co-clusters (Cx, Cz), tilde_p(X,Z)
[nrow ncol] = size(p);
tilde_p = zeros(nrow, ncol);
k = 1;
for i = 1:nrow
  for j = 1:ncol
    rc = subs(k, :);
    r = rc(1); c = rc(2);
    k = k+1;
    tilde_p(i,j) = cluster_p(r,c) * sum(p(i,:)) / sum(cluster_p(r,:)) * sum(p(:,j)) / sum(cluster_p(:,c));
  end
end

clearvars -except tilde_p cluster_p
