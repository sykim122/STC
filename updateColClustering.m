function [Cz] = updateColClustering(p, q, tilde_p, tilde_q, Cz, lambda)

% compute p(X|z)
pXz = p./repmat(sum(p,1), size(p,1), 1);

% compute q(Y|z)
qYz = q./repmat(sum(q,1), size(q,1), 1);

% compute tilde_p(X|tilde_z)
tilde_pXz = tilde_p./repmat(sum(tilde_p,1), size(tilde_p,1), 1);

for i = 1:size(tilde_p,1)
  tilde_pXtz(i,:) = accumarray(Cz, tilde_p(i,:)')';
end

% compute tilde_q(Y|tilde_z)
tilde_qYz = tilde_q./repmat(sum(tilde_q,1), size(tilde_q,1), 1);

for i = 1:size(tilde_q,1)
  tilde_qYtz(i,:) = accumarray(Cz, tilde_q(i,:)')';
end

% find Cz minimizing objective function
pz = sum(p,1); qz = sum(q,1);

for zc = 1:size(tilde_pXtz,2)
  for z = 1:size(pXz,2)
    temp_X(zc, z) = pz(z) * KLDiv(pXz(:,z)', tilde_pXtz(:,zc)');
    temp_Y(zc, z) = qz(z) * KLDiv(qYz(:,z)', tilde_qYtz(:,zc)');    
  end
end

temp = temp_X + lambda * temp_Y;

[mindist Cz] = min(temp);
Cz = Cz';

clearvars -except Cz 
