function p = jointprob_Z(X, conf)

nX = size(X, 1);
nY = conf.numWords;

for i = 1:nX
  p(i,:) = accumarray(X(i,:)', 1, [nY 1])';
end

p = p / (size(X(:),1) - sum(X(:)==0));

% normalize p to [0 1]
p = (p-min(p(:)))/(max(p(:))-min(p(:)));

clearvars -except p
