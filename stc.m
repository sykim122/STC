
conf.dataDir = '../Data/aux';
conf.targetDir = '../Data/target';
conf.per_target = 0.1; % percent of the target data
conf.numData = 760; % total 2940 / batos 760 / unjeongne 1481
conf.phowOpts = {'Step', 3, 'Color', 'gray'} ; % gray, hsv, rgb, opponent

conf.numWords = 500;

nrowcluster = 3;
ncolcluster = 20;

iter = 10;
lambda = 0.5;

%fid_p = fopen(fullfile(resultdir, 'p_clustering.txt'), 'w');
%fid_q = fopen(fullfile(resultdir, 'q_clustering.txt'), 'w');

% load target data(X), auxiliary data(Y)
[imX imY] = load_data(conf);

% get feature space(Z)
X = getFeature(imX, conf);
Y = getFeature(imY, conf);

% joint probability p(X,Z), q(Y,Z)
p = jointprob_Z(X, conf);
q = jointprob_Z(Y, conf);

% initialize clustering functions Cx, Cy, Cz;
rng(1);
[Cx c] = kmeans(p, nrowcluster);
[Cy c] = kmeans(q, nrowcluster);
[Cz c] = kmeans(p', ncolcluster); % p or q??

% initialize tilde_p(X,Z) and tilde_q(Y,Z)
[tilde_p cluster_p] = updateTildep(p, Cx, Cz);
[tilde_q cluster_q] = updateTildep(q, Cy, Cz);

for t = 1:iter
  % update Cx(X) based on p, tilde_p
  Cx = updateRowClustering(p, tilde_p, Cx);

  % update Cy(Y) based on q, tilde_q
  Cy = updateRowClustering(q, tilde_q, Cy);

  % update Cz(Z) based on p, q, tilde_p, tilde_q
  Cz = updateColClustering(p, q, tilde_p, tilde_q, Cz, lambda);

  % update tilde_p and tilde_q based on p(X,Z), Cx, Cz and q(Y,Z), Cy, Cz
  [tilde_p cluster_p] = updateTildep(p, Cx, Cz);
  [tilde_q cluster_q] = updateTildep(q, Cy, Cz);
  
end

disp('tilde_p(X,Z)')
tilde_p

disp('p(tilde_X, tilde_Z)')
cluster_p

disp('clustering on X')
Cx = Cx'

resultdir = fullfile('result', num2str(lambda));
for c = 1:nrowcluster
  dir = fullfile(resultdir, num2str(c));
  if ~exist(dir, 'dir') mkdir(dir); end 
end

for i = 1:length(Cx)
  copyfile(imX{i}, fullfile(resultdir, num2str(Cx(i)), sprintf('%d.jpg', i)));
end

disp('file copy complete')

clear
%fclose(fid_p);
%fclose(fid_q);
