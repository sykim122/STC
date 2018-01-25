function [vocab kdtree] = trainvocab(conf, desc)

%MODEL = model;

%d = cell(1, length(selTrain));

%for i = 1:length(selTrain)
%	d{i} = descrs{selTrain(i)};
%end

%if strcmp(feature, 'hist')
%	d = cat(2, d{:});
%	d = single(d);
%else

desc = vl_colsubset(cat(2, desc{:}), 10e4) ;
desc = single(desc) ;

%end

% Quantize the descriptors to get the visual words
vocab = vl_kmeans(desc, conf.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;

%MODEL.vocab = vocab ;

kdtree = vl_kdtreebuild(vocab);
%if strcmp(MODEL.quantizer, 'kdtree')
%  MODEL.kdtree = vl_kdtreebuild(vocab) ;
%end

