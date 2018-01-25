function X = getFeature(images, conf)

len = length(images);
desc = cell(1, len);
%ims = cell(1, conf.numData);

parfor i = 1:len
    im = imread(images{i}) ;
    %ims{i} = im;

    % PHOW feature
    im = standarizeImage(im);
    [drop, desc{i}] = vl_phow(im, conf.phowOpts{:}) ;
end

[vocab kdtree] = trainvocab(conf, desc);

X = [];
for i = 1:len
    X = [X; double(vl_kdtreequery(kdtree, vocab, ...
		single(desc{i}), ...
		'MaxComparisons', 50))];
end


clearvars -except X
