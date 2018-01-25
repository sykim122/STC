function [target auxiliary] = load_data(conf)

rng(1)


images = dir(fullfile(conf.dataDir, '*.png'))' ;
images = cellfun(@(x)fullfile(conf.dataDir,x),{images.name},'UniformOutput',false) ;


images = images(randperm(length(images), conf.numData));

idx = randperm(conf.numData, conf.numData*conf.per_target);
target = images(idx);

tmp = [1:conf.numData];
tmp(idx) = [];
auxiliary = images(tmp);

clearvars -except target auxiliary

