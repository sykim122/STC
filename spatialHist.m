function hists = spatialHist(conf, model, imgs, feature)

hists = cell(1, length(imgs)) ;

parfor ii = 1:length(imgs)
	im = imgs{ii};
	hists{ii} = getImageDescriptor(model, im, feature);
end

hists = cat(2, hists{:}) ;
save(conf.histPath, 'hists') ;
