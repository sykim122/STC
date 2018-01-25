function hist = getImageDescriptor(model, im, feature)
numWords = size(model.vocab, 2);   

if strcmp(feature, 'hist')
	im = imresize(im,[480 480]);
  	im = im2double(im);
  	width = size(im,2);
	height = size(im,1);
	descrs = get_hists(im);
	bins = double(vl_kdtreequery(model.kdtree, model.vocab, ...
					  single(descrs), ...
					  'MaxComparisons', 50)) ;
	hist = zeros(numWords, 1);
	hist = vl_binsum(hist, ones(size(bins)), bins);
	hist = single(hist / sum(hist));

else
	im = standarizeImage(im) ;
	width = size(im,2) ;
	height = size(im,1) ;


	% get PHOW features
	[frames, descrs] = vl_phow(im, model.phowOpts{:}) ;

	% quantize local descriptors into visual words
	    binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
					  single(descrs), ...
					  'MaxComparisons', 50)) ;

	for i = 1:length(model.numSpatialX)
	  binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames(1,:)) ;
	  binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames(2,:)) ;

	  % combined quantization
	  bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
			 binsy,binsx,binsa) ;
	  hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
	  hist = vl_binsum(hist, ones(size(bins)), bins) ;
	  hists{i} = single(hist / sum(hist)) ;
	end
	hist = cat(1,hists{:}) ;
	hist = hist / sum(hist) ;
end

