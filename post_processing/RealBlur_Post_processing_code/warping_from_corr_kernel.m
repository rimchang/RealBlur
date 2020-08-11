function [ latentReg, blurredReg, x_diff, y_diff ] = warping_from_corr_kernel( latent, blurred, psf )
%kernel 로부터 centroid를 추정하여 img warping, 이때, correlation filter란 것을 명심!
%   psf_estimate 후에 rot90으로 filp 해준다음 함수에 넣어줘야 한다!!
    rpsf = psf;
    % compute centorid
    labeledImage = bwlabel(true(size(rpsf)));
    centroid = regionprops(labeledImage, rpsf, 'Centroid', 'WeightedCentroid');

    x_diff = centroid.Centroid(1) - centroid.WeightedCentroid(1);
    y_diff = centroid.Centroid(2) - centroid.WeightedCentroid(2);                 
                        

    H = [1 0 0; 0 1 0; x_diff y_diff 1];
    latent = imtranslate(latent, [x_diff, y_diff]);
    blurred = blurred;
    [top, bottom, left, right] = bboxFromHomography(latent, H);
    
    latentReg = latent(top:bottom, left:right, :);
    blurredReg = blurred(top:bottom, left:right, :);  

end

