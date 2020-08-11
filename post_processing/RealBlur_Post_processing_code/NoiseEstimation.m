% -----------------------------------------------------------------------     
%Inputs:
% im_noisy:  the noisy image whose noise level requires to be estimated
% PatchSize: the predefined size of patches
% 
%Outputs:
% estsigma: Estimated result given by our method
%
% Last revision: 1-Dec-2015
%
% Authors: Guangyong Chen
% License: MIT License
%
% Copyright (c) 2015 Guangyong Chen
%
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this software and associated documentation files (the
% "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to
% permit persons to whom the Software is furnished to do so, subject to
% the following conditions:
% 
% The above copyright notice and this permission notice shall be
% included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
% NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
% LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
% OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
% WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%
% -----------------------------------------------------------------------
function [estsigma, CompCnt] =NoiseEstimation(im_noisy,PatchSize)

p_out = image2cols(im_noisy, PatchSize, 3);

mu = mean(p_out,2);
sigma=(p_out-repmat(mu,[1,size(p_out,2)])) ...
        *(p_out-repmat(mu,[1,size(p_out,2)]))'/(size(p_out,2));
eigvalue = (sort((eig(sigma)),'ascend'));
 
for CompCnt = size(p_out,1):-1:1
    Mean = mean(eigvalue(1:CompCnt));
    
    if(sum(eigvalue(1:CompCnt)>Mean) == sum(eigvalue(1:CompCnt)<Mean))
        break
    end
   
end
estsigma = sqrt(Mean);
