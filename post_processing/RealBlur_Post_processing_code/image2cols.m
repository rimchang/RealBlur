% -----------------------------------------------------------------------     
%Inputs:
% im:  the noisy image to be patchized
% pSz: the predefined size of patches
% stride: the predfined gap between neighbor patches
%
%Outputs:
% res: the set of decomposed patches
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
function res = image2cols(im, pSz, stride)
  res = [];

  range_y = 1:stride:(size(im,1)-pSz+1);
  range_x = 1:stride:(size(im,2)-pSz+1);
  channel = size(im,3);
  if (range_y(end)~=(size(im,1)-pSz+1))
    range_y = [range_y (size(im,1)-pSz+1)];
  end
  if (range_x(end)~=(size(im,2)-pSz+1))
    range_x = [range_x (size(im,2)-pSz+1)];
  end
  sz = length(range_y)*length(range_x);

  tmp = zeros(pSz^2*channel, sz);

  idx = 0;
  for y=range_y
    for x=range_x
      p = im(y:y+pSz-1,x:x+pSz-1,:);
      idx = idx + 1;
      tmp(:,idx) = p(:);
    end
  end

  res = [res, tmp];
      


return

