function [outputImage] = warping_with_resize_undistortion(A, homography, cameraParams, params)
% function is based on imresize_old function

bi_interp = 1; % we use only bicubic kernel
if nargin > 3
    defflt_reducedim = params.antialiasing; % only support downsample with antialiasing
    undistort = params.undistort;
    m = params.resize;
else
    defflt_reducedim = false; % only support downsample with antialiasing
    undistort = false;
    m = 1;
end

[so(1),so(2),thirdD] = size(A); % old image size
if isscalar(m)
    % m is the scale factor.
    sn = max(floor(m*so(1:2)),1); % new image size=(integer>0)
    sc = [m m];
else
    % m is new image size
    sn = m;
    sc = sn ./ so;
end

if bi_interp  && defflt_reducedim,
    if (~isa(A,'double')),%change format to double to perform imfilter
        A = im2double(A);
        classChanged = 1;
    end
    
    if defflt_reducedim,%Design anti-aliasing filter for reduced image
        drec = find(sn<so);% find direction of filtering
        for k = drec,% create filter for drec-direction
            h = 11;
            hh(k,:) = DesignFilter(h,sn(k)/so(k)); %#ok<AGROW>
        end;
        if length(drec)==1,%filters in one direction only
            % first direction is column, second is row
            h = reshape(hh(k,:),(h-1)*(k==1)+1,(h-1)*(k==2)+1);
        else % filters in both directions
            for k=1:thirdD,%loop if A matrix is 3D
                A(:,:,k) = imfilter(imfilter(A(:,:,k), hh(2,:),'replicate'),...
                    hh(1,:).','replicate');
            end
        end;
    end;
end

% define homography of downsampling
if m ~= 1
    a = [sc(2),         0,                0
        0,             sc(1),            0
        0.5*(1-sc(2)), 0.5*(1-sc(1)),    1];
    a = a * homography;
else
    a = homography;
end

T = maketform('projective', a);


[x, y] = meshgrid(1:sn(2), 1:sn(1));

% apply inverse transform of downsampling and homography
[xm, ym] = tforminv(T, x(:), y(:));

% below code is based on computeMap mehtod of ImageTransformer
ptsIn = [xm ym];

if undistort
    % apply distort
    ptsOut = vision.internal.calibration.distortPoints(ptsIn, cameraParams.IntrinsicMatrix, ...
        cameraParams.RadialDistortion, cameraParams.TangentialDistortion);
else
    ptsOut = ptsIn;
end

xm2 = reshape(ptsOut(:,1), sn);
ym2 = reshape(ptsOut(:,2), sn);

% wapring image
outputImage = zeros(sn(1),sn(2),thirdD);
for plane = 1:thirdD
    outputImage(:,:,plane) = interp2(A(:,:,plane),xm2,ym2, 'cubic', 0);
end


end

function b = DesignFilter(N,Wn)
% Modified from SPT v3 fir1.m and hanning.m
% first creates only first half of the filter
% and later mirrows it to the other half

odd = rem(N,2);
vec = 1:floor(N/2);
vec2 = pi*(vec-(1-odd)/2);

wind = .54-.46*cos(2*pi*(vec-1)/(N-1));
b = [fliplr(sin(Wn*vec2)./vec2).*wind Wn];% first half is ready
b = b([vec floor(N/2)+(1:odd) fliplr(vec)]);% entire filter
b = b/abs(polyval(b,1));% norm
end