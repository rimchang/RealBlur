function [y_est] = CBM3D(z, sigma_psd, profile, colorspace)
% BM3D For color images. Performs color transform to do block-matching in luminance domain.
% Arguments:
% 
% z: Noisy image, 3 channels (MxNx3)
% sigma_psd: Noise PSD, either MxN or MxNx3 (different PSDs for different channels)
% OR
% sigma_psd: Noise standard deviation, either float, or [float, float, float] for 3 different stds.
% profile: Settings for BM3D: BM3DProfile object or a string.
%  ('np', 'refilter', 'vn', 'vn_old', 'high', 'deb')
% colorspace: 'YCbCr' or 'opp' for choosing the color transform


if ~exist('profile','var')
    profile         = 'np'; %% default profile
end

% Color transform

if ~exist('colorspace', 'var')
    colorspace = 'opp';
end

[z, img_max, img_min, scale, A] = rgb_to(z, colorspace);

single_dim_psd = size(sigma_psd, 1) == 1 || size(sigma_psd, 2) == 1;

if (single_dim_psd && numel(sigma_psd) > 1) || size(sigma_psd, 3) == 3
    if single_dim_psd
        o = reshape(sigma_psd(:), [1, 3]).^2 * A.^2';
        psd = sqrt(o ./ (img_max - img_min).^2);

    else
       o = reshape(sigma_psd, [size(sigma_psd, 1) * size(sigma_psd, 2), 3]) * A.^2';
       o = reshape(o, [size(sigma_psd, 1), size(sigma_psd, 2), 3]);
       psd = o ./ permute((img_max - img_min).^2, [1, 3, 2]); 
    end
else
    if single_dim_psd
       psd = sigma_psd(:) .* sqrt(scale(:));
    else
       psd = sigma_psd .* permute(scale, [1, 3, 2]);
    end
end
% Call BM3D
y_est = BM3D(z, psd, profile);

% Inverse color transform
y_est = rgb_to(y_est, colorspace, true, img_max, img_min);


end

% Converts to normalized YCbCr/opp (or back), returns normalization values needed for inverse
function [o, o_max, o_min, scale, A] = rgb_to(img, colormode, inverse, o_max, o_min)

    if exist('colormode', 'var') && strcmp(colormode, 'opp')
        % Forward
        A =[1/3 1/3 1/3; 0.5  0  -0.5; 0.25  -0.5  0.25];
        % Inverse
        B =[1 1 2/3;1 0 -4/3;1 -1 2/3];
    else
        % YCbCr
        A = [0.299, 0.587, 0.114; -0.168737, -0.331263, 0.5;  0.5,  -0.418688,  -0.081313];
        B = [1.0000, 0.0000, 1.4020; 1.0000, -0.3441, -0.7141; 1.0000, 1.7720, 0.0000];
    end

    if exist('inverse', 'var') && inverse
        % The inverse transform
        o = (reshape(img, [size(img, 1) * size(img, 2), 3]) .* (o_max - o_min) + o_min) * B';
    else
        % The color transform
        o = reshape(img, [size(img, 1) * size(img, 2), 3]) * A';
        %o(:, 2:3) = o(:, 2:3) + 0.5;
        o_max = max(o, [], 1);
        o_min = min(o, [], 1);
        o = (o - o_min) ./ (o_max - o_min);
        scale = sum(A'.^2) ./ (o_max - o_min).^2;
    end
    
    o = reshape(o, [size(img, 1), size(img, 2), 3]);
end