function [y_est, blocks] = BM3D(z, sigma_psd, profile, stage_arg, blockmatches)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  BM3D is an algorithm for attenuation of additive spatially correlated
%  stationary (aka colored) Gaussian noise in grayscale and multichannel images.
%
%
%  FUNCTION INTERFACE:
%
%  y_est = BM3D(z, sigma_psd, profile)
%
%  INPUT ARGUMENTS:
%
%  -- required --
%
%         'z' : noisy image (M x N or M x N x C double array, intensities in range [0,1])
%               For multichannel images, block matching is performed on the first channel.
%  'sigma_psd' : noise power spectral density (M x N double nonnegative array)
%               OR
%               noise STD
%               OR
%               either of these in multichannel:
%               (M x N x C PSDs or 1 x C STDs)
%
% -- optional --
%
%   'profile' : 'np' --> Normal Profile (default)
%               'refilter' --> Apply refiltering
%               OR
%               a BM3DProfile object specifying the parameters
%               some other premade profiles also included from the previous versions
%               in BM3DProfile.m
%
%   'stage_arg' : Determines whether to perform hard-thresholding or wiener filtering.
%                 either BM3DProfile.HARD_THRESHOLDING, BM3DProfile.ALL_STAGES or an estimate
%                  of the noise-free image.
%                    - BM3DProfile.ALL_STAGES: Perform both.
%                    - BM3DProfile.HARD_THRESHOLDING: Perform hard-thresholding only.
%                    - ndarray, size of z: Perform Wiener Filtering with stage_arg as pilot.
%
%   'blockmatches' : Tuple {HT, Wiener}, with either value either:
%                      - false : Do not save blockmatches for phase
%                      (default)
%                      - true : Save blockmatches for phase
%                      - Pre-computed block-matching array returned by a
%                      previous call with [true]
%  OUTPUT:
%      'y_est'  denoised image  (M x N double array)
%      'y_est', {'blocks_ht', 'blocks_wie'} denoised image, plus HT and
%          Wiener blockmatches, if any storeBM values are set to True
%          (or [0] for missing block array, if only one calculated)
%
%
%  BASIC SIMULATION EXAMPLES:
%
%     Case 1)
%
%      % Read a grayscale noise-free image
%
%      y=im2double(imread('cameraman.tif'));
%
%      % Generate noisy observations corrupted by additive colored random noise
%        generated as convution of AWGN against with kernel 'k'
%
%      k=[-1;2;-1]*[1 4 1]/100;   % e.g., a diagonal kernel
%      z=y+imfilter(randn(size(y)),k(end:-1:1,end:-1:1),'circular');
%
%      % define 'sigma_psd' from the kernel 'k'
%
%      sigma_psd=abs(fft2(k,size(z,1),size(z,2))).^2*numel(z);
%
%      % Denoise 'z'
%      y_est = BM3D(z, sigma_psd);
%
%
%     Case 2)
%
%      % Read a grayscale noise-free image
%
%      y=im2double(imread('cameraman.tif'));
%
%      % Generate noisy observations corrupted by additive colored random noise
%      % generated as convution of AWGN against with kernel 'k'
%      [x2, x1]=meshgrid(ceil(-size(y,2)/2):ceil(size(y,2)/2)-1,ceil(-size(y,1)/2):ceil(size(y,1)/2)-1)
%      sigma_psd=ifftshift(exp(-((x1/size(y,1)).^2+(x2/size(y,2)).^2)*10))*numel(y)/100;
%      z=y+real(ifft2(fft2(randn(size(y))).*sqrt(sigma_psd)/sqrt(numel(y))));
%
%      % Denoise 'z'
%      y_est = BM3D(z, sigma_psd);
%
%     Case 3) If 'sigma_psd' is a singleton, this value is taken as sigma and
%             it is assumed that the noise is white variance sigma^2.
%
%      % Read a grayscale noise-free image
%
%      y=im2double(imread('cameraman.tif'));
%
%      % Generate noisy observations corrupted by additive white Gaussian noise with variance sigma^2
%      sigma=0.1;
%      z=y+sigma*randn(size(y));
%
%      y_est = BM3D(z, sigma);
%
%      % or, equivalently,
%      sigma_psd = ones(size(z))*sigma^2*numel(z)
%      y_est = BM3D(z, sigma_psd)
%
%
%      Case 4)   MULTICHANNEL PROCESSING
%
%      y_est = BM3D(cat(3, z1, z2, z3), sigma_psd, 'np'); 
%
%      Multiple PSDs are optionally handled in the same way.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (c) 2006-2019 Tampere University.
% All rights reserved.
% This work (software, material, and documentation) shall only
% be used for nonprofit noncommercial purposes.
% Any unauthorized use of this work for commercial or for-profit purposes
% is prohibited.
%
% AUTHORS:
%     Y. Mäkinen, L. Azzari, K. Dabov, A. Foi
%     email: ymir.makinen@tuni.fi
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('profile','var')
    profile         = 'np'; %% default profile
end

if isa(profile, 'string') || isa(profile, 'char')
    profile = BM3DProfile(profile);
end

if ~exist('stage_arg','var')
    stage_arg = profile.ALL_STAGES;  % By default, do both HT and Wie
elseif stage_arg == profile.WIENER_FILTERING
    disp('Error: If you wish to only do wiener filtering, pass the estimate y_hat instead of the WIENER_FILTERING value!')
    return
elseif isa(stage_arg, 'float') || isa(stage_arg, 'double')
    
    if numel(size(stage_arg)) < 2
        disp('Error: stage_arg must be either stage value from BM3DProfile or an estimate y_hat!')
        return
    end
    
    % Presume that stage_arg is an estimate for wiener.
    y_hat = stage_arg;
    stage_arg = profile.WIENER_FILTERING;

elseif ~isa(stage_arg, 'int8')
    disp('Error: stage_arg must be either stage value from BM3DProfile or an estimate y_hat!')
    return
end


% Define maximum pad size: pad size should be at least
% half the size of the correlation kernel, but needn't be larger
% (although it can be)
% To be sure, we pad total of the image size, but if the
% kernel size is approximately known, some computation time
% may be saved by specifying it in the profile.
if profile.max_pad_size(1) == -1
    pad_size = [ceil(size(z, 1)/2), ceil(size(z, 2)/2)];
else
    pad_size = profile.max_pad_size;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% convert z to double precision if needed
z = double(z);

% Check if sigmas were passed instead of PSDs.
single_dim_psd = size(sigma_psd, 1) == 1 || size(sigma_psd, 2) == 1;

% Get relevant info from the sigma_psd, including lambda and mu.
[sigma_psd2, psd_blur, psd_k, profile] = process_psd(sigma_psd, z, ...
    single_dim_psd, pad_size, profile);

% Blockatching save information for the binary
blockmatches_ht = int32(0);
blockmatches_wie = int32(0);

if exist('blockmatches', 'var')
   blockmatches_ht = int32(blockmatches{1});
   blockmatches_wie = int32(blockmatches{2});
end

ht_bm = {};
wie_bm = {};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Step 1. Produce the basic estimate by HT filtering
%%%%

if bitand(stage_arg, BM3DProfile.HARD_THRESHOLDING)
    
    [t_forward, t_inverse, hadper_trans_single_den, ...
        inverse_hadper_trans_single_den, Wwin2D] = get_transforms(profile, true);

    
    [y_hat, ht_bm] = bm3d_thr_colored_noise(z, hadper_trans_single_den, ...
        inverse_hadper_trans_single_den, profile.Nstep, profile.N1, profile.N2, ...
        profile.lambda_thr3D, profile.tau_match*profile.N1*profile.N1/(255*255), ...
        (profile.Ns-1)/2, single(t_forward), single(t_inverse)', ...
        [numel(profile.lambda_thr3D); profile.lambda_thr3D(:)], Wwin2D, single(psd_blur), ...
        profile.Nf, single(profile.gamma), profile.Kin, blockmatches_ht);


    % Re-filter    
    if (profile.denoise_residual)

       [remains, remains_PSD] = get_filtered_residual(z, y_hat, sigma_psd2, pad_size, ...
           profile.residual_thr, single_dim_psd);
       remains_PSD = process_psd_for_nf(remains_PSD, psd_k, profile);
        
        % Skip refiltering if there is a zero sigma_psd
        if(min(max(max(remains_PSD, [], 1), [], 2)) > 1e-5)
            % Re-filter
            [y_hat, ht_bm] = bm3d_thr_colored_noise(double(y_hat + remains), ...
                hadper_trans_single_den, inverse_hadper_trans_single_den,  ...
                profile.Nstep, profile.N1, profile.N2, profile.lambda_thr3D_re, ...
                profile.tau_match*profile.N1*profile.N1/(255*255), (profile.Ns-1)/2, ...
                single(t_forward), single(t_inverse)', ...
                [numel(profile.lambda_thr3D_re); profile.lambda_thr3D_re(:)], Wwin2D, ...
                single(remains_PSD), profile.Nf, single(profile.gamma), profile.Kin, ...
                blockmatches_ht);
           
        end


    end

    if profile.print_info
        disp('Hard-thresholding phase completed')
    end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Step 2. Produce the final estimate by Wiener filtering (using the
%%%%  hard-thresholding initial estimate)
%%%

if bitand(stage_arg, BM3DProfile.WIENER_FILTERING)

    [t_forward, t_inverse, hadper_trans_single_den, ...
        inverse_hadper_trans_single_den, Wwin2D] = get_transforms(profile, false);
  
    mu2 = reshape(profile.mu2(:), [1, 1, numel(profile.mu2)]);
    mu2_re = reshape(profile.mu2_re(:), [1, 1, numel(profile.mu2_re)]);
    if single_dim_psd
    	mu2 = sqrt(mu2);
    end
    
    % Wiener filtering

    [y_est, wie_bm] = bm3d_wiener_colored_noise(z, single(y_hat), hadper_trans_single_den, ...
        inverse_hadper_trans_single_den, profile.Nstep_wiener, profile.N1_wiener, ...
        profile.N2_wiener, profile.tau_match_wiener*profile.N1_wiener*profile.N1_wiener/(255*255), ...
        (profile.Ns_wiener-1)/2, zeros(profile.N1_wiener, profile.N1_wiener), single(t_forward), ...
        single(t_inverse)', Wwin2D, single(psd_blur.*mu2), profile.Nf, profile.Kin, ...
        blockmatches_wie);



    if (profile.denoise_residual)

        [remains, remains_PSD] = get_filtered_residual(z, y_est, sigma_psd2, pad_size, ...
            profile.residual_thr, single_dim_psd);
        remains_PSD = process_psd_for_nf(remains_PSD, psd_k, profile);
        
        if(min(max(max(remains_PSD, [], 1), [], 2)) > 1e-5) 

            % Re-filter
            [y_est, wie_bm] = bm3d_wiener_colored_noise(double(y_est + remains), y_est, ...
                hadper_trans_single_den, inverse_hadper_trans_single_den, profile.Nstep_wiener, ...
                profile.N1_wiener, profile.N2_wiener, ...
                profile.tau_match_wiener*profile.N1_wiener*profile.N1_wiener/(255*255), ...
                (profile.Ns_wiener-1)/2, zeros(profile.N1_wiener, profile.N1_wiener), ...
                single(t_forward), single(t_inverse)', Wwin2D, ...
                single(remains_PSD.*mu2_re), profile.Nf, profile.Kin, blockmatches_wie);

        end

    end    

    if profile.print_info
        disp('Wiener phase completed')
    end
else
    y_est = y_hat;
end

y_est = double(y_est);
blocks = {ht_bm, wie_bm};

return;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Some auxiliary functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Get residual, filtered by global FFT HT
function [remains, remains_PSD] = get_filtered_residual(z, y_hat, sigma_psd, pad_size, ...
    residual_thr, single_dim_psd)
    
    resid = fft2(padarray(z - double(y_hat), pad_size, 'both'));

    % Convolve & max filter the thresholded map for extra coefficients
    ksz = ceil(size(resid) / 150);
    ksz = ksz + 1 - mod(ksz, 2);
    k=fspecial('gaussian',[ksz(1), 1], size(resid, 1) / 500) * ...
        fspecial('gaussian',[1, ksz(2)], size(resid, 2) / 500);
    
    P = sigma_psd;
    if single_dim_psd  % It's std, not variance.
        P = sigma_psd.^2 * size(z, 1) * size(z, 2);
    end
    
    % Respahapes for convolution
    if (single_dim_psd && numel(size(y_hat)) == 3)
       P = reshape(P(:), [1, 1, numel(P)]);
    elseif (single_dim_psd && numel(size(y_hat)) == 2)
       P = reshape(P(:), [1, numel(P)]);
    end
    
    cc=convn(padarray(abs(resid) > residual_thr .* sqrt(P), (size(k) - 1)/2, 'circular'), k, 'valid');

    % Threshold mask
    msk = (cc > 0.01);
    
    % Residual + sigma_psd
    remains = ifft2((resid) .* msk);
    remains_PSD = P .* msk;

    % Crop the pad off
    remains = remains(1 + pad_size(1) : end - pad_size(1),...
                      1 + pad_size(2) : end - pad_size(2), :);
   
    % Also from the sigma_psd             
    temp_kernel = real(fftshift(fftshift(ifft2(sqrt(remains_PSD / (size(z, 1) * size(z, 2)))), 1), 2));
    temp_kernel = temp_kernel(1+pad_size(1):end-pad_size(1), 1+pad_size(2):end-pad_size(2), :);
    remains_PSD = abs(fft2(temp_kernel, size(z, 1), size(z, 2))).^2 * (size(z, 1)) * (size(z, 2));

end

function [Tforward, Tinverse] = get_transf_matrix(N, transform_type, dec_levels)
%
% Create forward and inverse transform matrices, which allow for perfect
% reconstruction. The forward transform matrix is normalized so that the
% l2-norm of each basis element is 1.
%
% [Tforward, Tinverse] = get_transf_matrix (N, transform_type, dec_levels)
%
%  INPUTS:
%
%   N               --> Size of the transform (for wavelets, must be 2^K)
%
%   transform_type  --> 'dct', 'dst', 'hadamard', or anything that is
%                       listed by 'help wfilters' (bi-orthogonal wavelets)
%                       'DCrand' -- an orthonormal transform with a DC and all
%                       the other basis elements of random nature
%
%   dec_levels      --> If a wavelet transform is generated, this is the
%                       desired decomposition level. Must be in the
%                       range [0, log2(N)-1], where "0" implies
%                       full decomposition.
%
%  OUTPUTS:
%
%   Tforward        --> (N x N) Forward transform matrix
%
%   Tinverse        --> (N x N) Inverse transform matrix
%

if ~exist('dec_levels','var')
    dec_levels = 0;
end

if N == 1
    Tforward = 1;
elseif strcmp(transform_type, 'hadamard') == 1
    Tforward    = hadamard(N);
elseif (N == 8) && strcmp(transform_type, 'bior1.5')==1 % hardcoded transform so that the wavelet toolbox is not needed to generate it
    Tforward =[ 0.343550200747110   0.343550200747110   0.343550200747110   0.343550200747110   0.343550200747110   0.343550200747110   0.343550200747110   0.343550200747110
               -0.225454819240296  -0.461645582253923  -0.461645582253923  -0.225454819240296   0.225454819240296   0.461645582253923   0.461645582253923   0.225454819240296
                0.569359398342840   0.402347308162280  -0.402347308162280  -0.569359398342840  -0.083506045090280   0.083506045090280  -0.083506045090280   0.083506045090280
               -0.083506045090280   0.083506045090280  -0.083506045090280   0.083506045090280   0.569359398342840   0.402347308162280  -0.402347308162280  -0.569359398342840
                0.707106781186550  -0.707106781186550                   0                   0                   0                   0                   0                   0
                                0                   0   0.707106781186550  -0.707106781186550                   0                   0                   0                   0
                                0                   0                   0                   0   0.707106781186550  -0.707106781186550                   0                   0
                                0                   0                   0                   0                   0                   0   0.707106781186550  -0.707106781186550];
elseif (N == 8) && strcmp(transform_type, 'dct')==1 % hardcoded transform so that the signal processing toolbox is not needed to generate it
    Tforward = [ 0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274;
        0.490392640201615   0.415734806151273   0.277785116509801   0.097545161008064  -0.097545161008064  -0.277785116509801  -0.415734806151273  -0.490392640201615;
        0.461939766255643   0.191341716182545  -0.191341716182545  -0.461939766255643  -0.461939766255643  -0.191341716182545   0.191341716182545   0.461939766255643;
        0.415734806151273  -0.097545161008064  -0.490392640201615  -0.277785116509801   0.277785116509801   0.490392640201615   0.097545161008064  -0.415734806151273;
        0.353553390593274  -0.353553390593274  -0.353553390593274   0.353553390593274   0.353553390593274  -0.353553390593274  -0.353553390593274   0.353553390593274;
        0.277785116509801  -0.490392640201615   0.097545161008064   0.415734806151273  -0.415734806151273  -0.097545161008064   0.490392640201615  -0.277785116509801;
        0.191341716182545  -0.461939766255643   0.461939766255643  -0.191341716182545  -0.191341716182545   0.461939766255643  -0.461939766255643   0.191341716182545;
        0.097545161008064  -0.277785116509801   0.415734806151273  -0.490392640201615   0.490392640201615  -0.415734806151273   0.277785116509801  -0.097545161008064];
elseif (N == 8) && strcmp(transform_type, 'dst')==1 % hardcoded transform so that the PDE toolbox is not needed to generate it
    Tforward = [ 0.161229841765317   0.303012985114696   0.408248290463863   0.464242826880013   0.464242826880013   0.408248290463863   0.303012985114696   0.161229841765317;
        0.303012985114696   0.464242826880013   0.408248290463863   0.161229841765317  -0.161229841765317  -0.408248290463863  -0.464242826880013  -0.303012985114696;
        0.408248290463863   0.408248290463863                   0  -0.408248290463863  -0.408248290463863                   0   0.408248290463863   0.408248290463863;
        0.464242826880013   0.161229841765317  -0.408248290463863  -0.303012985114696   0.303012985114696   0.408248290463863  -0.161229841765317  -0.464242826880013;
        0.464242826880013  -0.161229841765317  -0.408248290463863   0.303012985114696   0.303012985114696  -0.408248290463863  -0.161229841765317   0.464242826880013;
        0.408248290463863  -0.408248290463863                   0   0.408248290463863  -0.408248290463863                   0   0.408248290463863  -0.408248290463863;
        0.303012985114696  -0.464242826880013   0.408248290463863  -0.161229841765317  -0.161229841765317   0.408248290463863  -0.464242826880013   0.303012985114696;
        0.161229841765317  -0.303012985114696   0.408248290463863  -0.464242826880013   0.464242826880013  -0.408248290463863   0.303012985114696  -0.161229841765317];
elseif strcmp(transform_type, 'dct') == 1
    Tforward    = dct(eye(N));
elseif strcmp(transform_type, 'dst') == 1
    Tforward    = dst(eye(N));
elseif strcmp(transform_type, 'DCrand') == 1
    x = randn(N); x(1:end,1) = 1; [Q,~] = qr(x);
    if (Q(1) < 0)
        Q = -Q;
    end;
    Tforward = Q';
else %% a wavelet decomposition supported by 'wavedec'
    %%% Set periodic boundary conditions, to preserve bi-orthogonality
    dwtmode('per','nodisp');

    Tforward = zeros(N,N);
    for i = 1:N
        Tforward(:,i)=wavedec(circshift([1 zeros(1,N-1)],[dec_levels i-1]), log2(N), transform_type);  %% construct transform matrix
    end
end

%%% Normalize the basis elements
if ~((N == 8) && strcmp(transform_type, 'bior1.5')==1)
    Tforward = (Tforward' * diag(sqrt(1./sum(Tforward.^2,2))))';
end

%%% Compute the inverse transform matrix
Tinverse = inv(Tforward);

return;
end


function [psd] = process_psd_for_nf(sigma_psd, psd_k, profile)
    if profile.Nf == 0
        psd = sigma_psd;
        return;
    end
    
    % Reduce PSD size to start with
    max_ratio = 16;
    sigma_psd_copy = sigma_psd;
    single_kernel = ones(3, 3, 1) / 9;
    orig_ratio = max(size(sigma_psd)) / profile.Nf;
    ratio = orig_ratio;
    while ratio > max_ratio
        mid_corr = convn(padarray(sigma_psd_copy, [1, 1], 'circular'), ...
                         single_kernel, 'valid');
        sigma_psd_copy = mid_corr(2:3:end, 2:3:end, :);
        ratio = max(size(sigma_psd_copy)) / profile.Nf;
    end
    
    % Scale PSD because the binary expects it to be scaled by size
    sigma_psd_copy = sigma_psd_copy .* (ratio / orig_ratio).^2;
    if ~isempty(psd_k)
        sigma_psd_copy = convn(padarray(sigma_psd_copy, ...
                                        (size(psd_k) - 1) / 2, 'circular'), ...
                               psd_k, 'valid');
    end
    
    psd = sigma_psd_copy;

end

function [lambda, wielambdasq, lambda2, wielambdasq2] = estimate_psd_parameters(PSD65_full)
% Estimate parameters based on the sigma_psd

% Get the optimal parameters and matching features for a bunch of PSDs
load('param_matching_data.mat', 'features', 'maxes');

sz = 65;
indices_to_take = [1:2:10 12:5:32];

lambda = [];
wielambdasq = [];
lambda2 = [];
wielambdasq2 = [];

% Get separate parameters for each sigma_psd provided
for psd_num = 1:size(PSD65_full, 3)
    PSD65 = PSD65_full(:, :, psd_num);
    PSD65 = fftshift(PSD65);
    
    % Get features for this sigma_psd
    pcaxa = get_features(PSD65, sz, indices_to_take);

    % Calculate distances to other PSDs
    mm = mean(features, 2);
    centered_features = features - mm;
    corr_matx = centered_features * centered_features';
    corr_matx = corr_matx / 500;
    
    centered_pcax = pcaxa' - mm;
    
    [u, s, ~] = svd(corr_matx);
    centered_features = u * centered_features;
    centered_pcax = u * centered_pcax;
    centered_features = centered_features .* sqrt(diag(s));
    centered_pcax = centered_pcax .* sqrt(diag(s));

    diff_pcax = sqrt(sum(abs(centered_features - centered_pcax).^2, 1));

    % Take only smallest->best x %
    [~, dff_I] = sort(diff_pcax);

    %  Take 20 most similar PSDs into consideration
    count = 20;
    diff_indices = dff_I(1:count);

    % Invert, smaller -> bigger weight
    diff_inv = 1 ./ (diff_pcax + eps);
    diff_inv = diff_inv(diff_indices) ./ sum(diff_inv(diff_indices));

    % Weight
    param_idxs = sum(diff_inv .* maxes(diff_indices, :)', 2);

    lambdas = 2.5:0.1:5;
    wielambdasqs = 0.2:0.2:6;
    
    % Get parameters from indices - 
    % Interpolate lambdas and mu^2s from the list
    for ix = [1, 3]
        param_idx = max(1, param_idxs(ix));
        param_idx2 = max(1, param_idxs(ix+1));

        l1 = lambdas(floor(param_idx));
        l2 = lambdas(min(ceil(param_idx), numel(lambdas)));

        w1 = wielambdasqs(floor(param_idx2));
        w2 = wielambdasqs(min(ceil(param_idx2), numel(wielambdasqs)));

        param_smooth = param_idx - floor(param_idx);
        param_smooth2 = param_idx2 - floor(param_idx2);

        if ix == 1
            lambda = [lambda, l2 * param_smooth + l1 * (1 - param_smooth)];
            wielambdasq = [wielambdasq, w2 * param_smooth2 + w1 * (1 - param_smooth2)];
        elseif ix == 3
            lambda2 = [lambda2, l2 * param_smooth + l1 * (1 - param_smooth)];
            wielambdasq2 = [wielambdasq2, w2 * param_smooth2 + w1 * (1 - param_smooth2)];
        end

    end
end
end

% Calculate features for a sigma_psd from integrals
function f = get_features(sigma_psd, sz, indices_to_take)

    [I_rot, I_rot2] = pcax(sigma_psd);
    f1 = zeros(1, numel(indices_to_take));
    f2 = f1;
    
    % Extract features for a sigma_psd
    for ii = 1:numel(indices_to_take)
        rang = indices_to_take(ii);
        if ii > 1
            rang = indices_to_take(ii-1)+1:rang;
        end
        f1(ii) = sum(I_rot(ceil(sz/2) + rang - 1)) / numel(rang);
        f2(ii) = sum(I_rot2(ceil(sz/2) + rang - 1)) / numel(rang);
    end
    
    f = [f1 f2];
end

% Calculate integrals along principal axes of the sigma_psd
function [I_rot, I_rot2] = pcax(sigma_psd)

N=size(sigma_psd,1);
[G2, G1]=meshgrid(1:N,1:N);

trapz2D=@(G2,G1,sigma_psd) trapz(G1(:,1),trapz2(G2,sigma_psd,2),1);

Pn=sigma_psd/trapz2D(G2,G1,sigma_psd);

m2=trapz2D(G2,G1,Pn.*G2);
m1=trapz2D(G2,G1,Pn.*G1);
C=zeros(2);
O1=[2 1 1 0];
O2=[0 1 1 2];
for jj=[1 2 4]
    C(jj)=squeeze(trapz2D(G2,G1,Pn.*(G2-m2).^O1(jj).*(G1-m1).^O2(jj)));
end
C(3)=C(2);


[U, ~, ~]=svd(C);

N3 = 3 * N;
[G13N, G23N]=ndgrid((1:N3)-(N3+1)/2, (1:N3)-(N3+1)/2);

% Rotate PSDs and calculate integrals along the rotated PSDs
theta = angle(U(1, 1) + 1i * U(1, 2));
G2rot=G23N(N+1:2*N,N+1:2*N)*cos(theta)-G13N(N+1:2*N,N+1:2*N)*sin(theta);
G1rot=G13N(N+1:2*N,N+1:2*N)*cos(theta)+G23N(N+1:2*N,N+1:2*N)*sin(theta);
P_rot_handle = griddedInterpolant(G13N,G23N,repmat(sigma_psd,[3,3]),'linear','nearest');
P_rot = P_rot_handle(G1rot,G2rot);
I_rot = trapz2(G1, P_rot, 1);

theta2 = angle(U(2, 1) + 1i * U(2, 2));
G2rot=G23N(N+1:2*N,N+1:2*N)*cos(theta2)-G13N(N+1:2*N,N+1:2*N)*sin(theta2);
G1rot=G13N(N+1:2*N,N+1:2*N)*cos(theta2)+G23N(N+1:2*N,N+1:2*N)*sin(theta2);
P_rot2 = P_rot_handle(G1rot,G2rot);
I_rot2 = trapz2(G1, P_rot2, 1);
end

function I = trapz2(X,Y,dimm)

if dimm==2
I=sum((Y(:,2:end)+Y(:,1:end-1))/2.*(X(:,2:end)-X(:,1:end-1)),2);
else
I=sum((Y(2:end,:)+Y(1:end-1,:))/2.*(X(2:end,:)-X(1:end-1,:)),1);
end

end

% Process sigma_psd, get parameters if needed
function [sigma_psd2, psd_blur, psd_k, profile] = process_psd(sigma_psd, z, single_dim_psd, ...
    pad_size, profile)

% Get auto params if there is a relevant parameter to be calculated.
auto_params = profile.lambda_thr3D == profile.NO_VALUE || ...
              profile.mu2 == profile.NO_VALUE || ...
              (profile.denoise_residual && ...
                (profile.lambda_thr3D_re == profile.NO_VALUE || ...
                profile.mu2_re == profile.NO_VALUE) ...
              );


% Calculate the correlation kernel from the sigma_psd in order
% to resize the sigma_psd. (skip if we are not resizing the sigma_psd)
if (profile.denoise_residual || auto_params) && ~single_dim_psd
    temp_kernel = real(fftshift(fftshift(ifft2(sqrt(sigma_psd / (size(sigma_psd, 1) ...
                                                                 * size(sigma_psd, 2)))), 1), 2));
end

% We need a bigger sigma_psd if we are denoising residuals
if profile.denoise_residual && pad_size(1) && ~single_dim_psd
    
    extended_size = [size(z, 1) + pad_size(1)*2, size(z, 2) + pad_size(2) * 2];    
    % bigger sigma_psd
    sigma_psd2 = abs(fft2(temp_kernel, extended_size(1), extended_size(2))).^2 * ...
                     size(z, 1) * size(z, 2);
else
    sigma_psd2 = sigma_psd;
end

if auto_params && ~single_dim_psd
    % Estimate parameters based on the sigma_psd
    
    minus_size = ceil((size(z) - 65) / 2);
    temp_kernel65 = temp_kernel(1+minus_size(1):minus_size(1)+65, ...
                                1+minus_size(2):minus_size(2)+65, :);
    
    % Normalize kernels to var 1
    for i = 1:size(temp_kernel65, 3)
        temp_kernel65(:, :, i) = temp_kernel65(:, :, i) ./ sqrt(sum(sum(temp_kernel65(:, :, i).^2)));
    end
    
    PSD65 = abs(fft2(temp_kernel65, 65, 65)).^2 * 65 * 65;
    
    % Parameter estimation
    [lambda_thr3D, mu2, lambda_thr3D_re, mu2_re] = estimate_psd_parameters(PSD65);

else
    % For white noise, the result of the estimation is this.
    % No need to create a sigma_psd and run the script just for that.
    lambda_thr3D = 3.0;
    mu2 = 0.4;
    lambda_thr3D_re = 2.5;
    mu2_re = 3.6;
end

% Ensure sigma_psd resized to Nf is usable by convolving it a bit
if(profile.Nf > 0)
    psd_blur = process_psd_for_nf(sigma_psd, [], profile);
    psd_k = fspecial('gaussian',[1+2*(floor(0.5*size(psd_blur, 1)/profile.Nf)), 1], ...
                                 1+2*(floor(0.5*size(psd_blur, 1)/profile.Nf)) / 20) * ...
                                 fspecial('gaussian',[1, 1+2*(floor(0.5*size(psd_blur, 2)/profile.Nf))], ...
                                 1+2*(floor(0.5*size(psd_blur, 2)/profile.Nf)) / 20);
    psd_k = psd_k/sum(psd_k(:));

    psd_blur = convn(padarray(psd_blur, (size(psd_k) - 1) / 2, 'circular'), psd_k, 'valid');
else
    psd_blur = sigma_psd;
end

% Replace things which had no value previously
if profile.lambda_thr3D == profile.NO_VALUE; profile.lambda_thr3D = lambda_thr3D; end
if profile.mu2 == profile.NO_VALUE; profile.mu2 = mu2; end
if profile.lambda_thr3D_re == profile.NO_VALUE; profile.lambda_thr3D_re = lambda_thr3D_re; end
if profile.mu2_re == profile.NO_VALUE; profile.mu2_re = mu2_re; end

end

function [t_forward, t_inverse, hadper_trans_single_den, inverse_hadper_trans_single_den, Wwin2D] = ...
    get_transforms(profile, stage_ht)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Create transform matrices, etc.
%%%%

if stage_ht
    % get (normalized) forward and inverse transform matrices
    [t_forward, t_inverse] = get_transf_matrix(profile.N1, profile.transform_2D_HT_name, profile.decLevel); 
else
    % get (normalized) forward and inverse transform matrices
    [t_forward, t_inverse] = get_transf_matrix(profile.N1_wiener, profile.transform_2D_Wiener_name, 0); 
end

if ((strcmp(profile.transform_3rd_dim_name, 'haar') == 1) || ...
        (strcmp(profile.transform_3rd_dim_name(end-2:end), '1.1') == 1))
    %%% If Haar is used in the 3-rd dimension, then a fast internal transform is used,
    %%% thus no need to generate transform matrices.
    hadper_trans_single_den         = {};
    inverse_hadper_trans_single_den = {};
else
    %%% Create transform matrices. The transforms are later applied by
    %%% matrix-vector multiplication for the 1D case.
    for hpow = 0:ceil(log2(max(profile.N2,profile.N2_wiener)))
        h = 2^hpow;
        [Tfor3rd, Tinv3rd]   = get_transf_matrix(h, profile.transform_3rd_dim_name, 0);
        hadper_trans_single_den{h}         = single(Tfor3rd);
        inverse_hadper_trans_single_den{h} = single(Tinv3rd');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 2D Kaiser windows used in the aggregation of block-wise estimates
%%%%
% hardcode the window function so that the signal processing toolbox is not needed by default
if profile.beta_wiener==2 && profile.beta==2 && profile.N1_wiener==8 && profile.N1==8
    Wwin2D = [ 0.1924    0.2989    0.3846    0.4325    0.4325    0.3846    0.2989    0.1924;
        0.2989    0.4642    0.5974    0.6717    0.6717    0.5974    0.4642    0.2989;
        0.3846    0.5974    0.7688    0.8644    0.8644    0.7688    0.5974    0.3846;
        0.4325    0.6717    0.8644    0.9718    0.9718    0.8644    0.6717    0.4325;
        0.4325    0.6717    0.8644    0.9718    0.9718    0.8644    0.6717    0.4325;
        0.3846    0.5974    0.7688    0.8644    0.8644    0.7688    0.5974    0.3846;
        0.2989    0.4642    0.5974    0.6717    0.6717    0.5974    0.4642    0.2989;
        0.1924    0.2989    0.3846    0.4325    0.4325    0.3846    0.2989    0.1924];
else
    if stage_ht
         % Kaiser window used in the aggregation of the HT part
        Wwin2D = kaiser(profile.N1, profile.beta) * kaiser(profile.N1, profile.beta)';
    else
        % Kaiser window used in the aggregation of the Wiener filt. part
        Wwin2D = kaiser(profile.N1_wiener, profile.beta_wiener) * kaiser(profile.N1_wiener, profile.beta_wiener)';
    end
end

end
