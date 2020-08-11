function [y_est] = BM3DDEB(z, sigma_psd, psf, profile)
% BM3D Deblurring. Performs regularization, then denoising, for both
% hard-thresholding and wiener.
%
% Arguments:
% 
% z: Noisy blurred image. either MxN or MxNxC where C is the channel count.
% 
% sigmaPSD: Noise PSD, either MxN or MxNxC (different PSDs for different channels)
% OR
% sigmaPSD: Noise standard deviation, either float, or length C list of floats
% 
% psf: Blur point-spread function in space domain.
% 
% --optional--
%
% profile: Settings for BM3D: BM3DProfile object or a string.
% (default='np')
% ('np', 'refilter', 'vn', 'vn_old', 'high', 'deb')
% 

if ~exist('profile','var')
    profile         = 'np'; %% default profile
end


%%%% Resize PSD
single_dim_psd = size(sigma_psd, 1) == 1 || size(sigma_psd, 2) == 1;
if single_dim_psd
    sigma_psd = ones(size(z)) .* permute(sigma_psd(:), [2, 3, 1]) .^ 2 * size(z, 1) * size(z, 2);
end

%%%% Fixed regularization parameters (obtained empirically after a rough optimization)
Regularization_alpha_RI = 4e-4;
Regularization_alpha_RWI = 5e-3;

%%%% Step 1.1. Regularized Inversion
big_v  = zeros(size(z, 1), size(z, 2));
big_v(1:size(psf, 1), 1:size(psf, 2)) = psf;
big_v = circshift(big_v, -round([(size(psf, 1) - 1) / 2, (size(psf, 2) - 1) / 2])); % pad PSF with zeros to whole image domain, and center it
V = fft2(big_v); 
RI = conj(V)./( (abs(V).^2) + Regularization_alpha_RI .* sigma_psd + eps); % Transfer Matrix for RI    %% Standard Tikhonov Regularization
zRI = real(ifft2( fft2(z).* RI ));   % Regularized Inverse Estimate (RI OBSERVATION)
PSD_zRI = sigma_psd .* abs(RI).^2;

y_hat = BM3D(zRI, PSD_zRI, profile, BM3DProfile.HARD_THRESHOLDING);

%%%% Step 2.1. Regularized Wiener Inversion
Wiener_Pilot = abs(fft2(double(y_hat)));   %%% Wiener reference estimate
RWI  = conj(V).*Wiener_Pilot.^2./(Wiener_Pilot.^2.*(abs(V).^2) + Regularization_alpha_RWI .* sigma_psd + eps);   % Transfer Matrix for RWI (uses standard regularization 'a-la-Tikhonov')
zRWI = real(ifft2(fft2(z).*RWI));   % RWI OBSERVATION
PSD_zRWI = sigma_psd .* abs(RWI).^2;

y_est = BM3D(zRWI, PSD_zRWI, profile, y_hat);

end