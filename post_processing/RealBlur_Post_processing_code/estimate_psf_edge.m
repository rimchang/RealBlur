function psf = estimate_psf_edge(blurred, latent, psf_size, reg_strength)
  L = psf2otf([0,-1,0;-1,4,-1;0,-1,0], size(blurred)); % laplacian derivative
  bx = imfilter(blurred, [0,-1,1], 'replicate');
  by = imfilter(blurred, [0;-1;1], 'replicate');
  lx = imfilter(latent, [0,-1,1], 'replicate');
  ly = imfilter(latent, [0;-1;1], 'replicate');
  ll = imfilter(lx, [1,-1,0], 0) + imfilter(ly, [1;-1;0], 0);
  Bx = fft2(bx);
  By = fft2(by);
  Lx = fft2(lx);
  Ly = fft2(ly);
  K = (conj(Lx).*Bx + conj(Ly).*By) ./ (conj(Lx).*Lx + conj(Ly).*Ly + reg_strength.*L);% + reg_strength); CLS reconstruction
  psf = otf2psf(K, psf_size);
  
  psf(psf < max(psf(:))*0.05) = 0;
  psf = psf / sum(psf(:));
end
