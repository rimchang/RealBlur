function [ mean_v, std_v ] = compute_mean_std( img)


img_r = img(:,:,1);
img_g = img(:,:,2);
img_b = img(:,:,3);

mean_r = mean(img_r(:));
mean_g = mean(img_g(:));
mean_b = mean(img_b(:));

std_r = std(img_r(:));
std_g = std(img_g(:));
std_b = std(img_b(:));

mean_v = [mean_r mean_g mean_b];
std_v = [std_r std_g std_b];

mean_v = reshape(mean_v, 1, 1, 3);
std_v = reshape(std_v, 1, 1, 3);
end

