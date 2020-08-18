#!/bin/bash

ln -sf `pwd`/dataset/BSD_B_Centroid ./DeblurGANv2/dataset/
ln -sf `pwd`/dataset/BSD_B_Centroid ./SRN-Deblur/training_set/

ln -sf `pwd`/dataset/RealBlur-J_ECC_IMCORR_centroid_itensity_ref ./DeblurGANv2/dataset/
ln -sf `pwd`/dataset/RealBlur-J_ECC_IMCORR_centroid_itensity_ref ./SRN-Deblur/training_set/
ln -sf `pwd`/dataset/RealBlur-J_ECC_IMCORR_centroid_itensity_ref ./SRN-Deblur/testing_set/

ln -sf `pwd`/dataset/RealBlur-R_BM3D_ECC_IMCORR_centroid_itensity_ref ./DeblurGANv2/dataset/
ln -sf `pwd`/dataset/RealBlur-R_BM3D_ECC_IMCORR_centroid_itensity_ref ./SRN-Deblur/training_set/
ln -sf `pwd`/dataset/RealBlur-R_BM3D_ECC_IMCORR_centroid_itensity_ref ./SRN-Deblur/testing_set/

ln -sf `pwd`/dataset/train ./DeblurGANv2/dataset/
ln -sf `pwd`/dataset/train ./SRN-Deblur/training_set/

ln -sf `pwd`/dataset/test ./DeblurGANv2/dataset/
ln -sf `pwd`/dataset/test ./SRN-Deblur/testing_set/

cp dataset/DeblurGAN-v2/checkpoints/* DeblurGANv2/checkpoints -r
cp dataset/SRN-Deblur/checkpoints/* SRN-Deblur/checkpoints -r
