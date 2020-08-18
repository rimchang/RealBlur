!/bin/bash

mkdir dataset
tar -zxvf BSD_B_Centroid.tar.gz -C ./dataset/
tar -zxvf GOPRO_Large.zip -C ./dataset/
tar -zxvf RealBlur.tar.gz -C ./dataset/
tar -zxvf trained_model.tar.gz -C ./dataset/


ln -s `pwd`/dataset/BSD_B_Centroid ./DeblurGANv2/dataset/
ln -s `pwd`/dataset/BSD_B_Centroid ./SRN-Deblur/training_set/
ln -s `pwd`/dataset/RealBlur-J_ECC_IMCORR_centroid_itensity_ref ./DeblurGANv2/dataset/
ln -s `pwd`/dataset/RealBlur-J_ECC_IMCORR_centroid_itensity_ref ./SRN-Deblur/training_set/
ln -s `pwd`/dataset/RealBlur-J_ECC_IMCORR_centroid_itensity_ref ./SRN-Deblur/testing_set/
ln -s `pwd`/dataset/RealBlur-R_BM3D_ECC_IMCORR_centroid_itensity_ref ./DeblurGANv2/dataset/
ln -s `pwd`/dataset/RealBlur-R_BM3D_ECC_IMCORR_centroid_itensity_ref ./SRN-Deblur/training_set/
ln -s `pwd`/dataset/RealBlur-R_BM3D_ECC_IMCORR_centroid_itensity_ref ./SRN-Deblur/testing_set/

cp dataset/DeblurGAN-v2/checkpoints/* DeblurGANv2/checkpoints -r
cp dataset/SRN-Deblur/checkpoints/* SRN-Deblur/checkpoints -r
