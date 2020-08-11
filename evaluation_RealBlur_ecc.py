import argparse
import glob
import skimage
from skimage import io
from skimage import color
import os
import numpy as np
from skimage.metrics import structural_similarity
from skimage.util.dtype import dtype_range
from multiprocessing import Pool
from skimage import util
import cv2

parser = argparse.ArgumentParser(description='eval arg')
parser.add_argument('--result_dir', type=str, default='../../../result/RealBlur_J')
parser.add_argument('--gt_root', type=str, default='RealBlur-J_ECC_IMCORR_centroid_itensity_ref')
parser.add_argument('--core', type=int, default=4)
args = parser.parse_args()



def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  z = deblurred
  c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

  warp_mode = cv2.MOTION_HOMOGRAPHY
  warp_matrix = np.eye(3, 3, dtype=np.float32)

  # Specify the number of iterations.
  number_of_iterations = 100

  termination_eps = 0

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              number_of_iterations, termination_eps)

  # Run the ECC algorithm. The results are stored in warp_matrix.
  (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

  target_shape = x.shape
  shift = warp_matrix

  zr = cv2.warpPerspective(
    zs,
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_REFLECT)

  cr = cv2.warpPerspective(
    np.ones_like(zs, dtype='float32'),
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0)

  zr = zr * cr
  xr = x * cr

  return zr, xr, cr, shift


def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)


def im2uint8(image):
  image = np.clip(image * 255, 0, 255) + 0.5  # round color value
  image = image.astype('uint8')
  return image


def evaluation_folder(args_list):
  model, gt_root, out_root = args_list
  print(model, gt_root, out_root)

  out_root = os.path.join(out_root, model.split('/')[-1])
  if not os.path.exists(out_root):
    os.mkdir(out_root)

  imgList = glob.glob(model + '/*.png')

  cnt = 0
  deblur_psnr_list = []
  deblur_ssim_list = []
  blur_psnr_list = []
  blur_ssim_list = []
  f = open(os.path.join(out_root, 'psnr.txt'), 'wt')
  for j, img_path in enumerate(imgList):
    print(img_path)
    if 'fail' in img_path or 'ker' in img_path:
      continue

    img_name_split = img_path.split('/')[-1]
    img_name_split = img_name_split.split('_')

    scene_name = img_name_split[0]

    if scene_name == 'scene244':
      scene_name = 'scene244_reflect'

    img_name = img_name_split[-1]

    deblurred = io.imread(img_path).astype('float32') / 255
    blurred = io.imread(os.path.join(gt_root, scene_name, 'blur', 'blur_' + img_name)).astype('float32') / 255
    gt = io.imread(os.path.join(gt_root, scene_name, 'gt', 'gt_' + img_name)).astype('float32') / 255

    aligned_deblurred, aligned_xr1, cr1, shift = image_align(deblurred, gt)
    aligned_blurred, aligned_xr2, cr2, shift = image_align(blurred, gt)
    aligned_blurred = blurred
    aligned_xr2 = gt
    cr2 = np.ones_like(blurred, dtype='float32')

    # it is recomended by nah et al.


    deblur_ssim_pre, deblur_ssim_map = structural_similarity(aligned_xr1, aligned_deblurred, multichannel=True, gaussian_weights=True,
                                        use_sample_covariance=False, data_range = 1.0, full=True)
    deblur_ssim_map = deblur_ssim_map * cr1

    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    deblur_ssim = deblur_ssim_map[pad:-pad,pad:-pad,:]
    crop_cr1 = cr1[pad:-pad,pad:-pad,:]
    deblur_ssim = deblur_ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
    deblur_ssim = np.mean(deblur_ssim)

    #blur_ssim = structural_similarity(aligned_xr2, aligned_blurred, multichannel=True, gaussian_weights=True,
    #                                  use_sample_covariance=False, data_range = 1.0, full=True)

    blur_ssim = 0
    print(deblur_ssim, deblur_ssim_pre)

    # only compute mse on valid region
    deblur_psnr = compute_psnr(aligned_xr1, aligned_deblurred, cr1, data_range=1)
    blur_psnr = compute_psnr(aligned_xr2, aligned_blurred, cr2, data_range=1)

    deblur_psnr_list.append(deblur_psnr)
    deblur_ssim_list.append(deblur_ssim)
    blur_psnr_list.append(blur_psnr)
    blur_ssim_list.append(blur_ssim)

    vis_image = np.concatenate([aligned_blurred, aligned_deblurred, aligned_xr1], axis=1)

    deblur_out_name = os.path.join(out_root, '_'.join(img_name_split))
    vis_img_out_name = 'vis_%s_blur_%s_PSNR_%5.5f_%5.5f_SSIM_%5.5f_%5.5f.jpg' % (scene_name, img_name[:-4], blur_psnr, deblur_psnr, blur_ssim, deblur_ssim)
    vis_img_out_name = os.path.join(out_root, vis_img_out_name)
    io.imsave(deblur_out_name, im2uint8(deblurred))
    io.imsave(vis_img_out_name, im2uint8(vis_image))
    f.write("%s %5.5f %5.5f\n" % ('_'.join(img_name_split), deblur_psnr, deblur_ssim))
    cnt += 1

  f2 = open(os.path.join(out_root, 'result.txt'), 'wt')
  f2.write("deblur_psnr : %4.4f \n" % np.mean(deblur_psnr_list))
  f2.write("deblur_ssim : %4.4f \n" % np.mean(deblur_ssim_list))
  f2.write("blur_psnr : %4.4f \n" % np.mean(blur_psnr_list))
  f2.write("blur_ssim : %4.4f \n" % np.mean(blur_ssim_list))
  f2.write("cnt : %4.4f \n" % cnt)
  f2.close()
  f.close()

if __name__ == '__main__':

  if skimage.__version__ != '0.17.2':
    print("please use skimage==0.17.2 and python3")
    exit()

  result_dir = args.result_dir
  model_list = glob.glob(result_dir + '/*')
  models = []
  for model_path in model_list:
    if os.path.isdir(model_path):
      models.append(model_path)
  print(models)
  if len(models) == 0:
    models = [result_dir+'/']

  gt_root = args.gt_root
  gt_roots = [gt_root for j in models]

  out_root = './result_%s' % (models[0].split('/')[-2])
  if not os.path.exists(out_root):
    os.mkdir(out_root)
  out_roots = [out_root for j in models]

  with Pool(args.core) as p:
    p.map(evaluation_folder, zip(models, gt_roots, out_roots))
