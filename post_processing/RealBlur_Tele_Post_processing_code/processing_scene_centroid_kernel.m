function [ new_homography_cell ] = processing_scene_centroid_kernel( leftimg_cell, rightimg_cell, homography_cell, outbasedir, warping_params, ker_size)

        if nargin > 5
            ker_size = ker_size;
        else
            ker_size = 101;
        end

        gtout = fullfile(outbasedir,'gt');
        blurout = fullfile(outbasedir,'blur');
        Anaglyphout = fullfile(outbasedir,'Anaglyph');
        gifout = fullfile(outbasedir,'gif');
        kernelout = fullfile(outbasedir,'kernel');
        tformout = fullfile(outbasedir,'tform');
        
        if (~exist(outbasedir, 'dir')); mkdir(outbasedir); end
        if (~exist(gtout, 'dir')); mkdir(gtout); end
        if (~exist(blurout, 'dir')); mkdir(blurout); end
        if (~exist(Anaglyphout, 'dir')); mkdir(Anaglyphout); end
        if (~exist(gifout, 'dir')); mkdir(gifout); end
        if (~exist(kernelout, 'dir')); mkdir(kernelout); end
        if (~exist(tformout, 'dir')); mkdir(tformout); end
        
        new_homography_cell = cell(1, size(leftimg_cell, 2));
        
        for j = 1:1
            gtoutname = fullfile(gtout, sprintf('gt_%d.png', j));
            bluroutname = fullfile(blurout, sprintf('blur_%d.png', j));
            Anaglyphoutname = fullfile(Anaglyphout, sprintf('Anaglyph_%d.jpg', j));
            gifoutname = fullfile(gifout, sprintf('gif_%d.gif', j));
            tformoutname = fullfile(tformout, sprintf('tform_%d.mat', j));
            
            ori_fixed = im2double(rightimg_cell{1, j});
            ori_moving = im2double(leftimg_cell{1, j});
                        
            new_homography_cell{1,j} = homography_cell{1,j};
            
            identity = [1 0 0; 0 1 0; 0 0 1;];
            fixedReg = warping_with_resize_undistortion(ori_fixed, identity, warping_params.camera_param.params.CameraParameters2, warping_params);
            movingReg = warping_with_resize_undistortion(ori_moving, new_homography_cell{1,j}, warping_params.camera_param.params.CameraParameters1, warping_params);
                                        
            [top, bottom, left, right] = bboxFromHomography(movingReg, new_homography_cell{1,j});         
            movingReg = movingReg(top:bottom, left:right, :);
            fixedReg = fixedReg(top:bottom, left:right, :);
            
            writeGIF(im2double(movingReg),im2double(fixedReg), gifoutname);
            imwrite(stereoAnaglyph(movingReg,fixedReg), Anaglyphoutname);
            imwrite(movingReg, gtoutname);
            imwrite(fixedReg, bluroutname);
            save_parfor(tformoutname, new_homography_cell{1,j});
        end
        
        parfor j = 2:size(leftimg_cell,2)
            gtoutname = fullfile(gtout, sprintf('gt_%d.png', j));
            bluroutname = fullfile(blurout, sprintf('blur_%d.png', j));
            Anaglyphoutname = fullfile(Anaglyphout, sprintf('Anaglyph_%d.jpg', j));
            AnaglyphRefoutname = fullfile(Anaglyphout, sprintf('Anaglyph_%d_ref.jpg', j));
            gifoutname = fullfile(gifout, sprintf('gif_%d.gif', j));
            kerneloutname = fullfile(kernelout, sprintf('kernel_%d.jpg', j));
            tformoutname = fullfile(tformout, sprintf('tform_%d.mat', j));
                        
            ori_latent = im2double(leftimg_cell{1,j});
            ori_blurred = im2double(rightimg_cell{1,j});           
            
            identity = [1 0 0; 0 1 0; 0 0 1;];
            blurred = warping_with_resize_undistortion(ori_blurred, identity,  warping_params.camera_param.params.CameraParameters2, warping_params);
            latent = warping_with_resize_undistortion(ori_latent, homography_cell{1,j}, warping_params.camera_param.params.CameraParameters1, warping_params);
                    
            % crop invalid region for estimating kernel
            [top, bottom, left, right] = bboxFromHomography(latent, homography_cell{1,j});            
            latent_PRE = latent(top:bottom, left:right, :);
            blurred_PRE = blurred(top:bottom, left:right, :);
            
            psf_height = ker_size;
            psf_width = ker_size;
            psf = estimate_psf_edge(blurred_PRE, latent_PRE, [psf_height, psf_width], 1000.1);

            rpsf = real(psf);
            rpsf = rpsf / sum(rpsf(:));
            rpsf = rot90(rpsf,2);
            
            % compute tanslation matrix
            [temp2, temp2, x_diff, y_diff] = warping_from_corr_kernel(latent_PRE, blurred_PRE, rpsf);
            H = [1 0 0; 0 1 0; x_diff y_diff 1];
            
            % visulalize blur kernel
            vis_rpsf = zeros(psf_height, psf_width, 3);      
            vis_rpsf(:,:,1) = rpsf/max(max(rpsf));
            vis_rpsf(:,:,2) = rpsf/max(max(rpsf));
            vis_rpsf(:,:,3) = rpsf/max(max(rpsf));            
            

            new_homography_cell{1,j} = homography_cell{1,j} * H;
            
            % warping
            latentReg = warping_with_resize_undistortion(ori_latent, new_homography_cell{1,j}, warping_params.camera_param.params.CameraParameters1, warping_params);
            blurredReg = blurred;
                                    
            [top, bottom, left, right] = bboxFromHomography(latent, new_homography_cell{1,j});         
            latentReg = latentReg(top:bottom, left:right, :);
            blurredReg = blurredReg(top:bottom, left:right, :);  
            
            writeGIF(latentReg, blurredReg, gifoutname);
            imwrite(stereoAnaglyph(latent_PRE, blurred_PRE), AnaglyphRefoutname);
            imwrite(stereoAnaglyph(latentReg, blurredReg), Anaglyphoutname);
            imwrite(latentReg, gtoutname);
            imwrite(blurredReg, bluroutname);
            imwrite(vis_rpsf, kerneloutname);
            save_parfor(tformoutname, new_homography_cell{1,j});
            
        end
        
end

