function processing_scene_local_centroid_kernel( leftimg_cell, rightimg_cell, outbasedir)

        gtout = fullfile(outbasedir,'gt');
        blurout = fullfile(outbasedir,'blur');
        Anaglyphout = fullfile(outbasedir,'Anaglyph');
        gifout = fullfile(outbasedir,'gif');
        kernelout = fullfile(outbasedir,'kernel');
        resulttxtout = fullfile(outbasedir,'kernel', 'displacement.txt');
                
        if (~exist(outbasedir, 'dir')); mkdir(outbasedir); end
        if (~exist(gtout, 'dir')); mkdir(gtout); end
        if (~exist(blurout, 'dir')); mkdir(blurout); end
        if (~exist(Anaglyphout, 'dir')); mkdir(Anaglyphout); end
        if (~exist(gifout, 'dir')); mkdir(gifout); end
        if (~exist(kernelout, 'dir')); mkdir(kernelout); end 
        fid = fopen(resulttxtout, 'wt');
        
        
        for j = 1:1
            gtoutname = fullfile(gtout, sprintf('gt_%d.png', j));
            bluroutname = fullfile(blurout, sprintf('blur_%d.png', j));
            Anaglyphoutname = fullfile(Anaglyphout, sprintf('Anaglyph_%d.jpg', j));
            gifoutname = fullfile(gifout, sprintf('gif_%d.gif', j));
            
            leftimg = leftimg_cell{1, j};
            rightimg = rightimg_cell{1, j};
            
            blurred_linear = im2double(rightimg);
            latent_linear = im2double(leftimg);
                        
            cfixedReg = blurred_linear;
            cmovingReg = latent_linear; 
                        
            writeGIF(im2double(cmovingReg),im2double(cfixedReg), gifoutname);
            imwrite(stereoAnaglyph(cmovingReg,cfixedReg), Anaglyphoutname);
            imwrite(cmovingReg, gtoutname);
            imwrite(cfixedReg, bluroutname);              
        end
        
        write_line_cell = {1, size(leftimg_cell,2)};
        parfor j = 2:size(leftimg_cell,2)
            gtoutname = fullfile(gtout, sprintf('gt_%d.png', j));
            bluroutname = fullfile(blurout, sprintf('blur_%d.png', j));
            Anaglyphoutname = fullfile(Anaglyphout, sprintf('Anaglyph_%d.jpg', j));
            gifoutname = fullfile(gifout, sprintf('gif_%d.gif', j));           
            kerneloutname = fullfile(kernelout, sprintf('kernel_%d.jpg', j));    
            
            leftimg = leftimg_cell{1,j};
            rightimg = rightimg_cell{1,j};
            
            latent = im2double(leftimg);
            blurred = im2double(rightimg);
                        
            psf_height = 151;
            psf_width = 151;
            
            % global kernel estimation
            psf = estimate_psf_edge(blurred, latent, [psf_height, psf_width], 1000.1);
            
            
            rpsf = real(psf);
            rpsf = rpsf / sum(rpsf(:));
            rpsf = rot90(rpsf,2);
            
            [sub_latentReg, sub_blurredReg, x_diff, y_diff] = warping_from_corr_kernel(latent, blurred, rpsf);
            
            
            vis_rpsf = zeros(psf_height, psf_width, 3);
            vis_rpsf(:,:,1) = rpsf/max(max(rpsf));
            vis_rpsf(:,:,2) = rpsf/max(max(rpsf));
            vis_rpsf(:,:,3) = rpsf/max(max(rpsf));
            
            write_line_cell{1,j} = sprintf('kernel_%d.jpg x_diff : %4.4f y_diff : %4.4f,', j, x_diff, y_diff);
            
            kerneloutname = fullfile(kernelout, sprintf('kernel_%d.jpg', j));
            imwrite(vis_rpsf, kerneloutname);            
            
            
            % locally analysis
            subimg_height = ceil(size(latent,1)/2);
            subimg_width = ceil(size(latent,2)/2);
            
            sub_latent_cell = {};
            sub_latent_cell{1,1} = latent(1:subimg_height,1:subimg_width,:);
            sub_latent_cell{1,2} = latent(1:subimg_height,subimg_width+1:end,:);
            sub_latent_cell{1,3} = latent(subimg_height+1:end,1:subimg_width,:);
            sub_latent_cell{1,4} = latent(subimg_height+1:end,subimg_width+1:end,:);
            
            sub_blurred_cell = {};
            sub_blurred_cell{1,1} = blurred(1:subimg_height,1:subimg_width,:);
            sub_blurred_cell{1,2} = blurred(1:subimg_height,subimg_width+1:end,:);
            sub_blurred_cell{1,3} = blurred(subimg_height+1:end,1:subimg_width,:);
            sub_blurred_cell{1,4} = blurred(subimg_height+1:end,subimg_width+1:end,:);
            
            for k=1:4
                
                sub_blurred = sub_blurred_cell{1,k};
                sub_latent = sub_latent_cell{1,k};
                psf = estimate_psf_edge(sub_blurred, sub_latent, [psf_height, psf_width], 1000.1);
                
                
                rpsf = real(psf);
                rpsf = rpsf / sum(rpsf(:));
                rpsf = rot90(rpsf,2);
                
                [sub_latentReg, sub_blurredReg, x_diff, y_diff] = warping_from_corr_kernel(sub_latent, sub_blurred, rpsf);
                
                
                vis_rpsf = zeros(psf_height, psf_width, 3);
                vis_rpsf(:,:,1) = rpsf/max(max(rpsf));
                vis_rpsf(:,:,2) = rpsf/max(max(rpsf));
                vis_rpsf(:,:,3) = rpsf/max(max(rpsf));
                
                write_line_cell{1,j} = strcat(write_line_cell{1,j}, sprintf('kernel_%d_sub%d.jpg x_diff : %4.4f y_diff : %4.4f,', j, k, x_diff, y_diff));
                %fprintf(fid, sprintf('kernel_%d_sub%d.jpg x_diff : %4.4f y_diff : %4.4f \n', j, k, abs(x_diff), abs(y_diff)));
                
                kerneloutname = fullfile(kernelout, sprintf('kernel_%d_sub%d.jpg', j, k));
                imwrite(vis_rpsf, kerneloutname);
            end
                        
            
            latentReg = latent;
            blurredReg = blurred;
                        
            writeGIF(latentReg, blurredReg, gifoutname);
            imwrite(stereoAnaglyph(latentReg, blurredReg), Anaglyphoutname);
            imwrite(latentReg, gtoutname);
            imwrite(blurredReg, bluroutname);           
            imwrite(vis_rpsf, kerneloutname);
        end
        
        % write centroid error to txt file
        for j = 2:size(leftimg_cell,2)
            line_split = strsplit(write_line_cell{1,j}, ',');
            for t=1:5
                fprintf(fid, strcat(line_split{1,t},'\n'));
            end            
        end
        
        
        fclose(fid);
end

