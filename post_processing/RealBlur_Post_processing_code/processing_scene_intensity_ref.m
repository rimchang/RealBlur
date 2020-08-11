function [ leftimg_cell ,rightimg_cell ] = processing_scene_intensity_ref( leftimg_cell, rightimg_cell, outbasedir)

        gtout = fullfile(outbasedir,'gt');
        blurout = fullfile(outbasedir,'blur');
        Anaglyphout = fullfile(outbasedir,'Anaglyph');
        gifout = fullfile(outbasedir,'gif');
        kernelout = fullfile(outbasedir,'kernel');
        
        if (~exist(outbasedir, 'dir')); mkdir(outbasedir); end
        if (~exist(gtout, 'dir')); mkdir(gtout); end
        if (~exist(blurout, 'dir')); mkdir(blurout); end
        if (~exist(Anaglyphout, 'dir')); mkdir(Anaglyphout); end
        if (~exist(gifout, 'dir')); mkdir(gifout); end
        if (~exist(kernelout, 'dir')); mkdir(kernelout); end 
        
        
        for j = 1:1
            gtoutname = fullfile(gtout, sprintf('gt_%d.png', j));
            bluroutname = fullfile(blurout, sprintf('blur_%d.png', j));
            Anaglyphoutname = fullfile(Anaglyphout, sprintf('Anaglyph_%d.jpg', j));
            gifoutname = fullfile(gifout, sprintf('gif_%d.gif', j));
                        
            blurred = im2double(rightimg_cell{1, j});
            latent = im2double(leftimg_cell{1, j});
            
            [blurred_mean, blurred_std] = compute_mean_std(blurred);
            [latent_mean, latent_std] = compute_mean_std(latent);
                
            alpha = blurred_std./latent_std;
            beta = blurred_mean - alpha .* latent_mean;
            latent = alpha .* latent + beta;
                   
            
            leftimg_cell{1, j} = latent;
            rightimg_cell{1, j} = blurred;
            
            
            writeGIF(im2double(latent),im2double(blurred), gifoutname);
            imwrite(stereoAnaglyph(latent,blurred), Anaglyphoutname);
            imwrite(latent, gtoutname);
            imwrite(blurred, bluroutname);              
        end
        
        parfor j = 2:size(leftimg_cell,2)
            gtoutname = fullfile(gtout, sprintf('gt_%d.png', j));
            bluroutname = fullfile(blurout, sprintf('blur_%d.png', j));
            Anaglyphoutname = fullfile(Anaglyphout, sprintf('Anaglyph_%d.jpg', j));
            gifoutname = fullfile(gifout, sprintf('gif_%d.gif', j));            
            
            latent = im2double(leftimg_cell{1,j});
            blurred = im2double(rightimg_cell{1,j});
                        
            % intensity alignment using reference_img
            latent = alpha .* latent + beta;
                                   
            leftimg_cell{1, j} = latent;
            rightimg_cell{1, j} = blurred;
                        
            writeGIF(latent, blurred, gifoutname);
            imwrite(stereoAnaglyph(latent, blurred), Anaglyphoutname);
            imwrite(latent, gtoutname);
            imwrite(blurred, bluroutname);           
        end
        
end

