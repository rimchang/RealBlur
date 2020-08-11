function [ new_leftimg_cell ,new_rightimg_cell ] = processing_scene_bm3d( leftimg_cell, rightimg_cell, outbasedir)
        addpath('bm3d');
        gtout = fullfile(outbasedir,'gt');
        blurout = fullfile(outbasedir,'blur');
        Anaglyphout = fullfile(outbasedir,'Anaglyph');
        gifout = fullfile(outbasedir,'gif');
        noiseout = fullfile(outbasedir,'noise');
        noisetxtout = fullfile(outbasedir,'noise', 'noise.txt');
        
        if (~exist(outbasedir, 'dir')); mkdir(outbasedir); end
        if (~exist(gtout, 'dir')); mkdir(gtout); end
        if (~exist(blurout, 'dir')); mkdir(blurout); end
        if (~exist(Anaglyphout, 'dir')); mkdir(Anaglyphout); end
        if (~exist(gifout, 'dir')); mkdir(gifout); end
        if (~exist(noiseout, 'dir')); mkdir(noiseout); end 
        
        new_leftimg_cell = cell(1, size(leftimg_cell, 2));
        new_rightimg_cell = cell(1, size(leftimg_cell, 2));
        
       
        fid = fopen(noisetxtout, 'wt');
        for i = 1:size(leftimg_cell,2)
            latent = lin2rgb(leftimg_cell{1,i});
            temp_noise_level = NoiseEstimation(latent, 8);
            fprintf(fid, sprintf('%d : %4.4f \n', i, temp_noise_level));
        end
        fclose(fid);
        
        
        parfor j = 1:size(leftimg_cell,2)
            gtoutname = fullfile(gtout, sprintf('gt_%d.png', j));
            bluroutname = fullfile(blurout, sprintf('blur_%d.png', j));
            Anaglyphoutname = fullfile(Anaglyphout, sprintf('Anaglyph_%d.jpg', j));
            gifoutname = fullfile(gifout, sprintf('gif_%d.gif', j));              
            noiseoutname = fullfile(noiseout, sprintf('noise_%d.png', j)); 
            

            latent = lin2rgb(leftimg_cell{1,j});
            noise_level = NoiseEstimation(latent, 8);        
            %disp([noise_level, j]);
            latentsrgb = CBM3D(latent, noise_level*1.5);
            latentReg = rgb2lin(latentsrgb);
            %imshow(cat(2, latent, latentsrgb))
            
            blurredReg = rightimg_cell{1,j};
            
            new_leftimg_cell{1, j} = latentReg;
            new_rightimg_cell{1, j} = blurredReg;
            
            writeGIF(latentReg, blurredReg, gifoutname);
            imwrite(stereoAnaglyph(latentReg, blurredReg), Anaglyphoutname);
            imwrite(latentReg, gtoutname);
            imwrite(blurredReg, bluroutname);    
            imwrite(cat(2, latent, latentsrgb), noiseoutname);
        end
        
end

