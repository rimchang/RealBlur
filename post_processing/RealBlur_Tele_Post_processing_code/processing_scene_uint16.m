function [ leftimg_cell ,rightimg_cell ] = processing_scene_uint16( leftimg_cell, rightimg_cell, outbasedir)

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
            
            leftimg = leftimg_cell{1, j};
            rightimg = rightimg_cell{1, j};
            
            cmovingReg = im2double(rightimg);
            cfixedReg = im2double(leftimg);
            
            writeGIF(im2double(cfixedReg),im2double(cmovingReg), gifoutname);
            imwrite(stereoAnaglyph(cfixedReg,cmovingReg), Anaglyphoutname);
            imwrite(im2uint16(cfixedReg), gtoutname);
            imwrite(im2uint16(cmovingReg), bluroutname);              
        end
        
        parfor j = 2:size(leftimg_cell,2)
            gtoutname = fullfile(gtout, sprintf('gt_%d.png', j));
            bluroutname = fullfile(blurout, sprintf('blur_%d.png', j));
            Anaglyphoutname = fullfile(Anaglyphout, sprintf('Anaglyph_%d.jpg', j));
            gifoutname = fullfile(gifout, sprintf('gif_%d.gif', j));            
            
            leftimg = leftimg_cell{1,j};
            rightimg = rightimg_cell{1,j};
            
            latent = im2double(leftimg);
            blurred = im2double(rightimg);   
            
            blurredReg = blurred;
            latentReg = latent;            
            
            writeGIF(latentReg, blurredReg, gifoutname);
            imwrite(stereoAnaglyph(latentReg, blurredReg), Anaglyphoutname);
            imwrite(im2uint16(latentReg), gtoutname);
            imwrite(im2uint16(blurredReg), bluroutname);           
        end
        
end

