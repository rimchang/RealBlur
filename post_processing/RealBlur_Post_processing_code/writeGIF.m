function writeGIF(latent, blurred, dst)
    [IND1,map1] = rgb2ind(latent, 256);
    %IND = im2double(IND);
    imwrite(IND1, map1, dst,'gif', 'Loopcount',inf,'DelayTime',3); 
    [IND2,map2] = rgb2ind(blurred, 256);
    %IND = im2double(IND);
    imwrite(IND2, map2, dst,'gif','WriteMode','append','DelayTime',3); 

end

