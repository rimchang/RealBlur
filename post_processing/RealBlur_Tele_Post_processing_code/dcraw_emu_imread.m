function [ outimg, exif_struct ] = dcraw_emu_imread(fname, temp_file)

[exifdata, exif_struct, nf] = run_exiftool(fname);

copyfile(fname, temp_file);

test = 'dcraw_emu';
TS=[ '"' test '" -w -W -T -g 1 1 -q 3 -4"' temp_file '"']; 
%TS=[ '"' test '" -w -W -T -g 1 1 -fbdd 2 "' temp_file '"']; 
[status, exifdata] = system(TS); 

tiff_name = strcat(temp_file, '.tiff');
outimg = imread(tiff_name);

end

