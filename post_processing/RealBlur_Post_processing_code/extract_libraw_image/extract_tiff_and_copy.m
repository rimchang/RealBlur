function [ ] = extract_tiff_and_copy( source_fname, target_fname )
    %[exifdata, exif_struct, nf] = run_exiftool(source_fname);

    copyfile(source_fname, target_fname);

    test = 'dcraw_emu';
    TS=[ '"' test '" -w -W -T -g 1 1 -q 3 -4 "' target_fname '"']; 
    %TS=[ '"' test '" -w -W -T -g 1 1 -fbdd 2 "' temp_file '"']; 
    [status, exifdata] = system(TS); 

    delete(target_fname);

end

