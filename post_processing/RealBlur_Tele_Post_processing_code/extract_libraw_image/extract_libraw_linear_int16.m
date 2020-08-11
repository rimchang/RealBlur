rootdir = '../../RealBlur_Tele_Original';
outdir = '../../RealBlur_Tele_libraw_linear_int16';


if (~exist(outdir, 'dir')); mkdir(outdir); end

directorys = dir(rootdir);
directorys=directorys(~ismember({directorys.name},{'.','..'}));

delete(gcp('nocreate'))
p = parpool(6);


for i = 1:size(directorys, 1)
    tic;
    directory = directorys(i);
    basedir = fullfile(rootdir, directory.name);
    leftfolder = fullfile(basedir,'left');
    rightfolder = fullfile(basedir,'right');
    
    outbasedir = fullfile(outdir, directory.name);
    gtout = fullfile(outbasedir,'gt_linear');
    blurout = fullfile(outbasedir,'blur_linear');
    
    
    if (~exist(basedir, 'dir')); mkdir(basedir); end
    if (~exist(gtout, 'dir')); mkdir(gtout); end
    if (~exist(blurout, 'dir')); mkdir(blurout); end
    
    
    leftList = dir(fullfile(leftfolder, '*.JPG'));
    rightList = dir(fullfile(rightfolder, '*.JPG'));
    
    leftnameList = extractfield(leftList, 'name');
    rightnameList = extractfield(rightList, 'name');
    
    leftnameList = sort(leftnameList);
    rightnameList = sort(rightnameList);
    
    parfor j = 1:size(leftnameList, 2)
        gtoutname = fullfile(gtout, sprintf('gt_%d.png', j));
        bluroutname = fullfile(blurout, sprintf('blur_%d.png', j));
        
        % raw-processing
        temp_right_tiff = fullfile(blurout, sprintf('blur_%s.ARW', num2str(j)));
        temp_left_tiff = fullfile(gtout, sprintf('gt_%s.ARW', num2str(j)));
        extract_tiff_and_copy(strrep(fullfile(rightfolder, rightnameList{j}), 'JPG', 'ARW'), temp_right_tiff);
        extract_tiff_and_copy(strrep(fullfile(leftfolder, leftnameList{j}), 'JPG', 'ARW'), temp_left_tiff);
        
    end
    toc;
end

delete(p);