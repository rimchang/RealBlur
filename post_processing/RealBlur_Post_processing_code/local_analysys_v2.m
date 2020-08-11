clear all;

%rootdir = '/media/ubuntu/My Book/processed_SRD_data_v4/RealBlur-J/RealBlur-J_ECC_NO_IMCORR_centroid_local_analysis';
%rootdir = '/media/ubuntu/My Book/processed_SRD_data_v4/RealBlur-J/RealBlur-J_ECC_IMCORR_centroid_local_analysis';
rootdir = '/media/ubuntu/My Book/processed_SRD_data_v4/RealBlur-J_only_ref/RealBlur-J_ECC_local_analysis';


scene_dir = dir(rootdir);
scene_dir=scene_dir(~ismember({scene_dir.name},{'.','..'}));


fileID = fopen('RealBlur_J_test_list_no_ref.txt','r');
test_data_list = textscan(fileID,'%s %s\n');
test_data_list = test_data_list{1};

fileID = fopen('RealBlur_J_train_list.txt','r');
train_data_list = textscan(fileID,'%s %s\n');
train_data_list = train_data_list{1};

data_list = [test_data_list; train_data_list];

total_x_diff = 0;
total_y_diff = 0;
total_distance = 0;
cnt = 0;
for scene_i = 1:234
    scene = scene_dir(scene_i);
    
    if contains(scene.name, 'scene') == 0
        continue;
    end
    
    
    temp_line = data_list{1};
    
    basedir = fullfile(rootdir, scene.name);
    kerfolder = fullfile(basedir,'kernel');
    fileID = fopen(fullfile(kerfolder, 'displacement.txt'),'r');
    dis = textscan(fileID,'%s %s %s %f %s %s %f\n');
    
    kername= dis{1,1};
    x_diff = dis{1,4};
    y_diff = dis{1,7};
    for i=1:size(dis{1,1},1)
        if contains(kername{i,1}, 'sub') == 0
            continue;
        end
        
        img_n = kername{i};
        img_n = strsplit(img_n,'_');
        img_n = img_n{2};
        
        data_line = strrep(temp_line, 'scene230', scene.name);
        data_line = strrep(data_line , 'gt_7.png', sprintf('gt_%s.png', img_n));
        
        if sum(strcmp(data_list, data_line)) == 0
            continue
        end
        
        total_x_diff = total_x_diff + x_diff(i);
        total_y_diff = total_y_diff + y_diff(i);
        total_distance = total_distance + sqrt(x_diff(i)^2 +y_diff(i)^2);
        cnt = cnt +1 ;
    end
    
end

avg_dis = total_distance / cnt;
disp(avg_dis);



