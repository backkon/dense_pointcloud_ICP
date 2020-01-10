clc;clear;
path=pwd;
file = dir(fullfile(path,'*.ply'));    %读取所有ply格式文件
filenames = {file.name}';
filelength = size(filenames,1);        %ply格式文件数

for idx = 1 : filelength               %批处理
    filedir = strcat(path, filenames(idx));
    ptcloud=pcread(filenames{idx});   %ply格式文件用pcread读取
    figure(idx);
    pcshow(ptcloud);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title(filenames{idx});
    Data(:,1)= double(ptcloud.Location(1:1:end,1));   %提取所有点的三维坐标
    Data(:,2)= double(ptcloud.Location(1:1:end,2));
    Data(:,3)= double(ptcloud.Location(1:1:end,3)); 
    namesplit=strsplit(filenames{idx},'.');           %分割ply文件的名称，分成文件名与ply后缀名
    frontname=namesplit{1};                           %提取文件名，舍弃后缀名
%     fid=fopen(strcat(frontname,'.txt'),'wt');      
    eval(['fid=fopen(''',frontname,'.txt'',''wt'');']);      
    [b1,b2]=size(Data);    
    for i=1:b1                   %将二维数组Data写入txt格式文件中
        for j=1:b2-1
            fprintf(fid,'%.4f\t ',Data(i,j));           %所有坐标数据保留小数点后四位
        end
        fprintf(fid,'%.4f\n',Data(i,b2));
    end
    clear Data;                 
    fclose(fid);
end