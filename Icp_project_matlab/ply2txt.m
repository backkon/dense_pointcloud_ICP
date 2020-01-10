clc;clear;
path=pwd;
file = dir(fullfile(path,'*.ply'));    %��ȡ����ply��ʽ�ļ�
filenames = {file.name}';
filelength = size(filenames,1);        %ply��ʽ�ļ���

for idx = 1 : filelength               %������
    filedir = strcat(path, filenames(idx));
    ptcloud=pcread(filenames{idx});   %ply��ʽ�ļ���pcread��ȡ
    figure(idx);
    pcshow(ptcloud);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title(filenames{idx});
    Data(:,1)= double(ptcloud.Location(1:1:end,1));   %��ȡ���е����ά����
    Data(:,2)= double(ptcloud.Location(1:1:end,2));
    Data(:,3)= double(ptcloud.Location(1:1:end,3)); 
    namesplit=strsplit(filenames{idx},'.');           %�ָ�ply�ļ������ƣ��ֳ��ļ�����ply��׺��
    frontname=namesplit{1};                           %��ȡ�ļ�����������׺��
%     fid=fopen(strcat(frontname,'.txt'),'wt');      
    eval(['fid=fopen(''',frontname,'.txt'',''wt'');']);      
    [b1,b2]=size(Data);    
    for i=1:b1                   %����ά����Dataд��txt��ʽ�ļ���
        for j=1:b2-1
            fprintf(fid,'%.4f\t ',Data(i,j));           %�����������ݱ���С�������λ
        end
        fprintf(fid,'%.4f\n',Data(i,b2));
    end
    clear Data;                 
    fclose(fid);
end