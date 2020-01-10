clc;
clear;
global ptcloud_p0;
global q0;
global numTotal;
ptcloud_p0=pcread('E:\pg_one\CG\final_project\ICP\P0.ply');
ptcloud_q0=pcread('E:\pg_one\CG\final_project\ICP\Q0.ply');

q0 = ptcloud_q0.Location;% Object
numTotal = size(ptcloud_p0.Location, 1);


time(10) = 0;
score(10) = 0;
selectNum = 300:10:350;
for i = 1:length(selectNum)
    [time(i), score(i)] = exp2_func(selectNum(i));
    if score(i)>=0.5
        fprintf('��%d��û�дﵽ����Ҫ��\n',i);
        time(i) = 0;
    else
    fprintf('��%d�ε������\n',i);
    end
end
figure(1);
t = time(time ~= 0);
[~,pos]=ismember(t,time);
s = selectNum(pos);
plot(s,t);
xlabel('�������');
ylabel('����ʱ��');
title('����ʱ������������Ĺ�ϵ����');
%save('time.mat','t');
%save('selectNum.mat','s');