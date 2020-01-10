%% error随迭代次数的变化关系
close all;
clear;
clc;

ptcloud_p0=pcread('E:\pg_one\CG\final_project\ICP\P0.ply');
ptcloud_q0=pcread('E:\pg_one\CG\final_project\ICP\Q0.ply');
p0 = ptcloud_p0.Location;
q0 = ptcloud_q0.Location;% Object
selectNum = 10;
numTotal = size(p0, 1);
q0Norm = sum(q0.^2, 2)';
q0Norm = repmat(q0Norm, [selectNum, 1]);
rng(1);

tic;
error(200)=0;
for i = 1:200
    mSelected = randsample(numTotal, selectNum);
    p0select = p0(mSelected, :);
    p0selectNorm = sum(p0select.^2, 2);
    p0selectNorm = repmat(p0selectNorm, [1, numTotal]);
    dotMatrix = p0select * q0';
    distanceMatrix = p0selectNorm + q0Norm - 2*dotMatrix;
    [~, index] = min(distanceMatrix,[], 2);
    q0select = q0(index, :);
    
    p0selectCentroid = mean(p0select);
    q0selectCentroid = mean(q0select);
    T = q0selectCentroid - p0selectCentroid;
    p0select = p0select - p0selectCentroid;
    q0select = q0select - q0selectCentroid;
    H = p0select' * q0select;
    [U, S, V] = svd(H);
    R = U*V';
    
    error(i) = sum(sum((q0 - p0*R - T).^2)) / numTotal;
    p0 = p0*R + T;
    if error(i) < 0.5
        break;
    end
end
figure(1);
x = 1:1:i;
error = error(1:i);
plot(x,error);
xlabel('迭代次数');
ylabel('error');
title('error随迭代次数的变化曲线');