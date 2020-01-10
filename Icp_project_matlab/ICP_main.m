%% This procedure intends to realize initial ICP algorithm
close all;
clear;
clc;

ptcloud_p0=pcread('E:\pg_one\CG\final_project\ICP\P0.ply');
ptcloud_q0=pcread('E:\pg_one\CG\final_project\ICP\newQ0\Q0.ply');
p0 = ptcloud_p0.Location;
q0 = ptcloud_q0.Location;% Object
selectNum = 100;
numTotal = size(p0, 1);
q0Norm = sum(q0.^2, 2)';
q0Norm = repmat(q0Norm, [selectNum, 1]);
rng(1);

tic;
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
    p0select = p0select - p0selectCentroid;
    q0select = q0select - q0selectCentroid;
    H = p0select' * q0select;
    [U, S, V] = svd(H);
    R = U*V';
    T = q0selectCentroid - p0selectCentroid * R;
    
    error = sum(sum((q0 - p0*R - T).^2)) / numTotal;
    p0 = p0*R + T;
    if error < 0.5
        break;
    end
end
fprintf('旋转矩阵：\n');
disp(R);
fprintf('平移向量：\n');
disp(T);
fprintf('\n');
toc
ptcloud_p0_tran = pointCloud(p0);
pcwrite(ptcloud_p0_tran,'P0_tran.ply');
figure(1);
pcshowpair(ptcloud_p0,ptcloud_q0);
figure(2);
pcshowpair(ptcloud_p0_tran,ptcloud_q0);