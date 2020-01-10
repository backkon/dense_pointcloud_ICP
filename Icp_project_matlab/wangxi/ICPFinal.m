%% This procedure intends to realize initial ICP algorithm
close all;
clear;
clc;
tic;
p0 = dlmread('P0.txt');
q0 = dlmread('Q0New.txt');% Object
numTotal = length(p0);
selectNum = 10;
q0Norm = sum(q0.^2, 2)';
q0Norm = repmat(q0Norm, [selectNum, 1]);
rng(1);


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
    p0 = p0*R + T;

    if rem(i, 10) == 0
        indexall = knnsearch(q0, p0);
        q0Temp = q0(indexall, :);
        error = sum(sum((q0Temp - p0).^2)) / numTotal;
        if error < 0.5
            dlmwrite(sprintf('P0Rotation_%d.txt', selectNum), p0);
            break;
        end
    end    
end
time = toc;