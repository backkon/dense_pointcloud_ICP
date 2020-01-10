%% This procedure intends to realize initial ICP algorithm
close all;
clear;
clc;

q0 = dlmread('Q0New.txt');% Object
q0NormInitial = sum(q0.^2, 2)';
rng(1);

nCount = 1;
timeAll = zeros(10, 1);
errorAll = zeros(10, 1);
for selectNum = 5:5:50
    p0 = dlmread('P0.txt');
    numTotal = length(p0);
    q0Norm = repmat(q0NormInitial, [selectNum, 1]);
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
        p0 = p0*R + T;
    end
    time = toc;
    timeAll(nCount, 1) = time;
    indexall = knnsearch(q0, p0);
    q0Temp = q0(indexall, :);
    error = sum(sum((q0Temp - p0).^2)) / numTotal;
    errorAll(nCount, 1) = error;
    nCount = nCount + 1;
end

save errorAll errorAll;
save timeAll timeAll;