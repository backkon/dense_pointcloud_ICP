%% This procedure intends to realize initial ICP algorithm
close all;
clear;
clc;

q0 = dlmread('Q0New.txt');% Object
rng(1);

nCount = 1;
timeAll = zeros(10, 1);
errorAll = zeros(10, 1);
iterator = zeros(10, 1);
for selectNum = 5:5:50
    p0 = dlmread('P0.txt');
    numTotal = length(p0);
    tic;
    for i = 1:200
        mSelected = randsample(numTotal, selectNum);
        p0select = p0(mSelected, :);
        index = knnsearch(q0, p0select);
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
            p0 = p0*R + T;
            if error < 0.5
                dlmwrite(sprintf('P0Rotation_%d.txt', selectNum), p0);
                break;
            end
        end    
    end
    time = toc;
    timeAll(nCount, 1) = time;
    errorAll(nCount, 1) = error;
    iterator(nCount, 1) = i;
    nCount = nCount + 1;
end