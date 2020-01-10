function [time,score] = exp2_func(selectNum)
global ptcloud_p0;
global q0;
global numTotal;

p0 = ptcloud_p0.Location;
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
    T = q0selectCentroid - p0selectCentroid;
    p0select = p0select - p0selectCentroid;
    q0select = q0select - q0selectCentroid;
    H = p0select' * q0select;
    [U, ~, V] = svd(H);
    R = U*V';
    
    score = sum(sum((q0 - p0*R - T).^2)) / numTotal;
    p0 = p0*R + T;
    if score < 0.5
        break;
    end
end
time = toc;
end

