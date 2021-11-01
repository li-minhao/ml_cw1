function [trainX,trainY,testX,testY] = crossGroup(X,Y,testRatio)
    trainIndices = crossvalind('HoldOut', size(X, 1), testRatio);
    testIndices = ~trainIndices;

    % obtain train set and test set
    trainX = X(trainIndices, :);
    trainY = Y(trainIndices, :);
    testX = X(testIndices, :);
    testY = Y(testIndices, :);
end

