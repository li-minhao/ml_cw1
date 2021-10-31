function [acc, pre, re] = train_and_test(X, Y, testRatio, method)
    trainIndices = crossvalind('HoldOut', size(X, 1), testRatio);
    testIndices = ~trainIndices;

    % obtain train set and test set
    train_X = X(trainIndices, :);
    train_Y = Y(trainIndices, :);
    test_X = X(testIndices, :);
    test_Y = Y(testIndices, :);
    
    % train model
    if method == "classification"
        Mdl= fitcsvm(train_X,train_Y,"KernelFunction","linear","BoxConstraint",1);
    elseif method == "regression"
%         Mdl = 
    end

    % predict the label for test set
    predictLabels = predict(Mdl,test_X);
    acc = accuracy(predictLabels,test_Y);
    pre = precision(predictLabels,test_Y);
    re  = recall(predictLabels,test_Y);
end

