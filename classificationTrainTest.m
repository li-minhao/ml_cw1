function [acc, pre, re] = classificationTrainTest(X, Y, testRatio)
    [trainX,trainY,testX,testY] = crossGroup(X,Y,testRatio);
    
    % train model
    Mdl = fitcsvm(trainX, trainY, "KernelFunction", "linear", "BoxConstraint", 1);

    % predict the label for test set
    predictLabels = predict(Mdl,testX);
    acc = accuracy(predictLabels,testY);
    pre = precision(predictLabels,testY);
    re  = recall(predictLabels,testY);
end

