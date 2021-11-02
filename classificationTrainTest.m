function [avg_acc, avg_pre, avg_re] = classificationTrainTest(X, Y, k)
    acc = [];
    pre = [];
    re  = [];
    for i=1:k
        [trainX,trainY,testX,testY] = KFoldGroup(X,Y,k,i,randperm(size(X,1)));

        % train model
        Mdl = fitcsvm(trainX, trainY, "Standardize",true,"KernelFunction", "linear", "BoxConstraint", 1);

        % predict the label for test set
        predictLabels = predict(Mdl,testX);
        acc = [acc,accuracy(predictLabels,testY)];
        pre = [pre,precision(predictLabels,testY)];
        re  = [re,recall(predictLabels,testY)];
    end
    avg_acc = mean(acc);
    avg_pre = mean(pre);
    avg_re  = mean(re);
end

