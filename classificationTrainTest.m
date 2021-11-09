function [avg_acc, avg_pre, avg_re] = classificationTrainTest(X, Y, k)
    acc = zeros(1,k);
    pre = zeros(1,k);
    re  = zeros(1,k);
    for i=1:k
        [trainX,trainY,testX,testY] = KFoldGroup(X,Y,k,i,randperm(size(X,1)));

        % train model
        Mdl = fitcsvm(trainX, trainY, "Standardize",true,"KernelFunction", "linear", "BoxConstraint", 1);

        % predict the label for test set
        predictLabels = predict(Mdl,testX);
        acc(i) = accuracy(predictLabels,testY);
        pre(i) = precision(predictLabels,testY);
        re(i)  = recall(predictLabels,testY);
    end
    avg_acc = mean(acc);
    avg_pre = mean(pre);
    avg_re  = mean(re);
end

