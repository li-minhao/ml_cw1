function avg_RMSE = task1_regression(X,Y,k,epsilon)
    RMSE = zeros(1,k);
    for i=1:k
        [trainX,trainY,testX,testY] = KFoldGroup(X,Y,k,i,randperm(size(X,1)));
    
         Mdl = fitrsvm(trainX, trainY,'Standardize',true, 'Epsilon', epsilon);
    
        predictedLbls = predict(Mdl,testX);
        RMSE(i) = rmse(predictedLbls,testY);
    end
    avg_RMSE = mean(RMSE);
end

