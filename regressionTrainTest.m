function avg_RMSE = regressionTrainTest(X,Y,k,epsilon)
    RMSE = [];
    for i=1:k
        [trainX,trainY,testX,testY] = KFoldGroup(X,Y,k,i,randperm(size(X,1)));
    
         Mdl = fitrsvm(trainX, trainY,'Standardize',true, 'Epsilon', epsilon);
    
        predictedLbls = predict(Mdl,testX);
        RMSE = [RMSE, rmse(predictedLbls,testY)];
    end
    avg_RMSE = mean(RMSE);
end

