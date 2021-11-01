function RMSE = regressionTrainTest(X,Y,testRatio,epsilon)
    [trainX,trainY,testX,testY] = crossGroup(X,Y,testRatio);
    
    Mdl = fitrsvm(trainX, trainY, 'Epsilon', epsilon);
    
    predictedLbls = predict(Mdl,testX);
    RMSE = rmse(predictedLbls,testY);
end

