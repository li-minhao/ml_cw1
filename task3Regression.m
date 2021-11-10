% Regression task for task 3
% Use the optimise hyperparameters to apply 10-fold cv
%% Data preparation
wine = table2array(readtable('winequality-white.csv'));
indices = randperm(size(wine,1),round(size(wine,1)/20));
wine = wine(indices,:);
X = wine(:,1:end-1);
y = wine(:,end);

[X,has_NaN] = preprocess(X,y);
if has_NaN
    fprintf("Wine quality dataset has missing value.\n\n");
end
k = 10;


%% Linear:
fprintf('=========== Linear ============\n')
RMSE_l = [];
for i=1:k
    fprintf('Fold %d: ',i);
    [trainX,trainY,testX,testY] = KFoldGroup(X,y,k,i,randperm(size(X,1)));
    Mdl = fitrsvm(trainX, trainY,'Standardize',true);
    X_pdt = predict(Mdl, testX);
    RMSE_l(i) = rmse(x_pdt,testY);
    fprintf('RMSE: %.3f\n',RMSE_l(i))
end
fprintf('%d-Fold mean RMSE: %.6f\n',k,mean(RMSE_l))


%% Gaussian RBF kernel:
fprintf('=========== RBF kernal ============\n')
load('RBFRegression.mat')
RMSE_r = [];
for i=1:k
    fprintf('Fold %d: ',i);
    [trainX,trainY,testX,testY] = KFoldGroup(X,y,k,i,randperm(size(X,1)));
    % Use 10 optimse hyperparameters train model
    X_pdt = zeros(size(testX,1),size(best_C_r,2));
    % Train k models based on the k best hyperparameters
    for j = 1:size(best_C_r,2)
        BoxConstraint_r = best_C_r(j);
        KernelScale_r = best_sigma_r(j);
        Epsilon_r = best_Epsilon_r(j);
        Mdl = fitrsvm(X_train,y_train,'Standardize',true,'KernelFunction','RBF','BoxConstraint',BoxConstraint_r,'KernelScale',KernelScale_r,'Epsilon',Epsilon_r);
        X_pdt(:,j) = predict(Mdl, testX);
    end
    % take the average of k1 models with the best hyperparameter
    avg_predict = mean(X_pdt,2);
    % Calculate performance
    RMSE_r(i) = rmse(avg_predict,testY);
    fprintf('RMSE: %.3f\n',RMSE_r(i))
end
fprintf('%d-Fold mean RMSE: %.6f\n',k,mean(RMSE_r))


%% Polynomial kernel:
fprintf('=========== Poly kernal ============\n')
load('PolyRegression.mat')
RMSE_p = [];
for i=1:k
    fprintf('Fold %d: ',i);
    [trainX,trainY,testX,testY] = KFoldGroup(X,y,k,i,randperm(size(X,1)));
    % Use 10 optimse hyperparameters train model
    X_pdt = zeros(size(testX,1),size(best_C_p,2));
    % Train k1 models based on the k1 best hyperparameters
    for j = 1:size(best_C_p,2)
        BoxConstraint_p = best_C_p(j);
        PolynomialOrder_p = best_q_p(j);
        Epsilon_p = best_Epsilon_p(j);
        Mdl = fitrsvm(X_train,y_train,'Standardize',true,'KernelFunction','polynomial','BoxConstraint',BoxConstraint_p,'PolynomialOrder',PolynomialOrder_p,'Epsilon',Epsilon_p);
        X_pdt(:,j) = predict(Mdl, testX);
    end
    % take the average of k1 models with the best hyperparameter
    avg_predict = mean(X_pdt,2);
    % Calculate performance
    RMSE_p(i) = rmse(avg_predict,testY);
    fprintf('RMSE: %.3f\n',RMSE_p(i))
end
fprintf('%d-Fold mean RMSE: %.6f\n',k,mean(RMSE_p))
