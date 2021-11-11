close all;
clear;
addpath(genpath(".."));


%% set k for k-fold cross validation
k = 10;


%% Data preparation
% load iris and remove the first category for binary classification
iris = readtable("iris.csv");
X_iris = table2array(iris(:,1:4));
Y_iris = iris(:,5);
Y_iris = double(categorical(table2cell(Y_iris)));
X_iris = X_iris(51:150,:);
Y_iris = Y_iris(51:150,:);

% load wine_quality
wine = table2array(readtable('winequality-white.csv'));
load('dataset/wine_indices.mat');
wine = wine(index,:);
X_wine = wine(:,1:end-1);
Y_wine = wine(:,end);
[X_wine,has_NaN] = preprocess(X_wine,Y_wine);
if has_NaN
    fprintf("Wine quality dataset has missing value.\n\n");
end


%% Linear kernel (Classification):
fprintf('=========== Linear kernal (Classification) ============\n\n')
load('hyper_parameter/linear_classification.mat')
acc = [];
for i=1:k
    fprintf('Fold-%d:',i);
    [trainX,trainY,testX,testY] = KFoldGroup(X_iris,Y_iris,k,i,randperm(size(X_iris,1)));
    % Use 10 optimse hyperparameters train model
    X_pdt = zeros(size(testX,1),size(best_C_l,2));
    % Train k1 models based on the k1 best hyperparameters
    for j = 1:size(best_C_l,2)
        BoxConstraint = best_C_l(j);
        Mdl = fitcsvm(trainX,trainY,'Standardize',true,'KernelFunction','linear','BoxConstraint',BoxConstraint);
        X_pdt(:,j) = predict(Mdl, testX);
    end
    % take the average of k1 models with the best hyperparameter
    avg_predict = mean(X_pdt,2);
    % Calculate performance
    acc(i) = accuracy(avg_predict,testY);
    fprintf('accuracy:%.3f\n',acc(i))
end
fprintf('\n%d-Fold mean accuracy:%.6f\n\n',k,mean(acc))


%% RBF kernel (Classification):
fprintf('=========== RBF kernal (Classification) ============\n\n')
load('hyper_parameter/RBF_classification.mat')
acc = [];
for i=1:k
    fprintf('Fold-%d:',i);
    [trainX,trainY,testX,testY] = KFoldGroup(X_iris,Y_iris,k,i,randperm(size(X_iris,1)));
    % Use 10 optimse hyperparameters train model
    X_pdt = zeros(size(testX,1),size(best_sigma_rbf,2));
    % Train k1 models based on the k1 best hyperparameters
    for j = 1:size(best_C_rbf,2)
        BoxConstraint = best_C_rbf(j);
        KernelScale = best_sigma_rbf(j);
        Mdl = fitcsvm(trainX,trainY,'Standardize',true,'KernelFunction','RBF','BoxConstraint',BoxConstraint,'KernelScale',KernelScale);
        X_pdt(:,j) = predict(Mdl, testX);
    end
    % take the average of k1 models with the best hyperparameter
    avg_predict = mean(X_pdt,2);
    % Calculate performance
    acc(i) = accuracy(avg_predict,testY);
    fprintf('accuracy:%.3f\n',acc(i))
end
fprintf('\n%d-Fold mean accuracy:%.6f\n\n',k,mean(acc))


%% Poly kernel (Classification):
fprintf('=========== Poly kernal (Classification) ============\n\n')
load('hyper_parameter/poly_classification.mat')
acc = [];
for i=1:k
    fprintf('Fold-%d:',i);
    [trainX,trainY,testX,testY] = KFoldGroup(X_iris,Y_iris,k,i,randperm(size(X_iris,1)));
    % Use 10 optimse hyperparameters train model
    X_pdt = zeros(size(testX,1),size(best_C_poly,2));
    % Train k1 models based on the k1 best hyperparameters
    for j = 1:size(best_C_poly,2)
        BoxConstraint = best_C_poly(j);
        PolynomialOrder = best_q_poly(j);
        Mdl = fitcsvm(trainX,trainY,'Standardize',true,'KernelFunction','polynomial','BoxConstraint',BoxConstraint,'PolynomialOrder',PolynomialOrder);
        X_pdt(:,j) = predict(Mdl, testX);
    end
    % take the average of k1 models with the best hyperparameter
    avg_predict = mean(X_pdt,2);
    % Calculate performance
    acc(i) = accuracy(avg_predict,testY);
    fprintf('accuracy:%.3f\n',acc(i))
end
fprintf('\n%d-Fold mean accuracy:%.6f\n\n',k,mean(acc))


%% Linear kernel (Regression):
fprintf('=========== Linear kernal (Regression) ============\n\n')
load('hyper_parameter/linear_SVR.mat')
RMSE_l = [];
for i=1:k
    fprintf('Fold %d: ',i);
    [trainX,trainY,testX,testY] = KFoldGroup(X_wine,Y_wine,k,i,randperm(size(X_wine,1)));
    % Use 10 optimse hyperparameters train model
    X_pdt = zeros(size(testX,1),size(best_C_l,2));
    % Train k models based on the k best hyperparameters
    for j = 1:size(best_C_l,2)
        BoxConstraint_l = best_C_l(j);
        Epsilon_l = best_Epsilon_l(j);
        Mdl = fitrsvm(trainX,trainY,'Standardize',true,'KernelFunction','RBF','BoxConstraint',BoxConstraint_l,'Epsilon',Epsilon_l);
        X_pdt(:,j) = predict(Mdl, testX);
    end
    % take the average of k1 models with the best hyperparameter
    avg_predict = mean(X_pdt,2);
    % Calculate performance
    RMSE_l(i) = rmse(avg_predict,testY);
    fprintf('RMSE: %.3f\n',RMSE_l(i))
end
fprintf('\n%d-Fold mean RMSE: %.6f\n\n',k,mean(RMSE_l))


%% Gaussian RBF kernel (Regression):
fprintf('=========== RBF kernal (Regression) ============\n\n')
load('hyper_parameter/RBF_regression.mat')
RMSE_r = [];
for i=1:k
    fprintf('Fold %d: ',i);
    [trainX,trainY,testX,testY] = KFoldGroup(X_wine,Y_wine,k,i,randperm(size(X_wine,1)));
    % Use 10 optimse hyperparameters train model
    X_pdt = zeros(size(testX,1),size(best_C_r,2));
    % Train k models based on the k best hyperparameters
    for j = 1:size(best_C_r,2)
        BoxConstraint_r = best_C_r(j);
        KernelScale_r = best_sigma_r(j);
        Epsilon_r = best_Epsilon_r(j);
        Mdl = fitrsvm(trainX,trainY,'Standardize',true,'KernelFunction','RBF','BoxConstraint',BoxConstraint_r,'KernelScale',KernelScale_r,'Epsilon',Epsilon_r);
        X_pdt(:,j) = predict(Mdl, testX);
    end
    % take the average of k1 models with the best hyperparameter
    avg_predict = mean(X_pdt,2);
    % Calculate performance
    RMSE_r(i) = rmse(avg_predict,testY);
    fprintf('RMSE: %.3f\n',RMSE_r(i))
end
fprintf('\n%d-Fold mean RMSE: %.6f\n',k,mean(RMSE_r))


%% Polynomial kernel (Regression):
fprintf('=========== Poly kernal (Regression) ============\n\n')
load('hyper_parameter/poly_regression.mat')
RMSE_p = [];
for i=1:k
    fprintf('Fold %d: ',i);
    [trainX,trainY,testX,testY] = KFoldGroup(X_wine,Y_wine,k,i,randperm(size(X_wine,1)));
    % Use 10 optimse hyperparameters train model
    X_pdt = zeros(size(testX,1),size(best_C_p,2));
    % Train k1 models based on the k1 best hyperparameters
    for j = 1:size(best_C_p,2)
        BoxConstraint_p = best_C_p(j);
        PolynomialOrder_p = best_q_p(j);
        Epsilon_p = best_Epsilon_p(j);
        Mdl = fitrsvm(trainX,trainY,'Standardize',true,'KernelFunction','polynomial','BoxConstraint',BoxConstraint_p,'PolynomialOrder',PolynomialOrder_p,'Epsilon',Epsilon_p);
        X_pdt(:,j) = predict(Mdl, testX);
    end
    % take the average of k1 models with the best hyperparameter
    avg_predict = mean(X_pdt,2);
    % Calculate performance
    RMSE_p(i) = rmse(avg_predict,testY);
    fprintf('RMSE: %.3f\n',RMSE_p(i))
end
fprintf('\n%d-Fold mean RMSE: %.6f\n\n',k,mean(RMSE_p))

