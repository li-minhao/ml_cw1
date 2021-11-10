% Classification task for task 3
% Use the optimise hyperparameters to apply 10-fold cv
%% Data preparation
iris = readtable('iris.csv');
X = iris(:,1:4);
X = table2array(X);
Y = iris(:,5);
Y = table2cell(Y);
idx = ~strcmp(Y,'Iris-setosa');
X = X(idx,:);
Y = strcmp(Y(idx,:),'Iris-versicolor');
k = 10;


%% RBF kernel:
fprintf('=========== RBF kernal ============\n\n')
load('RBFClassification.mat')
acc = [];
for i=1:k
    fprintf('Fold-%d:',i);
    [trainX,trainY,testX,testY] = KFoldGroup(X,Y,k,i,randperm(size(X,1)));
    % Use 10 optimse hyperparameters train model
    X_pdt = zeros(size(testX,1),size(best_sigma,2));
    % Train k1 models based on the k1 best hyperparameters
    for j = 1:size(best_C,2)
        BoxConstraint = best_C(j);
        KernelScale = best_sigma(j);
        Mdl = fitcsvm(trainX,trainY,'Standardize',true,'KernelFunction','RBF','BoxConstraint',BoxConstraint,'KernelScale',KernelScale);
        X_pdt(:,j) = predict(Mdl, testX);
    end
    % take the average of k1 models with the best hyperparameter
    avg_predict = mean(X_pdt,2);
    % Calculate performance
    acc(i) = accuracy(avg_predict,testY);
    fprintf('accuracy:%.3f\n',acc(i))
end
fprintf('%d-Fold mean accuracy:%.6f\n\n',k,mean(acc))

%% Poly kernel:
fprintf('=========== Poly kernal ============\n\n')
load('PolyClassification.mat')
acc = [];
for i=1:k
    fprintf('Fold-%d:',i);
    [trainX,trainY,testX,testY] = KFoldGroup(X,Y,k,i,randperm(size(X,1)));
    % Use 10 optimse hyperparameters train model
    X_pdt = zeros(size(testX,1),size(best_C,2));
    % Train k1 models based on the k1 best hyperparameters
    for j = 1:size(best_C,2)
        BoxConstraint = best_C(j);
        PolynomialOrder = best_q(j);
        Mdl = fitcsvm(trainX,trainY,'Standardize',true,'KernelFunction','polynomial','BoxConstraint',BoxConstraint,'PolynomialOrder',PolynomialOrder);
        X_pdt(:,j) = predict(Mdl, testX);
    end
    % take the average of k1 models with the best hyperparameter
    avg_predict = mean(X_pdt,2);
    % Calculate performance
    acc(i) = accuracy(avg_predict,testY);
    fprintf('accuracy:%.3f\n',acc(i))
end
fprintf('%d-Fold mean accuracy:%.6f\n',k,mean(acc))