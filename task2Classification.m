%% Classification task for Task 2
%% Data preparation
iris = readtable('iris.csv');
X = iris(:,1:4);
X = table2array(X);
Y = iris(:,5);
Y = table2cell(Y);
idx = ~strcmp(Y,'Iris-setosa');
X = X(idx,:);
Y = strcmp(Y(idx,:),'Iris-versicolor');
D = [X,Y];
% Initialize the hyperparameter
C = linspace(0.5,2,10);
sigma = linspace(1,5,5);
q = linspace(1,5,5);
k1 = 10;
k2 = 5;

%% RBF kernal
% Training
[best_C, best_sigma, correspond_inacc, correspond_outacc, support_vec_num, support_vec_percentage] = RBFClassificationCV(D, k1, k2, C, sigma);
fprintf('=========== RBF kernal ============\n')
% report the result
fprintf('Best %d models:\n',k1)
for i = 1:k1
    fprintf('C: %.3f, ',best_C(i))
    fprintf('Sigma:%.2f, ',best_sigma(i))
    fprintf('sv_num:%d(%.1f%%), ',support_vec_num(i),support_vec_percentage(i))
    fprintf('OuterTestAcc:%.4f, ',correspond_outacc(i))
    fprintf('InnerBestAcc:%.4f\n',correspond_inacc(i))
end
fprintf('\nAverage test accuracy:%.6f\n',mean(correspond_outacc))
fprintf('Average support vector:%.1f(%.1f%%)\n\n',mean(support_vec_num),mean(support_vec_percentage))


%% Polynomial kernal
% Training
[best_C, best_q, correspond_inacc, correspond_outacc, support_vec_num, support_vec_percentage] = PolyClassificationCV(D, 10, 5, C, q);
% Report the result
fprintf('=========== Poly kernal ============\n')
fprintf('\nBest %d models:\n',k1)
for i = 1:k1
    fprintf('C: %.3f, ',best_C(i))
    fprintf('q:%.2f, ',best_q(i))
    fprintf('sv_num:%d(%.1f%%), ',support_vec_num(i),support_vec_percentage(i))
    fprintf('OuterTestAcc:%.4f, ',correspond_outacc(i))
    fprintf('InnerBestAcc:%.4f\n',correspond_inacc(i))
end
fprintf('\nAverage test accuracy:%.6f\n',mean(correspond_outacc))
fprintf('Average support vector:%.1f(%.1f%%)\n\n',mean(support_vec_num),mean(support_vec_percentage))

%% Task3:
fprintf('\n task3\n')
% Use the optimise hyperparameters to apply 10-fold cv
k = 10;
%% RBF kernel:
fprintf('=========== RBF kernal ============\n')
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
fprintf('%d-Fold mean accuracy:%.6f\n',k,mean(acc))

%% Poly kernel:
fprintf('=========== Poly kernal ============\n')
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



