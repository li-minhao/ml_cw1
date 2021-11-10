close all;
clear;
addpath(genpath(".."));


%% set k for k-fold cross validation
k = 10;


%% load Iris data
iris = readtable("iris.csv");
X = table2array(iris(:,1:4));
Y = iris(:,5);
Y = double(categorical(table2cell(Y)));

% remove the first category for binary classification
X = X(51:150,:);
Y = Y(51:150,:);

% preprocess the data
[X,hasNaN_iris] = preprocess(X,Y);
if hasNaN_iris
    fprintf("Iris dataset has missing value.\n\n");
else
    fprintf("Iris dataset has no missing value.\n\n");
end
 

%% train classification model and test
acc = linear_classification(X, Y, 10);
fprintf("The accuracy of iris is: ");
disp(acc);


%% load wine data
wine = readtable("winequality-white.csv");

% randomly pick 5% data to accelerate computing
% len = size(wine,1);
% index = randperm(len,round(len/20));
% if 0    % Please do not modify this file when testing
%     save('WineIndices.mat', 'index')
% end

% make sure use the same dataset for both task1 and task2
load('WineIndices.mat');
wine = wine(index,:);

X_wine = table2array(wine(:,1:11));
Y_wine = table2array(wine(:,12));

% preprocess the data
[X_wine,hasNaN_wine] = preprocess(X_wine,Y_wine);
if hasNaN_wine
    fprintf("Wine dataset has missing value.\n");
else
    fprintf("Wine dataset has no missing value.\n");
end


%% train regression model and test
epsilon =  linspace(0.1,1.5,20);
RMSE = zeros(1, 20);
for i=1:20
    RMSE(i) = linear_regression(X_wine, Y_wine, k, epsilon(i));
end

plot(epsilon,RMSE);
title('task1 regression');
xlabel('epsilon');
ylabel('RMSE');

[sortRMSE, index] = sort(RMSE,'ascend');
bestEpsilon = epsilon(index(1));

fprintf("The best RMSE is: ");
disp(sortRMSE(1));
fprintf("That epsilon is: ");
disp(bestEpsilon);

averageRMSE = mean(RMSE);
fprintf("The average RMSE is: ");
disp(averageRMSE);



