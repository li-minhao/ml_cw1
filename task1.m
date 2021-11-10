close all;
clear;

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
[acc, pre, re] = task1_classification(X, Y, 10);
f1 = 2*(pre*re)/(pre+re);
fprintf("The accuracy of iris is: ");
disp(acc);
fprintf("The precision of iris is: ");
disp(pre);
fprintf("The recall of iris is: ");
disp(re);
fprintf("The f1 score of iris is: ");
disp(f1);

%% load wine data
wine = readtable("winequality-white.csv");

% randomly pick 20% data becasuse this dataset has too much data
len = size(wine,1);
index = randperm(len,round(len/5));
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
    RMSE(i) = task1_regression(X_wine, Y_wine, k, epsilon(i));
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


