close all;
clear;

%% set k for k-fold cross validation
k = 10;

%% load Iris data
iris = readtable("iris.csv");
X = table2array(iris(:,1:4));
Y = iris(:,5);
Y = double(categorical(table2cell(Y)));
 
X = X(51:150,:);
Y = Y(51:150,:);
 
%% train classification model and test
[X,hasNaN_iris] = preprocess(X,Y);
if hasNaN_iris
    fprintf("Iris dataset has missing value.\n\n");
else
    fprintf("Tris dataset has no missing value.\n\n");
end

[acc, pre, re] = classificationTrainTest(X, Y, 10);
fprintf("The accuracy of iris is: ")
disp(acc)
fprintf("The precision of iris is: ")
disp(pre)
fprintf("The recall of iris is: ")
disp(re)

%% load wine data
wine = readtable("winequality-white.csv");

% random pick 20% data becasuse this dataset has too much data
len = size(wine,1);
index = randperm(round(len/5));
wine = wine(index,:);

X_wine = table2array(wine(:,1:11));
Y_wine = table2array(wine(:,12));

[X_wine,hasNaN_wine] = preprocess(X_wine,Y_wine);
if hasNaN_wine
    fprintf("Wine dataset has missing value.\n");
else
    fprintf("Wine dataset has no missing value.\n");
end

%% train regression model and test
epsilon =  linspace(0.1,1,30);
RMSE = [];
for i=1:30
    RMSE = [RMSE, regressionTrainTest(X_wine, Y_wine, k, epsilon(i))];
end

plot(epsilon,RMSE)
title('task1 regression')
xlabel('epsilon')
ylabel('RMSE')
