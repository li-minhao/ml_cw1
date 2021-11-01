close all;
clear;

%% load Iris data
M = readtable("iris.csv");
X = table2array(M(:,1:4));
Y = M(:,5);
Y = double(categorical(table2cell(Y)));

X = X(51:150,:);
Y = Y(51:150,:);

%% train classification model and test
testRatio = 0.3;
[acc, pre, re] = classificationTrainTest(X, Y, testRatio);

%% train regression model and test
testRatio = 0.3;
epsilon = 0.3;
RMSE = regressionTrainTest(X, Y, testRatio, epsilon);
