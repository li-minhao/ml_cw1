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
[acc, pre, re] = classificationTrainTest(X, Y, 10);


%% load wine data
wine = readtable("winequality-white.csv");

% random pick 10% data becasuse this dataset has too much data
len = size(wine,1);
index = randperm(round(len/10));
wine = wine(index,:);

X_wine = table2array(wine(:,1:11));
Y_wine = table2array(wine(:,12));


%% train regression model and test
epsilon = 0.3;
RMSE = regressionTrainTest(X_wine, Y_wine, k, epsilon);
