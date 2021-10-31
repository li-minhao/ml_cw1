close all;
clear;

%% load Iris data
M = readtable("iris.csv");
X = table2array(M(:,1:4));
Y = M(:,5);
Y = double(categorical(table2cell(Y)));

X = X(51:150,:);
Y = Y(51:150,:);

%% train model and test
[acc, pre, re] = train_and_test(X, Y, 0.3, "classification");
