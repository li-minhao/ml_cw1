%% sample for task 2
iris = readtable('iris.csv');
X = iris(:,1:4);
X = table2array(X);
Y = iris(:,5);
Y = table2cell(Y);
idx = ~strcmp(Y,'Iris-setosa');
X = X(idx,:);
Y = Y(idx,:);
y = strcmp(Y,'Iris-versicolor');
D = [X,y];
C = linspace(0.5,1,10);
sigma = linspace(0.5,1,10);

% RBF kenal
RBFNestCrossValidation(D, 10, 5, C, sigma)




