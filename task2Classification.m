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
C = linspace(0.5,2,20);
sigma = linspace(1,5,20);
q = linspace(1,5,5);

% RBF kernal
RBFNestCrossValidation(D, 10, 5, C, sigma)

% Polynomial kernal
PolyKerNestCrossValidation(D, 10, 5, C, q)


