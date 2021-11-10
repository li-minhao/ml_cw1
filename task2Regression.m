%% regression task for task 2
wine = table2array(readtable('winequality-white.csv'));
indices = randperm(round(size(wine,1)/10));
wine = wine(indices,:);

C = linspace(0.5,1,2);
sigma = linspace(1,5,10);
q = linspace(1,4,4);
epsilon = linspace(0.1,1,5);

% RBF kernal
RBFRegressionCV(wine, 10, 5, C, sigma, epsilon);

% Polynomial kernal
PolyRegressionCV(wine, 10, 5, C, q, epsilon);

