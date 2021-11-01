%% regression task for task 2
wine_red = table2array(readtable('winequality-red.csv'));
wine_white = table2array(readtable('winequality-white.csv'));
wine = [wine_red;wine_white];

C = linspace(0.5,1,5);
sigma = linspace(1,5,5);
q = linspace(1,5,5);
epsilon = linspace(0,2,5);

% RBF kernal
[best_epsilon, inRMSE, outRMSE] = RegressionCrossValidation(wine, 10, 5, epsilon);

% report the result
fprintf('best_epsilon\n');
disp(best_epsilon);
fprintf('correspond_inRMSE\n');
disp(inRMSE);
fprintf('correspond_outRMSE\n');
disp(outRMSE);
