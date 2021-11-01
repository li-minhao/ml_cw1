%% regression task for task 2
wine_red = table2array(readtable('winequality-red.csv'));
wine_white = table2array(readtable('winequality-white.csv'));
wine = [wine_red;wine_white];

epsilon = linspace(1,2,20);


% RBF kernal
[best_epsilon, inRMSE, outRMSE] = RegressionCrossValidation(wine, 10, 5, epsilon);

% report the result
fprintf('best_epsilon\n');
disp(best_epsilon);
fprintf('correspond_inRMSE\n');
disp(inRMSE);
fprintf('correspond_outRMSE\n');
disp(outRMSE);
