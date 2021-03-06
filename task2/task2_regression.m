%% Regression task for task 2
close all;
clear;
addpath(genpath(".."));
path_name = 'hyperparameter/';


%% Data preparation
% load wine
wine = table2array(readtable('winequality-white.csv'));
load('dataset/wine_indices.mat');
wine = wine(index,:);
X = wine(:,1:end-1);
y = wine(:,end);

[X,has_NaN] = preprocess(X,y);
if has_NaN
    fprintf("Wine quality dataset has missing value.\n\n");
end

% Initialize the hyperparameter
C = linspace(0.5,1,5);
sigma = linspace(1,5,5);
q = linspace(1,3,3);
epsilon = linspace(0.1,1,5);
k1 = 10;
k2 = 5;


%% Linear kernel training
fprintf('Regression model using linear kernels in training...\n\n')
[best_C_l, best_Epsilon_l, inRMSE_l, outRMSE_l, support_vec_num_l, support_vec_percentage_l] = SVR_linear_cv(X, y, k1, k2, C, epsilon);
% save the best hyperparameters found in a file
save([path_name,'linear_SVR.mat'],'best_C_l','best_Epsilon_l')


%% RBF kernel training
fprintf('Regression model using RBF kernels in training...\n\n')
[best_C_r, best_sigma_r, best_Epsilon_r, inRMSE_r, outRMSE_r, support_vec_num_r, support_vec_percentage_r] = regression_RBF_cv(X, y, k1, k2, C, sigma, epsilon);
% save the best hyperparameters found in a file
save([path_name,'RBF_regression.mat'],'best_C_r','best_sigma_r','best_Epsilon_r')


%% Polynomial kernel Training
fprintf('Regression model using Polynomial kernels in training...\n\n')
[best_C_p, best_q_p, best_Epsilon_p, inRMSE_p, outRMSE_p, support_vec_num_p, support_vec_percentage_p] = regression_poly_cv(X, y, k1, k2, C, q, epsilon);
% save the best hyperparameters found in a file
save([path_name,'poly_regression.mat'],'best_C_p','best_q_p','best_Epsilon_p')


%% Linear kernel report the result
fprintf('=========== Linear kernal ============\n\n')
fprintf('Best %d models: \n\n',k1)
for i = 1:k1
    fprintf('C: %.3f, ', best_C_l(i))
    fprintf('Epsilon: %.3f, ', best_Epsilon_l(i))
    fprintf('sv_num: %d(%.1f%%), ', support_vec_num_l(i), support_vec_percentage_l(i))
    fprintf('OuterTestRMSE: %.4f, ', inRMSE_l(i))
    fprintf('InnerBestRMSE: %.4f\n', outRMSE_l(i))
end
fprintf('\nAverage test RMSE: %.6f\n', mean(outRMSE_l))
fprintf('Average support vector: %.1f(%.1f%%)\n\n', mean(support_vec_num_l), mean(support_vec_percentage_l))


%% RBF kernel report the result
fprintf('=========== RBF kernal ============\n\n')
fprintf('Best %d models: \n\n',k1)
for i = 1:k1
    fprintf('C: %.3f, ', best_C_r(i))
    fprintf('Sigma: %.2f, ', best_sigma_r(i))
    fprintf('Epsilon: %.3f, ', best_Epsilon_r(i))
    fprintf('sv_num: %d(%.1f%%), ', support_vec_num_r(i), support_vec_percentage_r(i))
    fprintf('OuterTestRMSE: %.4f, ', inRMSE_r(i))
    fprintf('InnerBestRMSE: %.4f\n', outRMSE_r(i))
end
fprintf('\nAverage test RMSE: %.6f\n', mean(outRMSE_r))
fprintf('Average support vector: %.1f(%.1f%%)\n\n', mean(support_vec_num_r), mean(support_vec_percentage_r))


%% Polynomial kernal report the result
fprintf('=========== Poly kernal ============\n\n')
fprintf('Best %d models: \n\n',k1)
for i = 1:k1
    fprintf('C: %.3f, ', best_C_p(i))
    fprintf('q: %.2f, ', best_q_p(i))
    fprintf('Epsilon: %.3f, ', best_Epsilon_p(i))
    fprintf('sv_num: %d(%.1f%%), ', support_vec_num_p(i), support_vec_percentage_p(i))
    fprintf('OuterTestRMSE: %.4f, ', inRMSE_p(i))
    fprintf('InnerBestRMSE: %.4f\n', outRMSE_p(i))
end
fprintf('\nAverage test RMSE: %.6f\n', mean(outRMSE_p))
fprintf('Average support vector: %.1f(%.1f%%)\n\n', mean(support_vec_num_p), mean(support_vec_percentage_p))

