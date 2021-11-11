%% Classification task for Task 2
close all;
clear;
addpath(genpath("./dataset"));
path_name = 'hyper_parameter/';


%% Data preparation
% load iris
iris = readtable("iris.csv");
X = table2array(iris(:,1:4));
Y = iris(:,5);
Y = double(categorical(table2cell(Y)));

% remove the first category for binary classification
X = X(51:150,:);
Y = Y(51:150,:);

% Initialize the hyperparameter
C = linspace(0.5,2,10);
sigma = linspace(1,5,5);
q = linspace(1,5,5);
k1 = 10;
k2 = 5;


%% Linear kernel
% Training
fprintf('Classification model linear using RBF kernels in training...\n\n')
[best_C_l, correspond_inacc_l, correspond_outacc_l, support_vec_num_l, support_vec_percentage_l] = classification_linear_cv(X,Y, k1, k2, C);

% save the best hyperparameters found in a file
save([path_name,'linear_classification.mat'],'best_C_l')


%% RBF kernel
% Training
fprintf('Classification model using RBF kernels in training...\n\n')
[best_C_rbf, best_sigma_rbf, correspond_inacc_r, correspond_outacc_r, support_vec_num_r, support_vec_percentage_r] = classification_RBF_cv(X,Y, k1, k2, C, sigma);
% save the best hyperparameters found in a file
save([path_name,'RBF_classification.mat'],'best_C_rbf','best_sigma_rbf')


%% Polynomial kernel
% Training
fprintf('Classification model using Polynomial kernels in training...\n\n')
[best_C_poly, best_q_poly, correspond_inacc_p, correspond_outacc_p, support_vec_num_p, support_vec_percentage_p] = classification_poly_cv(X, Y, 10, 5, C, q);
% save the best hyperparameters found in a file
save([path_name,'poly_classification.mat'],'best_C_poly','best_q_poly')


%% Report the result for Linear kernel
fprintf('=========== Linear kernal ============\n')
fprintf('Best %d models:\n',k1)
for i = 1:k1
    fprintf('C: %.3f, ',best_C_l(i))
    fprintf('sv_num:%d(%.1f%%), ',support_vec_num_l(i),support_vec_percentage_l(i))
    fprintf('OuterTestAcc:%.4f, ',correspond_outacc_l(i))
    fprintf('InnerBestAcc:%.4f\n',correspond_inacc_l(i))
end
fprintf('\nAverage test accuracy:%.6f\n',mean(correspond_outacc_l))
fprintf('Average support vector:%.1f(%.1f%%)\n\n',mean(support_vec_num_l),mean(support_vec_percentage_l))


%% report the result for RBF kernel
fprintf('=========== RBF kernal ============\n')
fprintf('Best %d models:\n',k1)
for i = 1:k1
    fprintf('C: %.3f, ',best_C_rbf(i))
    fprintf('Sigma:%.2f, ',best_sigma_rbf(i))
    fprintf('sv_num:%d(%.1f%%), ',support_vec_num_r(i),support_vec_percentage_r(i))
    fprintf('OuterTestAcc:%.4f, ',correspond_outacc_r(i))
    fprintf('InnerBestAcc:%.4f\n',correspond_inacc_r(i))
end
fprintf('\nAverage test accuracy:%.6f\n',mean(correspond_outacc_r))
fprintf('Average support vector:%.1f(%.1f%%)\n\n',mean(support_vec_num_r),mean(support_vec_percentage_r))


%% Report the result for polynomial kernel
fprintf('=========== Poly kernal ============\n')
fprintf('\nBest %d models:\n',k1)
for i = 1:k1
    fprintf('C: %.3f, ',best_C_poly(i))
    fprintf('q:%.2f, ',best_q_poly(i))
    fprintf('sv_num:%d(%.1f%%), ',support_vec_num_p(i),support_vec_percentage_p(i))
    fprintf('OuterTestAcc:%.4f, ',correspond_outacc_p(i))
    fprintf('InnerBestAcc:%.4f\n',correspond_inacc_p(i))
end
fprintf('\nAverage test accuracy:%.6f\n',mean(correspond_outacc_p))
fprintf('Average support vector:%.1f(%.1f%%)\n\n',mean(support_vec_num_p),mean(support_vec_percentage_p))


