%% Classification task for Task 2
%% Data preparation
iris = readtable('iris.csv');
X = iris(:,1:4);
X = table2array(X);
Y = iris(:,5);
Y = table2cell(Y);
idx = ~strcmp(Y,'Iris-setosa');
X = X(idx,:);
Y = strcmp(Y(idx,:),'Iris-versicolor');

C = linspace(0.5,2,10);
sigma = linspace(1,5,5);
q = linspace(1,5,5);
k1 = 10;
k2 = 5;

%% RBF kernal
% Training
fprintf('Classification model using RBF kernels in training...\n\n')
[best_C_rbf, best_sigma_rbf, correspond_inacc_r, correspond_outacc_r, support_vec_num_r, support_vec_percentage_r] = RBFClassificationCV(X,Y, k1, k2, C, sigma);
% save the best hyperparameters found in a file
save('RBFClassification.mat','best_C_rbf','best_sigma_rbf')

%% Polynomial kernal
% Training
fprintf('Classification model using Polynomial kernels in training...\n\n')
[best_C_poly, best_q_poly, correspond_inacc_p, correspond_outacc_p, support_vec_num_p, support_vec_percentage_p] = PolyClassificationCV(X, Y, 10, 5, C, q);
% save the best hyperparameters found in a file
save('PolyClassification.mat','best_C_poly','best_q_poly')

%% report the result for RBF kernel
fprintf('=========== RBF kernal  ============\n')
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


