%% Task 2
%% Data preparation
iris = readtable('iris.csv');
X = iris(:,1:4);
X = table2array(X);
Y = iris(:,5);
Y = table2cell(Y);
idx = ~strcmp(Y,'Iris-setosa');
X = X(idx,:);
Y = strcmp(Y(idx,:),'Iris-versicolor');
D = [X,Y];
% Initialize the hyperparameter
C = linspace(0.5,2,10);
sigma = linspace(1,5,5);
q = linspace(1,5,5);
k1 = 10;
k2 = 5;

%% RBF kernal
[best_C, best_sigma, correspond_inacc, correspond_outacc, support_vec_num, support_vec_percentage] = RBFClassificationCV(D, k1, k2, C, sigma);
fprintf('=========== RBF kernal ============\n')
% report the result
fprintf('Best %d models:\n',k1)
for i = 1:k1
    fprintf('C: %.3f, ',best_C(i))
    fprintf('Sigma:%.2f, ',best_sigma(i))
    fprintf('sv_num:%d(%.1f%%), ',support_vec_num(i),support_vec_percentage(i))
    fprintf('OutTestAcc:%.4f, ',correspond_outacc(i))
    fprintf('InnerBestAcc:%.4f\n',correspond_inacc(i))
end
fprintf('\nAverage test accuracy:%.6f\n',mean(correspond_outacc))
fprintf('Average support vector:%.1f(%.1f%%)\n\n',mean(support_vec_num),mean(support_vec_percentage))


%% Polynomial kernal
% Training
[best_C, best_q, correspond_inacc, correspond_outacc, support_vec_num, support_vec_percentage] = PolyClassificationCV(D, 10, 5, C, q);
% Report the result
fprintf('=========== Poly kernal ============\n')
fprintf('\nBest %d models:\n',k1)
for i = 1:k1
    fprintf('C: %.3f, ',best_C(i))
    fprintf('q:%.2f, ',best_q(i))
    fprintf('sv_num:%d(%.1f%%), ',support_vec_num(i),support_vec_percentage(i))
    fprintf('OutTestAcc:%.4f, ',correspond_outacc(i))
    fprintf('InnerBestAcc:%.4f\n',correspond_inacc(i))
end
fprintf('\nAverage test accuracy:%.6f\n',mean(correspond_outacc))
fprintf('Average support vector:%.1f(%.1f%%)\n\n',mean(support_vec_num),mean(support_vec_percentage))

