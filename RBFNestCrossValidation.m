function RBFNestCrossValidation(D, k1, k2, C, sigma)
% function RBFNestCrossValidation
% SVM using Gaussian RBF kernel.
% Input dataset D, outer k1 fold, inner k2 fold, box constraint C
% and kernel scale sigma.
% Report the ratio of support vector, the best hyperparameter chosen 
% and their corresponding accuracy.

    outSplitNum = round(size(D,1)/k1); %calculate the number of sample splitted by outer KFold
    inSplitNum = round((size(D,1)-outSplitNum)/k2);
    randidx_out = randperm(size(D,1)); %generate random index to shuffle the data
    randidx_in = randperm((size(D,1)-outSplitNum));
    best_C = [];
    best_sigma = [];
    correspond_inacc = [];
    correspond_outacc = [];
    support_vec_num = [];
    support_vec_percentage = [];
    for i = 1:k1
    % The outer cross validation
        % Randomly split the data
        idx1 = (1+(i-1)*outSplitNum);
        idx2 = min(size(D,1),i*outSplitNum);
        D_out = D(randidx_out(idx1:idx2),:);
        D_in = D;
        D_in(randidx_out(idx1:idx2),:) = [];
        % Initialise the accracy
        best_acc = 0;
        inner_acc = [];
        fprintf('Fold: %d\n',i)
        for m = 1:length(C)
            % Searching the hyperparameter C
            for n = 1:length(sigma)
            % Searching the hyperparameter sigma 
                KernelScale = sigma(n);
                BoxConstraint = C(m);
                for j = 1:k2
                % The inner cross validation
                    % Split the data
                    idx1 = (1+(j-1)*inSplitNum);
                    idx2 = min(size(D_in,1),j*inSplitNum);
                    D_val = D_in(randidx_in(idx1:idx2),:);
                    D_train = D_in;
                    D_train(randidx_in(idx1:idx2),:) = [];
                    % Extract features X and labels y
                    X_train = D_train(:,1:size(D_train,2)-1);
                    y_train = D_train(:,size(D_train,2));
                    X_val = D_val(:,1:size(D_train,2)-1);
                    y_val = D_val(:,size(D_train,2));
                    % Fit the model
                    M = fitcsvm(X_train,y_train,'Standardize',true,'KernelFunction','RBF','BoxConstraint',BoxConstraint,'KernelScale',KernelScale);
                    svInd = M.IsSupportVector;
                    % Make predictions on validation set
                    X_pdt = predict(M, X_val);
                    % Calculate accuracy
                    clf_acc = accuracy(X_pdt,y_val);
                    inner_acc = [inner_acc,clf_acc];
                end
                % find the mean accuracy of k results
                k_inner_acc = mean(inner_acc);
                fprintf('innerCV: C:%.3f, sigma:%.3f, svNum:%d(%.3f%%), ValAcc:%.6f, best_acc_sofar:%.6f\n',BoxConstraint,KernelScale,sum(svInd),sum(svInd)/length(X_train)*100,k_inner_acc,best_acc)
                if k_inner_acc > best_acc
                    %find the best accuracy and the hyperparameter
                    best_acc = k_inner_acc;
                    C_best = BoxConstraint;
                    sigma_best = KernelScale;
                end
            end
        end
        % append the best hyperparameter searched in inner cv
        
        correspond_inacc = [correspond_inacc, best_acc];
        best_C = [best_C, C_best];
        best_sigma = [best_sigma, sigma_best];
        % use the best hyperparameter to have outer cv
        % Extract features X and labels y
        X_train = D_in(:,1:size(D_train,2)-1);
        y_train = D_in(:,size(D_train,2));
        X_val = D_out(:,1:size(D_train,2)-1);
        y_val = D_out(:,size(D_train,2));
        % Fit the model
        M = fitcsvm(X_train,y_train,'Standardize',true,'KernelFunction','RBF','BoxConstraint',C_best,'KernelScale',sigma_best);
        svInd = M.IsSupportVector;
        % Make predictions on test set
        X_pdt = predict(M, X_val);
        % Calculate accuracy
        clf_acc = accuracy(X_pdt,y_val);
        correspond_outacc = [correspond_outacc,clf_acc];
        sv_num = sum(svInd);
        sv_per = sv_num/length(X_train)*100;
        support_vec_num = [support_vec_num, sv_num];
        support_vec_percentage = [support_vec_percentage, sv_per];
        fprintf('\nouterCV:outerFold:%d, C:%.3f, sigma:%.3f\nsvNum:%d(%.3f%%), estAcc:%.3f, testAcc:%.3f\n\n',i,C_best,sigma_best,sum(svInd),sum(svInd)/length(X_train)*100,best_acc,clf_acc)
    end
    % report the result
    fprintf('best_C\n')
    disp(best_C)
    fprintf('best_sigma\n')
    disp(best_sigma)
    fprintf('correspond_valacc\n')
    disp(correspond_inacc)
    fprintf('correspond_testacc\n')
    disp(correspond_outacc)
    fprintf('support_vec_num\n')
    disp(support_vec_num)
    fprintf('support_vec_percentage\n')
    disp(support_vec_percentage)
end