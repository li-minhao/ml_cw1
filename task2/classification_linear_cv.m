function [best_C, correspond_inacc, correspond_outacc, support_vec_num, support_vec_percentage] = classification_linear_cv(X, Y, k1, k2, C)
% function LinearNestCrossValidation
% SVM using Linear kernel 
% input dataset X, y, outer k1 fold, inner k2 fold, box constraint C
% report the best hyperparameter chosen and its correspond accuracy

    outSplitNum = floor(size(X,1)/k1); %calculate the number of sample splitted by outer KFold
    randidx_out = randperm(size(X,1)); %generate random index to shuffle the data
    randidx_in = randperm((size(X,1)-outSplitNum));
    best_C = [];
    correspond_inacc = [];
    correspond_outacc = [];
    support_vec_num = [];
    support_vec_percentage = [];
    for i = 1:k1
    % The outer cross validation
        % Randomly split the data
        [OuttrainX,OuttrainY,OuttestX,OuttestY] = KFoldGroup(X,Y,k1,i,randidx_out);
        % Initialise the accracy
        best_acc = 0;
        inner_acc = [];
        fprintf('Fold %d:\n',i)
        for m = 1:length(C)
            % Searching the hyperparameter C
                BoxConstraint = C(m);
                for j = 1:k2
                % The inner cross validation
                    % Split the data
                    [X_train,y_train,X_val,y_val] = KFoldGroup(OuttrainX,OuttrainY,k2,j,randidx_in);
                    % Fit the model
                    M = fitcsvm(X_train,y_train,'Standardize',true,'KernelFunction','linear','BoxConstraint',BoxConstraint);
                    svInd = M.IsSupportVector;
                    % Make predictions on validation set
                    X_pdt = predict(M, X_val);
                    % Calculate accuracy
                    clf_acc = accuracy(X_pdt,y_val);
                    inner_acc = [inner_acc,clf_acc];
                end
                % find the mean accuracy of k results
                k_inner_acc = mean(inner_acc);
                fprintf('innerCV: C:%.3f, svNum:%d(%.3f%%), ValAcc:%.6f, best_acc_sofar:%.6f\n',BoxConstraint,sum(svInd),sum(svInd)/length(X_train)*100,k_inner_acc,best_acc)
                if k_inner_acc > best_acc
                    %find the best accuracy and the hyperparameter
                    best_acc = k_inner_acc;
                    C_best = BoxConstraint;
                end
        end
        % append the best hyperparameter searched in inner cv
        correspond_inacc = [correspond_inacc, best_acc];
        best_C = [best_C, C_best];
        % use the best hyperparameter to have outer cv
        % Fit the model
        M = fitcsvm(OuttrainX,OuttrainY,'Standardize',true,'KernelFunction','linear','BoxConstraint',C_best);
        svInd = M.IsSupportVector;
        % Make predictions on test set
        X_pdt = predict(M, OuttestX);
        % Calculate accuracy
        clf_acc = accuracy(X_pdt,OuttestY);
        correspond_outacc = [correspond_outacc,clf_acc];
        sv_num = sum(svInd);
        sv_per = sv_num/length(X_train)*100;
        support_vec_num = [support_vec_num, sv_num];
        support_vec_percentage = [support_vec_percentage, sv_per];
        fprintf('\nouterCV:outerFold:%d, C:%.3f, \nsvNum:%d(%.3f%%), estAcc:%.3f, testAcc:%.3f\n\n',i,C_best,sum(svInd),sum(svInd)/length(X_train)*100,best_acc,clf_acc)
    end
end