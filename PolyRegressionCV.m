function [best_C, best_q, best_Epsilon, inRMSE, outRMSE, support_vec_num, support_vec_percentage] = PolyRegressionCV(X, Y, k1, k2, C, q, epsilon)
% input dataset D, outer k1 fold, inner k2 fold, C, q, and Epsilon
% report the best hyperparameter chosen and its correspond RMSE

    outSplitNum = floor(size(X,1)/k1); %calculate the number of sample splitted by outer KFold
    randidx_out = randperm(size(X,1)); %generate random index to shuffle the data
    randidx_in = randperm((size(X,1)-outSplitNum));
    best_C = [];
    best_q = [];
    best_Epsilon = [];
    inRMSE = [];
    outRMSE = [];
    support_vec_num = [];
    support_vec_percentage = [];
    for i = 1:k1
    % The outer cross validation
        % Randomly split the data
        [OuttrainX,OuttrainY,OuttestX,OuttestY] = KFoldGroup(X,Y,k1,i,randidx_out);
        % Initialise the accracy
        best_RMSE = inf;
        inner_RMSE = [];
        fprintf('Fold %d:\n',i)
        % Searching the hyperparameter C
        for m = 1:length(C)
            BoxConstraint = C(m);
            % Searching the hyperparameter q
            for n = 1:length(q)
                PolynomialOrder = q(n);
                % Searching the hyperparameter Epsilon
                for p = 1:length(epsilon)
                    Epsilon = epsilon(p);
                    for j = 1:k2
                    % The inner cross validation
                        % Split the data
                        [X_train,y_train,X_val,y_val] = KFoldGroup(OuttrainX,OuttrainY,k2,j,randidx_in);
                        % Fit the model
                        M = fitrsvm(X_train,y_train,'Standardize',true,'KernelFunction','polynomial','BoxConstraint',BoxConstraint,'PolynomialOrder',PolynomialOrder,'Epsilon',Epsilon);
                        svIdx = M.IsSupportVector;
                        % Make predictions on validation set
                        X_pdt = predict(M, X_val);
                        % Calculate accuracy
                        clf_RMSE = rmse(X_pdt,y_val);
                        inner_RMSE = [inner_RMSE,clf_RMSE];
                    end
                    % find the mean RMSE of k results
                    k_inner_RMSE = mean(inner_RMSE);
                    fprintf('innerCV: C:%.3f, q:%.3f, Epsilon:%.3f svNum:%d(%.3f%%), valRMSE:%.3f, best_RMSE:%.3f\n',BoxConstraint,PolynomialOrder,Epsilon,sum(svIdx),sum(svIdx)/length(X_train)*100,k_inner_RMSE,best_RMSE)
                    if k_inner_RMSE < best_RMSE
                        %find the best accuracy and the hyperparameter
                        best_RMSE = k_inner_RMSE;
                        C_best = BoxConstraint;
                        q_best = PolynomialOrder;
                        Epsilon_best = Epsilon;
                    end
                end
            end
        end
        % append the best hyperparameter searched in inner cv
        inRMSE = [inRMSE, best_RMSE];
        best_C = [best_C, C_best];
        best_q = [best_q, q_best];
        best_Epsilon = [best_Epsilon, Epsilon_best];
        % use the best hyperparameter to have outer cv
        % Fit the model
        M = fitrsvm(OuttrainX,OuttrainY,'Standardize',true,'KernelFunction','polynomial','BoxConstraint',C_best,'PolynomialOrder',q_best,'Epsilon',Epsilon_best);
        % Make predictions on test set
        X_pdt = predict(M, OuttestX);
        % Calculate accuracy
        clf_RMSE = rmse(X_pdt,OuttestY);
        outRMSE = [outRMSE,clf_RMSE];
        support_vec_num(end+1) = sum(svIdx);
        support_vec_percentage(end+1) = sum(svIdx)/length(X_train)*100;
        fprintf('\nouterCV: outerFold:%d, C:%.3f, q:%.3f Epsilon:%.3f svNum:%d(%.3f%%), estRMSE:%.3f, testRMSE:%.3f\n\n',i,C_best,q_best,Epsilon_best,sum(svIdx),sum(svIdx)/length(X_train)*100,best_RMSE,clf_RMSE)
    end
end