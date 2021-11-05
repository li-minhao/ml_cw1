function PolyRegressionCV(D, k1, k2, C, q, epsilon)
% input dataset D, outer k1 fold, inner k2 fold, C, q, and Epsilon
% report the best hyperparameter chosen and its correspond RMSE

    outSplitNum = floor(size(D,1)/k1); %calculate the number of sample splitted by outer KFold
    inSplitNum = floor((size(D,1)-outSplitNum)/k2);
    randidx_out = randperm(size(D,1)); %generate random index to shuffle the data
    randidx_in = randperm((size(D,1)-outSplitNum));
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
        idxHead = (1+(i-1)*outSplitNum);
        idxTail = i*outSplitNum;
        D_out = D(randidx_out(idxHead:idxTail),:);
        D_in = D;
        D_in(randidx_out(idxHead:idxTail),:) = [];
        % Initialise the accracy
        best_RMSE = inf;
        inner_RMSE = [];
        fprintf('Fold %d:\n',i)
        % Searching the hyperparameter C
        for m = 1:length(C)
            % Searching the hyperparameter q
            for n = 1:length(q)
                % Searching the hyperparameter Epsilon
                for p = 1:length(epsilon)
                    BoxConstraint = C(m);
                    PolynomialOrder = q(n);
                    Epsilon = epsilon(p);
                    for j = 1:k2
                    % The inner cross validation
                        % Split the data
                        idxHead = (1+(j-1)*inSplitNum);
                        idxTail = min(size(D_in,1),j*inSplitNum);
                        D_val = D_in(randidx_in(idxHead:idxTail),:);
                        D_train = setxor(D_in, D_val, 'row', 'stable');
                        % Extract features X and labels y
                        X_train = D_train(:,1:size(D_train,2)-1);
                        y_train = D_train(:,size(D_train,2));
                        X_val = D_val(:,1:size(D_train,2)-1);
                        y_val = D_val(:,size(D_train,2));
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
        % Extract features X and labels y
        X_train = D_in(:,1:size(D_train,2)-1);
        y_train = D_in(:,size(D_train,2));
        X_val = D_out(:,1:size(D_train,2)-1);
        y_val = D_out(:,size(D_train,2));
        % Fit the model
        M = fitrsvm(X_train,y_train,'Standardize',true,'KernelFunction','polynomial','BoxConstraint',C_best,'PolynomialOrder',q_best,'Epsilon',Epsilon_best);
        % Make predictions on test set
        X_pdt = predict(M, X_val);
        % Calculate accuracy
        clf_RMSE = rmse(X_pdt,y_val);
        outRMSE = [outRMSE,clf_RMSE];
        support_vec_num(end+1) = sum(svIdx);
        support_vec_percentage(end+1) = sum(svIdx)/length(X_train)*100;
        fprintf('outerCV: outerFold:%d, C:%.3f, q:%.3f Epsilon:%.3f svNum:%d(%.3f%%), estRMSE:%.3f, testRMSE:%.3f\n\n',i,C_best,q_best,Epsilon_best,sum(svIdx),sum(svIdx)/length(X_train)*100,best_RMSE,clf_RMSE)
    end
    % report the result
    fprintf('best_C\n')
    disp(best_C)
    fprintf('best_q\n')
    disp(best_q)
    fprintf('best_Epsilon\n');
    disp(best_Epsilon);
    fprintf('correspond_inRMSE\n');
    disp(inRMSE);
    fprintf('correspond_outRMSE\n');
    disp(outRMSE);
    fprintf('support_vec_num\n')
    disp(support_vec_num)
    fprintf('support_vec_percentage\n')
    disp(support_vec_percentage)
end