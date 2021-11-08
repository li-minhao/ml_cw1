function [best_C, best_sigma, best_Epsilon, inRMSE, outRMSE, support_vec_num, support_vec_percentage] = RBFRegressionCV(D, k1, k2, C, sigma, epsilon)
% input dataset D, outer k1 fold, inner k2 fold, C, sigma, and Epsilon
% report the best hyperparameter chosen and its correspond RMSE

    outSplitNum = floor(size(D,1)/k1); %calculate the number of sample splitted by outer KFold
    inSplitNum = floor((size(D,1)-outSplitNum)/k2);
    randidx_out = randperm(size(D,1)); %generate random index to shuffle the data
    randidx_in = randperm((size(D,1)-outSplitNum));
    best_C = [];
    best_sigma = [];
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
            BoxConstraint = C(m);
            % Searching the hyperparameter sigma
            for n = 1:length(sigma)
                KernelScale = sigma(n);
                % Searching the hyperparameter Epsilon
                for p = 1:length(epsilon)                    
                    Epsilon = epsilon(p);
                    for j = 1:k2
                    % The inner cross validation
                        % Split the data
                        idxHead = (1+(j-1)*inSplitNum);
                        idxTail = min(size(D_in,1),j*inSplitNum);
                        D_val = D_in(randidx_in(idxHead:idxTail),:);
                        D_train = D_in;
                        D_train(randidx_in(idxHead:idxTail),:) = [];
                        % Extract features X and labels y
                        X_train = D_train(:,1:size(D_train,2)-1);
                        y_train = D_train(:,size(D_train,2));
                        X_val = D_val(:,1:size(D_train,2)-1);
                        y_val = D_val(:,size(D_train,2));
                        % Fit the model
                        M = fitrsvm(X_train,y_train,'Standardize',true,'KernelFunction','RBF','BoxConstraint',BoxConstraint,'KernelScale',KernelScale,'Epsilon',Epsilon);
                        svIdx = M.IsSupportVector;
                        % Make predictions on validation set
                        X_pdt = predict(M, X_val);
                        % Calculate accuracy
                        clf_RMSE = rmse(X_pdt,y_val);
                        inner_RMSE = [inner_RMSE,clf_RMSE];
                    end
                    % find the mean RMSE of k results
                    k_inner_RMSE = mean(inner_RMSE);
                    fprintf('innerCV: C:%.3f, Sigma:%.3f, Epsilon:%.3f svNum:%d(%.3f%%), valRMSE:%.3f, best_RMSE:%.3f\n',BoxConstraint,KernelScale,Epsilon,sum(svIdx),sum(svIdx)/length(X_train)*100,k_inner_RMSE,best_RMSE)
                    if k_inner_RMSE < best_RMSE
                        %find the best accuracy and the hyperparameter
                        best_RMSE = k_inner_RMSE;
                        C_best = BoxConstraint;
                        sigma_best = KernelScale;
                        Epsilon_best = Epsilon;
                    end
                end
            end
        end
        % append the best hyperparameter searched in inner cv
        inRMSE = [inRMSE, best_RMSE];
        best_C = [best_C, C_best];
        best_sigma = [best_sigma, sigma_best];
        best_Epsilon = [best_Epsilon, Epsilon_best];
        % use the best hyperparameter to have outer cv
        % Extract features X and labels y
        X_train = D_in(:,1:size(D_train,2)-1);
        y_train = D_in(:,size(D_train,2));
        X_val = D_out(:,1:size(D_train,2)-1);
        y_val = D_out(:,size(D_train,2));
        % Fit the model
        M = fitrsvm(X_train,y_train,'Standardize',true,'KernelFunction','RBF','BoxConstraint',C_best,'KernelScale',sigma_best,'Epsilon',Epsilon_best);
        % Make predictions on test set
        X_pdt = predict(M, X_val);
        % Calculate accuracy
        clf_RMSE = rmse(X_pdt,y_val);
        outRMSE = [outRMSE,clf_RMSE];
        support_vec_num(end+1) = sum(svIdx);
        support_vec_percentage(end+1) = sum(svIdx)/length(X_train)*100;
        fprintf('\nouterCV: outerFold:%d, C:%.3f, Sigma:%.3f Epsilon:%.3f svNum:%d(%.3f%%), estRMSE:%.3f, testRMSE:%.3f\n\n',i,C_best,sigma_best,Epsilon_best,sum(svIdx),sum(svIdx)/length(X_train)*100,best_RMSE,clf_RMSE)
    end
end