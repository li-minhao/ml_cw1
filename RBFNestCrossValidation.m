function RBFNestCrossValidation(D, k1, k2, C, sigma)
% function RBFNestCrossValidation
% SVM using Gaussian RBF kernel 
% input dataset D, outer k1 fold, inner k2 fold, box constraint C
% and kernel scale sigma
% report the best hyperparameter chosen and its correspond accuracy

    outSplitNum = round(size(D,1)/k1); %calculate the number of sample splitted by outer KFold
    inSplitNum = round((size(D,1)-outSplitNum)/k2);
    randidx_out = randperm(size(D,1)); %generate random index to shuffle the data
    randidx_in = randperm((size(D,1)-outSplitNum));
    best_C = [];
    best_sigma = [];
    correspond_acc = [];
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
                    % Make predictions on validation set
                    X_pdt = predict(M, X_val);
                    % Calculate accuracy
                    clf_acc = accuracy(X_pdt,y_val);
                    inner_acc = [inner_acc,clf_acc];
                end
                % find the mean accuracy of k results
                k_inner_acc = mean(inner_acc);
                if k_inner_acc > best_acc
                    %find the best accuracy and the hyperparameter
                    best_acc = k_inner_acc;
                    C_best = BoxConstraint;
                    sigma_best = KernelScale;
                end
            end
        end
        % append the best hyperparameter searched in inner cv
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
        % Make predictions on test set
        X_pdt = predict(M, X_val);
        % Calculate accuracy
        clf_acc = accuracy(X_pdt,y_val);
        correspond_acc = [correspond_acc,clf_acc];
    end
    % report the result
    fprintf('best_C\n')
    disp(best_C)
    fprintf('best_sigma\n')
    disp(best_sigma)
    fprintf('correspond_acc\n')
    disp(correspond_acc)
end