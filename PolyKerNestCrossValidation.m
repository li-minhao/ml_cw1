function PolyKerNestCrossValidation(D, k1, k2, C, q)
% function RBFNestCrossValidation
% SVM using Polynomial kernel 
% input dataset D, outer k1 fold, inner k2 fold, box constraint C
% and Polynomial order q
% report the best hyperparameter chosen and its correspond accuracy

    outSplitNum = round(size(D,1)/k1); %calculate the number of sample splitted by outer KFold
    inSplitNum = round((size(D,1)-outSplitNum)/k2);
    randidx_out = randperm(size(D,1)); %generate random index to shuffle the data
    randidx_in = randperm((size(D,1)-outSplitNum));
    best_C = [];
    best_q = [];
    correspond_inacc = [];
    correspond_outacc = [];
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
            for n = 1:length(q)
            % Searching the hyperparameter q 
                PolynomialOrder = q(n);
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
                    M = fitcsvm(X_train,y_train,'Standardize',true,'KernelFunction','polynomial','BoxConstraint',BoxConstraint,'PolynomialOrder',PolynomialOrder);
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
                    q_best = PolynomialOrder;
                end
            end
        end
        % append the best hyperparameter searched in inner cv
        correspond_inacc = [correspond_inacc, best_acc];
        best_C = [best_C, C_best];
        best_q = [best_q, q_best];
        % use the best hyperparameter to have outer cv
        % Extract features X and labels y
        X_train = D_in(:,1:size(D_train,2)-1);
        y_train = D_in(:,size(D_train,2));
        X_val = D_out(:,1:size(D_train,2)-1);
        y_val = D_out(:,size(D_train,2));
        % Fit the model
        M = fitcsvm(X_train,y_train,'Standardize',true,'KernelFunction','polynomial','BoxConstraint',C_best,'PolynomialOrder',PolynomialOrder);
        % Make predictions on test set
        X_pdt = predict(M, X_val);
        % Calculate accuracy
        clf_acc = accuracy(X_pdt,y_val);
        correspond_outacc = [correspond_outacc,clf_acc];
    end
    % report the result
    fprintf('best_C\n')
    disp(best_C)
    fprintf('best_q\n')
    disp(best_q)
    fprintf('correspond_inacc\n')
    disp(correspond_inacc)
    fprintf('correspond_outacc\n')
    disp(correspond_outacc)
end