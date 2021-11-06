function [X_new,hasNaN]=preprocess(X,Y)
%function preprocess: check the missing data and correlation of the dataset
%Input: features X and target Y
%Output: features after filtering via correlation and a boolean value
%        hasNaN to describe if the dataset has missing value


    %% find the missing data
    X_NaN = size(find(isnan(X)),1);
    Y_NaN = size(find(isnan(Y)),1);

    hasNaN = true;
    if X_NaN+Y_NaN == 0
        hasNaN = false;
    end

    %% calculate the Pearson correlation
    corrMatrix = corr(X,Y,'type','Pearson');
    X_new = [];
    len = size(X,2);

    for i=1:len
        if abs(corrMatrix(i))>0.05
            X_new = [X_new X(:,i)];
        end
    end

end