function [trainX,trainY,testX,testY] = KFoldGroup(X,Y,k,n,rnd_index)
%function KFoldGroup: split the train and test set by hand
%Input:
%X:samples,
%Y:labels,
%k:the number of k fold
%n:output the n-th fold of of data. Note that n<=k
%rnd_index:the random row index of the sample, size(rnd_index,1)=size(X,1)

    SplitNum = round(size(X,1)/k);
    idxHead = (1+(n-1)*SplitNum);
    idxTail = min(size(X,1),n*SplitNum);
    % obtain train set and test set
    trainX = X;
    trainX(rnd_index(idxHead:idxTail),:) = [];
    trainY = Y;
    trainY(rnd_index(idxHead:idxTail),:) = [];
    testX = X(rnd_index(idxHead:idxTail),:);
    testY = Y(rnd_index(idxHead:idxTail),:);
end

