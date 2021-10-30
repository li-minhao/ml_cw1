clear
%% iris

%% load data
M = readtable("iris.csv");
X = M(:,1:4);
X = table2array(X);

Y = M(:,5);
Y = table2cell(Y);
Y = categorical(Y);
Y = double(Y);

X = X(51:150,:);
Y = Y(51:150,:);

%% cross validation
% set ratio
testRatio = 0.3;

trainIndices = crossvalind('HoldOut', size(X, 1), testRatio);
testIndices = ~trainIndices;

% obtain train set and test set
train_X = X(trainIndices, :);
train_Y = Y(trainIndices, :);

test_X = X(testIndices, :);
test_Y = Y(testIndices, :);

%% fit classfication svm model
% fit model
Mdl= fitcsvm(train_X,train_Y,"KernelFunction","linear","BoxConstraint",1);

% predict the label for test set
predictLabel = predict(Mdl,test_X);
acc = accuracy(predictLabel,test_Y)
pre = precision(predictLabel,test_Y)
re  = recall(predictLabel,test_Y)

