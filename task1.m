
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

%% fit classfication svm model
Mdl= fitcsvm(X,Y,"KernelFunction","linear","BoxConstraint",1);
classOrder = Mdl.ClassNames;
CVSVMModel = crossval(Mdl);
classLoss = kfoldLoss(CVSVMModel)