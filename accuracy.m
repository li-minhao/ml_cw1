
% input: train label result and the of test set
% output: accuracy

function acc = accuracy(train,test)
    n = size(train,1);
    m = 0;
    for i=1:n
        if train(i) == test(i)
            m = m+1;
        end
    end
    acc = m/n;
end