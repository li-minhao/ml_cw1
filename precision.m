
% input: train label result and the of test set
% output: precision 

function pre = precision(train,test)
    n = size(train,1);
    T = test(1);
    p = 0;
    q = 0;
    for i=1:n
        if train(i) == T
           p = p+1;
           if test(i) == T
               q = q+1;
           end
        end
    end
    pre = q/p;
end