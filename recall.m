
% input: train label result and the of test set
% output: recall 

function re = recall(train,test)
    n = size(train,1);
    T = test(1);
    p = 0;
    q = 0;
    for i=1:n
        if test(i) == T
           p = p+1;
           if train(i) == T
               q = q+1;
           end
        end
    end
    re = q/p;
end