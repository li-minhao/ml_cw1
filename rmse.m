function res = rmse(predicted,expected)
    res = sqrt(mean((predicted-expected).^2));
end

