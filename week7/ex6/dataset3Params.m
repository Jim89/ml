function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% PSUEDO CODE
% FOR EACH C
% FOR EACH SIGMA
% TRAIN THE SVM
% COMPUTE Jcv
% FIND COMBO OF C AND SIGMA WITH LOWEST Jcv

x1 = X(:,1);
x1 = x1(:);
x2 = X(:,2);
x2 = x2(:);


values = [0.01 0.03 0.1 0.3 1 3 10 30];
for C = 1:length(values);
    C_iter = values(C);
    for sig = 1:length(values);
        sigma_iter = values(sig);
        model = svmTrain(X, y, C_iter, @(x1, x2) gaussianKernel(x1, x2, sigma_iter));
        predictions = svmPredict(model, Xval);
        iter_error = mean(double(predictions ~= yval));
        C_val(C, sig) = C_iter;
        Sigma_val(C, sig) = sigma_iter;
        error(C, sig) = iter_error;
    end;
end;

[m i] = max(error == min(min(error)));
[mcol icol] = max(m)

col = icol;
row = max(i);

C = C_val(row, col);
sigma = Sigma_val(row, col);
% =========================================================================

end
