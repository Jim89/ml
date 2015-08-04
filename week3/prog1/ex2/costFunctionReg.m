function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
features = size(X, 2);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% set up constants
const = 1/m;
const2 = lambda / (2*m);

% common properties
thetax = (X*theta);
gthetax = sigmoid(thetax);
theta_0_squared = power(theta(1), 2);
theta_rest_squared = power(theta(2:end), 2);
theta_squared_sum = theta_0_squared + sum(theta_rest_squared);

% calculations for J(theta):
lhs = log(gthetax);             % lhs log
rhs = log(1 - gthetax);         % rhs log
lhs_mult = -y .* lhs;           % lhs expression
rhs_mult = (1-y) .* rhs;        % rhs expression   
diff = lhs_mult - rhs_mult;     % different
j_summat = sum(diff);             % jtheta different

% calculations for gradient
diff_grad = gthetax - y;

for i = 1:features
    if i == 1
        grad(i) = const .* sum(diff_grad .* X(:, i));
    else
        grad(i) = const .* sum(diff_grad .* X(:, i)) + lambda/m * theta(i);
    end
end    

% return values
J = (const * j_summat) + (const2 .* sum(theta_rest_squared)); % note we don't include theta0 in regularised J
grad = grad;

% =============================================================

end
