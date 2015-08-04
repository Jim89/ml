function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% set up constants
const = 1/m;

% common properties
thetax = (X*theta);
gthetax = sigmoid(thetax);

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
    grad(i) = const .* sum(diff_grad .* X(:, i));
end    

% return values
J = const * j_summat;
grad = grad;


% =============================================================

end
