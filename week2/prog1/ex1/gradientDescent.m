function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% set up constants
const = alpha * 1/m;        % never changes as alpha and m are constant
x0 = X(:, 1);               % static data
x1 = X(:, 2);               % static data

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
     
    % set up calculations
    theta0 = theta(1);
    theta1 = theta(2);
    
    % perform common calculation
    hx = X*theta;                   % work out predicted values
    e = (hx - y);                   % error = calculate residuals
    
    % adjustments
    temp0 = theta0 - const * sum(e .* x0);          % adjust theta0
    temp1 = theta1 - const * sum(e .* x1);          % adjust theta1

    % updadate theta
    theta = [temp0; temp1];                          % put adjusted values

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
