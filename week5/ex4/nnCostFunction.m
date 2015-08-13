function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
K = num_labels;
        
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% add the bias unit
X = [ones(m, 1) X];


% PART 1 - FEED FORWARD CALCULATIONS
% calculate hidden layer activations, then add bias unit
hidden_act = sigmoid(X * Theta1');
hidden_act = [ones(size(hidden_act, 1), 1) hidden_act];
% calculate output later
output_act = sigmoid(hidden_act * Theta2');

% PART 2 - CALCULATE THE COST FUNCTION
% compute cost function
for row = 1:m;
    % find the actual answer at each row
    ans = y(row);
    % set up a matrix for y vectors to be held in
    y_mat(row, :) = zeros(num_labels, 1);
    % recode the Kth value of y matrix to be 1 for the correct class
    y_mat(row, ans) = 1;
    % grab what the nnet calculated for each xi
    calculated = output_act(row, :);
    % grab the actual answer for each xi (i.e. yi)
    y_act = y_mat(row, :);
    % inner section of cost function - go accross classes
    for class = 1:K;
        % for each class, work out what out prediction was
        hx = calculated(class);
        % get the actual answer for that class ({0,1})
        y_prob = y_act(class);
        % work out difference for cost function
        diff(class) = (-y_prob * log(hx)) - ((1-y_prob)*log(1-hx));
    end;
    % for each row, sum the Kth differences
    inner(row) = sum(diff);
end;

% compute regularisation terms for the cost function
% theta 1 reg term
for j = 1:size(Theta1, 1);
    theta1(j) = sum((Theta1(j, 2:size(Theta1, 2))).^2);
end;
theta1_reg = sum(theta1);

% theta 2 reg term
for j = 1:size(Theta2, 1);
    theta2(j) = sum((Theta2(j, 2:size(Theta2, 2))).^2);
end;
theta2_reg = sum(theta2);    

% combined regularisation term
reg = (lambda/(2*m)) * (theta1_reg + theta2_reg);
        
% outside the loop - calculate the cost function   
J = ((1/m) * sum(inner)) + reg;

% PART 3 - BACKPROPAGATION:
Delta1 = 0;
Delta2 = 0;
for t = 1:m
    ans = y(t);
    y_mat(t, :) = zeros(num_labels, 1);
    y_mat(t, ans) = 1;    
    
% i. calculate necessary activations
    a1 = X(t, :);
    z2 = Theta1*a1';
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
  
% ii. calculate output delta
    delta3 = a3 - y_mat(t, :)';
    
% iii. calculate hidden layer delta
    delta2 = (Theta2(:, 2:end)'*delta3).*sigmoidGradient(z2);

    
% accumulate the gradient
    Delta1 = Delta1 + delta2*a1;
    Delta2 = Delta2 + delta3*a2';
end;

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end)+ ((lambda/m) * Theta1(:, 2:end));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end)+ ((lambda/m) * Theta2(:, 2:end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
