function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

l_lambda = [0; repmat(lambda,size(theta,1)-1,1)];

sigmoidTerm = sigmoid(X * theta);
J = (sum(y .* log(sigmoidTerm) + (1 - y) .* log(1 - sigmoidTerm)) ./ -m) + ((l_lambda' * (theta.^2)) / (2 * m));
grad = (((sigmoidTerm - y)' * X)' + (l_lambda .* theta)) ./ m;

% sigmoidTerm = X * theta;
% J = (sum(y .* log(sigmoid(sigmoidTerm)) + (1 - y) .* log(1 - sigmoid(sigmoidTerm))) ./ -m) + (l_lambda' * (theta.^2)) / (2 * m);
% grad = (((sigmoid(sigmoidTerm) - y)' * X) + (l_lambda' * theta)) ./ m;

% J = (sum(y .* log(sigmoid(sigmoidTerm)) + (1 - y) .* log(1 - sigmoid(sigmoidTerm))) ./ -m);
% grad = (((sigmoid(sigmoidTerm) - y)' * X)) ./ m;

% =============================================================

end
