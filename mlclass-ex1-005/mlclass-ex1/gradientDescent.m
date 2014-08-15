function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	derivatives = zeros(length(theta), 1);
	
	for j = 1:length(theta)
		derivatives(j) = computePartialDerivative(X, y, theta, j);
	end
	
	theta = theta - alpha * derivatives;
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

function d = computePartialDerivative(X, y, theta, j)
	
	m = length(y);
	d = 0;
	
	for i = 1:m
		d =  d + (((X(i,:) * theta) - y(i,1)) * X(i,j));
	end
	
	d = d / m;
	
end
