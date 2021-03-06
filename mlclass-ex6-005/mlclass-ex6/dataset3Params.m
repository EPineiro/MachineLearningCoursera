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


test_vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
best_error = 1000000;
best_C = C;
best_sigma = sigma;

for i = 1:length(test_vals)
	for j = 1: length(test_vals)
		
		model= svmTrain(X, y, test_vals(i), @(x1, x2) gaussianKernel(x1, x2, test_vals(j)));
		predictions = svmPredict(model, Xval);
		error = mean(double(predictions ~= yval));
	
		if error < best_error
			best_error = error;
			best_C = test_vals(i);
			best_sigma = test_vals(j);
		end
		
	end
end

C = best_C;
sigma = best_sigma;

% =========================================================================

end
