function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
[n  waste] =  size(theta);
% You need to return the following variables correctly 
J = 0;
t_sum = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

alpha = 1;
summ = 0;
hx = 0;
a=0;
b=0;
for i = 1:m
	f = X(i,:);
	hx = theta' * f';
	hx = sigmoid(hx);
	a = log(hx);
	a = -1*a*y(i,1);
	 
	b = 1-hx;
	b = log(b);
	b=(1 - y(i,1))*b;
	
	a = a-b;
	%hx = theta(1,1) + (theta(2,1) * X(i,2)); %h(tt)
	%hx = hx - y(i,1);  % h(tt) - y
	%hx = hx * hx;    %( h(tt) - y)^2
    summ = summ + a;  %sigma( h(tt) - y)^2
end



J = summ / m;


for i = 1:n
	t_sum = t_sum + ( theta(i,1) * theta(i,1) );
end
t_sum = (t_sum * lambda)/(2*m);


J = J + t_sum;
grad = zeros(size(theta));
display(J);
% =============================================================

end
