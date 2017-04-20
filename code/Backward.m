function [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a)
% [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a) computes the gradient
% updates to the deep network parameters and returns them in cell arrays
% 'grad_W' and 'grad_b'. This function takes as input:
%   - 'W' and 'b' the network parameters
%   - 'X' and 'Y' the single input data sample and ground truth output vector,
%     of sizes Nx1 and Cx1 respectively
%   - 'act_h' and 'act_a' the network layer pre and post activations when forward
%     forward propogating the input smaple 'X'

grad_W = W;
grad_b = b;

grad_h = act_h;
grad_a = act_a;

for i = length(W):-1:1 % numLayers-1
    
    if i == length(W)
        %% compute grad_a
        % Check: Y must be one-hot vector
        grad_a{i} = act_h{i} - Y;
    else
        %% compute grad_a
        grad_h{i} = sum(repmat(grad_a{i+1},[1,size(W{i+1},1)]) .* W{i+1}');
        grad_h{i} = grad_h{i}';
        grad_a{i} = act_h{i}.*(1-act_h{i}) .* grad_h{i};
    end

    if i ~= 1
        last_h = act_h{i-1};
    else
        last_h = X;
    end
    grad_W{i} = last_h * grad_a{i}'; % grad_a{i}'s dimension is succedding layer's #nodes * 1
                                     % act_h{i-1}'s dimension is 1 * previous layer's #nodes
    grad_b{i} = grad_a{i};
end
