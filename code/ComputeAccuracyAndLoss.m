function [accuracy, loss] = ComputeAccuracyAndLoss(W, b, data, labels)
% [accuracy, loss] = ComputeAccuracyAndLoss(W, b, X, Y) computes the networks
% classification accuracy and cross entropy loss with respect to the data samples
% and ground truth labels provided in 'data' and labels'. The function should return
% the overall accuracy and the average cross-entropy loss.

[outputs] = Classify(W, b, data);

[~, col_id] = max(outputs,[],2);

n = size(data,1);
row_id = (1:n)';

result_matrix = zeros(size(labels));
result_matrix(sub2ind(size(labels),row_id,col_id)) = 1;

numCorrect = sum(sum((labels & result_matrix)));

accuracy = numCorrect / n;

errors = sum(labels.*outputs,2);
loss = -sum(log(errors));

end
