num_epoch = 30;
classes = 26;
layers = [32*32, 400, classes];
learning_rate = 0.01;

load('../data/nist26_train.mat', 'train_data', 'train_labels')
load('../data/nist26_test.mat', 'test_data', 'test_labels')
load('../data/nist26_valid.mat', 'valid_data', 'valid_labels')

index = 123;
X = train_data(index,:)';
Y = train_labels(index,:)';


[W, b] = InitializeNetwork(layers);
[~, act_h, act_a] = Forward(W, b, X);
[grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a);

[~, loss] = ComputeAccuracyAndLoss(W, b, train_data(index,:), train_labels(index,:));

epsilon = 1e-4;

cnt_e = 0;
cnt = 0;

step = 100;
for i=1
    %% examine W
    for j=1:step:size(W{i},1)
        for k=1:step:size(W{i},2)
            newW = W;
            newW{i}(j,k) = W{i}(j,k) - epsilon;
            [~, loss1] = ComputeAccuracyAndLoss(newW, b, train_data(index,:), train_labels(index,:));
            newW{i}(j,k) = W{i}(j,k) + epsilon;
            [~, loss2] = ComputeAccuracyAndLoss(newW, b, train_data(index,:), train_labels(index,:));
            
            delta = abs(grad_W{i}(j,k) - (loss2-loss1)/(2*epsilon));
            l = grad_W{i}(j,k);
            r = (loss2-loss1)/(2*epsilon);
            
            fprintf('%e: %e vs %e\n', delta , l, r);
            if delta / l > 1
                cnt_e = cnt_e+1;
            end
            cnt = cnt+1;
        end
    end
    %% examine b
    for j=1:size(b{i},1)
        newb = b;
        newb{i}(j) = newb{i}(j) - epsilon;
        [~, loss1] = ComputeAccuracyAndLoss(W, newb, train_data(index,:), train_labels(index,:));
        newb{i}(j) = newb{i}(j) + epsilon;
        [~, loss2] = ComputeAccuracyAndLoss(W, newb, train_data(index,:), train_labels(index,:));

        delta = abs(grad_b{i}(j) - (loss2-loss1)/(2*epsilon));
        l = grad_b{i}(j);
        r = (loss2-loss1)/(2*epsilon);
        
        fprintf('%e: %e vs %e\n', delta , l, r);
        if abs(delta / l) > 1
            cnt_e = cnt_e+1;
        end
        cnt = cnt+1;
    end
    fprintf('Potential error rate: %f\n', cnt_e / cnt);
end
