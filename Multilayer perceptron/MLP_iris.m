%% Dataset
load fisheriris
[Ntotal, D]= size(meas);
M = 3;
features = meas;
Ybinary = zeros(Ntotal, 3);
Ybinary(1:50, 1) = 1;
Ybinary(51:100, 2) = 1;
Ybinary(101:150, 3) = 1;

rng(1); idx = randperm(150);
features = features(idx, :);
Ybinary = Ybinary(idx, :);

Ntrain = 100; Ntest = 50;
trainx = features(1:Ntrain, :);  Ytrain = Ybinary(1:Ntrain, :);
testx = features(Ntrain+1:end, :); Ytest = Ybinary(Ntrain+1:end, :);

amin = 0; amax = 1;
trainxp = trainx;
for i = 1:D
    trainx(:,i) = (amax - amin)*(trainx(:,i) - min(trainxp(:,i)))/(max(trainxp(:,i)) - min(trainxp(:,i))) + amin;
    testx(:,i) = (amax - amin)*(testx(:,i) - min(trainxp(:,i)))/(max(trainxp(:,i)) - min(trainxp(:,i))) + amin;
end
trainxp = trainx;
Ytrainp = Ytrain;


%% Network
Nl = [D+1, 10, M];   % Number of neurons in each layer (number of hidden units is tunable, more hidden layers can also be added simply by specifying number of hidden units in each consecutive layer)


%% Network parameters (all tunable)
eta = 0.1;
a = 0.5;
b = 0.4;
maxepochs = 500;


%% Initialization
W = cell(length(Nl) - 1, 1);
for l = 1:length(Nl)-1 
   W{l, 1} = 0.1*rand(Nl(l+1), Nl(l));
end
V = cell(length(Nl) - 1, 1);        % weighted sums at each layer
A = cell(length(Nl) - 1, 1);        % activations at each layer
Ader = cell(length(Nl) - 1, 1);     % activation gradients at each layer
Del = cell(length(Nl) - 1, 1);      % local gradients at each layer

errorcheck = zeros(maxepochs*Ntrain, 1);
count = 1;


%% Training
for epoch = 1:maxepochs
    
   % Randomize input order presentation
   I = randperm(Ntrain);
   trainx = trainxp(I, :);
   Ytrain = Ytrainp(I, :);
   
   for n = 1:Ntrain
       
      % Select a training data point and corresponding label
      X = [trainx(n, :) 1]';
      Y = Ytrain(n, :)';
      
      % Forward pass
      for l = 1:length(Nl) - 1
          if l == 1
              V{l, 1} = W{l, 1}*X;
          else
              V{l, 1} = W{l, 1}*A{l-1, 1}; 
          end
          [Al, Alder] = relu(V{l, 1});
          A{l, 1} = Al;
          Ader{l, 1} = Alder;
      end
      
      
      % Error
      e = Y - A{l, 1};
      
      % Calculate local gradients
      for l = length(Nl)-1:-1:1
          if l == length(Nl) - 1
             Del{l, 1} = e.*Ader{l, 1};
          else
             Del{l, 1} = Ader{l, 1}.*(sum(Del{l+1, 1}.*W{l+1, 1})');
          end
      end
      
      
      % Update parameters in backward pass
      for l = length(Nl)-1:-1:1
          if l == 1
              W{l, 1} = W{l, 1} + eta*(Del{l, 1}*X');
          else
              W{l, 1} = W{l, 1} + eta*(Del{l, 1}*A{l-1, 1}');
          end
      end
     
       
      % Store error
      errorcheck(count, 1) = mean(e.^2);
      count = count + 1;
       
   end
    
    
end

figure; 
set(gca, 'FontSize', 15);
plot(smooth(errorcheck, 100));
xlabel('Iterations', 'FontSize', 15);
ylabel('Squared loss', 'FontSize', 15);


%% Testing
train_acc = 0;
test_acc = 0;
conf_mat_train = zeros(M, M);
conf_mat_test = zeros(M, M);

for p = 1:2
    
    N = Ntrain*(p==1) + Ntest*(p==2);
    if p == 1
        XX = trainx;
        YY = Ytrain;
    else
        XX = testx;
        YY = Ytest;
    end
    
    for n = 1:N
        
        X = [XX(n, :) 1]';
        Y = YY(n, :)';
        [~, indtrue] = max(Y);
        
        % Forward pass
        for l = 1:length(Nl) - 1
          if l == 1
              Vl = W{l, 1}*X;
          else
              Vl = W{l, 1}*Al; 
          end
          [Al, Alder] = relu(Vl);
        end
        
        
        [~, ind] = max(Al);
        
        if p == 1
            conf_mat_train(indtrue, ind) = conf_mat_train(indtrue, ind) + 1;
            train_acc = train_acc + 1*(ind==indtrue);
        else
            conf_mat_test(indtrue, ind) = conf_mat_test(indtrue, ind) + 1;
            test_acc = test_acc + 1*(ind==indtrue);
        end
        
        
    end
    
end

fprintf('Confusion matrix for training data:\n')
conf_mat_train
fprintf('\n train accuracy = %f%\n', (train_acc/Ntrain)*100);
fprintf('Confusion matrix for test data:\n')
conf_mat_test
fprintf('\n test accuracy = %f%\n', (test_acc/Ntest)*100);