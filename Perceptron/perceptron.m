% Implements a perceptron online/batch update rule for learning a linear
% threshold boundary between two classes

% Inputs and parameters to be defined by user
rule = 1;      % 1 : implements perceptron batch update rule (updates on all
               % misclassified training samples at one go
               % 2 : implements perceptron online rule (updates only when a sample is misclassified)      
maxepoch = 50; 
eta = 0.1;     % Learning rate
randflag = 1;  % 1 : randomize order of training samples (for online rule)


% Load data
% trainx : Training data (Samples x dimensions)
% Ytrain : Training labels (Samples x no. of classes) e.g. [1 0] --> class 1
% crossx : Cross-validation data (Samples x dimensions)
% Ycross : Test labels (Samples x no. of classes) e.g. [1 0] --> class 1
% Two datasets available: 
% data_lin_sep: linearly separable data (2 clusters)
% data_nonlin_sep : non-linearly separable data (4 clusters)

load data_lin_sep
[N, D] = size(trainx);
x1 = trainx(Ytrain(:, 1) == 1,:);  % class 1 samples
x2 = trainx(Ytrain(:, 2) == 1,:);  % class 2 samples
[N1, ~] = size(x1); [N2, ~] = size(x2);


% Invert all training samples belonging to class 2
x2 = - x2;

% Initial augmented weight vector
a = 0.1*ones(1, (D+1)); 

% Augmented feature vectors
x1 = [ones(N1,1) x1];
x2 = [ - ones(N2,1) x2];
X = [x1; x2];

%Initialization
costvec = [];
errorvec = [];
epoch = 1;
I = 1:N;


while (epoch <= maxepoch)
    
    switch rule
        
        case 1
    
            % Check for misclassified samples
            b = sum(a.*X, 2);
            cost = - sum(b(b<0));
            misclassified = sum(X(b<0, :));
            errorvec = [errorvec (length(find(b<0))*100)/N];
            
            % Update weights
            a = a + eta * misclassified;
            
            
        case 2
            
            if randflag == 1
                I = randperm(N);
            end
            misclassified = 0;
            cost = 0;
            err = 0;
            
            % Take training samples one by one
            for i = 1:N
                x = X(I(i), :);
                b = a*x';
                
                % Update only when a sample is misclassified
                a = a + eta*x*(b<0);
                misclassified = misclassified + x*(b<0);
                cost = cost - b*(b<0);
                err = err + (b<0);
            end
            
            errorvec = [errorvec (err*100)/N];
            
    end
    
    costvec = [costvec cost];
    epoch = epoch + 1;
    
    if sum(misclassified) == 0
        break;
    end
    
end


% Plot training samples along with the classification boundary (only for
% 2D data)
if D == 2
    figure; hold on
    plot(trainx(Ytrain(:, 1) == 1, 1), trainx(Ytrain(:, 1) == 1, 2), 'og', 'MarkerFaceColor', 'g', 'MarkerSize', 12, 'MarkerEdgeColor', 'k');
    plot(trainx(Ytrain(:, 2) == 1, 1), trainx(Ytrain(:, 2) == 1, 2), 'ob', 'MarkerFaceColor', 'b', 'MarkerSize', 12, 'MarkerEdgeColor', 'k');
    xlabel('x_1', 'FontSize', 20); ylabel('x_2', 'FontSize', 20);
    xmin = min(trainx(:,1)); xmax = max(trainx(:,1));
    ymin = min(trainx(:,2)); ymax = max(trainx(:,2));
    x = xmin : (xmax-xmin)/25 : xmax;
    for i = 1:length(x)
        y(1,i) = -(a(1) + a(2)*x(1,i))/a(3);
    end
    plot(x, y, 'k-', 'LineWidth', 3); 
    axis([xmin xmax ymin ymax]);
    hold off
else
   error('Cannot plot classification boundary for dimension greater than 2'); 
end

%Plot classification error vs iterations
figure;
plot(errorvec, '-og', 'LineWidth', 3);
xlabel('Iterations', 'FontSize', 20); ylabel('Classification error', 'FontSize', 20);


% Test phase
[Ntest, D1] = size(crossx);
if D ~= D1
    error('Dimensions of training and test data are not same');
end

crossx = [ones(Ntest,1) crossx];
Ypred = zeros(Ntest, 2);
inference = sum(a.*crossx, 2);
Ypred(inference > 0, 1) = 1;
Ypred(inference < 0, 2) = 1;

class_err = (sum(abs(Ycross(:,1) - Ypred(:,1)))*100) / Ntest;