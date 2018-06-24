load data_lin_sep

x1 = trainx(Ytrain(:, 1)==1, :);
x2 = trainx(Ytrain(:, 2)==1, :);
[N1, d] = size(x1);
[N2, ~] = size(x2);
N = N1 + N2;

figure; hold on
plot(x1(:,1), x1(:,2), 'og', 'MarkerFaceColor', 'g', 'MarkerSize', 12, 'MarkerEdgeColor', 'k');
plot(x2(:,1), x2(:,2), 'ob', 'MarkerFaceColor', 'b', 'MarkerSize', 12, 'MarkerEdgeColor', 'k');
xlabel('x_1', 'FontSize', 20); ylabel('x_2', 'FontSize', 20);


%Weights and parameters
eta = 0.1;              %learning rate
a = ones((d+1),1);      %initial augmented weight vector
b = ones(N, 1);      %target vector

%update weight vector till convergence
x1 = [ones(N1,1) x1];
x2 = - [ones(N2,1) x2];
Y = [x1; x2];

% %% One-shot solution (not recommended for large datasets or if Y is singular)
% a = inv(Y'*Y)*Y'*b;


%% Gradient descent method
costvec = [];
errorvec = [];
iter =  1;
maxepoch = 20;

while iter <= maxepoch
    
    a = a + (eta/iter)*Y'*(b-Y*a);
    costvec = [costvec sum((Y*a - b).^2)];
    errorvec = [errorvec sum((Y*a)<0)*100/N];
    iter = iter + 1;
    
end

xmin = min(trainx(:,1)); xmax = max(trainx(:,1));  
x = xmin : (xmax - xmin)/25 : xmax;
for i = 1:length(x)
   y(1, i) = -(a(1) + a(2)*x(1, i))/a(3);
end
plot(x,y,'k-','LineWidth',3); hold off

%Plot classification error vs iterations
figure;
plot(1:maxepoch, errorvec, '-og', 'LineWidth', 3);
xlabel('Iterations','FontSize',30); ylabel('Classification error (\%)','FontSize',30);