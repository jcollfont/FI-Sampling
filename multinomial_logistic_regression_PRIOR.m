function [Fit, logLikelihood] = multinomial_logistic_regression_PRIOR(data, labels, precision, lambda, LAMBDA, Tolerance)
% Input:
%   lambda: regularization parameter
% Output:
%   Fit: trained Fit structure
%   llh: loglikelihood
if nargin < 6
    Tolerance = 1e-4; %set tolerance if not specified
end
if nargin < 5
    lambdaI=1e-8;
    LAMBDA=lambdaI*eye(size(data,1)+1);
end
if nargin < 4
    lambda = 1e-4; %set lambda if not specified
end
B=lambda*precision; %+LAMBDA; %%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = [data; ones(1,size(data,2))]; %append ones
[featureNum,dataNum] = size(data); %find size of data
classNum = max(labels); %number of classes
Max_Iterations = 100; %define max iterations
logLikelihood = -inf(1,Max_Iterations); %begin with -inf log likelihood
fc = featureNum*classNum;
index = (1:fc)'; %index list
dg = sub2ind([fc,fc],index,index);
T = sparse(labels,1:dataNum,1,classNum,dataNum,dataNum);
w = zeros(featureNum,classNum);
Ht = zeros(featureNum,classNum,featureNum,classNum);
for iteration = 2:Max_Iterations
    A = w'*data; %activations
    prior=-0.5*sum(diag(w'*B*w)); %2x2 matrix %%%%%%%%%%%%%%%%%%%%
    logY = bsxfun(@minus,A,logsumexp(A,1)); %4.104
    logLikelihood(iteration) = dot(T(:),logY(:))+prior;  % -0.5*lambda*(w'*B*w); %4.108
    if abs(logLikelihood(iteration)-logLikelihood(iteration-1)) < Tolerance; break; end
    Y = exp(logY);
    for i = 1:classNum
         for j = 1:classNum
            r = Y(i,:).*((i==j)-Y(j,:));
            Ht(:,i,:,j) = bsxfun(@times,data,r)*data'; %4.110
        end
    end
    G = data*(Y-T)'+B*w ; %Gradient calculated via 4.96 %%%%%%%%%%%%%%%%%%%%%%%%
    H = reshape(Ht,fc,fc);
    B2 =[B B; B B]; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H(dg) = H(dg)+B2(dg); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    w(:) = w(:)-H\G(:); %parameter vector updated via 4.92
end
logLikelihood = logLikelihood(2:iteration);
Fit.w = w; %output parameter vector









