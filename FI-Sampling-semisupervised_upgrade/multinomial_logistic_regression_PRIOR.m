function [Fit, logLikelihood] = multinomial_logistic_regression(data, labels, precision, lambdak, LAMBDA, lambda, Tolerance)
% Input:
%   lambda: regularization parameter
% Output:
%   Fit: trained Fit structure
%   llh: loglikelihood
if nargin < 7
    Tolerance = 1e-4; %set tolerance if not specified
end
if nargin < 6
    lambda = 1e-4; %set lambda if not specified
end
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
    logY = bsxfun(@minus,A,logsumexp(A,1)); %4.104
    %logY = logY - (lambdak/2)*(w'*(precision+LAMBDA)*w);
    logLikelihood(iteration) = dot(T(:),logY(:))-0.5*lambda*dot(w(:),w(:)); %4.108
    if abs(logLikelihood(iteration)-logLikelihood(iteration-1)) < Tolerance; break; end
    Y = exp(logY);
    for i = 1:classNum
         for j = 1:classNum
            r = Y(i,:).*((i==j)-Y(j,:));
            Ht(:,i,:,j) = bsxfun(@times,data,r)*data'; %4.110
        end
    end
    G = data*(Y-T)'+lambda*w; %Gradient calculated via 4.96
    H = reshape(Ht,fc,fc);
    H(dg) = H(dg)+lambda;
    w(:) = w(:)-H\G(:); %parameter vector updated via 4.92
end
logLikelihood = logLikelihood(2:iteration);
Fit.w = w; %output parameter vector









