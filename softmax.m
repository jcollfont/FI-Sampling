function [out] = softmax(x)
% Softmax function
dim = find(size(x)~=1,1);
if isempty(dim)
    dim = 1;
end
l = logsumexp(x,dim);
out = exp(x-l);