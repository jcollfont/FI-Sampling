function [label, probability] = multinomial_logistic_prediction(Fit, datain)
w = Fit.w; %pull parameter vector
datain = [datain; ones(1,size(datain,2))]; %append ones to data
a=w'*datain; %activations
probability = softmax(a);
[~, label] = max(probability,[],1);