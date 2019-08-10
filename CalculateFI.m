function [A, EstimatedUnlabeleds] = CalculateFI(UnlabeledIndices, feature_map, flatFeature_map, Fit, c_total)
EstimatedUnlabeleds=zeros(length(UnlabeledIndices),1);
A=zeros(size(feature_map,3)+1,size(feature_map,3)+1,length(UnlabeledIndices)); %Create zeros for FI matrix
%x=zeros(1,length(UnlabeledIndices));
for i=1:length(UnlabeledIndices) %walk through unlabeled points
    x=flatFeature_map(UnlabeledIndices(i),:); %at unlabeled point x
    [y, p] = multinomial_logistic_prediction(Fit, x');
    EstimatedUnlabeleds(i)=y;
    S=zeros(1,c_total);
    for c=1:c_total
        P=p(c);
        g=(1-P)*x;
        dLop=g*g'; %outer product
        S(c)=P*dLop;
    end
    A(:,:,i)=sum(S); %FI at x is outer product times posterior estimate summed over classes
end









