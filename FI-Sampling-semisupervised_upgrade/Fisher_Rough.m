%Fisher Rough Draft
clear;
ClusterImageGenerator2
%INPUT: initial labeled data set
image=image1; %loaded image
actuallabels=imagelabels; %actually classes

%to start we are going to ignore the spatial element and choose values
%based only on their pixel values
%So first we create a 1d empty index of all pixels that will be labeled:
listsize=length(image1)^2;
guessedlabels=zeros(listsize,1);

image=reshape(image,listsize,1);
actuallabels=reshape(actuallabels,listsize,1);

PoolNum=10; %number of labels in pool to begin
[pvalue,pindex]=datasample(actuallabels,PoolNum);
pindex=pindex';

k=1;
kk=1;
for i=1:PoolNum
    guessedlabels(pindex(i))=pvalue(i);
    if pvalue(i)==1
        class1data(k)=image(pindex(i));
        k=k+1;
    end
    if pvalue(i)==2
        class2data(kk)=image(pindex(i));
        kk=kk+1;
    end
end

%form a parameter estimate based on the current classes
[muhat1,sigmahat1] = normfit(class1data);
[muhat2,sigmahat2] = normfit(class2data);
%replace the above ^^^ with my own MLE calculation!!!
% 
% %% Optimization to get Xf
% UnlabeledIndices=find(guessedlabels==0);
% UnlabeledLength=listsize-(length(class1data)+length(class2data));
% 
% n=UnlabeledLength;
% cvx_begin %begin optimization
%     variable q(n)
%     OBJ = 0;
%     for i = 1:n
%         OBJ = OBJ + (q(i) * A(:,:,i));
%     end
%     OBJ = trace_inv(OBJ);
%     minimize(OBJ);
%     subject to
%         q >= 0; 
%         sum(q) == 1;   
% cvx_end %end optimization


UnlabeledIndices=find(guessedlabels==0);
%% Calculate the FI matrix
UnlabeledLength=listsize-(length(class1data)+length(class2data));
A=zeros(2,2,UnlabeledLength);
Prior=0.5;
for i=1:UnlabeledLength
    x=image(UnlabeledIndices(i));
    G1=normpdf(x,muhat1,sigmahat1);
    G2=normpdf(x,muhat2,sigmahat2);
    P1=Prior*G1/(G1+G2);
    P2=Prior*G2/(G1+G2);
    dLmu1=(x-muhat1)/sigmahat1;
    dLmu2=(x-muhat2)/sigmahat2;
    dLsigma1=((x-muhat1)^2)/(2*(sigmahat1)^2);
    dLsigma2=((x-muhat2)^2)/(2*(sigmahat2)^2);
    dL1=vertcat(dLmu1,dLsigma1);
    dL2=vertcat(dLmu2,dLsigma2);
    dL1op=dL1*dL1';
    dL2op=dL2*dL2';
    A(:,:,i)=P1*dL1op+P2*dL2op;
end

%% Optimization to get Marginal Distribution q(x)
n=UnlabeledLength;
cvx_begin %begin optimization
    variable q(n)
    OBJ = 0;
    for i = 1:n
        OBJ = OBJ + (q(i) * A(:,:,i));
    end
    OBJ = trace_inv(OBJ);
    minimize(OBJ);
    subject to
        q >= 0; 
        sum(q) == 1;   
cvx_end %end optimization

%% Sample from Marginal
qnum = 5;
queries = discretesample(q, qnum)
queriesUnique=unique(queries);
for i=1:length(queriesUnique)
    queriesUnique(i)=UnlabeledIndices(queriesUnique(i));
end
%Now queriesUnique is unique indices to label from the image

%% Label the queries
for i=1:length(queriesUnique)
    guessedlabels(queriesUnique(i))=actuallabels(queriesUnique(i));
    if guessedlabels(queriesUnique(i))==1
        class1data(k)=image(queriesUnique(i));
        k=k+1;
    end
    if guessedlabels(queriesUnique(i))==2
        class2data(k)=image(queriesUnique(i));
        kk=kk+1;
    end
end

%% 
%form a parameter estimate based on the current classes
[muhat1_new,sigmahat1_new] = normfit(class1data);
[muhat2_new,sigmahat2_new] = normfit(class2data);