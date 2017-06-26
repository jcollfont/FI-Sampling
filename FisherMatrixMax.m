clear;
%%% Fisher Rough Draft %%%
%Inputs: Image, Actual Labels, Labeled Pool Size, iterations (or
%confidence), # of classes
iterationNum=17;

%% Create Image and Labels
ClusterImageGenerator3 %Generate Image
image=image1; %load image
actuallabels=imagelabels; %actually class labels

%Ignore Spatial Elements of Image (Convert to 1d instead of 2)
listsize=length(image1)^2; %length of image
guessedlabels=zeros(listsize,1); %empty estimated labels

image=reshape(image,listsize,1); %make image list (of pixel values)
actuallabels=reshape(actuallabels,listsize,1); %make lables list

%% Initial Labeled Pool
PoolNum=10; %Number of samples in pool
[pvalue,pindex]=datasample(actuallabels,PoolNum); %randomly samples w/o replacement
pindex=pindex';

k=1;
kk=1;
for i=1:PoolNum
    guessedlabels(pindex(i))=pvalue(i); %save estimated labels with actual labels
    if pvalue(i)==1
        class1data(k)=image(pindex(i)); %Create list of class1 pixel values
        k=k+1;
    end
    if pvalue(i)==2
        class2data(kk)=image(pindex(i)); %Create list of class2 pixel values
        kk=kk+1;
    end
end

for iteration=1:iterationNum
    %% MLE Parameter Estimates
    % [muhat1,sigmahat1] = normfit(class1data);
    % [muhat2,sigmahat2] = normfit(class2data); %uses sqrt of ubiased estimator for sigma

    N1=length(class1data);
    muhat1MLE=(1/N1)*sum(class1data); %MLE of mean
    sigmahat1MLE=0;
    for i=1:N1
        sigmahat1MLE=sigmahat1MLE+(class1data(i)-muhat1MLE)^2;
    end
    sigmahat1MLE=sqrt((1/(N1-1))*sigmahat1MLE); %unbiased estimator of sigma

    N2=length(class2data);
    muhat2MLE=(1/N2)*sum(class2data); %MLE of mean
    sigmahat2MLE=0;
    for i=1:N2
        sigmahat2MLE=sigmahat2MLE+(class2data(i)-muhat2MLE)^2;
    end
    sigmahat2MLE=sqrt((1/(N2-1))*sigmahat2MLE); %unbiased estimator of sigma
    
    %% Calculate the FI matrix
    UnlabeledIndices=find(guessedlabels==0); %collect unlabeled indices
    UnlabeledLength=length(UnlabeledIndices);
    A=zeros(2,2,UnlabeledLength); %Create zeros for FI matrix
    Prior=0.5; %Assume priors set at 50/50 for now (true for first test image)
    for i=1:UnlabeledLength %walk through unlabeled points
        x=image(UnlabeledIndices(i)); %at unlabeled point x
        G1=normpdf(x,muhat1MLE,sigmahat1MLE); %class 1 gaussian value at x
        G2=normpdf(x,muhat2MLE,sigmahat2MLE);
        P1=Prior*G1/(G1+G2); %class 1 posterior estimate
        P2=Prior*G2/(G1+G2); %class 2 ""
        dLmu1=(x-muhat1MLE)/(sigmahat1MLE^2); %derivative of log likelihood mean
        dLmu2=(x-muhat2MLE)/(sigmahat2MLE^2); %class 2 ""
        dLsigma1=((x-muhat1MLE)^2)/(2*(sigmahat1MLE^2)^2); %derivative of log likelihood sigma
        dLsigma2=((x-muhat2MLE)^2)/(2*(sigmahat2MLE^2)^2); %class 2 ""
        dL1=vertcat(dLmu1,dLsigma1); %creat derivative of log likelihood matrix
        dL2=vertcat(dLmu2,dLsigma2); %class 2 ""
        dL1op=dL1*dL1'; %class 1 outer product
        dL2op=dL2*dL2'; %class 2 outer product
        A(:,:,i)=P1*dL1op+P2*dL2op; %FI at x is outer product times posterior estimate summed over classes
    end

    %% Find maximum entry in A
    trA=zeros(UnlabeledLength,1); %Create zeros for trace of FI matrix
    for i=1:UnlabeledLength
        trA(i)=trace(A(:,:,i));
    end
    [max_value,new_index]=max(trA);
    
    %% Plot stuff and label new point
    xax = [0:.01:256];
    norm1 = normpdf(xax,muhat1MLE,sigmahat1MLE);
    norm2 = normpdf(xax,muhat2MLE,sigmahat2MLE);
    figure()
    plot(class1data',ones(length(class1data))*0.1,'bx')
    hold on
    title(['Iteration # ' num2str(iteration)])
    xlabel('pixel value')
    plot(class2data',ones(length(class2data))*0.1,'rx')
    plot(xax,norm1,'--b')
    plot(xax,norm2,'--r')

    guessedlabels(new_index)=actuallabels(new_index);
    if actuallabels(new_index)==1
        class1data(k)=image(new_index);
        plot(image(new_index),0.1,'bo','MarkerSize',10)
        k=k+1;
    end
    if actuallabels(new_index)==2
        class2data(kk)=image(new_index);
        plot(image(new_index),0.1,'ro','MarkerSize',10)
        kk=kk+1;
    end
    
    hold off
    
end

