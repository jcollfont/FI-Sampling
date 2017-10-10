clear;
%%% Fisher Rough Draft %%%
%Inputs: Image, Actual Labels, Labeled Pool Size, iterations (or
%confidence), # of classes
IterationNum=20;
c_total=4;
PoolNum=20; %Number of samples in initial labeled pool

%% Create Image and Labels
ClusterImageGenerator3 %Generate Image
image=image1; %load image
Knownlabels=imagelabels; %actually class labels

%Ignore Spatial Elements of Image (Convert to 1d instead of 2)
listsize=length(image1)^2; %length of image
NewLabels=zeros(listsize,1); %empty estimated labels

image=reshape(image,listsize,1); %make image list (of pixel values)
Knownlabels=reshape(Knownlabels,listsize,1); %make lables list

%% Initial Labeled Pool
[PoolClass,PoolIndex]=datasample(Knownlabels,PoolNum); %randomly samples w/o replacement
PoolIndex=PoolIndex';

for i=1:PoolNum %Add labels to current list
    NewLabels(PoolIndex(i))=PoolClass(i);
end

for c=1:c_total %create class lists
    class{c} = image(PoolIndex(find(PoolClass==c)));
end

%% Iterative Loop
%figID = figure;
for iteration=1:IterationNum
    %% MLE Parameter Estimates
    % [muhat1,sigmahat1] = normfit(class1data);
    % [muhat2,sigmahat2] = normfit(class2data); %uses sqrt of ubiased estimator for sigma
    for c=1:c_total
        n=length(class{c});
        muhat{c}=(1/n)*sum(class{c});
        sigmahat{c}=0;
        for i=1:n
            sigmahat{c}=sigmahat{c}+(class{c}(i)-muhat{c})^2;
        end
        sigmahat{c}=sqrt((1/(n-1))*sigmahat{c});
        if isnan(sigmahat{c})==1
            sigmahat{c}=1;
        end
    end
    
    %% Calculate the FI matrix
    UnlabeledIndices=find(NewLabels==0); %collect unlabeled indices
    UnlabeledLength=length(UnlabeledIndices);
    A=zeros(2,2,UnlabeledLength); %Create zeros for FI matrix
    Prior=1/c_total; %Assume priors set equal for all classes for now (true for first test image)
    for i=1:UnlabeledLength %walk through unlabeled points
        x=image(UnlabeledIndices(i)); %at unlabeled point x
        for c=1:c_total
            G(c)=normpdf(x,muhat{c},sigmahat{c});
        end
        for c=1:c_total
            P=Prior*G(c)/sum(G);
            dLmu=(x-muhat{c})/(sigmahat{c}^2); %derivative of log likelihood mean
            dLsigma=((x-muhat{c})^2)/(2*(sigmahat{c}^2)^2); %derivative of log likelihood sigma
            dL=vertcat(dLmu,dLsigma); %creat derivative of log likelihood matrix
            dLop=dL*dL'; %outer product
            S(:,:,c)=P*dLop;
        end
        A(:,:,i)=sum(S,3); %FI at x is outer product times posterior estimate summed over classes
    end

    %% Find maximum entry in A
    trA=zeros(UnlabeledLength,1); %Create zeros for trace of FI matrix
    for i=1:UnlabeledLength
        trA(i)=trace(A(:,:,i));
    end
    [max_value,new_index]=max(trA);
    
    %% Plot stuff and label new point
    xax = linspace(0,256,1000);
	  for c=1:c_total
        norm{c}=normpdf(xax,mu{c},sigma{c});
        normest{c}=normpdf(xax,muhat{c},sigmahat{c});
    end
	
    figure() %figID
    hold on
    for c=1:c_total
        plot(class{c}',ones(length(class{c}),1)*(max(norm{c})/2),'bx')
        plot(xax,normest{c},'--b') %Estimated Distribution
        plot(xax,norm{c},'-k') %Actual Distrubution
    end
    title(['Iteration # ' num2str(iteration)])
    xlabel('pixel value')
%    legend('Class 1 Data','Class 2 Data','Class 1 Estimated','Class 2 Estimated','Class 1 Actual','Class 2 Actual')

    NewLabels(new_index)=Knownlabels(UnlabeledIndices(new_index));
    for c=1:c_total
        if Knownlabels(UnlabeledIndices(new_index))==c
            class{c}(end+1)=image(UnlabeledIndices(new_index));
            plot(image(UnlabeledIndices(new_index)),(max(norm{c})/2),'or','MarkerSize',5,'LineWidth',2)
        end
    end
    
    yyaxis left;
    plot(image(UnlabeledIndices(:)),trA,'bx');
    
    hold off;
    pause(0.5);
    
end

