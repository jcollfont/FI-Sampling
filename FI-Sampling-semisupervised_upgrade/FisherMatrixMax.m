clear;
%%% Fisher Rough Draft %%%
%Inputs: Image, Actual Labels, Labeled Pool Size, iterations (or
%confidence), # of classes
IterationNum=40;
c_total=4;
PoolNum=50; %Number of samples in initial labeled pool

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
    
    %% Fit logistic Regression to Current Pool
    [labels,data]=class_breakdown(class,c_total);
%     L=zeros(1,c_total);
%     for c=1:c_total
%         L(c)=length(class{c});
%     end
%     for c=1:c_total
%         l{c}=ones(L(c),1)*c;
%     end
%     labels=l{1};
%     data=class{1};
%     for c=2:c_total
%         labels=vertcat(labels,l{c});
%         data=vertcat(data,class{c});
%     end
    
%     sp=categorical(labels);
%     B=mnrfit(data,sp);

    [Fit, llh, G] = multinomial_logistic_regression(data', labels');
    %% Calculate the FI matrix
    UnlabeledIndices=find(NewLabels==0); %collect unlabeled indices
    UnlabeledLength=length(UnlabeledIndices);
    A=zeros(2,2,UnlabeledLength); %Create zeros for FI matrix
    x=zeros(1,UnlabeledLength);
    for i=1:UnlabeledLength %walk through unlabeled points
        x=image(UnlabeledIndices(i)); %at unlabeled point x
        [y, p] = multinomial_logistic_prediction(Fit, x);
        for c=1:c_total
            P=p(c);
            %What if you evaluate the logistic regression for each new
            %point???
            class_temp=class;
            class{c}(end+1)=x;
            [labels_temp,data_temp]=class_breakdown(class_temp,c_total);
            [Fit_temp, llh_temp, G_temp] = multinomial_logistic_regression(data_temp', labels_temp');
            g=G_temp(:,c);
            dLop=g*g'; %outer product
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
    
%     yyaxis left;
%     plot(image(UnlabeledIndices(:)),trA,'bx');
%     
%     hold off;
    pause(0.1);
    
end

