clear;
%%% Fisher Rough Draft %%%
%Inputs: Image, Actual Labels, Labeled Pool Size, iterations (or
%confidence), # of classes
IterationNum=300;
c_total=2;
PoolNum=20; %Number of samples in initial labeled pool

%TRY WITH REAL LABELED IMAGE (WILL HAVE TO FIGURE OUT INDEXING ISSUES)
%2. query user to label random pixels of pool size
%3. After all pool is labeled run through algorithm
%4. Query user to label pixel/group with each iteration
%5. Output current image guess with each iteration

%% Create Image and Labels
% ClusterImageGenerator2 %Generate Image
% image=image1; %load image
% Knownlabels=imagelabels; %actually class labels

% %Ignore Spatial Elements of Image (Convert to 1d instead of 2)
% listsize=length(image1)^2; %length of image
% NewLabels=zeros(listsize,1); %empty estimated labels
% 
% image=reshape(image,listsize,1); %make image list (of pixel values)
% Knownlabels=reshape(Knownlabels,listsize,1); %make lables list

im=rgb2gray(imread('flower_test.jpg'));
image=double(im) + 1;

%% Create Feature Map
[features] = Image2Features(image);

%% Initial Labeled Pool
[PoolClass,PoolIndex]=datasample(Knownlabels,PoolNum); %randomly samples w/o replacement
PoolIndex=PoolIndex';

for i=1:PoolNum %Add labels to current list
    NewLabels(PoolIndex(i))=PoolClass(i);
end

for c=1:c_total %create class lists
    class{c} = features(PoolIndex(find(PoolClass==c)),:);
end

f=1;
%% Iterative Loop
%figID = figure;
for iteration=1:IterationNum
    %% MLE Parameter Estimates
    for c=1:c_total
        n=size(class{c},1);
        muhat{c}=(1/n)*sum(class{c}(:,5)); %class{c}(:,5)
        sigmahat{c}=0;
        for i=1:n
            sigmahat{c}=sigmahat{c}+((class{c}(i,5))-muhat{c})^2; %5
        end
        sigmahat{c}=sqrt((1/(n-1))*sigmahat{c});
        if isnan(sigmahat{c})==1
            sigmahat{c}=1;
        end
    end
    
    %% Fit logistic Regression to Current Pool
    [labels,data]=class_breakdown(class,c_total);
    [Fit, llh] = multinomial_logistic_regression(data', labels);
    %% Calculate the FI matrix
    UnlabeledIndices=find(NewLabels==0); %collect unlabeled indices
    UnlabeledLength=length(UnlabeledIndices);
    A=zeros(size(features,2)+1,size(features,2)+1,UnlabeledLength); %Create zeros for FI matrix
    x=zeros(1,UnlabeledLength);
    for i=1:UnlabeledLength %walk through unlabeled points
        x=features(UnlabeledIndices(i),:); %at unlabeled point x
        [y, p] = multinomial_logistic_prediction(Fit, x');
        for c=1:c_total
            P=p(c);
            g=(1-P)*x;
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
	
    figure(f) %figID
    hold on
    for c=1:c_total
        plot(class{c}(:,1)',ones(size(class{c},5),1)*(max(norm{c})/2),'bx') %5
        plot(xax,normest{c},'--b') %Estimated Distribution
        plot(xax,norm{c},'-k') %Actual Distrubution
    end
    title(['Iteration # ' num2str(iteration)])
    xlabel('pixel value')
%    legend('Class 1 Data','Class 2 Data','Class 1 Estimated','Class 2 Estimated','Class 1 Actual','Class 2 Actual')

    NewLabels(new_index)=Knownlabels(UnlabeledIndices(new_index));
    for c=1:c_total
        if Knownlabels(UnlabeledIndices(new_index))==c
            class{c}=[class{c};features(UnlabeledIndices(new_index),:)];
            plot(features(UnlabeledIndices(new_index),5),(max(norm{c})/2),'or','MarkerSize',5,'LineWidth',2) %(new_index),5)
            break;
        end
    end
    
    yyaxis right;
    plot(image(UnlabeledIndices(:)),trA,'bx');
    hold off;
    if iteration~=1 & iteration~=IterationNum
        close(figure(f))
    end
    pause(0.1);
    f=f+1;
end

