clear;
%%% Fisher Rough Draft %%%

%what does the filter look like? *Todo
%catch up with semi-supervised learning (include in the current iteration
%of code) *Todo
%%% gives you a prior for the weight vectors. You can look at if this makes
%%% sense. Does it bias your weight vectors in a logical direction?


%Inputs: Image, Actual Labels, Labeled Pool Size, iterations (or
%confidence), # of classes
IterationNum=300;
c_total=2;
PoolNum=16; %Number of samples in initial labeled pool

im=rgb2gray(imread('1flower.jpeg')); %converts truecolor to intensity

%Need to make image square so that Degree matrix is square
if size(im,1)>size(im,2)
    ldiff=size(im,1)-size(im,2);
    %im=im(ldiff:end-1,:);
    im=im(ldiff:end-200,1:end-199);
elseif size(im,1)<size(im,2)
    ldiff=size(im,2)-size(im,1);
    %im=im(:,ldiff:end-1);
    im=im(1:end-199,ldiff:end-200);
else
    %do Nothing -- dimensions are squared
end

%im=imnoise(im,'gaussian',0,0.005);
im1=double(im) + 1; %convert to numbers between 1 and 256 (double)

[feature_map] = Image2FeatureMap(im1); % Create Feature Map

coordinates=zeros(size(im1,1)*size(im1,2),2);
c=1:size(im1,2);
c=c';
coordinates(:,2)=repmat(c,size(im1,1),1);
r=1;c=1;
for i=1:size(coordinates,1)
    if c<size(im1,2)
        coordinates(i,1)=r;
        c=c+1;
    else
        coordinates(i,1)=r;
        c=1;
        r=r+1;
    end
end
flatImage=reshape(im1,(size(im,1)*size(im,2)),1); %make image list (of pixel values)
flatFeature_map=reshape(feature_map,(size(feature_map,1)^2),9);


% %% Calculate Graph Laplacian
% sigma=1;
% AdjacMat=zeros(size(flatImage,1),size(flatImage,1)); %Change to feature map!
% for i=1:size(flatImage,1)
%     for j=1:size(flatImage,1)
%         if
%         AdjacMat(i,j)=exp((-1/(2*sigma^2))*norm(flatImage(i)-flatImage(j)));
%     end
% end
 
% %% Initial Labeled Pool
% [PoolIndex]=datasample(1:(length(flatImage)),PoolNum); %randomly samples w/o replacement
% PoolIndex=PoolIndex';
% 
% im2=im2double(im);
% PoolClass=zeros(1,length(PoolIndex));
% 
% prompt = {['Enter displayed points class (1 to ',num2str(c_total),'):']};
% dlg_title = 'Label Class';
% num_lines = 1;
% defaultans = {'0','hsv'};
% p=zeros(1,2);
% for i=1:length(PoolIndex)
%     handles.H=figure();
%     imshow(im2)
%     hold on
%     p=coordinates(PoolIndex(i),:);
%     plot(p(2),p(1),'go','LineWidth',3)
%     answer=inputdlg(prompt,dlg_title,num_lines,defaultans);
%     PoolClass(i)=str2num(answer{1});
%     close(handles.H)
% end
% 
% NewLabels=zeros(length(flatImage),1); %empty estimated labels
% for i=1:PoolNum %Add labels to current list
%     NewLabels(PoolIndex(i))=PoolClass(i);
% end
% 
% fdepth=size(feature_map,3);
% for c=1:c_total %create class lists
%     PI=PoolIndex(find(PoolClass==c));
%     for i=1:length(PI)
%         for ii=1:fdepth
%             class{c}(i,ii) = feature_map(coordinates(PI(i),1),coordinates(PI(i),2),ii);
%         end
%     end
% end
% 
% save('1flowerTrained.mat')
% 
% %% Iterative Loop
% for iteration=1:IterationNum 
%     %% Fit logistic Regression to Current Pool
%     [labels,data]=class_breakdown(class,c_total);
%     [Fit, llh] = multinomial_logistic_regression(data', labels);
%     %% Calculate the FI matrix
%     UnlabeledIndices=find(NewLabels==0); %collect unlabeled indices
%     EstimatedUnlabeleds=zeros(length(UnlabeledIndices),1);
%     A=zeros(size(feature_map,3)+1,size(feature_map,3)+1,length(UnlabeledIndices)); %Create zeros for FI matrix
%     x=zeros(1,length(UnlabeledIndices));
%     for i=1:length(UnlabeledIndices) %walk through unlabeled points
%         x=feature_map(coordinates(UnlabeledIndices(i),1),coordinates(UnlabeledIndices(i),2),:); %at unlabeled point x
%         x=squeeze(x);
%         [y, p] = multinomial_logistic_prediction(Fit, x);
%         EstimatedUnlabeleds(i)=y;
%         for c=1:c_total
%             P=p(c);
%             g=(1-P)*x';
%             dLop=g*g'; %outer product
%             S(:,:,c)=P*dLop;
%         end
%         A(:,:,i)=sum(S,3); %FI at x is outer product times posterior estimate summed over classes
%     end
% 
%     %Plot Estimated Unlabeleds
%     Estimates=zeros(length(NewLabels),1);
%     for i=1:length(EstimatedUnlabeleds)
%         Estimates(UnlabeledIndices(i))=EstimatedUnlabeleds(i);
%     end
%     v=find(NewLabels~=0);
%     for i=1:length(v)
%         Estimates(v(i))=NewLabels(v(i));
%     end
%     
%     %plot heatmap
%     Estimate_Matrix=zeros(size(im,1),size(im,2));
%     for i=1:length(coordinates)
%         Estimate_Matrix(coordinates(i,1),coordinates(i,2))=Estimates(i);
%     end
%     figure()
%     colormap('gray')
%     imagesc(Estimate_Matrix)
%     colorbar
%     
%     %% Find maximum entry in A
%     trA=zeros(length(UnlabeledIndices),1); %Create zeros for trace of FI matrix
%     for i=1:length(UnlabeledIndices)
%         trA(i)=trace(A(:,:,i));
%     end
%     [max_value,new_index]=max(trA);
% 
%     trA_new=zeros(length(NewLabels),1);
%     for i=1:length(UnlabeledIndices)
%         trA_new(UnlabeledIndices(i))=trA(i);
%     end
%     %plot heatmap
%     A_heat=zeros(size(im,1),size(im,2));
%     for i=1:length(coordinates)
%         A_heat(coordinates(i,1),coordinates(i,2))=trA_new(i);
%     end
%     figure()
%     colormap('hot')
%     imagesc(A_heat)
%     colorbar
%    
%     %% Plot stuff and label new point	
%     prompt = {['Enter displayed points class (1 to ',num2str(c_total),'):']};
%     dlg_title = 'Label Class';
%     num_lines = 1;
%     defaultans = {'0','hsv'};
%     handles.H=figure();
%     imshow(im2)
%     hold on
%     p=coordinates(new_index,:);
%     plot(p(2),p(1),'go','LineWidth',3)
%     answer=inputdlg(prompt,dlg_title,num_lines,defaultans);
%     num=str2num(answer{1});
%     NewLabels(UnlabeledIndices(new_index))=num;
%     feature=zeros(1,fdepth);
%     for i=1:fdepth
%         feature(i)=feature_map(coordinates(UnlabeledIndices(new_index),1),coordinates(UnlabeledIndices(new_index),2),i);
%     end
%     class{num}=[class{num};feature];
%     close(handles.H)
%     pause(0.1);
% end
% 
