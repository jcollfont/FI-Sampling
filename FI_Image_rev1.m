%% Fisher Rough Draft
clear;

%% Inputs: Image, Actual Labels, Labeled Pool Size, iterations (or
%confidence), # of classes
IterationNum=10;
c_total=2;
PoolNum=50; %Number of samples in initial labeled pool

im=rgb2gray(imread('BSDS300/images/train/135069.jpg')); %converts truecolor to intensity
% figure()
% imshow(im);

seg = readSeg('BSDS300/human/color/1105/135069.seg');
[rs,cs]=size(seg);
for i=1:rs %eliminate single pixel segmentations from original image
    for ii=1:cs
        if seg(i,ii)==6
            seg(i,ii)=0;
        end
        if seg(i,ii)==5
            seg(i,ii)=0;
        end
    end
end

% %Use only two classes of this image (birds vs background)
% seg2class=zeros(rs,cs);
% for i=1:rs
%     for ii=1:cs
%         if seg(i,ii)==1
%             seg2class(i,ii)=1; %background is 1
%         end
%     end
% end
% for i=1:rs
%     for ii=1:cs
%         if seg2class(i,ii)==0
%             seg2class(i,ii)=2; %birds are 2
%         end
%     end
% end

%Use only two classes of this image (birds vs background)
seg2class=zeros(rs,cs);
for i=1:rs
    for ii=1:cs
        if seg(i,ii)==1
            seg2class(i,ii)=1; %background is 1
        end
        if seg(i,ii)==2
            seg2class(i,ii)=2; %birds are 2
        end
        if seg(i,ii)==3
            seg2class(i,ii)=1; %tails are 2
        end
        if seg(i,ii)==4
            seg2class(i,ii)=2; %birds are 2
        end
    end
end

% %Use three classes of this image (birds vs tail vs background)
% seg2class=zeros(rs,cs);
% for i=1:rs
%     for ii=1:cs
%         if seg(i,ii)==1
%             seg2class(i,ii)=1; %background is 1
%         end
%     end
% end
% for i=1:rs
%     for ii=1:cs
%         if seg(i,ii)==3
%             seg2class(i,ii)=3; %tail is 3
%         end
%     end
% end
% 
% for i=1:rs
%     for ii=1:cs
%         if seg2class(i,ii)==0
%             seg2class(i,ii)=2; %birds are 2
%         end
%     end
% end

% figure()
% imshow(mat2gray(seg2class))

%% Square off image (im, seg, seg2class)
if size(im,1)>size(im,2)
    ldiff=size(im,1)-size(im,2);
    im=im(floor(ldiff/2):end-floor(ldiff/2),:);
    seg=seg(floor(ldiff/2):end-floor(ldiff/2),:);
    seg2class=seg2class(floor(ldiff/2):end-floor(ldiff/2),:);
elseif size(im,1)<size(im,2)
    ldiff=size(im,2)-size(im,1);
    im=im(:,floor(ldiff/2):end-floor(ldiff/2));
    seg=seg(:,floor(ldiff/2):end-floor(ldiff/2));
    seg2class=seg2class(:,floor(ldiff/2):end-floor(ldiff/2));
else
    %do Nothing -- dimensions are squared
end

scalefactor=0.3;
im=imresize(im,scalefactor);
seg=imresize(seg,scalefactor,'nearest');
seg2class=imresize(seg2class,scalefactor,'nearest');

%%imshow(im)
%%im=imnoise(im,'gaussian',0,0.005);
im1=double(im) + 1; %convert to numbers between 1 and 256 (double)

%% Create Feature Map(s)
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
flatClass=reshape(seg2class,(size(seg2class,1)*size(seg2class,2)),1);
flatFeature_map=reshape(feature_map,(size(feature_map,1)^2),9);
flatFeature_map2=mean(flatFeature_map,2);

%% Calculate Graph Laplacian
% sigma=10;
% tic
% AdjacMat=zeros(size(flatFeature_map2,1),size(flatFeature_map2,1)); %Change to feature map!
% toc
% for i=1:size(flatImage,1)
%     for j=1:size(flatImage,1)
%         a=exp((-1/(2*sigma^2))*(norm(flatFeature_map2(i,:)-flatFeature_map2(j,:)))^2);
%         if a<0.0001
%             AdjacMat(i,j)=0;
%         else
%             AdjacMat(i,j)=a;
%         end
%     end
% end
% toc
% 
% del=diag(sum(AdjacMat,1))-AdjacMat;
% toc
% 
% save('bird_del_p3_11_2_17.mat','del','-v7.3');
% toc

load('bird_del_p3_11_2_17.mat');

%% Initial Labeled Pool
a=1;
while a==1
    [PoolIndex]=datasample(1:(length(flatImage)),PoolNum); %randomly samples w/o replacement
    PoolIndex=PoolIndex';

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

    % NewLabels=zeros(length(flatImage),1); %empty estimated labels
    % for i=1:PoolNum %Add labels to current list
    %     NewLabels(PoolIndex(i))=PoolClass(i);
    % end

    % fdepth=size(feature_map,3);
    % for c=1:c_total %create class lists
    %     PI=PoolIndex(find(PoolClass==c));
    %     for i=1:length(PI)
    %         for ii=1:fdepth
    %             class{c}(i,ii) = feature_map(coordinates(PI(i),1),coordinates(PI(i),2),ii);
    %         end
    %     end
    % end

    NewLabels=zeros(length(flatImage),1); %empty estimated labels
    for i=1:PoolNum %Add labels to current list
        NewLabels(PoolIndex(i))=flatClass(PoolIndex(i));
    end

    fdepth=size(feature_map,3);
    for c=1:c_total %create class lists
        PI=find(NewLabels==c);
        class{c} = flatFeature_map(PI,:);
    end
    
    if max(NewLabels)-min(NewLabels)==c_total
        a=0;
    else
        a=1;
        %didn't sample all classes
        disp('sample failure, resampling initial pool')
    end
end

%% Loop through prior weightings
div1=50;
div2=10;
RecordedAccuracies=zeros(div1*div2,5);
q=1;
for lambda=10.^linspace(-8,1,div1)
    for lambdaI=10.^linspace(-8,1,div2)
        flatFeature_map_ones = [flatFeature_map ones(size(flatFeature_map,1),1)]; %append ones
        precision=flatFeature_map_ones'*del*flatFeature_map_ones;
        LAMBDA=lambdaI*eye(size(flatFeature_map_ones,2));

        %% Iterative Loop
        for iteration=1:IterationNum 
            %% Fit logistic Regression to Current Pool
            [labels,data]=class_breakdown(class,c_total);
            [Fit, llh] = multinomial_logistic_regression_PRIOR(data', labels', precision, lambda, LAMBDA);
            %[Fit, llh] = multinomial_logistic_regression(data', labels');
            %% Calculate the FI matrix
            UnlabeledIndices=find(NewLabels==0); %collect unlabeled indices
            EstimatedUnlabeleds=zeros(length(UnlabeledIndices),1);
            A=zeros(size(feature_map,3)+1,size(feature_map,3)+1,length(UnlabeledIndices)); %Create zeros for FI matrix
            x=zeros(1,length(UnlabeledIndices));
            for i=1:length(UnlabeledIndices) %walk through unlabeled points
                x=feature_map(coordinates(UnlabeledIndices(i),1),coordinates(UnlabeledIndices(i),2),:); %at unlabeled point x
                x=squeeze(x);
                [y, p] = multinomial_logistic_prediction(Fit, x);
                EstimatedUnlabeleds(i)=y;
                for c=1:c_total
                    P=p(c);
                    g=(1-P)*x';
                    dLop=g*g'; %outer product
                    S(:,:,c)=P*dLop;
                end
                A(:,:,i)=sum(S,3); %FI at x is outer product times posterior estimate summed over classes
            end

            %Estimated Unlabeleds
            Estimates=zeros(length(NewLabels),1);
            for i=1:length(EstimatedUnlabeleds)
                Estimates(UnlabeledIndices(i))=EstimatedUnlabeleds(i);
            end
            v=find(NewLabels~=0);
            for i=1:length(v)
                Estimates(v(i))=NewLabels(v(i));
            end

            %plot heatmap
            Estimate_Matrix=zeros(size(im,1),size(im,2));
            for i=1:length(coordinates)
                Estimate_Matrix(coordinates(i,1),coordinates(i,2))=Estimates(i);
            end    
            if iteration == IterationNum
%             if mod(iteration,20)==0 %iteration==IterationNum
                figure()
                colormap('gray')
                imagesc(Estimate_Matrix)
                colorbar
                title(['Final Estimate for lambda =',num2str(lambda)])
%             end
            end

            %% Find maximum entry in A
            trA=zeros(length(UnlabeledIndices),1); %Create zeros for trace of FI matrix
            for i=1:length(UnlabeledIndices)
                trA(i)=trace(A(:,:,i));
            end
            [max_value,new_index]=max(trA);

            trA_new=zeros(length(NewLabels),1);
            for i=1:length(UnlabeledIndices)
                trA_new(UnlabeledIndices(i))=trA(i);
            end

%             %plot heatmap
%             A_heat=zeros(size(im,1),size(im,2));
%             for i=1:length(coordinates)
%                 A_heat(coordinates(i,1),coordinates(i,2))=trA_new(i);
%             end
%             if mod(iteration,20)==0 %iteration==IterationNum
%                 figure()
%                 colormap('hot')
%                 imagesc(A_heat)
%                 colorbar
%             end

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
        %     
        %     
        %     NewLabels(UnlabeledIndices(new_index))=num;
        %     feature=zeros(1,fdepth);
        %     for i=1:fdepth
        %         feature(i)=feature_map(coordinates(UnlabeledIndices(new_index),1),coordinates(UnlabeledIndices(new_index),2),i);
        %     end
        %     class{num}=[class{num};feature];
        %     close(handles.H)
        %     pause(0.1);
        %     
        %     NewLabels(UnlabeledIndices(new_index))=flatClass(UnlabeledIndices(new_index));
        %     feature=zeros(1,fdepth);
        %     for i=1:fdepth
        %         feature(i)=feature_map(coordinates(UnlabeledIndices(new_index),1),coordinates(UnlabeledIndices(new_index),2),i);
        %     end
        %     class{num}=[class{num};feature];
        %     close(handles.H)
        %     pause(0.5);

            NewLabels(UnlabeledIndices(new_index))=flatClass(UnlabeledIndices(new_index));
            class{flatClass(UnlabeledIndices(new_index))}=[class{flatClass(UnlabeledIndices(new_index))};flatFeature_map(new_index,:)];
            pause(0.5);

        end

        %% Accuracy Measurement

        %percent accuracy
        flatEstimate=reshape(Estimate_Matrix,(size(Estimate_Matrix,1)*size(Estimate_Matrix,2)),1); %flat estimate
        
        %separate accuracies for two classes
        acc1=0;acc2=0;
        for v = 1:length(flatEstimate)
            if flatClass(v) == 1
                if flatClass(v)==flatEstimate(v)
                    acc1=acc1+1;
                end
            end
            if flatClass(v) == 2
                if flatClass(v)==flatEstimate(v)
                    acc2=acc2+1;
                end
            end
        end
        c1sum=sum(flatClass==1);
        c2sum=sum(flatClass==2);
        
        accuracyList=flatEstimate==flatClass;
        accuracyTotal=(sum(accuracyList)/length(accuracyList))*100;
        
        accuracy1=(acc1/c1sum)*100;
        accuracy2=(acc2/c2sum)*100;
        
        
        %disp(['Accuracy for ', num2str(lambda), ' is ',num2str(accuracy)]);
        RecordedAccuracies(q,1)=lambda;
        RecordedAccuracies(q,2)=lambdaI;
        RecordedAccuracies(q,3)=accuracyTotal;
        RecordedAccuracies(q,4)=accuracy1;
        RecordedAccuracies(q,5)=accuracy2;
        q=q+1;
    end
end

save('11_3_17_test1.mat');

