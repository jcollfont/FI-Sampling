%% Inputs: Image, Actual Labels, Labeled Pool Size, iterations (or
%confidence), # of classes
clear;
IterationNum=2000;
c_total=2;
PoolIterations=6; %always add plus 1 for random drawing
PoolNum=500; %Number of samples in initial labeled pool
lambdaspan=10; %10^-1; %3.7365e+12; %10.^linspace(-6,1,8);
lambdaIspan=0;
KernelSize=9;

im=rgb2gray(imread('135069.jpg')); %converts truecolor to intensity

seg = readSeg('135069.seg');
[rs,cs]=size(seg);
for i=1:rs %eliminate single pixel segmentations from original image
    for ii=1:cs
        if seg(i,ii)==6
            seg(i,ii)=1;
        end
        if seg(i,ii)==5
            seg(i,ii)=1;
        end
    end
end

%% Use only two classes of this image (birds vs background)
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

%% Square off image (im, seg, seg2class)
if size(im,1)>size(im,2)
    ldiff=size(im,1)-size(im,2);
    im=im(floor(ldiff/2):end-floor(ldiff/2)-1,:);
    seg=seg(floor(ldiff/2):end-floor(ldiff/2)-1,:);
    seg2class=seg2class(floor(ldiff/2):end-floor(ldiff/2)-1,:);
elseif size(im,1)<size(im,2)
    ldiff=size(im,2)-size(im,1);
    im=im(:,floor(ldiff/2):end-floor(ldiff/2)-1);
    seg=seg(:,floor(ldiff/2):end-floor(ldiff/2)-1);
    seg2class=seg2class(:,floor(ldiff/2):end-floor(ldiff/2)-1);
else
    %do Nothing -- dimensions are squared
end

%% Shrink Image further (assumed square)
PixRm=100;
im=im(floor(PixRm/2):end-floor(PixRm/2)-1,floor(PixRm/2):end-floor(PixRm/2)-1);
seg=seg(floor(PixRm/2):end-floor(PixRm/2)-1,floor(PixRm/2):end-floor(PixRm/2)-1);
seg2class=seg2class(floor(PixRm/2):end-floor(PixRm/2)-1,floor(PixRm/2):end-floor(PixRm/2)-1);

scalefactor=0.6;
im=imresize(im,scalefactor);
seg=imresize(seg,scalefactor,'nearest');
seg2class=imresize(seg2class,scalefactor,'nearest');

%imshow(im)
im=imnoise(im,'gaussian',0,0.005);
imdouble=double(im) + 1; %convert to numbers between 1 and 256 (double)

%% Create Feature Map(s)
[feature_map] = Image2FeatureMap2(imdouble,KernelSize); % Create Feature Map

% imDis=im;
% ZRm=floor(KernelSize);
% im=im(floor(ZRm/2):end-floor(ZRm/2)-1,floor(ZRm/2):end-floor(ZRm/2)-1);
% imdouble=imdouble(floor(ZRm/2):end-floor(ZRm/2)-1,floor(ZRm/2):end-floor(ZRm/2)-1);
% seg=seg(floor(ZRm/2):end-floor(ZRm/2)-1,floor(ZRm/2):end-floor(ZRm/2)-1);
% seg2class=seg2class(floor(ZRm/2):end-floor(ZRm/2)-1,floor(ZRm/2):end-floor(ZRm/2)-1);
% feature_map=feature_map(floor(ZRm/2):end-floor(ZRm/2)-1,floor(ZRm/2):end-floor(ZRm/2)-1,:);

flatImage=reshape(imdouble,(size(im,1)*size(im,2)),1); %make image list (of pixel values)
flatClass=reshape(seg2class,(size(seg2class,1)*size(seg2class,2)),1);
flatFeature_map=fliplr(reshape(feature_map,(size(feature_map,1)^2),KernelSize^2));
%flatFeature_mapMEAN=mean(flatFeature_map,2);

disp('Now loading Del...');
%tic

%% Calculate Graph Laplacian
sigma=10;
AdjacMat=zeros(size(flatFeature_map,1),size(flatFeature_map,1));

%tic
for i=1:size(flatImage,1)
    for j=1:size(flatImage,1)
        a=exp((-1/(2*sigma^2))*(norm(flatFeature_map(i,:)-flatFeature_map(j,:)))^2);
%         if a<0.0001
%             AdjacMat(i,j)=0;
%         else
            AdjacMat(i,j)=a;
%         end
    end
end
% toc

del=diag(sum(AdjacMat,1))-AdjacMat;
%toc

%save('bird_delp6_12_5_17.mat','del','-v7.3');
save('bird_delNOISEp6_12_11_17.mat','del','-v7.3');
%toc

load('bird_delNOISEp6_12_11_17.mat');
%toc
disp('Del Loaded');

%% Loop through prior weightings

q=1;
for lambda=lambdaspan
    disp(['For lambda = ',num2str(lambda),' ...']);
    for lambdaI=lambdaIspan
        disp(['For lambdaI = ',num2str(lambdaI),' ...']);
        %% Initial Labeled Pool
        for PoolIteration=1:PoolIterations
            disp(['For Pool Iteration = ',num2str(PoolIteration),' ...']);
            count=0;
            a=1;
            while a==1
                %PoolIndex=randperm(length(flatImage),PoolNum); %randomly samples w/o replacement
                PoolIndex = datasample(1:length(flatImage),PoolNum,'Replace',false);%,'Weights',PoolWeights);
                PoolIndex=PoolIndex';

                NewLabels=zeros(size(flatImage,1),1); %empty estimated labels
                NewLabels(PoolIndex)=flatClass(PoolIndex);

                if max(NewLabels)-min(NewLabels)==c_total
                    a=0;
                else
                    a=1;
                    %didn't sample all classes
                end
            end
            
            if mod(PoolIteration,2) == 1 %odd
                sPoolIndex=PoolIndex;
                sNewLabels=NewLabels;
                random=1;
            elseif mod(PoolIteration,2) == 0 %even
                clear PoolIndex;
                clear NewLabels;
                PoolIndex=sPoolIndex;
                NewLabels=sNewLabels;
                random=0;
            else
                random=0;
            end
            
            Output(q).PoolIt(PoolIteration).InitalPool=PoolIndex;
            save('Output10_12_12_17.mat','Output','-v7.3');

            flatFeature_map_ones = [flatFeature_map ones(size(flatFeature_map,1),1)]; %append ones
            precision=flatFeature_map_ones'*del*flatFeature_map_ones;
            LAMBDA=lambdaI*eye(size(flatFeature_map_ones,2));
               
            clear class;
            for c=1:c_total %create class lists
                PI=find(NewLabels==c);
                class{c} = flatFeature_map(PI,:);
            end

            AccuracyVsIterationTotal=zeros(1,IterationNum);
            AccuracyVsIterationClass1=zeros(1,IterationNum);
            AccuracyVsIterationClass2=zeros(1,IterationNum);

            % Iterative Loop
            for iteration=1:IterationNum 
                %% Fit logistic Regression to Current Pool
                [labels,data]=class_breakdown(class,c_total);
                [Fit, llh] = multinomial_logistic_regression_PRIOR(data', labels', precision, lambda, LAMBDA);
                %[Fit, llh] = multinomial_logistic_regression(data', labels',lambda);

                UnlabeledIndices=find(NewLabels==0); %collect unlabeled indices
                if random == 1
                    new_index = datasample(1:length(UnlabeledIndices),1,'Replace',false);

                    EstimatedUnlabeleds=zeros(length(UnlabeledIndices),1);
                    for i=1:length(UnlabeledIndices) %walk through unlabeled points
                        x=flatFeature_map(UnlabeledIndices(i),:); %at unlabeled point x
                        [y, p] = multinomial_logistic_prediction(Fit, x');
                        EstimatedUnlabeleds(i)=y;
                    end

                else
                    % Calculate the FI matrix
                [A, EstimatedUnlabeleds]=CalculateFI_11_26_17(UnlabeledIndices, feature_map, flatFeature_map, Fit, c_total);


                if max(max(max(A)))==0
                    disp('A HAS GONE TO ZERO');
                    %pause;
                    count=count+1;
                    new_index = datasample(1:length(UnlabeledIndices),1,'Replace',false);

                    trA=zeros(length(UnlabeledIndices),1); %Create zeros for trace of FI matrix
                    for i=1:length(UnlabeledIndices)
                        trA(i)=trace(A(:,:,i));
                    end
                else
                    % Find maximum entry in A
                    trA=zeros(length(UnlabeledIndices),1); %Create zeros for trace of FI matrix
                    for i=1:length(UnlabeledIndices)
                        trA(i)=trace(A(:,:,i));
                    end
                    [max_value,new_index]=max(trA);
                end

%                 if mod(iteration,50)==0 | iteration ==1
%                 HeatPlots_11_26_17(Estimate_Matrix, iteration, NewLabels, UnlabeledIndices,trA, im,Fit);
%                 pause(0.5);
%                 end

                end

                %Estimated Unlabeleds
                Estimates=zeros(size(flatImage,1),1);
                Estimates(UnlabeledIndices)=EstimatedUnlabeleds;
                EstimatesAndLabels=NewLabels;
                EstimatesAndLabels(UnlabeledIndices)=EstimatedUnlabeleds;

               %plot heatmap
                %Estimate_Matrix=zeros(size(im,2),size(im,1));
                Estimate_Matrix=reshape(EstimatesAndLabels,size(im,2),size(im,1));
%                 for i=1:length(coordinates)
%                     Estimate_Matrix(coordinates(i,2),coordinates(i,1))=EstimatesAndLabels(i);
%                 end

                NewLabels(UnlabeledIndices(new_index))=flatClass(UnlabeledIndices(new_index));
                class{flatClass(UnlabeledIndices(new_index))}=[class{flatClass(UnlabeledIndices(new_index))};flatFeature_map(UnlabeledIndices(new_index),:)];
                %pause(0.5);

                %Accuracy Measurements
                %Separate accuracies for two classes
                C1List=(Estimates==1);
                C2List=(Estimates==2);
                CList=(Estimates~=0);
                C1ListAcc=flatClass(C1List)==C1List(C1List);
                C2ListAcc=flatClass(C2List)==(C2List(C2List)*2);
                CListAcc=flatClass(CList)==Estimates(CList);

                accTotal=(sum(CListAcc)/length(CListAcc));
                accuracy1=(sum(C1ListAcc)/length(C1ListAcc));
                accuracy2=(sum(C2ListAcc)/length(C2ListAcc));

                disp(['Total Accuracy = ',num2str(accTotal),' for iteration ',num2str(iteration)]);

                AccuracyVsIterationTotal(iteration)=accTotal;
                AccuracyVsIterationClass1(iteration)=accuracy1;
                AccuracyVsIterationClass2(iteration)=accuracy2;

                Output(q).PoolIt(PoolIteration).CurrentIt(iteration).ParameterV=Fit.w;
                Output(q).PoolIt(PoolIteration).CurrentIt(iteration).Sample=UnlabeledIndices(new_index);
                save('Output10_12_12_17.mat','Output','-v7.3');
            end
            Output(q).Lambda=lambda;
            Output(q).lambdaEye=lambdaI;
            Output(q).PoolIt(PoolIteration).Count=count;
            Output(q).PoolIt(PoolIteration).AccuracyTotal=AccuracyVsIterationTotal;
            Output(q).PoolIt(PoolIteration).Accuracy1=AccuracyVsIterationClass1;
            Output(q).PoolIt(PoolIteration).Accuracy2=AccuracyVsIterationClass2;
        end
    end
    q=q+1;
end

save('Output10_12_12_17.mat','Output','-v7.3');
