%% Inputs: Image, Actual Labels, Labeled Pool Size, iterations (or
%confidence), # of classes
IterationNum=1000;
c_total=2;
PoolIterations=1;
PoolNum=30; %Number of samples in initial labeled pool
lambdaspan=1; %10.^linspace(-8,0,9);
lambdaIspan=10^-6;

GenerateBWimage1

im=(imread('BWtest.jpg')); %converts truecolor to intensity

load('BWtestTruth.mat');
seg2class=seg;

%% Shrink Image further (assumed square)

%scalefactor=1.0;
%im=imresize(im,scalefactor);
%seg2class=imresize(seg2class,scalefactor,'nearest');

%imshow(im)
im=imnoise(im,'gaussian',0,0.005);
imdouble=double(im) + 1; %convert to numbers between 1 and 256 (double)

%% Create Feature Map(s)
[feature_map] = Image2FeatureMap2(imdouble,9); % Create Feature Map

coordinates=zeros(size(imdouble,1)*size(imdouble,2),2); %Create list of coordinates
c=1:size(imdouble,2);
c=c';
coordinates(:,2)=repmat(c,size(imdouble,1),1);
r=1;c=1;
for i=1:size(coordinates,1)
    if c<size(imdouble,2)
        coordinates(i,1)=r;
        c=c+1;
    else
        coordinates(i,1)=r;
        c=1;
        r=r+1;
    end
end

flatImage=reshape(imdouble,(size(im,1)*size(im,2)),1); %make image list (of pixel values)
flatClass=reshape(seg2class,(size(seg2class,1)*size(seg2class,2)),1);
flatFeature_map=reshape(feature_map,(size(feature_map,1)^2),9^2);
%flatFeature_mapMEAN=mean(flatFeature_map,2);

disp('Now loading Del...');
%tic

%% Calculate Graph Laplacian
% sigma=10;
% AdjacMat=zeros(size(flatFeature_map,1),size(flatFeature_map,1));
% 
% % tic
% for i=1:size(flatImage,1)
%     for j=1:size(flatImage,1)
%         a=exp((-1/(2*sigma^2))*(norm(flatFeature_map(i,:)-flatFeature_map(j,:)))^2);
%         if a<0.0001
%             AdjacMat(i,j)=0;
%         else
%             AdjacMat(i,j)=a;
%         end
%     end
% end
% % toc
% 
% del=diag(sum(AdjacMat,1))-AdjacMat;
% % toc
% 
% save('BWtest_del_11_26_17.mat','del','-v7.3');
%toc

load('BWtest_del_11_26_17.mat');
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
            
            a=1;
            while a==1
                PoolIndex=randperm(length(flatImage),PoolNum); %randomly samples w/o replacement
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
            
            BWOutput(q).PoolIt(PoolIteration).InitalPool=PoolIndex;
            save('BWOutput_11_26_17.mat','BWOutput','-v7.3');

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
                %[Fit, llh] = multinomial_logistic_regression(data', labels');
                
                % Calculate the FI matrix
                UnlabeledIndices=find(NewLabels==0); %collect unlabeled indices
                [A, EstimatedUnlabeleds]=CalculateFI_11_26_17(UnlabeledIndices, feature_map, flatFeature_map, Fit, c_total);
                
                if max(max(max(A)))==0
                    disp('A HAS GONE TO ZERO');
                    pause;
                end
                
                %Estimated Unlabeleds
                Estimates=zeros(size(flatImage,1),1);
                Estimates(UnlabeledIndices)=EstimatedUnlabeleds;
                EstimatesAndLabels=NewLabels;
                EstimatesAndLabels(UnlabeledIndices)=EstimatedUnlabeleds;

               %plot heatmap
                Estimate_Matrix=zeros(size(im,1),size(im,2));
                for i=1:length(coordinates)
                    Estimate_Matrix(coordinates(i,1),coordinates(i,2))=EstimatesAndLabels(i);
                end

                % Find maximum entry in A
                trA=zeros(length(UnlabeledIndices),1); %Create zeros for trace of FI matrix
                for i=1:length(UnlabeledIndices)
                    trA(i)=trace(A(:,:,i));
                end
                [max_value,new_index]=max(trA);
                
                %HeatPlots_11_26_17(Estimate_Matrix, iteration, NewLabels, UnlabeledIndices,trA, im, coordinates);

                NewLabels(UnlabeledIndices(new_index))=flatClass(UnlabeledIndices(new_index));
                class{flatClass(UnlabeledIndices(new_index))}=[class{flatClass(UnlabeledIndices(new_index))};flatFeature_map(new_index,:)];
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
                
                BWOutput(q).PoolIt(PoolIteration).CurrentIt(iteration).ParameterV=Fit.w;
                BWOutput(q).PoolIt(PoolIteration).CurrentIt(iteration).Sample=UnlabeledIndices(new_index);
                save('BWOutput_11_26_17.mat','BWOutput','-v7.3');
            end
            BWOutput(q).Lambda=lambda
            BWOutput(q).lambdaEye=lambdaI;
            BWOutput(q).PoolIt(PoolIteration).AccuracyTotal=AccuracyVsIterationTotal;
            BWOutput(q).PoolIt(PoolIteration).Accuracy1=AccuracyVsIterationClass1;
            BWOutput(q).PoolIt(PoolIteration).Accuracy2=AccuracyVsIterationClass2;
        end
    end
    q=q+1;
end

save('BWOutput_11_26_17.mat','BWOutput','-v7.3');

