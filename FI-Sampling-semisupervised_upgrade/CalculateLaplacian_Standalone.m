%Calculate the Laplacian as a standalone because computationally intensive
%load('10_2_17_test_rev1.mat')

clear;
IterationNum=300;
c_total=2;
PoolNum=16; %Number of samples in initial labeled pool

im=rgb2gray(imread('1flower.jpeg')); %converts truecolor to intensity

%Need to make image square so that Degree matrix is square
if size(im,1)>size(im,2)
    ldiff=size(im,1)-size(im,2);
%     im=im(ldiff:end-1,:);
    im=im((ldiff/2+.5)+50:end-(ldiff/2+.5)-70,50:end-70);
elseif size(im,1)<size(im,2)
    ldiff=size(im,2)-size(im,1);
%     im=im(:,ldiff:end-1);
    im=im(100:end-70,(ldiff/2+.5)+100:end-(ldiff/2-.5)-70);
else
    %do Nothing -- dimensions are squared
end

imshow(im)

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
flatFeature_map=mean(flatFeature_map,2);

%% Downsample here...

%% Calculate Graph Laplacian
sigma=10;
tic
AdjacMat=zeros(size(flatFeature_map,1),size(flatFeature_map,1)); %Change to feature map!
toc
for i=1:size(flatImage,1)
    for j=1:size(flatImage,1)
        a=exp((-1/(2*sigma^2))*(norm(flatFeature_map(i,:)-flatFeature_map(j,:)))^2);
        if a<0.001
            AdjacMat(i,j)=0;
        else
            AdjacMat(i,j)=a;
        end
    end
end
toc

del=diag(sum(AdjacMat,1))-AdjacMat;
toc

save('10_3_17_test_rev1.mat');
% save('10_3_17_Adjacency.mat','AdjacMat','del','-v7.3');