clear;

%% Inputs: Image, Actual Labels, Labeled Pool Size, iterations (or
%confidence), # of classes
IterationNum=10000;
c_total=2;
PoolIterations=10;
PoolNum=1000; %Number of samples in initial labeled pool
lambdaspan=10.^linspace(-8,0,9);
lambdaIspan=10^-8;

im=rgb2gray(imread('BSDS300/images/train/135069.jpg')); %converts truecolor to intensity

seg = readSeg('BSDS300/human/color/1105/135069.seg');
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

%% Use three classes of this image (birds vs tail vs background)
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

scalefactor=1.0;
im=imresize(im,scalefactor);
seg=imresize(seg,scalefactor,'nearest');
seg2class=imresize(seg2class,scalefactor,'nearest');

%%imshow(im)
%%im=imnoise(im,'gaussian',0,0.005);
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
flatFeature_mapMEAN=mean(flatFeature_map,2);

%% Calculate Graph Laplacian
sigma=10;
AdjacMat=zeros(size(flatFeature_map,1),size(flatFeature_map,1));

% tic
for i=1:size(flatImage,1)
    for j=1:size(flatImage,1)
        a=exp((-1/(2*sigma^2))*(norm(flatFeature_map(i,:)-flatFeature_map(j,:)))^2);
        if a<0.0001
            AdjacMat(i,j)=0;
        else
            AdjacMat(i,j)=a;
        end
    end
end
% toc

del=diag(sum(AdjacMat,1))-AdjacMat;
% toc

save('bird_del_11_15_17.mat','del','-v7.3');
%toc

%load('bird_del_11_7_17.mat');