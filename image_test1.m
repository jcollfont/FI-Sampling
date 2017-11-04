%Test of accessing image segmentations
clear;
close all;

im=rgb2gray(imread('BSDS300/images/train/135069.jpg')); %converts truecolor to intensity
figure()
imshow(im);

seg = readSeg('BSDS300/human/color/1105/135069.seg');
bmap = seg2bmap(seg);
[cmap,cid2sid] = seg2cmap(seg,bmap);
FG = readbin('fgdata/color/1105/135069-105.fg');

%Full map
[r,c]=size(seg);
segF=zeros(r,c);
for i=1:r
    for ii=1:c
        segF(i,ii)=seg(i,ii)*42;
    end
end
I=mat2gray(segF);
figure()
imshow(I)

%Map 1
seg1=zeros(r,c);
for i=1:r
    for ii=1:c
        if seg(i,ii)==1
            seg1(i,ii)=1;
        end
    end
end
I1=mat2gray(seg1);
figure()
imshow(I1)
title('Section 1')

%Map 2
seg2=zeros(r,c);
for i=1:r
    for ii=1:c
        if seg(i,ii)==2
            seg2(i,ii)=1;
        end
    end
end
I2=mat2gray(seg2);
figure()
imshow(I2)
title('Section 2')

%Map 3
seg3=zeros(r,c);
for i=1:r
    for ii=1:c
        if seg(i,ii)==3
            seg3(i,ii)=1;
        end
    end
end
I3=mat2gray(seg3);
figure()
imshow(I3)
title('Section 3')

%Map 4
seg4=zeros(r,c);
for i=1:r
    for ii=1:c
        if seg(i,ii)==4
            seg4(i,ii)=1;
        end
    end
end
I4=mat2gray(seg4);
figure()
imshow(I4)
title('Section 4')

