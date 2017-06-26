%Generate gaussian spread random images
c_total=2; %total number of classes
imagedim=256;
imagelabels=zeros(imagedim,imagedim);
for i1=1:imagedim
    for i2=1:imagedim
        if i1<floor(imagedim/2)
            imagelabels(i1,i2)=1;
        else
            imagelabels(i1,i2)=2;
        end
    end
end

image1=imagelabels;
mu1=imagedim/5;%(imagedim).*rand;
mu2=(imagedim/5)*4;%(imagedim).*rand;
sigma1=20;
sigma2=10;

for i1=1:imagedim
    for i2=1:imagedim
        if image1(i1,i2)==1
            image1(i1,i2)=normrnd(mu1,sigma1);
        elseif image1(i1,i2)==2
            image1(i1,i2)=normrnd(mu2,sigma2);
        end
        if image1(i1,i2)<1
            image1(i1,i2)=1;
        end
        if image1(i1,i2)>imagedim
            image1(i1,i2)=imagedim;
        end
    end
end
