%Generate gaussian spread random images
c_total=4; %total number of classes
imagedim=256;
imagelabels=zeros(imagedim,imagedim);
for i1=1:imagedim
    for i2=1:imagedim
        if i1<floor(imagedim/2)
            if i2<floor(imagedim/2)
                imagelabels(i1,i2)=1;
            else
                imagelabels(i1,i2)=2;
            end
        else
            if i2<floor(imagedim/2)
                imagelabels(i1,i2)=3;
            else
                imagelabels(i1,i2)=4;
            end
        end
    end
end

image1=imagelabels;
mu1=imagedim/5;%(imagedim).*rand;
mu2=(imagedim/5)*2;%(imagedim).*rand;
mu3=(imagedim/5)*3;%(imagedim).*rand;
mu4=(imagedim/5)*4;%(imagedim).*rand;
sigma1=3;
sigma2=sigma1;
sigma3=sigma1;
sigma4=sigma1;

for i1=1:imagedim
    for i2=1:imagedim
        if image1(i1,i2)==1
            image1(i1,i2)=normrnd(mu1,sigma1);
        elseif image1(i1,i2)==2
            image1(i1,i2)=normrnd(mu2,sigma2);
        elseif image1(i1,i2)==3
            image1(i1,i2)=normrnd(mu3,sigma3);
        else
            image1(i1,i2)=normrnd(mu4,sigma4);
        end
        if image1(i1,i2)<1
            image1(i1,i2)=1;
        end
        if image1(i1,i2)>imagedim
            image1(i1,i2)=imagedim;
        end
    end
end

