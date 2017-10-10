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
mu{1}=100;%(imagedim).*rand;
mu{2}=150;%(imagedim).*rand;
sigma{1}=20;
sigma{2}=20;

for i1=1:imagedim
    for i2=1:imagedim
        if image1(i1,i2)==1
            image1(i1,i2)=normrnd(mu{1},sigma{1});
        elseif image1(i1,i2)==2
            image1(i1,i2)=normrnd(mu{2},sigma{2});
        end
        if image1(i1,i2)<1
            image1(i1,i2)=1;
        end
        if image1(i1,i2)>imagedim
            image1(i1,i2)=imagedim;
        end
    end
end

