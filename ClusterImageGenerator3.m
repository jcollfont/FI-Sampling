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
mu{1}=180;%(imagedim).*rand;
mu{2}=200;%(imagedim).*rand;
mu{3}=100;
mu{4}=50;
sigma{1}=10;
sigma{2}=10;
sigma{3}=20;
sigma{4}=50;

for i1=1:imagedim
    for i2=1:imagedim
        if image1(i1,i2)==1
            image1(i1,i2)=normrnd(mu{1},sigma{1});
        elseif image1(i1,i2)==2
            image1(i1,i2)=normrnd(mu{2},sigma{2});
        elseif image1(i1,i2)==3
            image1(i1,i2)=normrnd(mu{3},sigma{3});
        elseif image1(i1,i2)==4
            image1(i1,i2)=normrnd(mu{4},sigma{4});
        end
        if image1(i1,i2)<1
            image1(i1,i2)=1;
        end
        if image1(i1,i2)>imagedim
            image1(i1,i2)=imagedim;
        end
    end
end
