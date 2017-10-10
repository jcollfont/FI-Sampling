function [feature_map] = Image2FeatureMap(image)
    rownum=size(image,1);
    columnnum=size(image,2);
    feature_map=zeros(rownum,columnnum,9);
    for i=1:rownum
        for ii=1:columnnum
            if i==1
                if ii==1
                    i1=0;
                    i2=0;
                    i3=0;
                    i4=0;
                    i5=image(i,ii);
                    i6=image(i,ii+1);
                    i7=0;
                    i8=image(i+1,ii);
                    i9=image(i+1,ii+1);
                elseif ii==columnnum
                    i1=0;
                    i2=0;
                    i3=0;
                    i4=image(i,ii-1);
                    i5=image(i,ii);
                    i6=0;
                    i7=image(i+1,ii-1);
                    i8=image(i+1,ii);
                    i9=0;
                else
                    i1=0;
                    i2=0;
                    i3=0;
                    i4=image(i,ii-1);
                    i5=image(i,ii);
                    i6=image(i,ii+1);
                    i7=image(i+1,ii-1);
                    i8=image(i+1,ii);
                    i9=image(i+1,ii+1);
                end
            elseif i==rownum
                if ii==1
                    i1=0;
                    i2=image(i-1,ii);
                    i3=image(i-1,ii+1);
                    i4=0;
                    i5=image(i,ii);
                    i6=image(i,ii+1);
                    i7=0;
                    i8=0;
                    i9=0;
                elseif ii==columnnum
                    i1=image(i-1,ii-1);
                    i2=image(i-1,ii);
                    i3=0;
                    i4=image(i,ii-1);
                    i5=image(i,ii);
                    i6=0;
                    i7=0;
                    i8=0;
                    i9=0;
                else
                    i1=image(i-1,ii-1);
                    i2=image(i-1,ii);
                    i3=image(i-1,ii+1);
                    i4=image(i,ii-1);
                    i5=image(i,ii);
                    i6=image(i,ii+1);
                    i7=0;
                    i8=0;
                    i9=0;
                end
            else
                if ii==1
                    i1=0;
                    i2=image(i-1,ii);
                    i3=image(i-1,ii+1);
                    i4=0;
                    i5=image(i,ii);
                    i6=image(i,ii+1);
                    i7=0;
                    i8=image(i+1,ii);
                    i9=image(i+1,ii+1);
                elseif ii==columnnum
                    i1=image(i-1,ii-1);
                    i2=image(i-1,ii);
                    i3=0;
                    i4=image(i,ii-1);
                    i5=image(i,ii);
                    i6=0;
                    i7=image(i+1,ii-1);
                    i8=image(i+1,ii);
                    i9=0;
                else
                    i1=image(i-1,ii-1);
                    i2=image(i-1,ii);
                    i3=image(i-1,ii+1);
                    i4=image(i,ii-1);
                    i5=image(i,ii);
                    i6=image(i,ii+1);
                    i7=image(i+1,ii-1);
                    i8=image(i+1,ii);
                    i9=image(i+1,ii+1);
                end
            end
            feature_map(i,ii,1)=i1;
            feature_map(i,ii,2)=i2;
            feature_map(i,ii,3)=i3;
            feature_map(i,ii,4)=i4;
            feature_map(i,ii,5)=i5;
            feature_map(i,ii,6)=i6;
            feature_map(i,ii,7)=i7;
            feature_map(i,ii,8)=i8;
            feature_map(i,ii,9)=i9;
        end
    end
end