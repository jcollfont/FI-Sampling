function [feature_map] = Image2FeatureMap2(image,kernelDim)
    im2=padarray(image,[floor(kernelDim/2) floor(kernelDim/2)],0);

    k=zeros(kernelDim,kernelDim);
    Cstack=zeros(size(im2,1),size(im2,2),kernelDim^2);
    for i=1:kernelDim^2
        k(i)=1;
        Cstack(:,:,i)=convn(image,k);
        k=zeros(kernelDim,kernelDim);
    end
    
    rm=size(im2,1)-size(image,1);
    rm=rm/2;
    Cstack=Cstack(1+rm:end-rm,1+rm:end-rm,:);
    
    feature_map = Cstack;
end