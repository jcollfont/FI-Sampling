function [labels, data] = class_breakdown(class,c_total)
% Input:
%   class: full matrix of features broken down by class
% Output:
%   labels: labels returned
%   data: data returned
L=zeros(1,c_total);
for c1=1:c_total
    L(c1)=size(class{c1},1);
end
for c1=1:c_total
    l{c1}=ones(L(c1),1)*c1;
end
labels=l{1};
data=class{1};
for c1=2:c_total
    labels=vertcat(labels,l{c1});
    data=vertcat(data,class{c1});
end

end