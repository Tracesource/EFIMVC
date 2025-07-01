function Zall = server_pretrain(Z,ind,lambda1)
numview = length(Z);
n = size(Z{1},2); 
Zall=zeros(lambda1,n);
for iv = 1:numview
    Zall = Zall+Z{iv};
end
Zall = Zall.*repmat(1./sum(ind,2)',lambda1,1);