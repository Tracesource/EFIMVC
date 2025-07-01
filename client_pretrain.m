function Z = client_pretrain(X,lambda1,lambda2,ind)
n = size(X,2);
P = diag(ones(lambda1,1));
indexM = find(ind);
rand('twister',12);
[~, A] = litekmeans(X(:,indexM)',lambda1,'MaxIter', 100,'Replicates',10);
A = A';
Z = zeros(lambda1,n);
options = optimset( 'Algorithm','interior-point-convex','Display','off');
H = A'*A+lambda2*eye(lambda1);
C1 = X'*A;
for ii=1:length(indexM)
    ti = -C1(indexM(ii),:);
    Z(:,indexM(ii)) = quadprog(H,ti',[],[],ones(1,lambda1),1,zeros(lambda1,1),ones(lambda1,1),[],options);
end