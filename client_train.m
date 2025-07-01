function [Z,S,obj_finall] = client_train(X,Z,m,lambda,ind,Zall,P)

%% initialize
maxIter = 30 ; % the number of iterations

numsample = size(X,2);
d = size(X,1); 
indexM = find(ind);
PZall = P*Zall;

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;

    
   %% optimize A
    C = Z*Z';
    D = X * Z';
    if cond(C)>1e12
        A = D*pinv(C);
    else
        A = D*inv(C);
    end

    %% optimize Z
    options = optimset( 'Algorithm','interior-point-convex','Display','off');
    H = A'*A+lambda*eye(m);
    C1 = X'*A+lambda*PZall';
    for ii=1:length(indexM)
        ti = -C1(indexM(ii),:);
        Z(:,indexM(ii)) = quadprog(H,ti',[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],options);
    end
    
    %% convergence
    obj(iter) = norm(X-A*Z,'fro')^2+lambda*norm(PZall.*repmat(ind',m,1)-Z,'fro')^2;
    
    if (iter>1) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
        D = eucl_dist2(A',A');
        sigma = 0.3*((m-1)*0.5-1)+0.8;
        S = exp(-D/(2*sigma^2));
        obj_finall = obj(iter);
        flag = 0;
    end
end
         
         
    
