function [Zall,P,obj_finall] = server_train(P,Zall,Z,S,beta,ind)

%% initialize
maxIter = 30 ; % the number of iterations

numview = length(Z);
numsample = size(Z{1},2);
m = size(Z{1},1);

%% initialize P calculate L
for iv = 1:numview
    S{iv} = (S{iv}+S{iv}')/2;
    D{iv} = diag(sum(S{iv}));
    L{iv} = D{iv} - S{iv};
    L_lmd = eig(L{iv});
    Lmx{iv}= max(L_lmd)*eye(m) - L{iv};
end

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;

    %% optimize P
    ZZ = Zall*Zall';
    parfor iv = 1:numview
        Pv=P{iv};
        fobj1 = 0;
        for rep = 1:100
            LPZZ = Lmx{iv}*Pv*ZZ';
            ZZall = Z{iv}*Zall';
            M = 2*beta*LPZZ+ZZall;
            [Um,~,Vm] = svd(M,'econ');
            Pv = Um*Vm';
            
            fobj2 = fobj1;
            fobj1 = beta*trace(Pv'*LPZZ)+trace(Pv'*ZZall);
            if rep>3 && ((fobj1-fobj2)/fobj1<1e-3)
                break;
            end
        end
        P{iv}=Pv;
    end

    %% optimize Z
    options = optimset( 'Algorithm','interior-point-convex','Display','off');
    Lp = 0;
    Zp = 0;
    for iv = 1:numview
        Lp = Lp+P{iv}'*L{iv}*P{iv};
        Zp = Zp+Z{iv}'*P{iv};
    end
    for ii=1:numsample
        ti = -Zp(ii,:);
        H = sum(ind(ii,:))*eye(m)+2*beta*Lp;
        H=(H+H')/2;
        Zall(:,ii) = quadprog(H,ti',[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],options);
    end

     %% convergence
     part1 = 0;
     part2 = 0;
     for iv = 1:numview
         part1 = part1 + norm(P{iv}*Zall.*repmat(ind(:,iv)',m,1)-Z{iv},'fro')^2;
         part2 = part2 + trace(Zall'*P{iv}'*L{iv}*P{iv}*Zall);
     end
    obj(iter) = part1 + 2*beta*part2;
    
    if (iter>1) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
        obj_finall = obj(iter);
        flag = 0;
    end
   
end
         
         
    
