clear;
clc;

addpath(genpath('./'));

resultdir1 = 'Results/';
if (~exist('Results', 'file'))
    mkdir('Results');
    addpath(genpath('Results/'));
end

resultdir2 = 'aResults/';
if (~exist('aResults', 'file'))
    mkdir('aResults');
    addpath(genpath('aResults/')); 
end

datadir='.\Data\';
dataname={'proteinFold'};
numdata = length(dataname); % number of the test datasets
numname = {'_Per0.1', '_Per0.2', '_Per0.3', '_Per0.4','_Per0.5', '_Per0.6', '_Per0.7', '_Per0.8', '_Per0.9'};

for idata = 1:1 :length(dataname)
    ResBest = zeros(10, 8);
    ResStd = zeros(10, 8);
    for dataIndex = 1:1:9
        datafile = [datadir, cell2mat(dataname(idata)), cell2mat(numname(dataIndex)), '.mat'];
        load(datafile);
        %data preparation...
        gt = truelabel{1};
        k = length(unique(gt));
        tic;
        [X1, ind] = findindex(data, index);
        numview = length(X1);
        n = size(X1{1},2); 

        time1 = toc;
        maxAcc = 0;
        TempLambda1 = [k,2*k,3*k];
        TempLambda2 = [0.001 0.1 1 100];
        TempLambda3 = [0.001 0.1 1 100];
        
        ACC{dataIndex} = zeros(length(TempLambda1),length(TempLambda2),length(TempLambda3));
        NMI{dataIndex} = zeros(length(TempLambda1), length(TempLambda2),length(TempLambda3));
        Purity{dataIndex} = zeros(length(TempLambda1), length(TempLambda2),length(TempLambda3));
        idx = 1;
        
        for LambdaIndex1 = 1 : length(TempLambda1)
            lambda1 = TempLambda1(LambdaIndex1);
            for LambdaIndex2 = 1 : length(TempLambda2)
                lambda2 = TempLambda2(LambdaIndex2);
                
                for iv = 1:numview
                    Z{iv} = client_pretrain(X1{iv},lambda1,lambda2,ind(:,iv));
                end
                Zall = server_pretrain(Z,ind,lambda1);

                for LambdaIndex3 = 1 : length(TempLambda3)
                    lambda3 = TempLambda3(LambdaIndex3);
                    disp([char(dataname(idata)), char(numname(dataIndex)), '-l1=', num2str(lambda1), '-l2=', num2str(lambda2), '-l3=', num2str(lambda3)]);
                    tic;
                    flag = 1;
                    iter = 0;
                    maxIter = 5;
                    for iv = 1:numview
                        P{iv} = diag(ones(lambda1,1));
                    end
                    while flag
                        iter = iter + 1;
                        parfor iv = 1:numview
                            [Z{iv},S{iv},obj_client(iv)] = client_train(X1{iv},Z{iv},lambda1,lambda2,ind(:,iv),Zall,P{iv});
                        end
                        
                        [Zall,P,obj_server] = server_train(P,Zall,Z,S,lambda3,ind);

                        obj(iter) = sum(obj_client)+obj_server;

                        if (iter>1) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
                            [UU,~,V]=mySVD(Zall',k);
                            flag = 0;
                        end
                    end

                    F = UU ./ repmat(sqrt(sum(UU .^ 2, 2)), 1, size(UU,2));

                    time2 = toc;
                    stream = RandStream.getGlobalStream;
                    reset(stream);
                    MAXiter = 1000; % Maximum number of iterations for KMeans
                    REPlic = 20; % Number of replications for KMeans
                    tic;
                    for rep = 1 : 20
                        pY = kmeans(F, k, 'maxiter', MAXiter, 'replicates', REPlic, 'emptyaction', 'singleton');
                        res(rep,:) = Clustering8Measure(gt, pY);
                    end
                    time3 = toc;
                    runtime(idx) = time1 + time2 + time3/20;
                    disp(['runtime:', num2str(runtime(idx))])
                    idx = idx + 1;
                    tempResBest(dataIndex, : ) = mean(res);
                    tempResStd(dataIndex, : ) = std(res);
                    ACC{dataIndex}(LambdaIndex1, LambdaIndex2, LambdaIndex3) = tempResBest(dataIndex, 1);
                    NMI{dataIndex}(LambdaIndex1, LambdaIndex2, LambdaIndex3) = tempResBest(dataIndex, 2);
                    Purity{dataIndex}(LambdaIndex1, LambdaIndex2, LambdaIndex3) = tempResBest(dataIndex, 3);
                    save([resultdir1, char(dataname(idata)), char(numname(dataIndex)), '-l1=', num2str(lambda1), '-l2=', num2str(lambda2), '-l3=', num2str(lambda3), ...
                        '-acc=', num2str(tempResBest(dataIndex,1)), '_result.mat'], 'tempResBest', 'tempResStd');
                    for tempIndex = 1 : 8
                        if tempResBest(dataIndex, tempIndex) > ResBest(dataIndex, tempIndex)
                            if tempIndex == 1
                                newZ = Z;
                                newF = F;
                            end
                            ResBest(dataIndex, tempIndex) = tempResBest(dataIndex, tempIndex);
                            ResStd(dataIndex, tempIndex) = tempResStd(dataIndex, tempIndex);
                        end
                    end
                end
            end
        end
        aRuntime{dataIndex} = mean(runtime);
        PResBest = ResBest(dataIndex, :);
        PResStd = ResStd(dataIndex, :);
        save([resultdir2, char(dataname(idata)), char(numname(dataIndex)), 'ACC_', num2str(max(ACC{dataIndex}(:))), '_result.mat'], 'ACC', 'NMI', 'Purity', 'aRuntime', ...
            'newZ', 'newF', 'PResBest', 'PResStd');
    end
    ResBest(10, :) = mean(ResBest(1:9, :));
    ResStd(10, :) = mean(ResStd(1:9, :));
    save([resultdir2, char(dataname(idata)), '_result.mat'], 'ResBest', 'ResStd','aRuntime','ACC');
end
