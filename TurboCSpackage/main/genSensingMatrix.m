function [ A ] = genSensingMatrix( M, N, typeMatrix )
% genSensingMatrix
%

switch typeMatrix
    %iid Gaussian Sensing Matrix
    case 'GaussM'
        A = randn(M,N)/sqrt(M);
    case 'GaussN'
        A = randn(M,N)/sqrt(N);
    case 'CGaussM'
        A = (randn(M,N) + 1i*randn(M,N)) / sqrt(2*M);
    case 'CGaussN'
        A = (randn(M,N) + 1i*randn(M,N)) / sqrt(2*N);
        
    %partical DCT Sensing Matrix
    case 'pDCT'
        perm = randperm(N);
        index_sel = perm(1:M);
        F = dctmtx(N);
        A = F(index_sel, :);
        
    %partical DFT Sensing Matrix
    case 'pDFT'
        perm = randperm(N);
        index_sel = perm(1:M);
        F = dftmtx(N)/sqrt(N);
        A = F(index_sel, :);
        
    case 'pInvDFT'
        perm = randperm(N);
        index_sel = perm(1:M);
        F = dftmtx(N)/sqrt(N);
        F = F';
        A = F(index_sel, :);
        
    %partical double DCT Sensing Matrix
    case 'pdouDCT'
        %d = random('Binomial',1,0.5,N,1);
        d = randi([0 1],N,1);
        d = 2 * d - 1;
        F = dctmtx(N);
        F = F * diag(d) * F';
        perm = randperm(N);
        index_sel = perm(1:M);
        A = F(index_sel, :);
        
    %partical double DFT Sensing Matrix
    case 'pdouDFT'
        %d = random('Binomial',1,0.5,N,1);
        d = randi([0 1],N,1);
        d = 2 * d - 1;
        F = dftmtx(N)/sqrt(N);
        E = F * diag(d) * F';
        perm = randperm(N);
        index_sel = perm(1:M);
        A = E(index_sel, :);
        
    case 'pDFT_RP'
        perm = randperm(N);
        index_sel = perm(1:M);
        F = dftmtx(N)/sqrt(N);
        A = F(index_sel, :);
        perm1 = randperm(N);
        perMat = eye(N);
        perMat1 = perMat(perm1,:);
        A = A * perMat1;
        
    case 'pDCT_RP'
        perm = randperm(N);
        index_sel = perm(1:M);
        F = dctmtx(N);
        A = F(index_sel, :);
        perm = randperm(N);
        perMat = eye(N);
        perMat = perMat(perm,:);
        A = A*perMat;
        
    otherwise
        error('Unrecognized sensing matrix.');
end


end

