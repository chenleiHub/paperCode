%% demo2
% This file provides several examples of Turbo Compressed Sensing based
% algorithm. The parameters are updated by expectation maximization.
%
% Coded by: Lei Chen
% E-mail: leichen094@gmail.com or chenlei_imail@163.com

%%
clc;
clear;

basePath = [fileparts(mfilename('fullpath')) filesep];
addpath([basePath '/main']) %add main function

rng('default'); % Initialization.
rng(1); % Choose a seed.

scenario = 2; % Choose a scenario.

%% Simulation 1
% This simulation is a test for Turbo-CS based algorithms with expectation
% maximization (EM) update.

if scenario == 1
    
SNRRange = 30;
NSIM = 100;
K = 32;
blockNumber = 1;
M = 51;
N = 128;
T = 1;
Iteration = 50;

signalPara.K = K; % Number of the non-zero elements
signalPara.blockNumber = blockNumber; % Non-zero block numbers
signalPara.N = N; % Dimension of the unknown signal
signalPara.T = T; % Dimension of the multiple measurement
signalPara.xMean = 0; % Mean value of the Gaussian distribution
signalPara.xVar = N/K;
%p01 = 0.0040;% for N = 2000, K = 500, grpNum = 2
%p01 = 0.0625;% for N = 128, K = 32, grpNum = 2
%p01 = 0.03125;% for N = 128, K = 32, grpNum = 1

NMSETurboCS = NaN*ones(NSIM,1); % Estimation error for Turbo-CS
NMSESTCS = NaN*ones(NSIM,1); % Estimation error for structured Turbo-CS
for nsnr = 1:length(SNRRange)
for nsim = 1:NSIM
    % Generation linear model
    x = genSignalMatrix('ComplexBlock', signalPara);
    A = genSensingMatrix(M, N, 'pDFT_RP');
    z = A * x;
    wvar = norm(z(:))^2/M*10^(-SNRRange(nsnr)/10);
    w = sqrt(wvar)*(randn(size(z))+1i*randn(size(z)))/sqrt(2);
    y = z + w;
    
    % EM initialization
    %wvar0 = norm(y,2)^2/(100+1)/M;
    wvar0 = wvar;
    xlambda0 = 0.3;
    xmean0 = 0;
    xvar0 = (norm(y,2)^2 - M*wvar0)/(norm(A,'fro')*xlambda0);

    % Initialize Module A
    gA = AwgnModuleA(y, A, wvar0);
    gA.isOrthMatrix = true;
    gA.isEMUpatePerIter = false;
    gA.isEMUpdateVar = false;  
    
    % Initialize Module B
    gX = BernGaussModuleB(xlambda0, xmean0, xvar0);
    gX.isComplex = true;
    gX.isEMUpatePerIter = false;
    gX.isEMUpdateLambda = true;
    gX.isEMUpdateMean = false;
    gX.isEMUpdateVar = true;
    
    % EM Turbo-CS
    tcsOpt = tcsOption();
    tcsOpt.nit = 50;
    tcsOpt.damping = 0.8;
    tcsOpt.tol = -1;
    tcsOpt.xTrue = x;
    tcsOpt.isShowResult = false;
    tcsOpt.isErrorRecord = false;
    tcsState = tcsState();
    tcsState.xApri = zeros(N,T);
    tcsState.vApri = gX.xvar;
    tcsState.N = N;
    outIter = 3;
    [ estFin ] = EMTurboCS(gA, gX, tcsOpt, tcsState, outIter);
    estimError = (norm(x - estFin.xBpost,'fro')/norm(x,'fro'))^2;
    NMSETurboCS(nsim,1) = estimError;
    fprintf('EM-Turbo-CS: ');
    fprintf('NSIM = %d, NMSE = %.4f, %.4f dB\n', nsim, estimError, 10*log10(estimError));
    
    % Initialize Module A
    gA2 = AwgnModuleA(y, A, wvar0);
    gA2.isOrthMatrix = true;
    gA2.isEMUpatePerIter = false;
    gA2.isEMUpdateVar = false;

    % Initialize Module B
    xP01 = 0.1*ones(1,T);
    xP10 = xlambda0.*xP01./(1 - xlambda0);
    gX2 = MarkovChainModuleB(xlambda0, xlambda0, xP01, xP10, xmean0, xvar0);
    gX2.isComplex = true;
    gX2.isEMUpatePerIter = false;
    gX2.isEMUpdateLambda0 = true;
    gX2.isEMUpdateTransProb = true;
    gX2.isEMUpdateMean = false;
    gX2.isEMUpdateVar = true; 
    
    % EM structured Turbo-CS
    tcsState2 = tcsState(); % Initialize a new state for structured Turbo-CS
    tcsState2.xApri = zeros(N,T);
    tcsState2.vApri = gX2.xvar;
    tcsState2.N = N;
    [ estFin2 ] = EMTurboCS(gA2, gX2, tcsOpt, tcsState2, outIter);
    estimError = (norm(x - estFin2.xBpost,'fro')/norm(x,'fro'))^2;
    NMSESTCS(nsim,1) = estimError;
    fprintf('EM-STCS: ');
    fprintf('NSIM = %d, NMSE = %.4f, %.4f dB\n', nsim, estimError, 10*log10(estimError));
    
    
end
end

fprintf('Mean error of EM-Turbo-CS: NMSE = %.4f, %.4f dB\n', mean(NMSETurboCS,1), 10*log10(mean(NMSETurboCS,1)));
fprintf('Mean error of EM-STCS: NMSE = %.4f, %.4f dB\n', mean(NMSESTCS,1), 10*log10(mean(NMSESTCS,1)));
%figure(1);
% plot(10*log10(mean(NMSESigTurboCS, 1)));
% xlabel('Iteration');
% ylabel('NMSE/dB');
% grid on;

end

%% Simulation 2
% This simulation is a test for Turbo-CS based algorithms with expectation
% maximization (EM) update. The algorithms are tested under a wide range of
% SNR and pilot numbers.

if scenario == 2
   
N = 128;
T = 1;
SNRRange = 20:-2:-10;
p = 0.05:0.05:1;
MRange = ceil(p*N);
% SNRRange = 30;
% MRange = 0.5*N;
NSIM = 100;

load H_suburban_macro_128;

NMSETurboCS = NaN*ones(length(SNRRange),length(MRange),NSIM); % Estimation error for Turbo-CS
NMSESTCS = NaN*ones(length(SNRRange),length(MRange),NSIM); % Estimation error for structured Turbo-CS
for nsnr = 1:length(SNRRange)
for nm = 1:length(MRange)
for nsim = 1:NSIM
    M = MRange(nm);
    SNR = SNRRange(nsnr);
    % Generation linear model

    %hAngular = genSignalMatrix('ComplexBlock', signalPara);
    hAngular = Hwtlt(:,:,nsim).';
    A = genSensingMatrix(M, N, 'pDFT_RP');
    z = A * hAngular;
    wvar = norm(z(:))^2/M*10^(-SNR/10);
    w = sqrt(wvar)*(randn(size(z))+1i*randn(size(z)))/sqrt(2);
    y = z + w;
    
    % EM initialization
    %wvar0 = norm(y,2)^2/(100+1)/M;
    wvar0 = wvar;
    xlambda0 = 0.5;
    xmean0 = 0;
    %xvar0 = (norm(y,2)^2 - M*wvar0)/(norm(A,'fro')*xlambda0);
    xvar0 = norm(y,2)^2/(norm(A,'fro')^2*((M/2)/N));

    % Initialize Module A
    gA = AwgnModuleA(y, A, wvar0);
    gA.isOrthMatrix = true;
    gA.isEMUpatePerIter = false;
    gA.isEMUpdateVar = false;  
    
    % Initialize Module B
    gX = BernGaussModuleB(xlambda0, xmean0, xvar0);
    gX.isComplex = true;
    gX.isEMUpatePerIter = false;
    gX.isEMUpdateLambda = true;
    gX.isEMUpdateMean = false;
    gX.isEMUpdateVar = true;
    
    % EM Turbo-CS
    tcsOpt = tcsOption();
    tcsOpt.nit = 30;
    tcsOpt.damping = 0.8;
    tcsOpt.tol = -1;
    tcsOpt.xTrue = hAngular;
    tcsOpt.isShowResult = false;
    tcsOpt.isErrorRecord = false;
    tcsState = tcsState();
    tcsState.xApri = zeros(N,T);
    tcsState.vApri = gX.xvar;
    tcsState.N = N;
    outIter = 5;
    [ estFin ] = EMTurboCS(gA, gX, tcsOpt, tcsState, outIter);
    estimError = (norm(hAngular - estFin.xBpost,'fro')/norm(hAngular,'fro'))^2;
    NMSETurboCS(nsnr,nm,nsim) = estimError;
    fprintf('EM-Turbo-CS: ');
    fprintf('NSIM = %d, SNR = %d, M = %d\n', nsim, SNR, M);
    fprintf(' NMSE = %.4f, %.4f dB\n', estimError, 10*log10(estimError));
    
    % Initialize Module A
    gA2 = AwgnModuleA(y, A, wvar0);
    gA2.isOrthMatrix = true;
    gA2.isEMUpatePerIter = false;
    gA2.isEMUpdateVar = false;

    % Initialize Module B
    xP01 = 0.1*ones(1,T);
    xP10 = xlambda0.*xP01./(1 - xlambda0);
    gX2 = MarkovChainModuleB(xlambda0, xlambda0, xP01, xP10, xmean0, xvar0);
    gX2.isComplex = true;
    gX2.isEMUpatePerIter = false;
    gX2.isEMUpdateLambda0 = true;
    gX2.isEMUpdateTransProb = true;
    gX2.isEMUpdateMean = false;
    gX2.isEMUpdateVar = true; 
    
    % EM structured Turbo-CS
    tcsState2 = tcsState(); % Initialize a new state for structured Turbo-CS
    tcsState2.xApri = zeros(N,T);
    tcsState2.vApri = gX2.xvar;
    tcsState2.N = N;
    [ estFin2 ] = EMTurboCS(gA2, gX2, tcsOpt, tcsState2, outIter);
    estimError = (norm(hAngular - estFin2.xBpost,'fro')/norm(hAngular,'fro'))^2;
    NMSESTCS(nsnr,nm,nsim) = estimError;
    fprintf('EM-STCS: ');
    fprintf('NSIM = %d, SNR = %d, M = %d\n', nsim, SNR, M);
    fprintf(' NMSE = %.4f, %.4f dB\n', estimError, 10*log10(estimError));
    
end
end
end

save NMSETurboCS NMSETurboCS;
save NMSESTCS NMSESTCS;

figure(1);
imagesc(10*log10(mean(NMSETurboCS,3)));
set(gca,'XTickLabel',MRange);
set(gca,'YTickLabel',SNRRange);
xlabel('Pilot Numbers');
ylabel('SNR [dB]');
%caxis([-20,0]);

figure(2);
imagesc(10*log10(mean(NMSESTCS,3)));
set(gca,'XTickLabel',MRange);
set(gca,'YTickLabel',SNRRange);
xlabel('Pilot Numbers');
ylabel('SNR [dB]');
%caxis([-20,0]);

end


