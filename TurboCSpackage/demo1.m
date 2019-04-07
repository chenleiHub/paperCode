%% demo1
% This file provides several examples of Turbo Compressed Sensing based
% algorithms with known signal prior.
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
% This simulation can repeat the results in Fig. 2 of J. Ma, X. Yuan, and
% L. Ping, "Turbo compressed sensing with partial DFT sensing matrix," IEEE
% Signal Process. Letters, vol. 22, no. 2, pp. 158-161, Feb. 2015.
% Note that the algorithm can be further accelerated by using fft.
if scenario == 1;
    
SNR = 50;
NSIM = 5;
K = 3277;
M = 5734;
N = 8192;
T = 1;
Iteration = 35;

signalPara.K = K; % Number of the non-zero elements
signalPara.N = N; % Dimension of the unknown signal
signalPara.T = T; % Dimension of the multiple measurement
signalPara.xMean = 0; % Mean value of the Gaussian distribution
signalPara.xVar = N/K; % Variance of the Gaussian distribution

NMSESig = NaN*ones(NSIM,Iteration);
for nsnr = 1:length(SNR)
for nsim = 1:NSIM
    % Generate linear model
    x = genSignalMatrix('ComplexBG', signalPara);
    A = genSensingMatrix(M, N, 'pDFT');
    z = A * x;
    wvar = norm(z(:))^2/M*10^(-SNR(nsnr)/10);
    w = sqrt(wvar)*(randn(size(z))+1i*randn(size(z)))/sqrt(2);
    y = z + w;
    
    % Initialize Module A
    gA = AwgnModuleA(y, A, wvar);
    gA.isOrthMatrix = true;

    % Initialize Module B
    xlambda = K/N;
    xmean = 0;
    xvar = N/K;
    gX = BernGaussModuleB(xlambda, xmean, xvar);
    gX.isComplex = true;
    
    optSim1 = tcsOption();
    optSim1.nit = Iteration;
    optSim1.damping = 1;
    optSim1.tol = -1;
    optSim1.xTrue = x;
    optSim1.isShowResult = true;
    optSim1.isErrorRecord = true;
    stateSim1 = tcsState();
    stateSim1.xApri = zeros(N,T);
    stateSim1.vApri = ones(1,T);
    stateSim1.N = N;
    [estFin] = tcsEst(gA, gX, optSim1, stateSim1);
    
    NMSESig(nsim,:) = estFin.errorRecord; % To generate NMSESig, opt.nit should be same as Iteration
    estimError = (norm(x - estFin.xBpost,'fro')/norm(x,'fro'))^2;
    fprintf('NSIM = %d, NMSE = %.4f, %.4f dB\n', nsim, estimError, 10*log10(estimError));
end
end

figure(1);
semilogy(mean(NMSESig, 1), 'r--');
xlabel('Iteration');
ylabel('MSE');
grid on;

end

%% Simulation 2
% This simulation can repeat the results in Fig.3 and Fig.4 of L. Chen,
% A. Liu, and X. Yuan, "Structured turbo compressed sensing for massive
% MIMO channel estimation using a Markov prior," IEEE Trans. on Veh. Tech.,
% vol. 67, no. 5, pp. 4635-4639, May 2018.

if scenario == 2

SNR = 30;
NSIM = 1000;
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
%P01 = 0.0040;% for N = 2000, K = 500, grpNum = 2
%P01 = 0.0625;% for N = 128, K = 32, grpNum = 2
%P01 = 0.03125;% for N = 128, K = 32, grpNum = 1

NMSESigTurboCS = NaN*ones(NSIM,Iteration); % Estimation error for Turbo-CS
NMSESigSTCS = NaN*ones(NSIM,Iteration); % Estimation error for Structured Turbo-CS
for nsnr = 1:length(SNR)
for nsim = 1:NSIM
    % Generation linear model
    x = genSignalMatrix('ComplexBlock', signalPara);
    A = genSensingMatrix(M, N, 'pDFT');
    z = A * x;
    wvar = norm(z(:))^2/M*10^(-SNR(nsnr)/10);
    w = sqrt(wvar)*(randn(size(z))+1i*randn(size(z)))/sqrt(2);
    y = z + w;
    
    % Initialize Module A
    gA = AwgnModuleA(y, A, wvar);
    gA.isOrthMatrix = true;
    % Initialize Module B
    xlambda = K/N;
    xmean = 0;
    xvar = N/K;
    gX = BernGaussModuleB(xlambda, xmean, xvar);
    gX.isComplex = true;
    
    opt = tcsOption();
    opt.nit = Iteration;
    opt.damping = 1;
    opt.tol = -1;
    opt.xTrue = x;
    opt.isShowResult = false;
    opt.isErrorRecord = true;
    state = tcsState();
    state.xApri = zeros(N,T);
    state.vApri = ones(1,T);
    state.N = N;
    [estFin] = tcsEst(gA, gX, opt, state);
    
    NMSESigTurboCS(nsim,:) = estFin.errorRecord; % To generate NMSESig, opt.nit should be same as Iteration.
    estimError = (norm(x - estFin.xBpost,'fro')/norm(x,'fro'))^2;
    fprintf('NSIM = %d, NMSE = %.4f, %.4f dB\n', nsim, estimError, 10*log10(estimError));
    
    % Note that if structured Turbo-CS is used directly with small N (e.g.
    % 128), the algorithm might be faced with convergence problem. 
    % ONE solution is to use damping in the algorithm, which is a very
    % effective method in the message passing or belief propagation algorithm.
    xlambda0 = xlambda;
    xP01 = blockNumber/K; % P01 = Pr(sn=0|sn-1=1)
    xP10 = blockNumber/(N-K); % P10 = Pr(sn=1|sn-1=0)
    gX2 = MarkovChainModuleB(xlambda0, xlambda, xP01, xP10, xmean, xvar);
    opt.damping = 0.85; % Silghtly damping can improve the convergence.
    state = tcsState();
    state.xApri = zeros(N,T);
    state.vApri = ones(1,T);
    state.N = N;
    [estFin2] = tcsEst(gA, gX2, opt, state);
    NMSESigSTCS(nsim,:) = estFin2.errorRecord;
    
    % ANOTHER solution (used in the paper) is using origin Turbo-CS in the
    % first iteration, then the structured Turbo-CS.
%     opt.nit = 1;
%     state = tcsState();
%     state.xApri = zeros(N,T);
%     state.vApri = ones(1,T);
%     state.N = N;
%     [estFin1] = tcsEst(gA, gX, opt, state);
%     xlambda0 = xlambda;
%     xP01 = blockNumber/K;
%     xP10 = blockNumber/(N-K);
%     gX2 = MarkovChainModuleB(xlambda0, xlambda, xP01, xP10, xmean, xvar);
%     opt.nit = Iteration - 1;
%     state.xApri = estFin1.xApri;
%     state.vApri = estFin1.vApri;
%     [estFin2] = tcsEst(gA, gX2, opt, state);
%     NMSESigSTCS(nsim,:) = [estFin1.errorRecord, estFin2.errorRecord];
    
    estimError2 = (norm(x - estFin2.xBpost,'fro')/norm(x,'fro'))^2;
    fprintf('NSIM = %d, NMSE = %.4f, %.4f dB\n', nsim, estimError2, 10*log10(estimError2));
    
end
end

figure(1);
plot(10*log10(mean(NMSESigTurboCS, 1)));
hold on;
plot(10*log10(mean(NMSESigSTCS, 1)));
xlabel('Iteration');
ylabel('NMSE/dB');
grid on;

save NMSESigTurboCS NMSESigTurboCS
save NMSESigSTCS NMSESigSTCS

end


%% Simulation 3
% This simulation can repeat the results in Fig.4 of L. Chen and X. Yuan,
% "Massive MIMO-OFDM channel estimation via structured turbo compressed
% sensing," in Proc. IEEE Int. Conf. on Commun. (ICC), 2018.

if scenario == 3;
    
SNR = 30;
NSIM = 10;
K = 64;
blockNumber = 4;
L = 16;
M = 103;
N = 256;
P = 32;
Iteration = 30;

signalPara.K = K; % Number of the non-zero elements on the angular domain
signalPara.B = blockNumber; % Number of the scatterings (blocks, same as the non-zero taps)
signalPara.L = L; % Number of the taps (all) on the delay domain
signalPara.N = N; % Dimension of the channel (same as the antennas)
signalPara.P = P; % Number of the pilots on different frequencies (> L)
signalPara.isShowChannel = true; % Plot the channel or not

NMSESigTurboCS = NaN*ones(NSIM,Iteration);
NMSESigSTCS_FS = NaN*ones(NSIM,Iteration);
NMSESigSTCS_FS_ST = NaN*ones(NSIM,Iteration);
NMSESigSTCS_FS_T = NaN*ones(NSIM,Iteration);
NMSESigSTCS_FS_B = NaN*ones(NSIM,Iteration);

for nsnr = 1:length(SNR)
for nsim = 1:NSIM
    % Generate channel
    H = genSignalMatrix('BlockChannel', signalPara);
    HFreqDomain = H.xFreqDomain;
    HFreqDomain = HFreqDomain/sqrt(var(HFreqDomain(:)));
    A = NaN*ones(M,N,P);
    Z = NaN*ones(M,P);
    for p = 1:P
        A_p = genSensingMatrix(M, N, 'pDFT_RP');
        A(:,:,p) = A_p;
        z_p = A_p*HFreqDomain(:,p);
        Z(:,p) = z_p;
    end
    wvar = norm(Z(:))^2/M/P*10^(-SNR(nsnr)/10);
    W = sqrt(wvar)*(randn(size(Z))+1i*randn(size(Z)))/sqrt(2);
    Y = Z + W;
    
    % Turbo Compressed Sensing (Turbo-CS)
    % Initialize Module A
    gA31 = AwgnModuleA(Y, A, wvar*ones(1,P));
    gA31.isOrthMatrix = true;
    % Initialize Beroulli Gaussian Module B
    xlambda = K/N*ones(1,P);
    xmean = zeros(1,P);
    index = HFreqDomain~=0;
    xvar = var(HFreqDomain(index))*ones(1,P);
    gX31 = BernGaussModuleB(xlambda, xmean, xvar);
    gX31.isComplex = true;   
    % Algorithm settings
    opt3 = tcsOption();
    opt3.nit = Iteration;
    opt3.damping = 1;
    opt3.tol = -1;
    opt3.xTrue = HFreqDomain;
    opt3.isShowResult = false;
    opt3.isErrorRecord = true;
    stateSim31 = tcsState();
    stateSim31.xApri = zeros(N,P);
    stateSim31.vApri = ones(1,P);
    stateSim31.N = N;
    [estFin31] = tcsEst(gA31, gX31, opt3, stateSim31);
    NMSESigTurboCS(nsim,:) = estFin31.errorRecord;
    estimError = (norm(HFreqDomain - estFin31.xBpost,'fro')/norm(HFreqDomain,'fro'))^2;
    fprintf('Turbo-CS: NSIM = %d, NMSE = %.4f, %.4f dB\n', nsim, estimError, 10*log10(estimError));
    
    % Structured Turbo Compressed Sensing with frequency support (STCS-FS)
    % Initialize Module A
    gA32 = AwgnModuleA(Y, A, wvar*ones(1,P));
    gA32.isOrthMatrix = true;
    % Initialize Frequency Support Module B
    xlambda0 = K/N;
    xsparsityAvg = K/N*ones(1,P);
    xP01 = blockNumber/K; % P01 = Pr(sn=0|sn-1=1)
    xP10 = blockNumber/(N-K); % P10 = Pr(sn=1|sn-1=0)
    xmean = zeros(1,P);
    index = HFreqDomain~=0;
    xvar = var(HFreqDomain(index))*ones(1,P);
    gX32 = FreqSuppModuleB(xlambda0, xsparsityAvg, xP01, xP10, xmean, xvar);
    gX32.isComplex = true;
    % Algorithm settings
    stateSim32 = tcsState();
    stateSim32.xApri = zeros(N,P);
    stateSim32.vApri = ones(1,P);
    stateSim32.M = M;
    stateSim32.N = N;
    stateSim32.T = P;
    [estFin32] = tcsEst(gA32, gX32, opt3, stateSim32);
    NMSESigSTCS_FS(nsim,:) = estFin32.errorRecord;
    estimError = (norm(HFreqDomain - estFin32.xBpost,'fro')/norm(HFreqDomain,'fro'))^2;
    fprintf('STCS-FS: NSIM = %d, NMSE = %.4f, %.4f dB\n', nsim, estimError, 10*log10(estimError));
    
    % Structured Turbo Compressed Sensing with frequency support and delay
    % dmoain soft-thresholding (STCS-FS-ST)
    % Initialize Module A
    gA33 = AwgnModuleA(Y, A, wvar*ones(1,P));
    gA33.isOrthMatrix = true;
    % Initialize Frequency Support Module B
    xlambda0 = K/N;
    xsparsityAvg = K/N*ones(1,P);
    xP01 = blockNumber/K; % P01 = Pr(sn=0|sn-1=1)
    xP10 = blockNumber/(N-K); % P10 = Pr(sn=1|sn-1=0)
    xmean = zeros(1,P);
    index = HFreqDomain~=0;
    xvar = var(HFreqDomain(index))*ones(1,P);
    gX33 = FreqSuppModuleB(xlambda0, xsparsityAvg, xP01, xP10, xmean, xvar);
    gX33.maxL = L;
    gX33.isComplex = true;
    gX33.isDelayDenoiser = true;
    gX33.isSoftThres = true;
    gX33.isTruncation = false;
    % Algorithm settings
    opt3.isDelayDenoiser = true;
    stateSim33 = tcsState();
    stateSim33.xApri = zeros(N,P);
    stateSim33.vApri = ones(1,P);
    stateSim33.M = M;
    stateSim33.N = N;
    stateSim33.T = P;
    [estFin33] = tcsEst(gA33, gX33, opt3, stateSim33);
    NMSESigSTCS_FS_ST(nsim,:) = estFin33.errorRecord;
    estimError = (norm(HFreqDomain - estFin33.xApri,'fro')/norm(HFreqDomain,'fro'))^2;
    fprintf('STCS-FS-ST: NSIM = %d, NMSE = %.4f, %.4f dB\n', nsim, estimError, 10*log10(estimError));
    
    % Structured Turbo Compressed Sensing with frequency support and delay
    % dmoain truncation (STCS-FS-T)
    % Initialize Module A
    gA34 = AwgnModuleA(Y, A, wvar*ones(1,P));
    gA34.isOrthMatrix = true;
    % Initialize Frequency Support Module B
    xlambda0 = K/N;
    xsparsityAvg = K/N*ones(1,P);
    xP01 = blockNumber/K; % P01 = Pr(sn=0|sn-1=1)
    xP10 = blockNumber/(N-K); % P10 = Pr(sn=1|sn-1=0)
    xmean = zeros(1,P);
    index = HFreqDomain~=0;
    xvar = var(HFreqDomain(index))*ones(1,P);
    gX34 = FreqSuppModuleB(xlambda0, xsparsityAvg, xP01, xP10, xmean, xvar);
    gX34.maxL = L;
    gX34.isComplex = true;
    gX34.isDelayDenoiser = true;
    gX34.isSoftThres = false;
    gX34.isTruncation = true;
    % Algorithm settings
    opt3.isDelayDenoiser = true;
    stateSim34 = tcsState();
    stateSim34.xApri = zeros(N,P);
    stateSim34.vApri = ones(1,P);
    stateSim34.M = M;
    stateSim34.N = N;
    stateSim34.T = P;
    [estFin34] = tcsEst(gA34, gX34, opt3, stateSim34);
    NMSESigSTCS_FS_T(nsim,:) = estFin34.errorRecord;
    estimError = (norm(HFreqDomain - estFin34.xApri,'fro')/norm(HFreqDomain,'fro'))^2;
    fprintf('STCS-FS-T: NSIM = %d, NMSE = %.4f, %.4f dB\n', nsim, estimError, 10*log10(estimError));
    
    % Structured Turbo Compressed Sensing with frequency support and delay
    % dmoain soft-thresholding and truncation (STCS-FS-B)
    % Initialize Module A
    gA35 = AwgnModuleA(Y, A, wvar*ones(1,P));
    gA35.isOrthMatrix = true;
    % Initialize Frequency Support Module B
    xlambda0 = K/N;
    xsparsityAvg = K/N*ones(1,P);
    xP01 = blockNumber/K; % P01 = Pr(sn=0|sn-1=1)
    xP10 = blockNumber/(N-K); % P10 = Pr(sn=1|sn-1=0)
    xmean = zeros(1,P);
    index = HFreqDomain~=0;
    xvar = var(HFreqDomain(index))*ones(1,P);
    gX35 = FreqSuppModuleB(xlambda0, xsparsityAvg, xP01, xP10, xmean, xvar);
    gX35.maxL = L;
    gX35.isComplex = true;
    gX35.isDelayDenoiser = true;
    gX35.isSoftThres = true;
    gX35.isTruncation = true;
    % Algorithm settings
    opt3.isDelayDenoiser = true;
    stateSim35 = tcsState();
    stateSim35.xApri = zeros(N,P);
    stateSim35.vApri = ones(1,P);
    stateSim35.M = M;
    stateSim35.N = N;
    stateSim35.T = P;
    [estFin35] = tcsEst(gA35, gX35, opt3, stateSim35);
    NMSESigSTCS_FS_B(nsim,:) = estFin35.errorRecord;
    estimError = (norm(HFreqDomain - estFin35.xApri,'fro')/norm(HFreqDomain,'fro'))^2;
    fprintf('STCS-FS-B: NSIM = %d, NMSE = %.4f, %.4f dB\n', nsim, estimError, 10*log10(estimError));
    
end
end

figure(1);
plot(10*log10(mean(NMSESigTurboCS, 1)),'-*');
hold on;
plot(10*log10(mean(NMSESigSTCS_FS, 1)),'-*');
plot(10*log10(mean(NMSESigSTCS_FS_ST, 1)),'-*');
plot(10*log10(mean(NMSESigSTCS_FS_T, 1)),'-*');
plot(10*log10(mean(NMSESigSTCS_FS_B, 1)),'-*');
xlabel('Iteration');
ylabel('NMSE/dB');
grid on;

end


%% Simulation 4
% This simulation can repeat the results in figures of X. Kuai, L. Chen, X.
% Yuan and A. Liu, "Structured turbo compressed sensing for downlink
% massive MIMO-OFDM channel estimation," arXiv preprint arXiv:1811.03316.

if scenario == 4;
    
SNR = 10;
NSIM = 10;
K = 64;
blockNumber = 4;
L = 16;
M = 103;
N = 256;
P = 32;
Iteration = 30;

signalPara.K = K; % Number of the non-zero elements on the angular domain
signalPara.B = blockNumber; % Number of the scatterings (blocks, same as the non-zero taps)
signalPara.L = L; % Number of the taps (all) on the delay domain
signalPara.N = N; % Dimension of the channel (same as the antennas)
signalPara.P = P; % Number of the pilots on different frequencies (> L)
signalPara.isShowChannel = false; % Plot the channel or not

NMSESigSTCS_DS = NaN*ones(NSIM,Iteration);

for nsnr = 1:length(SNR)
for nsim = 1:NSIM
    % Generate channel
    H = genSignalMatrix('BlockChannel', signalPara);
    HDelayDomain = H.xDelayDomain;
    A = NaN*ones(M,N,P);
    Z = NaN*ones(M,P);
    A_p = genSensingMatrix(M, N, 'pDFT_RP');
    for p = 1:P
        A(:,:,p) = A_p;
        z_p = A_p*HDelayDomain(:,p);
        Z(:,p) = z_p;
    end
    wvar = norm(Z(:))^2/M/P*10^(-SNR(nsnr)/10);
    W = sqrt(wvar)*(randn(size(Z))+1i*randn(size(Z)))/sqrt(2);
    Y = Z + W;
    
    % Structured Turbo Compressed Sensing with delay support (STCS-DS)
    % Initialize Module A
    gA4 = AwgnModuleA(Y, A, wvar*ones(1,P));
    gA4.isOrthMatrix = true;
    % Initialize Delay Support Module B
    xlambda0 = K/blockNumber/N*ones(1,P);
    xsparsityAvg = K/blockNumber/N*ones(1,P);
    xP01 = 1/(K/blockNumber)*ones(1,P);
    xP10 = 1/(N-(K/blockNumber))*ones(1,P);
    xmean = zeros(1,P);
    index = HDelayDomain~=0;
    xvar = var(HDelayDomain(index))*ones(1,P);
    xjudge = blockNumber/P*ones(1,P);
    %{
    for p = 1:P
        index = find(HDelayDomain~=0);
        if isempty(index)
            xvar(1,p) = 10*var(HDelayDomain(index));
            xjudge(1,p) = 0.01;
        else
            xvar(1,p) = var(HDelayDomain(index));
            xjudge(1,p) = blockNumber/P;
        end
    end
    %}
    gX4 = DelaySuppModuleB(xlambda0, xsparsityAvg, xP01, xP10, xmean, xvar, xjudge);
    gX4.isComplex = true;
    % Algorithm settings
    opt4 = tcsOption();
    opt4.nit = Iteration;
    opt4.damping = 0.95;
    opt4.tol = -1;
    opt4.xTrue = HDelayDomain;
    opt4.isShowResult = false;
    opt4.isErrorRecord = true;
    opt4.isDelayDenoiser = false;
    stateSim4 = tcsState();
    stateSim4.xApri = zeros(N,P);
    stateSim4.vApri = ones(1,P);
    %{
    vApri0 = NaN*ones(1,P);
    for p = 1:P
        index = find(HDelayDomain~=0);
        if isempty(index)
            vApri0(1,p) = 0.1;
        else
            vApri0(1,p) = 1;
        end
    end 
    stateSim4.vApri = vApri0;
    %}
    stateSim4.M = M;
    stateSim4.N = N;
    stateSim4.T = P;
    [estFin4] = tcsEst(gA4, gX4, opt4, stateSim4);
    
    NMSESigSTCS_DS(nsim,:) = estFin4.errorRecord;
    estimError = (norm(HDelayDomain - estFin4.xBpost,'fro')/norm(HDelayDomain,'fro'))^2;
    fprintf('STCS-DS: NSIM = %d, NMSE = %.4f, %.4f dB\n', nsim, estimError, 10*log10(estimError));
end
end

figure(1);
plot(10*log10(mean(NMSESigSTCS_DS, 1)),'-*');
xlabel('Iteration');
ylabel('NMSE/dB');
grid on;
    
end












