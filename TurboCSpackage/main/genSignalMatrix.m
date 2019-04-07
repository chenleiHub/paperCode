function [ x ] = genSignalMatrix( signalPrior, signalPara )
% genSignalMatrix
% 
%set(0,'defaultAxesFontName', '<ËÎÌו>'); % chinese characters

switch signalPrior
    % Real Bernoulli Gaussian distribution
    case 'BG'
        % The dimension of all the parameter is 1x1.
        K = signalPara.K; % Number of the non-zero elements
        N = signalPara.N; % Dimension of the unknown signal
        T = signalPara.T; % Dimension of the multiple measurement
        xMean = signalPara.xMean; % Mean value of the Gaussian distribution
        xVar = signalPara.xVar; % Variance of the Gaussian distribution
        
        x = NaN*ones(N,T);
        for t = 1:T
            loc = randperm(N, K);
            index = zeros(N,1);
            index(loc) = 1;
            xt = sqrt(xVar)*randn(N,1) + xMean;
            x(:,t) = xt.*index;
        end
    
    % Complex Bernoulli Gaussian distribution 
    case 'ComplexBG'
        % The dimension of all the parameter is 1x1.
        K = signalPara.K; % Number of the non-zero elements
        N = signalPara.N; % Dimension of the unknown signal
        T = signalPara.T; % Dimension of the multiple measurement
        xMean = signalPara.xMean; % Mean value of the Gaussian distribution
        xVar = signalPara.xVar; % Variance of the Gaussian distribution
        
        x = NaN*ones(N,T);
        for t = 1:T
            loc = randperm(N, K);
            index = zeros(N,1);
            index(loc) = 1;
            xt = sqrt(xVar)*(randn(N,1) + 1i*randn(N,1))/sqrt(2) + xMean;
            x(:,t) = xt.*index;
        end
    
    % Block sparsity 
    % In block sparsity, the non-zero elements cluster into several blocks.
    % Note that the method is used to generate Markov chain when N is
    % small. P01 = B/K and P10 = B/(N-K) are accurate.
    case 'Block'
        % The dimension of all the parameter is 1x1.
        K = signalPara.K; % Number of the non-zero elements
        B = signalPara.blockNumber; % Non-zero block numbers
        N = signalPara.N; % Dimension of the unknown signal
        T = signalPara.T; % Dimension of the multiple measurement
        xMean = signalPara.xMean; % Mean value of the Gaussian distribution
        xVar = signalPara.xVar; % Variance of the Gaussian distribution
        
        x = NaN*ones(N,T);
        for t = 1:T
            index = zeros(N,1);
            r = abs(randn(B,1)) + 1; % Bx1
            % Divide K into B blocks. The elements number of each block is
            % in blockElements.
            blockElements = round(r/sum(r,1)*K); % Bx1
            blockElements(B,1) = K - sum(blockElements(1:B-1,1));
            % Divide N into B parts. The percent is the same as K (or
            % r/sum(r,1)), then each block can be well located into
            % corresponding part.
            blockParts = round(blockElements/K*N); % Bx1
            blockParts(B,1) = N - sum(blockParts(1:B-1,1));
            blockLocations = cumsum(blockParts,1); % Bx1
            
            % Insert each block into corresponding part.
            for ii = 1:B
                loc = randperm(blockParts(ii,1) - blockElements(ii,1) - 2, 1);
                xPart = zeros(blockParts(ii,1),1);
                xPart(loc + 1:loc + blockElements(ii,1)) = 1;
                index(blockLocations(ii,1) - blockParts(ii,1) + 1:blockLocations(ii,1)) = xPart;
            end
            xt = sqrt(xVar)*randn(N,1) + xMean;
            x(:,t) = xt.*index;
        end
    
    % Complex block sparsity
    % More information can be found under BLOCK SPARSITY.
    case 'ComplexBlock'
        % The dimension of all the parameter is 1x1.
        K = signalPara.K; % Number of the non-zero elements
        B = signalPara.blockNumber; % Non-zero block numbers
        N = signalPara.N; % Dimension of the unknown signal
        T = signalPara.T; % Dimension of the multiple measurement
        xMean = signalPara.xMean; % Mean value of the Gaussian distribution
        xVar = signalPara.xVar; % Variance of the Gaussian distribution
        
        x = NaN*ones(N,T);
        for t = 1:T
            index = zeros(N,1);
            r = abs(randn(B,1)) + 1; % Bx1
            % Divide K into B blocks. The elements number of each block is
            % in blockElements.
            blockElements = round(r/sum(r,1)*K); % Bx1
            blockElements(B,1) = K - sum(blockElements(1:B-1,1));
            % Divide N into B parts. The percent is the same as K (or
            % r/sum(r,1)), then each block can be well located into
            % corresponding part.
            blockParts = round(blockElements/K*N); % Bx1
            blockParts(B,1) = N - sum(blockParts(1:B-1,1));
            blockLocations = cumsum(blockParts,1); % Bx1
            
            % Insert each block into corresponding part.
            for ii = 1:B
                loc = randperm(blockParts(ii,1) - blockElements(ii,1) - 2, 1);
                xPart = zeros(blockParts(ii,1),1);
                xPart(loc + 1:loc + blockElements(ii,1)) = 1;
                index(blockLocations(ii,1) - blockParts(ii,1) + 1:blockLocations(ii,1)) = xPart;
            end
            xt = sqrt(xVar)*(randn(N,1) + 1i*randn(N,1))/sqrt(2) + xMean;
            x(:,t) = xt.*index;
        end
    
    % Block sparsity on multiple vector
    % This generator is used to simulate massive MIMO-OFDM sparse channel.
    % The channel is generated on the delay domain first with several
    % non-zero taps. Then the channel is transformed to frequency domain by
    % DFT. Note that this is not a scattering channel model, and it is only
    % used for theoretical analysis. In short, it is not practical!
    case 'BlockChannel'
        % The dimension of all the parameter is 1x1.
        K = signalPara.K; % Number of the non-zero elements on the angular domain
        B = signalPara.B; % Number of the scatterings (blocks, same as the non-zero taps)
        L = signalPara.L; % Number of the taps (all) on the delay domain
        N = signalPara.N; % Dimension of the channel (same as the antennas)
        P = signalPara.P; % Number of the pilots on different frequencies (> L)
        isShowChannel = signalPara.isShowChannel; % Plot the channel or not
        
        % In the default settings, K = 64, B = 4, L = 16, N = 256, P = 32
        xDelayDomain = zeros(N,P); % elements in L+1:P is zero
        %xFreqDomain = zeros(N,P);
        
        % Generate non-zero taps
        blockElements = round(K/B); % The elements of each block is set as the same for simplification.
        blockParts = round(N/B);
        xDelayTmp = zeros(N,B);
        for ii = 1:B
            loc = randperm(blockParts - blockElements, 1);
            loc = loc + (ii - 1)*blockParts;
            xDelayTmp(loc:loc + blockElements - 1, ii) = 1;
        end
        
        % Locate the non-zero taps and generate delay domain channel
        loc = randperm(L, B);
        for ii = 1:B
            xDelayDomain(:,loc(ii)) = xDelayTmp(:,ii);
        end
        xDelayDomain = xDelayDomain.*(randn(N,P) + 1i*randn(N,P))/sqrt(2);
        
        % Generate frequency domain channel by using DFT
        F = dftmtx(P)/sqrt(P);
        xFreqDomain = xDelayDomain*F.';
        
        x.xFreqDomain = xFreqDomain;
        x.xDelayDomain = xDelayDomain;

        %size(find(xDelayDomain(:,5)~=0))
        
        if isShowChannel
            figure(1);
            subplot(121);
            %mesh(abs(xFreqDomain)/max(max(abs(xFreqDomain))));
            % Figure generated by surf and bar3 is not clear. 
            % 'shading flat' or 'shading interp' operation is necessary.
            surf(abs(xFreqDomain)/max(max(abs(xFreqDomain))));
            %bar3(abs(xFreqDomain)/max(max(abs(xFreqDomain))));
            shading interp;
            xlabel({'P pilot frequencies'; '(frequency domain)'});
            ylabel({'N antennas'; '(angular domain)'});
            zlabel('Normalized channel gains');
            %axis([1,P,1,N,0,1]); % show x-y-z dimension
            axis([1,P,1,N]); % show x-y dimension
            grid on;
            %view(-37.5,30);
            colorbar;
            
            %figure(2);
            subplot(122);
            %mesh(abs(xDelayDomain)/max(max(abs(xDelayDomain))));
            surf(abs(xDelayDomain)/max(max(abs(xDelayDomain))));
            %bar3(abs(xDelayDomain)/max(max(abs(xDelayDomain))));
            shading interp;
            xlabel({'P delay taps'; '(delay domain)'}); % Elements in L+1:P are zeros.
            ylabel({'N antennas'; '(angular domain)'});
            zlabel('Normalized channel gains');
            %axis([1,P,1,N,0,1]); % show x-y-z dimension
            axis([1,P,1,N]); % show x-y dimension
            grid on;
            %view(-37.5,30);
            colorbar;
        end
    
    % Generate massive MIMO channel for uniform linear array (ULA) and
    % uniform plannar array (UPA).
    case 'FlatFadingChannel'
        % The dimension of all the parameter is 1x1.
        NT1 = signalPara.NT1; % Number of antennas at base station 
        P = signalPara.P; % Number of scattering paths
        dim = signalPara.dim; % Dimension of the antenna array, 1 for ULA, 2 for UPA.
        isShowChannel = signalPara.isShowChannel; % Plot the channel or not
        if dim == 1 % For uniform linear array
            l = 0:1:NT1-1;
            l = l';
            theta = (rand(1,P) - 0.5)*pi; % Angle of departure
            phi = 0.5*sin(theta);
            beta = (randn(P,1) + 1i*randn(P,1))/sqrt(2); % Path gain
            a = exp(2i*pi*(l*phi)) * beta;
            h = sum(a,2);
            h = h/sqrt(var(h));
            F = dftmtx(NT1)/sqrt(NT1);
            t = F'*h;
            x.spatial = h;
            x.angular = t;
        end
        
        if signalPara.dim == 2
        end
        
        if isShowChannel
            if dim == 1
                stem(abs(t)/max(abs(t)));
                axis([0, NT1, 0, 1]);
                xlabel('Angular domain');
                ylabel('Channel gains');
                grid on;
            end
        end
        
    case 'FreqSelectChannel'
        
    otherwise
        error('Unrecognized signal prior.');   
     
end

end

