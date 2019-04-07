classdef FreqSuppModuleB < handle
    % FreqSuppModuleB
    % In this class, NxT unknown matrix is a common Markov chain in T dimensions. 
    
    properties
        xlambda0; % 1x1 sparsity of the first variable node
        xsparsityAvg; % 1x1 average sparsity of the NxT unknown matrix
        xP01; % 1x1 transition probability of the NxT unknown matrix, P01 = Pr(sn=0|sn-1=1)
        xP10; % 1x1 transition probability of the NxT unknown matrix, P10 = Pr(sn=1|sn-1=0)
        sparsityMax = 1-1e-11; % maximum value of the sparsity and binary probability
        sparsityMin = 1e-11; % minimum value of the sparsity and binary probability
        logMax = 20; % maximum value on the exponent
        logMin = -20; % minimum value on the exponent
        xmean; % 1xT mean of the NxT unknown matrix
        xmeanMax = Inf; % maximum mean value 
        xmeanMin = -Inf; % minimum mean value
        xvar; % 1xT variance of the NxT unknown matrix
        xvarMax = Inf; % maximum variance value
        xvarMin = 1e-11; % minimum variance value
        maxL; % max delay tap for truncation denoiser
        isComplex = false; % the value is complex or not
        isDelayDenoiser = false; % the denoiser in the delay domain is used or not
        isSoftThres = false; % the complex soft-thresholding denoiser is used or not
        isTruncation = false; % the truncation denoiser is used or not
        isEMUpatePerIter = false; % the expectation maximization is used per iteration or not
        isEMUpdateLambda0 = false; % the expectation maximization is used for sparsity update or not
        isEMUpdateTransProb = false; % the expectation maximization is used for transition probability update or not
        isEMUpdateMean = false; % the expectation maximization is used for mean update or not
        isEMUpdateVar = false; % the expectation maximization is used for variance update or not  
    end
    
    methods
        % Constructor
        function obj = FreqSuppModuleB(xlambda0, xsparsityAvg, xP01, xP10, xmean, xvar)
            if nargin > 0
                obj.xlambda0 = xlambda0;
                obj.xsparsityAvg = xsparsityAvg;
                obj.xP01 = xP01;
                obj.xP10 = xP10;
                obj.xmean = xmean;
                obj.xvar = xvar;
            end
        end
        
        % Estimator
        function [xBpost, vBpost, CBpost] = estim(obj, xBpri, vBpri)
            [N, T] = size(xBpri);
            meanMatrix = ones(N,1)*obj.xmean; % NxT matrix
            varMatrix = ones(N,1)*obj.xvar; % NxT matrix
            vBpriMatrix = ones(N,1)*vBpri; % NxT matrix
            CBpost = [];
            
            nuMatrix = vBpriMatrix.*varMatrix./(vBpriMatrix + varMatrix);
            gammaMatrix = nuMatrix.*(xBpri./vBpriMatrix + meanMatrix./varMatrix);
            if obj.isComplex
                tmp0 = abs(xBpri - meanMatrix).^2./(vBpriMatrix + varMatrix) - abs(xBpri).^2./vBpriMatrix;
                tmp0 = varMatrix./nuMatrix.*exp(tmp0);
            else
                tmp0 = abs(xBpri - meanMatrix).^2./(vBpriMatrix + varMatrix) - abs(xBpri).^2./vBpriMatrix;
                tmp0 = sqrt(varMatrix./nuMatrix).*exp(0.5*tmp0);
            end
            
            % piIn calculation
            piInMatrix = 1./(1 + tmp0);
            piInMatrix = min(max(piInMatrix, obj.sparsityMin), obj.sparsityMax);
            
            % Markov chain forward and backward propagation
            lambdaFwd = NaN*ones(N,1);
            lambdaBwd = NaN*ones(N,1);
            lambdaFwd(1,1) = obj.xlambda0;
            lambdaBwd(N,1) = 0.5;
            P01 = obj.xP01;
            P10 = obj.xP10;
            piInLogSum = sum(log(piInMatrix),2); % Nx1 sum matrix of log(piInMatrix)
            %piInLogSum = min(max(piInLogSum, obj.logMin), obj.logMax);
            oneMinusPiInLogSum = sum(log(1 - piInMatrix),2); % Nx1 sum matrix of log(1 - piInMatrix)
            %oneMinusPiInLogSum = min(max(oneMinusPiInLogSum, obj.logMin), obj.logMax);
            
            % Forward propagation
            for n = 2:1:N
                tmp = log(lambdaFwd(n-1,1)) + piInLogSum(n-1,1) ...
                    - log(1 - lambdaFwd(n-1,1)) - oneMinusPiInLogSum(n-1,1);
                tmp = min(max(tmp, obj.logMin), obj.logMax);
                tmp = exp(tmp); % 1x1 value
                lambdaFwd0 = (P10 + tmp*(1 - P01))/(1 + tmp);
                %lambdaFwd(n,:) = lambdaFwd0;
                lambdaFwd(n,1) = min(max(lambdaFwd0, obj.sparsityMin), obj.sparsityMax);
            end
            
            % Backward propagation
            for n = N-1:-1:1
                tmp = log(lambdaBwd(n+1,1)) + piInLogSum(n+1,1) ...
                    - log(1 - lambdaBwd(n+1,1)) - oneMinusPiInLogSum(n+1,1);
                tmp = min(max(tmp, obj.logMin), obj.logMax);
                tmp = exp(tmp); % 1x1 value
                lambdaBwd0 = (P01 + tmp*(1 - P01))/((1 - P10 + P01) + tmp*(1 - P01 + P10));
                %lambdaBwd(n,:) = lambdaBwd0;
                lambdaBwd(n,1) = min(max(lambdaBwd0, obj.sparsityMin), obj.sparsityMax);
            end
            
            % piOut calculation
            lambdaFwdMatrixLog = log(lambdaFwd)*ones(1,T); % NxT ln(lambdaFwd) matrix
            oneMinusLambdaFwdMatrixLog = log(1 - lambdaFwd)*ones(1,T); % NxT ln(1-lambdaFwd) matrix
            lambdaBwdMatrixLog = log(lambdaBwd)*ones(1,T); % NxT ln(lambdaBwd) matrix
            oneMinusLambdaBwdMatrixLog = log(1 - lambdaBwd)*ones(1,T); % NxT ln(1-lambdaBwd) matrix
            piInLogSumMatrix = piInLogSum*ones(1,T); % NxT sum_1_T(ln(piIn)) matrix 
            oneMinusPiInLogSumMatrix = oneMinusPiInLogSum*ones(1,T); % NxT sum_1_T(ln(1-piIn)) matrix
            
            tmp = lambdaFwdMatrixLog + lambdaBwdMatrixLog + piInLogSumMatrix - log(piInMatrix) ...
                - oneMinusLambdaFwdMatrixLog - oneMinusLambdaBwdMatrixLog ...
                - oneMinusPiInLogSumMatrix + log(1 - piInMatrix); % NxT matrix
            tmp = min(max(tmp, obj.logMin), obj.logMax);
            tmp = exp(tmp); 
            piOutMatrix = tmp./(1 + tmp);
            
            piiMatrix = piInMatrix.*piOutMatrix./((1 - piInMatrix).*(1 - piOutMatrix) + piInMatrix.*piOutMatrix);
            %piiMatrix = (1 - piOutMatrix)./piOutMatrix.*tmp0;
            %piiMatrix =  1./(1 + piiMatrix);
                    
            % xBpost calculation
            xBpost = gammaMatrix.*piiMatrix;
            
            % vBpost calculation
            vBpost = sum(piiMatrix.*(abs(gammaMatrix).^2 + nuMatrix) - abs(xBpost).^2, 1)/N;

            % Delay domain denoiser 
            if obj.isDelayDenoiser
                F = dftmtx(T)/sqrt(T);
                xBpostDelay = xBpost*(F').';
                % Complex soft-thresholding denoiser
                if obj.isSoftThres
                    threshold = sqrt(mean(vBpost, 2)); % 1x1 value
                    xBpostDelay = obj.complexSoftThres(xBpostDelay, threshold);
                end
                % Truncation operation
                if obj.isTruncation
                    maxL0 = obj.maxL;
                    xBpostDelay(:, maxL0+1:1:end) = 0;
                end
                xBpost = xBpostDelay*F.'; % NxT denoised matrix
                vBpost = NaN*ones(1,T);
            end
            
            % Expectation maximization
            if obj.isEMUpatePerIter
                if obj.isEMUpdateLambda0
                    % 1x1 Sparsity of the first variable
                    tmp = log(lambdaFwd(1,1)) + log(lambdaBwd(1,1)) + piInLogSum(1,1) ...
                        - log(1 - lambdaFwd(1,1)) - log(1 - lambdaBwd(1,1)) ...
                        - oneMinusPiInLogSum(1,1);
                    tmp = min(max(tmp, obj.logMin), obj.logMax);
                    tmp = exp(tmp);
                    lambda0 = tmp/(1 + tmp);
                    obj.xlambda0 = min(max(lambda0, obj.sparsityMin), obj.sparsityMax);
                end
                
                if obj.isEMUpdateTransProb
                    % Nx1 Posterior probability of s, from 1 to N
                    tmp = log(lambdaFwd) + log(lambdaBwd) + piInLogSum ...
                        - log(1 - lambdaFwd) - log(1 - lambdaBwd) - oneMinusPiInLogSum;
                    tmp = min(max(tmp, obj.logMin), obj.logMax);
                    tmp = exp(tmp); % Nx1
                    PS1 = tmp./(1 + tmp); % Nx1 Pr(sn=1|Y) matrix
                    
                    % N-1x1 Transaction probability, from 2 to N
                    tmp = log(lambdaFwd(1:N-1,1)) + piInLogSum(1:N-1,1) ...
                        + log(lambdaBwd(2:N,1)) + piInLogSum(2:N,1) ...
                        - log(1 - lambdaFwd(1:N-1,1)) - oneMinusPiInLogSum(1:N-1,1) ...
                        - log(1 - lambdaBwd(2:N,1)) - oneMinusPiInLogSum(2:N,1);
                    tmp = min(max(tmp, obj.logMin), obj.logMax);
                    tmp = exp(tmp); % N-1x1 matrix
                    
                    tmp2 = log(lambdaFwd(1:N-1,1)) + piInLogSum(1:N-1,1) ...
                        - log(1 - lambdaFwd(1:N-1,1)) - oneMinusPiInLogSum(1:N-1,1);
                    tmp2 = min(max(tmp2, obj.logMin), obj.logMax);
                    tmp2 = exp(tmp2); % N-1x1 matrix
                    
                    tmp3 = log(lambdaBwd(2:N,1)) + piInLogSum(2:N,1) ...
                        - log(1 - lambdaBwd(2:N,1)) - oneMinusPiInLogSum(2:N,1);
                    tmp3 = min(max(tmp3, obj.logMin), obj.logMax);
                    tmp3 = exp(tmp3); % N-1x1 matrix
                    
                    PS1S1 = tmp*(1 - P01)./((1 - P10) + tmp2*P01 + tmp3*P10 ...
                        + tmp*(1 - P01)); % N-1x1 Pr(sn=1,sn-1=1|Y) matrix
                    
                    P01New = sum(PS1(1:N-1,1) - PS1S1, 1)/sum(PS1(1:N-1,1), 1); % 1x1 matrix
                    %obj.xP01 = P01New;
                    obj.xP01 = min(max(P01New, obj.sparsityMin), obj.sparsityMax);
                    
                    P10New = sum(PS1(2:N,1) - PS1S1, 1)/((N-1) - sum(PS1(1:N-1,1),1)); % 1x1 matrix
                    %obj.xP10 = P10New;
                    obj.xP10 = min(max(P10New, obj.sparsityMin), obj.sparsityMax);
                    
                    %xsparsityAvg0 = mean(piiMatrix, 1); % 1xT matrix
                    %obj.xsparsityAvg = xsparsityAvg0;
                end
                
                if obj.isEMUpdateMean
                    mean0 = sum(gammaMatrix.*piiMatrix, 1)./sum(piiMatrix, 1);
                    if isreal(mean0)
                        obj.xmean = min(max(mean0, obj.xmeanMin), obj.xmeanMax);
                    else
                        obj.xmean = mean0;
                    end     
                end
                
                if obj.isEMUpdateVar
                    var0 = sum(piiMatrix.*(abs(gammaMatrix - meanMatrix).^2 + nuMatrix), 1)./sum(piiMatrix, 1);
                    obj.xvar = min(max(var0, obj.xvarMin), obj.xvarMax);
                end
            end
        end
        
        % Expectation maximization update
        function [] = expectMaxUpdate(obj, xBpri, vBpri)
            [N, T] = size(xBpri);
            meanMatrix = ones(N,1)*obj.xmean; % NxT matrix
            varMatrix = ones(N,1)*obj.xvar; % NxT matrix
            vBpriMatrix = ones(N,1)*vBpri; % NxT matrix
            
            nuMatrix = vBpriMatrix.*varMatrix./(vBpriMatrix + varMatrix);
            gammaMatrix = nuMatrix.*(xBpri./vBpriMatrix + meanMatrix./varMatrix);
            if obj.isComplex
                tmp = abs(xBpri - meanMatrix).^2./(vBpriMatrix + varMatrix) - abs(xBpri).^2./vBpriMatrix;
                tmp = varMatrix./nuMatrix.*exp(tmp);
            else
                tmp = abs(xBpri - meanMatrix).^2./(vBpriMatrix + varMatrix) - abs(xBpri).^2./vBpriMatrix;
                tmp = sqrt(varMatrix./nuMatrix).*exp(0.5*tmp);
            end
            
            % piIn calculation
            piInMatrix = 1./(1 + tmp);
            piInMatrix = min(max(piInMatrix, obj.sparsityMin), obj.sparsityMax);
            
            % Markov chain forward and backward propagation
            lambdaFwd = NaN*ones(N,1);
            lambdaBwd = NaN*ones(N,1);
            lambdaFwd(1,1) = obj.xlambda0;
            lambdaBwd(N,1) = 0.5;
            P01 = obj.xP01;
            P10 = obj.XP10;
            piInLogSum = sum(log(piInMatrix),2); % Nx1 sum matrix of log(piInMatrix)
            oneMinusPiInLogSum = sum(log(1 - piInMatrix),2); % Nx1 sum matrix of log(1 - piInMatrix)
            
            % Forward propagation
            for n = 2:1:N
                tmp = log(lambdaFwd(n-1,1)) + piInLogSum(n-1,1) ...
                    - log(1 - lambdaFwd(n-1,1)) - oneMinusPiInLogSum(n-1,1);
                tmp = min(max(tmp, obj.logMin), obj.logMax);
                tmp = exp(tmp); % 1x1 value
                lambdaFwd0 = (P10 + tmp*(1 - P01))/(1 + tmp);
                %lambdaFwd(n,:) = lambdaFwd0;
                lambdaFwd(n,:) = min(max(lambdaFwd0, obj.sparsityMin), obj.sparsityMax);
            end
            
            % Backward propagation
            for n = N-1:-1:1
                tmp = log(lambdaBwd(n+1,1)) + piInLogSum(n+1,1) ...
                    - log(1 - lambdaBwd(n+1,1)) - oneMinusPiInLogSum(n+1,1);
                tmp = min(max(tmp, obj.logMin), obj.logMax);
                tmp = exp(tmp); % 1x1 value
                lambdaBwd0 = (P01 + tmp*(1 - P01))/((1 - P10 + P01) + tmp*(1 - P01 + P10));
                %lambdaBwd(n,:) = lambdaBwd0;
                lambdaBwd(n,:) = min(max(lambdaBwd0, obj.sparsityMin), obj.sparsityMax);
            end
            
            % piOut calculation
            lambdaFwdMatrixLog = log(lambdaFwd)*ones(1,T); % NxT ln(lambdaFwd) matrix
            oneMinusLambdaFwdMatrixLog = log(1 - lambdaFwd)*ones(1,T); % NxT ln(1-lambdaFwd) matrix
            lambdaBwdMatrixLog = log(lambdaBwd)*ones(1,T); % NxT ln(lambdaBwd) matrix
            oneMinusLambdaBwdMatrixLog = log(1 - lambdaBwd)*ones(1,T); % NxT ln(1-lambdaBwd) matrix
            piInLogSumMatrix = piInLogSum*ones(1,T); % NxT sum_1_T(ln(piIn)) matrix 
            oneMinusPiInLogSumMatrix = oneMinusPiInLogSum*ones(1,T); % NxT sum_1_T(ln(1-piIn)) matrix
            
            tmp = lambdaFwdMatrixLog + lambdaBwdMatrixLog + piInLogSumMatrix - log(piInMatrix) ...
                - oneMinusLambdaFwdMatrixLog - oneMinusLambdaBwdMatrixLog ...
                - oneMinusPiInLogSumMatrix + log(1 - piInMatrix); % NxT matrix
            tmp = min(max(tmp, obj.logMin), obj.logMax);
            tmp = exp(tmp); 
            piOutMatrix = tmp./(1 + tmp);
            
            piiMatrix = piInMatrix.*piOutMatrix./((1 - piInMatrix).*(1 - piOutMatrix) + piInMatrix.*piOutMatrix);
            
            % Expectation maximization
            if obj.isEMUpdateLambda0
                % 1x1 Sparsity of the first variable
                tmp = log(lambdaFwd(1,1)) + log(lambdaBwd(1,1)) + piInLogSum(1,1) ...
                    - log(1 - lambdaFwd(1,1)) - log(1 - lambdaBwd(1,1)) ...
                    - oneMinusPiInLogSum(1,1);
                tmp = min(max(tmp, obj.logMin), obj.logMax);
                tmp = exp(tmp);
                lambda0 = tmp/(1 + tmp);
                obj.xlambda0 = min(max(lambda0, obj.sparsityMin), obj.sparsityMax);
            end

            if obj.isEMUpdateTransProb
                % Nx1 Posterior probability of s, from 1 to N
                tmp = log(lambdaFwd) + log(lambdaBwd) + piInLogSum ...
                    - log(1 - lambdaFwd) - log(1 - lambdaBwd) - oneMinusPiInLogSum;
                tmp = min(max(tmp, obj.logMin), obj.logMax);
                tmp = exp(tmp); % Nx1
                PS1 = tmp./(1 + tmp); % Nx1 Pr(sn=1|Y) matrix

                % N-1x1 Transaction probability, from 2 to N
                tmp = log(lambdaFwd(1:N-1,1)) + piInLogSum(1:N-1,1) ...
                    + log(lambdaBwd(2:N,1)) + piInLogSum(2:N,1) ...
                    - log(1 - lambdaFwd(1:N-1,1)) - oneMinusPiInLogSum(1:N-1,1) ...
                    - log(1 - lambdaBwd(2:N,1)) - oneMinusPiInLogSum(2:N,1);
                tmp = min(max(tmp, obj.logMin), obj.logMax);
                tmp = exp(tmp); % N-1x1 matrix

                tmp2 = log(lambdaFwd(1:N-1,1)) + piInLogSum(1:N-1,1) ...
                    - log(1 - lambdaFwd(1:N-1,1)) - oneMinusPiInLogSum(1:N-1,1);
                tmp2 = min(max(tmp2, obj.logMin), obj.logMax);
                tmp2 = exp(tmp2); % N-1x1 matrix

                tmp3 = log(lambdaBwd(2:N,1)) + piInLogSum(2:N,1) ...
                    - log(1 - lambdaBwd(2:N,1)) - oneMinusPiInLogSum(2:N,1);
                tmp3 = min(max(tmp3, obj.logMin), obj.logMax);
                tmp3 = exp(tmp3); % N-1x1 matrix

                PS1S1 = tmp*(1 - P01)./((1 - P10) + tmp2*P01 + tmp3*P10 ...
                    + tmp*(1 - P01)); % N-1x1 Pr(sn=1,sn-1=1|Y) matrix

                P01New = sum(PS1(1:N-1,1) - PS1S1, 1)/sum(PS1(1:N-1,1), 1); % 1x1 matrix
                %obj.xP01 = P01New;
                obj.xP01 = min(max(P01New, obj.sparsityMin), obj.sparsityMax);

                P10New = sum(PS1(2:N,1) - PS1S1, 1)/((N-1) - sum(PS1(1:N-1,1),1)); % 1x1 matrix
                %obj.xP10 = P10New;
                obj.xP10 = min(max(P10New, obj.sparsityMin), obj.sparsityMax);

                %xsparsityAvg0 = mean(piiMatrix, 1); % 1xT matrix
                %obj.xsparsityAvg = xsparsityAvg0;
            end

            if obj.isEMUpdateMean
                mean0 = sum(gammaMatrix.*piiMatrix, 1)./sum(piiMatrix, 1);
                if isreal(mean0)
                    obj.xmean = min(max(mean0, obj.xmeanMin), obj.xmeanMax);
                else
                    obj.xmean = mean0;
                end     
            end

            if obj.isEMUpdateVar
                var0 = sum(piiMatrix.*(abs(gammaMatrix - meanMatrix).^2 + nuMatrix), 1)./sum(piiMatrix, 1);
                obj.xvar = min(max(var0, obj.xvarMin), obj.xvarMax);
            end
        end
        
        % Message passing on Markov chain
        function [piOutMatrix] = binaryProbEstim(obj, piInMatrix)
            
            % Initialization
            [N, T] = size(piInMatrix);
            piInMatrix = min(max(piInMatrix, obj.sparsityMin), obj.sparsityMax);
            
            % Markov chain forward and backward propagation
            lambdaFwd = NaN*ones(N,1);
            lambdaBwd = NaN*ones(N,1);
            lambdaFwd(1,1) = obj.xlambda0;
            lambdaBwd(N,1) = 0.5;
            P01 = obj.xP01;
            P10 = obj.XP10;
            piInLogSum = sum(log(piInMatrix),2); % Nx1 sum matrix of log(piInMatrix)
            oneMinusPiInLogSum = sum(log(1 - piInMatrix),2); % Nx1 sum matrix of log(1 - piInMatrix)
            
            % Forward propagation
            for n = 2:1:N
                tmp = log(lambdaFwd(n-1,1)) + piInLogSum(n-1,1) ...
                    - log(1 - lambdaFwd(n-1,1)) - oneMinusPiInLogSum(n-1,1);
                tmp = min(max(tmp, obj.logMin), obj.logMax);
                tmp = exp(tmp); % 1x1 value
                lambdaFwd0 = (P10 + tmp*(1 - P01))/(1 + tmp);
                %lambdaFwd(n,:) = lambdaFwd0;
                lambdaFwd(n,:) = min(max(lambdaFwd0, obj.sparsityMin), obj.sparsityMax);
            end
            
            % Backward propagation
            for n = N-1:-1:1
                tmp = log(lambdaBwd(n+1,1)) + piInLogSum(n+1,1) ...
                    - log(1 - lambdaBwd(n+1,1)) - oneMinusPiInLogSum(n+1,1);
                tmp = min(max(tmp, obj.logMin), obj.logMax);
                tmp = exp(tmp); % 1x1 value
                lambdaBwd0 = (P01 + tmp*(1 - P10))/((1 - P10 + P01) + tmp*(1 - P01 + P10));
                %lambdaBwd(n,:) = lambdaBwd0;
                lambdaBwd(n,:) = min(max(lambdaBwd0, obj.sparsityMin), obj.sparsityMax);
            end
            
            % piOut calculation
            lambdaFwdMatrixLog = log(lambdaFwd)*ones(1,T); % NxT ln(lambdaFwd) matrix
            oneMinusLambdaFwdMatrixLog = log(1 - lambdaFwd)*ones(1,T); % NxT ln(1-lambdaFwd) matrix
            lambdaBwdMatrixLog = log(lambdaBwd)*ones(1,T); % NxT ln(lambdaBwd) matrix
            oneMinusLambdaBwdMatrixLog = log(1 - lambdaBwd)*ones(1,T); % NxT ln(1-lambdaBwd) matrix
            piInLogSumMatrix = piInLogSum*ones(1,T); % NxT sum_1_T(ln(piIn)) matrix 
            oneMinusPiInLogSumMatrix = oneMinusPiInLogSum*ones(1,T); % NxT sum_1_T(ln(1-piIn)) matrix
            
            tmp = lambdaFwdMatrixLog + lambdaBwdMatrixLog + piInLogSumMatrix - log(piInMatrix) ...
                - oneMinusLambdaFwdMatrixLog - oneMinusLambdaBwdMatrixLog ...
                - oneMinusPiInLogSumMatrix + log(1 - piInMatrix); % NxT matrix
            tmp = min(max(tmp, obj.logMin), obj.logMax);
            tmp = exp(tmp); 
            piOutMatrix = tmp./(1 + tmp);
            
            % Test for P01 and P10
            %{
            % 1x1 Sparsity of the first variable
            tmp = log(lambdaFwd(1,1)) + log(lambdaBwd(1,1)) + piInLogSum(1,1) ...
                - log(1 - lambdaFwd(1,1)) - log(1 - lambdaBwd(1,1)) ...
                - oneMinusPiInLogSum(1,1);
            tmp = min(max(tmp, log(obj.sparsityMin)), log(obj.sparsityMax));
            tmp = exp(tmp);
            lambda0 = tmp/(1 + tmp);
            
            % Nx1 Posterior probability of s, from 1 to N
            tmp = log(lambdaFwd) + log(lambdaBwd) + piInLogSum ...
                - log(1 - lambdaFwd) - log(1 - lambdaBwd) - oneMinusPiInLogSum;
            tmp = min(max(tmp, log(obj.sparsityMin)), log(obj.sparsityMax));
            tmp = exp(tmp); % Nx1
            PS1 = tmp./(1 + tmp); % Nx1 Pr(sn=1|Y) matrix

            % N-1x1 Transaction probability, from 2 to N
            tmp = log(lambdaFwd(1:N-1,1)) + piInLogSum(1:N-1,1) ...
                + log(lambdaBwd(2:N,1)) + piInLogSum(2:N,1) ...
                - log(1 - lambdaFwd(1:N-1,1)) - oneMinusPiInLogSum(1:N-1,1) ...
                - log(1 - lambdaBwd(2:N,1)) - oneMinusPiInLogSum(2:N,1);
            tmp = min(max(tmp, log(obj.sparsityMin)), log(obj.sparsityMax));
            tmp = exp(tmp); % N-1x1 matrix

            tmp2 = log(lambdaFwd(1:N-1,1)) + piInLogSum(1:N-1,1) ...
                - log(1 - lambdaFwd(1:N-1,1)) - oneMinusPiInLogSum(1:N-1,1);
            tmp2 = min(max(tmp2, log(obj.sparsityMin)), log(obj.sparsityMax));
            tmp2 = exp(tmp2); % N-1x1 matrix

            tmp3 = log(lambdaBwd(2:N,1)) + piInLogSum(2:N,1) ...
                - log(1 - lambdaBwd(2:N,1)) - oneMinusPiInLogSum(2:N,1);
            tmp3 = min(max(tmp3, log(obj.sparsityMin)), log(obj.sparsityMax));
            tmp3 = exp(tmp3); % N-1x1 matrix

            PS1S1 = tmp*(1 - P01)./((1 - P10) + tmp2*P01 + tmp3*P10 ...
                + tmp*(1 - P01)); % N-1x1 Pr(sn=1,sn-1=1|Y) matrix

            P01New = sum(PS1(1:N-1,1) - PS1S1, 1)/sum(PS1(1:N-1,1), 1); % 1x1 matrix
            P10New = sum(PS1(2:N,1) - PS1S1, 1)/((N-1) - sum(PS1(1:N-1,1),1)); % 1x1 matrix
            fprintf('Test End.\n');
            %}
        end
        
        % Complex soft threshold denoiser
        % Asymptotic analysis of complex LASSO via complex approximate
        % message passing Equ.(4)
        function [ output ] = complexSoftThres( ~, value0, threshold0 )
            [N, L] = size(value0);
            output = zeros(N,L);
            valueAbs = abs(value0); % sqrt(u^2 + v^2)
            index = valueAbs > threshold0; % find sqrt(u^2 + v^2) > lambda
            output(index) = value0(index) - threshold0./valueAbs(index).*value0(index);
        end
    end
    
end

