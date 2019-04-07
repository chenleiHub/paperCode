classdef DelaySuppModuleB < handle
    % DelaySuppModuleB
    % In this class, each Nx1 column vector of NxT unknown matrix is judged 
    % by an additional variable, which can detect the elements are all-zero 
    % or not. 
    
    properties
        xlambda0; % 1xT sparsity of the first variable node
        xsparsityAvg; % 1xT average sparsity of the NxT unknown matrix
        xP01; % 1xT transition probability of the NxT unknown matrix, P01 = Pr(sn=0|sn-1=1)
        xP10; % 1xT transition probability of the NxT unknown matrix, P10 = Pr(sn=1|sn-1=0)
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
        xjudge; % 1xT binary probability of the Nx1 column vector is non-zero
        isComplex = false; % the value is complex or not
        isEMUpatePerIter = false; % the expectation maximization is used per iteration or not
        isEMUpdateLambda0 = false; % the expectation maximization is used for sparsity update or not
        isEMUpdateTransProb = false; % the expectation maximization is used for transition probability update or not
        isEMUpdateMean = false; % the expectation maximization is used for mean update or not
        isEMUpdateVar = false; % the expectation maximization is used for variance update or not 
        isEMUpdateJudge = false; % the expectation maximization is used for all-zero judgement or not
    end
    
    methods
        % Constructor
        function obj = DelaySuppModuleB(xlambda0, xsparsityAvg, xP01, xP10, ...
                xmean, xvar, xjudge)
            if nargin > 0
                obj.xlambda0 = xlambda0;
                obj.xsparsityAvg = xsparsityAvg;
                obj.xP01 = xP01;
                obj.xP10 = xP10;
                obj.xmean = xmean;
                obj.xvar = xvar;
                obj.xjudge = xjudge;
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
                tmp = abs(xBpri - meanMatrix).^2./(vBpriMatrix + varMatrix) - abs(xBpri).^2./vBpriMatrix;
                tmp = varMatrix./nuMatrix.*exp(tmp);
            else
                tmp = abs(xBpri - meanMatrix).^2./(vBpriMatrix + varMatrix) - abs(xBpri).^2./vBpriMatrix;
                tmp = sqrt(varMatrix./nuMatrix).*exp(0.5*tmp);
            end
            
            % piIn calculation
            piInMatrix = 1./(1 + tmp);
            piInMatrix = min(max(piInMatrix, obj.sparsityMin), obj.sparsityMax);
            
            % Message passing on the Markov chain based loopy factor graph
            lambdaFwd = NaN*ones(N,T);
            lambdaFwdPrime = NaN*ones(N,T); % Note that lambdaFwd2(N,:) is useless.
            lambdaBwd = NaN*ones(N,T);
            lambdaBwdPrime = NaN*ones(N,T);
            
            thetaLeft = ones(N,1)*obj.xjudge; % NxT matrix
            thetaRight = NaN*ones(N,T);
            % Initialization
            P01 = obj.xP01; % 1xT matrix
            P10 = obj.xP10; % 1xT matrix
            lambdaBwd(N,:) = 0.5*ones(1,T);
            espilon = 1e-11;
            damping = 1;
            Iter = 5;
            for ii = 1:Iter
                % Forward propagation
                lambdaFwd0 = thetaLeft(1,:).*obj.xlambda0 + (1 - thetaLeft(1,:))*espilon; 
                lambdaFwd(1,:) = min(max(lambdaFwd0, obj.sparsityMin), obj.sparsityMax);
                for n = 2:1:N
                    lambdaFwd0 = piInMatrix(n-1,:).*lambdaFwd(n-1,:) ...
                        ./((1 - piInMatrix(n-1,:)).*(1 - lambdaFwd(n-1,:)) + piInMatrix(n-1,:).*lambdaFwd(n-1,:));
                    lambdaFwdPrime(n-1,:) = min(max(lambdaFwd0, obj.sparsityMin), obj.sparsityMax);
                    
                    lambdaFwd0 = lambdaFwdPrime(n-1,:).*thetaLeft(n,:).*(1 - P01) ...
                        + (1 - lambdaFwdPrime(n-1,:)).*thetaLeft(n,:).*P10 ...
                        + (1 - thetaLeft(n,:))*espilon;
                    lambdaFwd(n,:) = min(max(lambdaFwd0, obj.sparsityMin), obj.sparsityMax);
                end
                lambdaFwd0 = piInMatrix(N,:).*lambdaFwd(N,:) ...
                        ./((1 - piInMatrix(N,:)).*(1 - lambdaFwd(N,:)) + piInMatrix(N,:).*lambdaFwd(N,:));
                lambdaFwdPrime(N,:) = min(max(lambdaFwd0, obj.sparsityMin), obj.sparsityMax);
                if ii > 1
                    lambdaFwd = damping*lambdaFwd + (1 - damping)*lambdaFwd_old;
                    lambdaFwdPrime = damping*lambdaFwdPrime + (1 - damping)*lambdaFwdPrime_old;
                end
                lambdaFwd_old = lambdaFwd;
                lambdaFwdPrime_old = lambdaFwdPrime;
                
                % Backward propagation
                for n = N-1:-1:1
                    lambdaBwd0 = piInMatrix(n+1,:).*lambdaBwd(n+1,:) ...
                        ./((1 - piInMatrix(n+1,:)).*(1 - lambdaBwd(n+1,:)) + piInMatrix(n+1,:).*lambdaBwd(n+1,:));
                    lambdaBwdPrime(n+1,:) = min(max(lambdaBwd0, obj.sparsityMin), obj.sparsityMax);
                    
                    lambdaBwd0 = ((1 - lambdaBwdPrime(n+1,:)).*(1 - thetaLeft(n+1,:))*(1 - espilon) ...
                        + (1 - lambdaBwdPrime(n+1,:)).*thetaLeft(n+1,:).*P01 ...
                        + lambdaBwdPrime(n+1,:).*(1 - thetaLeft(n+1,:))*espilon ...
                        + lambdaBwdPrime(n+1,:).*thetaLeft(n+1,:).*(1 - P01)) ...
                        ./(2*(1 - espilon)*(1 - lambdaBwdPrime(n+1,:)).*(1 - thetaLeft(n+1,:)) ...
                        + 2*espilon*lambdaBwdPrime(n+1,:).*(1 - thetaLeft(n+1,:)) ...
                        + (1 - lambdaBwdPrime(n+1,:)).*thetaLeft(n+1,:).*(1 - P10 + P01) ...
                        + lambdaBwdPrime(n+1,:).*thetaLeft(n+1,:).*(1 - P01 + P10));
                    lambdaBwd(n,:) = min(max(lambdaBwd0, obj.sparsityMin), obj.sparsityMax);
                end
                lambdaBwd0 = piInMatrix(1,:).*lambdaBwd(1,:) ...
                        ./((1 - piInMatrix(1,:)).*(1 - lambdaBwd(1,:)) + piInMatrix(1,:).*lambdaBwd(1,:));
                lambdaBwdPrime(1,:) = min(max(lambdaBwd0, obj.sparsityMin), obj.sparsityMax);
                if ii > 1
                    lambdaBwd = damping*lambdaBwd + (1 - damping)*lambdaBwd_old;
                    lambdaBwdPrime = damping*lambdaBwdPrime + (1 - damping)*lambdaBwdPrime_old;
                end
                lambdaBwd_old = lambdaBwd;
                lambdaBwdPrime_old = lambdaBwdPrime;
                
                % Update thetaRight
                lambda0 = ((1 - lambdaBwdPrime(1,:)).*(1 - obj.xlambda0) + lambdaBwdPrime(1,:).*obj.xlambda0) ...
                    ./((1 - lambdaBwdPrime(1,:)).*(1 - obj.xlambda0) + lambdaBwdPrime(1,:).*obj.xlambda0 ...
                    + (1 - lambdaBwdPrime(1,:))*(1 - espilon) + lambdaBwdPrime(1,:)*espilon);
                thetaRight(1,:) = min(max(lambda0, obj.sparsityMin), obj.sparsityMax);
                for n = 2:1:N
                    lambda0 = (lambdaFwdPrime(n-1,:).*lambdaBwdPrime(n,:).*(1 - P01) ...
                        + lambdaFwdPrime(n-1,:).*(1 - lambdaBwdPrime(n,:)).*P01 ...
                        + (1 - lambdaFwdPrime(n-1,:)).*lambdaBwdPrime(n,:).*P10 ...
                        + (1 - lambdaFwdPrime(n-1,:)).*(1 - lambdaBwdPrime(n,:)).*(1 - P10)) ...
                        ./(lambdaBwdPrime(n,:)*espilon + (1 - lambdaBwdPrime(n,:))*(1 - espilon) ...
                        + lambdaFwdPrime(n-1,:).*lambdaBwdPrime(n,:).*(1 - P01) ...
                        + lambdaFwdPrime(n-1,:).*(1 - lambdaBwdPrime(n,:)).*P01 ...
                        + (1 - lambdaFwdPrime(n-1,:)).*lambdaBwdPrime(n,:).*P10 ...
                        + (1 - lambdaFwdPrime(n-1,:)).*(1 - lambdaBwdPrime(n,:)).*(1 - P10));
                    thetaRight(n,:) = min(max(lambda0, obj.sparsityMin), obj.sparsityMax);
                end
                if ii > 1
                    thetaRight = damping*thetaRight + (1 - damping)*thetaRight_old;
                end
                thetaRight_old = thetaRight;
                
                % Update thetaLeft
                thetaLogSumMatrix = ones(N,1)*sum(log(thetaRight),1);
                oneMinusThetaLogSumMatrix = ones(N,1)*sum(log(1 - thetaRight),1);
                xjudgeMatrix = ones(N,1)*obj.xjudge;
                tmp = log(1 - xjudgeMatrix) - log(xjudgeMatrix) + oneMinusThetaLogSumMatrix ...
                    - log(1 - thetaRight) - thetaLogSumMatrix + log(thetaRight);
                tmp = min(max(tmp, obj.logMin), obj.logMax);
                thetaLeft = 1./(1 + exp(tmp));
                if ii > 1
                    thetaLeft = damping*thetaLeft + (1 - damping)*thetaLeft_old;
                end
                thetaLeft_old = thetaLeft;               
            end
            
            % Calculate piOutMatrix
            piOutMatrix = lambdaFwd.*lambdaBwd./((1 - lambdaFwd).*(1 - lambdaBwd) + lambdaFwd.*lambdaBwd);
            piOutMatrix = min(max(piOutMatrix, obj.sparsityMin), obj.sparsityMax);
        
            piiMatrix = piInMatrix.*piOutMatrix./((1 - piInMatrix).*(1 - piOutMatrix) + piInMatrix.*piOutMatrix);
            
            % xBpost calculation
            xBpost = gammaMatrix.*piiMatrix;
            
            % vBpost calculation
            vBpost = sum(piiMatrix.*(abs(gammaMatrix).^2 + nuMatrix) - abs(xBpost).^2, 1)/N;
            
            % Expectation maximization 
            if obj.isEMUpatePerIter
                if obj.isEMUpdateLambda0
                    % 1xT Sparsity of the first variable 
                    lambda0 = piiMatrix(1,:);
                    obj.xlambda0 = min(max(lambda0, obj.sparsityMin), obj.sparsityMax);
                end
                
                if obj.isEMUpdateTransProb
                    % N-1xT Transaction probability, from 2 to N
                    PS0S0 = (1 - lambdaFwd(1:N-1,:)).*(1 - piInMatrix(1:N-1,:))...
                        .*(1 - piInMatrix(2:N,:)).*(1 - lambdaBwd(2:N,:)).*(1 - ones(N-1,1)*P10);
                    PS1S0 = (1 - lambdaFwd(1:N-1,:)).*(1 - piInMatrix(1:N-1,:))...
                        .*piInMatrix(2:N,:).*lambdaBwd(2:N,:).*(ones(N-1,1)*P10);
                    PS0S1 = lambdaFwd(1:N-1,:).*piInMatrix(1:N-1,:)...
                        .*(1 - piInMatrix(2:N,:)).*(1 - lambdaBwd(2:N,:)).*(ones(N-1,1)*P01);
                    PS1S1 = lambdaFwd(1:N-1,:).*piInMatrix(1:N-1,:)...
                        .*piInMatrix(2:N,:).*lambdaBwd(2:N,:).*(1 - ones(N-1,1)*P01);
                    PS1S1Normal = PS1S1./(PS0S0 + PS1S0 + PS0S1 + PS1S1); % N-1xT matrix

                    P01New = sum(piiMatrix(1:N-1,:) - PS1S1Normal, 1)./sum(piiMatrix(1:N-1,:), 1); % 1xT matrix
                    %obj.xP01 = P01New;
                    obj.xP01 = min(max(P01New, obj.sparsityMin), obj.sparsityMax);

                    P10New = sum(piiMatrix(2:N,:) - PS1S1Normal, 1)./((N-1) - sum(piiMatrix(1:N-1,:), 1));
                    %obj.xP10 = P10New;
                    obj.xP10 = min(max(P10New, obj.sparsityMin), obj.sparsityMax); 
                    
                    %xsparsityAvg0 = mean(piiMatrix, 1);
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
                
                if obj.isEMUpdateJudge
                    tmp = sum(log(1 - thetaRight),1) + log(1 - obj.xjudge) ...
                        - sum(log(thetaRight),1) - log(obj.xjudge);
                    tmp = min(max(tmp, obj.logMin), obj.logMax);
                    judge0 = 1./(1 + exp(tmp));
                    obj.xjudge = min(max(judge0, obj.sparsityMin), obj.sparsityMax);
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
            
            % Message passing on the Markov chain based loopy factor graph
            lambdaFwd = NaN*ones(N,T);
            lambdaFwdPrime = NaN*ones(N,T); % Note that lambdaFwd2(N,:) is useless.
            lambdaBwd = NaN*ones(N,T);
            lambdaBwdPrime = NaN*ones(N,T);
            
            thetaLeft = ones(N,1)*obj.xjudge; % NxT matrix
            thetaRight = NaN*ones(N,T);
            % Initialization
            P01 = obj.xP01; % 1xT matrix
            P10 = obj.xP10; % 1xT matrix
            lambdaBwd(N,:) = 0.5*ones(1,T);
            espilon = 1e-11;
            damping = 1;
            Iter = 5;
            for ii = 1:Iter
                % Forward propagation
                lambdaFwd0 = thetaLeft(1,:).*obj.xlambda0 + (1 - thetaLeft(1,:))*espilon; 
                lambdaFwd(1,:) = min(max(lambdaFwd0, obj.sparsityMin), obj.sparsityMax);
                for n = 2:1:N
                    lambdaFwd0 = piInMatrix(n-1,:).*lambdaFwd(n-1,:) ...
                        ./((1 - piInMatrix(n-1,:)).*(1 - lambdaFwd(n-1,:)) + piInMatrix(n-1,:).*lambdaFwd(n-1,:));
                    lambdaFwdPrime(n-1,:) = min(max(lambdaFwd0, obj.sparsityMin), obj.sparsityMax);
                    
                    lambdaFwd0 = lambdaFwdPrime(n-1,:).*thetaLeft(n,:).*(1 - P01) ...
                        + (1 - lambdaFwdPrime(n-1,:)).*thetaLeft(n,:).*P10 ...
                        + (1 - thetaLeft(n,:))*espilon;
                    lambdaFwd(n,:) = min(max(lambdaFwd0, obj.sparsityMin), obj.sparsityMax);
                end
                lambdaFwd0 = piInMatrix(N,:).*lambdaFwd(N,:) ...
                        ./((1 - piInMatrix(N,:)).*(1 - lambdaFwd(N,:)) + piInMatrix(N,:).*lambdaFwd(N,:));
                lambdaFwdPrime(N,:) = min(max(lambdaFwd0, obj.sparsityMin), obj.sparsityMax);
                if ii > 1
                    lambdaFwd = damping*lambdaFwd + (1 - damping)*lambdaFwd_old;
                    lambdaFwdPrime = damping*lambdaFwdPrime + (1 - damping)*lambdaFwdPrime_old;
                end
                lambdaFwd_old = lambdaFwd;
                lambdaFwdPrime_old = lambdaFwdPrime;
                
                % Backward propagation
                for n = N-1:-1:1
                    lambdaBwd0 = piInMatrix(n+1,:).*lambdaBwd(n+1,:) ...
                        ./((1 - piInMatrix(n+1,:)).*(1 - lambdaBwd(n+1,:)) + piInMatrix(n+1,:).*lambdaBwd(n+1,:));
                    lambdaBwdPrime(n+1,:) = min(max(lambdaBwd0, obj.sparsityMin), obj.sparsityMax);
                    
                    lambdaBwd0 = ((1 - lambdaBwdPrime(n+1,:)).*(1 - thetaLeft(n+1,:))*(1 - espilon) ...
                        + (1 - lambdaBwdPrime(n+1,:)).*thetaLeft(n+1,:).*P01 ...
                        + lambdaBwdPrime(n+1,:).*(1 - thetaLeft(n+1,:))*espilon ...
                        + lambdaBwdPrime(n+1,:).*thetaLeft(n+1,:).*(1 - P01)) ...
                        ./(2*(1 - espilon)*(1 - lambdaBwdPrime(n+1,:)).*(1 - thetaLeft(n+1,:)) ...
                        + 2*espilon*lambdaBwdPrime(n+1,:).*(1 - thetaLeft(n+1,:)) ...
                        + (1 - lambdaBwdPrime(n+1,:)).*thetaLeft(n+1,:).*(1 - P10 + P01) ...
                        + lambdaBwdPrime(n+1,:).*thetaLeft(n+1,:).*(1 - P01 + P10));
                    lambdaBwd(n,:) = min(max(lambdaBwd0, obj.sparsityMin), obj.sparsityMax);
                end
                lambdaBwd0 = piInMatrix(1,:).*lambdaBwd(1,:) ...
                        ./((1 - piInMatrix(1,:)).*(1 - lambdaBwd(1,:)) + piInMatrix(1,:).*lambdaBwd(1,:));
                lambdaBwdPrime(1,:) = min(max(lambdaBwd0, obj.sparsityMin), obj.sparsityMax);
                if ii > 1
                    lambdaBwd = damping*lambdaBwd + (1 - damping)*lambdaBwd_old;
                    lambdaBwdPrime = damping*lambdaBwdPrime + (1 - damping)*lambdaBwdPrime_old;
                end
                lambdaBwd_old = lambdaBwd;
                lambdaBwdPrime_old = lambdaBwdPrime;
                
                % Update thetaRight
                lambda0 = ((1 - lambdaBwdPrime(1,:)).*(1 - obj.xlambda0) + lambdaBwdPrime(1,:).*obj.xlambda0) ...
                    ./((1 - lambdaBwdPrime(1,:)).*(1 - obj.xlambda0) + lambdaBwdPrime(1,:).*obj.xlambda0 ...
                    + (1 - lambdaBwdPrime(1,:))*(1 - espilon) + lambdaBwdPrime(1,:)*espilon);
                thetaRight(1,:) = min(max(lambda0, obj.sparsityMin), obj.sparsityMax);
                for n = 2:1:N
                    lambda0 = (lambdaFwdPrime(n-1,:).*lambdaBwdPrime(n,:).*(1 - P01) ...
                        + lambdaFwdPrime(n-1,:).*(1 - lambdaBwdPrime(n,:)).*P01 ...
                        + (1 - lambdaFwdPrime(n-1,:)).*lambdaBwdPrime(n,:).*P10 ...
                        + (1 - lambdaFwdPrime(n-1,:)).*(1 - lambdaBwdPrime(n,:)).*(1 - P10)) ...
                        ./(lambdaBwdPrime(n,:)*espilon + (1 - lambdaBwdPrime(n,:))*(1 - espilon) ...
                        + lambdaFwdPrime(n-1,:).*lambdaBwdPrime(n,:).*(1 - P01) ...
                        + lambdaFwdPrime(n-1,:).*(1 - lambdaBwdPrime(n,:)).*P01 ...
                        + (1 - lambdaFwdPrime(n-1,:)).*lambdaBwdPrime(n,:).*P10 ...
                        + (1 - lambdaFwdPrime(n-1,:)).*(1 - lambdaBwdPrime(n,:)).*(1 - P10));
                    thetaRight(n,:) = min(max(lambda0, obj.sparsityMin), obj.sparsityMax);
                end
                if ii > 1
                    thetaRight = damping*thetaRight + (1 - damping)*thetaRight_old;
                end
                thetaRight_old = thetaRight;
                
                % Update thetaLeft
                thetaLogSumMatrix = ones(N,1)*sum(log(thetaRight),1);
                oneMinusThetaLogSumMatrix = ones(N,1)*sum(log(1 - thetaRight),1);
                xjudgeMatrix = ones(N,1)*obj.xjudge;
                tmp = log(1 - xjudgeMatrix) - log(xjudgeMatrix) + oneMinusThetaLogSumMatrix ...
                    - log(1 - thetaRight) - thetaLogSumMatrix + log(thetaRight);
                tmp = min(max(tmp, obj.logMin), obj.logMax);
                thetaLeft = 1./(1 + exp(tmp));
                if ii > 1
                    thetaLeft = damping*thetaLeft + (1 - damping)*thetaLeft_old;
                end
                thetaLeft_old = thetaLeft;               
            end
            
            % Calculate piOutMatrix
            piOutMatrix = lambdaFwd.*lambdaBwd./((1 - lambdaFwd).*(1 - lambdaBwd) + lambdaFwd.*lambdaBwd);
            piOutMatrix = min(max(piOutMatrix, obj.sparsityMin), obj.sparsityMax);
        
            piiMatrix = piInMatrix.*piOutMatrix./((1 - piInMatrix).*(1 - piOutMatrix) + piInMatrix.*piOutMatrix);
            
            if obj.isEMUpdateLambda0
                % 1xT Sparsity of the first variable 
                lambda0 = piiMatrix(1,:);
                obj.xlambda0 = min(max(lambda0, obj.sparsityMin), obj.sparsityMax);
            end

            if obj.isEMUpdateTransProb
                % N-1xT Transaction probability, from 2 to N
                PS0S0 = (1 - lambdaFwd(1:N-1,:)).*(1 - piInMatrix(1:N-1,:))...
                    .*(1 - piInMatrix(2:N,:)).*(1 - lambdaBwd(2:N,:)).*(1 - ones(N-1,1)*P10);
                PS1S0 = (1 - lambdaFwd(1:N-1,:)).*(1 - piInMatrix(1:N-1,:))...
                    .*piInMatrix(2:N,:).*lambdaBwd(2:N,:).*(ones(N-1,1)*P10);
                PS0S1 = lambdaFwd(1:N-1,:).*piInMatrix(1:N-1,:)...
                    .*(1 - piInMatrix(2:N,:)).*(1 - lambdaBwd(2:N,:)).*(ones(N-1,1)*P01);
                PS1S1 = lambdaFwd(1:N-1,:).*piInMatrix(1:N-1,:)...
                    .*piInMatrix(2:N,:).*lambdaBwd(2:N,:).*(1 - ones(N-1,1)*P01);
                PS1S1Normal = PS1S1./(PS0S0 + PS1S0 + PS0S1 + PS1S1); % N-1xT matrix

                P01New = sum(piiMatrix(1:N-1,:) - PS1S1Normal, 1)./sum(piiMatrix(1:N-1,:), 1); % 1xT matrix
                %obj.xP01 = P01New;
                obj.xP01 = min(max(P01New, obj.sparsityMin), obj.sparsityMax);

                P10New = sum(piiMatrix(2:N,:) - PS1S1Normal, 1)./((N-1) - sum(piiMatrix(1:N-1,:), 1));
                %obj.xP10 = P10New;
                obj.xP10 = min(max(P10New, obj.sparsityMin), obj.sparsityMax); 

                %xsparsityAvg0 = mean(piiMatrix, 1);
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

            if obj.isEMUpdateJudge
                tmp = sum(log(1 - thetaRight),1) + log(1 - obj.xjudge) ...
                    - sum(log(thetaRight),1) - log(obj.xjudge);
                tmp = min(max(tmp, obj.logMin), obj.logMax);
                judge0 = 1./(1 + exp(tmp));
                obj.xjudge = min(max(judge0, obj.sparsityMin), obj.sparsityMax);
            end
        end
    end
    
end

