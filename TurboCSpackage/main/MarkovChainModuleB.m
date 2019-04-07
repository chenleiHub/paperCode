classdef MarkovChainModuleB < handle
    % MarkovChainModuleB
    % 
    
    properties
        xlambda0; % 1xT sparsity of the first variable node 
        xsparsityAvg; % 1xT average sparsity of the NxT unknown matrix
        xP01; % 1xT transition probability of the NxT unknown matrix, P01 = Pr(sn=0|sn-1=1)
        xP10; % 1xT transition probability of the NxT unknown matrix, P10 = Pr(sn=1|sn-1=0)
        sparsityMax = 1-1e-11; % maximum value of the sparsity and binary probability
        sparsityMin = 1e-11; % minimum value of the sparsity and binary probability
        xmean; % 1xT mean of the NxT unknown matrix
        xmeanMax = Inf; % maximum mean value 
        xmeanMin = -Inf; % minimum mean value
        xvar; % 1xT variance of the NxT unknown matrix
        xvarMax = Inf; % maximum variance value
        xvarMin = 1e-11; % minimum variance value
        logMax = 20; % maximum value on the exponent
        logMin = -20; % minimum value on the exponent
        isComplex = false; % the value is complex or not
        isEMUpatePerIter = false; % the expectation maximization is used per iteration or not
        isEMUpdateLambda0 = false; % the expectation maximization is used for sparsity update or not
        isEMUpdateTransProb = false; % the expectation maximization is used for transition probability update or not
        isEMUpdateMean = false; % the expectation maximization is used for mean update or not
        isEMUpdateVar = false; % the expectation maximization is used for variance update or not       
    end
    
    methods
        % Constructor
        function obj = MarkovChainModuleB(xlambda0, xsparsityAvg, xP01, xP10, xmean, xvar)
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
            lambdaFwd = NaN*ones(N,T);
            lambdaBwd = NaN*ones(N,T);        
            lambdaFwd(1,:) = obj.xlambda0;
            lambdaBwd(N,:) = 0.5*ones(1,T);
            P01 = obj.xP01;
            P10 = obj.xP10;
            % Forward propagation
            for n = 2:1:N
                % 1xT matrix
                lambdaFwd0 = ((1 - lambdaFwd(n-1,:)).*(1 - piInMatrix(n-1,:)).*P10 + lambdaFwd(n-1,:).*piInMatrix(n-1,:).*(1 - P01))...
                    ./((1 - lambdaFwd(n-1,:)).*(1 - piInMatrix(n-1,:)) + lambdaFwd(n-1,:).*piInMatrix(n-1,:));
                %lambdaFwd(n,:) = lambdaFwd0;
                lambdaFwd(n,:) = min(max(lambdaFwd0, obj.sparsityMin), obj.sparsityMax);
            end
            % Backward propagation
            for n = N-1:-1:1
                % 1xT matrix
                lambdaBwd0 = ((1 - lambdaBwd(n+1,:)).*(1 - piInMatrix(n+1,:)).*P01 + lambdaBwd(n+1,:).*piInMatrix(n+1,:).*(1 - P01))...
                    ./((1 - lambdaBwd(n+1,:)).*(1 - piInMatrix(n+1,:)).*(1 - P10 + P01) + lambdaBwd(n+1,:).*piInMatrix(n+1,:).*(1 - P01 + P10));
                %lambdaBwd(n,:) = lambdaBwd0;
                lambdaBwd(n,:) = min(max(lambdaBwd0, obj.sparsityMin), obj.sparsityMax);
            end
            
            % piOut calculation
            piOutMatrix = lambdaFwd.*lambdaBwd./((1 - lambdaFwd).*(1 - lambdaBwd) + lambdaFwd.*lambdaBwd);
            piOutMatrix = min(max(piOutMatrix, obj.sparsityMin), obj.sparsityMax);

            %piiMatrix = 1./(1 + (1 - piOutMatrix)./piOutMatrix.*tmp);
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
            lambdaFwd = NaN*ones(N,T);
            lambdaBwd = NaN*ones(N,T);        
            lambdaFwd(1,:) = obj.xlambda0;
            lambdaBwd(N,:) = 0.5*ones(1,T);
            P01 = obj.xP01;
            P10 = obj.xP10;
            % Forward propagation
            for n = 2:1:N
                % 1xT matrix
                lambdaFwd0 = ((1 - lambdaFwd(n-1,:)).*(1 - piInMatrix(n-1,:)).*P10 + lambdaFwd(n-1,:).*piInMatrix(n-1,:).*(1 - P01))...
                    ./((1 - lambdaFwd(n-1,:)).*(1 - piInMatrix(n-1,:)) + lambdaFwd(n-1,:).*piInMatrix(n-1,:));
                %lambdaFwd(n,:) = lambdaFwd0;
                lambdaFwd(n,:) = min(max(lambdaFwd0, obj.sparsityMin), obj.sparsityMax);
            end
            % Backward propagation
            for n = N-1:-1:1
                % 1xT matrix
                lambdaBwd0 = ((1 - lambdaBwd(n+1,:)).*(1 - piInMatrix(n+1,:)).*P01 + lambdaBwd(n+1,:).*piInMatrix(n+1,:).*(1 - P01))...
                    ./((1 - lambdaBwd(n+1,:)).*(1 - piInMatrix(n+1,:)).*(1 - P10 + P01) + lambdaBwd(n+1,:).*piInMatrix(n+1,:).*(1 - P01 + P10));
                %lambdaBwd(n,:) = lambdaBwd0;
                lambdaBwd(n,:) = min(max(lambdaBwd0, obj.sparsityMin), obj.sparsityMax);
            end
            
            % piOut calculation
            piOutMatrix = lambdaFwd.*lambdaBwd./((1 - lambdaFwd).*(1 - lambdaBwd) + lambdaFwd.*lambdaBwd);
            piOutMatrix = min(max(piOutMatrix, obj.sparsityMin), obj.sparsityMax);

            %piiMatrix = 1./(1 + (1 - piOutMatrix)./piOutMatrix.*tmp);
            piiMatrix = piInMatrix.*piOutMatrix./((1 - piInMatrix).*(1 - piOutMatrix) + piInMatrix.*piOutMatrix);
            
            % Expectation maximization 
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
                
                P10New = sum(piiMatrix(2:N,:) - PS1S1Normal, 1)./((N-1) - sum(piiMatrix(1:N-1,:), 1)); % 1xT matrix   
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
        end
        
        % Message passing on Markov chain
        function [piOutMatrix] = binaryProbEstim(obj, piInMatrix)
            
            % Initialization
            [N, T] = size(piInMatrix);
            piInMatrix = min(max(piInMatrix, obj.sparsityMin), obj.sparsityMax);
            
            % Markov chain forward and backward propagation
            lambdaFwd = NaN*ones(N,T);
            lambdaBwd = NaN*ones(N,T);        
            lambdaFwd(1,:) = obj.xlambda0;
            lambdaBwd(N,:) = 0.5*ones(1,T);
            P01 = obj.xP01;
            P10 = obj.xP10;
            % Forward propagation
            for n = 2:1:N
                % 1xT matrix
                lambdaFwd(n,:) = ((1 - lambdaFwd(n-1,:)).*(1 - piInMatrix(n-1,:)).*P10 + lambdaFwd(n-1,:).*piInMatrix(n-1,:).*(1 - P01))...
                    ./((1 - lambdaFwd(n-1,:)).*(1 - piInMatrix(n-1,:)) + lambdaFwd(n-1,:).*piInMatrix(n-1,:));
            end
            % Backward propagation
            for n = N-1:-1:1
                % 1xT matrix
                lambdaBwd(n,:) = ((1 - lambdaBwd(n+1,:)).*(1 - piInMatrix(n+1,:)).*P01 + lambdaBwd(n+1,:).*piInMatrix(n+1,:).*(1 - P01))...
                    ./((1 - lambdaBwd(n+1,:)).*(1 - piInMatrix(n+1,:)).*(1 - P10 + P01) + lambdaBwd(n+1,:).*piInMatrix(n+1,:).*(1 - P01 + P10));
            end
            
            % piOut calculation
            piOutMatrix = lambdaFwd.*lambdaBwd./((1 - lambdaFwd).*(1 - lambdaBwd) + lambdaFwd.*lambdaBwd);
            piOutMatrix = min(max(piOutMatrix, obj.sparsityMin), obj.sparsityMax);
            
            % Test for P01 and P10
            %{ 
            piiMatrix = piInMatrix.*piOutMatrix./((1 - piInMatrix).*(1 - piOutMatrix) + piInMatrix.*piOutMatrix);
            % 1xT Sparsity of the first variable 
            lambda0 = piiMatrix(1,:);
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
            P10New = sum(piiMatrix(2:N,:) - PS1S1Normal, 1)./((N-1) - sum(piiMatrix(1:N-1,:), 1)); % 1xT matrix   
            fprintf('Test End.\n');
            %}
        end
    end
    
end

