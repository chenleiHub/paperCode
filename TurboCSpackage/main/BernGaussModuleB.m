classdef BernGaussModuleB < handle
    % BernGaussModuleB
    %   
    
    properties
        xlambda % 1xT sparsity of the NxT unknown matrix
        xlambdaMax = 1-1e-11; % maximum value of the sparsity
        xlambdaMin = 1e-11; % minimum value of the sparsity
        xmean; % 1xT mean of the NxT unknown matrix
        xmeanMax = Inf; % maximum mean value 
        xmeanMin = -Inf; % minimum mean value
        xvar; % 1xT variance of the NxT unknown matrix
        xvarMax = Inf; % maximum variance value
        xvarMin = 1e-11; % minimum variance value
        logMax = Inf; % maximum value on the exponent
        logMin = -Inf; % minimum value on the exponent
        isComplex = false; % the value is complex or not
        isEMUpatePerIter = false; % the expectation maximization is used per iteration or not
        isEMUpdateLambda = false; % the expectation maximization is used for sparsity update or not
        isEMUpdateMean = false; % the expectation maximization is used for mean update or not
        isEMUpdateVar = false; % the expectation maximization is used for variance update or not
    end
    
    methods
        % Constructor
        function obj = BernGaussModuleB(xlambda, xmean, xvar)
            if nargin > 0
                obj.xlambda = xlambda;
                obj.xmean = xmean;
                obj.xvar = xvar;
            end
        end
        
        % Estimator
        function [xBpost, vBpost, CBpost] = estim(obj, xBpri, vBpri)
            [N, ~] = size(xBpri);    
            lambdaMatrix = ones(N,1)*obj.xlambda; % NxT matrix
            meanMatrix = ones(N,1)*obj.xmean; % NxT matrix
            varMatrix = ones(N,1)*obj.xvar; % NxT matrix
            vBpriMatrix = ones(N,1)*vBpri; % NxT matrix
            CBpost = [];
            
            nuMatrix = vBpriMatrix.*varMatrix./(vBpriMatrix + varMatrix);
            gammaMatrix = nuMatrix.*(xBpri./vBpriMatrix + meanMatrix./varMatrix);
            if obj.isComplex
                tmp = abs(xBpri - meanMatrix).^2./(vBpriMatrix + varMatrix) - abs(xBpri).^2./vBpriMatrix;
                %tmp = min(max(tmp, obj.logMin), obj.logMax);
                tmp = varMatrix./nuMatrix.*exp(tmp);
            else
                tmp = abs(xBpri - meanMatrix).^2./(vBpriMatrix + varMatrix) - abs(xBpri).^2./vBpriMatrix;
                %tmp = min(max(tmp, obj.logMin), obj.logMax);
                tmp = sqrt(varMatrix./nuMatrix).*exp(0.5*tmp);
            end
            piiMatrix = 1./(1 + (1 - lambdaMatrix)./lambdaMatrix.*tmp);
            
            % xBpost calculation
            xBpost = gammaMatrix.*piiMatrix;
            
            % vBpost calculation
            vBpost = sum(piiMatrix.*(abs(gammaMatrix).^2 + nuMatrix) - abs(xBpost).^2, 1)/N;
            
            % Expectation maximization 
            if obj.isEMUpatePerIter
                if obj.isEMUpdateLambda 
                    lambda0 = sum(piiMatrix, 1)/N;
                    obj.xlambda = min(max(lambda0, obj.xlambdaMin), obj.xlambdaMax);
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
            [N, ~] = size(xBpri);    
            lambdaMatrix = ones(N,1)*obj.xlambda; % NxT matrix
            meanMatrix = ones(N,1)*obj.xmean; % NxT matrix
            varMatrix = ones(N,1)*obj.xvar; % NxT matrix
            vBpriMatrix = ones(N,1)*vBpri; % NxT matrix
            
            nuMatrix = vBpriMatrix.*varMatrix./(vBpriMatrix + varMatrix);
            gammaMatrix = nuMatrix.*(xBpri./vBpriMatrix + meanMatrix./varMatrix);
            if obj.isComplex
                tmp = abs(xBpri - meanMatrix).^2./(vBpriMatrix + varMatrix) - abs(xBpri).^2./vBpriMatrix;
                %tmp = min(max(tmp, obj.logMin), obj.logMax);
                tmp = varMatrix./nuMatrix.*exp(tmp);
            else
                tmp = abs(xBpri - meanMatrix).^2./(vBpriMatrix + varMatrix) - abs(xBpri).^2./vBpriMatrix;
                %tmp = min(max(tmp, obj.logMin), obj.logMax);
                tmp = sqrt(varMatrix./nuMatrix).*exp(0.5*tmp);
            end
            piiMatrix = 1./(1 + (1 - lambdaMatrix)./lambdaMatrix.*tmp);
            
            % Expectation maximization 
            if obj.isEMUpdateLambda 
                lambda0 = sum(piiMatrix, 1)/N;
                obj.xlambda = min(max(lambda0, obj.xlambdaMin), obj.xlambdaMax);
            end
            
            if obj.isEMUpdateMean
                mean0 = sum(gammaMatrix.*piiMatrix, 1)./sum(piiMatrix, 1);
                obj.xmean = min(max(mean0, obj.xmeanMin), obj.xmeanMax);
            end
            
            if obj.isEMUpdateVar
                var0 = sum(piiMatrix.*(abs(gammaMatrix - meanMatrix).^2 + nuMatrix), 1)./sum(piiMatrix, 1);
                obj.xvar = min(max(var0, obj.xvarMin), obj.xvarMax);
            end
        end

    end
    
end

