classdef AwgnModuleA < handle
    % AwgnModuleA
    % 
    
    properties
        y; % MxT measurement matrix
        A; % MxNxT sensing matrix
        wvar; % 1xT noise variance matrix
        wvarMax = Inf; % maximum value of noise variance
        wvarMin = 1e-11; % minimum value of noise variance
        isComplex = false; % the value is complex or not (the formula is the same)
        isOrthMatrix = false; % the sensing matrix is partial orthogonal or not
        isEMUpatePerIter = false; % the expectation maximization is used per iteration or not
        isEMUpdateVar = false; % the expectation maximization is used for noise variance update or not
    end
    
    methods
        % Constructor
        function obj = AwgnModuleA(y, A, wvar)
            if nargin > 0
                obj.y = y;
                obj.A = A;
                obj.wvar = wvar;
            end
        end
        
        % Size
        function [nrow, ncol] = sizeInput(obj)
            [nrow, ncol] = size(obj.y);
        end
        
        % Estimator
        function [xApost, vApost, CApost] = estim(obj, xApri, vApri)
            [M, N, T] = size(obj.A);
            if obj.isOrthMatrix
                xApost = NaN*ones(N,T);
                vApost = NaN*ones(1,T);
                CApost = NaN*ones(N,N,T);
                for t = 1:T
                    xApri_t = xApri(:,t);
                    vApri_t = vApri(1,t);
                    A_t = obj.A(:,:,t);
                    wvar_t = obj.wvar(1,t);
                    y_t = obj.y(:,t);
                    
                    % xApost calculation
                    xApost_t = xApri_t + vApri_t/(vApri_t + wvar_t)*A_t'*(y_t - A_t*xApri_t);
                    xApost(:,t) = xApost_t;
                    
                    % vApost calculation
                    vApost(1,t) = vApri_t - M/N*vApri_t^2/(vApri_t + wvar_t);
                    
                    % CApost calculation
                    CApost(:,:,t) = vApost(1,t)*eye(N);
                end
            else
                xApost = NaN*ones(N,T);
                vApost = NaN*ones(1,T);
                CApost = NaN*ones(N,N,T);
                for t = 1:T
                    xApri_t = xApri(:,t);
                    vApri_t = vApri(1,t);
                    A_t = obj.A(:,:,t);
                    wvar_t = obj.wvar(1,t);
                    y_t = obj.y(:,t);
                    
                    % xApost calculation
                    xApost_t = xApri_t + vApri_t*eye(N)*A_t'*inv(A_t*(vApri_t*eye(N))*A_t'...
                        + wvar_t*eye(M))*(y_t - A_t*xApri_t);
                    xApost(:,t) = xApost_t;
                    
                    % vApost calculation
                    CApost_t = vApri_t*eye(N) - vApri_t*eye(N)*A_t'*inv(A_t*(vApri_t*eye(N))*A_t'...
                        + wvar_t*eye(M))*A_t*(vApri_t*eye(N));
                    vApost(1,t) = trace(CApost_t)/N;
                    
                    % CApost calculation
                    CApost(:,:,t) = CApost_t;
                end
            end
            
            if obj.isEMUpatePerIter && obj.isEMUpdateVar
%                 wvar0 = NaN*ones(1,T);
%                 for t = 1:T
%                     xApri_t = xApri(:,t);
%                     vApri_t = vApri(1,t);
%                     A_t = obj.A(:,:,t);
%                     y_t = obj.y(:,t);
%                     wvar_t = obj.wvar(1,t);
%                     for ii = 1:20
%                         xApost_t = xApri_t + vApri_t/(vApri_t + wvar_t)*A_t'*(y_t - A_t*xApri_t);
%                         vApost(1,t) = vApri_t - M/N*vApri_t^2/(vApri_t + wvar_t);
%                         CApost_t = vApost(1,t)*eye(N);
%                         wvar_t = real((norm(y_t - A_t*xApost_t,2))^2 + trace(A_t*CApost_t*A_t'))/M;
%                     end
%                     wvar0(1,t) = wvar_t;
%                 end
%                 obj.wvar = min(max(wvar0, obj.wvarMin), obj.wvarMax);
                
                wvar0 = NaN*ones(1,T);
                for t = 1:T
                    xApost_t = xApost(:,t);
                    CApost_t = CApost(:,:,t);
                    A_t = obj.A(:,:,t);
                    y_t = obj.y(:,t);
                    %wvar0(1,t) = real((norm(y_t - A_t*xApost_t,2))^2 + trace(A_t*CApost_t*A_t'))/M;
                    wvar0(1,t) = real((norm(y_t - A_t*xApost_t,2))^2)/M;
                end
                obj.wvar = min(max(wvar0, obj.wvarMin), obj.wvarMax);
            end
        end
        
        % Expectation maximization update
        function [] = expectMaxUpdate(obj, xApost, CApost)
            if obj.isEMUpdateVar
                [M, ~, T] = size(obj.A);
                wvar0 = NaN*ones(1,T);
                for t = 1:T
                    xApost_t = xApost(:,t);
                    CApost_t = CApost(:,:,t);
                    A_t = obj.A(:,:,t);
                    y_t = obj.y(:,t);
                    wvar0(1,t) = real((norm(y_t - A_t*xApost_t,2))^2 + trace(A_t*CApost_t*A_t'))/M;  
                    %wvar0(1,t) = real((norm(y_t - A_t*xApost_t,2))^2)/M;
                end
                obj.wvar = min(max(wvar0, obj.wvarMin), obj.wvarMax);
            end
        end
    end
    
end

