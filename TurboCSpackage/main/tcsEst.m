function [ estFin ] = tcsEst(gA, gX, opt, state)
% tcsEst
% 

nit = opt.nit;
damping = opt.damping;
tol = opt.tol;
xTrue = opt.xTrue;
isErrorRecord = opt.isErrorRecord;
isShowResult = opt.isShowResult;
isDelayDenoiser = opt.isDelayDenoiser;

xApri = state.xApri; % NxT prior mean matrix
vApri = state.vApri; % 1xT prior variance matrix
M = state.M; % dimension of the measurement, e.g., Y is MxT matrix 
N = state.N; % dimension of the unknown matrix, e.g., X is NxT matrix
T = state.T; % dimension of the multiple measurement 



%% Main iterations
stop = false;
it = 0;
while ~stop
    
    % Iteration count
    it = it + 1;
    
    % Check for final iteration
    if it >= nit
        stop = true;
    end
    
    % Module A estimation, xApri, xApost NxT matrix, vApri, vApost, 1xT matrix 
    [xApost, vApost, CApost] = gA.estim(xApri, vApri);
    
    % Update extrinsic
    vAext = vApost.*vApri./(vApri - vApost); 
    vAext = min(1e11, max(1e-11, vAext));
    xAext = (xApost./(ones(N,1)*vApost) - xApri./(ones(N,1)*vApri)).*(ones(N,1)*vAext);
    if it > 1
        vAext = damping*vAext + (1-damping)*vAext_old;
        xAext = damping*xAext + (1-damping)*xAext_old;
    end
    vAext_old = vAext;
    xAext_old = xAext;
    vBpri = vAext;
    xBpri = xAext;
    %clear xAext vAext;

    % Module B estimation, xBpri, xBpost NxT matrix, vBpri, vBpost, 1xT matrix 
    [xBpost, vBpost, ~] = gX.estim(xBpri, vBpri);
    
    % Update extrinsic
    if isDelayDenoiser 
        xBext = extrinsicEstim(xBpost, xBpri, vBpri, gX);
        %xBext = Extrinsic_Est( xBpost, xBpri, vBpri, sigPar );
        wvar0 = gA.wvar;
        A0 = gA.A;
        y0 = gA.y;
        vBext = NaN*ones(1,T);
        for t = 1:T
            vBext(1,t) = ( norm(y0(:,t) - A0(:,:,t)*xBext(:,t))^2 - M*wvar0(1,t))/M;
        end
        vBext = min(1e11, max(1e-11, vBext));
    else
        vBext = vBpost.*vBpri./(vBpri - vBpost);
        vBext = min(1e11, max(1e-11, vBext));
        xBext = (xBpost./(ones(N,1)*vBpost) - xBpri./(ones(N,1)*vBpri)).*(ones(N,1)*vBext);
    end
    if it > 1
        vBext = damping*vBext + (1-damping)*vBext_old;
        xBext = damping*xBext + (1-damping)*xBext_old;
    end
    vBext_old = vBext;
    xBext_old = xBext;
    vApri = vBext;
    xApri = xBext;
    %clear xBext vBext;
    
    % Record estimation error per iteration
    if isErrorRecord
        if isDelayDenoiser
            estimError = (norm(xTrue - xApri,'fro')/norm(xTrue,'fro'))^2;
            estFin.errorRecord(it) = estimError;
            if isShowResult
                fprintf('Iter = %d, NMSE = %.4f, %.4f dB\n', it, estimError, 10*log10(estimError));
            end
        else
            estimError = (norm(xTrue - xBpost,'fro')/norm(xTrue,'fro'))^2;
            estFin.errorRecord(it) = estimError;
            if isShowResult
                fprintf('Iter = %d, NMSE = %.4f, %.4f dB\n', it, estimError, 10*log10(estimError));
            end
        end
    end
    
    % Check for convergence
    if (it > 1) && (stop==false)
        if norm(xBpost_old - xBpost, 'fro')/norm(xBpost, 'fro') < tol
            stop = true;
        end
    end
    xBpost_old = xBpost;
    
end % main iteration end

% Prepare for the output
estFin.xApri = xApri;
estFin.vApri = vApri;
estFin.xApost = xApost;
estFin.vApost = vApost;
estFin.CApost = CApost;
estFin.xBpri = xBpri;
estFin.vBpri = vBpri;
estFin.xBpost = xBpost;
%estFin.vBpost = vBpost;


end

