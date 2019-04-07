function [ estFinOut ] = EMTurboCS( gA, gX, tcsOpt, tcsState, outIter )
% EMTurboCS
% In this function, Turbo Compressed Sensing based algorithms are divided
% into inner iteration and outer iteration. The two stage iteration can
% avoid time consuming EM update per iteration and make the algorithm more 
% robust.  

    [~, N, T] = size(gA.A);

    for it = 1:outIter
        % Inner iteration
        [estFin] = tcsEst(gA, gX, tcsOpt, tcsState);
        
        % Expectation maximization update
        if it < outIter
            gA.expectMaxUpdate(estFin.xApost, estFin.CApost);
            gX.expectMaxUpdate(estFin.xBpri, estFin.vBpri);
        end
        
        % Reinitialization
%         state.xApri = estFin.xApri;
%         state.vApri = estFin.vApri;
        tcsState.xApri = zeros(N,T);
        tcsState.vApri = gX.xvar;
        
    end
    
    estFinOut.xBpost = estFin.xBpost;

end

