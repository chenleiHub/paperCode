classdef tcsOption
    % tcsOption
    % This is a class which includes the parameter settings of Turbo-CS
    % based algorithms.
    
    properties
        nit = 100; % Default iterations
        
        % Damping is widely used in the message passing or belief
        % propagation based algorithms. A smaller factor can improve the
        % convergence (e.g. 0.95). Damping is not used in default settings.
        damping = 1; 
        
        % When the residual is smaller than tol, the algorithm will stop.
        tol = -1; % continue until default iterations
        
        % True signal is kept for error calculation per iteration.
        xTrue = [];
        
        % If use delay domain denoiser or not.
        isDelayDenoiser = false;
        
        % If print result per iteration or not.
        isShowResult = false;
        
        % If save estimation error per iteration or not.
        isErrorRecord = false;
    end
    
    methods
        % Constructor
        function opt = tcsOption()
        end
    end
    
end

