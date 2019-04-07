classdef tcsState
    % tcsState
    % This is a class which includes the initial value and state of the
    % algorithm.
    
    properties
        M = NaN; % Number of measurement
        
        N = NaN; % Number of unknown signal
        
        T = NaN; % Number of multiple measurement
        
        xApri = []; % NxT mean value matrix to initialize the algorithm
        
        vApri = []; % 1xT variance matrix to initialize the algorithm
        
    end
    
    methods
        % Constructor
        function state = tcsState()
        end
    end
    
end

