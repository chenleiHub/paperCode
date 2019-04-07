function [ xBext ] = extrinsicEstim( xBpost, xBpri, vBpri, gX)
% extrinsicEstim
%  

[N, P] = size(xBpost);
delta = 1e-2;
Iter = 5;
div = NaN*ones(Iter,P);
cOptimal = NaN*ones(1,P);

for it = 1:Iter
    Ntilde = (randn(N,P) + 1i*randn(N,P))/sqrt(2);
    [xBpostTmp, ~, ~] = gX.estim(xBpri + delta*Ntilde, vBpri); % xBpostTmp NxP matrix, vBpostTmp 1xP matrix
    xBpostHat = xBpostTmp;
    diff = (xBpostHat - xBpost)/delta;
    div(it, :) = sum(diff.*Ntilde, 1);
end
aOptimal = mean(div, 1)/N;

for p = 1:P
    cOptimal(1,p) = real(xBpri(:,p)'*(xBpost(:,p) - aOptimal(1,p)*xBpri(:,p))) ...
        /((xBpost(:,p) - aOptimal(1,p)*xBpri(:,p))'*(xBpost(:,p) - aOptimal(1,p)*xBpri(:,p)));
end

xBext = (ones(N,1)*cOptimal).*(xBpost - (ones(N,1)*aOptimal).*xBpri);

end

