%% Pruebas PID
%
% Jorge F. García-Samartín
% www.gsamartin.es
% 2023-05-29

clear
close all

%% Configuración
Vobj = 8.5;
tol = 1e-5;
niter = 0;
error = 1e12;
maxAction = 400;
T = NaT(150, 1);
A = zeros(150);

%% PID
r = Robot();
r.Connect();

K = 500;
r.Measure();

Tor = datetime('now');
while abs(error) > tol && niter < 150
    niter = niter + 1;

    % Acción
    if error > 0 
        action = min(K*error, maxAction);
    else
        action = max(K*error, -maxAction);
    end
    r.WriteOneValveMillis(0, action);
    A(niter) = action;
    T(niter) = datetime('now');

    % Medida
    r.Measure()
    error = r.voltages(1,end) - Vobj;
end

T1 = seconds(T-Tor) * 1000;
figure;
subplot(1,2,1)
plot(T1(1:niter), r.voltages(1,3:end))
title('Measured voltages')

subplot(1,2,2)
plot(T1(1:niter), A(1:niter));
title('Milliseconds sent')