%% Modelo PCC en 3D. Basado en CerrilloVacas2023 y Jones2006
%
% Calcula el modelo cinemático inverso de un robot de tres cables
% utilizando el método PCC.
%
% l = MCI(x, a) devuelve las longitudes que tendrían los tres cables de un
% robot, conocida la posición de su extremo (x) y el diámetro de la
% circunferenca que forman estos (a). En x se puede incluir o no orientación.
% 
% [l, params] = MCI(x, a) devuelve, además de las longitudes de los cables,
% una estructura con los valores de lr (longitud media), phi
% (orientación) y kappa (curvatura)
%
% l = MCI(x, a, phi0) permite girar, en sentido antihorario, un ángulo phi0
% el sistema de referencia del robot.
%
% Jorge F. García-Samartín
% www.gsamartin.es
% 2023-05-04

function [l, params] = MCI(x, a, phi0)  

    %% Comprobaciones iniciales
    switch nargin
        case 2
            phi0 = pi/2;
        case 1
            phi0 = pi/2;
            a = 40;
    end

    if length(x) < 3
        error("Introduce un vector de, al menos, tres longitudes")
    end
    x = x(1:3);

    %% Modelado dependiente
    % Caso general
    if (sum(x(1:2) - [0 0]))
        phi = atan2(x(2),x(1));
        kappa = 2 * norm(x(1:2)) / norm(x)^2;
        if x(3) <= 0
            theta = acos(1 - kappa * norm(x(1:2)));
        else
            theta = 2*pi - acos(1 - kappa * norm(x(1:2)));
        end
        theta2 = wrapToPi(theta);
        lr = abs(theta2 / kappa);
    % Configuración singular
    else
        phi = 0;    % Cualquier valor es posible
        kappa = 0;
        lr = x(3);            
    end

    params.lr = lr;
    params.phi = phi;
    params.kappa = kappa;

    %% Modelado independiente
    phi_i = phi0 + [pi pi/3 -pi/3];
    l = lr * (1 + kappa*a*sin(phi + phi_i));

end