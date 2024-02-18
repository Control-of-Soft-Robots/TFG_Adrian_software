%% Measuring the sensors
function Measure(this)
    % Robot.Measure() sends the order to the microcontroller to
    % measure the sensors' values
    
    % Sending measure order
    writeline(this.serialDevice, "M");
end

function CallbackMeasurement(this)
    % Callback associated to the mesure
    %
    % Robot.CallbackMeasurement split the measures received 
    measurement = this.serialData;
    disp(measurement);
    measurement = split(measurement);
    measurement = str2double(measurement);
    measurement = measurement(2:1+this.nSensors);
    this.voltages(:,end+1) = measurement;
end

%% Kinematic modelling
function [l, params] = MCI(this, xP, phi0, a)
    % Calculate the inverse kinematic model of a three-wire robot 
    % using the PCC method.
    % 
    % l = Robot.MCI() returns the lengths of the three wires of a
    % robot, knowing the position of its end (x) and the diameter
    % of the circumference they form (a). Orientation may or may
    % not be included in x.
    % 
    % [l, params] = Robot.MCI() returns, in addition to the lengths
    % of the wires, a structure with the values of lr (average
    % length), phi (orientation) and kappa (curvature)
    % 
    % l = Robot.MCI(phi0) allows to rotate, counter-clockwise, an
    % angle phi0 the robot reference system.
    
    % Initial setup
    switch nargin
        case 1 
            xP = this.x(1:3);
            phi0 = pi/2;
            a = this.geom.phi0;
        
        case 2
            phi0 = this.geom.phi0;
            a = this.geom.radius;

        case 3
            a = this.geom.radius;
            
    end
    

    % Dependent modelling
    % General case
    if (sum(xP(1:2) - [0 0]))
        phi = atan2(xP(2),xP(1));
        kappa = 2 * norm(xP(1:2)) / norm(xP)^2;
        if xP(3) <= 0
            theta = acos(1 - kappa * norm(xP(1:2)));
        else
            theta = 2*pi - acos(1 - kappa * norm(xP(1:2)));
        end
        theta2 = wrapToPi(theta);
        lr = abs(theta2 / kappa);

    % Singular configuration
    else
        phi = 0;    % Cualquier valor es posible
        kappa = 0;
        lr = xP(3);            
    end

    params.lr = lr;
    params.phi = phi;
    params.kappa = kappa;

    %% Independent modelling
    phi_i = phi0 + [pi pi/3 -pi/3];
    l = lr * (1 + kappa*a*sin(phi + phi_i));
    this.l = l;
end


function [T, params] = MCD(this, l, a)  
    
    % Calcula el modelo cinemático directo de un robot de tres cables
    % utilizando el método PCC.
    %
    % T = MCD(l, a) devuelve la matriz de transformación homogénea que 
    % permite pasar de la base al extremo del robot, conocidas las 
    % longtiudes de sus cables (l) y el diámetro de la circunferencia 
    % que forman (a).
    % 
    % [T, params] = MCD(l, a) devuelve, además de la matriz de 
    % transformación homogénea, una estructura con los valores de 
    % lr (longitud media), phi(orientación) y kappa (curvatura).
    
    % Comprobaciones iniciales
    if length(l) ~= 3
        error("Introduce un vector de tres longitudes")
    end

    if nargin == 2
        a = this.geom.radius;
    end

    % Modelado dependiente
    % Caso general
    if ~all(l == l(1))
        lr = mean(l);
        phi = atan2(sqrt(3) * (-2*l(1) + l(2) + l(3)), 3 * (l(2) - l(3)));
        kappa = 2 * sqrt(l(1)^2 + l(2)^2 + l(3)^2 - l(1)*l(2) - ...
            l(3)*l(2) - l(1)*l(3)) / a / (l(1) + l(2) + l(3));
    % Posición singular
    else
        lr = l(1);
        phi = 0;
        kappa = 0;
    end

    params.lr = lr;
    params.phi = phi;
    params.kappa = kappa;

    % Modelado independiente
    if ~all(l == l(1))
        Trot = [cos(phi) -sin(phi) 0 0; 
            sin(phi) cos(phi) 0 0; 0 0 1 0; 0 0 0 1];
        Tarc = [cos(kappa*lr) 0 sin(kappa*lr) (1-cos(kappa*lr))/kappa; 
            0 1 0 0; -sin(kappa*lr) 0 cos(kappa*lr) sin(kappa*lr)/kappa; 
            0 0 0 1];
        T = Trot*Tarc;
    else
        T = [1 0 0 0; 0 1 0 0; 0 0 1 lr; 0 0 0 1];
    end

end

%% Neural Network
function [perform_pt, perform_vt] = NN_training(this, pos, volt, ...
    tiempo, capas_pt, capas_vt)
    % [perform_pt, perform_vt] = Robot.NN_training(pos, volt,
    % tiempo, capas_pt, capas_vt) trains and creates the two
    % required neuronal networks for the control system of the
    % robot

    n = fix(0.95*length(pos));

    this.net_pt = feedforwardnet(capas_pt);
    this.net_pt = train(this.net_pt,pos(1:n,:)',tiempo(1:n,:)');

    out_pt = this.net_pt(pos(n+1:end,:)');
    perform_pt = perf(this.net_pt,tiempo(n+1:end,:)',out_pt);

    this.net_vt = feedforwardnet(capas_vt);
    this.net_vt = train(this.net_vt,volt(1:n,:)',tiempo(1:n,:)');

    out_vt = this.net_vt(volt(n+1:end,:)');
    perform_vt = perf(this.net_vt,tiempo(n+1:end,:)',out_vt);
end

function NN_creation(this, network_pt, network_vt)
    % Robot.NN_creation(network_pt, network_vt) creates and
    % storages the already created neural networks

    this.net_pt = network_pt;
    this.net_vt = network_vt;
end


%% Control of a single segment
function [pos_final, error_pos] = Move(this, x)
    % [pos_final, error_pos] = Robot.Move(x) moves the robot to an
    % specified point (x) in the workspace of the robot
    
    niter = 0;
    action = zeros(1,3);
    err = [900 900 900];
    max_accion = 300;
    toler = 40;

    if length(x) ~= 3
        errordlg("Introduce un punto en el espacio " + ...
            "(vector fila de 3 componentes)","Execution Error");
        return
    end

    t_obj = this.net_pt(x');

    while (abs(err(1)) > toler || abs(err(2)) > toler || abs(err(3)) ...
            > toler) && niter < 10
        niter = niter + 1;
        this.Measure();
        pause(0.1)
        vol_current = this.getVoltages();

        t_current = this.net_vt(vol_current);
        err = t_obj - t_current;
        pause(0.1)

        for i = 1:3
            if err(i) > 0 
                action(i) = min(err(i), max_accion);
            else
                action(i) = max(err(i), -max_accion);
            end
        end
    
        this.WriteSegmentMillis(action);
        pause(0.5)
    end

    pos_final_raw = this.CapturePosition;
    pos_final = pos_final_raw(2,:);
    error_pos = norm(pos_final - x);

end

function [pos_final, error_pos, pos_inter] = Move_debug(this, x)
    % [pos_final, error_pos, pos_inter] = Robot.Move_debug(x) moves
    % the to an specified point (x) in the workspace of the robot,
    % but capturing every position in between the inicial point and 
    % the final point
    
    niter = 0;
    action = zeros(1,3);
    pos_inter = zeros(20,3);
    err = [900 900 900];
    max_accion = 300;
    toler = 20;

    if length(x) ~= 3
        errordlg("Introduce un punto en el espacio " + ...
            "(vector fila de 3 componentes)","Execution Error");
        return
    end

    t_obj = this.net_pt(x');

    while (abs(err(1)) > toler || abs(err(2)) > toler || abs(err(3)) ...
            > toler) && niter < 20
        niter = niter + 1;
        pos_raw = this.CapturePosition;
        pos_inter(niter,:) = pos_raw(2,:);

        this.Measure();
        pause(0.1)
        vol_current = this.getVoltages();

        t_current = this.net_vt(vol_current);
        err = t_obj - t_current;
        pause(0.1)

        for i = 1:3
            if err(i) > 0 
                action(i) = min(err(i), max_accion);
            else
                action(i) = max(err(i), -max_accion);
            end
        end
    
        this.WriteSegmentMillis(action);
        pause(0.5)
    end

    pos_final_raw = this.CapturePosition;
    pos_final = pos_final_raw(2,:);
    error_pos = norm(pos_final - x);

end