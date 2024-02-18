%% Class for robot control
%
% Jorge F. García-Samartín
% www.gsamartin.es
% 2023-05-16

classdef Robot < handle

    properties (Access = public)
        serialDevice;                       % Serial port to which Arduino is connected
        realMode = 1;                       % 1 if working with the real robot
        deflatingTime = 1500;               % Default deflating time
        deflatingRatio = 2;                 % Relation between deflation and inflation time
        maxAction = 200;                    % Maximum action of each valve in each iteration of the control loop
        max_millis = 1000;                  % Maximum pressure in each valve
        nValves = 9;                        % Number of valves
        nSensors = 3;                       % Number of sensors
        base = [0 0 0];                     % Position of the centre of the basis (in cameras'coordinates)
        bottom = [0 0 0];                   % Position of the centre of the bottom (in cameras'coordinates)
        origin = zeros(5,3);                % Initial positions and orientation of red, green and blue dots
        millisSentToValves;                 % Milliseconds of air each valve has received
        voltages;                           % Voltages read using the INAs
        positions;                          % Positions read using the cameras
        serialData = '';                    % Data received using the serial port
        max_min;                            % Upper and lower limits of the sensor's measurements
        tol = 20;                           % Tolerance of the error in the control loop
        net_pt;                             % Neuronal network with positions as inputs
        net_vt;                             % Neuronal network with voltages as inputs

        % Geometric parameters
        geom;
        R = [1 0 0;0 0 1;0 -1 0];           % Rotation matrix (from camera to our axes)

        % Cameras
        cam_xz;
        cam_yz;
        img_xz;
        img_yz;
        dx = 0;
        dy = 0;
        dz = 0;
        z_offset = 0;
        calOK;
        OP_mode = 0;
        cam;                            % For auxiliar parameters

        % Position and lengths
        x = zeros(6,1);
        l = zeros(3,1);
    end

    events
        NewMeasure
    end

    methods
        %% Constructor and destructor
        function this = Robot(nValves)
            % Class constructor
            %
            % Robot(nValves) creates the object to handle a robot with
            % nValves valves.
            %
            % Default values:
            % nValves: 9

            switch nargin
                case 1
                    this.nValves = nValves;
            end

            this.millisSentToValves = zeros(1, this.nValves);
            this.voltages = zeros(this.nSensors, 1);
            this.positions = zeros(6, 1);

            % Geometric parameters
            this.geom.trihDistance = 77;                    % Distance (mm) from the centre of the green ball to the centre of the bottom of the robot
            this.geom.segmLength = 90;                      % Length of a segment
            this.geom.radius = 40;                          % Radius of the sensors' circle
            this.geom.phi0 = pi/2;
            this.geom.zoff_modo2 = 7;
            this.geom.xoff_modo2 = 69;

            % Camera representation
            this.MakeAxis();

            % Measure callback
            addlistener(this, 'NewMeasure', @(~,~) this.CallbackMeasurement);
            
        end

        function this = delete(this)
            % Class destructor
            %
            % Robot.delete() destroys the object and deletes all its
            % properties.

            p = properties(this);

            for i = length(p)
                delete(p{i});
            end
        end

        %% Connection to Arduino
        function this = Connect(this, serial, freq)
            % Stablish connexion with the Arduino using the serial port
            % 
            % Robot.Connect(serial, freq) connects to the serial port specified
            % in serial at frequency freq.
            %
            % Default values:
            % serial: 'COM3'
            % freq: 9600

            switch nargin
                case 2
                    freq = 9600;
                case 1
                    freq = 9600;
                    slist = serialportlist();
                    serial = slist(1);
            end
    
            this.serialDevice = serialport(serial, freq);
            configureTerminator(this.serialDevice,"CR/LF")
            configureCallback(this.serialDevice, "terminator", @(varargin)this.ReadSerialData)

            % Sending info of the robot
            writeline(this.serialDevice, "i" + this.realMode);

        end

        function this = Disconnect(this)
            % Erase connexion with the Arduino
            %
            % Robot.Disconnect() clears the existent connection with the
            % Arduino and returns Robot.serialDevice value to 0

            delete(this.serialDevice)
            disp("Connection has been removed")

        end

        function Rearme(this)
            writeline(this.serialDevice, 'R');
        end
        
        %% Camera callibration and capture
        function MakeAxis(this)
            % Creates a subplot with two axis, in where to show the images
            % captured by the camera, in case of the default ones have been
            % closed.
            %
            % Robot.MakeAxis() creates the suplot in where the images of
            % the camera and the captured positions will be shown.
            
            figure;
            title('Images captured by the cameras')
            this.cam.UIAxes_yz = subplot(1,2,1);
            title(this.cam.UIAxes_yz,'CAM_YZ');
            this.cam.UIAxes_xz = subplot(1,2,2);
            title(this.cam.UIAxes_xz,'CAM_XZ');
        end

        function CalibrateCameras(this)
            % Calculates the extrinsic parameters of the cameras and loads
            % the intrinsic ones.
            %
            % Robot.CallibrateCameras() returns in variable this.cam all
            % the necessary parameters to work with the cameras and two
            % figures displaying its possition relative to the chessboard

            % Select the cameras
            nCams = 0;
            for i = 1:length(webcamlist)
                if strcmp(webcam(i).Name, 'USB Live camera') && ~nCams
                    this.cam_xz = webcam(i);
                    nCams = 1;
                    continue
                end
                if strcmp(webcam(i).Name, 'USB Live camera') && nCams == 1
                    this.cam_yz = webcam(i);
                    break
                end
            end

            % New reference frame
            this.OP_mode = 0;
           
            % Extrinsic parameters
            tempX = load('par.mat','cameraParams_yz');
            this.cam.cameraParams_yz = tempX.cameraParams_yz;
            tempX = load('par.mat','cameraParams_xz');
            this.cam.cameraParams_xz = tempX.cameraParams_xz;
            [this.cam.rotyz, this.cam.transyz] = findCameraPose(this.cam_yz, this.cam.cameraParams_yz);
            [this.cam.rotxz, this.cam.transxz] = findCameraPose(this.cam_xz, this.cam.cameraParams_xz);
            disp('Snapshots have been taken')
            
            % Figure with cameras' position
            figure;
            this.cam.matyz = cameraMatrix(this.cam.cameraParams_yz, this.cam.rotyz, this.cam.transyz);
            this.cam.matxz = cameraMatrix(this.cam.cameraParams_xz, this.cam.rotxz, this.cam.transxz);   
            this.cam.BLamp.Color = "Green";
            hold off
        end

        function [pos_rel, pos_fixed, nattempts] = CapturePosition(this)
            % Capture pictures with the cameras and returns end tip
            % position.
            % 
            % The posiition of the centre of the bottom and the Euler
            % angles are also stored in the last column of this.positions array
            %
            % pos = Robot.CapturePosition() Capture pictures with the
            % cameras and returns matrix pos, which contains, by rows
            % the position of the green, red and blue dots, the estimated
            % position of the centre of the bottom and the Euler angles,
            % expressed with respect to the centre of the robot base.
            % 
            % [pos2, pos] = Robot.CapturePosition() also returns the
            % position expressed with respect to the cameras' reference
            % frame.
            
            nattempts = 0;
            pos = [-1 -1 -1];

            if ~isgraphics(this.cam.UIAxes_xz) || ~isgraphics(this.cam.UIAxes_yz)
                this.MakeAxis();
            end

            while find(pos == [-1 -1 -1])
                
                nattempts = nattempts + 1;

                % Take pictures with both cameras
                this.img_yz = snapshot(this.cam_yz);
                this.img_xz = snapshot(this.cam_xz);
                imshow(this.img_yz,'Parent',this.cam.UIAxes_yz);
                imshow(this.img_xz, 'Parent', this.cam.UIAxes_xz);
                
                % Searching for red, green and blue dots in the images
                yz_r = findPoint(this.img_yz, 'o',"yz");
                yz_g = findPoint(this.img_yz, 'g',"yz");
                yz_b = findPoint(this.img_yz, 'b',"yz");
                xz_r = findPoint(this.img_xz, 'o',"xz");
                xz_g = findPoint(this.img_xz, 'g',"xz");
                xz_b = findPoint(this.img_xz, 'b',"xz");
    
                % Plotting
                hold(this.cam.UIAxes_yz, 'on')
                hold(this.cam.UIAxes_xz, 'on')
                plot(yz_g(1) ,yz_g(2), 'o', 'Parent',this.cam.UIAxes_yz, 'Color','green');
                plot(xz_g(1), xz_g(2), 'o', 'Parent',this.cam.UIAxes_xz, 'Color','green')
                plot(yz_b(1), yz_b(2), 'o', 'Parent',this.cam.UIAxes_yz, 'Color','red');
                plot(xz_b(1), xz_b(2), 'o', 'Parent',this.cam.UIAxes_xz, 'Color','red')
                plot(yz_r(1), yz_r(2), 'o', 'Parent',this.cam.UIAxes_yz, 'Color','red');
                plot(xz_r(1), xz_r(2), 'o', 'Parent',this.cam.UIAxes_xz, 'Color','red')
                plot([yz_b(1) yz_g(1)], [yz_b(2) yz_g(2)], 'g', 'Parent',this.cam.UIAxes_yz);
                plot([yz_b(1) yz_r(1)], [yz_b(2) yz_r(2)], 'g', 'Parent',this.cam.UIAxes_yz);
                plot([yz_g(1) yz_r(1)], [yz_g(2) yz_r(2)], 'g', 'Parent',this.cam.UIAxes_yz);
                plot([xz_b(1) xz_g(1)], [xz_b(2) xz_g(2)], 'g', 'Parent',this.cam.UIAxes_xz);
                plot([xz_b(1) xz_r(1)], [xz_b(2) xz_r(2)], 'g', 'Parent',this.cam.UIAxes_xz);
                plot([xz_g(1) xz_r(1)], [xz_g(2) xz_r(2)], 'g', 'Parent',this.cam.UIAxes_xz);
                 
                this.cam.r = getCoordinates(yz_r,xz_r,this.cam.matyz, this.cam.matxz, this.dx, this.dy, this.dz, this.z_offset);
                this.cam.g = getCoordinates(yz_g,xz_g,this.cam.matyz, this.cam.matxz, this.dx, this.dy, this.dz, this.z_offset);
                this.cam.b = getCoordinates(yz_b,xz_b,this.cam.matyz, this.cam.matxz, this.dx, this.dy, this.dz, this.z_offset);
                
                hold on
    
                [this.cam.rot_m, this.cam.euler_m] = getRotations(this.cam.r, this.cam.g, this.cam.b);
                
                % From rotation matrix to quaternion
                quaternion = rotm2quat(this.cam.rot_m);
                disp(quaternion);
                hold(this.cam.UIAxes_yz, 'on');
                    
                % From rotation matrix to Euler angles
                euler = rotm2eul(this.cam.rot_m);
    
                hold(this.cam.UIAxes_yz, 'on');
                
                rv = [this.cam.r(1) - this.cam.g(1), this.cam.r(2) - this.cam.g(2), this.cam.r(3) - this.cam.g(3)];
                bv = [this.cam.b(1) - this.cam.g(1), this.cam.b(2) - this.cam.g(2), this.cam.b(3) - this.cam.g(3)];
                zv = cross(rv, bv);
                rv = rv / norm(rv);
                bv = bv / norm(bv);
                zv = zv / norm(zv);
    
                quiver3(this.cam.g(1), this.cam.g(2), this.cam.g(3), rv(1), rv(2), rv(3), 'Parent', this.cam.UIAxes_yz, 'Color', 'red');
                quiver3(this.cam.g(1), this.cam.g(2), this.cam.g(3), bv(1) , bv(2), bv(3), 'Parent', this.cam.UIAxes_yz, 'Color', 'blue');
                quiver3(this.cam.g(1), this.cam.g(2), this.cam.g(3), zv(1) , zv(2), zv(3), 'Parent', this.cam.UIAxes_yz, 'Color', 'green');
    
                valuestr = "Green: " + num2str(this.cam.g(1)) + ", " + num2str(this.cam.g(2)) + ", " + num2str(this.cam.g(3)) + newline + "red: " + num2str(this.cam.r(1)) + ", " + num2str(this.cam.r(2)) + ", " + num2str(this.cam.r(3)) + newline + "blue: " + num2str(this.cam.b(1)) + ", " + num2str(this.cam.b(2)) + ", " + num2str(this.cam.b(3)) + newline + "euler: " + num2str(rad2deg(euler(1))) + ", " + num2str(rad2deg(euler(2))) + ", " + num2str(rad2deg(euler(3)))+   newline +    " dy = " + num2str(this.dy) + "" +  " dx = " + num2str(this.dx);
                disp(valuestr);
    
                hold off
                hold off  

%                 pos = [this.cam.g; this.cam.r; this.cam.b];
                pos = this.cam.g;
            
                % Warning
                if nattempts > 10
                    disp("10 attempts of capture have been done and failed")
                    break
                end

            end

            % Coordinate system transformation 
            if (this.OP_mode == 0)
                this.bottom = pos(1,:) + [0 -this.geom.trihDistance 0];
                this.base = this.bottom + [0 -this.geom.segmLength 0];
                this.origin = [pos; this.bottom; euler];
                this.OP_mode = 2;
            elseif (this.OP_mode == 1)
                this.bottom = pos(1,:) + [0 -this.geom.zoff_modo2 -this.geom.xoff_modo2];
                this.base = this.bottom + [0 -this.geom.segmLength 0];
                this.origin = [pos; this.bottom; euler];
                this.OP_mode = 2;
            end

            %a = this.bottom + nanmean(removeOutliers(pos - this.origin(1:3,:)));
            a = this.bottom + (pos - this.origin(1,:));
            pos_fixed = [pos; a; euler];
            pos_rel = [pos_fixed(1:2,:) - this.base; euler];

            % Storing
            this.positions(:,end+1) = [pos_rel(2,:) euler]';

        end

        function Set_OP_mode(this,op)
            this.OP_mode = op;
        end
        
        %% Sending pressure to valves
        function Deflate(this)
            % Robot.Deflate() deflates all the valves, sending negative
            % pressure during the time specified in this.deflatingTime,
            % which default value is 1000

            deflateMillis = "w,1";
            for i = 1:this.nValves
                deflateMillis = strcat(deflateMillis, ",-", int2str(this.deflatingTime * this.deflatingRatio));
            end
            writeline(this.serialDevice, deflateMillis);

            pause(this.deflatingTime / 500);        % Pause works with seconds. A pause of 2 times the deflating time is done
        end

        function WriteOneValveMillis(this, valv, millis)
            % Robot.WriteOneValveMillis(valv, millis) sends to valve valv
            % air during the time specified in millis.
            %
            % The value of millis can  be possitive (inflating the valve)
            % or negative (deflating the valve). Negativa values are
            % multiplied by Robot.deflatingRatio

            if millis > 0
                writeline(this.serialDevice, "f," + int2str(valv) + "," + int2str(millis));
            else
                writeline(this.serialDevice, "e," + int2str(valv) + "," + int2str(-millis * this.deflatingRatio));
            end

            this.millisSentToValves(valv+1) = this.millisSentToValves(valv+1) + millis;
        end

        function WriteSegmentMillis(this, millis)
            % Robot.WriteSegmentMillis(millis) sends to one whole segment
            % an array of times, specified in millis

            % Values can be positive or negative

            for i = 0:2
                this.WriteOneValveMillis(i,millis(i+1));
            end
        end
        
        %% Reading valve state and sensors
        function millis = GetMillisSent(this)
            % millis = GetMillisSent() returns a 1 x Robot.nValves array 
            % containing the volume of air sent to each valve.
            % 
            % The values stored in millis are the values sent by the user,
            % not the ones which have been really sent (they ARE NOT
            % multplied by Robot.deflatingRatio).

            millis = this.millisSentToValves;

        end

        function vol = getVoltages(this)
            % vol = Robot.getVoltages() returns the last measurement of the
            % sensors

            vol = this.voltages(:,end);
        end

        function millis = GetMillis(this)
            % millis = GetMillis() returns a 1 x Robot.nValves array 
            % containing the volume of air sent to each valve.
            % 
            % The values stored in millis are the values read after
            % communicating with the Arduino (they ARE multplied by
            % Robot.deflatingRatio).

            write(this.serialDevice, 'r', "char");
            millis = str2double(readline(this.serialDevice));
            
        end

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

        function ResetVoltagesPositions(this)
            % Reset the values of Robot.voltages and Robot.positions to
            % default (Robot.voltages will be a nSensors x 1 array and
            % Robot.positions, a 6 x 1 array)
          
            this.voltages = zeros(this.nSensors, 1);
            this.positions = zeros(6, 1);
        end

        function data = ReadSerialData(this)
            % Callback associated to the serial port specified in
            % Robot.serialDevice
            % 
            % data = Robot.ReadSerialData() returns a string containing the data
            % read by the serial port.

            data = readline(this.serialDevice);
            this.serialData = data;
            dataChar = char(data);

            switch dataChar(1)
                case 'M'
                    notify(this, 'NewMeasure');
            end
        end

        %% Sensor calibration
        function CalibrateSensor(this, nSensor)
            % Robot.CalibrateSensor(nSensor) finds the maximum and the
            % minimum values of the measurement of each sensor

            this.Measure
            for i = 1:100000000
            end
            this.max_min(3,1) = this.voltages(3,end);

            this.WriteOneValveMillis(nSensor, this.max_millis);
            pause(4)
            this.Measure
            for i = 1:100000000
            end
            this.max_min(nSensor + 1,2) = this.voltages(nSensor + 1,end);
            this.Deflate();

        end

        function Calibrate(this)
            % Robot.Calibrate() calibrate a whole segment using the method
            % CalibrateSensor

            for k = 0:2
                this.CalibrateSensor(k);
            end

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
            % T = MCD(l, a) devuelve la matriz de transformación homogénea que permite
            % pasar de la base al extremo del robot, conocidas las longtiudes de sus
            % cables (l) y el diámetro de la circunferencia que forman (a).
            % 
            % [T, params] = MCD(l, a) devuelve, además de la matriz de transformación
            % homogénea, una estructura con los valores de lr (longitud media), phi
            % (orientación) y kappa (curvatura).
            
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
                kappa = 2 * sqrt(l(1)^2 + l(2)^2 + l(3)^2 - l(1)*l(2) - l(3)*l(2) - l(1)*l(3)) / a / (l(1) + l(2) + l(3));
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
                Trot = [cos(phi) -sin(phi) 0 0; sin(phi) cos(phi) 0 0; 0 0 1 0; 0 0 0 1];
                Tarc = [cos(kappa*lr) 0 sin(kappa*lr) (1-cos(kappa*lr))/kappa; 0 1 0 0; -sin(kappa*lr) 0 cos(kappa*lr) sin(kappa*lr)/kappa; 0 0 0 1];
                T = Trot*Tarc;
            else
                T = [1 0 0 0; 0 1 0 0; 0 0 1 lr; 0 0 0 1];
            end
        
        end

        %% Neural Network
        function [perform_pt, perform_vt] = NN_training(this, pos, volt, tiempo, capas_pt, capas_vt)
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
                errordlg("Introduce un punto en el espacio (vector fila de 3 componentes)","Execution Error");
                return
            end

            t_obj = this.net_pt(x');

            while (abs(err(1)) > toler || abs(err(2)) > toler || abs(err(3)) > toler) && niter < 10
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
                errordlg("Introduce un punto en el espacio (vector fila de 3 componentes)","Execution Error");
                return
            end

            t_obj = this.net_pt(x');

            while (abs(err(1)) > toler || abs(err(2)) > toler || abs(err(3)) > toler) && niter < 20
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

    end
end