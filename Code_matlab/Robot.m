%% Class for robot control
%
% Jorge F. García-Samartín
% www.gsamartin.es
% 2023-05-16

classdef Robot < handle

    properties (Access = public)
        serialDevice;                   % Serial port to which Arduino is connected
        deflatingTime = 1000;           % Default deflating time
        deflatingRatio = 1.4;           % Relation between deflation and inflation time
        nValves = 9;                    % Number of valves
        millisSentToValves;             % Milliseconds of air each valve has received

        % Cameras
        cam_xz;
        cam_yz;
        img_xz;
        img_yz;
        dx = 0;
        dy = 0;
        dz = 0;
        z_offset = 0;
        cam;                            % For auxiliar parameters
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

            % Camera representation
            figure;
            title('Images captured by the cameras')
            this.cam.UIAxes_yz = subplot(1,2,1);
            this.cam.UIAxes_yz.title = 'Camera_YZ';
            this.cam.UIAxes_xz = subplot(1,2,2);
            this.cam.UIAxes_xz.title = 'Camera_XZ';
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
            % Stablish conexion with the Arduino using the serial port
            % 
            % Robot.Connect(serial, freq) connects to the serial port specified
            % in serial at frequency freq.
            %
            % Default values:
            % serial: 'COM3'
            % freq: 9600

            switch nargin
                case 1
                    freq = 9600;
                case 0
                    freq = 3600;
                    serial = 'COM3';
            end
    
            this.serialDevice = serialport(serial, freq);
        end
        
        %% Camera callibration and capture
        function CallibrateCameras(this)
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
           
            % Extrinsic parameters
            tempX = load('par.mat','cameraParams_yz');
            this.cam.cameraParams_yz = tempX.cameraParams_yz;
            tempX = load('par.mat','cameraParams_xz');
            this.cam.cameraParams_xz = tempX.cameraParams_xz;
            [this.cam.rotyz, this.cam.transyz] = findCameraPose(this.cam_yz, this.cam.cameraParams_yz);
            [this.cam.rotxz, this.cam.transxz] = findCameraPose(this.cam_xz, this.cam.cameraParams_xz);
            
            % Figure with cameras' position
            figure;
            this.cam.matyz = cameraMatrix(this.cam.cameraParams_yz, this.cam.rotyz, this.cam.transyz);
            this.cam.matxz = cameraMatrix(this.cam.cameraParams_xz, this.cam.rotxz, this.cam.transxz);   
            this.cam.BLamp.Color = "Green";
            hold off
        end

        function SetZero(this)
            % Transform rotation matrixes of the cameras to make the
            % original green dot the home position

            % Taking images with the cameras
            this.img_yz = snapshot(this.cam_yz);
            this.img_xz = snapshot(this.cam_xz);
            imshow(this.img_yz,'Parent',this.cam.UIAxes_yz);
            imshow(this.img_xz, 'Parent', this.cam.UIAxes_xz);
            
            % Searching for red, green and blue dots
            yz_r = findPoint(this.img_yz, 'o',"yz");
            yz_g = findPoint(this.img_yz, 'g',"yz");
            yz_b = findPoint(this.img_yz, 'b',"yz");
            xz_r = findPoint(this.img_xz, 'o',"xz");
            xz_g = findPoint(this.img_xz, 'g',"xz");
            xz_b = findPoint(this.img_xz, 'b',"xz");

            % Coordinates of these points
            this.cam.r = getCoordinates(yz_r,xz_r,this.cam.matyz, this.cam.matxz, this.dx, this.dy, this.dz, this.cam.z_offset);
            this.cam.g = getCoordinates(yz_g,xz_g,this.cam.matyz, this.cam.matxz, this.dx, this.dy, this.dz, this.cam.z_offset);
            this.cam.b = getCoordinates(yz_b,xz_b,this.cam.matyz, this.cam.matxz, this.dx, this.dy, this.dz, this.cam.z_offset);

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
            
            this.dx = this.cam.g(1);
            this.dy = this.cam.g(2);
            this.dz = this.cam.g(3);

        end

        function CapturePosition(this)
            % Capture pictures with the cameras and returns end tip
            % position.
            %
            
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

        end
        
        %% Sending pressure to a valve
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
                writeline(this.serialDevice, "e" + int2str(valv) + "," + int2str(-millis * this.deflatingRatio));
            end

            this.millisSentToValves(valv) = this.millisSentToValves(valv) + millis;
        end
        
        %% Reading valve state
        function millis = GetMillisSent(this)
            % millis = GetMillisSent() returns a 1 x Robot.nValves array 
            % containing the volume of air sent to each valve.
            % 
            % The values stored in millis are the values sent by the user,
            % not the ones which have been really sent (they ARE NOT
            % multplied by Robot.deflatingRatio).

            millis = this.millisSentToValves;

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
    end
end