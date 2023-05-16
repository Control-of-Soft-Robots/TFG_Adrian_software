function [fitresult, gof] = createFit(L1, R1)
%CREATEFIT(L1,R1)
%  Create a fit.
%
%  Data for 'untitled fit 2' fit:
%      X Input: L1
%      Y Output: R1
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  See also FIT, CFIT, SFIT.

%  Auto-generated by MATLAB on 11-Apr-2023 13:32:53

load('car_sensor.mat')
L = ida_el(503:end);
V = ida(503:end);
L1 = L(1:167);
R1 = R(1:167);

%% Fit: 'untitled fit 2'.
[xData, yData] = prepareCurveData( L1, R1 );

% Set up fittype and options.
ft = fittype( '25*L0*(x+a*3.0289)/(L0-nu*((x+a*3.0289)-L0))^2', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [0.94 -Inf 0.1];
opts.StartPoint = [0.94 0.795199901137063 0.392227019534168];
opts.Upper = [Inf Inf 0.5];

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

% Plot fit with data.
figure( 'Name', 'untitled fit 2' );
h = plot( fitresult, xData, yData );
legend( h, 'R1 vs. L1', 'untitled fit 2', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 'L1', 'Interpreter', 'none' );
ylabel( 'R1', 'Interpreter', 'none' );
grid on

