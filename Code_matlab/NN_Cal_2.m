MAX_MILLIS = 900;
N_ITER_MAX = 1050;
SAVE = 100;

pos = zeros(N_ITER_MAX,3);
vol = zeros(N_ITER_MAX,3);
t = zeros(N_ITER_MAX,3);

R.Measure();
pause(0.1)
    
pos_raw = R.CapturePosition();
vol_raw = R.getVoltages();
pos(1,:) = pos_raw(2,:);
vol(1,:) = vol_raw';

for i = 1:N_ITER_MAX
   
    while 1

        t(i,:) = -50 + 50*randi(fix((MAX_MILLIS/50 + 1)), [1 3]);
    
        indice = find(t(i,:) == min(t(i,:)));
    
        if (t(i,indice(1)) ~= 0) && (isempty(find(t(i,:) == 0, 1)))
            t(i,indice(1)) = 0;
        end

        if isempty(find(ismember(t(i,:),t(1:i-1,:),'rows') == 1,1))
            break;
        end
    end
    

    R.WriteSegmentMillis(t(i,:));
    pause(2 * MAX_MILLIS / 1000)
    R.Measure();
    pause(0.1)
    
    pos_raw = R.CapturePosition();
    vol_raw = R.getVoltages();

    pos(i,:) = pos_raw(2,:);
    vol(i,:) = vol_raw';

    if ~mod(i,SAVE)
        save(strcat('DatasetNN/prueba_',num2str(i/SAVE)),'pos','vol','t');
    end
    
    R.Deflate();
    pause(0.5)

end