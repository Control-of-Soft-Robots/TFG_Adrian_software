function [punto, angulos, presiones, idx_] = buscador_posicion(data, punto, euler,index)

if euler ~= -1
    busqueda_angulo = true;
else
    busqueda_angulo = false;
end

if index ~= -1
    busqueda_indice = true;
else
    busqueda_indice = false;
end

if busqueda_indice == false
    if busqueda_angulo == false
        for i = 1:data.nummuestras  
               arr = [data.outputs(i,1), data.outputs(i,2),data.outputs(i,3) ; punto];
               distArr(i) = pdist(arr, 'euclidean');   
        end
            [~,idx] = min(distArr);
            idx_ = idx;
            punto = [data.outputs(idx,1), data.outputs(idx,2),data.outputs(idx,3)];
            angulos = [data.outputs(idx,4), data.outputs(idx,5),data.outputs(idx,6)];
            presiones = [data.inputs(idx,1),data.inputs(idx,2),data.inputs(idx,3),data.inputs(idx,4),data.inputs(idx,5),data.inputs(idx,6),data.inputs(idx,7),data.inputs(idx,8),data.inputs(idx,9) ];
        else
            punto_(1:3) = punto;
            punto_(4:6) = euler;
            for i = 1:data.nummuestras  
               arr = [data.outputs(i,1), data.outputs(i,1),data.outputs(i,1),data.outputs(i,4), data.outputs(i,5),data.outputs(i,6) ; punto_];
               distArr(i) = pdist(arr, 'euclidean');   
            end  
            [~,idx] = min(distArr);
            idx_ = idx;
            punto = [data.outputs(idx,1), data.outputs(idx,1),data.outputs(idx,1)];
            angulos = [data.outputs(idx,4), data.outputs(idx,5),data.outputs(idx,6)];
            presiones = [data.inputs(idx,1),data.inputs(idx,2),data.inputs(idx,3),data.inputs(idx,4),data.inputs(idx,5),data.inputs(idx,6),data.inputs(idx,7),data.inputs(idx,8),data.inputs(idx,9) ];
    end
else
    for idx = 1:data.nummuestras
        if data.index(idx) == index
            angulos = [data.outputs(idx,4), data.outputs(idx,5),data.outputs(idx,6)];
            presiones = [data.inputs(idx,1),data.inputs(idx,2),data.inputs(idx,3),data.inputs(idx,4),data.inputs(idx,5),data.inputs(idx,6),data.inputs(idx,7),data.inputs(idx,8),data.inputs(idx,9) ];
            punto = [data.outputs(idx,1), data.outputs(idx,1),data.outputs(idx,1)];
            idx_ = idx;
        end
    end
end
end