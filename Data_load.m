function Start_Data= Data_load(File_path)
Property= {'RPM50'; 'Throttle'; 'Engine_Load'; 'Air_Temp'; 'Coolant'; 'Rotor_Air_Temp'; 'Fuel_PW_10';...
    'Fuel_Pressure'; 'Volt_12'};
Property_index= [3, 4, 5, 6, 7, 8, 9, 12, 14];
Data_origin= readtable(File_path);
TimeStamp= Date2Time(Data_origin(:, 1));
Time= TimeStamp- TimeStamp(1);
% Data_name= ['Engine_Data_',num2str(i-2)];
Engine_Data= table(Time);
for j= 1: length(Property)
    eval(['Engine_Data.',Property{j}, '=', 'Data_origin{:,',num2str(Property_index(j)),'};']);
end
% 提取引擎啟動階段之資料
Start_idx= find(Engine_Data.RPM50>= 900);
if isempty(Start_idx)
    Time= (0:1:123)';
    Start_Data= table(Time);
    for j= 1:length(Property)
        Start_property= zeros(length(Time), 1);
        eval(['Start_Data.',Property{j}, '=', 'Start_property;']);
    end
else
    Start_time= Time(Start_idx);
    Time= Start_time- Start_time(1);
    Start_Data= table(Time);
    for j= 1:length(Property)
        Start_property= Engine_Data{Start_idx, j+1};
        eval(['Start_Data.',Property{j}, '=', 'Start_property;']);
    end
end
end