%% load data
close all; clear; clc
% load Param_1110323(第一季保) data
Folder_dir= dir('Param_1110323');
Property= {'RPM50'; 'Throttle'; 'Engine_Load'; 'Air_Temp'; 'Coolant'; 'Rotor_Air_Temp'; 'Fuel_PW_10';...
    'Fuel_Pressure'; 'Volt_12'};
Property_name= {'RPM50'; 'Throttle'; 'Engine Load'; 'Air Temp'; 'Coolant'; 'Rotor Air Temp'; 'Fuel PW 10';...
    'Fuel Pressure'; 'Volt 12'};
for i= 3:length(Folder_dir)
    File_path= [Folder_dir(i).folder,'\',Folder_dir(i).name];
    eval(['Start_1S_',num2str(i-2), '= Data_load(File_path);']);
    idx_ID= find(Folder_dir(i).name=='(');
    ID_1(i-2, :)= [Folder_dir(i).name(idx_ID+1:idx_ID+6)];
end
Fault_index_1= [2, 19, 37]; Normal_index_1= 1:49; Normal_index_1(Fault_index_1)= []; 
Fault_type= ["油門角超出標準3%"; "啟動後沒多久熄火"; "致動器異常"; "油壓低於標準值"; "滑油滲漏量稍多";...
    "箱內啟動器故障"]; % 箱內啟動器故障 只有一筆數據

% load 第二次季保 data
Folder_dir= dir('第二季保');
for i= 3:length(Folder_dir)
    File_path= [Folder_dir(i).folder,'\',Folder_dir(i).name];
    eval(['Start_2S_',num2str(i-2), '= Data_load(File_path);']);
    idx_ID= find(Folder_dir(i).name=='(');
    ID_2(i-2, :)= [Folder_dir(i).name(idx_ID+1:idx_ID+6)];
%     ID_testing(i-2, :)= [Folder_dir(i).name(end-11:end-6)];
end

% Fault_data_ID= ['FA3012'; 'FD3003'; 'FA3038'];
Fault_index_2= [36, 37, 42]; Normal_index_2= 1:49; Normal_index_2(Fault_index_2)= []; 

% load 第三次收取 data
Folder_dir= dir('SXW11042');
for i= 3:length(Folder_dir)
    File_path= [Folder_dir(i).folder,'\',Folder_dir(i).name];
    eval(['Start_3S_',num2str(i-2), '= Data_load(File_path);']);
    idx_ID= find(Folder_dir(i).name=='(');
    ID_3(i-2, :)= [Folder_dir(i).name(idx_ID+1:idx_ID+6)];
end
Fault_index_3= [6, 55, 72]; Normal_index_3= 1:74; Normal_index_3(Fault_index_3)= []; 
%% Data preprocessing- z score
clc
% CV1
% % training related
% ID_training= [ID_1; ID_2]; train_file= {'Start_1S_'; 'Start_2S_'}; 
% train_num= [size(ID_1,1); size(ID_2,1)];
% Normal_data_index= [Normal_index_1, Normal_index_2+ length(ID_1)]; %Normal data index for training data
% Fault_data_index= [Fault_index_1, Fault_index_2+ length(ID_1)];
% N= size(ID_training,1); % number of training data
% M= length(Property); % number of engine parameters
% K= 6; % number of feature
% N_G= 3; % number of gaussian models for GMM
% F1_idx_train= 2; F2_idx_train= 19; F3_idx_train= [37, length(ID_1)+ 42]; F4_idx_train= length(ID_1)+ [36, 37];
% F5_idx_train= []; F6_idx_train= [];
% % testing related
% ID_testing= ID_3; test_file= {'Start_3S_'}; test_num= size(ID_3,1);
% N_t= size(ID_testing,1); % number of testing data
% F1_idx_test= []; F2_idx_test= []; F3_idx_test= []; F4_idx_test= []; F5_idx_test= [55]; F6_idx_test= [6, 72];

% CV2
% training related
% ID_training= [ID_2; ID_3]; train_file= {'Start_2S_'; 'Start_3S_'}; 
% train_num= [size(ID_2,1); size(ID_3,1)];
% Normal_data_index= [Normal_index_2, Normal_index_3+ length(ID_2)]; %Normal data index for training data
% Fault_data_index= [Fault_index_2, Fault_index_3+ length(ID_2)];
% N= size(ID_training,1); % number of training data
% M= length(Property); % number of engine parameters
% K= 6; % number of feature
% N_G= 3; % number of gaussian models for GMM
% F1_idx_train= []; F2_idx_train= []; F3_idx_train= [42]; F4_idx_train= [36, 37];
% F5_idx_train= [55]+ length(ID_2); F6_idx_train= [6, 72]+ length(ID_2);
% % testing related
% ID_testing= ID_1; test_file= {'Start_1S_'}; test_num= size(ID_1,1);
% N_t= size(ID_testing,1); % number of testing data
% F1_idx_test= [2]; F2_idx_test= [19]; F3_idx_test= [37]; F4_idx_test= []; F5_idx_test= []; F6_idx_test= [];

% CV3
% training related
ID_training= [ID_1; ID_3]; train_file= {'Start_1S_'; 'Start_3S_'}; 
train_num= [size(ID_1,1); size(ID_3,1)];
Normal_data_index= [Normal_index_1, Normal_index_3+ length(ID_1)]; %Normal data index for training data
Fault_data_index= [Fault_index_1, Fault_index_3+ length(ID_1)];
N= size(ID_training,1); % number of training data
M= length(Property); % number of engine parameters
K= 6; % number of feature
N_G= 3; % number of gaussian models for GMM
F1_idx_train= [2]; F2_idx_train= [19]; F3_idx_train= [37]; F4_idx_train= [];
F5_idx_train= [55]+ length(ID_1); F6_idx_train= [6, 72]+ length(ID_1);
% testing related
ID_testing= ID_2; test_file= {'Start_2S_'}; test_num= size(ID_2,1);
N_t= size(ID_testing,1); % number of testing data
F1_idx_test= []; F2_idx_test= []; F3_idx_test= [42]; F4_idx_test= [36, 37]; F5_idx_test= []; F6_idx_test= [];

cnt= 1;
for j= 1:size(train_num, 1)
    ID_num= train_num(j);
    for i= 1:ID_num
        eval(['Start_Training_',num2str(cnt), '= ',train_file{j}, num2str(i),';']);
        cnt= cnt+1;
    end
end

for i= 1:N_t
     eval(['Start_Testing_',num2str(i), '= ', test_file{:}, num2str(i),';']);
end

Mu_z= zeros(M,K); Sigma_z= zeros(M,K);
for j= 1:M
    for i= 1: N
        eval(['Start_', Property{j}, '_Feature(i,:)= ', 'Feature_Extract(Start_Training_', num2str(i),...
            '.', Property{j}, ');']);
    end
    % standardize (use normal data mean and standard deviation)
    eval(['Feature_normal= Start_', Property{j}, '_Feature(Normal_data_index, :);']);
    for i= 1: K
        Mu_z(j, i)= mean(Feature_normal(:, i)); Sigma_z(j, i)= std(Feature_normal(:, i), 1);
        eval(['Start_', Property{j}, '_Feature(:, i)= ', '(Start_', Property{j},...
            '_Feature(:, i)- Mu_z(j,i) )/(Sigma_z(j,i)+ 1e-10);']);
    end
end

for j= 1:M
    for i= 1: N_t
        eval(['Start_', Property{j}, '_Feature_test(i,:)= ', 'Feature_Extract(Start_Testing_', num2str(i),...
            '.', Property{j}, ');']);
    end
    % standardize (use normal training data mean and standard deviation)
    for i= 1: K
        eval(['Start_', Property{j}, '_Feature_test(:, i)= ', '(Start_', Property{j},...
            '_Feature_test(:, i)- Mu_z(j,i) )/(Sigma_z(j,i)+ 1e-10);']);
    end
end
%% generate fault data (Feature)
rng(1)
gen_no= 20; percent= 3/100; cnt= length(ID_training);
for i= 1:length(Fault_type)
    eval(['F',num2str(i),'_idx_gen= [];']);
end
for i= 1:length(Fault_type)
    % generate from training fault data
    eval(['F_idx_train= F',num2str(i),'_idx_train;'])
    if ~isempty(F_idx_train)
        for j= 1: length(F_idx_train)
            for k= 1:M
                eval(['Feature_gen= repmat(Start_', Property{k},...
                    '_Feature(F_idx_train(j), :), gen_no,1)'...
                    '.*(1+ percent*randn(gen_no, K));']);
                eval(['Start_', Property{k},'_Feature(cnt+1:cnt+gen_no,:)= Feature_gen;']);
            end
            eval(['F',num2str(i),'_idx_gen =[F',num2str(i),'_idx_gen, cnt+1:cnt+gen_no];'])
            ID_training(cnt+1:cnt+gen_no,:)= repmat(ID_training(F_idx_train(j),:), gen_no, 1);
            cnt= cnt+ gen_no;
        end
    end
    
    % generate from testing fault data
    eval(['F_idx_test= F',num2str(i),'_idx_test;'])
    if ~isempty(F_idx_test)
        for j= 1: length(F_idx_test)
            for k= 1:M
                eval(['Feature_gen= repmat(Start_', Property{k},...
                    '_Feature_test(F_idx_test(j), :), gen_no,1)'...
                    '.*(1+ percent*randn(gen_no, K));']);
                eval(['Start_', Property{k},'_Feature(cnt+1:cnt+gen_no,:)= Feature_gen;']);
            end
            eval(['F',num2str(i),'_idx_gen =[F',num2str(i),'_idx_gen, cnt+1:cnt+gen_no];'])
            ID_training(cnt+1:cnt+gen_no,:)= repmat(ID_testing(F_idx_test(j),:), gen_no, 1);
            cnt= cnt+ gen_no;
        end
    end
end
N_gen= cnt- N; % no. of artificial data
if train_num(2) > 49
    disp_num= [train_num(1); train_num(2)/2; train_num(2)/2; N_gen/4*ones(4,1)];
else
    disp_num= [train_num; N_gen/4*ones(4,1)];
end

%% Training process
%% Choose no. of PC to build GMM model & calculate the probability of abnormality
% train the Gaussian model by normal data
Thres= 0.9;
PC= 4; min_PC= 1;

Pi= zeros(M,N_G); Mu= zeros(N_G, PC, M); PC_coord= zeros(K, PC, M);
W= zeros(M, N_G);
for j= 1:N_G
    eval("Var_"+ j+ "= zeros(PC,PC, M);");
end

for i= 1: M
    eval(['Feature_normal= Start_', Property{i}, '_Feature(Normal_data_index, :);']);
    [U, S, V]= svd(Feature_normal);
    figure
    plot(1:K, cumsum(diag(S))/sum(diag(S)),'-o', 'LineWidth', 2)
    hold on; grid on
    plot(1:K, Thres*ones(K,1), 'r--', 'LineWidth', 2);
    singular_value= diag(S);
    title(replace(Property{i}, '_', ' '))
    xlim([1, K])
    set(gca, 'FontSize', 20); xlabel('No. of Singular Value'); ylabel('Cumulative Sum')
    min_PC= max(min_PC, find(cumsum(singular_value)/sum(singular_value)>= Thres, 1));
    PC_coord(:, :, i)= V(:, 1:PC);
    SPE_normal= sum((Feature_normal*(eye(K)- V(:, 1:PC)*V(:, 1:PC)')).^2,2);
    Feature_reduce= Feature_normal*V(:,1:PC);

    [W(i,:), Mu(:,:,i), Var_est]= GMM_modeling(Feature_reduce, N_G);
    for j= 1:N_G
        eval("Var_"+ j+ "(:,:,i)= Var_est(:,:,"+ j+");");
        eval("Pi(i,j)= sqrt((2*pi)^(PC) * det(Var_est(:, :, "+ j +")));")
    end
end
%% test the result of training data
clc
Prob_training= zeros(N+N_gen, M); SPE_tr= zeros(N+N_gen,M);
for j= 1:M
    eval(['Feature= Start_', Property{j}, '_Feature;']);
    % [U, S, V]= svd(Feature);
    SPE_tr(:,j)= sum((Feature*(eye(K)- PC_coord(:, :, j)*PC_coord(:, :, j)')).^2,2);
    Feature_reduce= Feature*PC_coord(:, :, j);
    for i= 1:N+ N_gen
        prob_training= zeros(1, N_G);
        for k= 1:N_G
            eval("G= mvnpdf(Feature_reduce(i,:), Mu("+ k +", :, j), Var_"+ k +"(:, :, j));");
            prob_training(k)= Pi(j,k)*G;
        end
        Prob_training(i,j)= max(prob_training);
    end
end

cnt= 0;
for i= 1:size(disp_num, 1)
    Prob= Prob_training(cnt+1: cnt+ disp_num(i),:);
    figure
    H= heatmap(-log10(Prob));
    H.XData = categorical(Property_name); H.YDisplayLabels= ID_training(cnt+1: cnt+ disp_num(i),:);
    H.Colormap= [linspace(0, 1, 100)', ones(100, 1), zeros(100,1); ones(100,1), linspace(1, 0, 100)', zeros(100,1)];
    caxis([0, -log10(exp(-0.5*16))]) % -log10(exp(-0.5*9))
    cnt= cnt+ disp_num(i);
end
%% Multiclass logistic regression
% mark the label
Type= ["Normal"; "油門角超出標準3%"; "啟動後沒多久熄火"; "致動器異常"; "油壓低於標準值"; "滑油滲漏量稍多";...
    "箱內啟動器故障"];
class= size(Type, 1); % No. of engine contition (type)
Label= zeros(N+ N_gen,class); Label(Normal_data_index, 1)= 1;
for i= 1:length(Fault_type)
    eval(['F_idx= [F',num2str(i),'_idx_train, F',num2str(i),'_idx_gen];']);
    Label(F_idx, i+1)= 1;
end

% Softmax regression

% Standardized probability
learn_rate= 1e-3; iter_num= 20000; 
min_Pr= min(Prob_training); max_Pr= max(Prob_training);
Prob_tr_std= Prob_training;
% Standardized SPE
min_SPE= min(SPE_tr); max_SPE= max(SPE_tr);
SPE_tr_std= (SPE_tr- min(SPE_tr))./(max_SPE- min_SPE);

% training parameter
lambda= 1e-3;
Theta= SoftmaxRegression_Training([Prob_tr_std, SPE_tr_std], Label, learn_rate, lambda, iter_num);
% calculate the prob of each type
x= [ones(size(Prob_tr_std,1), 1), Prob_tr_std, SPE_tr_std];
Softmax = bsxfun(@times, exp(x*Theta), 1./sum(exp(x*Theta),2));
Softmax_1= Softmax(1:length(ID_1),:); Softmax_2= Softmax(length(ID_1)+1:end,:);
[Prob, Ind] = max(Softmax,[],2);
% test the result of all 
cnt= 0;
for i= 1:size(disp_num, 1)
    Softmax_ID= Softmax(cnt+1:cnt+disp_num(i),:);
    figure;
    for j= 1:class
        subplot(1,class,j)
        if j== 1
            H= heatmap(round(Softmax_ID(:,j)*100,2), 'FontSize', 10);
            H.XData = categorical(Type(1)); H.YDisplayLabels= ID_training(cnt+1: cnt+ disp_num(i),:);
            H.Colormap= flip([linspace(0, 1, 100)', ones(100, 1), zeros(100,1); ones(100,1),...
                linspace(1, 0, 100)', zeros(100,1)]);
            colorbar off;
        else
            H= heatmap(round(Softmax_ID(:,j)*100,2), 'FontSize', 10);
            H.XData = categorical(Type(j)); H.YDisplayLabels= ID_training(cnt+1: cnt+ disp_num(i),:);
            H.Colormap= [linspace(0, 1, 100)', ones(100, 1), zeros(100,1); ones(100,1),...
                linspace(1, 0, 100)', zeros(100,1)];
            colorbar off;
        end
        caxis([0, 100])     
    end
    cnt= cnt+ disp_num(i);
end

Status_train= ones(length(ID_training),1);
for i= 1:length(Status_train)
    Status_train(i)= find(Label(i,:)== 1);
end
Acc_train= sum(Ind == Status_train)/length(Status_train)*100;
disp("Accuracy of training data: "+ round(Acc_train,2)+ "%")
figure
P= confusionchart(Num2Class(Status_train, Type), Num2Class(Ind, Type));
set(P, 'FontSize',20)
%% Testing process
%% test the result of testing data
Prob_test= zeros(N_t, M); SPE_t= zeros(N_t, M);
for j= 1:M
    eval(['Feature= Start_', Property{j}, '_Feature_test;']);
    % [U, S, V]= svd(Feature);
    SPE_t(:,j)= sum((Feature*(eye(K)- PC_coord(:, :, j)*PC_coord(:, :, j)')).^2,2);
    Feature_reduce= Feature*PC_coord(:, :, j);
    for i= 1:N_t
        prob_test= zeros(1, N_G);
        for k= 1:N_G
            eval("G= Gaussian(Feature_reduce(i,:), Mu("+ k +", :, j), Var_"+ k +"(:, :, j));");
            prob_test(k)= Pi(j,k)*G;
        end
        Prob_test(i,j)= max(prob_test);
    end
end

if length(ID_testing) > 49
    disp_num_test= [length(ID_testing)/2; length(ID_testing)/2];
else
    disp_num_test= length(ID_testing);
end
cnt= 0;
for i= 1:length(disp_num_test)
    Prob= Prob_test(cnt+1: cnt+ disp_num_test(i),:);
    figure
    H= heatmap(-log10(Prob));
    H.XData = categorical(Property_name); H.YDisplayLabels= ID_testing(cnt+1: cnt+ disp_num_test(i),:);
    H.Colormap= [linspace(0, 1, 100)', ones(100, 1), zeros(100,1); ones(100,1), linspace(1, 0, 100)', zeros(100,1)];
    caxis([0, -log10(exp(-0.5*16))]) % -log10(exp(-0.5*9))
    cnt= cnt+ disp_num_test(i);
end
%% Multiclass logistic regression on testing data

% calculate the prob of each type
Prob_t_std= Prob_test;
SPE_t_std= (SPE_t- min_SPE)./(max_SPE- min_SPE);
x= [ones(size(Prob_t_std,1), 1), Prob_t_std, SPE_t_std];
y= exp(x*Theta);
for i= 1:size(y,1)
    for j= 1:size(y,2)
        if y(i,j) == Inf
            y(i,j)= 1e+50;
        elseif y(i,j)== -Inf
            y(i,j)= -1e+50;
        end
    end
end
Softmax = bsxfun(@times, y, 1./sum(y,2));
[Prob_test_max, Ind_test] = max(Softmax,[],2);
cnt= 0;
for i= 1:length(disp_num_test)
    figure
    Softmax_ID= Softmax(cnt+1:cnt+disp_num_test(i),:);
    for j= 1:class
        subplot(1,class,j)
        if j== 1
            H= heatmap(round(Softmax_ID(:,j)*100,2), 'FontSize', 10);
            H.XData = categorical(Type(1)); H.YDisplayLabels= ID_testing(cnt+1:cnt+disp_num_test(i),:);
            H.Colormap= flip([linspace(0, 1, 100)', ones(100, 1), zeros(100,1); ones(100,1),...
                linspace(1, 0, 100)', zeros(100,1)]);
            colorbar off;
        else
            H= heatmap(round(Softmax_ID(:,j)*100,2), 'FontSize', 10);
            H.XData = categorical(Type(j)); H.YDisplayLabels= ID_testing(cnt+1:cnt+disp_num_test(i),:);
            H.Colormap= [linspace(0, 1, 100)', ones(100, 1), zeros(100,1); ones(100,1),...
                linspace(1, 0, 100)', zeros(100,1)];
            colorbar off;
        end
        caxis([0, 100])
    end
    cnt= cnt+ disp_num_test(i);
end


Status_test= ones(length(ID_testing),1);
for i= 1:length(Fault_type)
    eval(['F_idx_test= F',num2str(i),'_idx_test;']);
    Status_test(F_idx_test)= i+1;
end

Acc_test= sum(Ind_test == Status_test)/length(Status_test)*100;
disp("Accuracy of testing data: "+ round(Acc_test,2)+ "%")
figure
P= confusionchart(Num2Class(Status_test, Type), Num2Class(Ind_test, Type));
set(P,'FontSize',20)