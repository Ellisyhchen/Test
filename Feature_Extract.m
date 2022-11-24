function Feature= Feature_Extract(Data_property)
epsilon= 1e-10;
Mu= mean(Data_property); Variance= var(Data_property, 1); Sigma= sqrt(Variance); 
% Mu: mean 平均值, Sigma: Standard deviation 均方差 
RootMS= rms(Data_property);
Max= max(abs(Data_property));
% Max: Maximum Peak 最大絕對值 峰值
Mu3= mean((Data_property- Mu).^3); Mu4= mean((Data_property- Mu).^4);
% Mu3: 三階主動差 third central moment, Mu4: 四階主動差 Fourth central moment
Crest_factor= Max/ (RootMS+ epsilon); % 峰值因子
Skewness= Mu3/ (Sigma^3+ epsilon); % 偏度 
Kurtosis= Mu4/ (Sigma^4+ epsilon); % 峰度
Feature= [Mu, Sigma, RootMS, Crest_factor, Skewness, Kurtosis];
end