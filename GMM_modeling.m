function [w_est, mu_est, var_est]= GMM_modeling(x, N_G)
% N_G: number of gaussian model for one GMM
% rng('default')
iter= 1000; N= length(x); M= size(x,2); % N: number of training data, M: Dimension of x

[w_est, mu_est, var_est]= KMeans_PlusPlus(x, N_G);
% mu_est: N_GxM, var_est: MxMxN_G, w_est: 1xN_G

for i= 1:iter
    %% Calculating G 將(多維度)的數據點在不同(多維度)高斯函數中，所佔的機率值
    G= zeros(N,N_G);
    for k = 1:N_G
        var_est(:,:,k)= var_est(:,:,k)+ 1e-10*eye(M);
        G(:,k) = w_est(k)*mvnpdf(x, mu_est(k,:), var_est(:,:,k));
    end
    %% Calculating z (latent variable) : 計算每一個點，在不同高斯函數中，所佔的機率比例
    z= zeros(N,N_G);
    sum_row = sum(G,2);  %%... 單一數據點在所有Gaussian的機率總和
    for k = 1:N_G
        z(:,k) = G(:,k) ./ sum_row;
    end
    %% M-step. Updating MUs & SIGMAs
    sum_col = sum(z);  %%...  所有數據點在單一gaussian的機率總和
    w_est = 1/N * sum_col;
    mu_est = ( (z'*x)'./sum_col )';
    for k = 1:N_G
        temp1 = z(:,k).* (x - repmat(mu_est(k,:), N, 1));
        temp2 = temp1'*(x - repmat(mu_est(k,:), N, 1));
        var_est(:,:,k) = temp2/sum_col(k)+ 1e-10*eye(M);
    end
end
end
