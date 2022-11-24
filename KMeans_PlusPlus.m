function [w_init, mu_init, var_init]= KMeans_PlusPlus(x, N_G)
% N_G: number of gaussian model for one GMM
rng(0)
iter= 10000; N= length(x); M= size(x,2); % N: number of training data, M: Dimension of x
mu_init= zeros(N_G, M); w_init= zeros(1, N_G);
var_init= zeros(M, M, N_G);
for i= 1:N_G
    var_init(:,:,i)= eye(M);
end

% Choose initial center from data points 
x_center= x; idx= randi(N);
mu_init(1,:)= x_center(idx, :); x_center(idx,:)= [];
D_square= sum((x_center- mu_init(1,:)).^2, 2); Prob= D_square/ sum(D_square);
wheel= cumsum(Prob);
for i= 2:N_G
    idx= find(rand<= wheel,1);
    mu_init(i,:)= x_center(idx, :); x_center(idx,:)= [];
    D_square= zeros(length(x_center), 1);
    for j= 1:i
        D_square= D_square+ sum((x_center- mu_init(j,:)).^2, 2);
    end
    Prob= D_square/ sum(D_square);
    wheel= cumsum(Prob);
end
% K means
for i= 1:iter
    mu_new= zeros(size(mu_init)); var_new= zeros(size(var_init));
    c_num= zeros(1, N_G);
    for j= 1:N_G
        for k= 1:N
            [~,I]=min(sum((x(k,:)- mu_init).^2,2));
            cond= ( c_num(j)==0 || sum( abs(sum(x(k,:)-X(1:c_num(j),:),2))<= 1e-10) == 0 );
            % cond:避免重複選取，同一點不可存在於不同cluster中
            if I==j && cond
                c_num(j)= c_num(j)+1;
                X(c_num(j),:)= x(k,:);
            end
        end
        if exist('X')==0
            mu_new(j,:)= mu_init(j,:);
            var_new(:,:,j)= var_init(:,:,j);
        else
            mu_new(j,:)= mean(X,1);
            var_new(:, :, j)= cov(X);
        end
        
        clear X
    end   

    if norm(mu_new- mu_init) <= 1e-6
        mu_init= mu_new; var_init= var_new;
        break;
    end
    mu_init= mu_new; var_init= var_new;
end
% Calculate initial weight
w_init= c_num/ N;
end