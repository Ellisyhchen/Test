function theta= SoftmaxRegression_Training(feature, label, learn_rate, lambda, iter_num)
x= [ones(size(feature,1), 1), feature]; y= label;
rng(0)
theta= rand(size(x,2), size(y,2));
beta_1= 0.9; beta_2= 0.999; epsilon= 1e-8; thres= 1e-50;
% lambda:  lasso/ridge parameters
% ADAM
m= 0; v=0;
for i= 1:iter_num
    Softmax= exp(x*theta) ./ sum(exp(x*theta), 2);
    temp = Softmax- y;
    g = x'*temp+ lambda*theta;  %% Speed is faster  : M-by-K

    m_new= beta_1*m+ (1-beta_1)*g;
    v_new= beta_2*v+ (1-beta_2)*g.^2;
    m_hat= m_new/(1- beta_1^i);
    v_hat= v_new/(1- beta_2^i);
    theta_new= theta-  learn_rate* 1./(sqrt(v_hat)+ epsilon) .* m_hat;
    m= m_new; v= v_new;
    if sum(sum(sqrt((theta_new- theta).^2))) <= thres
        theta= theta_new;
        break;
    end
    theta=theta_new;
end
j=1;
end