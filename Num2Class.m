function class_vector= Num2Class(Ind_vector, class)
for i= 1:length(Ind_vector)
   class_vector(i,:)= class(Ind_vector(i),:); 
end
end