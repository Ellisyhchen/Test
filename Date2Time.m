function TimeStamp= Date2Time(Date_table)
Date= table2array(Date_table); Date= string(Date);
Year= zeros(size(Date)); Month= zeros(size(Date));
Day= zeros(size(Date)); Hr= zeros(size(Date));
Min= zeros(size(Date)); Sec= zeros(size(Date));
mSec= zeros(size(Date));
for i= 1: size(Date_table,1)
    Str1= split(Date(i),'年'); Year(i)= str2num(Str1(1));
    Str2= split(Str1(2),'月'); Month(i)= str2num(Str2(1));
    Str3= split(Str2(2),'日'); Day(i)= str2num(Str3(1));
    Str4= split(Str3(2),'時'); Hr(i)= str2num(Str4(1));
    Str5= split(Str4(2),'分'); Min(i)= str2num(Str5(1));
    Str6= split(Str5(2),'秒'); Sec(i)= str2num(Str6(1));
    mSec(i)= str2num(erase(Str6(2),'毫'));
end
TimeStamp= 3600*Hr+ 60*Min+ Sec+ 0.001*mSec;
end