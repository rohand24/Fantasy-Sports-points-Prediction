clc;
close all ;
clear all;
cd C:\Users\dcc\Desktop\Matlab_Projects\Machine_Intelligence\MI_Project
addpath C:\Users\dcc\Desktop\Matlab_Projects\Machine_Intelligence\Rohan_Matlab_1\HW_6\libsvm-3.18\windows


%Load Game Data
filename = 'C:\Users\dcc\Desktop\Matlab_Projects\Machine_Intelligence\MI_Project\Game123.csv';
delimiter = ',';
startRow = 2;
formatSpec = '%f%f%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false);
fclose(fileID);
Game = [dataArray{1:end-1}];
clearvars filename delimiter startRow formatSpec fileID dataArray ans;
%Game Data loaded.

%Load Player data (Feature selected data)
filename = 'C:\Users\dcc\Desktop\Matlab_Projects\Machine_Intelligence\MI_Project\AR_1300features.csv';
delimiter = ',';
startRow = 2;
formatSpec = '%*s%f%*s%f%f%f%f%f%f%f%f%f%f%*s%*s%f%*s%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false);
fclose(fileID);
AR1300features = [dataArray{1:end-1}];
clearvars filename delimiter startRow formatSpec fileID dataArray ans;
% Done loading Player data


%%Initialise varaibles
gid = Game(:,1);
week = Game(:,3);
Xdata = AR1300features(:,2:10);
year = AR1300features(:,12);
y = AR1300features(:,11);

%Get train and test index values
TestIndex = find(year(:,1)==2015);
index1 = find(year(:,1)==2010);
index2 = find(year(:,1)==2011);
index3 = find(year(:,1)==2012);
index4 = find(year(:,1)==2013);
index5 = find(year(:,1)==2014);
TrainIndex = [index1(2:end); index2; index3; index4];
TrainGTindex = [index2; index3; index4; index5];


TrainXdata = Xdata(TrainIndex,:);
TrainGT =y(TrainGTindex);
TestXdata = Xdata(index5,:);
TestGT = y(TestIndex);

    
% Cost value
C = [1 3];
i = 0;
for i=1:length(C)


    %SVM model training using LibSVM
    [TrainXdataNorm, mu, sigma] = featureNormalize(TrainXdata,0,0);
    eval(['model = svmtrain(TrainGT ,TrainXdataNorm,''-s 3 -t 2 -c ' num2str(C(i)) '-g  -p 0.25'');']); %'-n 3
    predictionsTrain = svmpredict(TrainGT, TrainXdataNorm, model);

    
    %SVM model Testing
    [TestXdataNorm, mu, sigma] = featureNormalize(TestXdata,mu, sigma);
    predictionsTest = svmpredict(TestGT, TestXdataNorm, model);

    
    %Plots
    
    figure(i);
    plot(1:18,predictionsTest,'go',1:18,TestGT,'r+','MarkerSize',10);
    hold on
    MAE = abs(predictionsTest - TestGT);    %Mean Absolute Error
    MSE = sum((predictionsTest - TestGT).^2)/length(TestGT);    
    RMSE = sqrt(MSE);                       %Root Mean Square Error
    AvgMAE = sum(MAE)/length(MAE);
    c = num2str(C(i));
    r = num2str(RMSE)
    m = num2str(AvgMAE)
    title(['C = ',c   ' RMSE = ',r ' MAE = ',m ]);
    legend('Predicted Output','Actual Output','Location','northeast')
    set(gca, 'Position', [0.25 0.25 0.5 0.5]);

end
