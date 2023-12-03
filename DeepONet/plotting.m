% This script is to plot the predicted solution of the combustion problem
% obtained using POD DeepONet

close all;
clear 
clc

delta_t = 1000;
curDir = pwd;
folder_name = [curDir '/Predictions'];
if exist(folder_name,'dir')
    rmdir(folder_name,'s');
end
mkdir(folder_name);

coord_x = load('Case_4/cordinates');
% coord_y = load('Case_4/cordinates').y;
y_pred = load('Case_4/pred').u250_pred';
y_true = load('Case_4/pred').u_ref';
case_num = load('Case_4/pred').test_id';

xx = -4.3e-03:1e-5:-2.1e-03;
yy = -9.45e-04:1e-5:2.36e-04;
[xq,yq] = meshgrid(xx,yy);

scrsz = get(groot, 'ScreenSize');
hFig  = figure('Position',[1 scrsz(4)/6 4*scrsz(3)/5 1.2*scrsz(4)/4]);

for i = 1:length(case_num)
    
    ii = case_num(i)+1;
    initial = griddata(coord(:,1),coord(:,2),initial_cond(:,ii),xq,yq,'linear');    
    truth = griddata(coord(:,1),coord(:,2),y_true(:,i),xq,yq,'linear');
    pred = griddata(coord(:,1),coord(:,2),y_pred(:,i),xq,yq,'linear');
    error = abs(abs(truth) - abs(pred));%./abs(truth);
    
    subplot(1,4,1);
    pcolor(xq,yq,exp(initial))
    colormap(jet); colorbar(); shading interp;
    axis off
    title('Initial mass fraction (Y_1)')
    
    min_clim = min(min(truth));
    max_clim = max(max(truth));
    clim = [min_clim, max_clim];
    
    subplot(1,4,2);
    pcolor(xq,yq,truth)
    colormap(jet); colorbar(); shading interp; caxis(clim);
    axis off
    title('Final mass fraction: True')
    
    subplot(1,4,3);
    pcolor(xq,yq,pred)
    colormap(jet); colorbar(); shading interp; caxis(clim);
    axis off
    title('Final mass fraction: Predicted');
    
    subplot(1,4,4);
    pcolor(xq,yq,error)
    colormap(jet); colorbar(); shading interp;
    axis off
    title('Error');
    sgtitle("T = T_{init} + " + delta_t + " \times \Delta t")
    
    saveas(hFig, [folder_name '/TestCase', num2str(i),'.png'])
    
end
