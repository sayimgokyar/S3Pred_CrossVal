clear; close all; clc;
%%
foldlim = 0;
for ff=1:5      %Fold number
    %% Single Channel Plot order: B1P, B1M, GRE
    f=figure('Position', [100 200 480 720], 'Name', ['Fold:', num2str(ff),' Single Channel Networks'], 'NumberTitle','off');
    tiledlayout(3,2,'TileSpacing','tight'); %or 'compact'
    axlims = [-1.0 2.5 0 50]; xtics = [-1:.5:2]; ytics = [0:axlims(4)/2:axlims(4)]; txtlims1=[0.5,0.75*axlims(4)]; txtlims2=[0.1,2.5];


    load(['Fold',num2str(ff),'_B1P.mat']);
    GeneratePlot(xxx', yyy', axlims, xtics, ytics, txtlims1, txtlims2); clear xxx yyy;

    load(['Fold',num2str(ff),'_B1M.mat']);
    GeneratePlot(xxx', yyy', axlims, xtics, ytics, txtlims1, txtlims2); clear xxx yyy;

    load(['Fold',num2str(ff),'_GRE.mat']);
    GeneratePlot(xxx', yyy', axlims, xtics, ytics, txtlims1, txtlims2); clear xxx yyy;
    
    %% Double Channel Plot order: B1PB1M, B1PGRE, B1MGRE
    f=figure('Position', [100 200 480 720], 'Name', ['Fold:', num2str(ff),' Double Channel Networks'], 'NumberTitle','off');
    tiledlayout(3,2,'TileSpacing','tight'); %or 'compact'
    axlims = [-1.0 2.5 0 50]; xtics = [-1:.5:2]; ytics = [0:axlims(4)/2:axlims(4)]; txtlims1=[0.5,0.75*axlims(4)]; txtlims2=[0.1,2.5];

    load(['Fold',num2str(ff),'_B1PB1M.mat']);
    GeneratePlot(xxx', yyy', axlims, xtics, ytics, txtlims1, txtlims2); clear xxx yyy;

    load(['Fold',num2str(ff),'_B1PGRE.mat']);
    GeneratePlot(xxx', yyy', axlims, xtics, ytics, txtlims1, txtlims2); clear xxx yyy;

    load(['Fold',num2str(ff),'_B1MGRE.mat']);
    GeneratePlot(xxx', yyy', axlims, xtics, ytics, txtlims1, txtlims2); clear xxx yyy;

    %% Three Channel Plot
    f=figure('Position', [100 200 480 480], 'Name', ['Fold:', num2str(ff),' Three Channel and VOP Comparison'], 'NumberTitle','off');
    tiledlayout(2,2,'TileSpacing','tight'); %or 'compact'
     axlims = [-1.0 2.5 0 50]; xtics = [-1:.5:2]; ytics = [0:axlims(4)/2:axlims(4)]; txtlims1=[0.5,0.75*axlims(4)]; txtlims2=[0.1,2.5];

    load(['Fold',num2str(ff),'_B1PB1MGRE.mat']); dx = length(xxx); foldlim = foldlim+dx;    %set the number of scans on the given fold!
    GeneratePlot(xxx', yyy', axlims, xtics, ytics, txtlims1, txtlims2); clear xxx yyy;

    %% VOP Plot
    axlims = [-1.0 8 0 50]; xtics = [-1,0,1:2:8]; ytics = [0:axlims(4)/2:axlims(4)]; txtlims1=[3.0,0.75*axlims(4)]; txtlims2=[1.1,0.5];

    load('VOP_all.mat');    yyy = VOP;
    GeneratePlot(xxx(foldlim-dx+1:foldlim), yyy(foldlim-dx+1:foldlim), axlims, xtics, ytics, txtlims1, txtlims2); clear xxx yyy;

end

function [] = GeneratePlot(xxx, yyy, axlims, xtics, ytics, txtlims1, txtlims2)
    nexttile;
    r = (yyy./xxx-1);   binsize = 0.2; nbins = round((max(r) - min(r))/binsize); binsize = (max(r) - min(r))/nbins;
    pd = fitdist(r, 'Normal'); disp(pd);    h3 = histfit(r, nbins);  h3(2).LineWidth = 4; grid on;
    txt = {['\mu = ', num2str(pd.mu, 3)], ['\sigma = ', num2str(pd.sigma, 3)]};
    text(txtlims1(1),txtlims1(2), txt, "FontSize",12);
    axis(axlims);  xticks(xtics); yticks(ytics);  %yticklabels({}); xticklabels({});  

    nexttile;
    lft = fittype({'x'});   [f3, gof3] = fit(double(xxx), double(yyy), lft);
    ax3 = plot(xxx, yyy, 'bo'); ax3.LineWidth=2; ax3.MarkerSize=2; hold on; grid on; axis([0 3 0 3]); xticks([0:1:3]); yticks([0:1:3]);
    ax3 = plot(f3); ax3.LineWidth = 3;
    xlabel(''); ylabel(''); legend off; %xticklabels({}); yticklabels({}); legend off;

    ci = confint(f3, 0.95);
    sigma3 = (ci(2)-ci(1))/2;
    txt = {['y = (',num2str(f3.a, 3), char(177), num2str(sigma3, 1),') x'], ['RMSE = ', num2str(100*gof3.rmse, 3), ' %']};
    text(txtlims2(1),txtlims2(2),txt, "FontSize",12);

end