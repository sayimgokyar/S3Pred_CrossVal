
GT=[];
PRED_B1P=[];
PRED_B1M=[];
PRED_GRE=[];
PRED_B1PB1M=[];
PRED_B1PGRE=[];
PRED_B1MGRE=[];
PRED_B1PB1MGRE=[];

for ii=1:5
    load(['Fold',num2str(ii),'_B1P.mat']);
    PRED_B1P=cat(1, PRED_B1P, yyy');    clear yyy;
    GT= cat(1, GT, xxx');   clear xxx;  %disp([num2str(ii),num2str(length(xxx))]);

    load(['Fold',num2str(ii),'_B1M.mat']);
    PRED_B1M=cat(1, PRED_B1M, yyy');    clear yyy;  %disp([num2str(ii),num2str(length(xxx))]);

    load(['Fold',num2str(ii),'_GRE.mat']);
    PRED_GRE=cat(1, PRED_GRE, yyy');    clear yyy;

    load(['Fold',num2str(ii),'_B1PB1M.mat']);
    PRED_B1PB1M=cat(1, PRED_B1PB1M, yyy');    clear yyy;

    load(['Fold',num2str(ii),'_B1PGRE.mat']);
    PRED_B1PGRE=cat(1, PRED_B1PGRE, yyy');    clear yyy;

    load(['Fold',num2str(ii),'_B1MGRE.mat']);
    PRED_B1MGRE=cat(1, PRED_B1MGRE, yyy');    clear yyy;

    load(['Fold',num2str(ii),'_B1PB1MGRE.mat']);
    PRED_B1PB1MGRE=cat(1, PRED_B1PB1MGRE, yyy');    clear yyy;

end


xxx = GT;

%% 3D Histogram and Fits for Single Channel Data
f=figure('Position', [100 200 480 720], 'Name', 'Single Channel Networks');
tiledlayout(3,2,'TileSpacing','tight'); %or 'compact'
for ii=1:3
    if ii==1
        yyy = PRED_B1P;
    elseif ii==2
        yyy = PRED_B1M;
    elseif ii==3
        yyy= PRED_GRE;
    else
        yyy=0; print('Error!');
    end
    nexttile;
    r = (yyy./xxx-1);   binsize = 0.2; nbins = round((max(r) - min(r))/binsize); binsize = (max(r) - min(r))/nbins;
    pd = fitdist(r, 'Normal'); disp(pd);    h3 = histfit(r, nbins);  h3(2).LineWidth = 4; grid on;
    txt = {['\mu = ', num2str(pd.mu, 3)], ['\sigma = ', num2str(pd.sigma, 3)]};
    text(-0.9,175,txt, "FontSize",12);
    axis([-1.0 2 0 200]);  xticks([-1:.5:2]); yticks([0:50:200]);  %yticklabels({}); xticklabels({});  


    nexttile;
    lft = fittype({'x'});   [f3, gof3] = fit(double(xxx), double(yyy), lft);
    ax3 = plot(xxx, yyy, 'bo'); ax3.LineWidth=2; ax3.MarkerSize=2; hold on; grid on; axis([0 3 0 3]); xticks([0:1:3]); yticks([0:1:3]);
    ax3 = plot(f3); ax3.LineWidth = 3;
    xlabel(''); ylabel(''); legend off; %xticklabels({}); yticklabels({}); legend off;

    ci = confint(f3, 0.95);
    sigma3 = (ci(2)-ci(1))/2;
    txt = {['y = (',num2str(f3.a, 3), char(177), num2str(sigma3, 1),') x'], ['RMSE = ', num2str(100*gof3.rmse, 3), ' %']};
    text(0.1,2.5,txt, "FontSize",12);

end

%% 3D Histogram and Fits for Double Channel Data
f=figure('Position', [100 200 480 720]);
tiledlayout(3,2,'TileSpacing','tight'); %or 'compact'
for ii=1:3
    if ii==1
        yyy = PRED_B1PB1M;
    elseif ii==2
        yyy = PRED_B1PGRE;
    elseif ii==3
        yyy= PRED_B1MGRE;
    else
        yyy=0; print('Error!');
    end
    nexttile;
    r = (yyy./xxx-1);   binsize = 0.2; nbins = round((max(r) - min(r))/binsize); binsize = (max(r) - min(r))/nbins;
    pd = fitdist(r, 'Normal'); disp(pd);    h3 = histfit(r, nbins);  h3(2).LineWidth = 4; grid on;
    txt = {['\mu = ', num2str(pd.mu, 3)], ['\sigma = ', num2str(pd.sigma, 3)]};
    text(-0.9,175,txt, "FontSize",12);
    axis([-1.0 2 0 200]);  xticks([-1:.5:2]); yticks([0:50:200]);  %yticklabels({}); xticklabels({});    


    nexttile;
    lft = fittype({'x'});   [f3, gof3] = fit(double(xxx), double(yyy), lft);
    ax3 = plot(xxx, yyy, 'bo'); ax3.LineWidth=2; ax3.MarkerSize=2; hold on; grid on; axis([0 3 0 3]); xticks([0:1:3]); yticks([0:1:3]);
    ax3 = plot(f3); ax3.LineWidth = 3;
    xlabel(''); ylabel(''); legend('off'); %xticklabels({}); yticklabels({});

    ci = confint(f3, 0.95);
    sigma3 = (ci(2)-ci(1))/2;
    txt = {['y = (',num2str(f3.a, 3), char(177), num2str(sigma3, 1),') x'], ['RMSE = ', num2str(100*gof3.rmse, 3), ' %']};
    text(0.1,2.5,txt, "FontSize",12);

end


%% 3D Histogram and Fits for Three Channel Data and VOP
f=figure('Position', [100 200 480 480]);
tiledlayout(2,2,'TileSpacing','tight'); %or 'compact'
for ii=1:2
    if ii==1
        yyy = PRED_B1PB1MGRE;
        axlims = [-1.0 2 0 200];
        xtics = [-1:.5:2];
        ytics = [0:50:200];
    elseif ii==2            %Implement VOP table code for this line
        load('VOP_all.mat');
        yyy = VOP;
        axlims = [-1.0 8 0 200];
        xtics = [-1:2:8];
        ytics = [0:50:200];
    else
        yyy=0; print('Error!');
    end
    nexttile;
    r = (yyy./xxx-1);   binsize = 0.2; nbins = round((max(r) - min(r))/binsize); binsize = (max(r) - min(r))/nbins;
    disp(num2str(min(r),3));
    pd = fitdist(r, 'Normal'); disp(pd);    h3 = histfit(r, nbins);  h3(2).LineWidth = 4; grid on;
    txt = {['\mu = ', num2str(pd.mu, 3)], ['\sigma = ', num2str(pd.sigma, 3)]};
    text(0.75,175,txt, "FontSize",12);
    axis(axlims);  xticks(xtics); yticks(ytics);  %yticklabels({}); xticklabels({});    


    nexttile;
    lft = fittype({'x'});   [f3, gof3] = fit(double(xxx), double(yyy), lft);
    ax3 = plot(xxx, yyy, 'bo'); ax3.LineWidth=2; ax3.MarkerSize=2; hold on; grid on; axis([0 4 0 4]); xticks([0,0.5,1:1:4]); yticks([0:1:4]);
    ax3 = plot(f3); ax3.LineWidth = 3;
    xlabel(''); ylabel(''); legend off;       %xticklabels({}); yticklabels({}); legend off;

    ci = confint(f3, 0.95);
    sigma3 = (ci(2)-ci(1))/2;
    txt = {['y = (',num2str(f3.a, 3), char(177), num2str(sigma3, 1),') x'], ['RMSE = ', num2str(100*gof3.rmse, 3), ' %']};
    text(1.1,0.5,txt, "FontSize",12);

end
