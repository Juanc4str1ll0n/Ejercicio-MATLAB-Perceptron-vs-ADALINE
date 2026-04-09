%  GRUPO 1 — CLASIFICACIÓN DE FRUTAS (Agroindustria)
%  Manzanas vs Peras por Peso (g) y Color (valor verde 0-255)
%  Algoritmos: Perceptrón (Hardlim) y ADALINE (Regla Delta/LMS)

clear; clc; close all;

fprintf('  CLASIFICACIÓN DE FRUTAS — Agroindustria\n');
fprintf('  Perceptrón vs ADALINE\n');

% DEFINICIÓN DE DATOS DE ENTRENAMIENTO

P = [180   195  220  240  210  185  230  200  ...  
     120   130  145  160  135  125  155  140; ... 
     120   100   90  110   80  130   95  115  ...  
     210   230  200  220  240  215  225  205];     

% Etiquetas: 1 = Manzana, 0 = Pera
T = [1  1  1  1  1  1  1  1   0  0  0  0  0  0  0  0];

[n_features, N] = size(P); 

fprintf('Datos cargados: %d muestras, %d características\n', N, n_features);
fprintf('Manzanas: %d | Peras: %d\n\n', sum(T==1), sum(T==0));


%  PASO 2: NORMALIZACIÓN (Min-Max al rango [0, 1])

P_min = min(P, [], 2);  
P_max = max(P, [], 2);  
P_norm = (P - P_min) ./ (P_max - P_min);

fprintf('Rango original:\n');
fprintf('  Peso:  [%.0f, %.0f] g\n', P_min(1), P_max(1));
fprintf('  Color: [%.0f, %.0f]\n\n', P_min(2), P_max(2));

%  PASO 3: PARÁMETROS DE ENTRENAMIENTO
alpha_perc  = 0.1;   
alpha_adal  = 0.01;  
max_epocas  = 200;   
umbral_MSE  = 0.005; 

% ENTRENAMIENTO DEL PERCEPTRÓN

fprintf('----------------------------------------------\n');
fprintf('  ENTRENANDO PERCEPTRÓN...\n');
fprintf('----------------------------------------------\n');

% Inicialización de pesos y bias
W_p = zeros(1, n_features);
b_p = 0;

errores_perc    = zeros(1, max_epocas);
mse_perc        = zeros(1, max_epocas);
epocas_conv_p   = max_epocas;

for epoca = 1:max_epocas
    error_total = 0;
    mse_ep = 0;

    % Recorrer todas las muestras (en orden aleatorio para mejor convergencia)
    idx = randperm(N);

    for k = 1:N
        i = idx(k);
        x = P_norm(:, i);  
        t = T(i);          
        net = W_p * x + b_p;

        %Función de activación ESCALÓN (Hardlim) 
        if net >= 0
            y = 1;
        else
            y = 0;
        end

        % - Error discreto 
        e = t - y; 

        %  Actualización de pesos (solo si hay error) 
        W_p = W_p + alpha_perc * e * x';
        b_p = b_p + alpha_perc * e;

        error_total = error_total + abs(e);
        mse_ep      = mse_ep + e^2;
    end

    errores_perc(epoca) = error_total;
    mse_perc(epoca)     = mse_ep / N;

    % Criterio de paro: sin errores en toda una época
    if error_total == 0
        epocas_conv_p = epoca;
        fprintf('  ✓ Convergió en época %d\n', epoca);
        break;
    end
end

% Calcular precisión final del Perceptrón
y_pred_p = zeros(1, N);
for i = 1:N
    net = W_p * P_norm(:,i) + b_p;
    y_pred_p(i) = net >= 0;
end
acc_p = sum(y_pred_p == T) / N * 100;

fprintf('  Pesos finales: W = [%.4f, %.4f]  b = %.4f\n', W_p(1), W_p(2), b_p);
fprintf('  Precisión: %.1f%%\n\n', acc_p);

% ENTRENAMIENTO DE ADALINE (Regla Delta / LMS)
fprintf('----------------------------------------------\n');
fprintf('  ENTRENANDO ADALINE...\n');
fprintf('----------------------------------------------\n');

% Inicialización
W_a = rand(1, n_features) * 0.01; 
b_a = 0;

mse_adal      = zeros(1, max_epocas);
epocas_conv_a = max_epocas;

for epoca = 1:max_epocas
    errores_cuad = zeros(1, N);
    idx = randperm(N);

    for k = 1:N
        i = idx(k);
        x = P_norm(:, i);
        t = T(i);

        % Salida LINEAL (Purelin — SIN escalón) 
        net = W_a * x + b_a;   

        % Error CONTINUO (antes de cuantizar) 
        e = t - net; 

        % Regla Delta (gradiente del MSE)
        W_a = W_a + 2 * alpha_adal * e * x';
        b_a = b_a + 2 * alpha_adal * e;

        errores_cuad(k) = e^2;
    end

    mse_adal(epoca) = mean(errores_cuad);

    % Criterio de paro por MSE
    if mse_adal(epoca) < umbral_MSE
        epocas_conv_a = epoca;
        fprintf('  ✓ Convergió en época %d  (MSE = %.6f)\n', epoca, mse_adal(epoca));
        break;
    end
end

% Calcular precisión final de ADALINE (se aplica escalón DESPUÉS del entrenamiento)
y_pred_a = zeros(1, N);
for i = 1:N
    net = W_a * P_norm(:,i) + b_a;
    y_pred_a(i) = net >= 0.5;  
end
acc_a = sum(y_pred_a == T) / N * 100;

fprintf('  Pesos finales: W = [%.4f, %.4f]  b = %.4f\n', W_a(1), W_a(2), b_a);
fprintf('  MSE final: %.6f\n', mse_adal(epocas_conv_a));
fprintf('  Precisión: %.1f%%\n\n', acc_a);

%  PASO 4: VISUALIZACIÓN — 4 FIGURAS

% Índices por clase
idx_manzana = find(T == 1);  
idx_pera    = find(T == 0);  

% Colores
color_manzana = [0.85 0.20 0.15]; 
color_pera    = [0.20 0.65 0.25];  
color_perc    = [0.20 0.40 0.80];  
color_adal    = [0.85 0.50 0.10];  
x1_rango = linspace(-0.1, 1.1, 300);  

% Calcular fronteras de decisión
% Perceptrón: W_p(1)*x1 + W_p(2)*x2 + b_p = 0  →  x2 = -(W_p(1)*x1 + b_p)/W_p(2)
if abs(W_p(2)) > 1e-6
    x2_frontera_p = -(W_p(1) * x1_rango + b_p) / W_p(2);
else
    x2_frontera_p = NaN(size(x1_rango));
end

% ADALINE: frontera en net = 0.5 (umbral entre 0 y 1)
if abs(W_a(2)) > 1e-6
    x2_frontera_a = (0.5 - W_a(1) * x1_rango - b_a) / W_a(2);
else
    x2_frontera_a = NaN(size(x1_rango));
end

%% figura 1: Datos en espacio original 
figure('Name','Figura 1 - Datos Originales','NumberTitle','off',...
       'Position',[50 500 600 450],'Color','w');

scatter(P(1, idx_manzana), P(2, idx_manzana), 120, ...
        color_manzana, 'filled', 'Marker','o', 'DisplayName','Manzana');
hold on;
scatter(P(1, idx_pera), P(2, idx_pera), 120, ...
        color_pera, 'filled', 'Marker','^', 'DisplayName','Pera');

for i = 1:N
    if T(i) == 1
        txt = sprintf('M%d', i);
    else
        txt = sprintf('P%d', i-8);
    end
    text(P(1,i)+2, P(2,i)+3, txt, 'FontSize', 8, 'Color', [0.3 0.3 0.3]);
end

grid on; box on;
xlabel('Peso (g)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Valor de Verde (0-255)', 'FontSize', 13, 'FontWeight', 'bold');
title('Datos Originales — Manzanas vs Peras', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
set(gca, 'FontSize', 11);

annotation('textbox', [0.13 0.01 0.8 0.06], 'String', ...
    '■ Manzanas: mayor peso, menor valor de verde   ▲ Peras: menor peso, mayor valor de verde', ...
    'EdgeColor', 'none', 'FontSize', 9, 'Color', [0.4 0.4 0.4], 'HorizontalAlignment','center');

%% FIGURA 2: Fronteras de decisión 
figure('Name','Figura 2 - Fronteras de Decisión','NumberTitle','off',...
       'Position',[670 500 650 450],'Color','w');


[xx, yy] = meshgrid(linspace(0,1,200), linspace(0,1,200));
zz_p = zeros(size(xx));
for r = 1:size(xx,1)
    for c = 1:size(xx,2)
        net = W_p(1)*xx(r,c) + W_p(2)*yy(r,c) + b_p;
        zz_p(r,c) = net >= 0;
    end
end

h_bg = imagesc([0 1], [0 1], zz_p);
colormap([0.95 0.85 0.85; 0.85 0.95 0.85]); 
set(gca, 'YDir', 'normal');
alpha(h_bg, 0.3);
hold on;


plot(x1_rango, x2_frontera_p, '-', 'Color', color_perc, ...
     'LineWidth', 2.5, 'DisplayName', sprintf('Perceptrón (α=%.2f)', alpha_perc));
plot(x1_rango, x2_frontera_a, '--', 'Color', color_adal, ...
     'LineWidth', 2.5, 'DisplayName', sprintf('ADALINE (α=%.3f)', alpha_adal));

% Datos normalizados
scatter(P_norm(1, idx_manzana), P_norm(2, idx_manzana), 130, ...
        color_manzana, 'filled', 'o', 'DisplayName','Manzana', ...
        'MarkerEdgeColor','w', 'LineWidth',1.5);
scatter(P_norm(1, idx_pera), P_norm(2, idx_pera), 130, ...
        color_pera, 'filled', '^', 'DisplayName','Pera', ...
        'MarkerEdgeColor','w', 'LineWidth',1.5);

axis([0 1 0 1]); grid on; box on;
xlabel('Peso normalizado', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Verde normalizado', 'FontSize', 13, 'FontWeight', 'bold');
title('Fronteras de Decisión — Perceptrón vs ADALINE', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 10);
set(gca, 'FontSize', 11);

% Anotaciones de región
text(0.05, 0.92, 'Región: Manzana', 'Color', color_manzana, 'FontSize', 10, 'FontWeight', 'bold');
text(0.65, 0.08, 'Región: Pera', 'Color', color_pera, 'FontSize', 10, 'FontWeight', 'bold');

%% figura 3: Error del Perceptrón y MSE de ADALINE por época 
figure('Name','Figura 3 - Convergencia','NumberTitle','off',...
       'Position',[50 30 650 420],'Color','w');

ep_p = 1:epocas_conv_p;
ep_a = 1:epocas_conv_a;

% Subplot 1: Perceptrón — Error total por época
subplot(2,1,1);
area(ep_p, errores_perc(ep_p), 'FaceColor', color_perc, 'FaceAlpha', 0.3, ...
     'EdgeColor', color_perc, 'LineWidth', 1.5);
hold on;
plot(ep_p, errores_perc(ep_p), 'o-', 'Color', color_perc, ...
     'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', color_perc);
xline(epocas_conv_p, '--k', sprintf('Convergencia (é=%d)', epocas_conv_p), ...
      'FontSize', 9, 'LabelHorizontalAlignment', 'left');
xlabel('Época', 'FontSize', 11);
ylabel('Errores totales |e|', 'FontSize', 11);
title('Perceptrón — Evolución del Error', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;
set(gca, 'FontSize', 10);

% Subplot 2: ADALINE — MSE por época
subplot(2,1,2);
semilogy(ep_a, mse_adal(ep_a), '-', 'Color', color_adal, 'LineWidth', 2.5);
hold on;
semilogy(ep_a, mse_adal(ep_a), '.', 'Color', color_adal, 'MarkerSize', 8);
yline(umbral_MSE, '--k', sprintf('Umbral MSE = %.3f', umbral_MSE), ...
      'FontSize', 9, 'LabelHorizontalAlignment', 'right');
xline(epocas_conv_a, '--', 'Color',[0.4 0.4 0.4], ...
      'Label', sprintf('Convergencia (é=%d)', epocas_conv_a), ...
      'FontSize', 9, 'LabelHorizontalAlignment', 'left');
xlabel('Época', 'FontSize', 11);
ylabel('MSE (escala logarítmica)', 'FontSize', 11);
title('ADALINE — Convergencia del MSE (Regla Delta)', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;
set(gca, 'FontSize', 10);

%% FIGURA 4: Tabla de resultados y métricas finales 
figure('Name','Figura 4 - Comparación Final','NumberTitle','off',...
       'Position',[720 30 620 420],'Color','w');

% Panel izquierdo: Matriz de confusión del Perceptrón
subplot(1,2,1);
cm_p = confusionmat(T, y_pred_p);
imagesc(cm_p);
colormap([0.95 0.95 0.95; 0.70 0.85 0.70]);
colorbar off;
axis square;
xticks([1 2]); xticklabels({'Pred: Pera','Pred: Manzana'});
yticks([1 2]); yticklabels({'Real: Pera','Real: Manzana'});
title(sprintf('Perceptrón\nPrecisión: %.1f%%', acc_p), 'FontSize', 12, 'FontWeight','bold');
set(gca, 'FontSize', 9);
for i = 1:2
    for j = 1:2
        text(j, i, num2str(cm_p(i,j)), 'HorizontalAlignment','center', ...
             'FontSize', 18, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.1]);
    end
end
xlabel('Predicción', 'FontSize', 10);
ylabel('Real', 'FontSize', 10);

% Panel derecho: Matriz de confusión de ADALINE
subplot(1,2,2);
cm_a = confusionmat(T, y_pred_a);
imagesc(cm_a);
colormap([0.95 0.95 0.95; 0.70 0.85 0.70]);
colorbar off;
axis square;
xticks([1 2]); xticklabels({'Pred: Pera','Pred: Manzana'});
yticks([1 2]); yticklabels({'Real: Pera','Real: Manzana'});
title(sprintf('ADALINE\nPrecisión: %.1f%%', acc_a), 'FontSize', 12, 'FontWeight','bold');
set(gca, 'FontSize', 9);
for i = 1:2
    for j = 1:2
        text(j, i, num2str(cm_a(i,j)), 'HorizontalAlignment','center', ...
             'FontSize', 18, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.1]);
    end
end
xlabel('Predicción', 'FontSize', 10);
ylabel('Real', 'FontSize', 10);

sgtitle('Matrices de Confusión — Comparación Final', 'FontSize', 13, 'FontWeight', 'bold');

% RESUMEN FINAL EN CONSOLA
fprintf('  RESUMEN COMPARATIVO FINAL\n');
fprintf('%-25s %-15s %-15s\n', 'Métrica', 'Perceptrón', 'ADALINE');
fprintf('%s\n', repmat('-',1,55));
fprintf('%-25s %-15d %-15d\n', 'Épocas hasta convergencia', epocas_conv_p, epocas_conv_a);
fprintf('%-25s %-15.4f %-15.6f\n', 'MSE final', mse_perc(epocas_conv_p), mse_adal(epocas_conv_a));
fprintf('%-25s %-14.1f%% %-14.1f%%\n', 'Precisión en entrenamiento', acc_p, acc_a);
fprintf('%-25s %-15.4f %-15.4f\n', 'Peso W1 (Peso)', W_p(1), W_a(1));
fprintf('%-25s %-15.4f %-15.4f\n', 'Peso W2 (Color)', W_p(2), W_a(2));
fprintf('%-25s %-15.4f %-15.4f\n', 'Bias b', b_p, b_a);
fprintf('%s\n\n', repmat('-',1,55));

% CLASIFICAR NUEVAS FRUTAS
fprintf('  PRUEBA CON NUEVAS MUESTRAS\n');

nuevas_frutas = [190; 105];  
nuevas_frutas2 = [140; 218];  

etiquetas = {'Pera', 'Manzana'};

for muestra = 1:2
    if muestra == 1
        xn = nuevas_frutas;
        fprintf('\nMuestra 1: Peso=190g, Verde=105\n');
    else
        xn = nuevas_frutas2;
        fprintf('\nMuestra 2: Peso=140g, Verde=218\n');
    end

    xn_norm = (xn - P_min) ./ (P_max - P_min);

    % Clasificar con Perceptrón
    net_p = W_p * xn_norm + b_p;
    clase_p = net_p >= 0;
    fprintf('  Perceptrón → %s  (net=%.4f)\n', etiquetas{clase_p+1}, net_p);

    % Clasificar con ADALINE
    net_a = W_a * xn_norm + b_a;
    clase_a = net_a >= 0.5;
    fprintf('  ADALINE    → %s  (net=%.4f)\n', etiquetas{clase_a+1}, net_a);
end

fprintf('\n¡Entrenamiento y evaluación completados!\n');
fprintf('Revisar las 4 figuras generadas.\n');