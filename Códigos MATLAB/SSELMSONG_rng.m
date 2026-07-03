clc; clear; close all;

%% 1) LECTURA DE DATOS Y MUESTREO 
fprintf(' Cargando base de datos\n');
C_data = readcell('BaseDeDatosSS2.xlsx');
M = str2double(strrep(string(C_data), ',', '.'));

if any(isnan(M(1, :))), X_total = M(2:end, 1:45); Y_total = M(2:end, 46);
else,                   X_total = M(:, 1:45);     Y_total = M(:, 46); end

X_hist   = X_total(1:49, :); Y_hist   = Y_total(1:49);
X_ciegos = X_total(50:end, :);

nL = size(X_hist, 1); nU = size(X_ciegos, 1);

% orden de las muestras ciegas y la validación cruzada
rng(42); 
idx_rand_global = randperm(nU);

%% 2) HIPERPARÁMETROS DEL GRID 
K = 5;
cv_indices = crossvalind('Kfold', nL, K);

NO_vec        = 1:2:35;         
C_vec         = 2.^(-10:2:10);   
lambda_vec_SS = 10.^(-6:1:6);    
num_Xu_vec    = floor(linspace(5, nU, 8)); 

total_iter = length(NO_vec) * length(C_vec) * length(lambda_vec_SS) * length(num_Xu_vec);
fprintf('Grid 4D Total: %d combinaciones por Fold\n', total_iter);

%% 3) PRE-CÁLCULO MASIVO (CV, SMOGN Y LAPLACIANOS)
options.NN = 5; options.GraphWeights = 'binary';
options.GraphDistanceFunction = 'euclidean';
options.LaplacianNormalize = 1; options.LaplacianDegree = 5;

cv_data = cell(K, 1);

for j = 1:K
   
    rng(42 + j); 
    
    idxTest  = (cv_indices == j); idxTrain = ~idxTest;
    X_train = X_hist(idxTrain, :); Y_train = Y_hist(idxTrain);
    X_test  = X_hist(idxTest, :);  Y_test  = Y_hist(idxTest);

    x_min = min(X_train, [], 1); x_max = max(X_train, [], 1);
    x_rng = x_max - x_min; x_rng(x_rng == 0) = 1;

    X_train_norm = 2 .* (X_train - x_min) ./ x_rng - 1;
    X_test_norm  = 2 .* (X_test  - x_min) ./ x_rng - 1;
    
    X_unlab_norm = 2 .* (X_ciegos - x_min) ./ x_rng - 1;
    X_unlab_consistent = X_unlab_norm(idx_rand_global, :);

    [X_sint, Y_sint] = aplicar_smogn(X_train_norm, Y_train);
    X_train_aug = [X_train_norm; X_sint];
    Y_train_aug = [Y_train; Y_sint];

    cv_data{j}.X_l = X_train_aug;     cv_data{j}.Y_l = Y_train_aug;
    cv_data{j}.Y_test = Y_test;       cv_data{j}.num_test = size(X_test_norm, 1);
    cv_data{j}.X_test_norm = X_test_norm; 
    cv_data{j}.Y_dummy = zeros(cv_data{j}.num_test, 1); 
    
    cv_data{j}.Xu_subsets = cell(length(num_Xu_vec), 1);
    cv_data{j}.L_matrices = cell(length(num_Xu_vec), 1);
    for iX = 1:length(num_Xu_vec)
        cant_Xu = num_Xu_vec(iX);
        X_u_sub = X_unlab_consistent(1:cant_Xu, :);
        cv_data{j}.Xu_subsets{iX} = X_u_sub;
        cv_data{j}.L_matrices{iX} = laplacian(options, [X_train_aug; X_u_sub]);
    end
end

%% 4) BÚSQUEA (Registro de TRAIN y TEST)
fprintf('\n Grid (NO x C x Lambda x Xu) \n');
RMSE_4D       = inf(length(NO_vec), length(C_vec), length(lambda_vec_SS), length(num_Xu_vec));
R_4D          = nan(length(NO_vec), length(C_vec), length(lambda_vec_SS), length(num_Xu_vec));
RMSE_train_4D = inf(length(NO_vec), length(C_vec), length(lambda_vec_SS), length(num_Xu_vec));
R_train_4D    = nan(length(NO_vec), length(C_vec), length(lambda_vec_SS), length(num_Xu_vec));

paras.NoDisplay = 1; paras.Kernel = 'sigmoid';
iter = 0;

for iX = 1:length(num_Xu_vec)
    for iL = 1:length(lambda_vec_SS)
        paras.lambda = lambda_vec_SS(iL);
        for iNO = 1:length(NO_vec)
            paras.NumHiddenNeuron = NO_vec(iNO);
            for iC = 1:length(C_vec)
                paras.C = C_vec(iC);
                iter = iter + 1;
                
                pred_all_test = zeros(nL, 1); real_all_test = zeros(nL, 1);
                pred_all_train = []; real_all_train = [];
                idx_s = 1;
                
                for j = 1:K
                    X_u_eval = cv_data{j}.Xu_subsets{iX};
                    L_eval   = cv_data{j}.L_matrices{iX};
                    
                    % Entrenamiento
                    elmModel = SSELMP(cv_data{j}.X_l, cv_data{j}.Y_l, X_u_eval, L_eval, paras);
                    
                    % Predicción Testeo
                    [~, ~, pred_t] = SSELMP_predict(cv_data{j}.X_test_norm, cv_data{j}.Y_dummy, elmModel);
                    pred_test = max(0, min(90, pred_t(:)));
                    
                    idx_e = idx_s + cv_data{j}.num_test - 1;
                    pred_all_test(idx_s:idx_e) = pred_test;
                    real_all_test(idx_s:idx_e) = cv_data{j}.Y_test(:);
                    idx_s = idx_e + 1;
                    
                    % Predicción Entrenamiento (para gráficos)
                    [~, ~, pred_tr] = SSELMP_predict(cv_data{j}.X_l, cv_data{j}.Y_l, elmModel);
                    pred_train = max(0, min(90, pred_tr(:)));
                    pred_all_train = [pred_all_train; pred_train];
                    real_all_train = [real_all_train; cv_data{j}.Y_l(:)];
                end
                
                % Métricas Testeo
                err_test = real_all_test - pred_all_test;
                RMSE_4D(iNO, iC, iL, iX) = sqrt(mean(err_test.^2));
                if std(real_all_test) > 0 && std(pred_all_test) > 0
                    mat_corr = corrcoef(real_all_test, pred_all_test);
                    R_4D(iNO, iC, iL, iX) = mat_corr(1,2);
                end
                
                % Métricas Entrenamiento
                err_train = real_all_train - pred_all_train;
                RMSE_train_4D(iNO, iC, iL, iX) = sqrt(mean(err_train.^2));
                if std(real_all_train) > 0 && std(pred_all_train) > 0
                    mat_corr_tr = corrcoef(real_all_train, pred_all_train);
                    R_train_4D(iNO, iC, iL, iX) = mat_corr_tr(1,2);
                end
                
                if mod(iter, 1000) == 0, fprintf(' Progreso: %d/%d (%.1f%%)\n', iter, total_iter, 100*iter/total_iter); end
            end
        end
    end
end

% Extracción del Óptimo Global 
[min_rmse_global, idx_opt] = min(RMSE_4D(:));
[iNO_opt, iC_opt, iL_opt, iX_opt] = ind2sub(size(RMSE_4D), idx_opt);

mejorNO_SS     = NO_vec(iNO_opt);
mejorC_SS      = C_vec(iC_opt);
mejorLambda_SS = lambda_vec_SS(iL_opt);
mejorXu_SS     = num_Xu_vec(iX_opt);
mejorR_SS      = R_4D(iNO_opt, iC_opt, iL_opt, iX_opt);

fprintf('\n ÓPTIMOS \n');
fprintf('NO: %d | C: %.2e | Lambda: %.2e | Xu: %d muestras\n', mejorNO_SS, mejorC_SS, mejorLambda_SS, mejorXu_SS);
fprintf('RMSE Test: %.4f kg | R Test: %.4f\n', min_rmse_global, mejorR_SS);

%% 5) ENTRENAMIENTO
fprintf('\n Entrenando Modelo \n');

x_min_f = min(X_hist, [], 1); x_max_f = max(X_hist, [], 1);
x_rng_f = x_max_f - x_min_f; x_rng_f(x_rng_f == 0) = 1;

X_hist_norm   = 2 .* (X_hist   - x_min_f) ./ x_rng_f - 1;
X_ciegos_norm = 2 .* (X_ciegos - x_min_f) ./ x_rng_f - 1;

% Fija el SMOGN del entrenamiento
rng(42); 
[X_sint_f, Y_sint_f] = aplicar_smogn(X_hist_norm, Y_hist);

X_train_final = [X_hist_norm; X_sint_f];
Y_train_final = [Y_hist; Y_sint_f];

X_ciegos_mezclados = X_ciegos_norm(idx_rand_global, :);
X_u_optimo_final   = X_ciegos_mezclados(1:mejorXu_SS, :);

L_final = laplacian(options, [X_train_final; X_u_optimo_final]);

paras.NumHiddenNeuron = mejorNO_SS;
paras.C               = mejorC_SS;
paras.lambda          = mejorLambda_SS;

modelo_final_SS = SSELMP(X_train_final, Y_train_final, X_u_optimo_final, L_final, paras);

%% 6) PREDICCIÓN FINAL 
fprintf('\n=== Predicción de las 152 cosechas ciegas ===\n');
[~, ~, pred_unlabeled] = SSELMP_predict(X_ciegos_norm, zeros(nU, 1), modelo_final_SS);
pred_unlabeled = max(0, min(90, pred_unlabeled(:)));

%% 7) EXPORTACIÓN A EXCEL
C_final = C_data;
offset = any(isnan(M(1, :))) * 1 + 49; 
for idx = 1:nU
    C_final{offset + idx, 46} = pred_unlabeled(idx);
end
nombre_excel = sprintf('BaseDeDatosNueva%s.xlsx', char(datetime('now', 'Format', 'yyyyMMdd_HHmmss')));
writecell(C_final, nombre_excel);
fprintf(' Archivo exportado con éxito: %s\n', nombre_excel);


%% 8) VISUALIZACIONES GRÁFICAS 
fprintf('\n=== Generando visualizaciones ===\n');

% Extraer vectores 1D (fijando las otras variables en sus óptimos)
rmse_test_no = squeeze(RMSE_4D(:, iC_opt, iL_opt, iX_opt));
rmse_train_no = squeeze(RMSE_train_4D(:, iC_opt, iL_opt, iX_opt));
r_test_no = squeeze(R_4D(:, iC_opt, iL_opt, iX_opt));
r_train_no = squeeze(R_train_4D(:, iC_opt, iL_opt, iX_opt));

% FIGURA 1: Capacidad Estructural (NO vs Métricas) 
figure('Position', [100, 100, 1000, 450], 'Name', 'Análisis de Neuronas Ocultas (SS-ELM)');

% Subplot A: NO vs RMSE
subplot(1,2,1);
plot(NO_vec, rmse_test_no, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'Color', [0 0.4470 0.7410], 'DisplayName', 'Testeo'); hold on;
plot(NO_vec, rmse_train_no, '-s', 'LineWidth', 2, 'MarkerSize', 6, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Entrenamiento');
plot(mejorNO_SS, rmse_test_no(iNO_opt), 'p', 'MarkerSize', 15, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'Óptimo Test');
xlabel('Neuronas Ocultas (NO)', 'FontSize', 11, 'FontWeight', 'bold'); 
ylabel('RMSE (kg)', 'FontSize', 11, 'FontWeight', 'bold');
title('A) Curva de Aprendizaje - RMSE', 'FontSize', 12);
grid on; legend('Location', 'best', 'FontSize', 10);

% Subplot B: NO vs R
subplot(1,2,2);
plot(NO_vec, r_test_no, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'Color', [0 0.4470 0.7410], 'DisplayName', 'Testeo'); hold on;
plot(NO_vec, r_train_no, '-s', 'LineWidth', 2, 'MarkerSize', 6, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Entrenamiento');
plot(mejorNO_SS, r_test_no(iNO_opt), 'p', 'MarkerSize', 15, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'Óptimo Test');
xlabel('Neuronas Ocultas (NO)', 'FontSize', 11, 'FontWeight', 'bold'); 
ylabel('Coeficiente de Correlación (R)', 'FontSize', 11, 'FontWeight', 'bold');
title('B) Curva de Aprendizaje - Correlación', 'FontSize', 12);
grid on; legend('Location', 'best', 'FontSize', 10);

% FIGURA 2: Mapas de Contorno (Lambda vs Xu) 
figure('Position', [150, 150, 1200, 500], 'Name', 'Mapas de Contorno SS-ELM (Lambda vs Xu)');

% matriz 2D exacta fijando NO y C en sus óptimos globales
[X_grid, L_grid] = meshgrid(num_Xu_vec, log10(lambda_vec_SS));
mat_rmse_2D = squeeze(RMSE_4D(iNO_opt, iC_opt, :, :));
mat_r_2D    = squeeze(R_4D(iNO_opt, iC_opt, :, :));

% Subplot A: RMSE
subplot(1,2,1);
contourf(X_grid, L_grid, mat_rmse_2D, 20, 'LineColor', 'none'); hold on;
%  
plot(mejorXu_SS, log10(mejorLambda_SS), 'p', 'MarkerSize', 18, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
hcb1 = colorbar; ylabel(hcb1, 'RMSE Error (Kg/Año)', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('Cantidad de Muestras de Testeo usadas como Entrenamiento (Xu)', 'FontSize', 11); 
ylabel('Regularización log_{10}(\lambda)', 'FontSize', 11);
title(sprintf('A) RMSE SS-ELM\nÓptimo: %.4f', min_rmse_global), 'FontSize', 12); 
grid on; axis tight;

% Subplot B: Correlación (R)
subplot(1,2,2);
contourf(X_grid, L_grid, mat_r_2D, 20, 'LineColor', 'none'); hold on;
% 
plot(mejorXu_SS, log10(mejorLambda_SS), 'p', 'MarkerSize', 18, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
hcb2 = colorbar; 
xlabel('Cantidad de Muestras de Testeo usadas como Entrenamiento (Xu)', 'FontSize', 11); 
ylabel('Regularización log_{10}(\lambda)', 'FontSize', 11);
title(sprintf('B) Correlación SS-ELM\nValor R en el Óptimo RMSE: %.4f', mejorR_SS), 'FontSize', 12); 
grid on; axis tight;

fprintf('--- Visualizaciones completadas exitosamente ---\n');
%% FUNCIONES AUXILIARES %%
function [X_sint, Y_sint] = aplicar_smogn(X_train, Y_train)
    umb_inf = prctile(Y_train, 20);
    umb_sup = prctile(Y_train, 80);
    grupos_raros = {find(Y_train <= umb_inf), find(Y_train >= umb_sup)};
    
    max_sint = size(X_train, 1) * 4; 
    X_sint = zeros(max_sint, size(X_train, 2));
    Y_sint = zeros(max_sint, 1);
    cnt = 0;
    
    for g = 1:2
        idx_rare = grupos_raros{g};
        n_rare = length(idx_rare);

        if n_rare > 1
            k_nn = min(5, n_rare - 1);
            X_r = X_train(idx_rare, :); Y_r = Y_train(idx_rare);
            D = pdist2(X_r, X_r);

            for ii = 1:n_rare
                dist_i = D(ii, :);
                dist_i_excl = dist_i([1:ii-1, ii+1:end]);
                maxD = median(dist_i_excl) / 2;

                [~, sorted_idx] = sort(dist_i);
                vecinos_idx = sorted_idx(2:k_nn+1);

                for s = 1:2 
                    id_local = vecinos_idx(randi(k_nn));
                    dist_vecino = D(ii, id_local);
                    case_X = X_r(ii, :); case_Y = Y_r(ii);
                    x_X = X_r(id_local, :); x_Y = Y_r(id_local);

                    if dist_vecino < maxD
                        w = rand();
                        new_X = case_X + w * (x_X - case_X);
                        d1 = norm(new_X - case_X); d2 = norm(new_X - x_X);
                        if (d1 + d2) == 0, new_Y = case_Y;
                        else, new_Y = (d2 * case_Y + d1 * x_Y) / (d1 + d2); end
                    else
                        pert = min(maxD, 0.02);
                        new_X = case_X + pert * randn(1, size(X_r, 2));
                        new_Y = case_Y + pert * randn() * std(Y_train);
                    end

                    new_X = max(-1, min(1, new_X));
                    new_Y = max(0, min(90, new_Y));

                    cnt = cnt + 1;
                    X_sint(cnt, :) = new_X;
                    Y_sint(cnt) = new_Y;
                end
            end
        end
    end
    X_sint = X_sint(1:cnt, :);
    Y_sint = Y_sint(1:cnt);
end