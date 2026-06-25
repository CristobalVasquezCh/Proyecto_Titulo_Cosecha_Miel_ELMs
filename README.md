# Predicción de Cosecha de Miel mediante Máquinas de Aprendizaje Extremo (ELM)

Este repositorio contiene el código fuente en MATLAB, los entornos de variables (`.mat`), las bases de datos climáticas y las firmas satelitales (NDVI) desarrollados para el Proyecto de Título de Ingeniería Civil Electrónica.

**Autor:** Cristóbal Vásquez Chávez  
**Universidad:** Universidad Católica del Maule, Chile  
**Año:** 2026

## Descripción del Proyecto
Esta investigación propone una arquitectura computacional basada en **Extreme Learning Machines (ELM)** para predecir cuantitativamente el volumen de cosecha de miel en el noroeste de Australia. 

Para enfrentar la escasez de datos empíricos (*Data Starvation*), el modelo integra:
* Datos meteorológicos históricos (Temperatura y Precipitación).
* Firmas multiespectrales (NDVI) extraídas desde Google Earth Engine (GEE) con software QGIS.
* Densificación de dominios desbalanceados mediante SMOGN.
* Pseudo-etiquetado y Aprendizaje Semisupervisado (SS-ELM).

## Estructura del Repositorio
Para garantizar la reproducibilidad y trazabilidad algorítmica de la tesis, el repositorio ha sido estructurado modularmente de la siguiente manera:

* **`/Base de Datos/`**: Contiene los archivos `.xlsx` con la matriz histórica ($N=49$) y la matriz expandida ($N=201$), tanto en sus versiones exclusivas de clima como enriquecidas con NDVI.
* **`/Códigos MATLAB/`**:
  * Scripts principales de ejecución (`.m`) para el entrenamiento, la optimización hiperparamétrica (*Grid Search*) y la validación cruzada (*5-Fold CV*).
  * **`/FuncionesELM/`**: Subrutinas matemáticas que ejecutan funciones de activación de la capa oculta.
  * **`/Workspaces/`**: Archivos `.mat` con las variables y métricas pre-calculadas, permitiendo la auditoría inmediata de los resultados y mapas de contorno sin necesidad de re-compilar.
* **`/Mapas NDVI (Abrir con QGIS)/`**: Raster espaciales descargados desde la colección MODIS que validan la composición de valor máximo mensual para las áreas de estudio.
* **`/Regression Learner/`**: Archivos `.mat` y resultados `.xlsx` exportados desde el entorno automatizado de MATLAB, utilizados para establecer el *Benchmarking* comparativo frente a las ELMs.
* **`Extraccion_NDVI_MODIS_GEE.js`**: Script original en JavaScript ejecutado en la plataforma de Google Earth Engine para la extracción satelital.

> **Nota sobre los Mapas Espaciales (NDVI)**: Debido a que los archivos raster de composición de valor máximo superan el límite de tamaño de GitHub (75MB+), la carpeta con los mapas georreferenciados para abrir en QGIS se encuentra alojada en el siguiente repositorio externo:
> [ Descargar Mapas NDVI desde Google Drive]([https://drive.google.com/drive/folders/1ZRshQ7XS2Qf6rbBTATEnGrv_guHjhxMZ?usp=sharing])
