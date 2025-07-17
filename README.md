# Actividad 1: Laboratorio Práctico en Google Colab

## Estructura del Proyecto

```
├── 01_Fundamentos_NumPy_Pandas
│   ├── 01_Fundamentos_NumPy_Pandas.ipynb
│   └── images
│       └── 01_iris_petal_ratio.png
│
├── 02_Visualizacion_Datos
│   ├── 02_Visualizacion_Datos.ipynb
│   └── images
│       ├── 02_titanic_genero.png
│       ├── 02_titanic_edad_clase.png
│       └── 02_wine_plot_interactivo.html
│
├── 03_Machine_Learning_Basico
│   └── 03_Machine_Learning_Basico.ipynb
│
├── 04_Deep_Learning_Intro
│   └── 04_Deep_Learning_Intro.ipynb
│
└── README.md
│
└── requirements.txt
```

## Informe Ejecutivo

### 1. Fundamentos NumPy y Pandas
En el análisis se identificaron diferencias claras entre tipos de flores al comparar la proporción entre el largo y ancho de sus pétalos; Esta métrica permitió visualizar patrones que podrían ser útiles para clasificarlas automáticamente.

- Se trabajó con el dataset **Iris**.
- Se analizaron y manipularon datos utilizando arrays y DataFrames.
- Se creó una nueva métrica "petal_ratio" que permitió observar diferencias entre especies de flores.

### Análisis de resultados

i. **Resumen estadístico por especie**:
   - La especie **virginica** tiene el mayor promedio de ´petal_ratio´, seguido de **versicolor** y finalmente **setosa**.
   - Esto indica que el largo del pétalo de *virginica* es desproporcionadamente mayor respecto a su ancho en comparación con las otras especies.
   - Esta diferencia puede ser utilizada como una variable discriminante para clasificación automática.

ii. **Nueva columna petal_ratio**:
   - Se incorporó exitosamente como una nueva variable en el DataFrame.
   - La inspección inicial (.head()) muestra variabilidad en los valores, que será útil para análisis posterior.

iii. **Distribución visual**:
   - El histograma muestra que la mayoría de los valores de **petal_ratio** se concentran entre 2 y 6.
   - Hay picos notables alrededor de los valores 2.5 y 4, lo que probablemente corresponde a la especie *setosa*.
   - Las colas hacia la derecha muestran la presencia de valores altos asociados a *virginica*, confirmando su mayor razón pétalo.

### Evidencias

- [Notebook 1](./01_Fundamentos_NumPy_Pandas/01_Fundamentos_NumPy_Pandas.ipynb)
- <img src="./01_Fundamentos_NumPy_Pandas/images/01_iris_petal_ratio.png" alt="01_iris_petal_ratio" width="300"/>

### Conclusión

La métrica petal_ratio resulta ser un indicador relevante para distinguir entre especies de Iris. Su distribución evidencia diferencias significativas en la morfología floral, siendo útil como variable de entrada para modelos de clasificación. La visualización complementa este hallazgo, mostrando patrones claros entre grupos.

---


### 2. Visualización de Datos
Se identificaron patrones clave que ayudan a comprender mejor los datos:

- En el caso del Titanic, los gráficos revelaron que la edad y el género influyeron significativamente en la distribución de los pasajeros por clase. Los adultos predominaban en primera clase, mientras que en tercera clase había mayor presencia de hombres y jóvenes. Esto sugiere desigualdad en el acceso a recursos dentro del barco.

- Para los vinos, el análisis visual mostró que, al comparar niveles de alcohol y acidez, se formaban agrupaciones por tipo de vino. Estas diferencias químicas se pueden aprovechar para clasificarlos automáticamente.

Se utilizaron herramientas gráficas estáticas y dinámicas para facilitar el análisis, permitiendo identificar patrones que no serían visibles solo con tablas numéricas.

### Evidencias

- [Notebook 2](./02_Visualizacion_Datos/02_Visualizacion_Datos.ipynb)
- <img src="./02_Visualizacion_Datos/images/02_titanic_edad_clase.png" alt="02_titanic_edad_clase.png" width="300"/>
- <img src="./02_Visualizacion_Datos/images/02_titanic_genero.png" alt="02_titanic_genero.png" width="300"/>
- [Ver gráfico interactivo Wine](https://steven-sanchez-uees.github.io/UEES-IA-Semana1-Grupo2/02_Visualizacion_Datos/images/02_wine_plot_interactivo.html)

### Conclusión

La visualización no solo hizo más comprensible la información, sino que expuso relaciones ocultas entre variables que pueden apoyar decisiones y automatizar procesos como segmentación o clasificación.

---

### 3. Machine Learning Basico
En esta actividad aplicamos técnicas de Machine Learning supervisado para resolver un problema clásico: predecir si una persona sobrevivió o no al hundimiento del Titanic, basándonos en sus características personales como clase, sexo, edad, número de familiares, tarifa pagada, entre otros.

Para esto, utilizamos tres modelos de clasificación muy populares:
- Regresión Logística
- Árbol de Decisión
- Random Forest

Luego de entrenarlos, comparamos su desempeño mediante métricas como accuracy, precision, recall y F1-score, y visualizamos sus resultados a través de gráficos y matrices de confusión.

i. **Resultados Generales de Precisión**
<br>Al evaluar la precisión general de cada modelo, encontramos lo siguiente:
````
    Regresión Logística: 81.01%
    Random Forest: 79.89%
    Árbol de Decisión: 78.77%
````

ii. **Distribución de Probabilidades – Regresión Logística**
<br>Este gráfico nos ayuda a entender cómo el modelo de regresión logística calcula la probabilidad de supervivencia para cada pasajero. Como se observa, hay una clara separación en las predicciones: muchas personas tienen probabilidades muy bajas o muy altas, lo cual indica que el modelo está bastante seguro en la mayoría de sus decisiones.

La línea roja marca el umbral de decisión del modelo (0.5), que separa a los que fueron clasificados como sobrevivientes y no sobrevivientes.

iii. **Matrices de Confusión**
<br>Las siguientes gráficas muestran cuántas predicciones fueron correctas o incorrectas para cada modelo. Esto permite identificar qué tan bien aciertan en ambas clases (sobrevivientes y no sobrevivientes):
- **Regresión Logística**
  - Predijo correctamente a 90 personas que no sobrevivieron.
  - Predijo correctamente a 55 personas que sí sobrevivieron.
  - Se equivocó con 15 personas (falsos positivos) y con 19 (falsos negativos).
- **Árbol de Decisión**
    <br>Tuvo más errores que la regresión logística, especialmente clasificando a personas que no sobrevivieron.
- **Random Forest**
    <br>Su rendimiento fue intermedio entre los dos modelos anteriores, con un número de aciertos y errores muy balanceado.

En general, la regresión logística mostró el mejor equilibrio entre precisión y sensibilidad (F1-score), lo cual la convierte en una buena opción cuando se busca un balance entre ambos aspectos.

### Evidencias
- [Notebook 3](./03_Machine_Learning_Basico/03_Machine_Learning_Basico.ipynb)
- <img src="./03_Machine_Learning_Basico/images/confusion_matrix_decision_tree.png" alt="confusion_matrix_decision_tree.png" width="300"/>
- <img src="./03_Machine_Learning_Basico/images/proba_distribution.png" alt="proba_distribution.png" width="300"/>
- [Ver todas las imágenes](./03_Machine_Learning_Basico/images/)

### Conclusiones
- La Regresión Logística se posicionó como el mejor modelo en este caso, con el mayor puntaje de precisión y un buen balance general.
- Random Forest también mostró resultados sólidos, especialmente en términos de precisión.
- Aunque el Árbol de Decisión fue el más simple, tuvo una leve desventaja frente a los otros modelos.

-------


-------
> *Desarrollado por el Grupo 2 – Universidad Espíritu Santo (UEES)*
> 
> • Steven Sánchez [@steven-sanchez-uees](https://github.com/steven-sanchez-uees)<br>
> • Joel Espín [@joel-espin-uees](https://github.com/joel-espin-uees)<br>
> • Cristina Gramal [@cristina-gramal](https://github.com/cristina-gramal)<br>
> • Veronica Ochoa [@veritochoah](https://github.com/veritochoah)<br>
> • Darío Pérez [@dario-perez-v](https://github.com/dario-perez-v)<br>
> • Tomas Guijo [@tguijo](https://github.com/tguijo)<br>

