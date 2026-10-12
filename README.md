# TP 7 — Clasificador Bayesiano, Punto B: Detección de Grooming

### Materia: Procesamiento de Aprendizaje Automático

 **[Ver el informe del modelo (PDF)](informe_modelo_grooming.pdf)**

---

## Descripción

Una organización que gestiona una plataforma de mensajería quiere detectar automáticamente posibles casos de *grooming* (acoso a menores en línea) a partir de los mensajes de texto que envían los usuarios.

En este trabajo se entrena un clasificador **Naive Bayes multinomial** con un conjunto de 20 mensajes etiquetados como `grooming` o `no grooming`. Después se evalúa su rendimiento y se usa para clasificar mensajes nuevos.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `modelo_grooming.py` | Código del modelo: datos, entrenamiento, evaluación y clasificación de mensajes nuevos |
| `informe_modelo_grooming.pdf` | Informe con accuracy, reporte de clasificación, matriz de confusión y clasificación de nuevos mensajes |
| `matriz_confusion.png` | Gráfico de la matriz de confusión (lo genera el script) |
| `resultados.json` | Resultados del modelo en formato JSON (lo genera el script) |

## Cómo ejecutarlo

Requiere Python 3.

```bash
pip install scikit-learn pandas matplotlib
python modelo_grooming.py
```

El script muestra los resultados en la consola y genera `matriz_confusion.png` y `resultados.json`.

## Metodología

1. **Datos:** los 20 mensajes provistos por la cátedra (10 de cada clase), sin modificaciones.
2. **Preprocesamiento:** se pasa el texto a minúsculas y se convierte cada mensaje en un conteo de palabras (*bag of words*) con `CountVectorizer`. No se eliminan las *stopwords*, porque palabras como "no", "nadie" o "solo" son señales clave de grooming ("no le digas a nadie").
3. **División:** 70% para entrenamiento (14 mensajes) y 30% para prueba (6 mensajes), estratificada y con `random_state=2` para que el resultado sea reproducible.
4. **Modelo:** `MultinomialNB` de scikit-learn.
5. **Evaluación:** accuracy, reporte de clasificación y matriz de confusión.

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy (conjunto de prueba) | **100%** (6 de 6) |
| Accuracy promedio en 200 sorteos distintos | 88% |
| Validación cruzada (5 particiones) | 80% |

### Clasificación de nuevos mensajes

| Mensaje | Clasificación |
|---|---|
| "¿Podés mandarme una foto tuya? No se la muestres a nadie." | grooming |
| "¿Jugamos Minecraft esta tarde?" | no grooming |
| "No le digas a tus papás que hablamos, ¿sí?" | grooming |
| "¿Terminaste el trabajo práctico de historia?" | no grooming |

