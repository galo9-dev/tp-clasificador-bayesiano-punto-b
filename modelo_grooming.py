"""
Modelo de detección de grooming con Naive Bayes
------------------------------------------------
Entrena un clasificador Multinomial Naive Bayes sobre mensajes etiquetados
como "grooming" / "no grooming", lo evalúa y clasifica mensajes nuevos.

Requisitos:  pip install scikit-learn pandas matplotlib
Ejecución:   python modelo_grooming.py
"""

import json

import matplotlib
matplotlib.use("Agg")  # permite guardar gráficos sin ventana
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------
datos = [
    ("¿Cuántos años tenés? No le digas a nadie que hablamos.", "grooming"),
    ("Hola, ¿cómo estás? ¿Querés jugar algo online?", "no grooming"),
    ("Sos muy linda, no le muestres esto a tus papás.", "grooming"),
    ("¿Terminaste la tarea de matemáticas?", "no grooming"),
    ("Podés mandarme una foto tuya, pero que sea solo para mí.", "grooming"),
    ("¿Querés que te ayude con el juego que no podías pasar?", "no grooming"),
    ("No le contés a nadie que hablamos por acá, ¿sí?", "grooming"),
    ("¿Te gustó la película de ayer?", "no grooming"),
    ("¿Estás solo en casa ahora?", "grooming"),
    ("¿Querés hacer una videollamada para estudiar juntos?", "no grooming"),
    ("No le digas a tus padres que te escribí, es nuestro secreto.", "grooming"),
    ("¿Cómo te fue en el examen de historia?", "no grooming"),
    ("Mandame una foto tuya, pero que nadie más la vea.", "grooming"),
    ("¿Jugamos Minecraft esta tarde?", "no grooming"),
    ("Sos muy especial para mí, no le cuentes a nadie lo que hablamos.", "grooming"),
    ("¿Querés que te pase los apuntes de biología?", "no grooming"),
    ("¿Podés mostrarme cómo estás vestida ahora?", "grooming"),
    ("¿Tenés ganas de salir a andar en bici mañana?", "no grooming"),
    ("No hace falta que le digas a nadie que hablamos tanto.", "grooming"),
    ("¿Querés que estudiemos juntos para el parcial?", "no grooming"),
]
df = pd.DataFrame(datos, columns=["mensaje", "etiqueta"])

print("=" * 60)
print("DATASET")
print("=" * 60)
print(f"Total de mensajes: {len(df)}")
print(df["etiqueta"].value_counts().to_string(), "\n")

# ---------------------------------------------------------------------------
# 2. División entrenamiento / prueba (70 / 30, estratificada)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["mensaje"],
    df["etiqueta"],
    test_size=0.3,
    stratify=df["etiqueta"],
    random_state=2,  # semilla: fija el sorteo para que el resultado sea reproducible
)

# ---------------------------------------------------------------------------
# 3. Modelo: bolsa de palabras + Naive Bayes multinomial
#    No se eliminan stopwords: "no", "nadie", "solo" son señales clave.
# ---------------------------------------------------------------------------
modelo = make_pipeline(
    CountVectorizer(lowercase=True),
    MultinomialNB(),
)
modelo.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 4. Evaluación
# ---------------------------------------------------------------------------
y_pred = modelo.predict(X_test)
clases = ["grooming", "no grooming"]

accuracy = accuracy_score(y_test, y_pred)
reporte_txt = classification_report(y_test, y_pred, labels=clases, zero_division=0)
reporte_dict = classification_report(
    y_test, y_pred, labels=clases, zero_division=0, output_dict=True
)
matriz = confusion_matrix(y_test, y_pred, labels=clases)

print("=" * 60)
print("1) PRECISIÓN DEL MODELO (ACCURACY)")
print("=" * 60)
print(f"Accuracy: {accuracy:.2f} ({accuracy * 100:.0f}%)\n")

print("=" * 60)
print("2) REPORTE DE CLASIFICACIÓN")
print("=" * 60)
print(reporte_txt)

print("=" * 60)
print("3) MATRIZ DE CONFUSIÓN")
print("=" * 60)
print(pd.DataFrame(
    matriz,
    index=[f"real: {c}" for c in clases],
    columns=[f"pred: {c}" for c in clases],
).to_string(), "\n")

print("Mensajes del conjunto de prueba:")
for msg, real, pred in zip(X_test, y_test, y_pred):
    marca = "OK " if real == pred else "ERR"
    print(f"  [{marca}] real={real:<12} pred={pred:<12} | {msg}")
print()

fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay(matriz, display_labels=clases).plot(
    ax=ax, cmap="Blues", colorbar=False
)
ax.set_title("Matriz de confusión")
ax.set_xlabel("Predicción")
ax.set_ylabel("Valor real")
plt.tight_layout()
plt.savefig("matriz_confusion.png", dpi=200)
plt.close()

# Validación cruzada: estimación más robusta con un dataset tan chico
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    make_pipeline(CountVectorizer(lowercase=True), MultinomialNB()),
    df["mensaje"], df["etiqueta"], cv=cv,
)
print("Validación cruzada (5 folds) sobre los 20 mensajes:")
print(f"  Accuracy por fold: {[round(float(s), 2) for s in cv_scores]}")
print(f"  Promedio: {cv_scores.mean():.2f}  (desvío: {cv_scores.std():.2f})\n")

# Robustez: repetir el sorteo 70/30 con 200 semillas distintas
accs = []
for semilla in range(200):
    a, b, c, d = train_test_split(
        df["mensaje"], df["etiqueta"], test_size=0.3,
        stratify=df["etiqueta"], random_state=semilla,
    )
    m = make_pipeline(CountVectorizer(lowercase=True), MultinomialNB()).fit(a, c)
    accs.append(accuracy_score(d, m.predict(b)))
accs = pd.Series(accs)
print("Repetición del sorteo con 200 semillas distintas:")
print(f"  Accuracy promedio: {accs.mean():.2f}")
print(f"  Mínimo: {accs.min():.2f}  |  Máximo: {accs.max():.2f}")
print(f"  Sorteos con 100% de acierto: {(accs == 1).mean() * 100:.0f}%\n")

# ---------------------------------------------------------------------------
# 5. Clasificación de mensajes nuevos
# ---------------------------------------------------------------------------
nuevos = [
    "¿Podés mandarme una foto tuya? No se la muestres a nadie.",
    "¿Jugamos Minecraft esta tarde?",
    "No le digas a tus papás que hablamos, ¿sí?",
    "¿Terminaste el trabajo práctico de historia?",
]
pred_nuevos = modelo.predict(nuevos)
proba_nuevos = modelo.predict_proba(nuevos)
idx_g = list(modelo.classes_).index("grooming")

print("=" * 60)
print("4) CLASIFICACIÓN DE NUEVOS MENSAJES")
print("=" * 60)
for msg, pred, proba in zip(nuevos, pred_nuevos, proba_nuevos):
    print(f"  {pred:<12} (P(grooming)={proba[idx_g]:.2f}) | {msg}")

# Resultados en JSON (los usa el generador del informe PDF)
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump({
        "n_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": accuracy,
        "reporte": reporte_dict,
        "matriz": matriz.tolist(),
        "clases": clases,
        "test": [
            {"mensaje": m, "real": r, "pred": p}
            for m, r, p in zip(X_test, y_test, y_pred)
        ],
        "cv_scores": cv_scores.tolist(),
        "multi": {
            "media": float(accs.mean()), "min": float(accs.min()),
            "max": float(accs.max()), "pct_100": float((accs == 1).mean()),
            "distribucion": accs.round(2).value_counts().sort_index().to_dict(),
        },
        "nuevos": [
            {"mensaje": m, "pred": p, "p_grooming": float(pr[idx_g])}
            for m, p, pr in zip(nuevos, pred_nuevos, proba_nuevos)
        ],
    }, f, ensure_ascii=False, indent=2)