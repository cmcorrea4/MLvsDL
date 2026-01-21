import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
st.set_page_config(
    page_title="Comparador ML vs Deep Learning",
    layout="wide"
)

st.title("🤖 Comparador: Machine Learning vs Deep Learning")
st.markdown(
    """
    Esta aplicación compara un **modelo clásico de Machine Learning**
    con un **modelo de Deep Learning (Red Neuronal)** usando el mismo dataset.
    """
)

# -------------------------------
# DATASET
# -------------------------------
st.header("📊 Dataset")
data = load_iris()
X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

st.write("Dataset usado: **Iris** (clasificación multiclase)")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Parámetros del modelo DL")

hidden_layers = st.sidebar.selectbox(
    "Capas ocultas",
    [(10,), (20,), (20, 10), (50, 20)]
)

max_iter = st.sidebar.slider(
    "Iteraciones (entrenamiento)",
    100, 1000, 300
)

# -------------------------------
# MACHINE LEARNING
# -------------------------------
st.header("📌 Machine Learning clásico")
start_ml = time.time()

ml_model = LogisticRegression(max_iter=300)
ml_model.fit(X_train, y_train)

ml_time = time.time() - start_ml
y_pred_ml = ml_model.predict(X_test)
acc_ml = accuracy_score(y_test, y_pred_ml)

# -------------------------------
# DEEP LEARNING (MLP)
# -------------------------------
st.header("🧠 Deep Learning (Red Neuronal)")

start_dl = time.time()

dl_model = MLPClassifier(
    hidden_layer_sizes=hidden_layers,
    max_iter=max_iter,
    random_state=42
)

dl_model.fit(X_train, y_train)

dl_time = time.time() - start_dl
y_pred_dl = dl_model.predict(X_test)
acc_dl = accuracy_score(y_test, y_pred_dl)

# -------------------------------
# RESULTADOS
# -------------------------------
st.header("📈 Resultados")

col1, col2 = st.columns(2)

with col1:
    st.subheader("ML clásico")
    st.metric("Accuracy", f"{acc_ml:.3f}")
    st.metric("Tiempo entrenamiento (s)", f"{ml_time:.2f}")

with col2:
    st.subheader("Deep Learning")
    st.metric("Accuracy", f"{acc_dl:.3f}")
    st.metric("Tiempo entrenamiento (s)", f"{dl_time:.2f}")

# -------------------------------
# MATRIZ DE CONFUSIÓN
# -------------------------------
st.header("📉 Matriz de Confusión")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

cm_ml = confusion_matrix(y_test, y_pred_ml)
cm_dl = confusion_matrix(y_test, y_pred_dl)

ax[0].imshow(cm_ml)
ax[0].set_title("ML clásico")
ax[0].set_xlabel("Predicción")
ax[0].set_ylabel("Real")

ax[1].imshow(cm_dl)
ax[1].set_title("Deep Learning")
ax[1].set_xlabel("Predicción")
ax[1].set_ylabel("Real")

st.pyplot(fig)

# -------------------------------
# CONCLUSIONES
# -------------------------------
st.header("🧠 Conclusión didáctica")

if acc_dl > acc_ml:
    st.success(
        "El modelo de Deep Learning obtiene mejor desempeño, "
        "pero con mayor costo computacional."
    )
else:
    st.warning(
        "El modelo clásico es suficiente para este problema simple."
    )

st.info(
    """
    **Mensaje clave del curso**  
    - Deep Learning **no siempre es la mejor opción**
    - Para datasets pequeños, ML clásico suele ser más eficiente  
    - DL cobra valor con **más datos y mayor complejidad**
    """
)

