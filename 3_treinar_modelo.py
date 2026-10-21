"""
ETAPA 3 - CRIAÇÃO E TREINAMENTO DA REDE NEURAL (Keras/TensorFlow)

Arquitetura:
  Entrada  → Dense(256) + BatchNorm + Dropout(0.4)
           → Dense(128) + BatchNorm + Dropout(0.3)
           → Dense(64)  + BatchNorm + Dropout(0.2)
  Saída    → Dense(n_classes, softmax)

Por que essas camadas?
  • BatchNormalization: estabiliza o treinamento, normaliza as ativações.
  • Dropout: previne overfitting (a rede não "decora" as amostras de treino).
  • Softmax na saída: converte logits em probabilidades que somam 1.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import confusion_matrix, classification_report
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Configurações ────────────────────────────────────────────────────────────
SEED        = 42
EPOCHS      = 100
BATCH_SIZE  = 16
TEST_SIZE   = 0.2     # 20 % dos dados para teste
VAL_SIZE    = 0.15    # 15 % dos dados de treino para validação

BASE_DIR  = os.path.dirname(__file__)
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── 1. Carregar dados ────────────────────────────────────────────────────────
def carregar_dados():
    X       = np.load(os.path.join(PROC_DIR, "X.npy"))
    y_raw   = np.load(os.path.join(PROC_DIR, "y.npy"))
    classes = np.load(os.path.join(PROC_DIR, "classes.npy"), allow_pickle=True)
    return X, y_raw, classes


# ── 2. Pré-processamento ─────────────────────────────────────────────────────
def preprocessar(X, y_raw, classes):
    # Normalização Z-score: média 0, desvio 1 — essencial para redes densas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # One-hot encoding dos rótulos
    n_classes = len(classes)
    y_onehot  = keras.utils.to_categorical(y_raw, num_classes=n_classes)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test, y_idx_train, y_idx_test = train_test_split(
        X_scaled, y_onehot, y_raw,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_raw,          # mantém proporção das classes
    )

    return X_train, X_test, y_train, y_test, y_idx_train, y_idx_test, scaler


# ── 3. Construção do modelo ──────────────────────────────────────────────────
def construir_modelo(input_dim: int, n_classes: int) -> keras.Model:
    modelo = keras.Sequential(name="VoiceCommandNet")

    # Camada 1
    modelo.add(layers.Dense(256, input_dim=input_dim))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.Activation("relu"))
    modelo.add(layers.Dropout(0.4))

    # Camada 2
    modelo.add(layers.Dense(128))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.Activation("relu"))
    modelo.add(layers.Dropout(0.3))

    # Camada 3
    modelo.add(layers.Dense(64))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.Activation("relu"))
    modelo.add(layers.Dropout(0.2))

    # Saída
    modelo.add(layers.Dense(n_classes, activation="softmax"))

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


# ── 4. Treinamento ───────────────────────────────────────────────────────────
def treinar(modelo, X_train, y_train):
    callbacks = [
        # Para se a val_loss não melhorar por 15 épocas consecutivas
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        # Reduz o LR quando val_loss estagna
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    historico = modelo.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    return historico


# ── 5. Visualizações ─────────────────────────────────────────────────────────
def plotar_historico(historico) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Histórico de Treinamento", fontsize=14)

    # Acurácia
    ax1.plot(historico.history["accuracy"],     label="Treino",    color="steelblue")
    ax1.plot(historico.history["val_accuracy"], label="Validação", color="darkorange")
    ax1.set_title("Acurácia por Época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Acurácia")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(historico.history["loss"],     label="Treino",    color="steelblue")
    ax2.plot(historico.history["val_loss"], label="Validação", color="darkorange")
    ax2.set_title("Loss por Época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "historico_treinamento.png"), dpi=120)
    plt.show()


def plotar_matriz_confusao(y_real, y_pred, classes) -> None:
    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
    )
    plt.title("Matriz de Confusão", fontsize=14)
    plt.ylabel("Rótulo Real")
    plt.xlabel("Rótulo Previsto")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "matriz_confusao.png"), dpi=120)
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  TREINAMENTO DA REDE NEURAL – Voice Game")
    print("=" * 60)

    # 1. Dados
    X, y_raw, classes = carregar_dados()
    print(f"\n  Dados carregados: {X.shape[0]} amostras, {len(classes)} classes")
    print(f"  Classes: {list(classes)}")

    # 2. Pré-processamento
    X_train, X_test, y_train, y_test, y_idx_train, y_idx_test, scaler = preprocessar(X, y_raw, classes)
    print(f"\n  Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

    # 3. Modelo
    modelo = construir_modelo(X_train.shape[1], len(classes))
    modelo.summary()

    # 4. Treino
    print("\n  Iniciando treinamento...\n")
    historico = treinar(modelo, X_train, y_train)

    # 5. Avaliação no conjunto de teste
    print("\n  Avaliando no conjunto de teste...")
    loss, acc = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"\n  ╔══════════════════════════════╗")
    print(f"  ║  Loss de Teste  : {loss:.4f}      ║")
    print(f"  ║  Acurácia Teste : {acc * 100:.2f} %   ║")
    print(f"  ╚══════════════════════════════╝")

    # Previsões para métricas detalhadas
    y_pred_prob = modelo.predict(X_test)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    print("\n  Relatório de Classificação:\n")
    print(classification_report(y_idx_test, y_pred, target_names=classes))

    # 6. Visualizações
    plotar_historico(historico)
    plotar_matriz_confusao(y_idx_test, y_pred, classes)

    # 7. Mostra acertos e erros individualmente
    print("\n  Exemplos de Acertos e Erros:")
    print(f"  {'#':<5} {'Real':<12} {'Previsto':<12} {'Confiança':<10} {'Status'}")
    print("  " + "-" * 55)
    for i in range(min(20, len(y_idx_test))):
        real     = classes[y_idx_test[i]]
        previsto = classes[y_pred[i]]
        confianca = np.max(y_pred_prob[i]) * 100
        status   = "✔ ACERTO" if real == previsto else "✘ ERRO"
        print(f"  {i+1:<5} {real:<12} {previsto:<12} {confianca:<10.1f} {status}")

    # 8. Salva modelo e scaler
    modelo.save(os.path.join(MODEL_DIR, "voice_model.h5"))
    np.save(os.path.join(MODEL_DIR, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(MODEL_DIR, "scaler_scale.npy"), scaler.scale_)
    np.save(os.path.join(MODEL_DIR, "classes.npy"), classes)

    print(f"\n  ✔  Modelo salvo em models/voice_model.h5")
    print("\n" + "=" * 60)
    print("  Treinamento concluído! Execute agora: 4_jogo_demo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
