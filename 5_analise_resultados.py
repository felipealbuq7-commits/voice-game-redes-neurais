"""
ETAPA 5 - ANÁLISE DE RESULTADOS (para apresentação)
Carrega o modelo treinado, faz previsões no conjunto de teste
e gera todos os gráficos e discussões sobre acertos/erros.

Útil para responder:
  • Onde a rede erra mais?
  • Por que ela erra?
  • Como a IA "aprende"?
  • Quais são as limitações?
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tensorflow import keras

# ── Configurações ────────────────────────────────────────────────────────────
SEED      = 42
TEST_SIZE = 0.2

BASE_DIR  = os.path.dirname(__file__)
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def main():
    print("=" * 65)
    print("  ANÁLISE DE RESULTADOS – Voice Game")
    print("=" * 65)

    # ── Carrega dados e modelo ────────────────────────────────────────────────
    X       = np.load(os.path.join(PROC_DIR, "X.npy"))
    y_raw   = np.load(os.path.join(PROC_DIR, "y.npy"))
    classes = np.load(os.path.join(PROC_DIR, "classes.npy"), allow_pickle=True)
    modelo  = keras.models.load_model(os.path.join(MODEL_DIR, "voice_model.h5"))
    mean    = np.load(os.path.join(MODEL_DIR, "scaler_mean.npy"))
    scale   = np.load(os.path.join(MODEL_DIR, "scaler_scale.npy"))

    # ── Pré-processamento idêntico ao treino ─────────────────────────────────
    X_scaled = (X - mean) / scale
    _, X_test, _, y_test_idx = train_test_split(
        X_scaled, y_raw, test_size=TEST_SIZE, random_state=SEED, stratify=y_raw
    )
    y_test_onehot = keras.utils.to_categorical(y_test_idx, num_classes=len(classes))

    # ── Previsões ─────────────────────────────────────────────────────────────
    y_probs = modelo.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_probs, axis=1)

    loss, acc = modelo.evaluate(X_test, y_test_onehot, verbose=0)

    # ── Relatório de texto ────────────────────────────────────────────────────
    print(f"\n  Acurácia geral: {acc * 100:.2f}%   |   Loss: {loss:.4f}")
    print("\n  Relatório por classe:\n")
    print(classification_report(y_test_idx, y_pred, target_names=classes))

    # ── Exemplos de acertos e erros ───────────────────────────────────────────
    acertos = [(i, y_probs[i]) for i in range(len(y_test_idx)) if y_test_idx[i] == y_pred[i]]
    erros   = [(i, y_probs[i]) for i in range(len(y_test_idx)) if y_test_idx[i] != y_pred[i]]

    print(f"\n  Total acertos: {len(acertos)}  |  Total erros: {len(erros)}")
    print(f"\n  {'#':<4} {'Real':<12} {'Previsto':<12} {'Confiança':<10}")
    print("  " + "-" * 45)

    for i, probs in erros[:10]:
        real     = classes[y_test_idx[i]]
        previsto = classes[y_pred[i]]
        conf     = np.max(probs) * 100
        print(f"  {i:<4} {real:<12} {previsto:<12} {conf:.1f}%  ← ERRO")

    # ── Gráficos ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Análise Completa do Modelo – Voice Game", fontsize=16, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Matriz de Confusão
    ax1 = fig.add_subplot(gs[0, :2])
    cm  = confusion_matrix(y_test_idx, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title("Matriz de Confusão", fontsize=13)
    ax1.set_ylabel("Real")
    ax1.set_xlabel("Previsto")

    # 2. Acurácia por classe
    ax2 = fig.add_subplot(gs[0, 2])
    acuracias_por_classe = cm.diagonal() / cm.sum(axis=1)
    cores = ["#27ae60" if a >= 0.8 else "#f39c12" if a >= 0.6 else "#e74c3c"
             for a in acuracias_por_classe]
    ax2.barh(classes, acuracias_por_classe * 100, color=cores, edgecolor="black")
    ax2.set_xlim(0, 110)
    ax2.set_xlabel("Acurácia (%)")
    ax2.set_title("Acurácia por Classe", fontsize=13)
    for i, v in enumerate(acuracias_por_classe):
        ax2.text(v * 100 + 1, i, f"{v*100:.1f}%", va="center", fontsize=9)

    # 3. Distribuição de confiança (acertos vs erros)
    ax3 = fig.add_subplot(gs[1, 0])
    conf_acertos = [np.max(y_probs[i]) for i, _ in acertos]
    conf_erros   = [np.max(y_probs[i]) for i, _ in erros]
    ax3.hist(conf_acertos, bins=15, alpha=0.7, color="#27ae60", label=f"Acertos (n={len(acertos)})")
    ax3.hist(conf_erros,   bins=15, alpha=0.7, color="#e74c3c", label=f"Erros (n={len(erros)})")
    ax3.axvline(0.7, color="black", linestyle="--", label="Limiar 70%")
    ax3.set_title("Distribuição de Confiança", fontsize=13)
    ax3.set_xlabel("Confiança do Modelo")
    ax3.set_ylabel("Frequência")
    ax3.legend()

    # 4. Histograma de erros por classe
    ax4 = fig.add_subplot(gs[1, 1])
    erros_por_classe = {c: 0 for c in classes}
    for i, _ in erros:
        erros_por_classe[classes[y_test_idx[i]]] += 1
    ax4.bar(erros_por_classe.keys(), erros_por_classe.values(),
            color="#e74c3c", edgecolor="black")
    ax4.set_title("Erros por Classe", fontsize=13)
    ax4.set_xlabel("Comando")
    ax4.set_ylabel("Quantidade de Erros")

    # 5. Curva de aprendizado (carregada do historico se existir)
    ax5 = fig.add_subplot(gs[1, 2])
    hist_path = os.path.join(MODEL_DIR, "historico_treinamento.png")
    if os.path.exists(hist_path):
        from matplotlib.image import imread
        img = imread(hist_path)
        ax5.imshow(img)
        ax5.axis("off")
        ax5.set_title("Histórico de Treino", fontsize=13)
    else:
        ax5.text(0.5, 0.5, "Execute\n3_treinar_modelo.py\nprimeiro",
                 ha="center", va="center", fontsize=12, color="gray")
        ax5.axis("off")

    plt.savefig(os.path.join(MODEL_DIR, "analise_completa.png"), dpi=120, bbox_inches="tight")
    plt.show()
    print(f"\n  ✔  Análise salva em models/analise_completa.png")

    # ── Discussão impressa ────────────────────────────────────────────────────
    _imprimir_discussao(classes, acuracias_por_classe, len(acertos), len(erros), acc)


def _imprimir_discussao(classes, acuracias, n_acertos, n_erros, acc_geral):
    """Imprime a discussão guiada para a apresentação."""
    classe_pior  = classes[np.argmin(acuracias)]
    classe_melhor = classes[np.argmax(acuracias)]

    print("\n" + "=" * 65)
    print("  DISCUSSÃO – Como a IA aprende e suas limitações")
    print("=" * 65)

    print(f"""
  📊 RESULTADOS GERAIS
  ─────────────────────────────────────────────────────────────
  Acurácia geral: {acc_geral * 100:.1f}%
  Acertos: {n_acertos}  |  Erros: {n_erros}
  Melhor classe: '{classe_melhor}' ({np.max(acuracias)*100:.1f}%)
  Pior classe: '{classe_pior}' ({np.min(acuracias)*100:.1f}%)

  🧠 COMO A REDE APRENDE?
  ─────────────────────────────────────────────────────────────
  1. Cada áudio é convertido em MFCC: 40 números que representam
     como as frequências mudam ao longo do tempo (parecido com
     o que o ouvido humano faz naturalmente).

  2. A rede neural ajusta milhares de pesos durante o treino
     usando Backpropagation + Gradiente Descendente, minimizando
     a diferença entre o que ela prevê e o rótulo real (Loss).

  3. Dropout e BatchNorm evitam que a rede "decore" os dados de
     treino — ela precisa generalizar para vozes novas.

  ⚠️  POSSÍVEIS CAUSAS DOS ERROS
  ─────────────────────────────────────────────────────────────
  • Poucas amostras: com apenas 10 gravações por comando, a rede
    tem pouco material para aprender variações naturais da voz.

  • Sons parecidos: palavras com sons similares (ex: "parar" e
    "atacar") podem ser confundidas pelos MFCCs.

  • Ruído ambiental: microfone capta sons externos que o modelo
    não viu no treino.

  • Variação de falante: o modelo foi treinado com a voz de uma
    pessoa; outra pessoa pode ter padrão de frequência diferente.

  🔧 MELHORIAS POSSÍVEIS
  ─────────────────────────────────────────────────────────────
  • Gravar mais amostras (50+ por comando) com pessoas diferentes.
  • Aplicar Data Augmentation: pitch shift, ruído, time stretch.
  • Usar CNN 2D no espectrograma em vez de Dense layers.
  • Testar RNNs (LSTM) que capturam dependência temporal do áudio.
""")


if __name__ == "__main__":
    main()
