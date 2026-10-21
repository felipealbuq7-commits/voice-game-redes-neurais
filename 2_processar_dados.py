"""
ETAPA 2 - PROCESSAMENTO DE DADOS COM LIBROSA
Carrega os arquivos .wav, extrai features (MFCC) e salva
os arrays prontos para o treinamento da rede neural.

O que são MFCCs?
  Mel-Frequency Cepstral Coefficients – representação compacta
  do espectro de frequências do áudio, muito usada em
  reconhecimento de fala porque imita como o ouvido humano percebe o som.

Saída:
  data/processed/X.npy  – features (amostras × features)
  data/processed/y.npy  – rótulos numéricos
  data/processed/classes.npy – mapeamento índice → nome do comando
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ── Configurações ────────────────────────────────────────────────────────────
SAMPLE_RATE  = 22050
DURACAO_SEG  = 1.5
N_MFCC       = 40     # quantidade de coeficientes MFCC
N_SAMPLES    = int(SAMPLE_RATE * DURACAO_SEG)

BASE_DIR      = os.path.dirname(__file__)
AUDIO_DIR     = os.path.join(BASE_DIR, "data", "raw_audio")
PROC_DIR      = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)


def extrair_mfcc(caminho: str) -> np.ndarray:
    """
    Carrega um arquivo .wav com Librosa e extrai os MFCCs.
    Retorna um vetor 1-D de tamanho N_MFCC.
    """
    # Carrega o áudio garantindo sample rate fixo e mono
    audio, sr = librosa.load(caminho, sr=SAMPLE_RATE, mono=True)

    # Padeia ou corta para ter sempre o mesmo comprimento
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]

    # Extrai os MFCCs e calcula a média temporal → vetor de tamanho N_MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)   # shape: (N_MFCC,)


def visualizar_exemplo(caminho: str, comando: str) -> None:
    """Plota a forma de onda e o espectrograma MFCC de um exemplo."""
    audio, sr = librosa.load(caminho, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(f"Visualização do áudio – comando: '{comando}'", fontsize=14)

    # Forma de onda
    librosa.display.waveshow(audio, sr=sr, ax=axes[0], color="steelblue")
    axes[0].set_title("Forma de onda (Waveform)")
    axes[0].set_xlabel("Tempo (s)")
    axes[0].set_ylabel("Amplitude")

    # Espectrograma MFCC
    img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=axes[1], cmap="viridis")
    axes[1].set_title("MFCCs (Mel-Frequency Cepstral Coefficients)")
    axes[1].set_ylabel("Coeficiente MFCC")
    fig.colorbar(img, ax=axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(PROC_DIR, f"exemplo_{comando}.png"), dpi=120)
    plt.show()
    print(f"  ✔  Gráfico salvo em data/processed/exemplo_{comando}.png")


def main() -> None:
    print("=" * 60)
    print("  PROCESSAMENTO DE DADOS COM LIBROSA")
    print("=" * 60)

    # Descobre as classes (pastas dentro de raw_audio)
    classes = sorted([
        d for d in os.listdir(AUDIO_DIR)
        if os.path.isdir(os.path.join(AUDIO_DIR, d))
    ])
    if not classes:
        print("\n❌  Nenhuma pasta de áudio encontrada em data/raw_audio/")
        print("   Execute primeiro o script 1_gravar_audios.py")
        return

    print(f"\nClasses encontradas: {classes}\n")

    X, y = [], []

    for idx, comando in enumerate(classes):
        pasta = os.path.join(AUDIO_DIR, comando)
        arquivos = [f for f in os.listdir(pasta) if f.endswith(".wav")]
        print(f"  Processando '{comando}' ({len(arquivos)} amostras)...")

        for arq in arquivos:
            caminho = os.path.join(pasta, arq)
            try:
                features = extrair_mfcc(caminho)
                X.append(features)
                y.append(idx)
            except Exception as e:
                print(f"    ⚠  Erro em {arq}: {e}")

        # Visualiza o primeiro áudio de cada classe
        if arquivos:
            visualizar_exemplo(os.path.join(pasta, arquivos[0]), comando)

    X = np.array(X)
    y = np.array(y)

    print(f"\n  Shape dos dados: X={X.shape}, y={y.shape}")
    print(f"  Distribuição por classe:")
    for i, c in enumerate(classes):
        print(f"    [{i}] {c}: {np.sum(y == i)} amostras")

    # Salva os dados processados
    np.save(os.path.join(PROC_DIR, "X.npy"),       X)
    np.save(os.path.join(PROC_DIR, "y.npy"),       y)
    np.save(os.path.join(PROC_DIR, "classes.npy"), np.array(classes))

    print(f"\n  ✔  Dados salvos em data/processed/")

    # Visualiza distribuição das amostras
    plt.figure(figsize=(8, 4))
    contagens = [np.sum(y == i) for i in range(len(classes))]
    bars = plt.bar(classes, contagens, color="steelblue", edgecolor="black")
    plt.title("Distribuição das amostras por comando")
    plt.xlabel("Comando")
    plt.ylabel("Número de amostras")
    for bar, val in zip(bars, contagens):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(val), ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PROC_DIR, "distribuicao.png"), dpi=120)
    plt.show()
    print("  ✔  Gráfico de distribuição salvo.")

    print("\n" + "=" * 60)
    print("  Processamento concluído! Execute agora: 3_treinar_modelo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
