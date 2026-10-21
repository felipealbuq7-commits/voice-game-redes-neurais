"""
ETAPA 1 - COLETA DE DADOS
Script para gravar comandos de voz do jogo.

Comandos disponíveis:
  - pular    : personagem pula
  - atacar   : personagem ataca
  - correr   : personagem corre
  - parar    : personagem para
  - agachar  : personagem agacha

Como usar:
  python 1_gravar_audios.py
  Siga as instruções na tela para gravar cada comando.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time

# ── Configurações de áudio ──────────────────────────────────────────────────
SAMPLE_RATE = 22050   # Hz – mesmo valor que o Librosa usa por padrão
DURACAO_SEG = 1.5     # segundos por gravação
REPETICOES  = 10      # quantas amostras gravar por comando

# Comandos do jogo (cada um vira uma pasta dentro de data/raw_audio)
COMANDOS = ["pular", "atacar", "correr", "parar", "agachar"]

BASE_DIR  = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw_audio")


def gravar_amostra(comando: str, indice: int) -> None:
    """Grava uma única amostra de áudio e salva em .wav."""
    pasta = os.path.join(AUDIO_DIR, comando)
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, f"{comando}_{indice:03d}.wav")

    print(f"\n  ▶  Fale '{comando.upper()}' em 3... ", end="", flush=True)
    for i in (3, 2, 1):
        print(f"{i}... ", end="", flush=True)
        time.sleep(0.8)
    print("GRAVANDO!", flush=True)

    audio = sd.rec(
        int(DURACAO_SEG * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()  # aguarda fim da gravação

    sf.write(caminho, audio, SAMPLE_RATE)
    print(f"  ✔  Salvo em: {caminho}")


def main() -> None:
    print("=" * 60)
    print("  GRAVADOR DE COMANDOS DE VOZ – Voice Game")
    print("=" * 60)
    print(f"\nCada comando será gravado {REPETICOES} vezes ({DURACAO_SEG}s cada).")
    print("Tente variar um pouco o tom/velocidade para deixar o dataset mais robusto.\n")

    for comando in COMANDOS:
        input(f"\n>>> Pressione ENTER para começar a gravar '{comando.upper()}' <<<")
        for i in range(1, REPETICOES + 1):
            print(f"\n  Amostra {i}/{REPETICOES}")
            gravar_amostra(comando, i)
            time.sleep(0.5)  # pequena pausa entre gravações

    print("\n" + "=" * 60)
    print("  Coleta concluída! Áudios salvos em data/raw_audio/")
    print("=" * 60)


if __name__ == "__main__":
    main()
