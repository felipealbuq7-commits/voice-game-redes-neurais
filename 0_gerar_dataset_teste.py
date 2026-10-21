"""
SCRIPT AUXILIAR – Gerar dataset sintético para testes
Use esse script APENAS se quiser testar o pipeline sem gravar áudios.
Ele cria arquivos .wav com tons sintéticos distintos por comando,
simulando uma separação simples entre classes.

⚠️ Esse dataset NÃO representa voz real — serve apenas para validar
   que o pipeline roda sem erros. Para o projeto real, use o
   script 1_gravar_audios.py.
"""

import os
import numpy as np
import soundfile as sf

SAMPLE_RATE  = 22050
DURACAO_SEG  = 1.5
N_SAMPLES    = int(SAMPLE_RATE * DURACAO_SEG)
AMOSTRAS     = 20   # amostras por comando

BASE_DIR  = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw_audio")

# Frequência base diferente para cada comando (Hz)
# Em dados reais isso seria substituído por padrões de voz
COMANDOS_FREQ = {
    "pular":   [300, 450, 600],
    "atacar":  [800, 1000, 1200],
    "correr":  [200, 350, 500],
    "parar":   [1400, 1600, 1800],
    "agachar": [100, 150, 200],
}


def gerar_audio_sintetico(freqs: list, idx: int) -> np.ndarray:
    """Gera um sinal com múltiplas frequências + ruído leve."""
    t = np.linspace(0, DURACAO_SEG, N_SAMPLES)
    audio = np.zeros(N_SAMPLES)
    for freq in freqs:
        # Variação leve para simular amostras diferentes
        variacao = freq * (1 + np.random.uniform(-0.05, 0.05))
        audio += np.sin(2 * np.pi * variacao * t)
    # Normaliza e adiciona ruído
    audio = audio / len(freqs)
    audio += np.random.normal(0, 0.02, N_SAMPLES)
    return audio.astype(np.float32)


def main():
    print("=" * 55)
    print("  GERANDO DATASET SINTÉTICO")
    print("=" * 55)

    for comando, freqs in COMANDOS_FREQ.items():
        pasta = os.path.join(AUDIO_DIR, comando)
        os.makedirs(pasta, exist_ok=True)

        for i in range(1, AMOSTRAS + 1):
            audio  = gerar_audio_sintetico(freqs, i)
            caminho = os.path.join(pasta, f"{comando}_{i:03d}.wav")
            sf.write(caminho, audio, SAMPLE_RATE)

        print(f"  ✔  '{comando}': {AMOSTRAS} amostras criadas")

    print(f"\n  Dataset criado em data/raw_audio/")
    print("  Agora execute: python 2_processar_dados.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
