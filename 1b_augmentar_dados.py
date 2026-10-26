"""
ETAPA 1B - DATA AUGMENTATION
Gera variações dos áudios reais gravados para aumentar o dataset.

Técnicas aplicadas em cada áudio original:
  - Variação 1: ruído leve + leve variação de velocidade (esticar)
  - Variação 2: shift de pitch (voz levemente mais aguda/grave)

Resultado: ~3x mais amostras sem gravar nada novo.

Como usar:
  python 1b_augmentar_dados.py
"""

import os
import numpy as np
import soundfile as sf
import librosa

# ── Configurações ────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
DURACAO_SEG = 2.0
N_SAMPLES   = int(SAMPLE_RATE * DURACAO_SEG)

BASE_DIR  = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw_audio")


def pad_ou_cortar(audio: np.ndarray) -> np.ndarray:
    """Garante que o áudio tenha exatamente N_SAMPLES."""
    if len(audio) < N_SAMPLES:
        return np.pad(audio, (0, N_SAMPLES - len(audio)))
    return audio[:N_SAMPLES]


def augmentar_ruido(audio: np.ndarray) -> np.ndarray:
    """Adiciona ruído gaussiano leve."""
    ruido = np.random.normal(0, 0.005, len(audio))
    return np.clip(audio + ruido, -1.0, 1.0).astype(np.float32)


def augmentar_pitch(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Shift de pitch sem alterar a duração."""
    shifted = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=semitones)
    return pad_ou_cortar(shifted).astype(np.float32)


def augmentar_velocidade(audio: np.ndarray, rate: float) -> np.ndarray:
    """Altera levemente a velocidade (time stretch)."""
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return pad_ou_cortar(stretched).astype(np.float32)


def contar_existentes(pasta: str) -> int:
    return len([f for f in os.listdir(pasta) if f.endswith(".wav")])


def main() -> None:
    print("=" * 60)
    print("  DATA AUGMENTATION – Voice Game")
    print("=" * 60)

    comandos = sorted([
        d for d in os.listdir(AUDIO_DIR)
        if os.path.isdir(os.path.join(AUDIO_DIR, d))
    ])

    if not comandos:
        print("\n❌  Nenhuma pasta encontrada em data/raw_audio/")
        print("   Execute primeiro o script 1_gravar_audios.py")
        return

    total_gerados = 0

    for comando in comandos:
        pasta = os.path.join(AUDIO_DIR, comando)

        # Pega só os áudios originais (sem os que já foram augmentados)
        originais = sorted([
            f for f in os.listdir(pasta)
            if f.endswith(".wav") and "_aug" not in f
        ])

        if not originais:
            print(f"\n  ⚠  '{comando}': nenhum áudio original encontrado, pulando.")
            continue

        # Remove augmentações anteriores para não duplicar
        augmentados_antigos = [f for f in os.listdir(pasta) if "_aug" in f]
        for f in augmentados_antigos:
            os.remove(os.path.join(pasta, f))

        print(f"\n  '{comando.upper()}': {len(originais)} originais → gerando variações...")

        gerados = 0
        for arq in originais:
            caminho = os.path.join(pasta, arq)
            nome_base = arq.replace(".wav", "")

            try:
                audio, _ = librosa.load(caminho, sr=SAMPLE_RATE, mono=True)
                audio = pad_ou_cortar(audio)

                # Variação 1: ruído leve
                v1 = augmentar_ruido(audio)
                sf.write(os.path.join(pasta, f"{nome_base}_aug1.wav"), v1, SAMPLE_RATE)

                # Variação 2: pitch levemente mais agudo (+1.5 semitons)
                v2 = augmentar_pitch(audio, semitones=1.5)
                sf.write(os.path.join(pasta, f"{nome_base}_aug2.wav"), v2, SAMPLE_RATE)

                # Variação 3: pitch levemente mais grave (-1.5 semitons)
                v3 = augmentar_pitch(audio, semitones=-1.5)
                sf.write(os.path.join(pasta, f"{nome_base}_aug3.wav"), v3, SAMPLE_RATE)

                gerados += 3

            except Exception as e:
                print(f"    ⚠  Erro em {arq}: {e}")

        total_original = len(originais)
        total_com_aug  = total_original + gerados
        print(f"  ✔  {gerados} variações geradas → total: {total_com_aug} amostras")
        total_gerados += gerados

    print(f"\n  Total de amostras geradas: {total_gerados}")
    print("\n" + "=" * 60)
    print("  Augmentation concluída! Execute agora: 2_processar_dados.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
