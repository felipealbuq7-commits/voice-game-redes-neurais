"""
ETAPA 4 - DEMO DO JOGO
Interface gráfica com Tkinter que mostra o personagem reagindo
ao comando de voz reconhecido pela rede neural.

Controles:
  • Clique em "🎤 FALAR COMANDO" (ou aperte ESPAÇO) para gravar
  • O personagem executa a ação identificada
  • O histórico de comandos aparece no lado direito
"""

import os
import time
import threading
import numpy as np
import sounddevice as sd
import librosa
import tkinter as tk
from tkinter import font as tkfont
from tensorflow import keras

# ── Configurações ────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
DURACAO_SEG = 1.5
N_MFCC      = 40
N_SAMPLES   = int(SAMPLE_RATE * DURACAO_SEG)

BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Representação ASCII do personagem ────────────────────────────────────────
ESTADOS = {
    "idle": [
        "    O    ",
        "   /|\\  ",
        "   / \\  ",
        "        ",
        " Parado ",
    ],
    "pular": [
        "  \\O/   ",
        "   |    ",
        "  / \\   ",
        "   ^^^  ",
        " Pulou! ",
    ],
    "atacar": [
        "    O~~~",
        "   /|   ",
        "   / \\  ",
        "        ",
        " Atacou!",
    ],
    "correr": [
        "    O>  ",
        "   /|   ",
        "    |\\  ",
        "  >>    ",
        " Correu!",
    ],
    "parar": [
        "    O   ",
        "   /|\\  ",
        "   | |  ",
        "        ",
        " Parou! ",
    ],
    "agachar": [
        "        ",
        "    O   ",
        "   /|\\  ",
        " _/ \\_  ",
        "Agachou!",
    ],
}

CORES_ACAO = {
    "pular":   "#27ae60",
    "atacar":  "#e74c3c",
    "correr":  "#2980b9",
    "parar":   "#f39c12",
    "agachar": "#8e44ad",
    "idle":    "#7f8c8d",
}

EMOJIS = {
    "pular":   "⬆️",
    "atacar":  "⚔️",
    "correr":  "🏃",
    "parar":   "✋",
    "agachar": "⬇️",
    "idle":    "💤",
}


# ── Modelo ────────────────────────────────────────────────────────────────────
class ModeloVoz:
    def __init__(self):
        self.modelo    = keras.models.load_model(os.path.join(MODEL_DIR, "voice_model.h5"))
        self.classes   = np.load(os.path.join(MODEL_DIR, "classes.npy"), allow_pickle=True)
        self.mean      = np.load(os.path.join(MODEL_DIR, "scaler_mean.npy"))
        self.scale     = np.load(os.path.join(MODEL_DIR, "scaler_scale.npy"))

    def prever(self, audio: np.ndarray) -> tuple[str, float]:
        """Recebe array de áudio e retorna (comando, confiança)."""
        # Padeia ou corta
        if len(audio) < N_SAMPLES:
            audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
        else:
            audio = audio[:N_SAMPLES]

        # Extrai MFCCs
        mfcc = librosa.feature.mfcc(y=audio.flatten(), sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        features = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # Normaliza com os parâmetros do treino
        features = (features - self.mean) / self.scale

        # Previsão
        probs   = self.modelo.predict(features, verbose=0)[0]
        idx     = np.argmax(probs)
        return self.classes[idx], float(probs[idx])


# ── Interface ────────────────────────────────────────────────────────────────
class JogoVoz:
    def __init__(self, root: tk.Tk):
        self.root   = root
        self.modelo = ModeloVoz()
        self.estado_atual = "idle"
        self.gravando     = False
        self.historico    = []

        self._construir_ui()
        self._atualizar_personagem("idle")
        root.bind("<space>", lambda e: self._iniciar_gravacao())

    # ── Construção da UI ──────────────────────────────────────────────────────
    def _construir_ui(self):
        self.root.title("🎮 Voice Game – Reconhecimento de Voz")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        fonte_titulo  = tkfont.Font(family="Consolas", size=18, weight="bold")
        fonte_personagem = tkfont.Font(family="Courier New", size=22, weight="bold")
        fonte_info    = tkfont.Font(family="Consolas", size=12)
        fonte_botao   = tkfont.Font(family="Consolas", size=14, weight="bold")
        fonte_hist    = tkfont.Font(family="Consolas", size=10)

        # Título
        tk.Label(
            self.root, text="🎮 VOICE GAME 🎮",
            font=fonte_titulo, bg="#1a1a2e", fg="#e94560"
        ).pack(pady=(15, 5))

        tk.Label(
            self.root, text="Fale um comando e veja o personagem agir!",
            font=fonte_info, bg="#1a1a2e", fg="#aaaaaa"
        ).pack()

        # Frame principal
        frame_main = tk.Frame(self.root, bg="#1a1a2e")
        frame_main.pack(padx=20, pady=10, fill="both")

        # ── Área do personagem (esquerda) ──
        frame_personagem = tk.Frame(frame_main, bg="#16213e", relief="solid", bd=2)
        frame_personagem.pack(side="left", padx=(0, 10), pady=5)

        tk.Label(
            frame_personagem, text="PERSONAGEM",
            font=fonte_info, bg="#16213e", fg="#aaaaaa"
        ).pack(pady=(8, 0))

        self.lbl_personagem = tk.Label(
            frame_personagem, text="",
            font=fonte_personagem, bg="#16213e", fg="#00d2ff",
            justify="left", padx=30, pady=20,
        )
        self.lbl_personagem.pack()

        # Ação atual
        self.lbl_acao = tk.Label(
            frame_personagem, text="",
            font=fonte_titulo, bg="#16213e", fg="#ffffff",
            padx=20, pady=5,
        )
        self.lbl_acao.pack()

        # Confiança
        self.lbl_confianca = tk.Label(
            frame_personagem, text="",
            font=fonte_info, bg="#16213e", fg="#aaaaaa",
            pady=5,
        )
        self.lbl_confianca.pack()

        # ── Histórico (direita) ──
        frame_hist = tk.Frame(frame_main, bg="#16213e", relief="solid", bd=2)
        frame_hist.pack(side="left", pady=5, fill="both", expand=True)

        tk.Label(
            frame_hist, text="HISTÓRICO",
            font=fonte_info, bg="#16213e", fg="#aaaaaa"
        ).pack(pady=(8, 0))

        self.txt_historico = tk.Text(
            frame_hist, width=28, height=12,
            font=fonte_hist, bg="#0f3460", fg="#ffffff",
            relief="flat", padx=8, pady=8, state="disabled",
        )
        self.txt_historico.pack(padx=10, pady=8)

        # ── Barra de status ──
        self.lbl_status = tk.Label(
            self.root, text="Pronto. Pressione ESPAÇO ou clique no botão.",
            font=fonte_info, bg="#1a1a2e", fg="#aaaaaa",
        )
        self.lbl_status.pack(pady=5)

        # Barra de progresso da gravação
        self.canvas_prog = tk.Canvas(
            self.root, width=400, height=20, bg="#16213e",
            relief="flat", bd=0, highlightthickness=0,
        )
        self.canvas_prog.pack()
        self.barra = self.canvas_prog.create_rectangle(0, 0, 0, 20, fill="#e94560", outline="")

        # ── Botão ──
        self.btn_gravar = tk.Button(
            self.root,
            text="🎤  FALAR COMANDO  [ESPAÇO]",
            font=fonte_botao,
            bg="#e94560", fg="white",
            activebackground="#c0392b", activeforeground="white",
            relief="flat", padx=20, pady=12, cursor="hand2",
            command=self._iniciar_gravacao,
        )
        self.btn_gravar.pack(pady=15)

        # Legenda de comandos
        legenda = "Comandos: " + "  |  ".join(
            f"{EMOJIS.get(c, '')} {c}" for c in self.modelo.classes
        )
        tk.Label(
            self.root, text=legenda,
            font=fonte_hist, bg="#1a1a2e", fg="#555577",
        ).pack(pady=(0, 10))

    # ── Personagem ────────────────────────────────────────────────────────────
    def _atualizar_personagem(self, estado: str, confianca: float = 0.0):
        linhas  = ESTADOS.get(estado, ESTADOS["idle"])
        cor     = CORES_ACAO.get(estado, "#00d2ff")
        emoji   = EMOJIS.get(estado, "")
        display = "\n".join(linhas[:-1])   # sem a última linha (legenda)

        self.lbl_personagem.config(text=display, fg=cor)
        if estado != "idle":
            self.lbl_acao.config(
                text=f"{emoji}  {estado.upper()}",
                fg=cor, bg="#16213e",
            )
            self.lbl_confianca.config(
                text=f"Confiança: {confianca * 100:.1f}%",
                fg="#aaaaaa",
            )
        else:
            self.lbl_acao.config(text="💤  IDLE", fg="#7f8c8d", bg="#16213e")
            self.lbl_confianca.config(text="Aguardando comando...", fg="#555577")

    def _voltar_idle(self, delay: float = 2.0):
        self.root.after(int(delay * 1000), lambda: self._atualizar_personagem("idle"))

    # ── Gravação e previsão ───────────────────────────────────────────────────
    def _iniciar_gravacao(self):
        if self.gravando:
            return
        threading.Thread(target=self._gravar_e_prever, daemon=True).start()

    def _gravar_e_prever(self):
        self.gravando = True
        self.root.after(0, lambda: self.btn_gravar.config(state="disabled", bg="#555577"))
        self.root.after(0, lambda: self.lbl_status.config(
            text="🔴 GRAVANDO...", fg="#e94560"
        ))

        # Animação da barra de progresso
        steps  = 30
        delay  = DURACAO_SEG / steps
        for i in range(steps + 1):
            prog = int(400 * i / steps)
            self.root.after(0, lambda p=prog: self.canvas_prog.coords(self.barra, 0, 0, p, 20))
            time.sleep(delay)

        # Grava
        audio = sd.rec(N_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()

        # Prevê
        self.root.after(0, lambda: self.lbl_status.config(
            text="⚙️  Analisando...", fg="#f39c12"
        ))
        time.sleep(0.1)

        try:
            comando, confianca = self.modelo.prever(audio)
            self.root.after(0, lambda: self._exibir_resultado(comando, confianca))
        except Exception as e:
            self.root.after(0, lambda: self.lbl_status.config(
                text=f"Erro: {e}", fg="#e74c3c"
            ))

        # Zera barra e reativa botão
        self.root.after(0, lambda: self.canvas_prog.coords(self.barra, 0, 0, 0, 20))
        self.root.after(0, lambda: self.btn_gravar.config(state="normal", bg="#e94560"))
        self.gravando = False

    def _exibir_resultado(self, comando: str, confianca: float):
        self._atualizar_personagem(comando, confianca)

        # Status
        cor_status = "#27ae60" if confianca >= 0.7 else "#f39c12"
        self.lbl_status.config(
            text=f"Reconhecido: '{comando.upper()}' com {confianca*100:.1f}% de confiança",
            fg=cor_status,
        )

        # Histórico
        timestamp = time.strftime("%H:%M:%S")
        emoji     = EMOJIS.get(comando, "")
        nivel     = "✔" if confianca >= 0.7 else "⚠"
        entrada   = f"{timestamp} {nivel} {emoji} {comando:<10} ({confianca*100:.0f}%)\n"
        self.historico.append(entrada)

        self.txt_historico.config(state="normal")
        self.txt_historico.insert("end", entrada)
        self.txt_historico.see("end")
        self.txt_historico.config(state="disabled")

        # Volta ao idle após 2 segundos
        self._voltar_idle(2.0)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    app  = JogoVoz(root)
    root.mainloop()


if __name__ == "__main__":
    main()
