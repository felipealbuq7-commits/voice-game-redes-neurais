# 🎮 Voice Game – Reconhecimento de Comandos de Voz com Redes Neurais

Projeto acadêmico que utiliza **Librosa** para processamento de áudio e **Keras/TensorFlow**
para treinar uma rede neural capaz de identificar comandos de voz e executar ações em um personagem de jogo.

---

## 📁 Estrutura do Projeto

```
voice_game/
├── 0_gerar_dataset_teste.py   # Gera áudios sintéticos para testar o pipeline
├── 1_gravar_audios.py         # Grava seus próprios comandos de voz
├── 2_processar_dados.py       # Extrai MFCCs com Librosa
├── 3_treinar_modelo.py        # Treina a rede neural (Keras/TensorFlow)
├── 4_jogo_demo.py             # Demo interativo do jogo
├── 5_analise_resultados.py    # Análise de acertos/erros para apresentação
├── requirements.txt
└── data/
    ├── raw_audio/             # Áudios .wav por classe
    └── processed/             # Arrays .npy prontos para treino
└── models/                    # Modelo salvo + gráficos
```

---

## 🚀 Como executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Coletar dados
**Opção A – Gravar sua própria voz (recomendado):**
```bash
python 1_gravar_audios.py
```
Grave 10 amostras de cada comando: `pular`, `atacar`, `correr`, `parar`, `agachar`.

**Opção B – Dataset sintético (só para testar o pipeline):**
```bash
python 0_gerar_dataset_teste.py
```

### 3. Processar os áudios com Librosa
```bash
python 2_processar_dados.py
```
Extrai os **MFCCs** e salva os arrays em `data/processed/`.

### 4. Treinar a rede neural
```bash
python 3_treinar_modelo.py
```
Treina, avalia e salva o modelo em `models/voice_model.h5`.

### 5. Jogar!
```bash
python 4_jogo_demo.py
```
Clique em **🎤 FALAR COMANDO** (ou aperte **ESPAÇO**) e fale o comando.

### 6. Análise para apresentação
```bash
python 5_analise_resultados.py
```

---

## 🧠 Como funciona a IA?

```
Áudio (.wav)
    ↓
Librosa: extrai MFCCs (40 coeficientes que resumem o som)
    ↓
Rede Neural (Dense 256 → 128 → 64 → softmax)
    ↓
Probabilidade para cada comando
    ↓
Personagem executa a ação!
```

### O que são MFCCs?
Mel-Frequency Cepstral Coefficients — uma representação compacta
das frequências do áudio que imita como o ouvido humano percebe o som.
São amplamente usados em reconhecimento de fala.

---

## 📊 Arquitetura da Rede Neural

| Camada         | Neurônios | Ativação | Regularização      |
|----------------|-----------|----------|--------------------|
| Dense 1        | 256       | ReLU     | BatchNorm + Drop40%|
| Dense 2        | 128       | ReLU     | BatchNorm + Drop30%|
| Dense 3        | 64        | ReLU     | BatchNorm + Drop20%|
| Saída          | N classes | Softmax  | —                  |

**Otimizador:** Adam (lr=1e-3)  
**Loss:** Categorical Crossentropy  
**Callbacks:** EarlyStopping + ReduceLROnPlateau

---

## 🎮 Comandos do Jogo

| Comando  | Emoji | Ação do personagem |
|----------|-------|--------------------|
| pular    | ⬆️    | Personagem pula    |
| atacar   | ⚔️    | Personagem ataca   |
| correr   | 🏃    | Personagem corre   |
| parar    | ✋    | Personagem para    |
| agachar  | ⬇️    | Personagem agacha  |

---

## ⚠️ Limitações e melhorias

**Causas de erros:**
- Poucas amostras de treino (10 por comando)
- Sons parecidos entre palavras ("parar" × "atacar")
- Ruído ambiental não visto no treino
- Variação entre diferentes falantes

**Como melhorar:**
- Gravar 50+ amostras com pessoas diferentes
- Data Augmentation (pitch shift, ruído sintético)
- Usar CNN 2D sobre o espectrograma
- Testar LSTM para capturar padrões temporais
