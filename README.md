# rPPG — batimentos cardíacos pela webcam

Fotopletismografia remota (rPPG): o volume de sangue na pele varia a cada
batimento e muda sutilmente a cor do rosto. Este projeto captura o rosto pela
webcam, extrai a média RGB de ROIs de pele (testa e bochechas) e estima o BPM
pelo espectro do sinal. Feito para estudo: cada etapa é uma função pequena e
trocável.

A saída sai enquadrada em **9:16** (pronta para vídeo vertical), com um HUD que
mostra o BPM, a qualidade do sinal, a onda do pulso e um **coração que bate em
sincronia com o pulso detectado** — a fase vem de um ajuste senoidal na
frequência estimada, com um PLL simples para a animação não tremer.

## Instalação

Requer Python 3.10–3.12 (MediaPipe não suporta 3.13 ainda).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(ou, com [uv](https://docs.astral.sh/uv/): `uv venv --python 3.11 && uv pip install -r requirements.txt`)

O modelo `face_landmarker.task` do MediaPipe é baixado automaticamente na
primeira execução (fica em `models/`).

## Uso

```bash
python main.py                          # webcam ao vivo, 9:16
python main.py --method chrom           # método inicial (green|chrom|pos)
python main.py --rois forehead          # só a testa
python main.py --video sujeito1.avi     # arquivo de vídeo (ex.: UBFC-rPPG)
python main.py --plot                   # abre também os plots matplotlib
```

Teclas na janela de vídeo: `1`/`2`/`3` trocam o método (GREEN/CHROM/POS),
`r` começa/para a gravação do sinal (salva `rec_*.npz`), `q` sai.

### Enquadramento e exportação de vídeo

O recorte 9:16 acompanha o rosto com suavização, então dá para gravar direto:

```bash
# clipe de 30 s em 1080x1920, com o HUD, sem precisar de tela
python main.py --save clipe.mp4 --duration 30 --output-size 1080 1920 --no-window

python main.py --zoom 1.3                # aproxima o recorte no rosto
python main.py --no-portrait             # quadro cheio, sem recorte
python main.py --no-follow               # recorte fixo no centro
```

Dica: filme em `--output-size 1080 1920`; a câmera é aberta em 1280x720, então o
recorte 9:16 sai com 405 px de largura e é ampliado. Se a sua webcam suportar
1080p, ajuste `capture_size` no `config.py` para `(1920, 1080)` e a saída fica
bem mais nítida.

### Análise offline

```bash
python analyze.py rec_20260109_120000.npz
python analyze.py rec.npz --ref oximetro.csv   # CSV: timestamp,bpm -> MAE/RMSE
```

Roda os 3 métodos sobre a gravação e plota sinal, espectro e BPM ao longo do
tempo lado a lado.

### Testes

```bash
python -m pytest tests/ -v
```

## Estrutura

| arquivo | o que faz |
|---|---|
| `rppg/capture.py` | webcam/vídeo, timestamps reais, tenta desligar auto-exposição/WB |
| `rppg/roi.py` | Face Landmarker (MediaPipe Tasks), máscaras poligonais de testa e bochechas |
| `rppg/methods.py` | GREEN, CHROM (de Haan 2013), POS (Wang 2017) — mesma assinatura, fácil de trocar |
| `rppg/signal_processing.py` | reamostragem uniforme, detrend, Butterworth 0.7–4 Hz, FFT + SNR, fase do pulso |
| `rppg/ui.py` | enquadramento 9:16, coração batendo, HUD desenhado no vídeo |
| `rppg/viz.py` | plots matplotlib ao vivo (sinal, pulso, espectro) |
| `config.py` | todos os parâmetros experimentáveis num lugar só |

## Dicas de captura

- **Luz natural difusa** (janela) é o ideal. Evite LED/fluorescente que pisca
  em 100/120 Hz — vira ruído de aliasing no sinal.
- **Fique parado** e com expressão neutra; movimento é a maior fonte de erro.
- Rosto bem iluminado, sem sombras duras, a ~0.5–1 m da câmera.
- Espere ~10 s para a janela de análise encher. A qualidade aparece no SNR e na
  cor da barra (vermelho / âmbar / verde) e da onda — valores de rPPG são
  modestos por natureza, então SNR baixo é comum e não interrompe nada na tela.
- Se a câmera permitir, desligue exposição/white balance automáticos (o script
  tenta e loga o resultado; no macOS geralmente o driver não deixa).

## Limitações

- **Movimento e fala** degradam muito o sinal (POS é o mais robusto).
- **Iluminação instável** (nuvens, telas, LEDs) contamina a banda cardíaca.
- **Tom de pele**: a literatura reporta erro maior em peles mais escuras com
  câmeras RGB comuns (menos luz refletida, menor amplitude do pulso).
- Maquiagem, barba densa e óculos reduzem a área útil de pele.
- **Não é um dispositivo médico.** É um experimento didático; não use para
  qualquer decisão de saúde.

## Referências

- Verkruysse, Svaasand, Nelson. *Remote plethysmographic imaging using ambient light.* Optics Express, 2008. (GREEN)
- de Haan, Jeanne. *Robust pulse rate from chrominance-based rPPG.* IEEE TBME, 2013. (CHROM)
- Wang, den Brinker, Stuijk, de Haan. *Algorithmic principles of remote PPG.* IEEE TBME, 2017. (POS)
