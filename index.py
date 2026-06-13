import cv2
import numpy as np

# -------------------------------------------------------
# contadorCarros_Melhorado.py
# -------------------------------------------------------

video = cv2.VideoCapture('./camera.mp4')

if not video.isOpened():
    print("ERRO: não foi possível abrir o vídeo. Verifique o nome do arquivo.")
    exit()

subtrator = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)

# ── Configurações — AJUSTE CONFORME SEU VÍDEO ──────────────
LINHA_Y      = 240   # posição Y da linha de contagem
MIN_AREA     = 1200  # área mínima para ser considerado carro
MAX_DIST     = 80    # distância máxima (px) para associar ao mesmo carro
MAX_AUSENCIA = 5     # quantos frames o sistema "lembra" de um carro que sumiu da tela
# ───────────────────────────────────────────────────────────

contador = 0
proximo_id = 0
carros = {}

COR_LINHA = (0, 255, 255)
COR_CARRO = (0, 200, 0)
COR_TEXTO = (255, 255, 255)
COR_PONTO = (255, 0, 255)

while True:
    ret, frame = video.read()
    if not ret:
        print("Fim do vídeo.")
        break

    frame = cv2.resize(frame, (800, 600))
    
    # 1. Subtração e Morfologia
    mascara = subtrator.apply(frame)
    _, mascara = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.erode(mascara, kernel, iterations=1)
    mascara = cv2.dilate(mascara, kernel, iterations=3)

    # 2. Encontrar os blocos em movimento
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centros_atuais = []

    for cnt in contornos:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        centros_atuais.append((cx, cy, x, y, w, h))

    # 3. Rastreamento (Tracking)
    # Aumenta o tempo de ausência de todos os carros conhecidos
    for cid in carros.keys():
        carros[cid]['ausente'] += 1

    for (cx, cy, x, y, w, h) in centros_atuais:
        melhor_id = None
        menor_dist = MAX_DIST

        # Procura qual carro antigo está mais perto dessa nova posição
        for cid, dados in carros.items():
            px, py = dados['centro']
            dist = np.hypot(cx - px, cy - py)
            
            if dist < menor_dist:
                menor_dist = dist
                melhor_id = cid

        lado_atual = "acima" if cy < LINHA_Y else "abaixo"

        if melhor_id is None:
            # É um carro novo que apareceu
            melhor_id = proximo_id
            proximo_id += 1
            carros[melhor_id] = {
                'centro': (cx, cy),
                'ausente': 0,
                'lado_anterior': lado_atual,
                'contado': False
            }
        else:
            # É um carro que já conhecemos -> atualiza os dados
            carro = carros[melhor_id]
            carro['centro'] = (cx, cy)
            carro['ausente'] = 0  # Ele apareceu, então zeramos a ausência

            # Lógica de contagem
            if not carro['contado']:
                if carro['lado_anterior'] == 'acima' and lado_atual == 'abaixo':
                    contador += 1
                    carro['contado'] = True  # Trava para não contar de novo
                    print(f"Carro #{contador} (ID {melhor_id}) passou!")

            carro['lado_anterior'] = lado_atual

        # Desenha na tela (Bounding box e ID)
        cv2.rectangle(frame, (x, y), (x + w, y + h), COR_CARRO, 2)
        cv2.putText(frame, f"ID: {melhor_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_CARRO, 2)
        cv2.circle(frame, (cx, cy), 5, COR_PONTO, -1)

    # 4. Limpeza da Memória
    # Se um carro ficou ausente por mais de MAX_AUSENCIA frames, apagamos ele
    carros = {cid: dados for cid, dados in carros.items() if dados['ausente'] < MAX_AUSENCIA}

    # 5. Interface Visual
    cv2.line(frame, (0, LINHA_Y), (800, LINHA_Y), COR_LINHA, 2)
    cv2.putText(frame, "LINHA DE CONTAGEM", (10, LINHA_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_LINHA, 1)

    # Fundo semi-transparente para o texto do contador
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (320, 68), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"Carros: {contador}", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.3, COR_TEXTO, 2)

    cv2.imshow("Contador de Carros (Melhorado)", frame)
    cv2.imshow("Mascara (Visao do Computador)", cv2.resize(mascara, (400, 300)))

    if cv2.waitKey(30) == 27:  # ESC para sair
        break

video.release()
cv2.destroyAllWindows()
print(f"\n[SUCESSO] Total final: {contador} carros contados.")