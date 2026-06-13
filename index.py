import cv2
import numpy as np
video = cv2.VideoCapture('./camera.mp4')

if not video.isOpened():
    print("ERRO: não foi possível abrir o vídeo. Verifique o nome do arquivo.")
    exit()

subtrator = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)

LINHA_Y      = 240  
MIN_AREA     = 1200  
MAX_DIST     = 80    
MAX_AUSENCIA = 5     

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
    mascara = subtrator.apply(frame)
    _, mascara = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.erode(mascara, kernel, iterations=1)
    mascara = cv2.dilate(mascara, kernel, iterations=3)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centros_atuais = []

    for cnt in contornos:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        centros_atuais.append((cx, cy, x, y, w, h))

    for cid in carros.keys():
        carros[cid]['ausente'] += 1

    for (cx, cy, x, y, w, h) in centros_atuais:
        melhor_id = None
        menor_dist = MAX_DIST
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
            carro = carros[melhor_id]
            carro['centro'] = (cx, cy)
            carro['ausente'] = 0 
            if not carro['contado']:
                if carro['lado_anterior'] == 'acima' and lado_atual == 'abaixo':
                    contador += 1
                    carro['contado'] = True  
                    print(f"Carro #{contador} (ID {melhor_id}) passou!")

            carro['lado_anterior'] = lado_atual

        cv2.rectangle(frame, (x, y), (x + w, y + h), COR_CARRO, 2)
        cv2.putText(frame, f"ID: {melhor_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_CARRO, 2)
        cv2.circle(frame, (cx, cy), 5, COR_PONTO, -1)

    carros = {cid: dados for cid, dados in carros.items() if dados['ausente'] < MAX_AUSENCIA}
    cv2.line(frame, (0, LINHA_Y), (800, LINHA_Y), COR_LINHA, 2)
    cv2.putText(frame, "LINHA DE CONTAGEM", (10, LINHA_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_LINHA, 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (320, 68), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"Carros: {contador}", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.3, COR_TEXTO, 2)

    cv2.imshow("Contador de Carros (Melhorado)", frame)
    cv2.imshow("Mascara (Visao do Computador)", cv2.resize(mascara, (400, 300)))

    if cv2.waitKey(30) == 27: 
        break

video.release()
cv2.destroyAllWindows()
print(f"\n[SUCESSO] Total final: {contador} carros contados.")
