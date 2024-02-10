#DETECÇÃO FACIAL

import cv2

# Função para converter o objeto JavaScript em uma imagem OpenCV
def js_para_imagem(js_reply):
    """
    Params:
            js_reply: Objeto JavaScript contendo a imagem da webcam
    Returns:
            img: Imagem OpenCV no formato BGR
    """
    # Decodifica a imagem base64
    imagem_bytes = js_reply.split(',')[1].encode()
    # Converte os bytes em um array numpy
    np_bytes = np.frombuffer(imagem_bytes, dtype=np.uint8)
    # Decodifica o array numpy em uma imagem OpenCV no formato BGR
    img = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)

    return img

# Função para converter a caixa delimitadora OpenCV em uma string de bytes base64
def caixa_delimitadora_para_bytes(caixa_array):
    """
    Params:
            caixa_array: Array numpy (pixels) contendo a caixa delimitadora a ser sobreposta no fluxo de vídeo.
    Returns:
            bytes: String de bytes da imagem base64
    """
    # Converte o array em uma imagem PIL
    caixa_PIL = PIL.Image.fromarray(caixa_array, 'RGBA')
    buffer_io = io.BytesIO()
    # Formata a caixa delimitadora em PNG para retorno
    caixa_PIL.save(buffer_io, format='PNG')
    # Formata a string de retorno
    caixa_bytes = 'data:image/png;base64,{}'.format((str(base64.b64encode(buffer_io.getvalue()), 'utf-8')))

    return caixa_bytes

# Inicializa o classificador Haar Cascade para detecção facial
classificador_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Função para capturar a foto
def tirar_foto(nome_arquivo='foto.jpg', qualidade=0.8):
    # Criação da janela de captura de imagem
    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("Não foi possível abrir a câmera.")
        return

    # Captura da foto
    ret, frame = captura.read()
    if not ret:
        print("Não foi possível capturar a imagem.")
        captura.release()
        return

    # Salva a imagem capturada
    cv2.imwrite(nome_arquivo, frame)

    # Fecha a janela de captura de imagem
    captura.release()

    return nome_arquivo

# Captura da foto e detecção de rostos
try:
    nome_arquivo = tirar_foto('foto.jpg')
    print('Foto salva em {}'.format(nome_arquivo))

    # Carrega a imagem capturada
    img = cv2.imread(nome_arquivo)

    # Converte a imagem para escala de cinza
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecta os rostos na imagem usando o classificador Haar Cascade
    faces = classificador_face.detectMultiScale(img_cinza)

    # Desenha os retângulos ao redor dos rostos detectados na imagem
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Exibe a imagem com os rostos detectados
    cv2.imshow('Rostos Detectados', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as erro:
    # Erros serão lançados se o usuário não tiver uma webcam ou se não conceder permissão para acessá-la.
    print(str(erro))
