import sounddevice as sd  # Importa a biblioteca para capturar áudio
import numpy as np  # Importa a biblioteca para operações numéricas
import whisper  # Importa a biblioteca Whisper para transcrição
from scipy.io.wavfile import write  # Importa a função para salvar arquivos WAV
import os  # Importa a biblioteca para operações do sistema

# Parâmetros de gravação
sample_rate = 44100  # Taxa de amostragem (Hz)
duration = 5  # Duração de cada gravação (segundos)
overlap = 1  # Duração da sobreposição (segundos)
arquivo_saida = 'saida_audio_sounddevice.wav'  # Nome do arquivo de saída

# Função para gravar áudio com sobreposição
def gravar_audio_com_sobreposicao():
    audio_data = []  # Lista para armazenar os dados de áudio
    total_duration = duration + overlap  # Duração total da gravação com sobreposição
    for start in np.arange(0, 10, duration - overlap):  # Loop para gravar áudio
        print(f"Gravando de {start:.1f} a {start + duration:.1f} segundos...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)  # Captura áudio
        sd.wait()  # Aguarda a finalização da gravação
        audio_data.append(audio)  # Adiciona o áudio capturado à lista
    return np.concatenate(audio_data)  # Concatena todos os segmentos de áudio

# Carrega o modelo Whisper uma vez
model = whisper.load_model("base")  # Carrega o modelo de transcrição

# Loop contínuo para gravação e transcrição
try:
    while True:  # Mantém o loop até ser interrompido
        print("\nGravando novo áudio...")
        audio_data = gravar_audio_com_sobreposicao()  # Captura áudio com sobreposição
        
        # Salva o áudio em um arquivo WAV temporário
        write(arquivo_saida, sample_rate, audio_data)  # Salva o áudio capturado
        
        # Transcreve o áudio
        print("Transcrevendo o áudio...")
        result = model.transcribe(arquivo_saida)  # Transcreve o áudio usando Whisper

        # Exibe o texto transcrito
        print("Texto transcrito:")
        print(result['text'])  # Imprime o texto resultante

        # Apaga o arquivo de áudio temporário (opcional)
        if os.path.exists(arquivo_saida):  # Verifica se o arquivo existe
            os.remove(arquivo_saida)  # Remove o arquivo após a transcrição

except KeyboardInterrupt:  # Captura a interrupção do teclado
    print("Loop de gravação e transcrição interrompido.")  # Mensagem de finalização
