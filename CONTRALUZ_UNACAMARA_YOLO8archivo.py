import cv2
import serial
import os
import time
import csv
import psutil
from gpiozero import LED, OutputDevice
from sympy import false
from ultralytics import YOLO

# Inicialización del puerto serial
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
comando = "e".encode('utf-8')
comando2 = "o".encode('utf-8')

duracion = 2
bandera = False
inicio = time.time()
contador = 0

# Configuración del LED y relé
ledrojo = LED(27)
relay = OutputDevice(17)

# Estado inicial
relay.on()
ledrojo.off()
#arduino.write(comando2)

# Variables para captura
cpt = 10
maxFrames = 1
error_detectado = False

# Cargar modelo
model = YOLO("best_ncnn_model")
# model = YOLO("modelo_completo_openvino_model")

# Captura de video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FPS, 50)

# Crear carpeta de errores
error_folder = "/mnt/central/BANCO_IMAGENES/ERRORES_DESPINCE"
if not os.path.exists(error_folder):
    os.makedirs(error_folder)

error_data ="/mnt/central/BANCO_IMAGENES/ERRORES_DESPINCE/data.txt"
if not os.path.exists(error_data):
    with open(error_data,"w") as f:
        pass

# Crear carpeta para guardar CSV si no existe
csv_folder = "/home/dislana/Documentos/"
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)


# === Funciones de monitoreo ===
def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

def get_temperature():
    try:
        temp = os.popen("vcgencmd measure_temp").readline()
        return temp.replace("temp=", "").replace("'C\n", "")
    except:
        return "N/A"

# === Variables de medición de rendimiento ===
frame_count = 0
start_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se puede leer el fotograma. Reintentando...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(0)
            continue

        # Preprocesamiento
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # === Medir tiempo de inferencia ===
        infer_start = time.time()
        results = model.predict(source=gray_3ch)
        infer_end = time.time()

        inference_time = (infer_end - infer_start) * 1000  # ms
        fps = 1.0 / (infer_end - infer_start)

        # === Monitorear CPU y temperatura ===
        cpu_usage = get_cpu_usage()
        temperature = get_temperature()

        # Anotar resultados en imagen
        annotated = results[0].plot()

        # Mostrar mediciones en pantalla
        cv2.putText(annotated, f"Inference: {inference_time:.2f} ms", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated, f"CPU: {cpu_usage:.1f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated, f"Temp: {temperature} C", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Reset estado de relé y led
        ledrojo.off()
        relay.on()

        #arduino.write(comando2)

        error_actual = False
        # Analizar detecciones
        for result in results:
            for box in result.boxes:
                cls = box.cls.item()
                conf = box.conf.item()
                label = model.names[int(cls)]

                print(f"Etiqueta detectada: {label}, Confianza: {conf:.2f}")

                if label in ["error", "error1","error2"] and conf > 0.30:

                    error_actual = True
                    relay.off()
                    ledrojo.on()
                    arduino.write(comando)
                    cadena = arduino.readline().decode().strip()
                    #print(f"esp32: {cadena}")
                    if cadena and bandera == False:
                        print(f"encoder: {cadena}")
                        with open(error_data, 'a') as file:
                            try:
                                numero = float(cadena)
                                file.write(f"{label},{cadena}\n")
                            except ValueError:
                                pass
                        bandera = True
                    break

        if not error_actual:
            bandera = False

        # Mostrar imagen
        cv2.imshow("Video", annotated)
        #cv2.imshow("Video",gray_3ch )

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Salida por tecla 'q'")
            break

finally:
    print("Cerrando programa. Apagando relé y liberando recursos.")
    relay.off()
    ledrojo.off()
    arduino.close()
    relay.close()
    ledrojo.close()
    cap.release()
    cv2.destroyAllWindows()

    # Mostrar estadísticas finales
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Frames capturados: {frame_count}")
    print(f"Tiempo total: {elapsed_time:.2f} segundos")
    print(f"FPS promedio: {frame_count / elapsed_time:.2f}")

