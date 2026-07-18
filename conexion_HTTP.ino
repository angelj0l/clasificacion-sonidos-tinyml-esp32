#include <Arduino.h>
#include "driver/i2s.h"
#include <TensorFlowLite_ESP32.h>

// TENSORFLOW LITE MICRO
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"

// MICROFRONTEND
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"

// MODELO ENTRENADO
#include "model_data.h"

//Conectividad
#include <WiFi.h>
#include <HTTPClient.h>

// ==========================================
// CONFIGURACIÓN DE RED Y SERVIDOR
// ==========================================
const char* ssid = "*****";
const char* password = "******";

// URL de tu backend 
const char* serverName = "*************"; //servidor 

// ==========================================
// CONFIGURACIÓN DE AUDIO E I2S
// ==========================================
#define SAMPLE_RATE 16000
#define DURATION_SEC 3
#define TOTAL_SAMPLES (SAMPLE_RATE * DURATION_SEC)

#define I2S_WS 25
#define I2S_SD 33
#define I2S_SCK 26

// ==========================================
// CONFIGURACIÓN DEL ESPECTROGRAMA E IA
// ==========================================
#define N_MELS 26   
#define N_FRAMES 301

// PARÁMETROS DE CUANTIZACIÓN 
const float input_scale = 0.0036601307801902294f; 
const int input_zero_point = -128;   

const char* CLASSES[] = {"Musica", "Trafico", "Neutro"};

// ==========================================
// OBJETOS GLOBALES Y BUFFERS EN PSRAM
// ==========================================
int16_t* audio_buffer = NULL;
float (*mel)[N_FRAMES] = NULL;
FrontendState frontend;

// Punteros de TensorFlow Lite
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor Arena (Memoria RAM reservada para la red neuronal)
constexpr int kTensorArenaSize = 150 * 1024; // 150 KB
uint8_t* tensor_arena = NULL; //  ES UN PUNTERO 

// ==========================================
// INICIALIZACIÓN DE I2S
// ==========================================
void initI2S() {
    Serial.print("Iniciando bus I2S... ");
    i2s_config_t config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S_MSB,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 512,
        .use_apll = false
    };

    i2s_pin_config_t pins = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };

    if (i2s_driver_install(I2S_NUM_0, &config, 0, NULL) != ESP_OK) {
        Serial.println("❌ ERROR");
        while(1);
    }
    i2s_set_pin(I2S_NUM_0, &pins);
    i2s_zero_dma_buffer(I2S_NUM_0);
    Serial.println("✅ OK");
}

// ==========================================
// INICIALIZACIÓN DEL FRONTEND DE AUDIO
// ==========================================
void initFrontend() {
    Serial.print("Iniciando motor Log-Mel... ");
    FrontendConfig config;
    config.window.size_ms = 30;
    config.window.step_size_ms = 10;
    config.filterbank.num_channels = N_MELS;
    config.noise_reduction.smoothing_bits = 10;
    config.pcan_gain_control.enable_pcan = false;
    config.log_scale.enable_log = true;
    config.filterbank.lower_band_limit = 125.0;
    config.filterbank.upper_band_limit = 7500.0;

    if (FrontendPopulateState(&config, &frontend, SAMPLE_RATE) == 0) {
        Serial.println("❌ ERROR");
        while (1);
    }
    Serial.println("✅ OK");
}

// ==========================================
// INICIALIZACIÓN DE TENSORFLOW LITE
// ==========================================
void initTinyML() {
    Serial.print("Cargando modelo TFLite... ");
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    tflite_model = tflite::GetModel(model_tflite);
    /*if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("❌ ERROR: Esquema incompatible");
        while(1);
    }*/

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("❌ ERROR: Memoria insuficiente en Tensor Arena");
        while(1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.println("✅ OK");
}

// ==========================================
// CAPTURAR AUDIO
// ==========================================
void recordAudio() {
    int32_t raw;
    size_t bytes_read;
    int count = 0;

    while (count < TOTAL_SAMPLES) {
        i2s_read(I2S_NUM_0, &raw, sizeof(raw), &bytes_read, portMAX_DELAY);
        if (bytes_read > 0) {
            audio_buffer[count++] = (int16_t)(raw >> 14); 
        }
    }
}

// ==========================================
// GENERAR LOG-MEL SPECTROGRAM
// ==========================================
void computeLogMel() {
    int frame = 0;
    int16_t* ptr = audio_buffer;
    int remaining = TOTAL_SAMPLES;

    while (remaining > 0 && frame < N_FRAMES) {
        size_t read_samples;
        FrontendOutput out = FrontendProcessSamples(&frontend, ptr, remaining, &read_samples);

        if (out.values != nullptr) {
            for (int i = 0; i < N_MELS; i++) {
                mel[i][frame] = (float)out.values[i];
            }
            frame++;
        }
        ptr += read_samples;
        remaining -= read_samples;
    }

    // Rellenar con ceros si el audio terminó antes de los 301 frames
    for (int f = frame; f < N_FRAMES; f++) {
        for (int i = 0; i < N_MELS; i++) mel[i][f] = 0.0f;
    }
}

// ==========================================
// CONEXION WIFI
// ==========================================
void connectWiFi() {
    Serial.print("Conectando a Wi-Fi");
    WiFi.begin(ssid, password);
    
    // Intento de conexión con timeout para no bloquear el sistema eternamente
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n✅ Wi-Fi Conectado");
        Serial.print("Dirección IP: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\n❌ ERROR: No se pudo conectar al Wi-Fi.");
    }
}

// ==========================================
// ENVIO DE DATOS
// ==========================================
void sendDataToCloud(String prediction, float p_musica, float p_trafico, float p_neutro) {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(serverName);
        http.addHeader("Content-Type", "application/json");

        // Construimos el JSON con los 4 datos
        String jsonPayload = "{\"clase\":\"" + prediction + "\", " +
                             "\"prob_musica\":" + String(p_musica) + ", " +
                             "\"prob_trafico\":" + String(p_trafico) + ", " +
                             "\"prob_neutro\":" + String(p_neutro) + "}";
        
        int httpResponseCode = http.POST(jsonPayload);
        if (httpResponseCode > 0) {
            Serial.println("📡 Datos enviados correctamente.");
        } else {
            Serial.println("❌ ERROR en envío Wi-Fi.");
        }
        http.end();
    }
}

// ==========================================
// SETUP PRINCIPAL
// ==========================================
void setup() {
    Serial.begin(115200);
    connectWiFi();
    delay(2000);
    Serial.println("\n=================================");
    Serial.println("SISTEMA DE CLASIFICACIÓN ACÚSTICA");
    Serial.println("=================================");

    // Inicializar PSRAM del WROVER
    Serial.print("Comprobando PSRAM... ");
    if (!psramInit()) { 
        Serial.println("❌ ERROR");
        while (1); 
    }
    Serial.println("✅ OK");

    // Asignar memoria dinámica a los buffers
    Serial.print("Asignando buffers en PSRAM... ");
    audio_buffer = (int16_t*) ps_malloc(TOTAL_SAMPLES * sizeof(int16_t));
    mel = (float (*)[N_FRAMES]) ps_malloc(N_MELS * N_FRAMES * sizeof(float));
    
    // Mandamos el modelo a la PSRAM
    tensor_arena = (uint8_t*) ps_malloc(kTensorArenaSize);
    
    if (audio_buffer == NULL || mel == NULL || tensor_arena == NULL) {
        Serial.println("❌ ERROR: Sin memoria PSRAM");
        while(1);
    }
    Serial.println("✅ OK");

    // Iniciar subsistemas
    initI2S();
    initFrontend();
    initTinyML();

    Serial.println("\n🎯 SISTEMA INICIADO - COMENZANDO INFERENCIA...");
}

// ==========================================
// LOOP PRINCIPAL (CICLO DE INFERENCIA)
// ==========================================
void loop() {
    unsigned long time_start_total = millis();

    // 1. CAPTURA DE AUDIO
    Serial.print("\n🎤 Grabando 3 segundos... ");
    unsigned long t1 = millis();
    recordAudio();
    Serial.print("OK ("); Serial.print(millis() - t1); Serial.println(" ms)");

    // 2. PROCESAMIENTO DIGITAL DE SEÑALES (DSP)
    Serial.print("🧠 Generando Log-Mel... ");
    unsigned long t2 = millis();
    computeLogMel();
    Serial.print("OK ("); Serial.print(millis() - t2); Serial.println(" ms)");

    // 3. PIPELINE DE CUANTIZACIÓN
    Serial.print("⚙️  Cuantizando e inyectando tensor... ");
    unsigned long t3 = millis();
    int tensor_index = 0;
    
    // Leemos transponiendo los ejes (Frames en X, Mels en Y)
    for (int f = 0; f < N_FRAMES; f++) {
        for (int m = 0; m < N_MELS; m++) {
            float raw_val = mel[m][f]; 
            
            // EL PIPELINE ESTRICTO: Log-Mel -> /15 -> Cuantización INT8
            float scaled_val = raw_val / 15.0f;
            int32_t quantized_val = round(scaled_val / input_scale) + input_zero_point;
            
            // Clipping a los límites de INT8
            if (quantized_val > 127) quantized_val = 127;
            if (quantized_val < -128) quantized_val = -128;
            
            input->data.int8[tensor_index++] = (int8_t)quantized_val;
        }
    }
    Serial.print("OK ("); Serial.print(millis() - t3); Serial.println(" ms)");

    // 4. INFERENCIA RED NEURONAL
    Serial.print("🚀 Ejecutando Inferencia... ");
    unsigned long t4 = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("❌ ERROR: Falló el modelo");
        return;
    }
    Serial.print("OK ("); Serial.print(millis() - t4); Serial.println(" ms)");

    // =========================================================
    // 5. OBTENCIÓN DE RESULTADOS
    // =========================================================
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    
    int max_index = 0;
    float max_prob = 0.0;
    float sum_exp = 0.0;
    float probabilidades_crudas[3];
    
    Serial.println("\n----------------------------------");
    
    // A. Leer y aplicar factor de temperatura (T = 15.0)
    for (int i = 0; i < 3; i++) {
        int8_t out_val = output->data.int8[i];
        
        // 1. Extraer el logit base des-cuantizado
        float prob_base = (out_val - output_zero_point) * output_scale;
        
        // 2. Aplicar la exponencial con el factor de estiramiento
        probabilidades_crudas[i] = exp(prob_base * 40.0); 
        sum_exp += probabilidades_crudas[i];
    }
    
    // Arreglo para guardar los porcentajes finales antes de enviarlos
    float porcentajes_finales[3];

    // B. Calcular porcentajes finales relativos y encontrar al ganador
    for (int i = 0; i < 3; i++) {
        // 3. Normalizar para que la suma total sea 100%
        float prob_final = probabilidades_crudas[i] / sum_exp; 
        porcentajes_finales[i] = prob_final * 100.0; // Guardamos el porcentaje
        
        Serial.printf("> %s: %.1f%%\n", CLASSES[i], porcentajes_finales[i]); 
        
        // 4. Guardar el índice con mayor probabilidad
        if (prob_final > max_prob) {
            max_prob = prob_final; 
            max_index = i; 
        }
    }
    
    Serial.println("----------------------------------");
    Serial.printf("🎯 RESULTADO: %s \n", CLASSES[max_index]); 


    // >>> ENVIAR TODOS LOS DATOS POR WI-FI <<<
    // Según arreglo CLASSES, el índice 0 es Música, 1 es Tráfico, y 2 es Neutro 
    sendDataToCloud(CLASSES[max_index], porcentajes_finales[0], porcentajes_finales[1], porcentajes_finales[2]);

    Serial.printf("⏱️ TIEMPO TOTAL DE CICLO: %lu ms\n", (millis() - time_start_total)); 
    Serial.println("=================================="); 

    // Pequeño retardo antes de la siguiente captura
    delay(500); 
}
