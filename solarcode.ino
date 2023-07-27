#include <solar_grp4_inferencing.h>
#include <Servo.h>

// Servo motor setup
Servo servoMotor;
const int servoPin = 9;

// LDR pins setup
const int eastLDRPin = A0;
const int westLDRPin = A1;

// Variables to store LDR values
int eastLDRValue = 0;
int westLDRValue = 0;

// Buffer for the features
static float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

// Variables to store the current and predicted angles
int currentAngle = 0;
int predictedAngle = 0;

/**
 * @brief      Copy raw feature data in out_ptr
 *             Function called by inference library
 *
 * @param[in]  offset   The offset
 * @param[in]  length   The length
 * @param      out_ptr  The out pointer
 *
 * @return     0
 */
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, &features[offset], length * sizeof(float));
    return 0;
}

void print_inference_result(ei_impulse_result_t result);

// Function to get analog values from LDRs
void get_analog_values() {
    // Read LDR values
    eastLDRValue = analogRead(eastLDRPin);
    westLDRValue = analogRead(westLDRPin);

    // Update features array with LDR values
    features[0] = static_cast<float>(eastLDRValue);
    features[1] = static_cast<float>(westLDRValue);

    // Print LDR values on Serial Monitor
    Serial.print("East LDR Value: ");
    Serial.print(eastLDRValue);
    Serial.print("\tWest LDR Value: ");
    Serial.println(westLDRValue);
}

/**
 * @brief      Arduino setup function
 */
void setup()
{
    // put your setup code here, to run once:
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo");

    // Servo motor setup
    servoMotor.attach(servoPin);
    // Set initial servo motor angle to 0 degrees
    currentAngle = 0;
    servoMotor.write(currentAngle);

    // LDR setup
    pinMode(eastLDRPin, INPUT);
    pinMode(westLDRPin, INPUT);
}

/**
 * @brief      Arduino main function
 */
void loop()
{
    ei_printf("Edge Impulse standalone inferencing (Arduino)\n");

    // Get analog values from LDRs
    get_analog_values();

    ei_impulse_result_t result = { 0 };

    // the features are stored into flash, and we don't want to load everything into RAM
    signal_t features_signal;
    features_signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    features_signal.get_data = &raw_feature_get_data;

    // invoke the impulse
    EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false /* debug */);
    if (res != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", res);
        return;
    }

    // print inference return code
    ei_printf("run_classifier returned: %d\r\n", res);
    print_inference_result(result);

    // Find the predicted class with the maximum value
    int maxClassIndex = 0;
    float maxValue = result.classification[0].value;
    for (int i = 1; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > maxValue) {
            maxValue = result.classification[i].value;
            maxClassIndex = i;
        }
    }

    // Calculate the predicted angle
    predictedAngle = map(maxClassIndex, 0, EI_CLASSIFIER_LABEL_COUNT - 1, 0, 180);

    // Move the servo motor towards the predicted angle in steps of 10 degrees
    while (currentAngle != predictedAngle) {
        if (currentAngle < predictedAngle) {
            currentAngle = min(currentAngle + 10, predictedAngle);
        } else {
            currentAngle = max(currentAngle - 10, predictedAngle);
        }
        servoMotor.write(currentAngle);
        delay(100);
    }

    // Print the servo motor angle after being moved
    Serial.print("Servo Motor Angle: ");
    Serial.println(currentAngle);

    delay(1000);
}

void print_inference_result(ei_impulse_result_t result) {
    // Print how long it took to perform inference
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
              result.timing.dsp,
              result.timing.classification,
              result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                  bb.label,
                  bb.value,
                  bb.x,
                  bb.y,
                  bb.width,
                  bb.height);
    }

    // Print the prediction results (classification)
#else
    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
        ei_printf("%.5f\r\n", result.classification[i].value);
    }
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif
}