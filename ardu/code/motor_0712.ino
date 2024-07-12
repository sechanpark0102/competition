const int potentiometerPin = A0;  // 가변저항 연결 핀
const int motor_f_1 = 12;
const int motor_f_2 = 13;
int motor_b_l_1 = 8;
int motor_b_l_2 = 9;
int motor_b_r_1 = 10;
int motor_b_r_2 = 11;

// 속도
int Deceleration_threshold = 20;
int speed = 120;
int speed_offset = 255;

void setup() {
  // 모터 설정
  pinMode(motor_f_1, OUTPUT);
  pinMode(motor_f_2, OUTPUT);
  pinMode(motor_b_l_1, OUTPUT);
  pinMode(motor_b_l_2, OUTPUT);
  pinMode(motor_b_r_1, OUTPUT);
  pinMode(motor_b_r_2, OUTPUT);
  Serial.begin(115200);
}

float dt = 0.01;
float err_prev_angle = 0.0;
float err_prev_dist = 0.0;
float err_integral_angle = 0.0;      // 각도 오차의 누적 초기화
float kp = 2.0;
float kd = 0.5;
float ki = 0.01;
unsigned long last_time = 0;
unsigned long last_data_time = 0; // 마지막 데이터 수신 시간을 저장할 변수
int steer_type = 0;

void loop() {

  
  if (Serial.available()) {
      unsigned long current_time = millis();
  dt = (current_time - last_time) / 1000.0; // 시간 간격을 초 단위로 계산
  last_time = current_time;
    String receivedString = Serial.readStringUntil('\n');
    int commaIndex = receivedString.indexOf(',');
    String line_angle_input_str = receivedString.substring(0, commaIndex);
    String dist_left_str = receivedString.substring(commaIndex + 1);
    String dist_right_str = receivedString.substring(commaIndex + 2);
    float angle = line_angle_input_str.toFloat();
    float dist_l = dist_left_str.toFloat();
    float dist_r = dist_right_str.toFloat();
    float curr_pos = analogRead(potentiometerPin);
    
    double err_angle = curr_pos - angle;
    double delta_err_angle = err_angle - err_prev_angle;
    double steer_angle = err_angle * kp + kd * (delta_err_angle) / dt + ki * err_integral_angle;
    
    Serial.print(steer_angle);
    Serial.print("//"); // 수신된 값을 다시 전송
    Serial.print(curr_pos); Serial.print("//"); Serial.print(dist_l); Serial.print("//"); Serial.println(dist_r);

    if (steer_angle > 0) {
      analogWrite(motor_f_1, 0);
      analogWrite(motor_f_2, steer_angle * 1.7);
    } else {
      analogWrite(motor_f_1, abs(steer_angle));
      analogWrite(motor_f_2, 0);
    }  

    // 바퀴 모터 제어 (상수로 제어 중)
    analogWrite(motor_b_l_1, 0);
    analogWrite(motor_b_l_2, 135);
    analogWrite(motor_b_r_1, 0);
    analogWrite(motor_b_r_2, 135);

    err_prev_angle = err_angle;
    err_integral_angle += err_angle * dt;
    
    // 데이터 수신 시간을 현재 시간으로 업데이트
    last_data_time = current_time;
  }

}

void stopMotors() {
  analogWrite(motor_f_1, 0);
  analogWrite(motor_f_2, 0);
  analogWrite(motor_b_l_1, 0);
  analogWrite(motor_b_l_2, 0);
  analogWrite(motor_b_r_1, 0);
  analogWrite(motor_b_r_2, 0);
}
