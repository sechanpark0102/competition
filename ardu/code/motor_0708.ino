const int potentiometerPin = A0;  // 가변저항 연결 핀
const int motor_f_1 = 12;
const int motor_f_2 = 13;
int motor_b_l_1 = 8;
int motor_b_l_2 = 9;
int motor_b_r_1 = 10;
int motor_b_r_2 = 11;

// 속도
int Deceleration_threshold = 20;
int speed = 50;
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

void loop() {
  if (Serial.available()) {
    String receivedString = Serial.readStringUntil('\n');
    float des_pos = receivedString.toFloat();
    Serial.println(receivedString);  // 수신된 값을 다시 전송
    
    float curr_pos = analogRead(potentiometerPin);

    if (abs(curr_pos - des_pos) > Deceleration_threshold) {
      if (curr_pos > des_pos) {
        analogWrite(motor_f_1, 0);
        analogWrite(motor_f_2, speed);
      } else {
        analogWrite(motor_f_1, speed);
        analogWrite(motor_f_2, 0);
      }
    } else {
      analogWrite(motor_f_1, 0);
      analogWrite(motor_f_2, 0);
    }

    // 바퀴 모터 제어 (상수로 제어 중)
    analogWrite(motor_b_l_1, 0);
    analogWrite(motor_b_l_2, 70);
    analogWrite(motor_b_r_1, 0);
    analogWrite(motor_b_r_2, 70);
  }
}
