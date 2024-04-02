const int num_of_leds = 7;
char control_sig = '0';

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  for(int i = 2; i <= 8; i++) pinMode(i, OUTPUT);
}

void turnOffAll() {
  for(int i = 2; i <= 8; i++) digitalWrite(i, LOW);
}

void direct_Up() {
  for(int i = 2; i <= 8; i++) {
    digitalWrite(i, HIGH);
    delay(20);
  }
}

void direct_Down() {
  for(int i = 8; i >= 2; i--) {
    digitalWrite(i, HIGH);
    delay(20);
  }
}

void loop() {
  if(Serial.available()) {
    char tmp = Serial.read();
     control_sig = (tmp != 12 && tmp != 10)? tmp : control_sig;
  }

  Serial.println(control_sig);
  switch(control_sig) {
    case '0' :
      Serial.println("turnOff");
      turnOffAll();
      break;
    case '1' :
      Serial.println("turnUp");
      direct_Up();
      break;
    case '2' :
      Serial.println("turnDown");
      direct_Down();
      break;
  }
  delay(500);
}


