void setup() {
  for(int i = 2; i< 10; i++) pinMode(i, OUTPUT);
  pinMode(12, INPUT); // 전적색 + 보행자 -> 1
  pinMode(13, INPUT); // 차량 -> 1
  Serial.begin(9600);
}

/*
    10
    01
    traffic_light_State = ((digitalRead(11) << 1) | (digitalRead(13))) - 1;
    if (traffic_light_state != 0) {
      함수포인터배열[traffic_light_state](int direction);
    }
*/

char res = '0';
int curr_port = 3;
int signal_margins[] = {10, 40};
int signal_margin = 0;
int traffic_light_state = 0; // 현재 신호등 상태
int traffic_light_before = 0; // 이전 신호등 상태


void turnOnCrosswalk(char direction) {
  // Serial.println("Crosswalk ON########");
  int offset = (direction == '2') ? 1 : -1;
  int pin = (offset > 0) ? 2 : 8;
  while(pin >= 2 && pin < 9) {
    digitalWrite(pin, HIGH);
    pin += offset;
    delay(50);
  }
}

void turnOnRoad(char direction) {
  //Serial.println("Road ON########");
  digitalWrite(9, HIGH);
}

void turnOff() {
  for(int i = 2; i < 10; i++) {
    digitalWrite(i, LOW);
  }
}

void clearInputBuffer() {
  while(Serial.available()) {
    Serial.read();
  }
}


void (* LEDOperation[2])(char) = {turnOnCrosswalk, turnOnRoad};

void loop() {
  // put your main code here, to run repeatedly:
  traffic_light_state = ((digitalRead(13) << 1) | (digitalRead(12)));
  // 0 -> 1 -> 2
  if(!traffic_light_state) {
    traffic_light_before = 0;
    turnOff();
  }
  else {
    if(traffic_light_state != traffic_light_before) {
      signal_margin = 0;
      turnOff();
    }

    if(Serial.available()) {
      res = Serial.read();
    } else {
      res = '0';
    }

    if(signal_margin == 0) {
      if(res != '0') {
        // Serial.println(res);
        if(traffic_light_state == 2) {
          turnOnRoad(res);
        } else {
          turnOnCrosswalk(res);
        }
        signal_margin = signal_margins[traffic_light_state - 1];
      } else {
        turnOff();
        signal_margin = 0;
      }
    } 
    else {
        signal_margin--;
    }
    traffic_light_before = traffic_light_state;
  }

  clearInputBuffer();
  delay(200);
}
