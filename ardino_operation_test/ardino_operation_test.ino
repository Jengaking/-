void setup() {
  // put your setup code here, to run once:
  pinMode(13, OUTPUT);
  Serial.begin(9600);
}

char flag = '1';

void loop() {
  if(Serial.available()) {
    flag = (char)Serial.read();
    if(flag != 13 && flag != 10) Serial.println(flag);
  }
}
