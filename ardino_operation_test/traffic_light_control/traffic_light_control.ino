// 8 G, 9 Y, 10 R
const int green = 8, red = 10;

void setup() {
  // put your setup code here, to run once:
  pinMode(green, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(red, OUTPUT);
  
  Serial.begin(9600);
}

char flag = '1';
const int time_unit = 500;
const int red_clock = 50;
const int green_clock = 50;
int cur_light = 0;
int light_count = green_clock;

int green_on = 1;

void blink_green_light() {
  if(green_on) digitalWrite(green, LOW);
  else digitalWrite(green, HIGH);

  green_on = (green_on + 1) % 2;
}

void loop() {
  if(light_count == 0) {
    cur_light = (cur_light + 1) % 2;
    light_count = (cur_light) ? green_clock : red_clock;
  }
  Serial.println(cur_light);
  switch (cur_light) {
    case 1 :
      digitalWrite(red, LOW);
      if(light_count > green_clock / 2) digitalWrite(green, HIGH);
      else blink_green_light();
      break;
    case 0 :
      digitalWrite(10, HIGH);
      digitalWrite(8, LOW);
      digitalWrite(9, LOW);
      green_on = 1;
  }
  delay(time_unit);
  light_count--;
}

