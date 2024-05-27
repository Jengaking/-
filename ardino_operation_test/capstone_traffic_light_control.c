
enum TRAFFIC_COLORS { PIN_WALK_RED = 2, PIN_WALK_GREEN = 3, PIN_CAR_GREEN, PIN_CAR_YELLOW, PIN_CAR_RED };

const int time_unit = 500; // 초록불 깜박임이 자연스럽도록 하기 위해 시간 단위를 작게 함.
int walk_units[2] = { 22500 / time_unit, 12500 / time_unit };
int car_units[3] = { 20000 / time_unit, 1000 / time_unit, 14000 / time_unit };

unsigned char walk_flag = 0;
int walk_cnt = walk_units[0] - 3; // R: 0, G: 1
unsigned char car_flag = 0; // G: 0, Y: 1, R: 2
int car_cnt = car_units[0];

void operate_walk_lights() {
    if (walk_cnt == 0) {
        walk_flag ^= 1;
        walk_cnt = walk_units[walk_flag];
    }
    // 신호등 로직 수정
    digitalWrite(PIN_WALK_RED, walk_flag ^ 0x01);
    if (walk_cnt <= 10 && walk_flag)  digitalWrite(PIN_WALK_GREEN, !digitalRead(PIN_WALK_GREEN)); // BLINKING 동작
    else digitalWrite(PIN_WALK_GREEN, walk_flag); // STATIC 동작
    
    walk_cnt--;
}

void operate_car_lights() {
    if (car_cnt == 0) {
        car_flag = (car_flag + 1) % 3;
        car_cnt = car_units[car_flag];
    }

    for (int i = 0; i < 3; i++) digitalWrite(i + PIN_CAR_GREEN, (car_flag == i));
    car_cnt--;
}

void setup() {
    for (int i = 2; i <= 6; i++) pinMode(i, OUTPUT);
}

void loop() {
    operate_walk_lights();
    operate_car_lights();
    delay(time_unit);
}