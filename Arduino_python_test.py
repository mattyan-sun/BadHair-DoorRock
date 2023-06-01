# -*- coding: utf-8 -*-

import serial
import time

def main():
    #  COMポートを開く
    print("Open Port")
    ser = serial.Serial("/dev/ttyUSB0", 115200)
    while True:
        #  LED点灯
        #ser.write(b"1")
        #time.sleep(1)
        #  LED消灯
        #ser.write(b"0")
        #time.sleep(1)

        #Aruduinoに送信
        #ser.write(bytes([123]))

        #Aruduinoから受信
        data = ser.read_all()
        print("data:{}".format(data))

        print("OK")

    print("Close Port")
    ser.close()

if __name__ == '__main__':
    main()
