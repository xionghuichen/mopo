import psutil
import time


if __name__ == '__main__':
    max_percent = -1
    while True:
        percent = psutil.virtual_memory().percent
        time.sleep(0.2)
        max_percent = max(percent, max_percent)
        print('{} / {}'.format(percent, max_percent))