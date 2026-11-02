"""找出一个空闲端口，供一键启动.bat 使用。

从 8501 开始往后找，避免"端口已被占用"导致双击启动失败。

注意：Windows 上如果已有进程监听 0.0.0.0:8501，新进程仍能 bind 127.0.0.1:8501
（不像 Linux 会直接报错），所以不能只靠 bind 判断，必须先试连一下。
"""

import socket

START, END = 8501, 8530


def is_busy(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True  # 能连上，说明有人在监听
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))  # 绑 0.0.0.0，能发现监听在任意地址上的进程
        return False
    except OSError:
        return True


for port in range(START, END):
    if not is_busy(port):
        print(port)
        break
else:
    print(START)
