# Author: Leo Vainio

import curses
from curses import wrapper
import socket

# Establish connection with server
s = socket.socket()
localhost = socket.gethostname()
port = 1234
s.connect((localhost, port))

# Receive the graphics from the server
def read_line():
    COLS = 100
    line = ""
    for col in range(COLS):
        line = line + s.recv(1).decode()
    return line
    
# Initiate curses and main game loop
def main(stdscr):
    stdscr.clear()

    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    CYAN_BLACK = curses.color_pair(1)

    stdscr.nodelay(True)    # don't wait for key input
    curses.curs_set(False)  # invisible cursor

    ROWS = 20
    while True:
        # read in graphics from server and draw them to the window
        for x in range(ROWS):
            line = read_line()
            stdscr.addstr(x, 0, line, CYAN_BLACK)

        # Constants for arrowkeys and 'q' (quit)
        KEY_QUIT = 113
        KEY_DOWN = 258
        KEY_UP = 259
        KEY_LEFT = 260
        KEY_RIGHT = 261

        # receive key input from user and send proper message to server
        key = stdscr.getch()
        if key == KEY_QUIT:
            break
        elif key == KEY_DOWN:
            s.send(b'DOWN\n')
        elif key == KEY_UP:
            s.send(b'UP\n')
        elif key == KEY_LEFT:
            s.send(b'LEFT\n')
        elif key == KEY_RIGHT:
            s.send(b'RIGHT\n')
        else:
            s.send(b'STILL\n')

        stdscr.refresh()
        stdscr.clear()
        curses.flushinp()

# Initiate curses and start application
wrapper(main)
s.close()