# main.py

import src as newt


def main():
    stt = newt.SpeechToText()
    listener = newt.Listener(stt)
    listener.start()


if __name__ == "__main__":
    main()
