# main.py

import src as newt


def main():
    stt = newt.SpeechToText(
        model_size="small", device="cpu", transcribe_beam_size=5, language="en"
    )
    listener = newt.Listener(stt)
    listener.start()


if __name__ == "__main__":
    main()
