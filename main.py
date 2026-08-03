# main.py

import src as newt


def main():
    match newt.cfg.stt_pipeline_mode:
        case "KWS":
            kws = newt.KeyWordSpotter()
            stt = newt.Whisper()
            listener = newt.Listener(stt, kws)
        case "DIRECT":
            stt = newt.Whisper()
            listener = newt.Listener(stt)

    listener.start()


if __name__ == "__main__":
    main()
