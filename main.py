# main.py

import src as newt


def main():
    op = newt.Operator()

    match newt.cfg.stt.pipeline_mode:
        case "KWS":
            kws = newt.KeyWordSpotter()
            stt = newt.Whisper()
            listener = newt.Listener(stt, kws)
        case "DIRECT":
            stt = newt.Whisper()
            listener = newt.Listener(stt)

    listener.start(op)


if __name__ == "__main__":
    main()
