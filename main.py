# main.py

import src as newt


def main():
    llm = newt.LLM()

    match newt.cfg.stt.pipeline_mode:
        case "KWS":
            kws = newt.KeyWordSpotter()
            stt = newt.Whisper()
            listener = newt.Listener(stt, kws)
        case "DIRECT":
            stt = newt.Whisper()
            listener = newt.Listener(stt)

    listener.register_text_operator(newt.operate)

    listener.start()


if __name__ == "__main__":
    main()
