#
# main.py
# Entrypoint for Atlas
#


def main(no_ui: bool = True) -> None:
    from atlas import Atlas

    atlas = Atlas(ignored_modules=["ui"] if no_ui else [])
    atlas.run()


if __name__ == "__main__":
    main(no_ui=True)
