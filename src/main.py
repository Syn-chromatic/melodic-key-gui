if __name__ == "__main__":
    import logging
    import matplotlib
    import multiprocessing

    logging.basicConfig(level=logging.INFO)
    matplotlib.use("QtAgg")
    multiprocessing.freeze_support()

    import sys
    import traceback
    from window_main import Main
    from PyQt6 import QtWidgets

    try:
        app = QtWidgets.QApplication(sys.argv)
        window = Main()
        window.show()
        app.exec()

    except Exception as error:
        error_tb = "".join(traceback.format_tb(error.__traceback__))
        logging.error(f"ERROR TRACEBACK:\n{error_tb}")
        logging.error(f"ERROR AT MAIM:\n{error}")
