from PyQt6.QtCore import QRunnable, QObject, pyqtSignal


class WorkerSignals(QObject):
    output = pyqtSignal(object)
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(Exception)


class GeneralWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(GeneralWorker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            output = self.fn(*self.args, **self.kwargs)
        except Exception as error:
            self.signals.error.emit(error)
        else:
            self.signals.output.emit(output)
        finally:
            self.signals.finished.emit()


class GeneralWorkerCallback(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(GeneralWorkerCallback, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.kwargs['progress_callback'] = self.signals.progress

    def run(self):
        try:
            output = self.fn(*self.args, **self.kwargs)
        except Exception as error:
            self.signals.error.emit(error)
        else:
            self.signals.output.emit(output)
        finally:
            self.signals.finished.emit()

