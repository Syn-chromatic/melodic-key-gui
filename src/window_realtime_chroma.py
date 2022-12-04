
import time

from PyQt6.QtCore import QThreadPool, QByteArray
from PyQt6.QtWidgets import QWidget, QGraphicsPixmapItem, QGraphicsScene
from PyQt6.QtGui import QImage, QPixmap

from utils.shared_dcs import AudioPipeline, ChromaPipeline
from utils.qrunnable_utils import GeneralWorker, GeneralWorkerCallback
from utils.audio_utils.audio_recorder import AudioRecorder, ChromaProcessor

from ui.realtime_chroma_ui import Ui_Form


class RealTimeWindow(QWidget):
    def __init__(self, audio_recorder: AudioRecorder):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.threadpool = QThreadPool()
        self.window_alive = True

        self.ui.instrumentFilter.setChecked(True)
        self.ui.absoluteFilter.setChecked(True)
        self.ui.nnFilter.setChecked(True)
        self.ui.mSmoothingFilter.setChecked(True)
        self.ui.mSmoothingValue.setValue(20)
        self.ui.minClipFilter.setChecked(True)
        self.ui.minClipValue.setValue(0.2)

        self.adpl = self.get_adpl_ui()
        self.chpl = self.get_chpl_ui()

        self.ui.applySettings.clicked.connect(self.set_adpl_chpl)
        self.chroma_processor = ChromaProcessor(self.adpl, self.chpl)
        self.audio_recorder = audio_recorder
        self.scene = QGraphicsScene()
        self.ui.graphicsChroma.setScene(self.scene)
        self.audio_io = None

    def set_adpl_chpl(self):
        self.adpl = self.get_adpl_ui()
        self.chpl = self.get_chpl_ui()
        self.chroma_processor.update_adpl(self.adpl)
        self.chroma_processor.update_chpl(self.chpl)

    def get_adpl_ui(self):
        adpl = AudioPipeline(
            hpass_fl_state=self.ui.highPassFilter.isChecked(),
            hpass_val=0,
            lpass_fl_state=self.ui.lowPassFilter.isChecked(),
            lpass_val=0,
            inst_fl_state=self.ui.instrumentFilter.isChecked(),
            save_out_state=False,
            calc_bpm_state=False,
            core_count=1,
        )
        return adpl

    def get_chpl_ui(self):
        chpl = ChromaPipeline(
            abs_fl_state=self.ui.absoluteFilter.isChecked(),
            nn_fl_state=self.ui.nnFilter.isChecked(),
            mds_fl_state=self.ui.mSmoothingFilter.isChecked(),
            mds_val=self.ui.mSmoothingValue.value(),
            min_clip_fl_state=self.ui.minClipFilter.isChecked(),
            min_clip_val=0.5,
        )
        return chpl

    def show_window(self):
        self.show()

        worker = GeneralWorker(
            self.audio_recorder.start_recording, self.chroma_processor
        )
        worker.signals.output.connect(self.finish_recording)
        self.threadpool.start(worker)

        worker_live = GeneralWorkerCallback(self.live_chromagram)
        worker_live.signals.progress.connect(self.add_sceneitem)
        self.threadpool.start(worker_live)

        lcd_thread = GeneralWorker(self.lcd_thread)
        self.threadpool.start(lcd_thread)


    def get_audio_io(self):
        return self.audio_io

    def finish_recording(self, audio_io):
        self.audio_io = audio_io

    def stop_recording(self):
        self.audio_recorder.stop_recording()
        self.window_alive = False

    def add_sceneitem(self, progress):
        self.ui.graphicsChroma.show()
        self.scene.addItem(self.qg_pixmap)

    def live_chromagram(self, progress_callback):
        while self.window_alive:
            chromagram_io = self.chroma_processor.save_chromagram()

            chromagram_bytes = chromagram_io.getvalue()
            image_bytearr = QByteArray().append(chromagram_bytes)
            qimage = QImage()
            qimage.loadFromData(image_bytearr)

            pixmap = QPixmap(qimage)
            pixmap = pixmap.scaled(
                self.ui.graphicsChroma.size().width(),
                self.ui.graphicsChroma.size().height(),
            )
            self.qg_pixmap = QGraphicsPixmapItem(pixmap)
            self.chroma_result = self.chroma_processor.get_result()

            # self.ui.keyResLabel.setText(self.chroma_result.key)
            # self.ui.probabilityLCD.display(self.chroma_result.probability)
            # self.ui.bpmLCD.display(self.chroma_result.bpm)
            progress_callback.emit(0)
            time.sleep(0.1)

    def lcd_thread(self):
        while self.window_alive:
            self.chroma_result = self.chroma_processor.get_result()
            self.ui.keyResLabel.setText(self.chroma_result.key)
            self.ui.probabilityLCD.display(self.chroma_result.probability)
            self.ui.bpmLCD.display(self.chroma_result.bpm)
            time.sleep(0.05)
