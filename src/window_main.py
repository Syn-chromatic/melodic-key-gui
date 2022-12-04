import os
import logging
from io import BytesIO
from pydub import AudioSegment
from multiprocessing import cpu_count

import traceback
from PyQt6.QtCore import QThreadPool, QBuffer, QIODevice, QByteArray
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from utils.qrunnable_utils import GeneralWorker
from utils.audio_utils.audio_pipeline import ChromaMT
from utils.audio_utils.audio_devices import AudioDevices
from utils.audio_utils.audio_recorder import AudioRecorder
from utils.shared_dcs import AudioPipeline


from ui.musicui import Ui_MainWindow
from window_chroma import ChromaDialog
from window_realtime_chroma import RealTimeWindow




class Main(QMainWindow):
    def __init__(self):
        self.ui.highPassCheckbox.setChecked(True)
        self.ui.lowPassCheckbox.setChecked(True)
        self.ui.instrumentFilterCheckbox.setChecked(True)
        self.ui.horizontalSlider.setRange(0, 0)
        self.ui.lowPassDial.setValue(2000)
        self.ui.highPassDial.setValue(1000)

        self.setAcceptDrops(True)

    def __new__(cls):
        logging.info("Created Main Instance")
        instance = super().__new__(cls)
        super(Main, instance).__init__()
        cls.ui = Ui_MainWindow()
        cls.ui.setupUi(instance)
        cls.audio_player = AudioPlayer()
        cls.audio_processor = AudioProcessor()
        return instance

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        extension = os.path.splitext(file_path)[1].lower()
        validExtensions = [".mp3", ".wav", ".ogg", ".flac"]
        if event.mimeData().hasUrls() and extension in validExtensions:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.audio_player.load_audio_file(file_path)
            event.accept()
        else:
            event.ignore()


class AudioPlayer(Main):
    def __init__(self):
        self.mediaPlayer = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.mediaPlayer.setAudioOutput(self.audio_output)

        self.mediaPlayer.mediaStatusChanged.connect(self.media_status_changed)
        self.mediaPlayer.errorOccurred.connect(self.media_error)
        self.mediaPlayer.durationChanged.connect(self.media_duration_changed)

        self.audioDevices = self.getAudioDevices()
        self.insertDevicesIntoList()
        self.ui.playButton.setEnabled(False)
        self.ui.pauseButton.setEnabled(False)
        self.currentFile = None

        self.ui.playButton.setEnabled(False)
        self.ui.pauseButton.setEnabled(False)
        self.ui.stopRecordButton.setEnabled(False)
        self.ui.startProcessingButton.setEnabled(False)

        self.threadpool = QThreadPool()

        self.ui.playButton.clicked.connect(self.play_audio)
        self.ui.pauseButton.clicked.connect(self.pause_audio)

        self.ui.recordButton.clicked.connect(self.recordAudio)

        self.ui.recordButton.clicked.connect(lambda: self.set_position(0))

        self.ui.stopRecordButton.clicked.connect(self.stopRecord)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)
        self.ui.horizontalSlider.sliderMoved.connect(self.set_position)

    def __new__(cls, *args, **kwargs):
        logging.info("Created AudioPlayer Instance")
        instance = super(Main, AudioPlayer).__new__(cls)
        super(Main, instance).__init__()
        return instance

    def media_duration_changed(self, duration):
        logging.info(f"MediaPlayer Duration: {duration}")

    def media_error(self, error):
        logging.warning(f"MediaPlayer Error: {error}")

    def media_status_changed(self, status):
        logging.info(f"MediaStatus changed: {status}")

    def load_audio_file(self, file: str):
        audio_io = BytesIO()
        audio_file = AudioSegment.from_file(file)
        audio_file.export(audio_io, format="wav", codec="pcm_s16le")
        self.load_audio_io(audio_io)

    def load_audio_io(self, audio_io: BytesIO):
        self.buf = QBuffer()
        self.file_bytearray = QByteArray().append(audio_io.getvalue())
        self.buf.setBuffer(self.file_bytearray)
        self.buf.open(QIODevice.OpenModeFlag.ReadOnly)
        self.mediaPlayer.setSourceDevice(self.buf)

        self.currentFile = audio_io
        self.ui.playButton.setEnabled(True)
        self.ui.pauseButton.setEnabled(True)
        self.ui.startProcessingButton.setEnabled(True)

    def play_audio(self):
        self.mediaPlayer.play()

    def pause_audio(self):
        self.mediaPlayer.pause()

    def stop_audio(self):
        self.mediaPlayer.stop()

    def switch_thread_status(self):
        self.ui.recordButton.setEnabled(not self.ui.recordButton.isEnabled())
        self.ui.stopRecordButton.setEnabled(not self.ui.stopRecordButton.isEnabled())

    def reset_functions(self):
        self.currentFile = None
        self.ui.playButton.setEnabled(False)
        self.ui.pauseButton.setEnabled(False)
        self.ui.startProcessingButton.setEnabled(False)
        self.ui.entireDuration.setText(self.humanize_time(int(0)))

    def position_changed(self, position):
        position_s = position / 1000
        timer = self.humanize_time(position_s)
        self.ui.currentDuration.setText(timer)
        self.ui.horizontalSlider.setValue(position)

    def duration_changed(self, duration):
        duration_s = duration / 1000
        timer = self.humanize_time(duration_s)
        self.ui.entireDuration.setText(timer)
        self.ui.horizontalSlider.setRange(0, duration)

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    @staticmethod
    def humanize_time(secs):
        mins, secs = divmod(secs, 60)
        hours, mins = divmod(mins, 60)
        return "%02d:%02d:%02d" % (hours, mins, secs)

    def getAudioDevices(self):
        audio_devices = AudioDevices()
        wasapi_devices = audio_devices.get_wasapi_devices()
        wasapi_ins = audio_devices.filter_to_input_devices(wasapi_devices)
        return wasapi_ins

    def insertDevicesIntoList(self):
        for i in self.audioDevices:
            device_name = i.name
            self.ui.devicesList.addItem(device_name)

    def recorder_error(self, error):
        logging.warning(f"AudioRecorder Error: {error}")

    def recordAudio(self):
        self.stop_audio()
        self.reset_functions()
        self.switch_thread_status()

        device = self.audioDevices[self.ui.devicesList.currentIndex()]
        self.audio_recorder = AudioRecorder(device)

        # self.ui.stopRecordButton.clicked.connect(self.audio_recorder.stop_recording)
        # chromagram_viewer = ChrogramViewer()

        self.realtime_window = RealTimeWindow(self.audio_recorder)
        self.realtime_window.show_window()

        # worker = GeneralWorker(self.audio_recorder.start_recording, chromagram_viewer)
        # worker.signals.output.connect(self.load_audio_io)
        # worker.signals.error.connect(self.recorder_error)
        # self.threadpool.start(worker)

    def stopRecord(self):
        self.switch_thread_status()
        self.realtime_window.stop_recording()

    def get_audio_file(self):
        return self.currentFile


class AudioProcessor(AudioPlayer):
    def __init__(self):
        self.threadcount = cpu_count()
        self.threadpool = QThreadPool()

        self.ui.startProcessingButton.clicked.connect(self.processAudio)
        self.ui.lowPassDial.valueChanged.connect(self.set_lowPassValue)
        self.ui.highPassDial.valueChanged.connect(self.set_highPassValue)

    def __new__(cls, *args, **kwargs):
        logging.info("Created AudioProcessor Instance")
        instance = super(Main, AudioProcessor).__new__(cls)
        super(Main, instance).__init__()
        return instance

    def set_lowPassValue(self, value):
        self.ui.lowPassCurrent.setText(str(value))

    def set_highPassValue(self, value):
        self.ui.highPassCurrent.setText(str(value))

    def processingState(self):
        self.ui.recordButton.setEnabled(False)
        self.ui.stopRecordButton.setEnabled(False)

        self.ui.saveOutputCheckbox.setEnabled(False)
        self.ui.highPassCheckbox.setEnabled(False)
        self.ui.lowPassCheckbox.setEnabled(False)
        self.ui.instrumentFilterCheckbox.setEnabled(False)

        self.ui.startProcessingButton.setEnabled(False)
        self.ui.startProcessingButton.setText("Processing..")

    def completedState(self):
        self.ui.recordButton.setEnabled(True)
        self.ui.saveOutputCheckbox.setEnabled(True)
        self.ui.highPassCheckbox.setEnabled(True)
        self.ui.lowPassCheckbox.setEnabled(True)
        self.ui.instrumentFilterCheckbox.setEnabled(True)
        self.ui.startProcessingButton.setEnabled(True)
        self.ui.startProcessingButton.setText("Start Processing")

    def audh_output(self, chromas):
        self.completedState()
        self.dialog_window = ChromaDialog()
        self.dialog_window.show_window(chromas)

    def audh_finish(self):
        logging.info("AudioProcessor finish thread")

    def audh_error(self, error):
        error_tb = "".join(traceback.format_tb(error.__traceback__))
        logging.error(f"ERROR TRACEBACK:\n{error_tb}")
        logging.error(f"ERROR AT AudioProcessor:\n{error}")
        logging.warning(f"AudioProcessor Error: {error}")

    def audh_progress(self):
        logging.info("AudioProcessor progress")

    def processAudio(self):
        adpl = AudioPipeline(
            hpass_fl_state=self.ui.highPassCheckbox.isChecked(),
            hpass_val=self.ui.highPassDial.value(),
            lpass_fl_state=self.ui.lowPassCheckbox.isChecked(),
            lpass_val=self.ui.lowPassDial.value(),
            inst_fl_state=self.ui.instrumentFilterCheckbox.isChecked(),
            save_out_state=self.ui.saveOutputCheckbox.isChecked(),
            calc_bpm_state=True,
            core_count=6,
        )

        audioHandler = ChromaMT(adpl)

        current_file = self.audio_player.currentFile
        worker = GeneralWorker(audioHandler.get_audio_decomp, current_file)
        worker.signals.output.connect(self.audh_output)
        worker.signals.finished.connect(self.audh_finish)
        worker.signals.error.connect(self.audh_error)
        self.threadpool.start(worker)
        self.processingState()






