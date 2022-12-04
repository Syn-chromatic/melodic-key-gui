from pyaudiowpatch import PyAudio

from abc import ABC
from typing import Type, Union
from dataclasses import dataclass

DeviceInfo = dict[str, Union[str, float]]
HostApiInfo = dict[str, Union[str, int]]


@dataclass
class AbstractDevice(ABC):
    name: str
    index: int
    input_channels: int
    output_channels: int
    is_loopback: bool
    sample_rate: int
    low_in_latency: float
    low_out_latency: float
    high_in_latency: float
    high_out_latency: float


@dataclass
class WASAPIDevice(AbstractDevice):
    name: str
    index: int
    input_channels: int
    output_channels: int
    is_loopback: bool
    sample_rate: int
    low_in_latency: float
    low_out_latency: float
    high_in_latency: float
    high_out_latency: float


@dataclass
class MMEDevice(AbstractDevice):
    name: str
    index: int
    input_channels: int
    output_channels: int
    is_loopback: bool
    sample_rate: int
    low_in_latency: float
    low_out_latency: float
    high_in_latency: float
    high_out_latency: float


@dataclass
class WDSDevice(AbstractDevice):
    name: str
    index: int
    input_channels: int
    output_channels: int
    is_loopback: bool
    sample_rate: int
    low_in_latency: float
    low_out_latency: float
    high_in_latency: float
    high_out_latency: float


@dataclass
class WDMKSDevice(AbstractDevice):
    name: str
    index: int
    input_channels: int
    output_channels: int
    is_loopback: bool
    sample_rate: int
    low_in_latency: float
    low_out_latency: float
    high_in_latency: float
    high_out_latency: float


class AudioDevicesBase:
    def __init__(self):
        self._api_type_map = self._get_api_type_map()
        self._devices: list[AbstractDevice] = self._get_all_devices()

    def _get_api_type_map(self) -> dict[int, Type[AbstractDevice]]:
        api_type_map = {
            1: WDSDevice,
            2: MMEDevice,
            11: WDMKSDevice,
            13: WASAPIDevice,
        }
        return api_type_map

    def _get_all_devices(self) -> list[AbstractDevice]:
        pya = PyAudio()
        devices: list[AbstractDevice] = []
        device_count = pya.get_device_count()
        for idx in range(device_count):
            device_info = pya.get_device_info_by_index(idx)
            device_info = dict(device_info)
            host_api_info = self._get_host_api_info(pya, device_info)
            if device_info and host_api_info:
                host_api_info = dict(host_api_info)
                device_object = self._create_device_object(device_info, host_api_info)
                if device_object is not None:
                    devices.append(device_object)
        pya.terminate()
        return devices

    def _get_host_api_idx(self, device_info: DeviceInfo):
        host_api_idx = device_info.get("hostApi")
        if host_api_idx is not None and isinstance(host_api_idx, int):
            return int(host_api_idx)

    def _get_host_api_info(self, pya: PyAudio, device_info: DeviceInfo):
        host_api_idx = self._get_host_api_idx(device_info)
        if isinstance(host_api_idx, int):
            host_api_info = pya.get_host_api_info_by_index(host_api_idx)
            return host_api_info

    def _get_host_api_type(self, host_api_info: HostApiInfo):
        host_api_type = host_api_info["type"]
        if isinstance(host_api_type, int):
            return host_api_type

    def _get_host_api_name(self, host_api_info: HostApiInfo):
        host_api_name = host_api_info["name"]
        if isinstance(host_api_name, str):
            return host_api_name

    def _get_device_name(self, device_info: DeviceInfo):
        device_name = device_info["name"]
        if isinstance(device_name, str):
            return device_name

    def _get_device_idx(self, device_info: DeviceInfo):
        device_idx = device_info["index"]
        if isinstance(device_idx, int):
            return int(device_idx)

    def _get_device_ins(self, device_info: DeviceInfo):
        device_ins = device_info["maxInputChannels"]
        if isinstance(device_ins, int):
            return int(device_ins)

    def _get_device_outs(self, device_info: DeviceInfo):
        device_outs = device_info["maxOutputChannels"]
        if isinstance(device_outs, int):
            return int(device_outs)

    def _get_device_sr(self, device_info: DeviceInfo):
        device_sr = device_info["defaultSampleRate"]
        if isinstance(device_sr, (float, int)):
            return int(device_sr)

    def _get_device_low_il(self, device_info: DeviceInfo):
        device_low_il = device_info["defaultLowInputLatency"]
        if isinstance(device_low_il, float):
            return device_low_il

    def _get_device_low_ol(self, device_info: DeviceInfo):
        device_low_ol = device_info["defaultLowOutputLatency"]
        if isinstance(device_low_ol, float):
            return device_low_ol

    def _get_device_high_il(self, device_info: DeviceInfo):
        device_high_il = device_info["defaultHighInputLatency"]
        if isinstance(device_high_il, float):
            return device_high_il

    def _get_device_high_ol(self, device_info: DeviceInfo):
        device_high_ol = device_info["defaultHighOutputLatency"]
        if isinstance(device_high_ol, float):
            return device_high_ol

    def _get_device_is_loopback(self, device_info: DeviceInfo):
        device_is_loopback = device_info["isLoopbackDevice"]
        if isinstance(device_is_loopback, bool):
            return device_is_loopback

    def _create_device_object(
        self, device_info: DeviceInfo, host_api_info: HostApiInfo
    ) -> Union[AbstractDevice, None]:
        device_name = self._get_device_name(device_info)
        device_idx = self._get_device_idx(device_info)
        device_ins = self._get_device_ins(device_info)
        device_outs = self._get_device_outs(device_info)
        device_sr = self._get_device_sr(device_info)
        device_low_il = self._get_device_low_il(device_info)
        device_low_ol = self._get_device_low_ol(device_info)
        device_high_il = self._get_device_high_il(device_info)
        device_high_ol = self._get_device_high_ol(device_info)
        device_is_lb = self._get_device_is_loopback(device_info)

        host_api_name = self._get_host_api_name(host_api_info)
        host_api_type = self._get_host_api_type(host_api_info)

        if (
            device_name is not None
            and device_idx is not None
            and device_ins is not None
            and device_outs is not None
            and device_sr is not None
            and device_low_il is not None
            and device_low_ol is not None
            and device_high_il is not None
            and device_high_ol is not None
            and host_api_name is not None
            and host_api_type is not None
            and device_is_lb is not None
        ):

            if host_api_type in self._api_type_map:
                device_class = self._api_type_map[host_api_type]
                device_object = device_class(
                    name=device_name,
                    index=device_idx,
                    input_channels=device_ins,
                    output_channels=device_outs,
                    is_loopback=device_is_lb,
                    sample_rate=device_sr,
                    low_in_latency=device_low_il,
                    low_out_latency=device_low_ol,
                    high_in_latency=device_high_il,
                    high_out_latency=device_high_ol,
                )
                return device_object


class AudioDevices(AudioDevicesBase):
    def __init__(self):
        super().__init__()

    def get_devices(self) -> list[AbstractDevice]:
        return self._devices

    def get_wasapi_devices(self) -> list[AbstractDevice]:
        wasapi_devices = []
        for device in self._devices:
            if isinstance(device, WASAPIDevice):
                wasapi_devices.append(device)
        return wasapi_devices

    def get_mme_devices(self) -> list[AbstractDevice]:
        mme_devices = []
        for device in self._devices:
            if isinstance(device, MMEDevice):
                mme_devices.append(device)
        return mme_devices

    def get_wds_devices(self) -> list[AbstractDevice]:
        wds_devices = []
        for device in self._devices:
            if isinstance(device, WDSDevice):
                wds_devices.append(device)
        return wds_devices

    def get_wdmks_devices(self) -> list[AbstractDevice]:
        wdmks_devices = []
        for device in self._devices:
            if isinstance(device, WDMKSDevice):
                wdmks_devices.append(device)
        return wdmks_devices

    def filter_to_input_devices(
        self, devices: list[AbstractDevice]
    ) -> list[AbstractDevice]:
        input_devices = []
        for device in devices:
            if device.input_channels > 0 or device.is_loopback:
                input_devices.append(device)
        return input_devices

    def filter_to_output_devices(
        self, devices: list[AbstractDevice]
    ) -> list[AbstractDevice]:
        output_devices = []
        for device in devices:
            if device.output_channels > 0:
                output_devices.append(device)
        return output_devices
