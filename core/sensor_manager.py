"""
SensorManager — shared sensor instance for the application.

Centralizes sensor configuration, connection, and measurement so multiple
pages (Sensor page, Color Analysis, etc.) reflect the same state.

Usage:
    sm = SensorManager.instance()
    sm.configure('virtual', noise_level=0.02)
    sm.connect()
    reading = sm.read(pattern_hint=(0.5, 0.5, 0.5))

Signals:
    sensor_changed       — backend / config changed
    connection_changed(bool)
    reading_received(SensorReading)
"""
from __future__ import annotations
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QObject, Signal

from sensor_module import (
    SensorInterface, SensorReading, VirtualSensor, CRColorimeterSensor,
    CR_DEFAULT_BAUDRATE, CR_DEFAULT_TIMEOUT,
)


class SensorManager(QObject):
    """Process-wide shared sensor state."""

    sensor_changed     = Signal()          # backend or config replaced
    connection_changed = Signal(bool)
    reading_received   = Signal(object)    # SensorReading

    _instance: Optional["SensorManager"] = None

    @classmethod
    def instance(cls) -> "SensorManager":
        if cls._instance is None:
            cls._instance = SensorManager()
        return cls._instance

    def __init__(self):
        super().__init__()
        self._sensor: Optional[SensorInterface] = None
        self._backend: str = "virtual"
        self._config: dict = {"noise_level": 0.02}
        # Build initial (unconnected) virtual sensor so read() doesn't NPE
        self._build_sensor()

    # ── Backend management ─────────────────────────────────────
    def configure(self, backend: str, **config) -> None:
        """Replace the current sensor with a freshly-configured one.
        Disconnects the previous instance if it was connected."""
        if self._sensor and self.is_connected():
            try:
                self._sensor.disconnect()
            except Exception:
                pass
            self.connection_changed.emit(False)
        self._backend = backend
        self._config = dict(config)
        self._build_sensor()
        self.sensor_changed.emit()

    def _build_sensor(self) -> None:
        if self._backend == "cr":
            self._sensor = CRColorimeterSensor(
                port=self._config.get("port", "COM3"),
                baudrate=self._config.get("baudrate", CR_DEFAULT_BAUDRATE),
                timeout=self._config.get("timeout", CR_DEFAULT_TIMEOUT),
            )
        else:
            self._sensor = VirtualSensor(
                noise_level=self._config.get("noise_level", 0.02),
                display_colorspace=self._config.get("display_colorspace", "BT.2020"),
                max_luminance=self._config.get("max_luminance", 100.0),
                black_level=self._config.get("black_level", 0.05),
                native_gamma=self._config.get("native_gamma", 2.2),
            )

    # ── Connection ─────────────────────────────────────────────
    def connect(self) -> bool:
        if self._sensor is None:
            self._build_sensor()
        try:
            ok = bool(self._sensor.connect())
        except Exception:
            ok = False
        self.connection_changed.emit(ok)
        return ok

    def disconnect(self) -> bool:
        if self._sensor is None:
            return False
        try:
            ok = bool(self._sensor.disconnect())
        except Exception:
            ok = False
        self.connection_changed.emit(False)
        return ok

    def is_connected(self) -> bool:
        return bool(self._sensor and self._sensor.is_connected())

    # ── Measurement ────────────────────────────────────────────
    def read(self, pattern_hint: Optional[tuple] = None) -> Optional[SensorReading]:
        """Perform a measurement. Auto-connects on first use for Virtual."""
        if self._sensor is None:
            self._build_sensor()
        if not self.is_connected():
            # Virtual auto-connects silently; CR refuses (user must press Connect)
            if self._backend == "virtual":
                self.connect()
            else:
                return None
        if pattern_hint is not None and hasattr(self._sensor, "set_pattern_hint"):
            try:
                self._sensor.set_pattern_hint(tuple(pattern_hint))
            except Exception:
                pass
        try:
            reading = self._sensor.read()
        except Exception as exc:
            # Return an invalid reading wrapper rather than raising
            import numpy as np, time as _t
            return SensorReading(
                rgb=np.zeros(3), xyz=np.zeros(3), cie_xy=(0.0, 0.0),
                luminance=0.0, timestamp=_t.time(),
                is_valid=False, error_message=str(exc))
        self.reading_received.emit(reading)
        return reading

    # ── Introspection ──────────────────────────────────────────
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def config(self) -> dict:
        return dict(self._config)

    @property
    def sensor(self) -> Optional[SensorInterface]:
        return self._sensor
