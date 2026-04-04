from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
import asyncio
import threading
import time
import logging
import json


class InputControl(ABC):
    """Generic interface for input controls that can modify parameters"""
    
    def __init__(self, parameter_name: str, min_value: float = 0.0, max_value: float = 1.0):
        self.parameter_name = parameter_name
        self.min_value = min_value
        self.max_value = max_value
        self.is_active = False
        self.update_callback: Optional[Callable[[str, float], None]] = None
    
    @abstractmethod
    async def start(self) -> None:
        """Start the input control"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the input control"""
        pass
    
    @abstractmethod
    def get_current_value(self) -> float:
        """Get the current normalized value (0.0 to 1.0)"""
        pass
    
    def set_update_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set callback for parameter updates"""
        self.update_callback = callback
    
    def normalize_value(self, raw_value: float) -> float:
        """Normalize raw input value to 0.0-1.0 range"""
        return max(0.0, min(1.0, raw_value))
    
    def scale_to_parameter(self, normalized_value: float) -> float:
        """Scale normalized value to parameter range"""
        return self.min_value + (normalized_value * (self.max_value - self.min_value))
    
    def _trigger_update(self, normalized_value: float) -> None:
        """Trigger parameter update if callback is set"""
        if self.update_callback:
            scaled_value = self.scale_to_parameter(normalized_value)
            self.update_callback(self.parameter_name, scaled_value)


class GamepadInput(InputControl):
    """Gamepad input control for parameter modification"""
    
    def __init__(self, parameter_name: str, min_value: float = 0.0, max_value: float = 1.0, 
                 gamepad_index: int = 0, axis_index: int = 0, deadzone: float = 0.1):
        super().__init__(parameter_name, min_value, max_value)
        self.gamepad_index = gamepad_index
        self.axis_index = axis_index
        self.deadzone = deadzone
        self.current_value = 0.0
        self._stop_event = threading.Event()
        self._thread = None
    
    async def start(self) -> None:
        """Start gamepad monitoring"""
        if self.is_active:
            return
        
        self.is_active = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_gamepad, daemon=True)
        self._thread.start()
        logging.info(f"GamepadInput: Started monitoring gamepad {self.gamepad_index}, axis {self.axis_index}")
    
    async def stop(self) -> None:
        """Stop gamepad monitoring"""
        if not self.is_active:
            return
        
        self.is_active = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        logging.info(f"GamepadInput: Stopped monitoring gamepad {self.gamepad_index}, axis {self.axis_index}")
    
    def get_current_value(self) -> float:
        """Get current normalized value"""
        return self.current_value
    
    def _monitor_gamepad(self) -> None:
        """Monitor gamepad input in background thread"""
        try:
            import pygame
            
            # Initialize pygame for gamepad support
            pygame.init()
            pygame.joystick.init()
            
            # Check if gamepad is available
            if pygame.joystick.get_count() <= self.gamepad_index:
                logging.error(f"GamepadInput: Gamepad {self.gamepad_index} not found")
                return
            
            # Initialize the gamepad
            joystick = pygame.joystick.Joystick(self.gamepad_index)
            joystick.init()
            
            logging.info(f"GamepadInput: Connected to {joystick.get_name()}")
            
            # Monitor gamepad input
            while not self._stop_event.is_set():
                pygame.event.pump()
                
                # Get axis value
                if self.axis_index < joystick.get_numaxes():
                    raw_value = joystick.get_axis(self.axis_index)
                    
                    # Apply deadzone
                    if abs(raw_value) < self.deadzone:
                        raw_value = 0.0
                    
                    # Convert from [-1, 1] to [0, 1] range
                    normalized_value = (raw_value + 1.0) / 2.0
                    
                    # Update current value
                    self.current_value = normalized_value
                    
                    # Trigger update callback
                    self._trigger_update(normalized_value)
                
                time.sleep(0.016)  # ~60 FPS polling
                
        except ImportError:
            logging.error("GamepadInput: pygame not installed. Install with: pip install pygame")
        except Exception as e:
            logging.error(f"GamepadInput: Error monitoring gamepad: {e}")
        finally:
            try:
                pygame.quit()
            except:
                pass


# MicrophoneInput moved to frontend - browser handles microphone access


class BreathInputSource:
    """
    Real-time breath detection input source. Connects as a WebSocket client to
    breath.py (ws://localhost:8765) and exposes two continuous signals for the
    parameter mapping system:
      - amplitude: smoothed breath envelope, rolling-max normalised to 0-1
      - bpm:       bpm_rolling value, clamped/normalised over a 5-30 BPM range

    Architecture notes for future steps:
    =========================================================================
    Step 2 (planned) — Speech detection layer:
      breath.py will emit a "speech" event with a `transcription` field when
      the speech gate fires.  To handle it, add a `_handle_speech` method and
      register it in `self._handlers["speech"]`.  The dispatcher already ignores
      unknown event types, so no structural changes are needed here until then.

    Step 3 (planned) — Whisper / LLM prompt integration:
      Route the transcription through a local Whisper model for accuracy, then
      optionally through a local LLM (Ollama) to convert speech into a prompt
      rewrite or parameter shift rather than a literal transcription.
      Handler chain: "speech" event → Whisper refinement → LLM → POST to a
      prompt-update endpoint (to be added in Step 3).
    =========================================================================
    """

    BREATH_WS_URL = "ws://localhost:8765"
    RECONNECT_DELAY = 3.0   # seconds between connection attempts
    BPM_MIN = 5.0
    BPM_MAX = 30.0

    def __init__(self):
        self._status: str = "DISCONNECTED"   # DISCONNECTED | CONNECTED | LIVE
        self._last_event_type: str = ""
        self._amplitude: float = 0.0         # normalised 0-1
        self._bpm: float = 0.0               # normalised 0-1
        self._amplitude_max: float = 1e-9    # rolling max for normaliser (avoid div-by-zero)
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

        # Event dispatcher — maps event-type string → handler.
        # Route known event types here; unknown types are silently ignored.
        # To add Step 2 speech support: self._handlers["speech"] = self._handle_speech
        self._handlers: Dict[str, Callable] = {
            "inhale":    self._handle_breath_phase,
            "exhale":    self._handle_breath_phase,
            "cycle_end": self._handle_breath_phase,
        }

    async def start(self) -> None:
        """Start the background WebSocket connection task."""
        if self._task and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run())
        logging.info("BreathInputSource: Started")

    async def stop(self) -> None:
        """Stop the background task and mark as disconnected."""
        if self._stop_event:
            self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._status = "DISCONNECTED"
        logging.info("BreathInputSource: Stopped")

    def get_status(self) -> Dict[str, Any]:
        """Return current connection state and signal values for the status endpoint."""
        return {
            "status":     self._status,
            "last_event": self._last_event_type,
            "amplitude":  round(self._amplitude, 4),
            "bpm":        round(self._bpm, 4),
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_breath_phase(self, event: Dict[str, Any]) -> None:
        """Handle inhale, exhale, and cycle_end events."""
        self._last_event_type = event.get("event", "")

        # Amplitude — rolling-max normaliser (same concept as mic RMS normalisation)
        raw_amp = float(event.get("amplitude", 0.0))
        if raw_amp > self._amplitude_max:
            self._amplitude_max = raw_amp
        self._amplitude = raw_amp / self._amplitude_max if self._amplitude_max > 0 else 0.0

        # BPM — clamp to [BPM_MIN, BPM_MAX] then normalise to [0, 1]
        bpm = float(event.get("bpm_rolling", 0.0))
        self._bpm = max(0.0, min(1.0,
            (bpm - self.BPM_MIN) / (self.BPM_MAX - self.BPM_MIN)
        ))

        self._status = "LIVE"

    # Step 2 placeholder — uncomment and implement when breath.py emits speech events:
    # def _handle_speech(self, event: Dict[str, Any]) -> None:
    #     transcription = event.get("transcription", "")
    #     # TODO Step 3: route transcription → Whisper → LLM → prompt update

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Reconnecting WebSocket client loop."""
        import websockets  # local import — optional dependency

        while not self._stop_event.is_set():
            try:
                self._status = "CONNECTED"
                async with websockets.connect(self.BREATH_WS_URL) as ws:
                    logging.info("BreathInputSource: Connected to breath.py at %s", self.BREATH_WS_URL)
                    async for raw_message in ws:
                        if self._stop_event.is_set():
                            break
                        try:
                            event = json.loads(raw_message)
                            event_type = event.get("event", "")
                            handler = self._handlers.get(event_type)
                            if handler:
                                handler(event)
                            # Unknown event types (e.g. future "speech") are silently ignored
                        except json.JSONDecodeError as e:
                            logging.warning("BreathInputSource: Malformed JSON: %s", e)
                        except Exception as e:
                            logging.warning("BreathInputSource: Error processing event: %s", e)

            except (ConnectionRefusedError, OSError):
                logging.debug(
                    "BreathInputSource: Cannot connect to %s — breath.py not running? "
                    "Retrying in %.0fs.", self.BREATH_WS_URL, self.RECONNECT_DELAY
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.warning("BreathInputSource: Unexpected connection error: %s", e)

            if not self._stop_event.is_set():
                self._status = "DISCONNECTED"
                # Wait before reconnecting; wake early if stop is requested
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.RECONNECT_DELAY)
                except asyncio.TimeoutError:
                    pass

        self._status = "DISCONNECTED"


class InputManager:
    """Manages multiple input controls"""
    
    def __init__(self):
        self.inputs: Dict[str, InputControl] = {}
        self.parameter_update_callback: Optional[Callable[[str, float], None]] = None
    
    def add_input(self, input_id: str, input_control: InputControl) -> None:
        """Add an input control"""
        input_control.set_update_callback(self._handle_parameter_update)
        self.inputs[input_id] = input_control
        logging.info(f"InputManager: Added input control {input_id} for parameter {input_control.parameter_name}")
    
    def remove_input(self, input_id: str) -> None:
        """Remove an input control"""
        if input_id in self.inputs:
            asyncio.create_task(self.inputs[input_id].stop())
            del self.inputs[input_id]
            logging.info(f"InputManager: Removed input control {input_id}")
    
    async def start_input(self, input_id: str) -> None:
        """Start a specific input control"""
        if input_id in self.inputs:
            await self.inputs[input_id].start()
    
    async def stop_input(self, input_id: str) -> None:
        """Stop a specific input control"""
        if input_id in self.inputs:
            await self.inputs[input_id].stop()
    
    async def start_all(self) -> None:
        """Start all input controls"""
        for input_control in self.inputs.values():
            await input_control.start()
    
    async def stop_all(self) -> None:
        """Stop all input controls"""
        for input_control in self.inputs.values():
            await input_control.stop()
    
    def set_parameter_update_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set callback for parameter updates from any input"""
        self.parameter_update_callback = callback
    
    def get_input_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all inputs"""
        status = {}
        for input_id, input_control in self.inputs.items():
            status[input_id] = {
                "parameter_name": input_control.parameter_name,
                "is_active": input_control.is_active,
                "current_value": input_control.get_current_value(),
                "min_value": input_control.min_value,
                "max_value": input_control.max_value
            }
        return status
    
    def _handle_parameter_update(self, parameter_name: str, value: float) -> None:
        """Handle parameter update from input controls"""
        if self.parameter_update_callback:
            self.parameter_update_callback(parameter_name, value)