import { writable, type Writable, get } from 'svelte/store';

export enum MediaStreamStatusEnum {
    INIT = "init",
    CONNECTED = "connected",
    DISCONNECTED = "disconnected",
}
export const onFrameChangeStore: Writable<{ blob: Blob }> = writable({ blob: new Blob() });

export const mediaDevices = writable<MediaDeviceInfo[]>([]);
export const mediaStreamStatus = writable(MediaStreamStatusEnum.INIT);
export const mediaStream = writable<MediaStream | null>(null);

const STORAGE_KEY = 'sdiff_selected_device_id';
const savedDeviceId = typeof localStorage !== 'undefined' ? (localStorage.getItem(STORAGE_KEY) || '') : '';
export const selectedDeviceId = writable<string>(savedDeviceId);
// Persist device selection across sessions
selectedDeviceId.subscribe(id => {
    if (typeof localStorage !== 'undefined' && id) {
        localStorage.setItem(STORAGE_KEY, id);
    }
});

export const mediaStreamActions = {
    async enumerateDevices() {
        await navigator.mediaDevices.enumerateDevices()
            .then(devices => {
                const cameras = devices.filter(device => device.kind === 'videoinput');
                mediaDevices.set(cameras);
                const saved = get(selectedDeviceId);
                // Restore last-used device if it still exists, otherwise leave unselected
                // (don't blindly default to cameras[0])
                if (saved && cameras.some(c => c.deviceId === saved)) {
                    // Already set correctly — no change needed
                } else if (!saved && cameras.length > 0) {
                    // First ever launch: pick first device but don't auto-start
                    selectedDeviceId.set(cameras[0].deviceId);
                } else if (saved && !cameras.some(c => c.deviceId === saved)) {
                    // Saved device no longer available — clear so user must pick
                    selectedDeviceId.set('');
                }
            })
            .catch(err => {
                console.error(err);
            });
    },
    async start(mediaDeviceID?: string) {
        const deviceId = mediaDeviceID || get(selectedDeviceId);
        const constraints = {
            audio: false,
            video: {
                width: 1024, height: 1024,
                ...(deviceId ? { deviceId: { exact: deviceId } } : {})
            }
        };

        await navigator.mediaDevices
            .getUserMedia(constraints)
            .then((stream) => {
                mediaStreamStatus.set(MediaStreamStatusEnum.CONNECTED);
                mediaStream.set(stream);
            })
            .catch((err) => {
                console.error(`${err.name}: ${err.message}`);
                mediaStreamStatus.set(MediaStreamStatusEnum.DISCONNECTED);
                mediaStream.set(null);
            });
    },
    async startScreenCapture() {
        const displayMediaOptions = {
            video: {
                displaySurface: "window",
            },
            audio: false,
            surfaceSwitching: "include"
        };

        let captureStream = null;

        try {
            captureStream = await navigator.mediaDevices.getDisplayMedia(displayMediaOptions);
            const videoTrack = captureStream.getVideoTracks()[0];

            console.log("Track settings:");
            console.log(JSON.stringify(videoTrack.getSettings(), null, 2));
            console.log("Track constraints:");
            console.log(JSON.stringify(videoTrack.getConstraints(), null, 2));
            mediaStreamStatus.set(MediaStreamStatusEnum.CONNECTED);
            mediaStream.set(captureStream);
        } catch (err) {
            console.error(err);
        }
    },
    async switchCamera(mediaDeviceID: string) {
        // Update the shared store immediately
        selectedDeviceId.set(mediaDeviceID);

        // Stop existing tracks before requesting new stream
        const current = get(mediaStream);
        if (current) {
            current.getTracks().forEach(track => track.stop());
        }

        const constraints = {
            audio: false,
            video: {
                width: 1024, height: 1024,
                deviceId: { exact: mediaDeviceID }
            }
        };

        await navigator.mediaDevices
            .getUserMedia(constraints)
            .then((stream) => {
                mediaStreamStatus.set(MediaStreamStatusEnum.CONNECTED);
                mediaStream.set(stream);
            })
            .catch((err) => {
                console.error(`${err.name}: ${err.message}`);
                mediaStreamStatus.set(MediaStreamStatusEnum.DISCONNECTED);
                mediaStream.set(null);
            });
    },
    async stop() {
        // Stop all tracks on the current stream
        const current = get(mediaStream);
        if (current) {
            current.getTracks().forEach(track => track.stop());
        }
        mediaStreamStatus.set(MediaStreamStatusEnum.DISCONNECTED);
        mediaStream.set(null);
    },
};
