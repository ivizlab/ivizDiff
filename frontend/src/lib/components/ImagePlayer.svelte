<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { getPipelineValues, pipelineValues } from '$lib/store';
  import { parseResolution, type ResolutionInfo } from '$lib/utils';
  import { onDestroy, createEventDispatcher } from 'svelte';

  import Button from '$lib/components/Button.svelte';

  const dispatch = createEventDispatcher();
  import Floppy from '$lib/icons/floppy.svelte';
  import { snapImage } from '$lib/utils';

  let isFullscreen = false;

  export let currentResolution: ResolutionInfo | undefined = undefined;
  export let collectParams: (() => any) | undefined = undefined;

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: console.log('ImagePlayer: isLCMRunning', isLCMRunning);
  let imageEl: HTMLImageElement;
  let localResolution: ResolutionInfo;

  // Reactive resolution parsing
  $: {
    if (currentResolution) {
      // Use prop if provided
      localResolution = currentResolution;
    } else if ($pipelineValues.resolution) {
      // Fallback to pipeline values
      localResolution = parseResolution($pipelineValues.resolution);
    } else {
      // Default fallback
      localResolution = {
        width: 512,
        height: 512,
        aspectRatio: 1,
        aspectRatioString: "1:1"
      };
    }
  }
  
  async function takeSnapshot() {
    if (isLCMRunning) {
      await snapImage(imageEl, {
        prompt: getPipelineValues()?.prompt,
        negative_prompt: getPipelineValues()?.negative_prompt,
        seed: getPipelineValues()?.seed,
        guidance_scale: getPipelineValues()?.guidance_scale
      });
    }
  }

  // Save Snapshot (PNG + embedded params)
  let showNameInput = false;
  let snapshotName = '';
  let savingSnapshot = false;
  let snapshotMsg = '';

  function openSnapshotDialog() {
    const now = new Date();
    snapshotName = 'snapshot_' +
      now.getFullYear() +
      String(now.getMonth() + 1).padStart(2, '0') +
      String(now.getDate()).padStart(2, '0') + '_' +
      String(now.getHours()).padStart(2, '0') +
      String(now.getMinutes()).padStart(2, '0') +
      String(now.getSeconds()).padStart(2, '0');
    showNameInput = true;
    snapshotMsg = '';
  }

  async function saveSnapshot() {
    if (savingSnapshot) return;
    savingSnapshot = true;
    try {
      const body: any = { name: snapshotName.trim() };
      if (collectParams) {
        body.params = await collectParams();
      }
      const res = await fetch('/api/save_snapshot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const result = await res.json();
      if (res.ok) {
        snapshotMsg = `Saved: ${result.filename}`;
        showNameInput = false;
        setTimeout(() => snapshotMsg = '', 3000);
      } else {
        snapshotMsg = `Error: ${result.detail || 'Save failed'}`;
      }
    } catch {
      snapshotMsg = 'Save failed';
    } finally {
      savingSnapshot = false;
    }
  }

  function handleSnapshotKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') saveSnapshot();
    if (e.key === 'Escape') showNameInput = false;
  }

  // Load Snapshot — pick a PNG, send to backend, dispatch params to page
  let loadSnapshotInput: HTMLInputElement;
  let loadingSnapshot = false;

  function openLoadDialog() {
    loadSnapshotInput.click();
  }

  async function handleLoadSnapshot(e: Event) {
    const input = e.target as HTMLInputElement;
    if (!input.files?.length) return;
    const file = input.files[0];
    input.value = '';
    loadingSnapshot = true;
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('/api/load_snapshot', { method: 'POST', body: formData });
      const result = await res.json();
      if (res.ok && result.params) {
        dispatch('paramsLoaded', result.params);
        snapshotMsg = 'Params restored!';
        setTimeout(() => snapshotMsg = '', 3000);
      } else {
        snapshotMsg = `Load error: ${result.detail || 'No params found'}`;
      }
    } catch {
      snapshotMsg = 'Load failed';
    } finally {
      loadingSnapshot = false;
    }
  }

  async function toggleFullscreen() {
    if (!imageEl) return;

    try {
      if (!document.fullscreenElement) {
        await imageEl.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.error('toggleFullscreen: Error toggling fullscreen:', err);
    }
  }

  function handleFullscreenChange() {
    isFullscreen = !!document.fullscreenElement;
  }

  // Listen for fullscreen changes
  if (typeof window !== 'undefined') {
    document.addEventListener('fullscreenchange', handleFullscreenChange);
  }

  // ── Video recording ──────────────────────────────────────────────────────
  let recordingCanvas: HTMLCanvasElement;
  let isRecording = false;
  let mediaRecorder: MediaRecorder | null = null;
  let recordedChunks: BlobPart[] = [];
  let rafId: number | null = null;
  let recordingMsg = '';

  function getBestMimeType(): string {
    for (const t of ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']) {
      if (MediaRecorder.isTypeSupported(t)) return t;
    }
    return '';
  }

  function drawLoop() {
    if (!isRecording) return;
    if (imageEl && recordingCanvas) {
      const ctx = recordingCanvas.getContext('2d');
      if (ctx) ctx.drawImage(imageEl, 0, 0);
    }
    rafId = requestAnimationFrame(drawLoop);
  }

  function startRecording() {
    if (!imageEl || isRecording) return;

    // Size canvas to actual pipeline output pixels
    const w = imageEl.naturalWidth  || localResolution?.width  || 512;
    const h = imageEl.naturalHeight || localResolution?.height || 512;
    recordingCanvas.width  = w;
    recordingCanvas.height = h;

    const mimeType = getBestMimeType();
    const stream = recordingCanvas.captureStream(60);

    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = onRecordingStopped;

    mediaRecorder.start(100); // collect chunks every 100 ms
    isRecording = true;
    drawLoop();
  }

  async function stopRecording() {
    if (!mediaRecorder || !isRecording) return;
    mediaRecorder.stop();
    isRecording = false;
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
  }

  async function onRecordingStopped() {
    const mimeType = mediaRecorder?.mimeType || 'video/webm';
    const blob = new Blob(recordedChunks, { type: mimeType });
    recordedChunks = [];
    mediaRecorder = null;

    const now = new Date();
    const defaultName = 'recording_' +
      now.getFullYear() +
      String(now.getMonth() + 1).padStart(2, '0') +
      String(now.getDate()).padStart(2, '0') + '_' +
      String(now.getHours()).padStart(2, '0') +
      String(now.getMinutes()).padStart(2, '0') +
      String(now.getSeconds()).padStart(2, '0') + '.webm';

    if ('showSaveFilePicker' in window) {
      try {
        const handle = await (window as any).showSaveFilePicker({
          suggestedName: defaultName,
          types: [{ description: 'WebM Video', accept: { 'video/webm': ['.webm'] } }],
        });
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
        recordingMsg = 'Video saved!';
      } catch (e: any) {
        if (e?.name !== 'AbortError') {
          // showSaveFilePicker failed — fall back to download link
          triggerDownload(blob, defaultName);
        }
        // AbortError = user cancelled — do nothing
      }
    } else {
      triggerDownload(blob, defaultName);
    }
    setTimeout(() => recordingMsg = '', 4000);
  }

  function triggerDownload(blob: Blob, name: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 15000);
    recordingMsg = 'Video downloaded!';
  }

  onDestroy(() => {
    if (typeof window !== 'undefined') {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    }
    if (isRecording) stopRecording();
  });
</script>

<div 
  class="relative w-full h-full flex items-center justify-center overflow-hidden rounded-lg border border-slate-300 bg-gray-50 dark:bg-gray-900"
  style="aspect-ratio: {localResolution?.aspectRatio || 1}"
>
  <!-- Hidden canvas used as source for MediaRecorder — never displayed -->
  <canvas bind:this={recordingCanvas} class="hidden"></canvas>

  <!-- svelte-ignore a11y-missing-attribute -->
  {#if isLCMRunning && $streamId}
    <img
      bind:this={imageEl}
      class="max-w-full max-h-full object-contain rounded-lg"
      src={'/api/stream/' + $streamId}
      alt="Generated output stream"
    />
    
    <!-- Snapshot name dialog -->
    {#if showNameInput}
      <div class="absolute bottom-0 left-0 right-0 bg-black bg-opacity-85 px-3 py-2 flex items-center gap-2 rounded-b-lg">
        <input
          bind:value={snapshotName}
          on:keydown={handleSnapshotKeydown}
          placeholder="filename (no extension)"
          class="flex-1 px-2 py-1 text-sm rounded bg-gray-800 text-white border border-gray-600 focus:outline-none focus:border-blue-400"
        />
        <button
          on:click={saveSnapshot}
          disabled={savingSnapshot}
          class="px-3 py-1 text-sm bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded"
        >{savingSnapshot ? '…' : 'Save'}</button>
        <button
          on:click={() => showNameInput = false}
          class="px-2 py-1 text-sm bg-gray-700 hover:bg-gray-600 text-white rounded"
        >✕</button>
      </div>
    {/if}

    <!-- Status messages -->
    {#if snapshotMsg || recordingMsg}
      <div class="absolute top-2 left-1/2 -translate-x-1/2 bg-black bg-opacity-75 text-white text-xs px-3 py-1 rounded-full">
        {snapshotMsg || recordingMsg}
      </div>
    {/if}

    <!-- Recording indicator -->
    {#if isRecording}
      <div class="absolute top-2 left-2 flex items-center gap-1.5 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded-full">
        <span class="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
        REC
      </div>
    {/if}

    <div class="absolute bottom-2 right-2 flex gap-2">
      <!-- Record / Stop recording -->
      {#if isRecording}
        <Button
          on:click={stopRecording}
          title="Stop recording and save"
          classList={'text-sm text-white bg-red-600 bg-opacity-80 hover:bg-opacity-100 p-2 shadow-lg rounded-lg backdrop-blur-sm transition-all'}
        >
          <!-- Stop square -->
          <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            <rect x="5" y="5" width="14" height="14" rx="2"/>
          </svg>
        </Button>
      {:else}
        <Button
          on:click={startRecording}
          disabled={!isLCMRunning}
          title="Record output video"
          classList={'text-sm text-white bg-black bg-opacity-50 hover:bg-opacity-70 p-2 shadow-lg rounded-lg backdrop-blur-sm transition-all'}
        >
          <!-- Record dot -->
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="6" fill="#ef4444" stroke="none"/>
            <circle cx="12" cy="12" r="9" stroke-width="2"/>
          </svg>
        </Button>
      {/if}

      <Button
        on:click={toggleFullscreen}
        disabled={!isLCMRunning}
        title={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
        classList={'text-sm text-white bg-black bg-opacity-50 hover:bg-opacity-70 p-2 shadow-lg rounded-lg backdrop-blur-sm transition-all'}
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          {#if isFullscreen}
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 9V6a3 3 0 0 1 3-3h6a3 3 0 0 1 3 3v3M6 15v3a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3v-3M9 12h6"></path>
          {:else}
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0-5 5M4 16v4m0 0h4m-4 0 5-5m11 1v4m0 0h-4m4 0-5-5"></path>
          {/if}
        </svg>
      </Button>
      <!-- Save PNG + params snapshot -->
      <Button
        on:click={openSnapshotDialog}
        disabled={!isLCMRunning}
        title="Save Snapshot (PNG with embedded params)"
        classList={'text-sm text-white bg-black bg-opacity-50 hover:bg-opacity-70 p-2 shadow-lg rounded-lg backdrop-blur-sm transition-all'}
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M3 9a2 2 0 0 1 2-2h.93a2 2 0 0 0 1.664-.89l.812-1.22A2 2 0 0 1 10.07 4h3.86a2 2 0 0 1 1.664.89l.812 1.22A2 2 0 0 0 18.07 7H19a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
          <circle stroke-linecap="round" stroke-linejoin="round" stroke-width="2" cx="12" cy="13" r="3" />
        </svg>
      </Button>
      <!-- Load params from snapshot PNG -->
      <Button
        on:click={openLoadDialog}
        disabled={loadingSnapshot}
        title="Load params from snapshot PNG"
        classList={'text-sm text-white bg-black bg-opacity-50 hover:bg-opacity-70 p-2 shadow-lg rounded-lg backdrop-blur-sm transition-all'}
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M4 16v1a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-1M12 12V4m0 8-3-3m3 3 3-3" />
        </svg>
      </Button>
      <input
        bind:this={loadSnapshotInput}
        type="file"
        accept=".png,image/png"
        class="hidden"
        on:change={handleLoadSnapshot}
      />
      <Button
        on:click={takeSnapshot}
        disabled={!isLCMRunning}
        title={'Take Snapshot'}
        classList={'text-sm text-white bg-black bg-opacity-50 hover:bg-opacity-70 p-2 shadow-lg rounded-lg backdrop-blur-sm transition-all'}
      >
        <Floppy classList={''} />
      </Button>
    </div>
  {:else}
    <div class="w-full h-full flex flex-col items-center justify-center text-gray-400 dark:text-gray-600">
      <div class="w-24 h-24 mb-4 opacity-30">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
          <circle cx="9" cy="9" r="2"/>
          <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
        </svg>
      </div>
      <p class="text-lg font-medium">Generated output will appear here</p>
      <p class="text-sm opacity-75">Click "Start Stream" to begin</p>
      {#if localResolution}
        <div class="text-xs mt-2 opacity-50">
          Ready for {localResolution.width}×{localResolution.height} ({localResolution.aspectRatioString})
        </div>
      {/if}
    </div>
  {/if}
</div>
