<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Button from './Button.svelte';

  let videos: Array<{name: string, path: string}> = [];
  let selectedPath: string = '';
  let running = false;
  let runningName: string | null = null;
  let statusInterval: ReturnType<typeof setInterval>;
  let videoDirectory: string = '';
  let fetchError: string = '';
  let startError: string = '';
  let speed: number = 1;

  const SPEED_OPTIONS = [1, 2, 3, 4, 5];

  onMount(async () => {
    await loadVideos();
    await pollStatus();
    statusInterval = setInterval(pollStatus, 3000);
  });

  onDestroy(() => {
    clearInterval(statusInterval);
  });

  async function loadVideos() {
    try {
      const res = await fetch('/api/virtcam/videos');
      if (res.ok) {
        const data = await res.json();
        videos = data.videos || [];
        videoDirectory = data.directory || '';
        if (videos.length > 0 && !selectedPath) {
          selectedPath = videos[0].path;
        }
      } else {
        fetchError = `API error ${res.status}`;
      }
    } catch (e) {
      fetchError = String(e);
    }
  }

  async function pollStatus() {
    try {
      const res = await fetch('/api/virtcam/status');
      if (res.ok) {
        const data = await res.json();
        running = data.running;
        runningName = data.video_name || null;
        if (data.speed != null) speed = data.speed;
        if (running && data.video_path && !selectedPath) {
          selectedPath = data.video_path;
        }
      }
    } catch (_) {}
  }

  async function start() {
    if (!selectedPath) return;
    startError = '';
    try {
      const res = await fetch('/api/virtcam/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: selectedPath, speed }),
      });
      if (res.ok) {
        running = true;
        runningName = selectedPath.split(/[\\/]/).pop() || selectedPath;
      } else {
        const data = await res.json().catch(() => ({}));
        startError = data.detail || `Error ${res.status}`;
      }
    } catch (e) {
      startError = String(e);
    }
  }

  async function stop() {
    try {
      await fetch('/api/virtcam/stop', { method: 'POST' });
      running = false;
      runningName = null;
    } catch (_) {}
  }

  async function setSpeed(s: number) {
    speed = s;
    if (running) {
      try {
        await fetch('/api/virtcam/speed', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ speed: s }),
        });
      } catch (_) {}
    }
  }
</script>

<div class="space-y-2">
  <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-3">
    <h4 class="text-sm font-semibold mb-2">Video Source</h4>

    {#if fetchError}
      <p class="text-xs text-red-500 break-all">Error: {fetchError}</p>
    {:else if videos.length === 0}
      <p class="text-xs text-gray-500 dark:text-gray-400">No videos found in:</p>
      <p class="text-xs font-mono text-gray-400 dark:text-gray-500 break-all">{videoDirectory || '(checking...)'}</p>
    {:else}
      <select
        bind:value={selectedPath}
        disabled={running}
        class="w-full px-2 py-1 text-sm bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded mb-2 disabled:opacity-50"
      >
        {#each videos as v}
          <option value={v.path}>{v.name}</option>
        {/each}
      </select>
    {/if}

    <!-- Speed control -->
    {#if videos.length > 0 || running}
      <div class="mb-2">
        <div class="flex items-center gap-1">
          <span class="text-xs text-gray-500 dark:text-gray-400 mr-1">Speed</span>
          {#each SPEED_OPTIONS as s}
            <button
              on:click={() => setSpeed(s)}
              class="px-2 py-0.5 text-xs rounded border transition-colors
                {speed === s
                  ? 'bg-blue-600 border-blue-600 text-white'
                  : 'bg-white dark:bg-gray-700 border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600'}"
            >{s}×</button>
          {/each}
          {#if speed > 1}
            <span class="text-xs text-gray-400 dark:text-gray-500 ml-1">{speed}× slower</span>
          {/if}
        </div>
      </div>
    {/if}

    <div class="flex items-center gap-2">
      {#if running}
        <Button on:click={stop} classList="text-xs bg-red-600 hover:bg-red-700 text-white px-3 py-1">
          Stop
        </Button>
        <span class="text-xs text-green-600 dark:text-green-400 truncate" title={runningName || ''}>
          ▶ {runningName}
        </span>
      {:else}
        <Button on:click={start} disabled={!selectedPath} classList="text-xs px-3 py-1">
          Start
        </Button>
        <span class="text-xs text-gray-500">Feeds OBS Virtual Camera</span>
      {/if}
    </div>
    {#if startError}
      <p class="text-xs text-red-500 mt-1 break-all">{startError}</p>
    {/if}
  </div>
</div>
