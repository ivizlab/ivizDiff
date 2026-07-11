<script lang="ts">
  import { onMount } from 'svelte';
  import Button from './Button.svelte';
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus } from '$lib/lcmLive';

  export let initialLoras: any[] = [];
  export let loraDir: string = '';
  export let getStreamData: (() => any[]) | null = null;

  interface LoraEntry {
    path: string;
    scale: number;
    enabled: boolean;
    name: string;
    trigger_word: string;
  }

  interface AvailableLora {
    name: string;
    filename: string;
    path: string;
  }

  $: isStreamRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;

  let showSection: boolean = true;
  let availableLoras: AvailableLora[] = [];
  let activeLoras: LoraEntry[] = [];
  let selectedPath: string = '';
  let applyStatus: string = '';
  let applyStatusError: boolean = false;
  let pendingRestart: boolean = false;
  let loading: boolean = false;
  let restarting: boolean = false;

  onMount(async () => {
    await loadAvailable();
    if (initialLoras?.length) {
      activeLoras = initialLoras.map(normEntry);
    }
  });

  function normEntry(e: any): LoraEntry {
    const p = String(e.path || '');
    return {
      path: p,
      scale: Number(e.scale ?? 1.0),
      enabled: e.enabled !== false,
      name: p.split('/').pop()?.replace(/\.(safetensors|pt)$/i, '') ?? p,
      trigger_word: String(e.trigger_word ?? ''),
    };
  }

  async function loadAvailable() {
    try {
      const res = await fetch('/api/loras');
      if (res.ok) {
        const data = await res.json();
        availableLoras = data.available ?? [];
        if (!activeLoras.length && data.active?.length) {
          activeLoras = data.active.map(normEntry);
        }
        if (data.lora_dir) loraDir = data.lora_dir;
      }
    } catch (e) {
      console.error('loadAvailable: failed', e);
    }
  }

  function addLora() {
    if (!selectedPath) return;
    if (activeLoras.some(l => l.path === selectedPath)) return;
    const found = availableLoras.find(a => a.path === selectedPath);
    activeLoras = [...activeLoras, {
      path: selectedPath,
      scale: 1.0,
      enabled: true,
      name: found?.name ?? selectedPath.split('/').pop()?.replace(/\.(safetensors|pt)$/i, '') ?? selectedPath,
      trigger_word: '',
    }];
    selectedPath = '';
    pendingRestart = true;
  }

  function removeLora(idx: number) {
    activeLoras = activeLoras.filter((_, i) => i !== idx);
    pendingRestart = true;
  }

  function handleScaleInput(idx: number, event: Event) {
    const v = parseFloat((event.target as HTMLInputElement).value);
    activeLoras[idx].scale = v;
    activeLoras = [...activeLoras];
    pendingRestart = true;
  }

  function handleEnabledToggle(idx: number) {
    activeLoras[idx].enabled = !activeLoras[idx].enabled;
    activeLoras = [...activeLoras];
    pendingRestart = true;
  }

  async function applyLoras() {
    loading = true;
    applyStatus = '';
    applyStatusError = false;
    try {
      const res = await fetch('/api/loras', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ loras: activeLoras.map(l => ({ path: l.path, scale: l.scale, enabled: l.enabled, trigger_word: l.trigger_word })) }),
      });
      const data = await res.json();
      if (!res.ok) {
        applyStatusError = true;
        applyStatus = 'Error: ' + (data.detail ?? 'Unknown error');
        return;
      }
      pendingRestart = false;
      if (isStreamRunning && getStreamData) {
        loading = false;
        restarting = true;
        applyStatus = 'Restarting pipeline…';
        try {
          lcmLiveActions.stop();
          await lcmLiveActions.start(getStreamData);
          applyStatus = 'Done';
        } catch (e) {
          applyStatusError = true;
          applyStatus = 'Restart failed.';
        } finally {
          restarting = false;
        }
      } else {
        applyStatus = 'Applied — start stream to activate';
      }
    } catch (e) {
      applyStatusError = true;
      applyStatus = 'Request failed.';
    } finally {
      loading = false;
      setTimeout(() => { applyStatus = ''; applyStatusError = false; }, 4000);
    }
  }

  // Options not already in the active list
  $: addableOptions = availableLoras.filter(a => !activeLoras.some(l => l.path === a.path));
</script>

<div class="space-y-2">
  <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
    <button
      on:click={() => showSection = !showSection}
      class="w-full p-3 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg border-b border-gray-200 dark:border-gray-700"
    >
      <h4 class="text-sm font-semibold">LoRAs</h4>
      <span class="text-sm">{showSection ? '−' : '+'}</span>
    </button>

    {#if showSection}
      <div class="p-3 space-y-3">

        <!-- Active LoRA list -->
        {#if activeLoras.length === 0}
          <p class="text-xs text-gray-500 dark:text-gray-400">No LoRAs active.</p>
        {:else}
          <div class="space-y-2">
            {#each activeLoras as lora, i}
              <div class="bg-gray-50 dark:bg-gray-700 rounded p-2 space-y-1">
                <div class="flex items-center justify-between gap-2">
                  <!-- Enable toggle -->
                  <input
                    type="checkbox"
                    checked={lora.enabled}
                    on:change={() => handleEnabledToggle(i)}
                    class="w-3.5 h-3.5 accent-blue-500 cursor-pointer flex-shrink-0"
                    title="Enable/disable this LoRA"
                  />
                  <!-- Name -->
                  <span class="text-xs font-mono truncate flex-1 {lora.enabled ? '' : 'opacity-40'}" title={lora.path}>
                    {lora.name}
                  </span>
                  <!-- Remove -->
                  <button
                    on:click={() => removeLora(i)}
                    class="text-xs text-red-400 hover:text-red-600 flex-shrink-0 px-1"
                    title="Remove"
                  >✕</button>
                </div>
                <!-- Scale slider -->
                <div class="flex items-center gap-2 {lora.enabled ? '' : 'opacity-40'}">
                  <span class="text-xs text-gray-500 w-8 text-right">{lora.scale.toFixed(2)}</span>
                  <input
                    type="range"
                    min="0" max="2" step="0.05"
                    value={lora.scale}
                    on:input={(e) => handleScaleInput(i, e)}
                    disabled={!lora.enabled}
                    class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
                  />
                </div>
                <!-- Trigger word -->
                <input
                  type="text"
                  bind:value={lora.trigger_word}
                  on:input={() => { activeLoras = [...activeLoras]; pendingRestart = true; }}
                  placeholder="trigger word (optional)"
                  class="w-full px-1.5 py-0.5 text-xs font-mono bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded focus:outline-none focus:ring-1 focus:ring-blue-400 placeholder-gray-400"
                />
              </div>
            {/each}
          </div>
        {/if}

        <!-- Active trigger word reminder -->
        {#if activeLoras.some(l => l.enabled && l.trigger_word)}
          {@const words = activeLoras.filter(l => l.enabled && l.trigger_word).map(l => l.trigger_word).join(', ')}
          <p class="text-xs text-amber-600 dark:text-amber-400">
            Active trigger words: <span class="font-mono">{words}</span> — add to prompt for full effect
          </p>
        {/if}

        <!-- Add LoRA row -->
        {#if addableOptions.length > 0}
          <div class="flex items-center gap-2">
            <select
              bind:value={selectedPath}
              class="flex-1 px-2 py-1 text-xs bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="">— select LoRA —</option>
              {#each addableOptions as opt}
                <option value={opt.path}>{opt.name}</option>
              {/each}
            </select>
            <Button on:click={addLora} disabled={!selectedPath} classList="text-xs px-2 py-1">Add</Button>
          </div>
        {:else if availableLoras.length === 0}
          <p class="text-xs text-gray-400">No .safetensors files found in<br/><span class="font-mono break-all">{loraDir}</span></p>
        {/if}

        <!-- Apply button -->
        <div class="pt-1 flex items-center gap-2">
          <Button
            on:click={applyLoras}
            disabled={loading || restarting}
            classList="text-xs px-3 py-1.5 {pendingRestart ? 'bg-amber-500 hover:bg-amber-600 text-white border-amber-500' : ''}"
          >
            {#if loading}
              Applying…
            {:else if restarting}
              Restarting…
            {:else if pendingRestart}
              {isStreamRunning ? 'Apply & Restart ●' : 'Apply LoRAs ●'}
            {:else}
              {isStreamRunning ? 'Apply & Restart' : 'Apply LoRAs'}
            {/if}
          </Button>
        </div>

        {#if applyStatus}
          <p class="text-xs {applyStatusError ? 'text-red-500' : restarting ? 'text-blue-500' : 'text-green-600'}">
            {applyStatus}
          </p>
        {/if}

      </div>
    {/if}
  </div>
</div>

<style>
  input[type="range"]::-webkit-slider-thumb {
    appearance: none;
    height: 14px;
    width: 14px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid white;
  }
  input[type="range"]::-moz-range-thumb {
    height: 14px;
    width: 14px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid white;
  }
  input[type="range"]::-webkit-slider-track {
    height: 6px;
    border-radius: 3px;
    background: #e5e7eb;
  }
  input[type="range"]::-moz-range-track {
    height: 6px;
    border-radius: 3px;
    background: #e5e7eb;
    border: none;
  }
  .dark input[type="range"]::-webkit-slider-track { background: #4b5563; }
  .dark input[type="range"]::-moz-range-track { background: #4b5563; }
</style>
