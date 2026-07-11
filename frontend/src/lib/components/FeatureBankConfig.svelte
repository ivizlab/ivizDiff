<script lang="ts">
  export let enabled: boolean = true;
  export let weight: number = 0.15;

  let showSection: boolean = true;

  async function postUpdate() {
    try {
      await fetch('/api/feature-bank', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled, weight }),
      });
    } catch (error) {
      console.error('FeatureBankConfig: update failed:', error);
    }
  }

  function handleToggle() {
    enabled = !enabled;
    postUpdate();
  }

  function handleWeightChange(event: Event) {
    weight = parseFloat((event.target as HTMLInputElement).value);
    postUpdate();
  }
</script>

<div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
  <button
    on:click={() => showSection = !showSection}
    class="w-full p-3 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg border-b border-gray-200 dark:border-gray-700"
  >
    <h4 class="text-sm font-semibold">Temporal Smoothing</h4>
    <span class="text-sm">{showSection ? '−' : '+'}</span>
  </button>

  {#if showSection}
    <div class="p-3 space-y-3">
      <!-- Enable toggle -->
      <div class="flex items-center justify-between">
        <span class="text-xs font-medium text-gray-600 dark:text-gray-400" title="Blend the last few denoised latents to reduce frame-to-frame structural flicker">Feature Bank</span>
        <button
          on:click={handleToggle}
          class="relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none {enabled ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'}"
        >
          <span class="inline-block h-3 w-3 transform rounded-full bg-white shadow transition-transform {enabled ? 'translate-x-5' : 'translate-x-1'}"></span>
        </button>
      </div>

      <!-- Weight slider -->
      {#if enabled}
        <div class="space-y-1">
          <div class="flex items-center justify-between">
            <label class="text-xs font-medium text-gray-600 dark:text-gray-400" title="How strongly previous frames blend into the current one. 0 = off, higher = smoother but less reactive.">Blend Weight</label>
            <span class="text-xs text-gray-600 dark:text-gray-400">{weight.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0"
            max="0.5"
            step="0.01"
            value={weight}
            on:input={handleWeightChange}
            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
          />
          <div class="flex justify-between text-xs text-gray-400">
            <span>off</span>
            <span>smooth</span>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  input[type="range"]::-webkit-slider-thumb {
    appearance: none;
    height: 16px;
    width: 16px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }
  input[type="range"]::-moz-range-thumb {
    height: 16px;
    width: 16px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }
  input[type="range"]::-webkit-slider-track {
    height: 8px;
    border-radius: 4px;
    background: #e5e7eb;
  }
  input[type="range"]::-moz-range-track {
    height: 8px;
    border-radius: 4px;
    background: #e5e7eb;
    border: none;
  }
  .dark input[type="range"]::-webkit-slider-track { background: #4b5563; }
  .dark input[type="range"]::-moz-range-track { background: #4b5563; }
</style>
