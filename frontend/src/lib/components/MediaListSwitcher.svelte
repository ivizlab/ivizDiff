<script lang="ts">
  import { mediaDevices, mediaStreamActions, selectedDeviceId } from '$lib/mediaStream';
  import Screen from '$lib/icons/screen.svelte';
  import { onMount } from 'svelte';

  $: {
    console.log($mediaDevices);
  }
  $: {
    console.log($selectedDeviceId);
  }

  onMount(() => {
    if ($mediaDevices.length > 0 && !$selectedDeviceId) {
      selectedDeviceId.set($mediaDevices[0].deviceId);
    }
  });
</script>

<div class="flex items-center justify-center text-xs">
  <button
    title="Share your screen"
    class="border-1 my-1 flex cursor-pointer gap-1 rounded-md border-gray-500 border-opacity-50 bg-black bg-opacity-30 p-1 font-medium text-white"
    on:click={() => mediaStreamActions.startScreenCapture()}
  >
    <span>Share</span>
    <Screen classList={''} />
  </button>
  {#if $mediaDevices}
    <select
      bind:value={$selectedDeviceId}
      on:change={() => mediaStreamActions.switchCamera($selectedDeviceId)}
      id="devices-list"
      class="border-1 block cursor-pointer rounded-md border-gray-800 border-opacity-50 bg-black bg-opacity-30 p-1 font-medium text-white"
    >
      {#each $mediaDevices as device, i}
        <option value={device.deviceId}>{device.label}</option>
      {/each}
    </select>
  {/if}
</div>
