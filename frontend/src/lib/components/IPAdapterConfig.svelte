<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';

  export let ipadapterInfo: any = null;
  export let currentScale: number = 1.0;
  export let currentWeightType: string = "linear";
  export let currentBlendWeight: number = 0.0;

  const dispatch = createEventDispatcher();

  // Style image upload state — slot A
  let styleImageFile: HTMLInputElement;
  let uploadingImage = false;
  let uploadStatus = '';
  export let currentStyleImage: string | null = null;

  // Style image upload state — slot B (blend target)
  let styleImageFileB: HTMLInputElement;
  let uploadingImageB = false;
  let uploadStatusB = '';
  export let currentStyleImageB: string | null = null;
  let blendWeight: number = 0.0;

  // Collapsible section state
  let showIPAdapter: boolean = true;
  let showAdvanced: boolean = false;


  // Available weight types
  const weightTypes = [
    "linear", "ease in", "ease out", "ease in-out", "reverse in-out", 
    "weak input", "weak output", "weak middle", "strong middle", 
    "style transfer", "composition", "strong style transfer", 
    "style and composition", "style transfer precise", "composition precise"
  ];

  async function updateIPAdapterScale(scale: number) {
    try {
      const response = await fetch('/api/ipadapter/update-scale', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          scale: scale,
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateIPAdapterScale: Failed to update scale:', result.detail);
      }
    } catch (error) {
      console.error('updateIPAdapterScale: Update failed:', error);
    }
  }

  function handleScaleChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const scale = parseFloat(target.value);
    
    // Update local state immediately for responsiveness
    currentScale = scale;
    
    updateIPAdapterScale(scale);
  }

  async function updateIPAdapterWeightType(weightType: string) {
    try {
      const response = await fetch('/api/ipadapter/update-weight-type', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          weight_type: weightType,
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateIPAdapterWeightType: Failed to update weight type:', result.detail);
      }
    } catch (error) {
      console.error('updateIPAdapterWeightType: Update failed:', error);
    }
  }

  function handleWeightTypeChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    const weightType = target.value;
    
    // Update local state immediately for responsiveness
    currentWeightType = weightType;
    
    updateIPAdapterWeightType(weightType);
  }

  async function uploadStyleImage() {
    if (!styleImageFile.files || styleImageFile.files.length === 0) {
      uploadStatus = 'Please select an image file';
      return;
    }

    const file = styleImageFile.files[0];
    if (!file.type.startsWith('image/')) {
      uploadStatus = 'Please select a valid image file';
      return;
    }

    uploadingImage = true;
    uploadStatus = 'Uploading style image...';

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/ipadapter/upload-style-image', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        uploadStatus = 'Style image uploaded successfully!';
        
        // Create preview URL for the uploaded image and keep it for display
        const reader = new FileReader();
        reader.onload = (e) => {
          currentStyleImage = e.target?.result as string;
        };
        reader.readAsDataURL(file);
        
        // Clear file input
        styleImageFile.value = '';
        
        setTimeout(() => {
          uploadStatus = '';
        }, 3000);
      } else {
        uploadStatus = `Error: ${result.detail || 'Failed to upload style image'}`;
      }
    } catch (error) {
      console.error('uploadStyleImage: Upload failed:', error);
      uploadStatus = 'Upload failed. Please try again.';
    } finally {
      uploadingImage = false;
    }
  }

  function selectStyleImage() {
    styleImageFile.click();
  }

  async function uploadStyleImageB() {
    if (!styleImageFileB.files || styleImageFileB.files.length === 0) {
      uploadStatusB = 'Please select an image file';
      return;
    }
    const file = styleImageFileB.files[0];
    if (!file.type.startsWith('image/')) {
      uploadStatusB = 'Please select a valid image file';
      return;
    }

    uploadingImageB = true;
    uploadStatusB = 'Uploading...';

    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch('/api/ipadapter/upload-style-image-b', {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      if (response.ok) {
        uploadStatusB = 'Uploaded!';
        const reader = new FileReader();
        reader.onload = (e) => { currentStyleImageB = e.target?.result as string; };
        reader.readAsDataURL(file);
        styleImageFileB.value = '';
        setTimeout(() => { uploadStatusB = ''; }, 3000);
      } else {
        uploadStatusB = `Error: ${result.detail || 'Upload failed'}`;
      }
    } catch (error) {
      uploadStatusB = 'Upload failed.';
    } finally {
      uploadingImageB = false;
    }
  }

  function selectStyleImageB() {
    styleImageFileB.click();
  }

  async function updateBlendWeight(t: number) {
    try {
      await fetch('/api/ipadapter/blend-weight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ blend_weight: t }),
      });
    } catch (error) {
      console.error('updateBlendWeight failed:', error);
    }
  }

  function handleBlendWeightChange(event: Event) {
    const target = event.target as HTMLInputElement;
    blendWeight = parseFloat(target.value);
    currentBlendWeight = blendWeight; // keep parent in sync for snapshot save
    updateBlendWeight(blendWeight);
  }

  // Update current scale and weight type when prop changes
  $: if (ipadapterInfo?.scale !== undefined) {
    currentScale = ipadapterInfo.scale;
  }
  $: if (ipadapterInfo?.weight_type !== undefined) {
    currentWeightType = ipadapterInfo.weight_type;
  }
  // Keep blend slider in sync with parent (e.g. snapshot restore)
  $: if (currentBlendWeight !== blendWeight) {
    blendWeight = currentBlendWeight;
  }
</script>

<div class="space-y-2">
  <!-- IPAdapter Section -->
  <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
    <button 
      on:click={() => showIPAdapter = !showIPAdapter}
      class="w-full p-3 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg border-b border-gray-200 dark:border-gray-700"
    >
      <h4 class="text-sm font-semibold">IPAdapter</h4>
      <span class="text-sm">{showIPAdapter ? '−' : '+'}</span>
    </button>
    {#if showIPAdapter}
      <div class="p-3">
        <!-- IPAdapter Status -->
        <div class="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-700 rounded mb-1">
          {#if ipadapterInfo?.enabled}
            <div class="w-2 h-2 bg-green-500 rounded-full"></div>
            <span class="text-sm font-medium text-green-800 dark:text-green-200">IPAdapter Enabled</span>
          {:else}
            <div class="w-2 h-2 bg-gray-400 rounded-full"></div>
            <span class="text-sm text-gray-600 dark:text-gray-400">Standard Mode</span>
          {/if}
        </div>

        {#if ipadapterInfo?.enabled}
          <!-- Style Image Upload -->
          <div class="space-y-2">
            <!-- Style Image Preview -->
            {#if currentStyleImage}
              <div>
                <img
                  src={currentStyleImage}
                  alt="Uploaded style image"
                  class="w-full max-w-32 h-24 object-cover rounded border border-gray-200 dark:border-gray-600"
                />
              </div>
            {:else if ipadapterInfo?.style_image_path}
              <div>
                <img
                  src={ipadapterInfo.style_image_path}
                  alt="Style image"
                  class="w-full max-w-32 h-24 object-cover rounded border border-gray-200 dark:border-gray-600"
                />
                <p class="text-xs text-gray-500">
                  {#if ipadapterInfo.style_image_path.includes('/api/ipadapter/uploaded-style-image')}
                    Uploaded
                  {:else if ipadapterInfo.style_image_path.includes('/api/default-image')}
                    Default (input.png)
                  {:else}
                    From config
                  {/if}
                </p>
              </div>
            {/if}

            <!-- Upload Button -->
            <div class="flex items-center gap-2">
              <Button
                on:click={selectStyleImage}
                disabled={uploadingImage}
                classList="text-sm px-3 py-1"
              >
                {uploadingImage ? 'Uploading...' : 'Upload Style Image'}
              </Button>
            </div>

            <!-- Hidden file input -->
            <input
              bind:this={styleImageFile}
              type="file"
              accept="image/*"
              class="hidden"
              on:change={uploadStyleImage}
            />

            <!-- Upload Status -->
            {#if uploadStatus}
              <p class="text-xs {uploadStatus.includes('Error') || uploadStatus.includes('Please') ? 'text-red-600' : 'text-green-600'}">
                {uploadStatus}
              </p>
            {/if}

            <!-- Style Image B (blend target) -->
            <div class="border-t border-gray-200 dark:border-gray-600 pt-2 mt-1 space-y-1">
              <p class="text-xs font-medium text-gray-500 dark:text-gray-400">Style B (blend target)</p>
              {#if currentStyleImageB}
                <img
                  src={currentStyleImageB}
                  alt="Style image B"
                  class="w-full max-w-32 h-24 object-cover rounded border border-gray-200 dark:border-gray-600"
                />
              {/if}
              <div class="flex items-center gap-2">
                <Button
                  on:click={selectStyleImageB}
                  disabled={uploadingImageB}
                  classList="text-sm px-3 py-1"
                >
                  {uploadingImageB ? 'Uploading...' : currentStyleImageB ? 'Replace B' : 'Load Style B'}
                </Button>
              </div>
              <input
                bind:this={styleImageFileB}
                type="file"
                accept="image/*"
                class="hidden"
                on:change={uploadStyleImageB}
              />
              {#if uploadStatusB}
                <p class="text-xs {uploadStatusB.includes('Error') || uploadStatusB.includes('Please') ? 'text-red-600' : 'text-green-600'}">
                  {uploadStatusB}
                </p>
              {/if}

              <!-- Blend slider — only shown when B is loaded -->
              {#if currentStyleImageB}
                <div class="space-y-1 pt-1">
                  <div class="flex items-center justify-between">
                    <label class="text-xs font-medium text-gray-600 dark:text-gray-400" title="0 = Style A only, 1 = Style B only">A → B Blend</label>
                    <span class="text-xs text-gray-600 dark:text-gray-400">{blendWeight.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={blendWeight}
                    on:input={handleBlendWeightChange}
                    class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
                  />
                </div>
              {/if}
            </div>

            <!-- Scale Control -->
            <div class="space-y-1">
              <div class="flex items-center justify-between">
                <label class="text-xs font-medium text-gray-600 dark:text-gray-400" title="How strongly the style image influences generation. Higher = stronger style.">Scale</label>
                <span class="text-xs text-gray-600 dark:text-gray-400">{currentScale.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="2"
                step="0.01"
                value={currentScale}
                on:input={handleScaleChange}
                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
              />
            </div>

            <!-- Advanced (collapsed by default) -->
            <button
              on:click={() => showAdvanced = !showAdvanced}
              class="w-full text-left text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 flex items-center justify-between px-1"
            >
              <span>Advanced</span>
              <span>{showAdvanced ? '−' : '+'}</span>
            </button>

            {#if showAdvanced}
              <!-- Weight Type Control -->
              <div class="space-y-1">
                <label class="text-xs font-medium text-gray-600 dark:text-gray-400" title="How IPAdapter influence is distributed across model layers">Weight Type</label>
                <select
                  value={currentWeightType}
                  on:change={handleWeightTypeChange}
                  class="w-full px-2 py-1 text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {#each weightTypes as weightType}
                    <option value={weightType}>{weightType}</option>
                  {/each}
                </select>
              </div>

              <!-- IPAdapter Info -->
              {#if ipadapterInfo?.model_path}
                <p class="text-xs text-gray-500 font-mono break-all">{ipadapterInfo.model_path}</p>
              {/if}
            {/if}
          </div>
        {:else}
          <p class="text-xs text-gray-600 dark:text-gray-400">
            Load a configuration with IPAdapter settings to enable style-guided generation.
          </p>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  /* Range slider styling */
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

  .dark input[type="range"]::-webkit-slider-track {
    background: #4b5563;
  }

  .dark input[type="range"]::-moz-range-track {
    background: #4b5563;
  }
</style> 