<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { Fields, PipelineInfo } from '$lib/types';
  import { PipelineMode } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import VirtCamControl from '$lib/components/VirtCamControl.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import ControlNetConfig from '$lib/components/ControlNetConfig.svelte';
  import IPAdapterConfig from '$lib/components/IPAdapterConfig.svelte';
  import FeatureBankConfig from '$lib/components/FeatureBankConfig.svelte';
  import LoraConfig from '$lib/components/LoraConfig.svelte';
  import BlendingControl from '$lib/components/BlendingControl.svelte';
  import ResolutionPicker from '$lib/components/ResolutionPicker.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import Warning from '$lib/components/Warning.svelte';
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus } from '$lib/lcmLive';
  import { mediaStreamActions, onFrameChangeStore } from '$lib/mediaStream';
  import { getPipelineValues, deboucedPipelineValues, pipelineValues } from '$lib/store';
  import { parseResolution, type ResolutionInfo } from '$lib/utils';
  import TextArea from '$lib/components/TextArea.svelte';
  import InputControl from '$lib/components/InputControl.svelte';

  let pipelineParams: Fields;
  let pipelineInfo: PipelineInfo;
  let controlnetInfo: any = null;
  let ipadapterInfo: any = null;
  let ipadapterScale: number = 1.0;
  let ipadapterWeightType: string = "linear";
  let ipadapterBlendWeight: number = 0.0;
  let ipadapterStyleImageBUrl: string | null = null;
  let ipadapterStyleImageAUrl: string | null = null;
  let tIndexList: number[] = [35, 45];
  let guidanceScale: number = 1.1;
  let delta: number = 0.7;
  let numInferenceSteps: number = 50;
  let seed: number = 2;
  let negativePrompt: string = '';
  let upscaleFactor: number = 1;
  let promptBlendingConfig: any = null;
  let seedBlendingConfig: any = null;
  let normalizePromptWeights: boolean = true;
  let normalizeSeedWeights: boolean = true;
  let featureBankEnabled: boolean = true;
  let featureBankWeight: number = 0.15;
  let loraList: any[] = [];
  let loraDir: string = '';
  let blendingResetKey: number = 0;
  let controlnetResetKey: number = 0;
  let pageContent: string;
  let isImageMode: boolean = false;
  let maxQueueSize: number = 0;
  let currentQueueSize: number = 0;
  let queueCheckerRunning: boolean = false;
  let warningMessage: string = '';

  let currentResolution: ResolutionInfo;
  let apiError: string = '';
  let isRetrying: boolean = false;
  
  // Reactive resolution parsing
  $: {
    if ($pipelineValues.resolution) {
      currentResolution = parseResolution($pipelineValues.resolution);
    } else if (pipelineParams?.width?.default && pipelineParams?.height?.default) {
      // Fallback to pipeline params
      currentResolution = {
        width: Number(pipelineParams.width.default),
        height: Number(pipelineParams.height.default),
        aspectRatio: Number(pipelineParams.width.default) / Number(pipelineParams.height.default),
        aspectRatioString: "1:1"
      };
    }
  }
  
  // Panel state management
  let showPromptBlending: boolean = true; // Default to expanded since it's the unified blending interface
  let showResolutionPicker: boolean = true; // Default to expanded
  let leftPanelCollapsed: boolean = false;
  let rightPanelCollapsed: boolean = false;

  // FPS tracking
  let fps = 0;
  let fpsInterval: number | null = null;

  onMount(() => {
    getSettings();
    updateFPS();
    fpsInterval = setInterval(updateFPS, 1000);
    // Enumerate video devices on load so the picker shows before clicking Start
    mediaStreamActions.enumerateDevices();
  });

  onDestroy(() => {
    if (fpsInterval) {
      clearInterval(fpsInterval);
    }
  });

  async function getSettings() {
    try {
      apiError = '';
      isRetrying = false;
      
      const response = await fetch('/api/settings');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const settings = await response.json();

      pipelineParams = settings.input_params.properties;
      pipelineInfo = settings.info.properties;
      
      // Initialize prompt value in store if not already set
      if (!($pipelineValues.prompt)) {
        pipelineValues.update(values => ({
          ...values,
          prompt: pipelineParams.prompt?.default || "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
        }));
      }
      
      controlnetInfo = settings.controlnet || null;
      ipadapterInfo = settings.ipadapter || null;
      ipadapterScale = settings.ipadapter?.scale || 1.0;
      ipadapterWeightType = settings.ipadapter?.weight_type || "linear";
      tIndexList = settings.t_index_list || [35, 45];
      guidanceScale = settings.guidance_scale || 1.1;
      delta = settings.delta || 0.7;
      numInferenceSteps = settings.num_inference_steps || 50;
      seed = settings.seed || 2;
      promptBlendingConfig = settings.prompt_blending || null;
      seedBlendingConfig = settings.seed_blending || null;
      normalizePromptWeights = settings.normalize_prompt_weights ?? true;
      normalizeSeedWeights = settings.normalize_seed_weights ?? true;
      featureBankEnabled = settings.feature_bank_enabled ?? true;
      featureBankWeight = settings.feature_bank_weight ?? 0.15;
      loraList = settings.loras ?? [];
      loraDir = settings.lora_dir ?? '';
      isImageMode = pipelineInfo.input_mode.default === PipelineMode.IMAGE;
      maxQueueSize = settings.max_queue_size;
      pageContent = settings.page_content;
      
      console.log('getSettings: promptBlendingConfig:', promptBlendingConfig);
      console.log('getSettings: current prompt in store:', $pipelineValues.prompt);
      
      // Load negative prompt from config
      if (settings.config_negative_prompt !== undefined) {
        negativePrompt = settings.config_negative_prompt;
      }

      // Update prompt in store if config prompt is available
      if (settings.config_prompt) {
        pipelineValues.update(values => ({
          ...values,
          prompt: settings.config_prompt
        }));
        console.log('getSettings: Updated prompt from config_prompt:', settings.config_prompt);
      }
      
      // Set initial resolution value if available
      if (settings.current_resolution) {
        pipelineValues.update(values => ({
          ...values,
          resolution: settings.current_resolution
        }));
      }
      
      console.log(pipelineParams);
      console.log('handleControlNetUpdate: ControlNet Info:', controlnetInfo);
      console.log('handleControlNetUpdate: T-Index List:', tIndexList);
      toggleQueueChecker(true);
      
    } catch (error) {
      console.error('Failed to load settings:', error);
      apiError = error instanceof Error ? error.message : 'Failed to connect to the API. Please check if the server is running.';
    }
  }

  async function retryConnection() {
    isRetrying = true;
    await getSettings();
  }

  function handleControlNetUpdate(event: CustomEvent) {
    controlnetInfo = event.detail.controlnet;
    
    // Update prompt if config prompt is available
    if (event.detail.config_prompt) {
      pipelineValues.update(values => ({
        ...values,
        prompt: event.detail.config_prompt
      }));
    }
    
    // Update t_index_list if available
    if (event.detail.t_index_list) {
      tIndexList = [...event.detail.t_index_list];
    }
    
    console.log('handleControlNetUpdate: ControlNet updated:', controlnetInfo);
    console.log('handleControlNetUpdate: T-Index List updated:', tIndexList);
  }

  async function handleTIndexListUpdate(newTIndexList: number[]) {
    try {
      const response = await fetch('/api/params', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          t_index_list: newTIndexList
        }),
      });

      if (response.ok) {
        tIndexList = [...newTIndexList]; // Update local state
        console.log('handleTIndexListUpdate: T-Index List updated:', tIndexList);
      } else {
        const result = await response.json();
        console.error('handleTIndexListUpdate: Failed to update t_index_list:', result.detail);
      }
    } catch (error) {
      console.error('handleTIndexListUpdate: Failed to update t_index_list:', error);
    }
  }

  async function handleResolutionUpdate(resolution: string) {
    try {
      const response = await fetch('/api/params', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ resolution }),
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('handleResolutionUpdate: Resolution updated successfully:', result.message);
        
        // Show success message - no restart needed for real-time updates
        if (result.message) {
          warningMessage = result.message;
          // Clear message after a few seconds
          setTimeout(() => {
            warningMessage = '';
          }, 3000);
        }
      } else {
        const result = await response.json();
        console.error('handleResolutionUpdate: Failed to update resolution:', result.detail);
        warningMessage = 'Failed to update resolution: ' + result.detail;
      }
    } catch (error: unknown) {
      console.error('handleResolutionUpdate: Failed to update resolution:', error);
      warningMessage = 'Failed to update resolution: ' + (error instanceof Error ? error.message : String(error));
    }
  }

  function toggleQueueChecker(start: boolean) {
    queueCheckerRunning = start && maxQueueSize > 0;
    if (start) {
      getQueueSize();
    }
  }
  
  async function getQueueSize() {
    if (!queueCheckerRunning) {
      return;
    }
    
    try {
      const response = await fetch('/api/queue');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      currentQueueSize = data.queue_size;
    } catch (error) {
      console.error('Failed to get queue size:', error);
      // Don't show error to user for queue size, just log it
      // This is a background operation that shouldn't interrupt the main flow
    }
    
    setTimeout(getQueueSize, 10000);
  }

  function getSreamdata() {
    if (isImageMode) {
      return [getPipelineValues(), $onFrameChangeStore?.blob];
    } else {
      return [$deboucedPipelineValues];
    }
  }

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: if ($lcmLiveStatus === LCMLiveStatus.DISCONNECTED) {
    disabled = false;
  }
  $: if ($lcmLiveStatus === LCMLiveStatus.TIMEOUT) {
    warningMessage = 'Session timed out. Please try again.';
  }
  
  // Watch for resolution changes
  let previousResolution: string = '';
  $: {
    if ($pipelineValues.resolution && $pipelineValues.resolution !== previousResolution && previousResolution !== '') {
      previousResolution = $pipelineValues.resolution;
      handleResolutionUpdate($pipelineValues.resolution);
    } else if ($pipelineValues.resolution && previousResolution === '') {
      previousResolution = $pipelineValues.resolution;
    }
  }
  
  let disabled = false;
  async function toggleLcmLive() {
    try {
      if (!isLCMRunning) {
        if (isImageMode) {
          await mediaStreamActions.start();
        }
        disabled = true;
        await lcmLiveActions.start(getSreamdata);
        disabled = false;
        toggleQueueChecker(false);
      } else {
        if (isImageMode) {
          mediaStreamActions.stop();
        }
        lcmLiveActions.stop();
        toggleQueueChecker(true);
      }
    } catch (e) {
      warningMessage = e instanceof Error ? e.message : '';
      disabled = false;
      toggleQueueChecker(true);
    }
  }

  async function updateFPS() {
    try {
      const response = await fetch('/api/fps');
      const data = await response.json();
      fps = data.fps;
    } catch (error) {
      console.error('updateFPS: Failed to fetch FPS:', error);
    }
  }

  async function getSnapshotParams() {
    // Fetch live preprocessor params for each controlnet (feedback_strength etc.)
    const cnCount = controlnetInfo?.controlnets?.length ?? 0;
    const preprocessorParamsList: any[] = await Promise.all(
      Array.from({ length: cnCount }, (_, i) =>
        fetch(`/api/preprocessors/current-params/${i}`)
          .then(r => r.ok ? r.json() : {})
          .catch(() => ({}))
      )
    );

    // Fetch style image data for self-contained snapshot
    let styleImageData: any = {};
    try {
      const imgRes = await fetch('/api/ipadapter/style-images-data');
      if (imgRes.ok) styleImageData = await imgRes.json();
    } catch (_) {}

    // Fetch live blending config — page-level promptBlendingConfig may be stale
    // if the user edited prompts in the UI (those edits go to backend, not back to page state)
    let blendingData: any = {};
    try {
      const bRes = await fetch('/api/blending/current');
      if (bRes.ok) blendingData = await bRes.json();
    } catch (_) {}

    return {
      t_index_list: tIndexList,
      guidance_scale: guidanceScale,
      delta: delta,
      num_inference_steps: numInferenceSteps,
      seed: seed,
      negative_prompt: negativePrompt,
      normalize_prompt_weights: normalizePromptWeights,
      normalize_seed_weights: normalizeSeedWeights,
      ipadapter_scale: ipadapterScale,
      ipadapter_blend_weight: ipadapterBlendWeight,
      ipadapter_weight_type: ipadapterWeightType,
      feature_bank_enabled: featureBankEnabled,
      feature_bank_weight: featureBankWeight,
      style_image_a_data: styleImageData.style_image_a || null,
      style_image_b_data: styleImageData.style_image_b || null,
      controlnets: (controlnetInfo?.controlnets || []).map((cn: any, i: number) => ({
        conditioning_scale: cn.strength ?? cn.conditioning_scale,
        feedback_strength: preprocessorParamsList[i]?.parameters?.feedback_strength ?? cn.feedback_strength ?? null,
      })),
      prompt_list: blendingData.prompt_blending || promptBlendingConfig || null,
      seed_list: blendingData.seed_blending || seedBlendingConfig || null,
    };
  }

  async function handleRestoreParams(event: CustomEvent) {
    const p = event.detail;
    if (!p) return;
    try {
      const calls: Promise<any>[] = [];

      // Bulk scalar params
      const scalarParams: any = {};
      if (p.t_index_list)              scalarParams.t_index_list      = p.t_index_list;
      if (p.guidance_scale  != null)   scalarParams.guidance_scale    = p.guidance_scale;
      if (p.delta           != null)   scalarParams.delta             = p.delta;
      if (p.num_inference_steps != null) scalarParams.num_inference_steps = p.num_inference_steps;
      if (p.seed            != null)   scalarParams.seed              = p.seed;
      if (p.negative_prompt          != null) scalarParams.negative_prompt           = p.negative_prompt;
      if (p.normalize_prompt_weights != null) scalarParams.normalize_prompt_weights  = p.normalize_prompt_weights;
      if (p.normalize_seed_weights   != null) scalarParams.normalize_seed_weights    = p.normalize_seed_weights;
      if (Object.keys(scalarParams).length > 0) {
        calls.push(fetch('/api/params', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(scalarParams)
        }));
      }

      // IPAdapter scale
      if (p.ipadapter_scale != null) {
        calls.push(fetch('/api/ipadapter/update-scale', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scale: p.ipadapter_scale })
        }));
      }

      // IPAdapter blend weight
      if (p.ipadapter_blend_weight != null) {
        calls.push(fetch('/api/ipadapter/blend-weight', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ blend_weight: p.ipadapter_blend_weight })
        }));
      }

      // IPAdapter weight type
      if (p.ipadapter_weight_type != null) {
        calls.push(fetch('/api/ipadapter/update-weight-type', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ weight_type: p.ipadapter_weight_type })
        }));
      }

      // Feature bank
      if (p.feature_bank_enabled != null || p.feature_bank_weight != null) {
        const fb: any = {};
        if (p.feature_bank_enabled != null) fb.enabled = p.feature_bank_enabled;
        if (p.feature_bank_weight  != null) fb.weight  = p.feature_bank_weight;
        calls.push(fetch('/api/feature-bank', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(fb)
        }));
      }

      // Style images — decode base64 and re-upload
      function base64ToFormData(b64: string): FormData {
        const bytes = atob(b64);
        const ab = new ArrayBuffer(bytes.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < bytes.length; i++) ia[i] = bytes.charCodeAt(i);
        const blob = new Blob([ab], { type: 'image/jpeg' });
        const fd = new FormData();
        fd.append('file', blob, 'style_image.jpg');
        return fd;
      }
      if (p.style_image_a_data) {
        calls.push(fetch('/api/ipadapter/upload-style-image', { method: 'POST', body: base64ToFormData(p.style_image_a_data) }));
      }
      if (p.style_image_b_data) {
        calls.push(fetch('/api/ipadapter/upload-style-image-b', { method: 'POST', body: base64ToFormData(p.style_image_b_data) }));
      }

      // ControlNet strengths + feedback_strength
      (p.controlnets || []).forEach((cn: any, i: number) => {
        if (cn.conditioning_scale != null) {
          calls.push(fetch('/api/controlnet/update-strength', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index: i, strength: cn.conditioning_scale })
          }));
        }
        if (cn.feedback_strength != null) {
          calls.push(fetch('/api/preprocessors/update-params', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ controlnet_index: i, preprocessor_params: { feedback_strength: cn.feedback_strength } })
          }));
        }
      });

      // Prompt blending
      if (p.prompt_list?.length) {
        calls.push(fetch('/api/blending', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt_list: p.prompt_list, prompt_interpolation_method: 'slerp' })
        }));
      }

      // Seed blending
      if (p.seed_list?.length) {
        calls.push(fetch('/api/blending', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ seed_list: p.seed_list, seed_interpolation_method: 'linear' })
        }));
      }

      await Promise.all(calls);

      // Directly update all Svelte state from params — don't rely on getSettings()
      // round-tripping through the backend config dicts which may not be updated yet.
      if (p.t_index_list)              tIndexList          = p.t_index_list;
      if (p.guidance_scale  != null)   guidanceScale       = p.guidance_scale;
      if (p.delta           != null)   delta               = p.delta;
      if (p.num_inference_steps != null) numInferenceSteps = p.num_inference_steps;
      if (p.seed            != null)   seed                = p.seed;
      if (p.ipadapter_scale        != null) ipadapterScale        = p.ipadapter_scale;
      if (p.ipadapter_blend_weight != null) ipadapterBlendWeight  = p.ipadapter_blend_weight;
      if (p.ipadapter_weight_type  != null) ipadapterWeightType   = p.ipadapter_weight_type;
      if (p.feature_bank_enabled   != null) featureBankEnabled    = p.feature_bank_enabled;
      if (p.feature_bank_weight    != null) featureBankWeight     = p.feature_bank_weight;
      if (p.negative_prompt          != null) negativePrompt          = p.negative_prompt;
      if (p.normalize_prompt_weights != null) normalizePromptWeights  = p.normalize_prompt_weights;
      if (p.normalize_seed_weights   != null) normalizeSeedWeights    = p.normalize_seed_weights;
      // Refresh ipadapter info so style_image_path picks up the newly uploaded A image
      if (p.style_image_a_data || p.style_image_b_data) {
        try {
          const s = await fetch('/api/settings');
          if (s.ok) {
            const sd = await s.json();
            ipadapterInfo = sd.ipadapter || ipadapterInfo;
          }
        } catch (_) {}
      }
      // Set preview URLs directly from base64 data so previews show immediately
      if (p.style_image_a_data) {
        ipadapterStyleImageAUrl = 'data:image/jpeg;base64,' + p.style_image_a_data;
      }
      if (p.style_image_b_data) {
        ipadapterStyleImageBUrl = '/api/ipadapter/uploaded-style-image-b?' + Date.now();
      }

      // Update controlnet strengths in controlnetInfo so sliders reflect restored values
      if (p.controlnets?.length && controlnetInfo?.controlnets?.length) {
        const updatedCNs = controlnetInfo.controlnets.map((cn: any, i: number) => {
          const restored = p.controlnets[i];
          if (!restored) return cn;
          return {
            ...cn,
            ...(restored.conditioning_scale != null ? { strength: restored.conditioning_scale } : {})
          };
        });
        controlnetInfo = { ...controlnetInfo, controlnets: updatedCNs };
      }

      // Prompts and seeds — set config and force BlendingControl re-init
      if (p.prompt_list?.length) promptBlendingConfig = p.prompt_list;
      if (p.seed_list?.length)   seedBlendingConfig   = p.seed_list;
      blendingResetKey++;
      controlnetResetKey++;
    } catch (e) {
      console.error('handleRestoreParams: failed:', e);
    }
  }

  async function refreshBlendingConfigs() {
    try {
      const response = await fetch('/api/blending/current');
      const data = await response.json();
      
      if (data.prompt_blending) {
        promptBlendingConfig = data.prompt_blending;
        console.log('refreshBlendingConfigs: Updated prompt blending:', promptBlendingConfig);
      }
      
      if (data.seed_blending) {
        seedBlendingConfig = data.seed_blending;
        console.log('refreshBlendingConfigs: Updated seed blending:', seedBlendingConfig);
      }
      
      if (data.normalize_prompt_weights !== undefined) {
        normalizePromptWeights = data.normalize_prompt_weights;
      }
      
      if (data.normalize_seed_weights !== undefined) {
        normalizeSeedWeights = data.normalize_seed_weights;
      }
      
      console.log('refreshBlendingConfigs: Blending configs refreshed');
    } catch (error) {
      console.error('refreshBlendingConfigs: Failed to refresh blending configs:', error);
    }
  }

  // Pipeline configuration upload
  let fileInput: HTMLInputElement;
  let uploading = false;
  let uploadStatus = '';

  async function uploadConfig() {
    if (!fileInput.files || fileInput.files.length === 0) {
      uploadStatus = 'Please select a YAML file';
      return;
    }

    const file = fileInput.files[0];
    if (!file.name.endsWith('.yaml') && !file.name.endsWith('.yml')) {
      uploadStatus = 'Please select a YAML file (.yaml or .yml)';
      return;
    }

    uploading = true;
    uploadStatus = 'Uploading configuration...';

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/controlnet/upload-config', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        uploadStatus = isLCMRunning
          ? 'Configuration loaded! Pipeline rebuilding — stream will resume in ~30–60s.'
          : 'Configuration loaded! Pipeline will rebuild when you start streaming.';
        fileInput.value = '';
        
        // Update ControlNet info
        if (result.controlnet) {
          controlnetInfo = result.controlnet;
        }
        
        // Update IPAdapter info
        if (result.ipadapter) {
          ipadapterInfo = result.ipadapter;
          ipadapterScale = result.ipadapter.scale || 1.0;
        }
        
        // Update streaming parameters
        if (result.t_index_list) {
          tIndexList = [...result.t_index_list];
        }
        if (result.guidance_scale !== undefined) {
          guidanceScale = result.guidance_scale;
        }
        if (result.delta !== undefined) {
          delta = result.delta;
        }
        if (result.num_inference_steps !== undefined) {
          numInferenceSteps = result.num_inference_steps;
        }
        if (result.seed !== undefined) {
          seed = result.seed;
        }
        
        // Update normalization settings
        if (result.normalize_prompt_weights !== undefined) {
          normalizePromptWeights = result.normalize_prompt_weights;
        }
        if (result.normalize_seed_weights !== undefined) {
          normalizeSeedWeights = result.normalize_seed_weights;
        }
        
        // Update blending configurations
        if (result.prompt_blending) {
          promptBlendingConfig = result.prompt_blending;
          showPromptBlending = true;  // Auto-expand if config has blending data
          console.log('uploadConfig: Updated prompt blending config:', promptBlendingConfig);
        }
        if (result.seed_blending) {
          seedBlendingConfig = result.seed_blending;
          console.log('uploadConfig: Updated seed blending config:', seedBlendingConfig);
        }
        
        // Update main prompt if config prompt is available
        if (result.config_prompt) {
          pipelineValues.update(values => ({
            ...values,
            prompt: result.config_prompt
          }));
        }
        
        // Update resolution if config resolution is available
        if (result.current_resolution) {
          pipelineValues.update(values => ({
            ...values,
            resolution: result.current_resolution
          }));
          console.log('uploadConfig: Updated resolution to:', result.current_resolution);
        }
        
        setTimeout(() => {
          uploadStatus = '';
        }, 4000);
      } else {
        uploadStatus = `Error: ${result.detail || 'Failed to load configuration'}`;
      }
    } catch (error) {
      console.error('uploadConfig: Upload failed:', error);
      uploadStatus = 'Upload failed. Please try again.';
    } finally {
      uploading = false;
    }
  }

  function selectFile() {
    fileInput.click();
  }

  async function setUpscaleFactor(factor: number) {
    try {
      await fetch('/api/upscale', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ factor })
      });
      upscaleFactor = factor;
    } catch (e) {
      console.error('setUpscaleFactor: failed:', e);
    }
  }


  let negativePromptDebounceTimer: ReturnType<typeof setTimeout> | null = null;
  function handleNegativePromptInput() {
    if (negativePromptDebounceTimer) clearTimeout(negativePromptDebounceTimer);
    negativePromptDebounceTimer = setTimeout(async () => {
      try {
        await fetch('/api/params', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ negative_prompt: negativePrompt })
        });
      } catch (e) {
        console.error('handleNegativePromptInput: failed:', e);
      }
    }, 400);
  }
</script>

<svelte:head>
  <script
    src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js"
  ></script>
</svelte:head>

<main class="h-screen flex flex-col overflow-hidden">
  <Warning bind:message={warningMessage}></Warning>
  
  <!-- Header Section -->
  <header class="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex-shrink-0">
    <div class="flex items-center justify-between">
      <div class="flex-1">
        {#if pageContent}
          <div class="text-center">
            {@html pageContent}
          </div>
        {/if}
        {#if maxQueueSize > 0}
          <p class="text-sm text-center mt-2">
            There are <span id="queue_size" class="font-bold">{currentQueueSize}</span>
            user(s) sharing the same GPU, affecting real-time performance. Maximum queue size is {maxQueueSize}.
            <a
              href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
              target="_blank"
              class="text-blue-500 underline hover:no-underline">Duplicate</a
            > and run it on your own GPU.
          </p>
        {/if}
      </div>
      
      <!-- Pipeline Configuration and Main Controls -->
      <div class="flex items-center gap-4">
        <!-- Pipeline Configuration -->
        <div class="flex items-center gap-2">
          <Button on:click={selectFile} disabled={uploading} classList="text-sm px-3 py-2">
            {uploading ? 'Uploading...' : 'Load YAML Config'}
          </Button>
        </div>
        
        <input
          bind:this={fileInput}
          type="file"
          accept=".yaml,.yml"
          class="hidden"
          on:change={uploadConfig}
        />
        
        <!-- Main Control Button -->
        <Button on:click={toggleLcmLive} {disabled} classList={'text-lg px-6 py-3 font-semibold'}>
          {#if isLCMRunning}
            Stop Stream
          {:else}
            Start Stream
          {/if}
        </Button>
      </div>
    </div>
    
    {#if uploadStatus}
      <div class="mt-2 text-center">
        <p class="text-sm {uploadStatus.includes('Error') || uploadStatus.includes('Please') ? 'text-red-600' : 'text-green-600'}">
          {uploadStatus}
        </p>
      </div>
    {/if}
  </header>

  {#if pipelineParams}
    <!-- Main Content Grid -->
    <div class="flex-1 grid grid-cols-12 gap-4 p-4 overflow-hidden">
      
      <!-- Left Panel - Input and Basic Controls -->
      <div class="col-span-12 lg:col-span-3 flex flex-col gap-4 overflow-hidden">
        <!-- Panel Header -->
        <div class="flex items-center justify-between flex-shrink-0">
          <h2 class="text-lg font-semibold">Input & Controls</h2>
          <button 
            on:click={() => leftPanelCollapsed = !leftPanelCollapsed}
            class="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            {leftPanelCollapsed ? '→' : '←'}
          </button>
        </div>
        
        {#if !leftPanelCollapsed}
          <!-- Fixed Video Input Section (Image Mode Only) -->
          {#if isImageMode}
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 flex-shrink-0">
              <h3 class="text-md font-medium mb-3">Video Input</h3>
              <VirtCamControl />
              <div class="mt-3">
                <VideoInput
                  width={Number(pipelineParams.width.default)}
                  height={Number(pipelineParams.height.default)}
                  {currentResolution}
                />
              </div>
            </div>
          {/if}

          <!-- Scrollable Controls Section -->
          <div class="flex-1 overflow-y-auto space-y-4">
            <!-- Resolution Picker -->
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <button 
                on:click={() => showResolutionPicker = !showResolutionPicker}
                class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
              >
                <h3 class="text-md font-medium">Resolution</h3>
                <span class="text-sm">{showResolutionPicker ? '−' : '+'}</span>
              </button>
              {#if showResolutionPicker}
                <div class="p-4 pt-0">
                  <ResolutionPicker {currentResolution} {pipelineParams} />
                </div>
              {/if}
            </div>

            <!-- Unified Blending Control -->
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <button 
                on:click={() => showPromptBlending = !showPromptBlending}
                class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
              >
                <h3 class="text-md font-medium">Blending Controls</h3>
                <span class="text-sm">{showPromptBlending ? '−' : '+'}</span>
              </button>
              {#if showPromptBlending}
                <div class="p-4 pt-0 space-y-3">
                  <!-- Negative Prompt -->
                  <div>
                    <label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Negative Prompt</label>
                    <textarea
                      bind:value={negativePrompt}
                      on:input={handleNegativePromptInput}
                      placeholder="blurry, low quality, ..."
                      rows="2"
                      class="w-full p-2 border rounded resize-none text-xs dark:bg-gray-700 dark:border-gray-600 text-gray-800 dark:text-gray-200"
                    ></textarea>
                  </div>
                  <BlendingControl
                    {promptBlendingConfig}
                    {seedBlendingConfig}
                    {normalizePromptWeights}
                    {normalizeSeedWeights}
                    currentPrompt={$pipelineValues.prompt}
                    resetKey={blendingResetKey}
                  />
                </div>
              {/if}
            </div>

            <!-- Input Control Section -->
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <InputControl />
            </div>
          </div>
        {/if}
      </div>

      <!-- Center Panel - Main Image Output -->
      <div class="col-span-12 lg:col-span-6 flex flex-col">
        <div class="flex-1 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 flex flex-col">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-lg font-semibold">Generated Output</h2>
            <div class="flex items-center gap-4">
              {#if isLCMRunning}
                <div class="flex items-center gap-2 px-3 py-1 bg-green-100 dark:bg-green-900 rounded-lg">
                  <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span class="text-sm font-medium text-green-800 dark:text-green-200">
                    {fps.toFixed(1)} FPS
                  </span>
                </div>
              {/if}
              <div class="text-sm text-gray-600 dark:text-gray-400">
                Status: {isLCMRunning ? 'Streaming' : 'Stopped'}
              </div>
            </div>
          </div>
          <div class="flex-1 flex items-center justify-center">
            <div class="w-full">
              <ImagePlayer {currentResolution} on:paramsLoaded={handleRestoreParams} collectParams={getSnapshotParams} />
            </div>
          </div>
        </div>
      </div>

      <!-- Right Panel - Advanced Controls -->
      <div class="col-span-12 lg:col-span-3 flex flex-col gap-4 overflow-y-auto">
        <!-- Panel Header -->
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold">Advanced Settings</h2>
          <button 
            on:click={() => rightPanelCollapsed = !rightPanelCollapsed}
            class="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            {rightPanelCollapsed ? '←' : '→'}
          </button>
        </div>
        
        {#if !rightPanelCollapsed}
          <IPAdapterConfig
            {ipadapterInfo}
            bind:currentScale={ipadapterScale}
            bind:currentBlendWeight={ipadapterBlendWeight}
            bind:currentStyleImage={ipadapterStyleImageAUrl}
            bind:currentStyleImageB={ipadapterStyleImageBUrl}
            currentWeightType={ipadapterWeightType}
          ></IPAdapterConfig>

          <ControlNetConfig
            {controlnetInfo}
            {tIndexList}
            bind:guidanceScale
            bind:delta
            bind:numInferenceSteps
            resetKey={controlnetResetKey}
            on:controlnetUpdated={handleControlNetUpdate}
            on:tIndexListUpdated={(e) => handleTIndexListUpdate(e.detail)}
            on:controlnetConfigChanged={getSettings}
          ></ControlNetConfig>

          <FeatureBankConfig
            bind:enabled={featureBankEnabled}
            bind:weight={featureBankWeight}
          ></FeatureBankConfig>

          <LoraConfig
            initialLoras={loraList}
            {loraDir}
            getStreamData={getSreamdata}
          ></LoraConfig>

          <!-- Output Upscale -->
          <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-3">
            <div class="flex items-center justify-between">
              <span class="text-sm font-medium">Output Upscale</span>
              <div class="flex gap-1">
                {#each [1, 2, 4] as f}
                  <button
                    on:click={() => setUpscaleFactor(f)}
                    class="px-2 py-1 text-xs rounded border transition-colors {upscaleFactor === f
                      ? 'bg-blue-500 text-white border-blue-500'
                      : 'border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700'}"
                  >{f === 1 ? 'Off' : `${f}×`}</button>
                {/each}
              </div>
            </div>
          </div>
        {/if}
      </div>
    </div>
  {:else if apiError}
    <!-- API Error -->
    <div class="flex-1 flex flex-col items-center justify-center gap-6 py-48 text-center">
      <div>
        <h2 class="text-2xl font-bold text-red-600 mb-2">API Connection Failed</h2>
        <p class="text-gray-600 dark:text-gray-400 mb-4 max-w-md">
          {apiError}
        </p>
        <Button 
          on:click={retryConnection} 
          disabled={isRetrying} 
          classList="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2"
        >
          {#if isRetrying}
            <Spinner classList="w-4 h-4 mr-2 animate-spin" />
            Retrying...
          {:else}
            Retry Connection
          {/if}
        </Button>
      </div>
    </div>
  {:else}
    <!-- Loading State -->
    <div class="flex-1 flex items-center justify-center">
      <div class="flex items-center gap-3 text-2xl">
        <Spinner classList={'animate-spin opacity-50'} />
        <p>Loading iVizDiff...</p>
      </div>
    </div>
  {/if}
</main>

<style lang="postcss">
  @reference "tailwindcss";
  
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }
  
  /* Custom scrollbar styling */
  :global(.overflow-y-auto::-webkit-scrollbar) {
    width: 6px;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-track) {
    @apply bg-gray-100 dark:bg-gray-800;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb) {
    @apply bg-gray-300 dark:bg-gray-600 rounded-full;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb:hover) {
    @apply bg-gray-400 dark:bg-gray-500;
  }
</style>
