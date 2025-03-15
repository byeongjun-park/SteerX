import torch


def load_steerx(model, device='cuda'):  # noqa: D103
    '''
    Model Zoo
    - mochi + MonST3R (Text-to-Video-to-4D)
    - hunyuan + MonST3R (Text-to-Video-to-4D)
    - cog + MonST3R (Image-to-Video-to-4D)
    - dimensionx_{left} + MV-DUSt3R (Image-to-MV-to-3D)
    - splatflow (Text-to-3D)
    '''
    
    if model == 'mochi':
        from steerx_diffusers.pipeline_steerx_mochi import MochisteerxPipeline  # noqa: PLC0415
        from steerx_diffusers.geometry_steering.monst_util import GeometryReward  # noqa: PLC0415

        pipe = MochisteerxPipeline.from_pretrained("genmo/mochi-1-preview", variant="bf16", torch_dtype=torch.bfloat16, cache_dir='.cache/models')  # noqa: E501
        reward_fn = GeometryReward(monst3r_ckpt='steerx_ckpt/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')

    elif model == 'hunyuan':
        from diffusers import HunyuanVideoTransformer3DModel  # noqa: PLC0415
        from steerx_diffusers.pipeline_steerx_hunyuan import HunyuanVideosteerxPipeline  # noqa: PLC0415
        from steerx_diffusers.geometry_steering.monst_util import GeometryReward  # noqa: PLC0415

        model_id = "hunyuanvideo-community/HunyuanVideo"
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16, cache_dir='.cache/models'
        )

        pipe = HunyuanVideosteerxPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16, cache_dir='.cache/models')  # noqa: E501
        reward_fn = GeometryReward(monst3r_ckpt='steerx_ckpt/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    
    elif model == 'cog':
        from steerx_diffusers.pipeline_steerx_cogvideox import CogVideoXImageToVideosteerxPipeline  # noqa: PLC0415
        from diffusers.schedulers import CogVideoXDPMScheduler  # noqa: PLC0415
        from steerx_diffusers.geometry_steering.monst_util import GeometryReward  # noqa: PLC0415

        pipe = CogVideoXImageToVideosteerxPipeline.from_pretrained("THUDM/CogVideoX1.5-5b-I2V", torch_dtype=torch.bfloat16, cache_dir='.cache/models')
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")  # note: for stochastic
        reward_fn = GeometryReward(monst3r_ckpt='steerx_ckpt/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')

    elif model == 'dimensionx':
        from steerx_diffusers.pipeline_steerx_cogvideox import CogVideoXImageToVideosteerxPipeline  # noqa: PLC0415
        from diffusers.schedulers import CogVideoXDPMScheduler  # noqa: PLC0415
        from steerx_diffusers.geometry_steering.mvdp_util import GeometryReward  # noqa: PLC0415
        
        pipe = CogVideoXImageToVideosteerxPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16, cache_dir='.cache/models')
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")  # note: for stochastic
        
        pipe.load_lora_weights('steerx_ckpt', weight_name=f"orbit_left_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / 256)

        reward_fn = GeometryReward(mvdust3rp_ckpt='steerx_ckpt/MVDp_s1.pth')

    elif model == 'splatflow':
        from steerx_diffusers.geometry_steering.splatflow_util import GeometryReward  # noqa: PLC0415
        from steerx_diffusers.pipeline_steerx_splatflow import SplatFlowsteerxPipeline  # noqa: PLC0415
        from steerx_diffusers.splatflow.mv_sd3_architecture import MultiViewSD3Transformer  # noqa: PLC0415

        rf_transformer = MultiViewSD3Transformer.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="transformer",
            ignore_mismatched_sizes=True,
            strict=False,
            low_cpu_mem_usage=False,
            cache_dir='.cache/models'
        )
        rf_transformer.adjust_output_input_channel_size(new_in_channels=38)

        rf_transformer = rf_transformer.to(dtype=torch.float16)
        rf_ckpt = torch.load('steerx_ckpt/splatflow_mvrf.pt', map_location="cpu", weights_only=False)['ema']
        rf_transformer.load_state_dict(rf_ckpt)

        pipe = SplatFlowsteerxPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", transformer=rf_transformer, torch_dtype=torch.float16, cache_dir='.cache/models')  # noqa: E501
        reward_fn = GeometryReward(decoder_ckpt='steerx_ckpt/splatflow_decoder.pt').to('cuda')

    else:
        raise ValueError(f"Model {model} not recognized")
    
    pipe.enable_model_cpu_offload()
    return pipe, reward_fn.to(device)
