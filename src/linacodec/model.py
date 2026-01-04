# Add this modified method to LinaCodecModel class in model.py

def load_distilled_wavlm(self, path: str, device=None):
    """
    Loads distilled wavlm model, 970m params --> 250m params
    
    Args:
        path: Path to checkpoint file
        device: Device to load model to. If None, uses cuda if available, else cpu
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    ckpt = torch.load(path, map_location=device)
    wavlm_model = wav2vec2_model(**ckpt["config"])
    result = wavlm_model.load_state_dict(ckpt["state_dict"], strict=False)
    self.wavlm_model = wavlm_model.to(device)
    self.distilled_layers = [6, 8]  # can set custom, 6-8 seems best


# Also update the encode method to use proper device-aware autocast:

@torch.inference_mode()
def encode(self, waveform: torch.Tensor, return_content: bool = True, return_global: bool = True) -> LinaCodecFeatures:
    """Extract content and/or global features from audio using LinaCodec model."""
    audio_length = waveform.size(0)
    padding = self._calculate_waveform_padding(audio_length)
    local_ssl_features, global_ssl_features = self.forward_ssl_features(waveform.unsqueeze(0), padding=padding)

    result = LinaCodecFeatures()
    
    # Determine device and dtype for autocast
    device_type = waveform.device.type
    use_autocast = device_type == 'cuda'
    
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_autocast):
        if return_content:
            content_embedding, token_indices, _, _ = self.forward_content(local_ssl_features)
            result.content_embedding = content_embedding.squeeze(0)
            result.content_token_indices = token_indices.squeeze(0)

        if return_global:
            global_embedding = self.forward_global(global_ssl_features)
            result.global_embedding = global_embedding.squeeze(0)

    return result


@torch.inference_mode()
def decode(
    self,
    global_embedding: torch.Tensor,
    content_token_indices: torch.Tensor | None = None,
    content_embedding: torch.Tensor | None = None,
    target_audio_length: int | None = None,
) -> torch.Tensor:
    """Synthesize audio from content and global features."""
    # Obtain content embedding if not provided
    if content_embedding is None:
        if content_token_indices is None:
            raise ValueError("Either content_token_indices or content_embedding must be provided.")
        content_embedding = self.decode_token_indices(content_token_indices)

    if target_audio_length is None:
        seq_len = content_embedding.size(0)
        target_audio_length = self._calculate_original_audio_length(seq_len)

    # Device-aware autocast
    device_type = content_embedding.device.type
    use_autocast = device_type == 'cuda'
    
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_autocast):
        mel_length = self._calculate_target_mel_length(target_audio_length)
        content_embedding = content_embedding.unsqueeze(0)
        global_embedding = global_embedding.unsqueeze(0)
        mel_spectrogram = self.forward_mel(content_embedding, global_embedding, mel_length=mel_length)

    return mel_spectrogram.squeeze(0)
