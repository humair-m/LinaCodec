import torch
from linacodec.vocoder.vocos import Vocos
from huggingface_hub import snapshot_download
from .model import LinaCodecModel
from .util import load_audio, load_vocoder, vocode

class LinaCodec:
    def __init__(self, model_path=None, device=None):
        """
        Initialize LinaCodec with device support.
        
        Args:
            model_path: Path to model weights, or None to download from HuggingFace
            device: Device to use ('cuda', 'cpu', or torch.device). 
                   If None, automatically selects cuda if available.
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        print(f"Using device: {self.device}")

        ## download from hf
        if model_path is None:
            model_path = snapshot_download("YatharthS/LinaCodec")

        ## loads linacodec model
        model = LinaCodecModel.from_pretrained(
            config_path=f"{model_path}/config.yaml", 
            weights_path=f'{model_path}/model.safetensors'
        ).eval().to(self.device)

        ## loads distilled wavlm model
        model.load_distilled_wavlm(f"{model_path}/wavlm_encoder.pth", device=self.device)
        model.distilled_layers = [6, 9]

        ## loads vocoder
        vocos = Vocos.from_hparams(f'{model_path}/vocoder/config.yaml').to(self.device)
        vocos.load_state_dict(torch.load(
            f'{model_path}/vocoder/pytorch_model.bin',
            map_location=self.device
        ))

        self.model = model
        self.vocos = vocos

    @torch.no_grad()
    def encode(self, audio_path):
        """encodes audio into discrete content tokens at a rate of 12.5 t/s or 25 t/s and 128 dim global embedding"""
        ## load audio and extract features
        audio = load_audio(audio_path, sample_rate=self.model.config.sample_rate).to(self.device)
        features = self.model.encode(audio)
        return features.content_token_indices, features.global_embedding

    @torch.no_grad()
    def decode(self, content_tokens, global_embedding):
        """decodes tokens and embedding into 48khz waveform"""
        # Ensure inputs are on correct device
        content_tokens = content_tokens.to(self.device)
        global_embedding = global_embedding.to(self.device)
        
        # Use autocast only if on CUDA
        if self.device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                mel_spectrogram = self.model.decode(
                    content_token_indices=content_tokens, 
                    global_embedding=global_embedding
                )
        else:
            mel_spectrogram = self.model.decode(
                content_token_indices=content_tokens, 
                global_embedding=global_embedding
            )

        ## decode mel spectrogram into 48khz audio
        waveform = vocode(self.vocos, mel_spectrogram.unsqueeze(0))
        return waveform
        
    def convert_voice(self, source_file, reference_file):
        """converts voice timbre, keeping content of source file but timbre of reference file"""
        ## get tokens and embedding
        speech_tokens, _ = self.encode(source_file)
        _, ref_global_embedding = self.encode(reference_file)

        ## decode to audio
        audio = self.decode(speech_tokens, ref_global_embedding)
        return audio

    def to(self, device):
        """Move model to specified device"""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = self.model.to(self.device)
        self.vocos = self.vocos.to(self.device)
        return self


# Usage examples:
if __name__ == "__main__":
    # Auto-detect device (uses GPU if available)
    codec = LinaCodec()
    
    # Force CPU
    codec_cpu = LinaCodec(device='cpu')
 
    converted = codec.convert_voice("source.wav", "reference.wav")
