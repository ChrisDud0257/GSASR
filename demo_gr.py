

import torch
import numpy as np
import gradio as gr
from PIL import Image
import math
import torch.nn.functional as F
import os
import tempfile
import time
import threading

from utils.hatropeamp import HATNOUP_ROPE_AMP
from utils.fea2gsropeamp import Fea2GS_ROPE_AMP
from utils.edsrbaseline import EDSRNOUP
from utils.hatropeamp import HATNOUP_ROPE_AMP
from utils.rdn import RDNNOUP
from utils.swinir import SwinIRNOUP
from utils.fea2gsropeamp import Fea2GS_ROPE_AMP
from utils.gaussian_splatting import generate_2D_gaussian_splatting_step
from utils.split_and_joint_image import split_and_joint_image
from huggingface_hub import hf_hub_download
import subprocess



# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global stop flag for interrupting inference
stop_inference = False
inference_lock = threading.Lock()

def load_model(
    pretrained_model_name_or_path: str = "mutou0308/GSASR",
    model_name: str = "HATL_SA1B",
    device: str | torch.device = "cuda"
):
    enc_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=os.path.join('GSASR_enhenced_ultra', model_name, 'encoder.pth')
        )
    dec_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=os.path.join('GSASR_enhenced_ultra', model_name, 'decoder.pth')
        )

    enc_weight = torch.load(enc_path, weights_only=True)['params_ema']
    dec_weight = torch.load(dec_path, weights_only=True)['params_ema']
    
    if model_name in ['EDSR_DIV2K', 'EDSR_DF2K']:
        encoder = EDSRNOUP()
        decoder = Fea2GS_ROPE_AMP()
    elif model_name in ['RDN_DIV2K', 'RDN_DF2K']:
        encoder = RDNNOUP()
        decoder = Fea2GS_ROPE_AMP(num_crossattn_blocks = 2)
    elif model_name in ['SwinIR_DIV2K', 'SwinIR_DF2K']:
        encoder = SwinIRNOUP()
        decoder = Fea2GS_ROPE_AMP(num_crossattn_blocks=2, num_crossattn_layers=4, num_gs_seed=256, window_size=16)
    elif model_name in ['HATL_SA1B']:
        encoder = HATNOUP_ROPE_AMP()
        decoder = Fea2GS_ROPE_AMP(channel=192, num_crossattn_blocks=4, num_crossattn_layers=4, num_selfattn_blocks=8, num_selfattn_layers=6,
                                  num_gs_seed=256, window_size=16)
    else:
        raise ValueError(f"args.model-{model_name} must be in ['EDSR_DIV2K', 'EDSR_DF2K', 'RDN_DIV2K', 'RDN_DF2K', 'SwinIR_DIV2K', 'SwinIR_DF2K', 'HATL_SA1B']")

    encoder.load_state_dict(enc_weight, strict=True)
    decoder.load_state_dict(dec_weight, strict=True)
    encoder.eval()
    decoder.eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    return encoder, decoder


def preprocess(x, denominator=16):
    """Preprocess image to ensure dimensions are multiples of denominator"""
    _, c, h, w = x.shape
    if h % denominator > 0:
        pad_h = denominator - h % denominator
    else:
        pad_h = 0
    if w % denominator > 0:
        pad_w = denominator - w % denominator
    else:
        pad_w = 0
    x_new = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
    return x_new

def postprocess(x, gt_size_h, gt_size_w):
    """Post-process by cropping to target size"""
    x_new = x[:, :, :gt_size_h, :gt_size_w]
    return x_new

def should_use_tile(image_height, image_width, threshold=1024):
    """Determine if tile processing should be used based on image resolution"""
    return max(image_height, image_width) > threshold

def set_stop_flag():
    """Set the global stop flag to interrupt inference"""
    global stop_inference
    with inference_lock:
        stop_inference = True
    return "🛑 Stopping inference...", gr.update(interactive=False)

def reset_stop_flag():
    """Reset the global stop flag"""
    global stop_inference
    with inference_lock:
        stop_inference = False

def check_stop_flag():
    """Check if inference should be stopped"""
    global stop_inference
    with inference_lock:
        return stop_inference
    
def super_resolution_inference(image, scale=4.0):
    """Super-resolution inference function with automatic tile processing"""

    if image is None:
        return None, "Please upload an image", None
    
    # Load model
    encoder, decoder = load_model(model_name="HATL_SA1B")

    # Reset stop flag at the beginning
    reset_stop_flag()
    
    # Fixed parameters
    tile_overlap = 16  # Fixed overlap size
    crop_size = 8     # Fixed crop size
    tile_size = 1024   # Fixed tile size for large images
    
    try:
        # Check for interruption
        if check_stop_flag():
            return None, "❌ Inference interrupted", None
            
        # Convert PIL image to numpy array
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            img_np = img_np[:, :, [2, 1, 0]]  # RGB to BGR
        
        # Convert to tensor
        img = torch.from_numpy(np.transpose(img_np.astype(np.float32) / 255., (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        
        # Check for interruption
        if check_stop_flag():
            return None, "❌ Inference interrupted", None
        
        # Calculate target size
        gt_size = [math.floor(scale * img.shape[2]), math.floor(scale * img.shape[3])]
        
        # Determine if tile processing should be used
        use_tile = should_use_tile(img.shape[2], img.shape[3])
        
        # Force AMP mixed precision
        with torch.inference_mode():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Check for interruption before main processing
                if check_stop_flag():
                    return None, "❌ Inference interrupted", None
                    
                if use_tile:
                    # Use tile processing
                    assert tile_size % 16 == 0, f"tile_size-{tile_size} must be divisible by 16"
                    assert 2 * tile_overlap < tile_size, f"2 * tile_overlap must be less than tile_size"
                    assert 2 * crop_size <= tile_overlap, f"2 * crop_size must be less than or equal to tile_overlap"
                    
                    with torch.no_grad():
                        output = split_and_joint_image(
                            lq=img, 
                            scale_factor=scale,
                            split_size=tile_size,
                            overlap_size=tile_overlap,
                            model_g=encoder,
                            model_fea2gs=decoder,
                            crop_size=crop_size,
                            scale_modify=torch.tensor([scale, scale]),
                            default_step_size=1.2,
                            cuda_rendering=True,
                            mode='scale_modify',
                            if_dmax=True,
                            dmax_mode='fix',
                            dmax=0.1
                        )
                else:
                    # Direct processing without tiles
                    lq_pad = preprocess(img, 16)  # denominator=16 for HATL
                    gt_size_pad = torch.tensor([math.floor(scale * lq_pad.shape[2]), 
                                            math.floor(scale * lq_pad.shape[3])])
                    gt_size_pad = gt_size_pad.unsqueeze(0)
                    
                    with torch.no_grad():
                        # Check for interruption before encoder
                        if check_stop_flag():
                            return None, "❌ Inference interrupted", None
                            
                        # Encoder output
                        encoder_output = encoder(lq_pad)  # b,c,h,w
                        
                        # Check for interruption before decoder
                        if check_stop_flag():
                            return None, "❌ Inference interrupted", None
                            
                        scale_vector = torch.tensor(scale, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        # Decoder output
                        batch_gs_parameters = decoder(encoder_output, scale_vector)
                        gs_parameters = batch_gs_parameters[0, :]
                        
                        # Check for interruption before gaussian rendering
                        if check_stop_flag():
                            return None, "❌ Inference interrupted", None
                        
                        # Gaussian rendering
                        b_output = generate_2D_gaussian_splatting_step(
                            gs_parameters=gs_parameters,
                            sr_size=gt_size_pad[0],
                            scale=scale,
                            sample_coords=None,
                            scale_modify=torch.tensor([scale, scale]),
                            default_step_size=1.2,
                            cuda_rendering=True,
                            mode='scale_modify',
                            if_dmax=True,
                            dmax_mode='fix',
                            dmax=0.1
                        )
                        output = b_output.unsqueeze(0)
        
        # Check for interruption before post-processing
        if check_stop_flag():
            return None, "❌ Inference interrupted", None
        
        # Post-processing
        output = postprocess(output, gt_size[0], gt_size[1])
        
        # Convert back to PIL image format
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # BGR to RGB
        output = (output * 255.0).round().astype(np.uint8)
        
        # Convert to PIL image
        output_pil = Image.fromarray(output)
        
        # Generate result information
        original_size = f"{img.shape[3]}x{img.shape[2]}"
        output_size = f"{output.shape[1]}x{output.shape[0]}"
        tile_info = f"Tile processing enabled (size: {tile_size})" if use_tile else "Direct processing (no tiles)"
        result_info = f"✅ Processing completed successfully!\nOriginal size: {original_size}\nSuper-resolution size: {output_size}\nScale factor: {scale:.2f}x\nProcessing mode: {tile_info}\nAMP acceleration: Force enabled\nOverlap size: {tile_overlap}\nCrop size: {crop_size}"
        
        return output_pil, result_info, output_pil
        
    except Exception as e:
        if check_stop_flag():
            return None, "❌ Inference interrupted", None
        return None, f"❌ Error during processing: {str(e)}", None

def predict(image, scale):
    """Gradio prediction function"""
    output_image, info, download_image = super_resolution_inference(image, scale)
    
    # If processing successful, save image for download
    if output_image is not None:
        # Create temporary filename
        timestamp = int(time.time())
        temp_filename = f"GSASR_SR_result_{scale}x_{timestamp}.png"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        # Save image
        output_image.save(temp_path, "PNG")
        
        return output_image, temp_path, "✅ Ready", gr.update(interactive=True)
    else:
        return output_image, None, info if info else "❌ Processing failed", gr.update(interactive=True)

# Create Gradio interface
with gr.Blocks(title="🚀 GSASR (2D Gaussian Splatting Super-Resolution)") as demo:
    gr.Markdown("# **🚀 GSASR (Generalized and efficient 2d gaussian splatting for arbitrary-scale super-resolution)**")
    gr.Markdown("Official demo for GSASR. Please refer to our [paper](https://arxiv.org/pdf/2501.06838), [project page](https://mt-cly.github.io/GSASR.github.io/), and [github](https://github.com/ChrisDud0257/GSASR) for more details.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            
            # Scale parameters
            with gr.Group():
                gr.Markdown("### SR Scale")
                scale_slider = gr.Slider(minimum=1.0, maximum=30.0, value=4.0, step=0.1, label="SR Scale")
            
            # Control buttons
            with gr.Row():
                submit_btn = gr.Button("🚀 Start Super-Resolution", variant="primary")
                stop_btn = gr.Button("🛑 Stop Inference", variant="stop")
        
        with gr.Column():
            output_image = gr.Image(type="pil", label="Super-Resolution Result")
            
            # Status display
            status_text = gr.Textbox(label="Status", value="✅ Ready", interactive=False)
            
            # Download component
            with gr.Group():
                gr.Markdown("### 📥 Download Super-Resolution Result")
                download_btn = gr.File(visible=True)
    
    # Event handlers
    submit_event = submit_btn.click(
        fn=predict,
        inputs=[input_image, scale_slider],
        outputs=[output_image, download_btn, status_text, stop_btn]
    )
    
    stop_btn.click(
        fn=set_stop_flag,
        inputs=[],
        outputs=[status_text, stop_btn],
        cancels=[submit_event]
    )
    
    # Example images
    gr.Markdown("### 📚 Example Images")
    gr.Markdown("Try these examples with different scales:")
    
    gr.Examples(
        examples=[
            ["assets/0846x4.png", 1.5],
            ["assets/0892x4.png", 2.8],
            ["assets/0873x4_cropped_120x120.png", 30.0]
        ],
        inputs=[input_image, scale_slider],
        examples_per_page=3,
        cache_examples=False,
        label="Examples"
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0") 