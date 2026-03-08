"""
Export CompressAI model to ONNX format
This creates an ONNX model that can be used with ONNX Runtime for 2-3x speedup
"""
import argparse
import torch
import onnx
from pathlib import Path
from compressai.zoo import models
import compressai

def export_model_to_onnx(model_name="bmshj2018-factorized", quality=1, metric="mse", 
                         output_path="model_onnx", device="cpu"):
    """
    Export CompressAI model to ONNX format
    
    Note: CompressAI models are complex and may not fully export to ONNX.
    We'll export the encoder and decoder separately.
    """
    
    print(f"{'='*60}")
    print(f"ONNX Export Tool")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Quality: {quality}")
    print(f"Metric: {metric}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Load pretrained model
    print("Loading pretrained model...")
    net = models[model_name](quality=quality, metric=metric, pretrained=True).to(device).eval()
    print(f"✓ Model loaded\n")
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy input
    batch_size = 1
    channels = 3
    height = 256
    width = 256
    dummy_input = torch.randn(batch_size, channels, height, width, device=device)
    
    print("Attempting full model export...")
    full_export_success = False
    try:
        # Try to export the full model
        with torch.no_grad():
            torch.onnx.export(
                net,
                dummy_input,
                str(output_dir / f"{model_name}_q{quality}_full.onnx"),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}
                }
            )
        print(f"✓ Full model exported successfully!")
        
        # Verify the export
        onnx_model = onnx.load(str(output_dir / f"{model_name}_q{quality}_full.onnx"))
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verification passed\n")
        full_export_success = True
        
    except Exception as e:
        print(f"✗ Full model export failed: {e}\n")
        print("This is common with CompressAI models due to complex entropy coding.")
    
    # Always try to export encoder/decoder separately (needed for compress())
    print("Exporting encoder and decoder separately (needed for compression)...\n")
    
    # Try exporting encoder and decoder separately
    try:
        # Export encoder (g_a)
        if hasattr(net, 'g_a'):
            print("Exporting encoder (g_a)...")
            torch.onnx.export(
                net.g_a,
                dummy_input,
                str(output_dir / f"{model_name}_q{quality}_encoder.onnx"),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['latent'],
                dynamic_axes={
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'latent': {0: 'batch_size', 2: 'latent_h', 3: 'latent_w'}
                }
            )
            print(f"✓ Encoder exported")
        
        # Export decoder (g_s) - need to get latent dimensions first
        with torch.no_grad():
            latent = net.g_a(dummy_input)
        
        if hasattr(net, 'g_s'):
            print("Exporting decoder (g_s)...")
            torch.onnx.export(
                net.g_s,
                latent,
                str(output_dir / f"{model_name}_q{quality}_decoder.onnx"),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['latent'],
                output_names=['output'],
                dynamic_axes={
                    'latent': {0: 'batch_size', 2: 'latent_h', 3: 'latent_w'},
                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}
                }
            )
            print(f"✓ Decoder exported")
        
        print(f"\n✓ Encoder/Decoder export completed")
        if full_export_success:
            print(f"Note: Using encoder+decoder for compression (full model for reference only)")
        else:
            print(f"⚠ Note: Entropy coding will use PyTorch (not exportable to ONNX)")
        
        return str(output_dir / f"{model_name}_q{quality}_encoder.onnx")
        
    except Exception as e2:
        print(f"✗ Encoder/Decoder export also failed: {e2}")
        if full_export_success:
            print(f"\n✓ Full model export available but can't be used for compression")
            print(f"⚠ Recommendation: Use multiprocessing optimization instead (4-8x speedup)")
            return str(output_dir / f"{model_name}_q{quality}_full.onnx")
        else:
            print(f"\n❌ ONNX export not possible for this model")
            print(f"Recommendation: Use multiprocessing optimization instead (4-8x speedup)")
            return None
    finally:
        print(f"\n{'='*60}")
        print(f"Export process completed")
        print(f"Files saved to: {output_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CompressAI model to ONNX")
    parser.add_argument("--model", type=str, default="bmshj2018-factorized",
                        help="Model name")
    parser.add_argument("--quality", type=int, default=1,
                        help="Quality level (1-8)")
    parser.add_argument("--metric", type=str, default="mse",
                        help="Metric (mse or ms-ssim)")
    parser.add_argument("--output", type=str, default="model_onnx",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    
    args = parser.parse_args()
    export_model_to_onnx(args.model, args.quality, args.metric, args.output, args.device)
