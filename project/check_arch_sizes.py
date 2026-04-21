import os
import torch
from pathlib import Path

root = Path("./Results")

pt_files = []

for path in root.rglob("*.pt"):
    pt_files.append(path)

print(f"Found {len(pt_files)} .pt files")
for f in pt_files:
    print(f)

print(f"Found {len(pt_files)} .pt files")



def summarize_state_dict(state_dict):
    encoder = {
        k: v for k, v in state_dict.items()
        if isinstance(v, torch.Tensor) and "encoder" in k.lower()
    }
    decoder = {
        k: v for k, v in state_dict.items()
        if isinstance(v, torch.Tensor) and "decoder" in k.lower()
    }

    enc_params = sum(v.numel() for v in encoder.values()) if encoder else None
    dec_params = sum(v.numel() for v in decoder.values()) if decoder else None

    return encoder, decoder, enc_params, dec_params

def analyze_pt_file(path):
    print(f"\n--- {path} ---")

    # 1) safest path first: plain tensor checkpoint / state_dict
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)

        if isinstance(obj, dict):
            state_dict = obj.get("state_dict", obj) if isinstance(obj.get("state_dict", obj), dict) else obj
            encoder, decoder, enc_params, dec_params = summarize_state_dict(state_dict)

            if encoder or decoder:
                print(f"type: checkpoint/state_dict")
                print(f"encoder params: {enc_params}")
                print(f"decoder params: {dec_params}")

                print("encoder keys:")
                for k, v in encoder.items():
                    print(f"  {k}: {tuple(v.shape)}")

                print("decoder keys:")
                for k, v in decoder.items():
                    print(f"  {k}: {tuple(v.shape)}")
            else:
                print("type: checkpoint/state_dict")
                print("no explicit encoder/decoder keys found")
                print("sample keys:")
                for k in list(state_dict.keys())[:10]:
                    print(f"  {k}")
            return

    except Exception as e:
        msg = str(e)

        # 2) TorchScript fallback
        if "TorchScript archives" in msg or "torch.jit.load" in msg:
            try:
                model = torch.jit.load(str(path), map_location="cpu")
                print("type: torchscript")

                # Try direct submodules if present
                if hasattr(model, "encoder") and hasattr(model, "decoder"):
                    enc_params = sum(p.numel() for p in model.encoder.parameters())
                    dec_params = sum(p.numel() for p in model.decoder.parameters())
                    print(f"encoder params: {enc_params}")
                    print(f"decoder params: {dec_params}")
                else:
                    # fallback: inspect named parameters
                    named = dict(model.named_parameters())
                    encoder = {k: v for k, v in named.items() if "encoder" in k.lower()}
                    decoder = {k: v for k, v in named.items() if "decoder" in k.lower()}

                    if encoder or decoder:
                        enc_params = sum(v.numel() for v in encoder.values()) if encoder else None
                        dec_params = sum(v.numel() for v in decoder.values()) if decoder else None
                        print(f"encoder params: {enc_params}")
                        print(f"decoder params: {dec_params}")

                        print("encoder keys:")
                        for k, v in encoder.items():
                            print(f"  {k}: {tuple(v.shape)}")

                        print("decoder keys:")
                        for k, v in decoder.items():
                            print(f"  {k}: {tuple(v.shape)}")
                    else:
                        print("no explicit encoder/decoder parameters found")
                        print("sample parameter names:")
                        for k in list(named.keys())[:10]:
                            print(f"  {k}")
                return

            except Exception as e2:
                print(f"failed torchscript load: {e2}")
                return

        # 3) trusted-file fallback only
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            print("loaded with weights_only=False (trusted files only)")

            if isinstance(obj, dict):
                state_dict = obj.get("state_dict", obj) if isinstance(obj.get("state_dict", obj), dict) else obj
                encoder, decoder, enc_params, dec_params = summarize_state_dict(state_dict)

                if encoder or decoder:
                    print(f"encoder params: {enc_params}")
                    print(f"decoder params: {dec_params}")
                else:
                    print("no explicit encoder/decoder keys found")
                    print("sample keys:")
                    for k in list(state_dict.keys())[:10]:
                        print(f"  {k}")
            else:
                if hasattr(obj, "encoder") and hasattr(obj, "decoder"):
                    enc_params = sum(p.numel() for p in obj.encoder.parameters())
                    dec_params = sum(p.numel() for p in obj.decoder.parameters())
                    print(f"encoder params: {enc_params}")
                    print(f"decoder params: {dec_params}")
                else:
                    print("loaded object, but no encoder/decoder attributes found")

        except Exception as e3:
            print(f"failed to load: {e3}")


for pt_file in root.rglob("*.pt"):
    analyze_pt_file(pt_file)