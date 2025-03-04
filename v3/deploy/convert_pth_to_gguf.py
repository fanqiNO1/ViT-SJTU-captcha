try:
    import gguf
except ImportError:
    raise ImportError(
        'Please install gguf package by running `pip install gguf`')
import torch


def main():
    # load ckpt
    ckpt_path = '../../models/v3.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    ckpt['cls_token_index'] = torch.tensor(0, dtype=torch.int32)
    ckpt['input_position'] = torch.arange(0, 17, dtype=torch.int32)
    # hyperparameters
    hparams = {
        'image_size': 28,
        'num_layers': 2,
        'num_classes': 26,
        'patch_size': 7,
        'in_channels': 1,
        'hidden_size': 16,
        'num_attention_heads': 4,
        'num_key_value_heads': 1,
        'intermediate_size': 32,
        'act_fn': 'SiLU'
    }
    # GGUF
    gguf_writer = gguf.GGUFWriter(path='../../models/v3.gguf',
                                  arch='ViTCaptchaV3')
    # prepare hyperparameters
    for k, v in hparams.items():
        v_type = gguf.GGUFValueType.get_type(v)
        print(f'Writing hyperparameter {k} with value {v} and type {v_type}')
        gguf_writer.add_key_value(k, v, v_type)
    # prepare weights
    for name, tensor in ckpt.items():
        # write
        print(f'Writing tensor {name} with shape {tensor.shape}')
        gguf_writer.add_tensor(name, tensor.cpu().numpy())
    # write
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()


if __name__ == '__main__':
    main()
