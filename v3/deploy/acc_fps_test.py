import ctypes
import os

from tqdm import tqdm


class ViTCaptchaV3:

    def __init__(
        self,
        model_path: str,
        lib_path: str | None = None,
        use_timer: bool = False,
    ):
        if lib_path is None:
            lib_path = 'build/lib/libvit_captcha.so'
        self.lib = ctypes.CDLL(lib_path)
        # bind
        # load_vit_captcha_v3
        self.lib.load_vit_captcha_v3.argtypes = [ctypes.c_char_p]
        self.lib.load_vit_captcha_v3.restype = ctypes.c_void_p
        # free_vit_captcha_v3
        self.lib.free_vit_captcha_v3.argtypes = [ctypes.c_void_p]
        self.lib.free_vit_captcha_v3.restype = None
        # build_vit_captcha_v3_graph
        self.lib.build_vit_captcha_v3_graph.argtypes = [ctypes.c_void_p]
        self.lib.build_vit_captcha_v3_graph.restype = ctypes.c_void_p
        # predict or predict_with_timer
        self.use_timer = use_timer
        if not use_timer:
            self.lib.predict.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.lib.predict.restype = ctypes.c_char_p
            self.predict = self.lib.predict
        else:
            self.lib.predict_with_timer.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p
            ]
            self.lib.predict_with_timer.restype = ctypes.POINTER(
                ctypes.c_double)
            self.predict = self.lib.predict_with_timer
        # load model
        self.model = self.lib.load_vit_captcha_v3(model_path.encode('utf-8'))
        # build graph
        self.graph = self.lib.build_vit_captcha_v3_graph(self.model)

    def __call__(self, image_path: str):
        result = self.predict(self.graph, image_path.encode('utf-8'))
        if not self.use_timer:
            return result.decode('utf-8')
        else:
            all_time, preprocess_time, predict_time = result[0], result[
                1], result[2]
            num_chars = int(result[3])
            captcha = ''
            for i in range(num_chars):
                captcha += chr(int(result[4 + i]))
            return captcha, all_time, preprocess_time, predict_time

    def __del__(self):
        self.lib.free_vit_captcha_v3(self.model)


def metric(preds, gts):
    assert len(preds) == len(gts)
    captcha_correct, captcha_total = 0, 0
    char_correct, char_total = 0, 0

    for (pred, gt) in zip(preds, gts):
        captcha_total += 1
        if pred == gt:
            captcha_correct += 1

        if len(pred) == len(gt):
            char_total += len(gt)
            for (p, g) in zip(pred, gt):
                if p == g:
                    char_correct += 1

    return (captcha_correct / captcha_total, char_correct / char_total,
            captcha_correct, captcha_total, char_correct, char_total)


def main():
    model = ViTCaptchaV3('../../models/v3.gguf', use_timer=True)
    # prepare captcha files
    captcha_files = os.listdir('../../labelled')
    # time are in milliseconds
    all_time, preprocess_time, predict_time = 0, 0, 0
    captcha_preds, captcha_gts = [], []
    num_chars = 0
    # predict
    for captcha_file in tqdm(captcha_files):
        pred, all_time_i, preprocess_time_i, predict_time_i = model(
            f'../../labelled/{captcha_file}')
        # record
        all_time += all_time_i
        preprocess_time += preprocess_time_i
        predict_time += predict_time_i
        captcha_preds.append(pred)
        captcha_gts.append(captcha_file.split('_')[0])
        num_chars += len(pred)
    # metric
    all_time /= 1000
    preprocess_time /= 1000
    predict_time /= 1000
    (captcha_acc, char_acc, captcha_correct, captcha_total, char_correct,
     char_total) = metric(captcha_preds, captcha_gts)
    print(f'The accuracy of char: {100 * char_acc:.2f}% '
          f'({char_correct}/{char_total})')
    print(f'The accuracy of captcha: {100 * captcha_acc:.2f}% '
          f'({captcha_correct}/{captcha_total})')
    print(f'The fps of char is {num_chars / predict_time:.2f} '
          f'({num_chars} chars in {predict_time:.2f}s)')
    print(f'The fps of captcha is {len(captcha_files) / all_time:.2f} '
          f'({len(captcha_files)} captchas in {all_time:.2f}s)')
    print(
        f'The fps of preprocess is {len(captcha_files) / preprocess_time:.2f} '
        f'({len(captcha_files)} captchas in {preprocess_time:.2f}s)')


if __name__ == '__main__':
    main()
