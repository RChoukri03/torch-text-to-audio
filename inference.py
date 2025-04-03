import os
import torch
import torchaudio

from text import arabic_to_buckwalter, buckwalter_to_arabic, buckwalter_to_phonemes, simplify_phonemes
from utils import progbar, read_lines_from_file
import utils.make_html as html
from models.tacotron2 import Tacotron2Wave


def infer(text_file, checkpoint_path, output_dir, batch_size=2, denoise=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Tacotron2Wave(checkpoint_path)
    model.to(device).eval()

    os.makedirs(f"{output_dir}/wavs", exist_ok=True)
    lines = read_lines_from_file(text_file)
    batches = [lines[i:i+batch_size] for i in range(0, len(lines), batch_size)]

    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html.make_html_start())
        idx = 0

        for batch in progbar(batches):
            wavs = model.tts(batch, batch_size=batch_size, denoise=denoise)
            for text_line, wav in zip(batch, wavs):
                wav_path = f'{output_dir}/wavs/static{idx}.wav'
                torchaudio.save(wav_path, wav.unsqueeze(0), 22050)

                buck = arabic_to_buckwalter(text_line)
                print(text_line,'---', buck,'----',buckwalter_to_arabic(buck))
                phonemes = simplify_phonemes(
                    buckwalter_to_phonemes(buck).replace(' ', '').replace('+', ' ')
                )
                f.write(html.make_sample_entry2(wav_path, buckwalter_to_arabic(buck), f"{idx}) {phonemes}"))
                idx += 1

        f.write(html.make_volume_script(0.5))
        f.write(html.make_html_end())

    print(f"Inference complete! Results saved to {output_dir}")


if __name__ == '__main__':
    infer(
        text_file='data/test_arab.txt',
        # checkpoint_path='pretrained/exp_tc2_adv/states_5000.pth',
        checkpoint_path='pretrained/tacotron2_ar_adv.pth',
        output_dir='samples/res_tc2_adv1',
        batch_size=2,
        denoise=0.005
    )
