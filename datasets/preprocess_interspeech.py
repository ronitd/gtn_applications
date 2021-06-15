import argparse
import json
import os
import torchaudio


SPLITS = [
    "gu-in-Train", "gu-in-Test"
]


def load_transcripts(path):
    filepath = path + '/transcription.txt'
    data = {}
    with open(filepath, "r", encoding="utf-8") as fid:
        lines = (l.strip().split() for l in fid)
        lines = ((l[0], " ".join(l[1:])) for l in lines)
        data.update(lines)
    return data


def path_from_key(key, split_path, ext):
    return split_path +'/Audios' + '/' + key + '.' + ext


def clean_text(text):
    return text.strip().lower()


def build_json(data_path, save_path, split):
    split_path = os.path.join(data_path, split)
    transcripts = load_transcripts(split_path)
    save_file = os.path.join(save_path, f"{split}.json")
    with open(save_file, 'w', encoding="utf-8") as fid:
        for k, t in transcripts.items():
            wav_file = path_from_key(k, split_path, ext="wav")
            audio = torchaudio.load(wav_file)
            duration = audio[0].numel() / audio[1]
            #t = clean_text(t)
            #print(t)
            datum = {'text' : t,
                     'duration' : duration,
                     'audio' : wav_file}
            #print(datum)
            #exit()
            json.dump(datum, fid, ensure_ascii=False)
            fid.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Preprocess librispeech dataset.")

    parser.add_argument("--data_path", type=str,
        help="Location of the librispeech root directory.")
    parser.add_argument("--save_path", type=str,
        help="The json is saved in <save_path>/{train-clean-100, ...}.json")
    args = parser.parse_args()

    for split in SPLITS:
        print("Preprocessing {}".format(split))
        build_json(args.data_path, args.save_path, split)
