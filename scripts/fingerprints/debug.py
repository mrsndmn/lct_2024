
import torch
import torchaudio

if __name__ == '__main__':

    # sr = 16000
    # piracy, _ = torchaudio.load("./data/rutube/compressed_val_audios/ydcrodwtz3mstjq1vhbdflx6kyhj3y0p.wav")
    # torchaudio.save("./ydcrodwtz3mstjq1vhbdflx6kyhj3y0p_segment.wav", piracy[:, 1539*sr:1685*sr], sample_rate=sr)

    # legal, _ = torchaudio.load("./data/rutube/compressed_index_audios/ded3d179001b3f679a0101be95405d2c.wav")
    # torchaudio.save("./ded3d179001b3f679a0101be95405d2c_segment.wav", legal[:, 546*sr:692*sr], sample_rate=sr)

    sr = 16000
    piracy, _ = torchaudio.load("./data/rutube/compressed_val_audios/ydcrodwtz3mstjq1vhbdflx6kyhj3y0p.wav")
    torchaudio.save("./data/rutube/compressed_val_audios/ydcrodwtz3mstjq1vhbdflx6kyhj3y0p.wav", piracy[:, 1600:], sample_rate=sr)

    legal, _ = torchaudio.load("./data/rutube/compressed_index_audios/ded3d179001b3f679a0101be95405d2c.wav")
    torchaudio.save("./ded3d179001b3f679a0101be95405d2c_segment.wav", legal[:, :-1600], sample_rate=sr)
