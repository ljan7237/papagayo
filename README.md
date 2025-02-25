# papagayo
A Birdsong Synthesizer

## Background

Papagayo is an exploration of birdsong, machine learning, audio processing, and generative music, resulting in two prototype tools:

1. Timbre Transfer
2. Music Generator

### Birdsong Dataset  
Australian birdsong recordings were sourced from [xeno-canto.org](https://www.xeno-canto.org/):  
- Superb Lyrebirds  
- Butcherbirds  
- Magpies  
- Kookaburras  
- Noisy Friarbirds  
- Lorikeets  
- Mixed Australian birdsong samples  

### ML Components
- **Self-Organizing Feature Maps (SOFM)**: Embeds audio data into a lower-dimensional representation.  
- **2D Convolutional Network**: Upsamples embeddings into high-dimensional magnitude spectrogram slices.  
- **Autoencoder + Kalman Filter**: Predicts the next embedding state (quick to train but lacks long-term dependencies; a 3D ConvNet or Transformer may be better suited).  

![Architecture Diagram](diagrams/Architecture_Papagayo.png)  

---

# Applications

## Timbre Transfer  
A trained SOFM can embed birdsong (or any sound). New input audio is projected through the SOFM, enabling distance comparisons between input song slices and birdsong slices. The closest matches are used to reconstruct the input song using birdsong samples.  

## Songbird/Music Generator  

1. **Self-Organizing Feature Map Embedding**  
   - Audio is processed via the Short-Time Fourier Transform (STFT) and mapped onto SOFMs.  
   - The SOFM embeddings provide a lower-dimensional representation of the input.  
   - New audio can be embedded and compared to existing embeddings.  

2. **Autoencoder + Kalman Filter for Next-Step Prediction**  
   - Given an embedding at time step *t*, the model predicts the embedding at *t+1*.  

3. **2D Convolutional Network Upsampling**  
   - Uses the topology of SOFM embeddings to upsample back to the original spectrogram space.  

Combining (i) trained SOFMs, (ii) ConvNet upsampling, and (iii) Autoencoder+Kalman filters, Papagayo generates music autoregressively.  

---

# Installation & Usage

## Installation  
```
pip install -r requirements.txt
```
(*Tested on macOS with an M3 chip; other OSs not verified.*)  

---

## Processing Audio
### Perform STFT on input audio  
```
python stft/stft.py --audio-dir path/to/audio --output-dir path/to/audio_slices/ --save-phase --num-workers 4
```
(Saves phase information for reconstruction.)  

---

## Embedding Audio Data
### Using a trained SOFM (e.g., butcherbirds 8×8 SOFM trained for 200 epochs):  
If embedding the raw training data:  
```
python embed/embed.py --raw-data path/to/audio_slices.npy \
    --trained-som data/models/butcherbird/som_net/butcherbirds_8x8_som_net_200_epochs.npy \
    --output-file data/models/butcherbird/training_data/embeddings/butcherbird_embeddings.npy
```
If embedding new/unseen data (requires raw training data for scaling):  
```
python embed/embed.py --raw-data path/to/audio_slices.npy \
    --trained-som data/models/butcherbird/som_net/butcherbirds_8x8_som_net_200_epochs.npy \
    --output-file path/to/new_input_audio_slices.npy
```

---

## Timbre Transfer (Finding Closest Matches)  
```
python match_embeddings/match_embeddings.py --new-data path/to/new_embedding.npy \
    --birdsong-embeddings path/to/birdsong_embedding.npy \
    --birdsong-slices path/to/birdsong/slices.npy \
    --output-slices path/to/reconstructed_slices.npy
```

---

## Generating a Timbre-Transferred Song  
```
python song_generator/generate.py --audio-path path/to/input/song.mp3 \
    --spectrogram-path path/to/song/slices.npy \
    --phase-path path/to/song/phase.npy \
    --output-wav path/to/output.wav \
    --output-mp3 path/to/output.mp3
```

---

## Post-Processing (Denoising, Compression, Smoothing)  
```
python audio_post_process/post_process.py --input-mp3 path/to/generated_song.mp3 \
    --output-wav path/to/generated_song_post_processed.wav \
    --output-mp3 path/to/generated_song_post_processed.mp3 \
    --prop-decrease 0.8 --compress --low-pass 3000 --high-pass 100
```

---

## Upsampling Audio  
```
python upsampler/upsample_inference.py --input-data path/to/embeddings.npy \
    --model-path path/to/prediction_cnn_model.pth \
    --output-path path/to/slice_inferences.npy
```

### Generating an Upsampled Song  
```
python song_generator/generate.py --audio-path path/to/original/audio.mp3 \
    --spectrogram-path path/to/upsampled_slices.npy \
    --phase-path path/to/original/audio/phase.npy \
    --output-wav path/to/generated/song.wav \
    --output-mp3 path/to/generated/song.mp3
```

---

## Autoregression (Music Generation)  
Requires:  
- A trained **SOFM network** (*train_som/train_som_net.ipynb, Google Colab*).  
- A trained **2D ConvNet upsampler** (*upsampler/train_upsampler.ipynb, Google Colab*).  
- A trained **Autoencoder + Kalman Filter** (*train_autoencoder/train_auto_encoder.py*).  

Given an initial embedding, the autoregressor recursively predicts the next embeddings, upscales them to the original audio space, and reconstructs a song.  

```
python autoregressor/autoregressor.py --initial-input path/to/starting_point_embedding.npy \
    --num-steps 20000 \
    --output-file AI_birdsong.npy --to-wav
```
(*Saves AI_birdsong.mp3 and AI_birdsong.wav*)  

---

# Examples  
Audio samples from:  
- Timbre Transfer  
- Upsampling  
- Autoregression  

Available in the `examples/` directory.  

---

# Limitations & Future Work  

## Current Limitations  
- **Autoencoder for next-state prediction is limited**: No long-term dependencies; consider 3D ConvNet (predicting with multiple past samples) or a Transformer.  
- **Distance measures**: Currently uses Frobenius norm—try cosine similarity, Manhattan distance, etc.  
- **Mono processing only**: Extend to stereo.  
- **Hyperparameter tuning**: Optimize SOFM, Autoencoder+Kalman Filter, and ConvNet models.  
- **Data cleaning**: Training data contains ambient noise.  
- **Phase reconstruction**: Needs improvement.  
- **Higher resolution slicing**: Experimentation needed.  
- **Testing**: More validation required.  

## Future Directions  
- **Interactive "Draw a Song" Instrument**  
  - Utilize SOFMs’ topology to allow users to "draw" a song.  
  - Could serve as a generative music tool.  

<!-- # papagayo
A Birdsong Synthesizer

## Background

Papagayo is an exploration in birdsong, machine learning, audio processing, and generative music from which two protoype tools have been produced: i) Timbre Transfer, ii) Music Generator.

Sets of Australian birdsong audio files were sourced from xeno-canto.org:
- Superb Lyrebirds  
- Butcherbirds
- Magpies
- Kookaburras
- Noisy Friarbirds
- Lorikeets
- Sample of Australian birdsong

Two tools have been prototyped:
 - Timbre Transfer
 - Songbird/Music Generator Tool

 # ML Components:
 - Self Organizing Feature Maps (embedding audio data in lower dimensional representation), see https://en.wikipedia.org/wiki/Self-organizing_map
 - 2d Convnet (upsample from lower dimensional embedding to higher dimensional magnitude spectrogram slices)
 - Autoencoder + Kalman Filter for next stage prediction (quick to train but lacks ability to account for long-term dependencies) - need to replace with different architecture (ie temporal 3d convnet or Transformer)

![Alt text](diagrams/Architecture_Papagayo.png)


# Applications:

## Timbre Transfer
Given Self Organizing Feature Maps SOFM can be trained to embed Birdsong (or any song...). Any new (input) audio can then be projected through the SOFM, which can then allow for distance comparisons of the input song slices to birdsong slices. Closest matches  can then be used to reconstitute the input song with birdsong.

## Songbird/Music Generator Tool

Audio from birdsong recordings (xeno-canto.org data sets) are processed using librosa's Short-Time Fourier Transform (STFT) and used as data for training SOFMs. Projections can produce node difference maps which are taken to be a lower dimensional representation (embedding) of a the higher dimensional audio slice input.

### Self Organizing Feature Map Embedding
New audio can then be processed and projected through SOFMs to produce embeddings which can be efficiently processed or compared to other embeddings (i.e. for timbre transfer) that map to magnitude spectrogram slices.

### Autoencoder+Kalman Filter for next step prediction
An Autoencoder and Kalman filter can be trained to predict, given a current embedding at step t, what the embedding will be at step t+1.

### 2DConvnet Upsampling
Given the topological nature of SOFM embeddings, 2DConvnets can be trained to upsample from the lower dimensional embedding to the higher dimension of the input space.

Combining i) Trained SOM netowrks producing embeddings of audio slices, ii) Convnets for upsampling and iii) Autoencoders and Kalman filters, an autoregressive music generator is produced.


# Usage

## Install requirements 
pip install -r requirements.txt
(Note requirements work for MAC OS and M3 Chip. Other OSs not tested.)

## Process Audio: do stft, if performing stft on input data to reconstruct, save phase.
python stft/stft.py --audio-dir path/to/audio --output-dir path/to/audio_slices/ --save-phase --num-workers 4

## Embed data (raw or new), 
Given a trained Self Organizing Feature Map, eg data/models/butcherbird/som_net/butcherbirds_8x8_som_net_200_epochs.npy

If embedding the raw data used for training the SOFM (i.e. no new data):

python embed/embed.py --raw-data path/to/audio_slices.npy --trained-som data/models/butcherbird/som_net/butcherbirds_8x8_som_net_200_epochs.npy --output-file data/models/butcherbird/training_data/embeddings/butcherbird_embeddings.npy

If embedding new/unseen data, raw_data is required to scale the new data.

python embed/embed.py --raw-data path/to/audio_slices.npy --trained-som data/models/butcherbird/som_net/butcherbirds_8x8_som_net_200_epochs.npy --output-file path/to/new_input_audio_slices.npy

## Get closest matches (timbre transfer)
python match_embeddings/match_embeddings.py --new-data path/to/new_embedding.npy --birdsong-embeddings path/to/birdsong_embedding.npy --birdsong-slices path/to/birdsong/slices.npy  --output-slices path/to/reconstructed slices.

## Generate song/timbre transfer (reconstitute input with birdsong)
python song_generator/generate.py --audio-path path/to/input/song.mp3 --spectrogram-path=path/to/song/slices.npy --phase-path path/to/song/phase.npy --output-wav path/to/output.wav --output-mp3 path/to/output.mp3 

## Post process (denoising, compression, smoothing)
python audio_post_process/post_process.py --input-mp3 path/to/generated_song.mp3  --output-wav path/to/generated_song_post_processed.wav --output-mp3 path/to/generated_song_post_processed.wav --prop-decrease 0.8 --compress --low-pass 3000 --high-pass 100

## Upsampling
python upsampler/upsample_inference.py --input-data path/to/embeddings.npy --model-path path/to/prediction_cnn_model.pth --output-path path/to/slice_inferences.npy

## Generate upsampled song
python song_generator/generate.py --audio-path path/to/original/audio.mp3 --spectrogram-path path/to/upsampled_slices.npy --phase-path path/to/original/audio/phase.npy --output-wav path/to/generated/song.wav --output-mp3 path/to/generated/song.wav

## Autoregression (MUSIC GENERATOR)

- Requires a trained SOFM net (See SOFM training notebook: train_som/train_som_net.ipynb used in Google Collab), trained Upsampler 2DConvnet (See training notebook: upsampler/train_upsampler.ipynb used in Google Collab) and a trained Autoencoder and Kalman Filter (see:  train_autoencoder/train_auto_encoder.py).

Using any starting point (a lower dimensional embedding) the autoregressor will propagate embedding predictions by feeding in as input the output from the previous step. The 2Dconvent upsamples from the embedding space to the audio slice space and which are processed using librosa's istft, and estimated phase, to produce wav and mp3 files of generated song.

python autoregressor/autoregressor.py --initial-input path/to/starting_point_embedding.npy --num-steps 20000 --output-file AI_birdsong.npy --to-wav # saves AI_birdsong.mp3 and AI_birdsong.wav

# Examples 
See examples of audio produced by i) Timbre Transfer, ii) Upsampling and iii) Autoregression in examples/

# Current Limitations/Todos:
 - Although an Autoencoder is quick to train, it is a poor choice for next-state prediction as it has no long term dependendcies. Instead, use other architectures like a 3Dconv net (predcition given multiple past samples) or Transformer.
 - Experiment with different distance measures (Currently using Frobenius, try cosine similarity or Manhattan etc... for distance measures.)
 - Currently processing Mono. Modify for Stereo.
 - Training/Hyper parameter optimization required for SOM, Autoencoder+Kalman Filter, ConvNet.
 - Data cleaning/preprocessing (training data is raw and captures ambient noise)
 - Experiment with Phase reconstruction.
 - Experiment with higher resolution slicing.
 - Testing 


# Future
Given the topological representation afforded by SOFMs, an application could be produced to allow users to "Draw a song". This could be a fun "instrument to play with.

 -->
