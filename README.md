# papagayo
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
- Given Self Organizing Feature Maps can be trained to embed Birdsong (or any song...)
- Any new audio can then be projected through the SOM, which can then allow for distance comparisons of the input song slices to birdsong.
- Closest matches can then be used to reconstitute the input song with birdsong.

## Songbird/Music Generator Tool

Audio from birdsong recordings (xeno-canto.org data sets) are processed (Short-Time Fourier Transform) and used as data for training Self Organizing Feature Maps (SOFMs). Projections can produce node difference maps which are taken to be a lower dimensional representation (embedding) of a the higher dimensional audio slice input.

### Self Organizing Feature Map Embedding
New audio can then be processed and projected through SOFMs to produce embeddings which can be efficiently processed or compared to other embeddings (i.e. for timbre transfer) that map to magnitude spectrogram slices.

### Autoencoder+Kalman Filter for next step prediction
An Autoencoder and Kalman filter can be trained to predict, given a current embedding at step t, what the embedding will be at step t+1.

### 2DConvnet Upsampling
Given the topological nature of SOM embeddings, 2DConvnets can be used to upsample from the lower dimensional embedding to the higher dimension of the input space.

Combining i) Trained SOM netowrks producing embeddings of audio slices, ii) Convnets for upsampling and iii) Autoencoders and Kalman filters, an autoregressive music generator is produced.

In this study, sets of Australian birdsong audio files were sourced from xeno-canto.org:
- Superb Lyre Birds  
- Butcherbirds
- Magpies
- Kookaburras
- Noisy Friar Birds
- Lorikeets
- Sample of Australian birdsong

# Usage


Given a trained Self Organizing Feature Map (See: train_som/train_som_net.ipynb used for training in Google Collab), trained Upsampler Convnet (upsampler/train_upsampler.ipynb, used in Google Collab) and a trained Autoencoder and Kalman Filter (trained locally using train_autoencoder/train_auto_encoder.py):

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
Given any starting point (a single lower dimensional embedding) an autoregressive architecture can be produced such that a next step prediction from the autoencoder+kalman filter can be fed then used as input (which will generate --num-steps slices to convert to mp3/wav)

python autoregressor/autoregressor.py --initial-input path/to/starting_point_embedding.npy --num-steps 20000 --output-file AI_birdsong.npy --to-wav # saves AI_birdsong.mp3 and AI_birdsong.wav

# Examples 
See examples of audio produced by i) Timbre Transfer, ii) Upsampling and iii) Autoregression in examples/

Current Limitations/Todos:
 - Although an Autoencoder is quick to train, it is a poor choice for next-state prediction as it has no long term dependendcies. Instead, use other architectures like a 3Dconv net (predcition given multiple past samples) or Transformer.
 - Experiment with different distance measures (Currently using Frobenius, try cosine similarity or Manhattan etc... for distance measures.)
 - Currently processing Mono. Modify for Stereo.
 - Training/Hyper parameter optimization required for SOM, Autoencoder+Kalman Filter, ConvNet.
 - Data cleaning/preprocessing (training data is raw and captures ambient noise)
 - Experiment with Phase reconstruction.







