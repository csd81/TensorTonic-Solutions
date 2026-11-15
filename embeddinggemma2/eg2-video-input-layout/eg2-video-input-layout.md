# Video Input Layout

EmbeddingGemma 2 has no separate video encoder. A clip is written out as a sequence of the modalities the model already understands: text timestamps, image segments from the vision encoder and audio segments from the audio encoder. This problem builds that sequence and counts the soft tokens the backbone will see, which is where the cost of a video embedding is decided.

## A video is a budgeted list of images

The model samples frames at one per second and treats each frame as an image. Every image costs a fixed number of soft tokens, 140 per frame in the paper's training setup, so a one-minute clip at full frame rate would already need 8,400 image tokens before any audio. The paper caps a video at a frame budget of $N_{\text{frm}} = 32$ and offers two ways to fit it.

- **Truncation.** Keep the first $N$ frames. This preserves fine temporal detail at the start and ignores everything later, which suits short actions or clips whose meaning is front-loaded.
- **Uniform sampling.** Keep $N$ frames spread evenly over the whole clip. This sees the entire story at a coarser time resolution, which suits long videos where the important moment could be anywhere.

The paper says only that reduction happens "sometimes via truncation" and "sometimes via uniform sampling". The exercise lets the caller choose the mode, and it fixes the uniform rule as $i_j = \lfloor j(F-1)/(N-1) + 1/2 \rfloor$ for a video of $F$ seconds. Dividing by $N - 1$ rather than $N$ guarantees that $i_0 = 0$ and $i_{N-1} = F - 1$, so the last moment of the video is always seen. Adding one half and flooring rounds every half step upward. Python's built-in rounding sends half steps to the nearest even integer, which would make the sampled seconds depend on parity.

## Timestamps tell the backbone when

Mean pooling over a bidirectional transformer is blind to how far apart two frames were. The paper writes a text timestamp in MM:SS format before each frame and before each audio segment, which gives the backbone explicit time information in a form its text embedding table already handles. Zero padding keeps every stamp five characters long.

## Native audio fills every second

With audio enabled, the soundtrack is cut into one-second segments at 25 soft tokens per second, and each segment follows the frame it comes after. The paper is explicit about what happens when frames are thinned: "if frames are truncated we might have multiple trailing audio segments, and if frames are uniformly sampled, we might have multiple audio segments between frames." The exercise follows that rule, so every second of the video contributes exactly one audio segment even when its frame was dropped. Sound therefore keeps full temporal coverage while vision is subsampled, which is cheap because 25 audio tokens per second is far less than one frame.

Encoding audio natively instead of transcribing it matters. On MSEB retrieval the paper reports 69.54 MRR@10 with native audio against 57.59 when the same model embeds ASR transcripts. Transcripts lose accents, homophones and every non-speech sound, while native audio tokens keep them available to the shared backbone. The whole layout reflects the paper's broader design, in which one transformer reads text, image and audio tokens in a single ordered sequence.

## A seven-second clip with three frames

Take $F = 7$, $N = 3$, uniform mode, $T = 10$ soft tokens per frame and audio on. The index rule gives $\lfloor 0 + 0.5 \rfloor = 0$, $\lfloor 3 + 0.5 \rfloor = 3$ and $\lfloor 6 + 0.5 \rfloor = 6$, so the kept seconds are 0, 3 and 6. Frame 0 is followed by audio for seconds 0, 1 and 2. Frame 3 is followed by seconds 3, 4 and 5. Frame 6 is followed only by second 6, since the clip ends there.

The layout has BOS, three image segments of four entries each, seven audio segments of four entries each and EOS, for $1 + 12 + 28 + 1 = 42$ entries. The soft token count is $3 \times 10 + 7 \times 25 = 205$. The first mode would keep seconds 0, 1 and 2 instead, with seconds 2 through 6 trailing after the last frame, and the count would stay 205.

## Common mistakes to avoid

- **Striding by $F/N$.** Indices $\lfloor jF/N \rfloor$ stop short of the final second, so the end of every long clip is lost.
- **Rounding half steps to even.** Built-in rounding turns 2.5 into 2 and 3.5 into 4, which breaks the round-half-up rule on exactly the cases that land between two seconds.
- **Writing raw or unpadded timestamps.** The backbone sees text, so 65 or 1:5 is a different token pattern from 01:05.
- **Attaching only one audio second per frame.** When frames are skipped, the seconds in between still need their own timestamped audio segments.
- **Stamping every audio segment with its frame's time.** Each audio segment marks when its own chunk begins, which differs from the frame time after the first segment.
- **Counting markers as soft tokens.** Only image and audio entries carry encoder outputs, and BOS, EOS, delimiters and timestamps do not.