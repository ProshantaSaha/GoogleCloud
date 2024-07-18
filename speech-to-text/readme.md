# Speech-To-Text Examples
>[!WARNING]
>The examples shared in this repository is shared as-is with no claim of completeness or support. I make no warranties on the cost of running these examples. I will do my best to list all the depenendcies used so that you can have a reference point should you need to update them in the future 

>[!NOTE]
>These examples assume:
>1. You already have access to an active GCP account
>2. You have a billing enabled on GCP
>3. You have enabled the necessary services (e.g. Vertex API, TPU API, etc)
>4. You understand how to service accounts, and provide them with the necessary privileges 

1. When speech to text fails at high rates it is important to check the audio quality of the file. In [check_audio_quality_stt.ipynb](https://github.com/ProshantaSaha/GoogleCloud/blob/sandbox/speech-to-text/check_audio_quality_stt.ipynb), I show an example of audio files that were artificially clipped to 4KHz to optimize for human voice. Since STT are mostly trained on full spectrum audio freqencies, the model is unable to get the transcript quite right when audio has been altered. 
