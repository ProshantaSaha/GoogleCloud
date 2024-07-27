# Speech-To-Text Examples
>[!NOTE]
>These examples assume:
>1. You already have access to an active GCP account
>2. You have a billing enabled on GCP
>3. You have enabled the necessary services (e.g. Vertex API, TPU API, etc)
>4. You understand how to use service accounts, and provide them with the necessary privileges 

1. When using TPU with limited amount of host memory it is important to limit the number of open file as the buffers used could exceed available host memory. In [check_audio_quality_stt.ipynb](https://github.com/ProshantaSaha/GoogleCloud/blob/main/speech-to-text/check_audio_quality_stt.ipynb), I show an example of audio files that were artificially clipped to 4KHz to optimize for human voice. Since STT are mostly trained on full spectrum audio freqencies, the model is unable to get the transcript quite right when audio has been altered. 
