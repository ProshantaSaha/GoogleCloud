# TPU optimizations
>[!NOTE]
>These examples assume:
>1. You already have access to an active GCP account
>2. You have a billing enabled on GCP
>3. You have enabled the necessary services (e.g. Vertex API, TPU API, etc)
>4. You understand how to use service accounts, and provide them with the necessary privileges 

1. When speech to text fails at high rates it is important to check the audio quality of the file. In [optimized_data_pipeline.ipynb](https://github.com/ProshantaSaha/GoogleCloud/blob/main/tpu/tfdata_OOM_issues/optimized_data_pipeline.ipynb), I show an example of code that is limiting the amount of open files by avoiding the use of AUTOTUNE, limiting the number of prefetched files, and avoiding data shuffle if possible. 