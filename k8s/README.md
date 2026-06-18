# Running the pipeline on Minikube

These steps run the containerized pipeline inside a local Kubernetes cluster.

1. Start the cluster:

   ```
   minikube start
   ```

2. Point your shell's Docker client at Minikube's daemon, then build the image there
   (so the cluster can use it without a registry):

   ```
   eval $(minikube docker-env)
   docker build -t hatespeech-pipeline:latest ..
   ```

3. Launch the job and watch it:

   ```
   kubectl apply -f job.yaml
   kubectl get jobs
   kubectl get pods
   ```

4. Check the logs once the pod completes:

   ```
   kubectl logs job/hatespeech-pipeline
   ```

5. Clean up:

   ```
   kubectl delete -f job.yaml
   ```

The job uses `imagePullPolicy: Never`, so the image must be built inside Minikube's
Docker daemon (step 2). Take screenshots of `kubectl get pods` and the job logs for
the submission.
