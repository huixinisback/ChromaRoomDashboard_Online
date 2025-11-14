# Quick Start: Deploy to Google Cloud Run

## Fastest Way (5 minutes)

1. **Install Google Cloud SDK** (if not installed):
   ```bash
   # Windows: Download from https://cloud.google.com/sdk/docs/install
   # Or use PowerShell:
   (New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe"); & $env:Temp\GoogleCloudSDKInstaller.exe
   ```

2. **Login and Set Project**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Enable APIs**:
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com
   ```

4. **Deploy**:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/chromaroom-dashboard
   gcloud run deploy chromaroom-dashboard --image gcr.io/YOUR_PROJECT_ID/chromaroom-dashboard --platform managed --region us-central1 --allow-unauthenticated --port 8080
   ```

5. **Get URL**: The command will output your dashboard URL!

## Test Locally First

```bash
# Build Docker image
docker build -t chromaroom-dashboard .

# Run locally
docker run -p 8080:8080 chromaroom-dashboard

# Visit http://localhost:8080
```

## Update After Changes

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/chromaroom-dashboard
gcloud run deploy chromaroom-dashboard --image gcr.io/YOUR_PROJECT_ID/chromaroom-dashboard --region us-central1
```

## Cost

- **Free tier**: 2 million requests/month
- **After free tier**: ~$0.40 per million requests
- **Very affordable** for most use cases!


