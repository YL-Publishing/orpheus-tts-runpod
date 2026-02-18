# Orpheus TTS 3B — RunPod Serverless Worker

Emotional TTS for the video pipeline via RunPod GPU cloud.

## Architecture

```
VPS (CPU, $30/yr)                    RunPod Serverless (GPU)
┌──────────────────┐                 ┌──────────────────────┐
│ orchestrator.py   │    HTTP API     │ Orpheus TTS 3B       │
│ produce_recap.py  │ ──────────────► │ RTX 4090 (24GB)      │
│ ffmpeg            │ ◄────────────── │ ~$0.05/chapter       │
│ sound library     │    WAV audio    │ Scales to zero       │
└──────────────────┘                 └──────────────────────┘
```

## Setup (One-Time)

### 1. Create RunPod Account

Go to https://www.runpod.io/ → Sign Up → Add $10 credit.

### 2. Build & Push Docker Image

**Option A: GitHub integration (easiest)**
- Push this directory to a GitHub repo
- In RunPod: Serverless → New Template → Connect GitHub repo
- RunPod builds the image automatically

**Option B: Build locally (needs Docker Desktop)**
```bash
cd runpod-orpheus/
docker build --platform linux/amd64 -t YOURDOCKERHUB/orpheus-tts-runpod:latest .
docker push YOURDOCKERHUB/orpheus-tts-runpod:latest
```

### 3. Create Serverless Endpoint

In RunPod Console:
1. Go to **Serverless** → **New Endpoint**
2. Select your template (from step 2)
3. GPU: **RTX 4090** (24GB)
4. Workers: **Min 0** / Max 1 (flex — scales to zero)
5. Idle timeout: 30s (keeps warm briefly after request)
6. Click Create

Copy the **Endpoint ID** (looks like `abc123xyz`).

### 4. Get API Key

RunPod Console → Settings → API Keys → Create Key.

### 5. Configure VPS

```bash
ssh -i ~/.ssh/yonglibrary_vps root@100.85.221.52

# Add to pipeline user's environment
cat >> /home/pipeline/.bashrc << 'EOF'
export TTS_ENGINE=orpheus
export RUNPOD_API_KEY=your_key_here
export ORPHEUS_ENDPOINT_ID=your_endpoint_id_here
EOF

# Apply patch
scp patch_orpheus_tts.py to VPS, then:
python3 /tmp/patch_orpheus_tts.py
```

### 6. Test

```bash
su - pipeline
cd /opt/video-pipeline
python3 -c "
from scripts.orpheus_client import OrpheusClient
c = OrpheusClient()
wav = c.generate('Hello, this is a test of Orpheus TTS.', voice='tara')
print(f'Got {len(wav)} bytes of audio')
with open('/tmp/test_orpheus.wav', 'wb') as f:
    f.write(wav)
print('Saved to /tmp/test_orpheus.wav')
"
```

## Emotion Tags

Orpheus supports inline emotion tags in text:

| Tag | Effect |
|-----|--------|
| `<laugh>` | Laughter |
| `<chuckle>` | Soft laugh |
| `<sigh>` | Sighing |
| `<gasp>` | Gasping |
| `<cough>` | Coughing |
| `<sniffle>` | Sniffling |
| `<groan>` | Groaning |
| `<yawn>` | Yawning |

Example: `"Well, that's interesting. <laugh> I hadn't expected that."`

## Voices

| Voice | Character | Best For |
|-------|-----------|----------|
| `tara` | Female, clear | Default narrator |
| `leo` | Male, warm | Male narrator |
| `dan` | Male, young | Young male characters |
| `jess` | Female, bright | Young female characters |
| `leah` | Female, mature | Authority, mature women |
| `zac` | Male, deep | Mature, authoritative men |
| `mia` | Female, light | Children, soft voices |
| `zoe` | Female, energetic | Excited, fast-paced |

## Cost

| Item | Cost |
|------|------|
| Per chapter (~5 min audio) | ~$0.05 |
| 15 chapters full run | ~$0.70 |
| Idle | $0.00 |
| Monthly (weekly renders) | ~$2-3 |
