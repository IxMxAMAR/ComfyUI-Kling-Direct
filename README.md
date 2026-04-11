# ComfyUI-Kling-Direct

Direct access to the full Kling AI API from ComfyUI. No proxy. No middleman. No mystery black box sitting between you and the actual API.

Just JWT-authenticated requests going straight to Kling's Singapore endpoint, doing exactly what you told them to do.

32 nodes. Everything Kling offers. All of it.

---

## Why does this exist?

Because Kling AI is genuinely impressive and you should be able to use the full API from ComfyUI without wondering what some wrapper is doing behind the scenes. This package gives you direct, transparent access — you provide your keys, it generates JWT tokens automatically, and your requests go out. That's the whole deal.

---

## Installation

**Option 1 — ComfyUI Manager (easiest)**

Search for `Kling Direct` in the Manager and install. Done.

**Option 2 — Registry**

```
comfy node registry-install kling-direct
```

**Option 3 — Manual (you like doing things yourself)**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/IxMxAMAR/ComfyUI-Kling-Direct
pip install requests PyJWT
```

---

## Getting Your API Keys

1. Go to [klingai.com](https://klingai.com) and sign up
2. Navigate to API Settings
3. Grab your `access_key` and `secret_key`
4. Plug them into the KlingAuth node
5. That's literally it

The auth node handles JWT generation automatically. You don't need to think about tokens, expiry, or any of that. Just give it your keys and connect it to whatever node you're using.

Your keys are password-masked in the UI because they should be.

---

## The 32 Nodes

### Video (8 nodes)

| Node | What it does |
|------|--------------|
| Text to Video | Generate video from a text prompt |
| Image to Video | Animate a still image |
| Video Omni | Multi-modal video generation |
| Video Extend | Make an existing video longer |
| Lip Sync | Sync lips to audio on a video |
| Advanced Lip Sync | Lip sync with more control |
| Motion Control | Drive motion with camera/subject controls |
| Avatar Generation | Generate avatar video from a reference |

### Image (6 nodes)

| Node | What it does |
|------|--------------|
| Image Generation | Text-to-image via Kling |
| Image Omni | Multi-modal image generation |
| Image Extend | Outpaint / extend an image |
| Virtual Try-On | Dress a person in a garment image |
| Multi-Shot | Up to 6 shots in one generation |
| Image Recognize | Analyze and describe an image |

### Audio (5 nodes)

| Node | What it does |
|------|--------------|
| Text to Audio | Generate sound effects from a description |
| Text to Speech | Basic TTS |
| TTS Advanced | TTS with more voice control |
| Video to Audio | Generate audio that matches a video |
| Voice Clone | Clone a voice from a reference clip |

### Effects (3 nodes)

| Node | What it does |
|------|--------------|
| Video Effects | Apply effects: hug, kiss, heart gesture, and more |
| Effect Templates | Use Kling's built-in effect templates |
| AI Upscale | Upscale images or videos with AI |

### Config and I/O (10 nodes)

| Node | What it does |
|------|--------------|
| KlingAuth | JWT auth — connect this to everything |
| Video Loader | Load a video file for use in the graph |
| Raw File Loader | Load any file as raw bytes |
| Raw File Saver | Save raw bytes to disk |
| Asset Upload | Upload a file to Kling's servers |
| Element Selector | Pick specific elements for generation |
| Camera Control | Define camera movement parameters |
| Voice Selector | Pick a voice for TTS nodes |
| Cloud Uploader | Upload assets to cloud storage |
| Fast Video Saver | Save video output quickly |

---

## Supported Models

- `kling-v3`
- `kling-v2-6`
- `kling-v2-master`
- `kling-v2-1`
- `kling-v1-6`

---

## A Few Things Worth Knowing

**Polling is adaptive.** When you submit a job, the nodes poll Kling's API at 3s intervals early on, backing off to 5s and then 8s as time passes. This keeps things responsive for fast jobs without hammering the API for slow ones.

**Errors are classified.** Transient failures (rate limits, temporary server errors) trigger retries. Permanent failures (bad input, auth issues) fail immediately. You get a useful error instead of a confusing timeout.

**Sound is a boolean.** Not a string like "true". An actual boolean. This matters when you're wiring things together and wondering why your video is silent.

**IS_CHANGED is on all API nodes.** Every execution sends a fresh request. Kling's generation is non-deterministic and that's the point — caching would just give you the same video every time, which is the opposite of what you want.

---

## Part of a Bigger Package

If you also use ElevenLabs or Google Gemini, this package is included in **ComfyUI-API-Toolkit** — a single install that bundles Kling Direct, ElevenLabs Pro, and Gemini nodes together. Useful if you're building multi-service workflows and don't want to manage three separate packages.

---

## Requirements

- ComfyUI
- Python 3.10+
- `requests`
- `PyJWT`
- A Kling API account

---

## License

MIT

---

*Made by IxMxAMAR*
