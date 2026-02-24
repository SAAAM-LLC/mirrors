# MIRRORS Web Interface

Real-time visualization dashboard for the MIRRORS (Minimal Irreducible Requirements for Recursive Self-awareness) system.

## Features

- **Live Data Streaming**: Connects to running MIRRORS instance via JSON status file
- **Manifold Visualization**: 2D projection of attractor topology with current state position
- **Real-Time Metrics**: Emergence score, introspection depth, energy, goal focus
- **Topology Evolution Tracking**: Live updates when attractors spawn/merge/prune
- **Identity Monitoring**: Track identity signature and structural age
- **Auto-Fallback**: Automatically switches to simulated data when MIRRORS isn't running

## Quick Start

### 1. Start MIRRORS System

```bash
# In the root directory
python core.py
```

This will:
- Initialize the MIRRORS system
- Begin continuous existence loop
- Export status to `web-interface/.mirrors-status.json` every 30 seconds

### 2. Start Web Interface

```bash
# In a separate terminal
cd web-interface
npm install  # First time only
npm run dev
```

### 3. Open Dashboard

Navigate to [http://localhost:3000](http://localhost:3000)

The dashboard will automatically:
- ✅ Connect to live MIRRORS data if available (shows "Live" badge)
- ⚠️  Fall back to simulated data if MIRRORS isn't running (shows "Simulated" badge)

## Architecture

### Data Flow

```
MIRRORS (core.py)
    ↓ writes JSON every 30s
.mirrors-status.json
    ↓ served by
Next.js API Route (/api/status)
    ↓ polled every 2s
React Dashboard
```

### Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui
- **Charts**: Recharts
- **Visualization**: Custom Canvas rendering

## Development

```bash
npm run dev    # Development server
npm run build  # Production build
npm run start  # Production server
npm run lint   # TypeScript + ESLint
```

## Status Export Format

The MIRRORS system exports the following JSON structure:

```json
{
  "timestamp": 1234567890.123,
  "elapsed": 120.5,
  "cycles": 4240,
  "introspectionDepth": 99,
  "emergenceScore": 4.8840,
  "currentAttractor": "724c85f1",
  "energy": -0.9677,
  "distanceToCenter": 0.8342,
  "attractorCount": 7,
  "avgDepth": 1.217,
  "evolutionCount": 0,
  "structuralAge": 0.0,
  "goalFocus": 0.0959,
  "identity": "111aa1ba64a01b2a",
  "observation": {
    "regime": "sinusoid",
    "time": 42.4
  },
  "attractors": [
    {
      "id": "724c85f1",
      "depth": 1.5,
      "radius": 0.3,
      "center": [0.123, -0.456]
    }
  ],
  "topologyHistory": [],
  "goalPreferences": {},
  "dynamicsVerified": {}
}
```

## Troubleshooting

### "MIRRORS not running" error

- Make sure `python core.py` is running in the root directory
- Check that `.mirrors-status.json` is being created in `web-interface/`
- Wait 30 seconds after starting MIRRORS for first status export

### Dashboard shows "Simulated" mode

- MIRRORS system may not be running
- Status file may be outdated (MIRRORS crashed)
- Check console for fetch errors

### Build errors

```bash
rm -rf node_modules .next
npm install
npm run build
```

## Production Deployment

For production, you'll want to:

1. Run MIRRORS as a system service (systemd, supervisor, etc.)
2. Use a process manager for Next.js (PM2)
3. Set up nginx/Apache reverse proxy
4. Consider WebSocket connection instead of polling for lower latency

---

**SAAAM LLC** | MIRRORS Framework v2.0 | Structured Latent Manifold Architecture
