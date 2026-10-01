# NaVIP Frontend

AI-Based Indoor Navigation for Visually Impaired Persons — Frontend Dashboard

## Tech Stack

- **React 18** — UI framework
- **Vite** — Build tool and dev server
- **TypeScript** — Type safety
- **React Router** — Client-side routing

## Project Structure

```
frontend/
├── src/
│   ├── components/       # Reusable UI components
│   │   ├── Layout.tsx
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   ├── Card.tsx
│   │   ├── PhaseStatus.tsx
│   │   ├── LoadingSpinner.tsx
│   │   ├── ErrorMessage.tsx
│   │   └── EmptyState.tsx
│   ├── pages/            # Page components
│   │   ├── Dashboard.tsx
│   │   ├── CameraInput.tsx
│   │   ├── PosePrediction.tsx
│   │   ├── Localization.tsx
│   │   ├── Area3Map.tsx
│   │   ├── Rooms.tsx
│   │   ├── ModelInfo.tsx
│   │   ├── Evaluation.tsx
│   │   └── FutureFeatures.tsx
│   ├── services/         # API service layer
│   │   ├── poseService.ts
│   │   ├── mapService.ts
│   │   ├── evaluationService.ts
│   │   └── roomService.ts
│   ├── api/              # API client
│   │   └── client.ts
│   ├── hooks/            # Custom React hooks
│   │   ├── SystemStatus.ts
│   │   └── PosePrediction.ts
│   ├── types/            # TypeScript type definitions
│   │   └── index.ts
│   ├── utils/            # Utility functions
│   │   ├── formatters.ts
│   │   └── constants.ts
│   ├── App.tsx           # Main app component
│   ├── main.tsx          # Entry point
│   └── styles.css        # Global styles
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── .env.example
```

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Environment Configuration

Create a `.env` file in the `frontend` directory:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Replace `http://localhost:8000` with your backend API URL.

### Running the Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Building for Production

```bash
npm run build
```

## Features

### Current Phase (Phase 4)

1. **Dashboard** — System overview and status
2. **Camera Input** — Upload RGB images for pose prediction
3. **Pose Prediction** — Display camera pose results
4. **Localization** — Visualize position on Area 3 map
5. **Area 3 Map** — Navigation map display
6. **Rooms** — Environment room information
7. **Model Info** — ResNet-18 PoseNet architecture details
8. **Evaluation** — Test results and metrics
9. **Future Features** — Locked upcoming phases (5-10)

### Future Phases (Locked)

- Phase 5: Navigation Graph
- Phase 6: Route Planning
- Phase 7: Real-Time Navigation
- Phase 8: Voice Guidance
- Phase 9: Obstacle / Hazard Assistance
- Phase 10: Complete Navigation System

## API Integration

The frontend expects the following backend endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/pose/predict` | POST | Predict camera pose from image |
| `/api/model/status` | GET | Check model availability |
| `/api/map/area3` | GET | Get Area 3 map data |
| `/api/map/area3/rooms` | GET | Get Area 3 rooms |
| `/api/map/area3/points` | GET | Get Area 3 map points |
| `/api/evaluation/results` | GET | Get evaluation results |
| `/api/rooms` | GET | Get all rooms |
| `/health` | GET | Backend health check |

If an endpoint is not available, the frontend will display appropriate error messages without fabricating data.

## Important Notes

- This frontend is designed for **Phase 4** backend functionality
- No fake data is presented as real backend output
- Future navigation features are visually locked
- The map is a **derived representation** from Area 3 geometry, not an official Stanford floor plan
- Evaluation results are experimental and not from the original NaVIP paper
