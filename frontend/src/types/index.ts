export interface CameraPosition {
  x: number;
  y: number;
  z: number;
}

export interface CameraOrientation {
  qw: number;
  qx: number;
  qy: number;
  qz: number;
}

export interface PosePrediction {
  position: CameraPosition;
  orientation: CameraOrientation;
  positionError?: number;
  orientationError?: number;
  groundTruthPosition?: CameraPosition;
  groundTruthOrientation?: CameraOrientation;
}

export interface Room {
  id: string;
  name: string;
  type: string;
  location?: string;
  availableForLocalization: boolean;
  bounds?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface MapPoint {
  x: number;
  y: number;
  z?: number;
  type?: string;
}

export interface Area3Map {
  id: string;
  name: string;
  description: string;
  points: MapPoint[];
  rooms: Room[];
  previewImage?: string;
  bounds: {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
  };
}

export interface EvaluationResult {
  testSamples: number;
  successfulPredictions: number;
  failedPredictions: number;
  meanPositionError: number;
  medianPositionError: number;
  meanOrientationError: number;
  medianOrientationError: number;
}

export interface ModelInfo {
  name: string;
  architecture: string;
  inputType: string;
  inputSize: string;
  positionOutput: string[];
  orientationOutput: string[];
  backbone: string;
  preprocessing: string;
}

export interface SystemStatus {
  backend: 'connected' | 'not_connected' | 'checking';
  model: 'available' | 'not_available' | 'checking';
  map: 'available' | 'not_available' | 'checking';
  dataset: string;
}

export interface Phase {
  id: number;
  name: string;
  status: 'completed' | 'in_progress' | 'locked';
  description: string;
}
