export const PHASES = [
  {
    id: 1,
    name: 'Project Setup',
    status: 'completed' as const,
    description: 'Project and environment setup',
  },
  {
    id: 2,
    name: 'Dataset Preparation',
    status: 'completed' as const,
    description: 'Stanford 2D-3D-S Area 3 dataset preparation',
  },
  {
    id: 3,
    name: 'AI Pose Prediction',
    status: 'completed' as const,
    description: 'ResNet-18 PoseNet camera pose prediction',
  },
  {
    id: 4,
    name: 'Map / Spatial Representation',
    status: 'completed' as const,
    description: '3D environment and camera-pose coordinate consistency verification',
  },
  {
    id: 5,
    name: 'Navigation Graph',
    status: 'locked' as const,
    description: 'Convert map representation into navigation graph',
  },
  {
    id: 6,
    name: 'Route Planning',
    status: 'locked' as const,
    description: 'Select destination and calculate route',
  },
  {
    id: 7,
    name: 'Real-Time Navigation',
    status: 'locked' as const,
    description: 'Continuous camera input and pose estimation',
  },
  {
    id: 8,
    name: 'Voice Guidance',
    status: 'locked' as const,
    description: 'Turn-by-turn audio instructions',
  },
  {
    id: 9,
    name: 'Obstacle / Hazard Assistance',
    status: 'locked' as const,
    description: 'Object detection and hazard warnings',
  },
  {
    id: 10,
    name: 'Complete Navigation System',
    status: 'locked' as const,
    description: 'Full integration of all components',
  },
];

export const MODEL_INFO = {
  name: 'ResNet-18 PoseNet',
  architecture: 'ResNet-18 + Dual Head',
  inputType: 'RGB Image',
  inputSize: '224 × 224',
  positionOutput: ['X', 'Y', 'Z'],
  orientationOutput: ['qw', 'qx', 'qy', 'qz'],
  backbone: 'ResNet-18',
  preprocessing: 'Image resize + ImageNet normalization',
};

export const EVALUATION_DATA = {
  testSamples: 559,
  successfulPredictions: 559,
  failedPredictions: 0,
  meanPositionError: 6.3581,
  medianPositionError: 6.3157,
  meanOrientationError: 67.46,
  medianOrientationError: 56.0,
};
