import React from 'react';
import Card from '../components/Card';
import { MODEL_INFO } from '../utils/constants';

const ModelInfo: React.FC = () => {
  return (
    <div className="page model-info-page">
      <div className="page-header">
        <h2>Model Information</h2>
        <p>ResNet-18 PoseNet architecture details</p>
      </div>

      <Card title="Model Overview">
        <div className="model-overview">
          <div className="info-row">
            <span className="info-label">Model:</span>
            <span className="info-value">{MODEL_INFO.name}</span>
          </div>
          <div className="info-row">
            <span className="info-label">Architecture:</span>
            <span className="info-value">{MODEL_INFO.architecture}</span>
          </div>
          <div className="info-row">
            <span className="info-label">Backbone:</span>
            <span className="info-value">{MODEL_INFO.backbone}</span>
          </div>
          <div className="info-row">
            <span className="info-label">Checkpoint:</span>
            <span className="info-value">navip_stanford_best.pth</span>
          </div>
        </div>
      </Card>

      <Card title="Input Specifications">
        <div className="input-specs">
          <div className="info-row">
            <span className="info-label">Input Type:</span>
            <span className="info-value">{MODEL_INFO.inputType}</span>
          </div>
          <div className="info-row">
            <span className="info-label">Input Size:</span>
            <span className="info-value">{MODEL_INFO.inputSize}</span>
          </div>
          <div className="info-row">
            <span className="info-label">Preprocessing:</span>
            <span className="info-value">{MODEL_INFO.preprocessing}</span>
          </div>
        </div>
      </Card>

      <Card title="Output Specifications">
        <div className="output-specs">
          <div className="output-section">
            <h4>Position Output</h4>
            <div className="output-values">
              {MODEL_INFO.positionOutput.map((axis) => (
                <span key={axis} className="output-badge">{axis}</span>
              ))}
            </div>
          </div>
          <div className="output-section">
            <h4>Orientation Output</h4>
            <div className="output-values">
              {MODEL_INFO.orientationOutput.map((axis) => (
                <span key={axis} className="output-badge">{axis}</span>
              ))}
            </div>
          </div>
        </div>
      </Card>

      <Card title="Architecture Pipeline">
        <div className="pipeline">
          <div className="pipeline-step">
            <span className="step-number">1</span>
            <span className="step-label">RGB Image</span>
          </div>
          <div className="pipeline-arrow">↓</div>
          <div className="pipeline-step">
            <span className="step-number">2</span>
            <span className="step-label">ResNet-18</span>
          </div>
          <div className="pipeline-arrow">↓</div>
          <div className="pipeline-step">
            <span className="step-number">3</span>
            <span className="step-label">Feature Representation</span>
          </div>
          <div className="pipeline-arrow">↓</div>
          <div className="pipeline-step">
            <span className="step-number">4</span>
            <span className="step-label">Position Head + Orientation Head</span>
          </div>
          <div className="pipeline-arrow">↓</div>
          <div className="pipeline-step">
            <span className="step-number">5</span>
            <span className="step-label">Camera Pose</span>
          </div>
        </div>
      </Card>

      <Card title="Implementation Note">
        <p className="implementation-note">
          This is our current implementation. The model architecture and training
          approach are based on the NaVIP base paper methodology, adapted for
          Stanford 2D-3D-S Area 3 dataset.
        </p>
      </Card>
    </div>
  );
};

export default ModelInfo;
