import React, { useState, useRef } from 'react';
import { usePosePrediction } from '../hooks/PosePrediction';
import Card from '../components/Card';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import EmptyState from '../components/EmptyState';

const CameraInput: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { predictPose, prediction, loading, error, clearPrediction } = usePosePrediction();

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
      }
      setSelectedImage(file);
      clearPrediction();

      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    clearPrediction();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handlePredict = async () => {
    if (selectedImage) {
      await predictPose(selectedImage);
    }
  };

  return (
    <div className="page camera-page">
      <div className="page-header">
        <h2>Camera / Image Input</h2>
        <p>Upload an RGB image for camera pose prediction</p>
      </div>

      <Card title="Image Upload">
        <div className="upload-section">
          <div className="upload-area">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              className="file-input"
              id="image-upload"
              aria-label="Upload RGB image"
            />
            <label htmlFor="image-upload" className="upload-label">
              <span className="upload-icon">📷</span>
              <span className="upload-text">Click to select an image</span>
              <span className="upload-hint">Supports JPG, PNG, BMP</span>
            </label>
          </div>

          {imagePreview && (
            <div className="image-preview">
              <img src={imagePreview} alt="Selected input" />
              <button
                onClick={handleRemoveImage}
                className="remove-image-btn"
                aria-label="Remove selected image"
              >
                ×
              </button>
            </div>
          )}

          {selectedImage && (
            <div className="image-info">
              <p><strong>File:</strong> {selectedImage.name}</p>
              <p><strong>Size:</strong> {(selectedImage.size / 1024).toFixed(2)} KB</p>
            </div>
          )}

          <button
            onClick={handlePredict}
            disabled={!selectedImage || loading}
            className="predict-btn"
          >
            {loading ? 'Processing...' : 'Predict Camera Pose'}
          </button>
        </div>
      </Card>

      {loading && <LoadingSpinner message="Predicting camera pose..." />}

      {error && <ErrorMessage message={error} onDismiss={clearPrediction} />}

      {prediction && (
        <Card title="Prediction Result">
          <div className="prediction-result">
            <div className="pose-section">
              <h4>Predicted Camera Position</h4>
              <div className="pose-values">
                <div className="pose-item">
                  <span className="pose-label">X:</span>
                  <span className="pose-value">{prediction.position.x.toFixed(4)} m</span>
                </div>
                <div className="pose-item">
                  <span className="pose-label">Y:</span>
                  <span className="pose-value">{prediction.position.y.toFixed(4)} m</span>
                </div>
                <div className="pose-item">
                  <span className="pose-label">Z:</span>
                  <span className="pose-value">{prediction.position.z.toFixed(4)} m</span>
                </div>
              </div>
            </div>

            <div className="pose-section">
              <h4>Predicted Orientation</h4>
              <div className="pose-values">
                <div className="pose-item">
                  <span className="pose-label">qw:</span>
                  <span className="pose-value">{prediction.orientation.qw.toFixed(4)}</span>
                </div>
                <div className="pose-item">
                  <span className="pose-label">qx:</span>
                  <span className="pose-value">{prediction.orientation.qx.toFixed(4)}</span>
                </div>
                <div className="pose-item">
                  <span className="pose-label">qy:</span>
                  <span className="pose-value">{prediction.orientation.qy.toFixed(4)}</span>
                </div>
                <div className="pose-item">
                  <span className="pose-label">qz:</span>
                  <span className="pose-value">{prediction.orientation.qz.toFixed(4)}</span>
                </div>
              </div>
            </div>

            {prediction.groundTruthPosition && (
              <div className="pose-section">
                <h4>Ground Truth Position</h4>
                <div className="pose-values">
                  <div className="pose-item">
                    <span className="pose-label">X:</span>
                    <span className="pose-value">{prediction.groundTruthPosition.x.toFixed(4)} m</span>
                  </div>
                  <div className="pose-item">
                    <span className="pose-label">Y:</span>
                    <span className="pose-value">{prediction.groundTruthPosition.y.toFixed(4)} m</span>
                  </div>
                  <div className="pose-item">
                    <span className="pose-label">Z:</span>
                    <span className="pose-value">{prediction.groundTruthPosition.z.toFixed(4)} m</span>
                  </div>
                </div>
              </div>
            )}

            {prediction.groundTruthOrientation && (
              <div className="pose-section">
                <h4>Ground Truth Orientation</h4>
                <div className="pose-values">
                  <div className="pose-item">
                    <span className="pose-label">qw:</span>
                    <span className="pose-value">{prediction.groundTruthOrientation.qw.toFixed(4)}</span>
                  </div>
                  <div className="pose-item">
                    <span className="pose-label">qx:</span>
                    <span className="pose-value">{prediction.groundTruthOrientation.qx.toFixed(4)}</span>
                  </div>
                  <div className="pose-item">
                    <span className="pose-label">qy:</span>
                    <span className="pose-value">{prediction.groundTruthOrientation.qy.toFixed(4)}</span>
                  </div>
                  <div className="pose-item">
                    <span className="pose-label">qz:</span>
                    <span className="pose-value">{prediction.groundTruthOrientation.qz.toFixed(4)}</span>
                  </div>
                </div>
              </div>
            )}

            {prediction.positionError !== undefined && (
              <div className="error-section">
                <h4>Position Error</h4>
                <p>{prediction.positionError.toFixed(4)} m</p>
              </div>
            )}

            {prediction.orientationError !== undefined && (
              <div className="error-section">
                <h4>Orientation Error</h4>
                <p>{prediction.orientationError.toFixed(2)}°</p>
              </div>
            )}
          </div>
        </Card>
      )}

      {!selectedImage && !loading && !prediction && (
        <EmptyState
          icon="📷"
          title="No image selected"
          description="Upload an RGB image to begin pose prediction"
        />
      )}
    </div>
  );
};

export default CameraInput;
