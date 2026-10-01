import React, { useState, useEffect } from 'react';
import { mapService } from '../services/mapService';
import Card from '../components/Card';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import EmptyState from '../components/EmptyState';
import type { Area3Map, MapPoint, Room } from '../types';

const Area3Map: React.FC = () => {
  const [mapData, setMapData] = useState<Area3Map | null>(null);
  const [rooms, setRooms] = useState<Room[]>([]);
  const [points, setPoints] = useState<MapPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMapData = async () => {
      setLoading(true);
      setError(null);

      try {
        const [mapResult, roomsResult, pointsResult] = await Promise.allSettled([
          mapService.getArea3Map(),
          mapService.getArea3Rooms(),
          mapService.getArea3Points(),
        ]);

        if (mapResult.status === 'fulfilled') {
          setMapData(mapResult.value);
        } else {
          throw new Error('Failed to load map data');
        }

        if (roomsResult.status === 'fulfilled') {
          setRooms(roomsResult.value);
        }

        if (pointsResult.status === 'fulfilled') {
          setPoints(pointsResult.value);
        }
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'Failed to load map data'
        );
      } finally {
        setLoading(false);
      }
    };

    fetchMapData();
  }, []);

  if (loading) {
    return <LoadingSpinner message="Loading Area 3 map data..." />;
  }

  if (error) {
    return (
      <div className="page map-page">
        <div className="page-header">
          <h2>Area 3 Map</h2>
          <p>Stanford 2D-3D-S Area 3 navigation map</p>
        </div>
        <ErrorMessage message={error} />
        <EmptyState
          icon="🗺️"
          title="Map data unavailable"
          description="Backend endpoint not available. The map data will load when the backend API is connected."
        />
      </div>
    );
  }

  return (
    <div className="page map-page">
      <div className="page-header">
        <h2>Area 3 Map</h2>
        <p>Stanford 2D-3D-S Area 3 navigation map</p>
      </div>

      <Card title="Area 3 Derived Navigation Map">
        <div className="map-container">
          <div className="map-placeholder">
            <div className="map-grid">
              <div className="map-room office">
                <span>Office</span>
              </div>
              <div className="map-room hallway">
                <span>Hallway</span>
              </div>
              <div className="map-room office2">
                <span>Office</span>
              </div>
              <div className="map-room lounge">
                <span>Lounge</span>
              </div>
              <div className="map-room conference">
                <span>Conference</span>
              </div>
              <div className="map-room wc">
                <span>WC</span>
              </div>
              <div className="map-room storage">
                <span>Storage</span>
              </div>
            </div>
            <p className="map-note">
              Area 3 Derived Navigation Map — Not an official Stanford floor plan
            </p>
          </div>
        </div>
      </Card>

      <Card title="Map Information">
        <div className="map-info">
          <div className="info-row">
            <span className="info-label">Map Name:</span>
            <span className="info-value">{mapData?.name || 'Area 3'}</span>
          </div>
          <div className="info-row">
            <span className="info-label">Description:</span>
            <span className="info-value">
              {mapData?.description || 'Derived from Stanford Area 3 3D geometry'}
            </span>
          </div>
          <div className="info-row">
            <span className="info-label">Total Points:</span>
            <span className="info-value">{points.length || 'N/A'}</span>
          </div>
          <div className="info-row">
            <span className="info-label">Total Rooms:</span>
            <span className="info-value">{rooms.length || 'N/A'}</span>
          </div>
        </div>
      </Card>

      {rooms.length > 0 && (
        <Card title="Rooms">
          <div className="rooms-grid">
            {rooms.map((room) => (
              <div key={room.id} className="room-card">
                <h4>{room.name}</h4>
                <p><strong>Type:</strong> {room.type}</p>
                {room.location && <p><strong>Location:</strong> {room.location}</p>}
                <p>
                  <strong>Available:</strong>{' '}
                  {room.availableForLocalization ? 'Yes' : 'No'}
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

export default Area3Map;
