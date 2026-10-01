import React, { useState, useEffect } from 'react';
import { roomService } from '../services/roomService';
import Card from '../components/Card';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import EmptyState from '../components/EmptyState';
import type { Room } from '../types';

const Rooms: React.FC = () => {
  const [rooms, setRooms] = useState<Room[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRooms = async () => {
      setLoading(true);
      setError(null);

      try {
        const result = await roomService.getRooms();
        setRooms(result);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'Failed to load rooms data'
        );
      } finally {
        setLoading(false);
      }
    };

    fetchRooms();
  }, []);

  if (loading) {
    return <LoadingSpinner message="Loading rooms data..." />;
  }

  if (error) {
    return (
      <div className="page rooms-page">
        <div className="page-header">
          <h2>Rooms / Environment</h2>
          <p>Area 3 room information</p>
        </div>
        <ErrorMessage message={error} />
        <EmptyState
          icon="🚪"
          title="Rooms data unavailable"
          description="Backend endpoint not available. Room data will load when the backend API is connected."
        />
      </div>
    );
  }

  return (
    <div className="page rooms-page">
      <div className="page-header">
        <h2>Rooms / Environment</h2>
        <p>Area 3 room information</p>
      </div>

      {rooms.length > 0 ? (
        <div className="rooms-container">
          <Card title={`Area 3 Rooms (${rooms.length})`}>
            <div className="rooms-grid">
              {rooms.map((room) => (
                <div key={room.id} className="room-card">
                  <div className="room-header">
                    <h4>{room.name}</h4>
                    <span className={`room-badge ${room.availableForLocalization ? 'available' : 'unavailable'}`}>
                      {room.availableForLocalization ? 'Available' : 'Unavailable'}
                    </span>
                  </div>
                  <div className="room-details">
                    <p><strong>Type:</strong> {room.type}</p>
                    {room.location && <p><strong>Location:</strong> {room.location}</p>}
                    <p>
                      <strong>Localization:</strong>{' '}
                      {room.availableForLocalization ? 'Yes' : 'No'}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      ) : (
        <EmptyState
          icon="🚪"
          title="No rooms data"
          description="No room information available from the backend"
        />
      )}
    </div>
  );
};

export default Rooms;
