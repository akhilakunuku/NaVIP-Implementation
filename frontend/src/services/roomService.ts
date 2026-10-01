import apiClient from '../api/client';
import type { Room } from '../types';

export const roomService = {
  async getRooms(): Promise<Room[]> {
    return apiClient.get<Room[]>('/api/rooms');
  },

  async getRoomById(id: string): Promise<Room> {
    return apiClient.get<Room>(`/api/rooms/${id}`);
  },
};
