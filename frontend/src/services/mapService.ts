import apiClient from '../api/client';
import type { Area3Map, MapPoint, Room } from '../types';

export const mapService = {
  async getArea3Map(): Promise<Area3Map> {
    return apiClient.get<Area3Map>('/api/map/area3');
  },

  async getArea3Rooms(): Promise<Room[]> {
    return apiClient.get<Room[]>('/api/map/area3/rooms');
  },

  async getArea3Points(): Promise<MapPoint[]> {
    return apiClient.get<MapPoint[]>('/api/map/area3/points');
  },

  async checkMapAvailability(): Promise<boolean> {
    try {
      await apiClient.get('/api/map/area3');
      return true;
    } catch {
      return false;
    }
  },
};
