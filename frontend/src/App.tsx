import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import CameraInput from './pages/CameraInput';
import PosePrediction from './pages/PosePrediction';
import Localization from './pages/Localization';
import Area3Map from './pages/Area3Map';
import Rooms from './pages/Rooms';
import ModelInfo from './pages/ModelInfo';
import Evaluation from './pages/Evaluation';
import FutureFeatures from './pages/FutureFeatures';

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="camera" element={<CameraInput />} />
          <Route path="pose" element={<PosePrediction />} />
          <Route path="localization" element={<Localization />} />
          <Route path="map" element={<Area3Map />} />
          <Route path="rooms" element={<Rooms />} />
          <Route path="model" element={<ModelInfo />} />
          <Route path="evaluation" element={<Evaluation />} />
          <Route path="future" element={<FutureFeatures />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
