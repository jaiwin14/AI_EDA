// src/components/ParticlesBackground.tsx

import React, { useCallback } from 'react';
import Particles from 'react-tsparticles';
import type { Container, Engine } from 'tsparticles-engine';
import { loadSlim } from 'tsparticles-slim';


const ParticlesBackground: React.FC = () => {
  // This memoized function loads the tsParticles engine
  const particlesInit = useCallback(async (engine: Engine) => {
    // You can initiate the tsParticles instance (engine) here, adding custom shapes or presets
    // This loads the slim version of tsParticles, which is smaller and faster
    await loadSlim(engine);
  }, []);

  const particlesLoaded = useCallback(async (_container: Container | undefined) => {
    // This function is called when the particles container is loaded
    // You can perform actions with the container here if needed
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    void _container; // Mark as intentionally unused
  }, []);

  return (
    <Particles
      id="tsparticles"
      init={particlesInit}
      loaded={particlesLoaded}
      options={{
        background: {
          color: {
            value: 'transparent', // Make background transparent
          },
        },
        fpsLimit: 60,
        interactivity: {
          events: {
            onHover: {
              enable: true,
              mode: 'repulse', // Pushes particles away on hover
            },
            resize: true,
          },
          modes: {
            repulse: {
              distance: 100,
              duration: 0.4,
            },
          },
        },
        particles: {
          color: {
            value: '#4a5568', // A subtle gray color for particles
          },
          links: {
            color: '#4a5568', // Color of the lines connecting particles
            distance: 150,
            enable: true, // This creates the "neural network" look
            opacity: 0.2, // Very light transparency for links
            width: 1,
          },
          move: {
            direction: 'none',
            enable: true,
            outModes: {
              default: 'out',
            },
            random: false,
            speed: 1, // Slow, floating speed
            straight: false,
          },
          number: {
            density: {
              enable: true,
              area: 800,
            },
            value: 80, // Number of particles
          },
          opacity: {
            value: 0.3, // Very light transparency for particles
          },
          shape: {
            type: 'circle',
          },
          size: {
            value: { min: 1, max: 3 }, // Particles will have random sizes
          },
        },
        detectRetina: true,
      }}
    />
  );
};

export default ParticlesBackground;