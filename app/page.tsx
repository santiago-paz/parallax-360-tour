"use client";

import { useEffect, useRef, useState } from "react";
import Parallax360Viewer from "./components/parallax360/Parallax360Viewer";
import { SCENES, FOV } from "./scenes";

export default function Home() {
  const [activeId, setActiveId] = useState(SCENES[0]?.id ?? "");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const mainRef = useRef<HTMLElement>(null);
  const active = SCENES.find((s) => s.id === activeId) ?? SCENES[0];

  useEffect(() => {
    const onChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const toggleFullscreen = () => {
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else if (mainRef.current?.requestFullscreen) {
      mainRef.current.requestFullscreen().catch(() => {});
    }
  };

  if (!active) {
    return (
      <main className="w-screen h-screen grid place-items-center bg-black text-white">
        <p>No scenes registered in <code>app/scenes.ts</code>.</p>
      </main>
    );
  }

  return (
    <main
      ref={mainRef}
      className="w-screen h-screen bg-black overflow-hidden relative"
      style={{ touchAction: "none" }}
    >
      <Parallax360Viewer
        config={{
          imageSrc: active.imageSrc,
          fov: FOV,
          depthSrc: active.depthSrc,
          layered: active.layered,
        }}
      />

      {SCENES.length > 1 && (
        <nav className="absolute top-4 left-1/2 -translate-x-1/2 z-10 flex gap-2 flex-wrap justify-center max-w-[80vw]">
          {SCENES.map((s) => {
            const isActive = s.id === active.id;
            return (
              <button
                key={s.id}
                onClick={() => setActiveId(s.id)}
                className={
                  "px-3 py-1.5 text-sm font-medium rounded-md transition-colors " +
                  (isActive
                    ? "bg-white text-black"
                    : "bg-black/40 text-white hover:bg-black/60 backdrop-blur-sm")
                }
              >
                {s.label}
              </button>
            );
          })}
        </nav>
      )}

      <button
        onClick={toggleFullscreen}
        aria-label={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
        title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
        className="absolute top-4 right-4 z-10 p-2 rounded-md bg-black/40 text-white hover:bg-black/60 backdrop-blur-sm transition-colors"
      >
        {isFullscreen ? (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M8 3v5H3" />
            <path d="M16 3v5h5" />
            <path d="M8 21v-5H3" />
            <path d="M16 21v-5h5" />
          </svg>
        ) : (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M3 8V3h5" />
            <path d="M21 8V3h-5" />
            <path d="M3 16v5h5" />
            <path d="M21 16v5h-5" />
          </svg>
        )}
      </button>
    </main>
  );
}
