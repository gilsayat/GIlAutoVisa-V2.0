import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  BackgroundColorOption,
  ComplianceReport,
  EdgeFeatherMode,
  getLayoutSpec,
  PrintLayout,
  ProcessingRequest,
  VisaPhotoProcessor,
} from './services/VisaPhotoProcessor';
import { PrintPreviewModal } from './components/PrintPreviewModal';

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(0)} KB`;
  return `${(kb / 1024).toFixed(2)} MB`;
}

export default function App() {
  const processorRef = useRef<VisaPhotoProcessor>(new VisaPhotoProcessor());
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const canvasBoxRef = useRef<HTMLDivElement | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [isEngineReady, setIsEngineReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [showGuides, setShowGuides] = useState(true);

  const [statusMessage, setStatusMessage] = useState(
    'Loading AI vision models...'
  );
  const [statusColor, setStatusColor] = useState('#88888D');

  const [layout, setLayout] = useState<PrintLayout>(PrintLayout.Print4x6);
  const [bgColor, setBgColor] = useState<BackgroundColorOption>(
    BackgroundColorOption.White
  );
  const [softEdge, setSoftEdge] = useState(true);

  const [compliance, setCompliance] = useState<ComplianceReport | null>(null);
  const [transformHud, setTransformHud] = useState<{
    zoomPct: number;
    tiltDeg: number;
  }>({ zoomPct: 100, tiltDeg: 0 });

  const [lastFileName, setLastFileName] = useState<string>('');
  const [lastOutputUrl, setLastOutputUrl] = useState<string | null>(null);
  const [lastOutputName, setLastOutputName] = useState<string>('');
  const [lastOutputSize, setLastOutputSize] = useState<string>('');

  // Print sheet preview modal state
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [sheetPreviewCanvas, setSheetPreviewCanvas] =
    useState<HTMLCanvasElement | null>(null);

  // Smart Canvas transform refs for synchronous 60fps interaction
  const panXRef = useRef(0);
  const panYRef = useRef(0);
  const zoomScaleRef = useRef(0);
  const tiltAngleRef = useRef(0);
  const dragStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const isDraggingRef = useRef(false);
  const isEngineReadyRef = useRef(false);

  const settingsRef = useRef({
    layout,
    bgColor,
    softEdge,
  });

  useEffect(() => {
    settingsRef.current = { layout, bgColor, softEdge };
  }, [layout, bgColor, softEdge]);

  useEffect(() => {
    isEngineReadyRef.current = isEngineReady;
  }, [isEngineReady]);

  // Pre-warm MediaPipe AI models in background on mount
  useEffect(() => {
    let active = true;
    processorRef.current.preWarmModels().then((aiReady) => {
      if (!active || isEngineReadyRef.current) return;
      if (aiReady) {
        setStatusMessage('AI Engine loaded and ready.');
        setStatusColor('#00FF7F');
      } else {
        setStatusMessage('Ready to process (offline mode).');
        setStatusColor('#00FF7F');
      }
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (lastOutputUrl) URL.revokeObjectURL(lastOutputUrl);
    };
  }, [lastOutputUrl]);

  const getCurrentRequest = useCallback((): ProcessingRequest => {
    const s = settingsRef.current;
    return {
      layout: s.layout,
      bgColor: s.bgColor,
      edgeMode: s.softEdge ? EdgeFeatherMode.Soft : EdgeFeatherMode.Sharp,
      tilt: tiltAngleRef.current,
      zoomOffset: zoomScaleRef.current,
      panX: panXRef.current,
      panY: panYRef.current,
    };
  }, []);

  const renderLivePreview = useCallback(() => {
    if (!isEngineReadyRef.current) return;
    const outCanvas = previewCanvasRef.current;
    if (!outCanvas) return;

    try {
      const req = getCurrentRequest();
      const frame = processorRef.current.renderFrame(req, true);
      const ctx = outCanvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, outCanvas.width, outCanvas.height);
      ctx.drawImage(frame, 0, 0, outCanvas.width, outCanvas.height);

      const report = processorRef.current.computeCompliance(req);
      setCompliance(report);
      setTransformHud({
        zoomPct: Math.round(100 + zoomScaleRef.current),
        tiltDeg: Number(tiltAngleRef.current.toFixed(1)),
      });
    } catch {
      // Ignore transient render errors
    }
  }, [getCurrentRequest]);

  // Re-render whenever settings change
  useEffect(() => {
    if (isEngineReady) {
      renderLivePreview();
    }
  }, [layout, bgColor, softEdge, isEngineReady, renderLivePreview]);

  const resetManualOffsets = useCallback(() => {
    panXRef.current = 0;
    panYRef.current = 0;
    zoomScaleRef.current = 0;
    tiltAngleRef.current = 0;
    if (isEngineReadyRef.current) {
      renderLivePreview();
    }
  }, [renderLivePreview]);

  const handleImageSelection = useCallback(
    async (file: File | undefined | null) => {
      if (!file) return;
      if (!file.type.startsWith('image/')) {
        setStatusMessage('Please select a valid JPG or PNG image.');
        setStatusColor('#FF5C5C');
        return;
      }

      setLastFileName(file.name);
      if (lastOutputUrl) {
        URL.revokeObjectURL(lastOutputUrl);
        setLastOutputUrl(null);
        setLastOutputName('');
        setLastOutputSize('');
      }
      setIsPreviewOpen(false);
      setSheetPreviewCanvas(null);
      setIsEngineReady(false);
      isEngineReadyRef.current = false;
      setCompliance(null);

      panXRef.current = 0;
      panYRef.current = 0;
      zoomScaleRef.current = 0;
      tiltAngleRef.current = 0;

      setIsProcessing(true);
      setStatusMessage('AI Processing: Landmarks, Matting & De-Fringing...');
      setStatusColor('#FFD700');

      await new Promise((r) => requestAnimationFrame(r));

      try {
        await processorRef.current.initializeMasterFile(file);
        isEngineReadyRef.current = true;
        setIsEngineReady(true);

        const sourceLabel =
          processorRef.current.getDetectionSource() === 'ai'
            ? 'AI Crop Ready. Drag to Pan, Scroll to Zoom.'
            : 'Ready (Offline Crop). Drag to Pan, Scroll to Zoom.';
        setStatusMessage(sourceLabel);
        setStatusColor('#00FF7F');
        requestAnimationFrame(() => renderLivePreview());
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Unknown initialization error';
        setStatusMessage(`Initialization Failed: ${message}`);
        setStatusColor('#FF5C5C');
      } finally {
        setIsProcessing(false);
      }
    },
    [lastOutputUrl, renderLivePreview]
  );

  // Attach non-passive wheel listener for smooth zoom & shift+scroll tilt
  useEffect(() => {
    const el = canvasBoxRef.current;
    if (!el) return;

    const handleWheel = (e: WheelEvent) => {
      if (!isEngineReadyRef.current) return;
      e.preventDefault();

      if (e.shiftKey) {
        const tiltDelta = e.deltaY < 0 ? 0.5 : -0.5;
        tiltAngleRef.current = Math.max(
          -25,
          Math.min(25, tiltAngleRef.current + tiltDelta)
        );
      } else {
        const zoomDelta = e.deltaY < 0 ? 2.0 : -2.0;
        zoomScaleRef.current = Math.max(
          -25,
          Math.min(45, zoomScaleRef.current + zoomDelta)
        );

        const rect = el.getBoundingClientRect();
        const cursorNormX =
          (e.clientX - rect.left - rect.width / 2) / rect.width;
        const cursorNormY =
          (e.clientY - rect.top - rect.height / 2) / rect.height;

        if (zoomDelta > 0) {
          panXRef.current -= cursorNormX * 2.0;
          panYRef.current -= cursorNormY * 2.0;
        } else {
          panXRef.current += cursorNormX * 2.0;
          panYRef.current += cursorNormY * 2.0;
        }
      }

      renderLivePreview();
    };

    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => el.removeEventListener('wheel', handleWheel);
  }, [renderLivePreview]);

  // Precision keyboard shortcuts: Arrows pan, +/- zoom, G guides, R reset, Escape close modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isPreviewOpen) {
        setIsPreviewOpen(false);
        return;
      }

      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'SELECT' ||
          target.tagName === 'TEXTAREA')
      ) {
        return;
      }

      if (!isEngineReadyRef.current) return;

      const step = e.shiftKey ? 2.0 : 0.5;
      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          panXRef.current -= step;
          renderLivePreview();
          break;
        case 'ArrowRight':
          e.preventDefault();
          panXRef.current += step;
          renderLivePreview();
          break;
        case 'ArrowUp':
          e.preventDefault();
          panYRef.current -= step;
          renderLivePreview();
          break;
        case 'ArrowDown':
          e.preventDefault();
          panYRef.current += step;
          renderLivePreview();
          break;
        case '+':
        case '=':
          e.preventDefault();
          zoomScaleRef.current = Math.min(45, zoomScaleRef.current + 2.0);
          renderLivePreview();
          break;
        case '-':
        case '_':
          e.preventDefault();
          zoomScaleRef.current = Math.max(-25, zoomScaleRef.current - 2.0);
          renderLivePreview();
          break;
        case 'g':
        case 'G':
          e.preventDefault();
          setShowGuides((prev) => !prev);
          break;
        case 'r':
        case 'R':
          e.preventDefault();
          resetManualOffsets();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPreviewOpen, renderLivePreview, resetManualOffsets]);

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isEngineReady || e.button !== 0) return;
    isDraggingRef.current = true;
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDraggingRef.current || !isEngineReadyRef.current) return;
    const rect = canvasBoxRef.current?.getBoundingClientRect();
    const size = rect ? rect.width : 600;

    const deltaX = ((e.clientX - dragStartRef.current.x) / size) * 100.0;
    const deltaY = ((e.clientY - dragStartRef.current.y) / size) * 100.0;

    panXRef.current += deltaX;
    panYRef.current += deltaY;
    dragStartRef.current = { x: e.clientX, y: e.clientY };

    renderLivePreview();
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDraggingRef.current) return;
    isDraggingRef.current = false;
    setIsDragging(false);
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    renderLivePreview();
  };

  const handleCanvasClick = () => {
    if (!isEngineReady && !isProcessing) {
      fileInputRef.current?.click();
    }
  };

  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'copy';
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      void handleImageSelection(file);
    }
  };

  const handleBrowseClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    fileInputRef.current?.click();
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      void handleImageSelection(file);
    }
    e.target.value = '';
  };

  const triggerDownload = (url: string, filename: string) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleOpenPreview = () => {
    if (!isEngineReady || isProcessing) return;
    try {
      const req = getCurrentRequest();
      const canvas = processorRef.current.renderSheetPreview(req, 1100);
      setSheetPreviewCanvas(canvas);
      setIsPreviewOpen(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Preview failed';
      setStatusMessage(`Error: ${message}`);
      setStatusColor('#FF5C5C');
    }
  };

  const handleGeneratePrint = async () => {
    if (!isEngineReady || isProcessing) return;

    setIsProcessing(true);
    setStatusMessage('Compiling 600 DPI Master Print Sheet...');
    setStatusColor('#0066FF');

    await new Promise((r) => requestAnimationFrame(r));

    try {
      const req = getCurrentRequest();
      const blob = await processorRef.current.generateHighResPrint(req);

      if (lastOutputUrl) URL.revokeObjectURL(lastOutputUrl);
      const url = URL.createObjectURL(blob);
      const baseName = lastFileName.replace(/\.[^/.]+$/, '') || 'Portrait';
      const outputFilename = `${baseName}_VisaReady.jpg`;
      const sizeStr = formatBytes(blob.size);

      setLastOutputUrl(url);
      setLastOutputName(outputFilename);
      setLastOutputSize(sizeStr);
      triggerDownload(url, outputFilename);

      setStatusMessage(`Generation Complete (${sizeStr}) — Ready for Lab Printer.`);
      setStatusColor('#10B981');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Generation failed';
      setStatusMessage(`Error: ${message}`);
      setStatusColor('#FF5C5C');
    } finally {
      setIsProcessing(false);
    }
  };

  const canvasBg =
    bgColor === BackgroundColorOption.White ? '#FFFFFF' : '#F2F2F2';
  const currentSpec = getLayoutSpec(layout);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#121214] text-white select-none">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,.jpg,.jpeg,.png"
        className="hidden"
        onChange={handleFileInputChange}
      />

      {/* LEFT PANE: Smart Interactive Canvas */}
      <div className="relative flex flex-1 flex-col items-center justify-center bg-[#0A0A0C] p-6 overflow-hidden">
        {/* Top Indeterminate Progress Bar */}
        {isProcessing && (
          <div className="absolute top-0 left-0 right-0 h-1 overflow-hidden bg-transparent z-20">
            <div className="h-full w-1/3 bg-[#1E90FF] animate-indeterminate" />
          </div>
        )}

        {/* Top Interaction Hint & Minimalist Guide Toggle */}
        <div className="absolute top-[14px] left-6 right-6 flex items-center justify-between text-[11px] text-[#55555A]">
          <div className="w-[100px] font-mono tabular-nums text-[#6E6E74]">
            {isEngineReady
              ? `${transformHud.zoomPct}% · ${transformHud.tiltDeg >= 0 ? '+' : ''}${transformHud.tiltDeg.toFixed(1)}°`
              : ''}
          </div>

          <div className="text-center pointer-events-none">
            Shift+Scroll: Tilt &nbsp;|&nbsp; Scroll: Zoom &nbsp;|&nbsp; Drag:
            Pan
          </div>

          <div className="w-[100px] flex justify-end">
            {isEngineReady && (
              <button
                type="button"
                onClick={() => setShowGuides((prev) => !prev)}
                className="text-[11px] text-[#888890] hover:text-white cursor-pointer transition-colors whitespace-nowrap"
                title="Toggle U.S. Visa alignment overlay (G)"
              >
                {showGuides ? 'Hide Guides (G)' : 'Show Guides (G)'}
              </button>
            )}
          </div>
        </div>

        {/* The 1:1 600x600 Smart Canvas */}
        <div
          ref={canvasBoxRef}
          onClick={handleCanvasClick}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          style={{
            backgroundColor: isEngineReady ? canvasBg : '#FFFFFF',
            borderColor: isDragOver ? '#1E90FF' : 'transparent',
          }}
          className={`relative aspect-square w-full max-w-[600px] max-h-[min(600px,calc(100vh-88px))] rounded-[2px] border-2 overflow-hidden touch-none ${
            isEngineReady
              ? isDragging
                ? 'cursor-grabbing'
                : 'cursor-grab'
              : 'cursor-pointer'
          }`}
        >
          {/* Initial State UI */}
          {!isEngineReady && !isProcessing && (
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="text-[56px] leading-none mb-[15px] text-[#1E90FF]">
                📸
              </div>
              <div className="text-[18px] font-bold text-[#888890]">
                Drop Portrait Here
              </div>
              <button
                type="button"
                onClick={handleBrowseClick}
                className="mt-[20px] w-[100px] h-[30px] bg-[#1E1E22] hover:bg-[#2A2A30] text-white text-[12px] border border-[#333338] cursor-pointer transition-colors"
              >
                Browse...
              </button>
            </div>
          )}

          {/* Real-Time Cutout Image Canvas */}
          <canvas
            ref={previewCanvasRef}
            width={600}
            height={600}
            className={`absolute inset-0 h-full w-full pointer-events-none ${
              isEngineReady ? 'block' : 'hidden'
            }`}
          />

          {/* Exact U.S. Visa Composition Template Overlay */}
          {isEngineReady && showGuides && (
            <svg
              viewBox="0 0 600 600"
              className="absolute inset-0 h-full w-full pointer-events-none opacity-95"
              xmlns="http://www.w3.org/2000/svg"
            >
              {/* Safe Eye Zone (56% to 69% from bottom = 44% to 31% from top) */}
              <rect
                x="0"
                y="186"
                width="600"
                height="78"
                fill="rgba(0, 255, 0, 0.28)"
              />

              {/* Center Line & Eye Line Guide (Y=225 = 62.5% from bottom) */}
              <line
                x1="260"
                y1="225"
                x2="340"
                y2="225"
                stroke="rgba(0, 0, 0, 0.88)"
                strokeWidth="2"
                strokeDasharray="4 4"
              />
              <line
                x1="300"
                y1="0"
                x2="300"
                y2="600"
                stroke="rgba(255, 0, 0, 0.63)"
                strokeWidth="1"
                strokeDasharray="4 4"
              />

              {/* Head Height Bounds (Min 50% = 300px, Max 69% = 414px) */}
              <line
                x1="0"
                y1="42"
                x2="600"
                y2="42"
                stroke="rgba(0, 0, 0, 0.63)"
                strokeWidth="1"
                strokeDasharray="4 4"
              />
              <line
                x1="0"
                y1="150"
                x2="600"
                y2="150"
                stroke="rgba(0, 0, 0, 0.63)"
                strokeWidth="1"
                strokeDasharray="4 4"
              />

              <line
                x1="565"
                y1="42"
                x2="565"
                y2="450"
                stroke="rgba(0, 0, 0, 0.82)"
                strokeWidth="2"
              />
              <line
                x1="550"
                y1="42"
                x2="580"
                y2="42"
                stroke="rgba(0, 0, 0, 0.82)"
                strokeWidth="2"
              />
              <line
                x1="550"
                y1="150"
                x2="580"
                y2="150"
                stroke="rgba(0, 0, 0, 0.82)"
                strokeWidth="2"
              />

              <line
                x1="0"
                y1="450"
                x2="600"
                y2="450"
                stroke="rgba(0, 0, 0, 0.63)"
                strokeWidth="1"
                strokeDasharray="4 4"
              />
              <line
                x1="550"
                y1="450"
                x2="580"
                y2="450"
                stroke="rgba(0, 0, 0, 0.82)"
                strokeWidth="2"
              />

              {/* Head Silhouette Overlay */}
              <path
                d="M 210,320 C 210,480 270,450 300,450 C 330,450 390,480 390,320 C 390,120 370,80 300,80 C 230,80 210,120 210,320 Z"
                stroke="rgba(0, 0, 0, 0.75)"
                strokeWidth="2"
                strokeDasharray="8 8"
                fill="none"
              />

              {/* Border Frame */}
              <rect
                x="1"
                y="1"
                width="598"
                height="598"
                stroke="rgba(0, 0, 0, 0.314)"
                strokeWidth="2"
                fill="none"
              />
            </svg>
          )}
        </div>
      </div>

      {/* RIGHT PANE: Minimalist Studio Automation Panel */}
      <aside className="flex w-[320px] shrink-0 flex-col justify-between bg-[#18181A] border-l border-[#252528] p-[25px] overflow-y-auto">
        <div>
          <h1 className="text-[26px] font-black leading-tight text-white mb-[5px]">
            AUTO ID
          </h1>
          <p className="text-[11px] text-[#77777C] mb-[26px]">
            Studio Automation Engine
          </p>

          <div className="text-[10px] font-bold tracking-wide text-[#66666B] mb-[10px]">
            OUTPUT SETTINGS
          </div>

          <select
            aria-label="Output Print Layout"
            value={layout}
            onChange={(e) => setLayout(e.target.value as PrintLayout)}
            className="w-full h-[36px] mb-[10px] px-[10px] bg-white text-black text-[13px] border border-[#333338] rounded-[2px] outline-none cursor-pointer"
          >
            <option value={PrintLayout.Print4x6}>
              Print Sheet 4x6 (6 Photos)
            </option>
            <option value={PrintLayout.Print4x8}>
              Print Sheet 4x8 (8 Photos)
            </option>
            <option value={PrintLayout.Print5x10}>
              Print Sheet 5x10 (8 Photos)
            </option>
            <option value={PrintLayout.Digital2x2}>
              Digital Single (2x2)
            </option>
          </select>

          <select
            aria-label="Background Color"
            value={bgColor}
            onChange={(e) =>
              setBgColor(e.target.value as BackgroundColorOption)
            }
            className="w-full h-[36px] mb-[20px] px-[10px] bg-white text-black text-[13px] border border-[#333338] rounded-[2px] outline-none cursor-pointer"
          >
            <option value={BackgroundColorOption.White}>
              Pure White Background
            </option>
            <option value={BackgroundColorOption.OffWhite}>
              Studio Off-White Background
            </option>
          </select>

          <div className="text-[10px] font-bold tracking-wide text-[#66666B] mb-[10px]">
            MATTE &amp; ALIGNMENT
          </div>

          <label className="flex items-center gap-2 mb-[16px] text-[13px] text-[#EEEEEE] cursor-pointer">
            <input
              type="checkbox"
              checked={softEdge}
              onChange={(e) => setSoftEdge(e.target.checked)}
              className="h-3.5 w-3.5 accent-[#0066FF] cursor-pointer"
            />
            <span>Photographic Soft Edge</span>
          </label>

          {/* Reset & Replace Tools */}
          <button
            type="button"
            onClick={resetManualOffsets}
            disabled={!isEngineReady}
            className="w-full h-[32px] bg-[#222226] hover:bg-[#2C2C32] disabled:opacity-45 disabled:cursor-not-allowed text-[#EEEEEE] border border-[#333338] text-[12px] cursor-pointer transition-colors"
          >
            ⟲ Snap to AI Crop
          </button>

          {isEngineReady && (
            <button
              type="button"
              onClick={handleBrowseClick}
              className="mt-2 w-full h-[30px] bg-transparent hover:bg-[#222226] text-[#888890] hover:text-[#EEEEEE] border border-[#2A2A2F] text-[11px] cursor-pointer transition-colors"
            >
              Load Different Portrait...
            </button>
          )}

          {/* Live U.S. Visa Compliance Metrics (Unboxed Tabular Readout) */}
          {compliance && (
            <div className="mt-[20px] pt-[16px] border-t border-[#252528]">
              <div className="flex items-center justify-between mb-[10px]">
                <span className="text-[10px] font-bold tracking-wide text-[#66666B]">
                  U.S. VISA COMPLIANCE
                </span>
                <span
                  className={`text-[10px] font-bold tracking-wide ${
                    compliance.overallPass
                      ? 'text-[#10B981]'
                      : 'text-[#F59E0B]'
                  }`}
                >
                  {compliance.overallPass ? 'PASS' : 'ADJUST'}
                </span>
              </div>

              <div className="divide-y divide-[#222226] text-[12px]">
                {[
                  compliance.headHeight,
                  compliance.eyeLevel,
                  compliance.centering,
                  compliance.headTilt,
                ].map((item) => (
                  <div
                    key={item.label}
                    className="flex items-center justify-between py-[6px]"
                  >
                    <span className="text-[#9A9AA0]">{item.label}</span>
                    <div className="flex items-center gap-2.5 font-mono tabular-nums text-[11px]">
                      <span className="text-[#E4E4E7]">{item.valueText}</span>
                      <span
                        className={`w-[44px] text-right font-sans font-semibold text-[10px] ${
                          item.pass ? 'text-[#10B981]' : 'text-[#F59E0B]'
                        }`}
                      >
                        {item.pass ? 'PASS' : 'ADJUST'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Export Action */}
        <div className="pt-5">
          <div
            style={{ color: statusColor }}
            className="text-[11px] text-center mb-[10px] min-h-[16px]"
          >
            {statusMessage}
          </div>

          <button
            type="button"
            disabled={!isEngineReady || isProcessing}
            onClick={handleOpenPreview}
            className="mb-[8px] w-full h-[34px] rounded-[3px] bg-[#222226] hover:bg-[#2C2C32] disabled:opacity-45 disabled:cursor-not-allowed text-[#E4E4E7] border border-[#333338] text-[12px] font-medium cursor-pointer transition-colors"
          >
            Preview {currentSpec.shortLabel}
          </button>

          <button
            type="button"
            disabled={!isEngineReady || isProcessing}
            onClick={handleGeneratePrint}
            className="w-full h-[48px] rounded-[4px] bg-[#0066FF] hover:bg-[#0057DB] disabled:bg-[#1E3354] disabled:text-[#7D93B5] disabled:cursor-not-allowed text-white font-bold text-[14px] cursor-pointer transition-colors"
          >
            GENERATE PRINT
          </button>

          {lastOutputUrl && (
            <button
              type="button"
              onClick={() => triggerDownload(lastOutputUrl, lastOutputName)}
              className="mt-[8px] w-full h-[28px] bg-transparent text-[#1E90FF] hover:underline text-[11px] cursor-pointer truncate"
            >
              Download {lastOutputName}
              {lastOutputSize ? ` (${lastOutputSize})` : ''}
            </button>
          )}

          <div className="mt-[16px] text-right text-[10px] italic text-[#444448]">
            Created by Gil
          </div>
        </div>
      </aside>

      {/* Print Sheet Preview Modal */}
      <PrintPreviewModal
        isOpen={isPreviewOpen}
        layout={layout}
        spec={currentSpec}
        previewCanvas={sheetPreviewCanvas}
        isExporting={isProcessing}
        onClose={() => setIsPreviewOpen(false)}
        onDownload={handleGeneratePrint}
      />
    </div>
  );
}
