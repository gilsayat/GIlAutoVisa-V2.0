import React, { useEffect, useRef } from 'react';
import { LayoutSpec, PrintLayout } from '../services/VisaPhotoProcessor';

interface PrintPreviewModalProps {
  isOpen: boolean;
  layout: PrintLayout;
  spec: LayoutSpec;
  previewCanvas: HTMLCanvasElement | null;
  isExporting: boolean;
  onClose: () => void;
  onDownload: () => void;
}

export const PrintPreviewModal: React.FC<PrintPreviewModalProps> = ({
  isOpen,
  layout,
  spec,
  previewCanvas,
  isExporting,
  onClose,
  onDownload,
}) => {
  const displayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!isOpen || !previewCanvas || !displayCanvasRef.current) return;
    const dst = displayCanvasRef.current;
    dst.width = previewCanvas.width;
    dst.height = previewCanvas.height;
    const ctx = dst.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, dst.width, dst.height);
    ctx.drawImage(previewCanvas, 0, 0);
  }, [isOpen, previewCanvas]);

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Print Sheet Preview"
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="flex max-h-[92vh] w-full max-w-[760px] flex-col bg-[#18181A] border border-[#2A2A2F] rounded-[4px] overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-[#252528]">
          <div>
            <h2 className="text-[14px] font-bold text-white leading-snug">
              {spec.label}
            </h2>
            <div className="mt-0.5 text-[11px] text-[#888890] font-mono tabular-nums">
              {spec.widthPx} × {spec.heightPx} px · {spec.widthInches}×
              {spec.heightInches}&quot; · 600 DPI
              {layout !== PrintLayout.Digital2x2 ? ' · Outer Crop Marks' : ''}
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close print preview"
            className="h-8 w-8 flex items-center justify-center text-[#888890] hover:text-white hover:bg-[#252528] rounded-[2px] text-[18px] leading-none cursor-pointer transition-colors"
          >
            ×
          </button>
        </div>

        {/* Sheet Preview Viewport */}
        <div className="flex-1 flex items-center justify-center bg-[#0A0A0C] p-6 overflow-auto min-h-[340px]">
          <canvas
            ref={displayCanvasRef}
            className="max-h-[64vh] max-w-full w-auto h-auto shadow-[0_12px_40px_rgba(0,0,0,0.65)] ring-1 ring-white/10"
          />
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between px-5 py-3.5 border-t border-[#252528] bg-[#18181A]">
          <span className="text-[11px] text-[#77777C]">
            {layout === PrintLayout.Digital2x2
              ? 'Single 2×2" 600 DPI JPEG ready for digital submission.'
              : 'Crop marks are positioned outside each photo edge for spotless trimming.'}
          </span>

          <div className="flex items-center gap-2.5 shrink-0 ml-4">
            <button
              type="button"
              onClick={onClose}
              className="h-[34px] px-4 bg-[#222226] hover:bg-[#2C2C32] text-[#EEEEEE] border border-[#333338] rounded-[3px] text-[12px] cursor-pointer whitespace-nowrap shrink-0 transition-colors"
            >
              Close
            </button>
            <button
              type="button"
              disabled={isExporting}
              onClick={onDownload}
              className="h-[34px] px-4 bg-[#0066FF] hover:bg-[#0057DB] disabled:bg-[#1E3354] disabled:text-[#7D93B5] text-white font-semibold rounded-[3px] text-[12px] cursor-pointer whitespace-nowrap shrink-0 transition-colors"
            >
              {isExporting ? 'Compiling...' : 'Download 600 DPI JPEG'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
