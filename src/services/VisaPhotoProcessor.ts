import {
  FaceLandmarker,
  FilesetResolver,
  ImageSegmenter,
} from '@mediapipe/tasks-vision';

export enum EdgeFeatherMode {
  Sharp = 'Sharp',
  Soft = 'Soft',
}

export enum PrintLayout {
  Digital2x2 = 'Digital2x2',
  Print4x6 = 'Print4x6',
  Print4x8 = 'Print4x8',
  Print5x10 = 'Print5x10',
}

export enum BackgroundColorOption {
  White = 'White',
  OffWhite = 'OffWhite',
}

export interface ProcessingRequest {
  layout: PrintLayout;
  bgColor: BackgroundColorOption;
  edgeMode: EdgeFeatherMode;
  tilt: number;
  zoomOffset: number;
  panX: number;
  panY: number;
}

export interface LayoutSpec {
  label: string;
  shortLabel: string;
  widthInches: number;
  heightInches: number;
  widthPx: number;
  heightPx: number;
  photoCount: number;
  cols: number;
  rows: number;
}

export interface ComplianceCheckItem {
  label: string;
  valueText: string;
  targetText: string;
  pass: boolean;
}

export interface ComplianceReport {
  headHeight: ComplianceCheckItem;
  eyeLevel: ComplianceCheckItem;
  centering: ComplianceCheckItem;
  headTilt: ComplianceCheckItem;
  overallPass: boolean;
  detectionSource: 'ai' | 'fallback';
  autoTiltDeg: number;
}

export function getLayoutSpec(layout: PrintLayout): LayoutSpec {
  const dpi = 600;
  switch (layout) {
    case PrintLayout.Print4x6:
      return {
        label: 'Print Sheet 4x6 (6 Photos)',
        shortLabel: '4×6" Sheet · 6 Photos',
        widthInches: 4,
        heightInches: 6,
        widthPx: 4 * dpi,
        heightPx: 6 * dpi,
        photoCount: 6,
        cols: 2,
        rows: 3,
      };
    case PrintLayout.Print4x8:
      return {
        label: 'Print Sheet 4x8 (8 Photos)',
        shortLabel: '4×8" Sheet · 8 Photos',
        widthInches: 4,
        heightInches: 8,
        widthPx: 4 * dpi,
        heightPx: 8 * dpi,
        photoCount: 8,
        cols: 2,
        rows: 4,
      };
    case PrintLayout.Print5x10:
      return {
        label: 'Print Sheet 5x10 (8 Photos)',
        shortLabel: '5×10" Sheet · 8 Photos',
        widthInches: 5,
        heightInches: 10,
        widthPx: 5 * dpi,
        heightPx: 10 * dpi,
        photoCount: 8,
        cols: 2,
        rows: 4,
      };
    case PrintLayout.Digital2x2:
    default:
      return {
        label: 'Digital Single (2x2)',
        shortLabel: '2×2" Digital Single',
        widthInches: 2,
        heightInches: 2,
        widthPx: 2 * dpi,
        heightPx: 2 * dpi,
        photoCount: 1,
        cols: 1,
        rows: 1,
      };
  }
}

interface DetectedFaceGeometry {
  found: boolean;
  eyeX: number;
  eyeY: number;
  chinY: number;
  foreheadY: number;
  faceWidth: number;
  eyeSlopeDeg: number;
  faceLeft: number;
  faceRight: number;
  faceTop: number;
  faceBottom: number;
  source: 'ai' | 'fallback';
}

const MEDIAPIPE_WASM_PATH = '/mediapipe/wasm';
const FACE_LANDMARKER_MODEL = '/mediapipe/models/face_landmarker.task';
const SELFIE_SEGMENTER_MODEL = '/mediapipe/models/selfie_segmenter.tflite';

// MediaPipe 478-point Face Mesh contour indices
const LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]; // Viewer-left eye
const RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]; // Viewer-right eye

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

function createCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  return canvas;
}

async function loadOrientedCanvas(file: File): Promise<HTMLCanvasElement> {
  if (typeof createImageBitmap === 'function') {
    try {
      const bitmap = await createImageBitmap(file, {
        imageOrientation: 'from-image',
      });
      const canvas = createCanvas(bitmap.width, bitmap.height);
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) throw new Error('Canvas 2D context unavailable.');
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
      return canvas;
    } catch {
      // Fallback to HTMLImageElement below
    }
  }

  return new Promise<HTMLCanvasElement>((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      try {
        const canvas = createCanvas(img.naturalWidth, img.naturalHeight);
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) {
          reject(new Error('Canvas 2D context unavailable.'));
          return;
        }
        ctx.drawImage(img, 0, 0);
        resolve(canvas);
      } finally {
        URL.revokeObjectURL(url);
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Unable to decode image file.'));
    };
    img.src = url;
  });
}

function resizeIfLarger(
  source: HTMLCanvasElement,
  maxDim: number
): HTMLCanvasElement {
  if (source.width <= maxDim && source.height <= maxDim) {
    const copy = createCanvas(source.width, source.height);
    const ctx = copy.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(source, 0, 0);
    return copy;
  }
  const ratio = Math.min(maxDim / source.width, maxDim / source.height);
  const w = Math.max(1, Math.round(source.width * ratio));
  const h = Math.max(1, Math.round(source.height * ratio));
  const resized = createCanvas(w, h);
  const ctx = resized.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, w, h);
  return resized;
}

/**
 * Patches a JPEG buffer's JFIF APP0 marker (or inserts one) so lab printers
 * read the exact 600x600 DPI resolution.
 */
function injectJpegDpi(jpegBytes: Uint8Array, dpi: number): Uint8Array {
  if (jpegBytes.length < 4 || jpegBytes[0] !== 0xff || jpegBytes[1] !== 0xd8) {
    return jpegBytes;
  }

  const hi = (dpi >> 8) & 0xff;
  const lo = dpi & 0xff;

  if (
    jpegBytes.length >= 20 &&
    jpegBytes[2] === 0xff &&
    jpegBytes[3] === 0xe0 &&
    jpegBytes[6] === 0x4a &&
    jpegBytes[7] === 0x46 &&
    jpegBytes[8] === 0x49 &&
    jpegBytes[9] === 0x46 &&
    jpegBytes[10] === 0x00
  ) {
    const patched = new Uint8Array(jpegBytes);
    patched[13] = 1; // units = pixels per inch (DPI)
    patched[14] = hi;
    patched[15] = lo;
    patched[16] = hi;
    patched[17] = lo;
    return patched;
  }

  const app0 = new Uint8Array([
    0xff, 0xe0, 0x00, 0x10,
    0x4a, 0x46, 0x49, 0x46, 0x00,
    0x01, 0x01,
    0x01,
    hi, lo,
    hi, lo,
    0x00, 0x00,
  ]);
  const output = new Uint8Array(jpegBytes.length + app0.length);
  output.set(jpegBytes.subarray(0, 2), 0);
  output.set(app0, 2);
  output.set(jpegBytes.subarray(2), 2 + app0.length);
  return output;
}

async function canvasToJpegBlobWithDpi(
  canvas: HTMLCanvasElement,
  quality: number,
  dpi: number
): Promise<Blob> {
  const rawBlob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (b) => {
        if (b) resolve(b);
        else reject(new Error('Failed to encode JPEG output.'));
      },
      'image/jpeg',
      quality
    );
  });

  const arrayBuffer = await rawBlob.arrayBuffer();
  const patchedBytes = injectJpegDpi(new Uint8Array(arrayBuffer), dpi);
  return new Blob([patchedBytes.buffer as ArrayBuffer], { type: 'image/jpeg' });
}

// MediaPipe/TFLite WASM routes C++ stderr (including benign "INFO: Created TensorFlow Lite XNNPACK delegate")
// to console.error via Emscripten printErr. Filter only those benign TFLite diagnostic lines while preserving real errors.
const BENIGN_TFLITE_LOG_RE =
  /^(INFO|WARNING):|[IW]\d{4}\s|Created TensorFlow Lite|XNNPACK|GL version:|OpenGL error checking|Feedback manager|Successfully destroyed WebGL/i;

let tfliteLogFilterDepth = 0;
let originalConsoleError: typeof console.error | null = null;

function enterTfliteLogFilter(): void {
  if (tfliteLogFilterDepth === 0 && typeof console !== 'undefined') {
    const baseConsoleError = console.error.bind(console);
    originalConsoleError = console.error;
    console.error = (...args: unknown[]) => {
      const first = typeof args[0] === 'string' ? args[0] : '';
      if (BENIGN_TFLITE_LOG_RE.test(first)) {
        return;
      }
      baseConsoleError(...args);
    };
  }
  tfliteLogFilterDepth++;
}

function exitTfliteLogFilter(): void {
  tfliteLogFilterDepth = Math.max(0, tfliteLogFilterDepth - 1);
  if (tfliteLogFilterDepth === 0 && originalConsoleError) {
    console.error = originalConsoleError;
    originalConsoleError = null;
  }
}

async function runWithTfliteLogFilterAsync<T>(fn: () => Promise<T>): Promise<T> {
  enterTfliteLogFilter();
  try {
    return await fn();
  } finally {
    exitTfliteLogFilter();
  }
}

function runWithTfliteLogFilterSync<T>(fn: () => T): T {
  enterTfliteLogFilter();
  try {
    return fn();
  } finally {
    exitTfliteLogFilter();
  }
}

export class VisaPhotoProcessor {
  private _faceLandmarker: FaceLandmarker | null = null;
  private _imageSegmenter: ImageSegmenter | null = null;
  private _aiInitPromise: Promise<boolean> | null = null;
  private _aiAvailable = false;

  private _masterOriginal: HTMLCanvasElement | null = null;
  private _masterOrigData: ImageData | null = null;
  private _masterAlphaData: Uint8ClampedArray | null = null;

  private _masterEyeX = 0;
  private _masterEyeY = 0;
  private _masterHairTopY = 0;
  private _masterChinY = 0;
  private _detectedEyeSlopeDeg = 0;
  private _autoTiltDeg = 0;
  private _detectionSource: 'ai' | 'fallback' = 'fallback';

  private _cachedSubjectKey = '';
  private _cachedSubjectCanvas: HTMLCanvasElement | null = null;

  public async preWarmModels(): Promise<boolean> {
    if (this._aiInitPromise) return this._aiInitPromise;

    this._aiInitPromise = runWithTfliteLogFilterAsync(async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_PATH);

        const [landmarker, segmenter] = await Promise.all([
          FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: FACE_LANDMARKER_MODEL,
              delegate: 'CPU',
            },
            runningMode: 'IMAGE',
            numFaces: 1,
          }),
          ImageSegmenter.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: SELFIE_SEGMENTER_MODEL,
              delegate: 'CPU',
            },
            runningMode: 'IMAGE',
            outputCategoryMask: false,
            outputConfidenceMasks: true,
          }),
        ]);

        this._faceLandmarker = landmarker;
        this._imageSegmenter = segmenter;
        this._aiAvailable = true;
        return true;
      } catch {
        this._aiAvailable = false;
        return false;
      }
    });

    return this._aiInitPromise;
  }

  public isAiReady(): boolean {
    return this._aiAvailable;
  }

  public getAutoTiltDeg(): number {
    return this._autoTiltDeg;
  }

  public getDetectionSource(): 'ai' | 'fallback' {
    return this._detectionSource;
  }

  public async initializeMasterFile(file: File): Promise<void> {
    this._cachedSubjectKey = '';
    this._cachedSubjectCanvas = null;

    // Ensure AI models have finished loading (or gracefully fallen back)
    await this.preWarmModels();

    const origCanvas = await loadOrientedCanvas(file);
    const origCtx = origCanvas.getContext('2d', { willReadFrequently: true });
    if (!origCtx) throw new Error('Unable to read image pixels.');

    this._masterOriginal = origCanvas;
    this._masterOrigData = origCtx.getImageData(
      0,
      0,
      origCanvas.width,
      origCanvas.height
    );

    const proxy = resizeIfLarger(origCanvas, 1024);
    const ratio = origCanvas.width / proxy.width;

    // 1. Face landmark mapping (MediaPipe 478-point -> fallback skin/eye detector)
    let faceGeom: DetectedFaceGeometry | null = null;
    if (this._faceLandmarker) {
      faceGeom = this.detectFaceWithMediaPipe(proxy);
    }
    if (!faceGeom || !faceGeom.found) {
      faceGeom = this.detectFaceFallback(proxy);
    }

    this._detectionSource = faceGeom.source;

    if (faceGeom.found) {
      this._masterEyeX = faceGeom.eyeX * ratio;
      this._masterEyeY = faceGeom.eyeY * ratio;
      this._masterChinY = faceGeom.chinY * ratio;
      this._detectedEyeSlopeDeg = faceGeom.eyeSlopeDeg;
      this._autoTiltDeg = clamp(-faceGeom.eyeSlopeDeg, -15, 15);
    } else {
      this._masterEyeX = origCanvas.width / 2;
      this._masterEyeY = origCanvas.height * 0.38;
      this._masterChinY = origCanvas.height * 0.65;
      this._detectedEyeSlopeDeg = 0;
      this._autoTiltDeg = 0;
    }

    const scaledFace: DetectedFaceGeometry = {
      found: faceGeom.found,
      eyeX: this._masterEyeX,
      eyeY: this._masterEyeY,
      chinY: this._masterChinY,
      foreheadY: faceGeom.foreheadY * ratio,
      faceWidth: faceGeom.faceWidth * ratio,
      eyeSlopeDeg: this._detectedEyeSlopeDeg,
      faceLeft: faceGeom.faceLeft * ratio,
      faceRight: faceGeom.faceRight * ratio,
      faceTop: faceGeom.faceTop * ratio,
      faceBottom: faceGeom.faceBottom * ratio,
      source: faceGeom.source,
    };

    // 2. Portrait foreground segmentation (MediaPipe SelfieSegmenter -> fallback flood-fill)
    let alphaMask: Uint8ClampedArray | null = null;
    if (this._imageSegmenter) {
      alphaMask = this.computeMaskWithMediaPipe(
        proxy,
        origCanvas.width,
        origCanvas.height,
        faceGeom
      );
    }
    if (!alphaMask) {
      alphaMask = this.computeFallbackMask(origCanvas, scaledFace);
      this._detectionSource = 'fallback';
    }
    this._masterAlphaData = alphaMask;

    // 3. Resolution-independent hair top scan
    this._masterHairTopY = scaledFace.found
      ? this.detectHairTop(
          this._masterAlphaData,
          origCanvas.width,
          origCanvas.height,
          scaledFace
        )
      : origCanvas.height * 0.18;
  }

  private getTransformParams(
    req: ProcessingRequest,
    targetSize: number
  ): {
    scale: number;
    targetEyeYOnCanvas: number;
    panXPx: number;
    panYPx: number;
    totalTiltDeg: number;
  } {
    const origH = this._masterOriginal ? this._masterOriginal.height : targetSize;
    let rawHeadHeight = Math.floor(this._masterChinY - this._masterHairTopY);
    if (rawHeadHeight <= 20) {
      rawHeadHeight = Math.floor(origH * 0.5);
    }

    // Center head height at 58% (1.16") — ideal midpoint of U.S. Visa 50%–69% spec
    const baseScale = (targetSize * 0.58) / rawHeadHeight;
    const scale = baseScale * (1.0 + req.zoomOffset / 100.0);
    // Align eye line at 37.5% from top (62.5% from bottom) — exact center of 56%–69% eye band
    const targetEyeYOnCanvas = targetSize * 0.375;

    const panXPx = req.panX * (targetSize * 0.01);
    const panYPx = req.panY * (targetSize * 0.01);
    const totalTiltDeg = this._autoTiltDeg + req.tilt;

    return { scale, targetEyeYOnCanvas, panXPx, panYPx, totalTiltDeg };
  }

  /**
   * Computes live U.S. Department of State 2x2" Visa Photo compliance metrics
   * from the exact transformed landmark geometry on the active canvas.
   */
  public computeCompliance(req: ProcessingRequest): ComplianceReport {
    const targetSize = 600;
    const { scale, targetEyeYOnCanvas, panXPx, panYPx, totalTiltDeg } =
      this.getTransformParams(req, targetSize);

    const rad = (totalTiltDeg * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);

    const transformPoint = (x: number, y: number): { x: number; y: number } => {
      const dx = (x - this._masterEyeX) * scale;
      const dy = (y - this._masterEyeY) * scale;
      const rx = dx * cos - dy * sin;
      const ry = dx * sin + dy * cos;
      return {
        x: targetSize / 2 + panXPx + rx,
        y: targetEyeYOnCanvas + panYPx + ry,
      };
    };

    const eyePt = transformPoint(this._masterEyeX, this._masterEyeY);
    const hairPt = transformPoint(this._masterEyeX, this._masterHairTopY);
    const chinPt = transformPoint(this._masterEyeX, this._masterChinY);

    const headHeightFrac = Math.max(0, (chinPt.y - hairPt.y) / targetSize);
    const headHeightPct = headHeightFrac * 100;
    const headHeightIn = headHeightFrac * 2.0;
    const headPass = headHeightPct >= 50.0 && headHeightPct <= 69.0;

    const eyeFromBottomFrac = 1.0 - eyePt.y / targetSize;
    const eyeFromBottomPct = eyeFromBottomFrac * 100;
    const eyeFromBottomIn = eyeFromBottomFrac * 2.0;
    const eyePass = eyeFromBottomPct >= 56.0 && eyeFromBottomPct <= 69.0;

    const centerOffsetPct = (eyePt.x / targetSize - 0.5) * 100;
    const centerPass = Math.abs(centerOffsetPct) <= 3.5;

    const effectiveTiltDeg = this._detectedEyeSlopeDeg + totalTiltDeg;
    const tiltPass = Math.abs(effectiveTiltDeg) <= 2.0;

    const signCenter =
      Math.abs(centerOffsetPct) < 0.05
        ? ''
        : centerOffsetPct > 0
          ? '+'
          : '';
    const signTilt =
      Math.abs(effectiveTiltDeg) < 0.05
        ? ''
        : effectiveTiltDeg > 0
          ? '+'
          : '';

    return {
      headHeight: {
        label: 'Head Height',
        valueText: `${headHeightPct.toFixed(1)}% (${headHeightIn.toFixed(2)}")`,
        targetText: '50–69%',
        pass: headPass,
      },
      eyeLevel: {
        label: 'Eye Level',
        valueText: `${eyeFromBottomPct.toFixed(1)}% (${eyeFromBottomIn.toFixed(2)}")`,
        targetText: '56–69%',
        pass: eyePass,
      },
      centering: {
        label: 'Centering',
        valueText: `${signCenter}${centerOffsetPct.toFixed(1)}%`,
        targetText: '±3.5%',
        pass: centerPass,
      },
      headTilt: {
        label: 'Head Level',
        valueText: `${signTilt}${effectiveTiltDeg.toFixed(1)}°`,
        targetText: '±2.0°',
        pass: tiltPass,
      },
      overallPass: headPass && eyePass && centerPass && tiltPass,
      detectionSource: this._detectionSource,
      autoTiltDeg: this._autoTiltDeg,
    };
  }

  public renderFrame(
    req: ProcessingRequest,
    previewMode: boolean
  ): HTMLCanvasElement {
    if (
      !this._masterOriginal ||
      !this._masterOrigData ||
      !this._masterAlphaData
    ) {
      throw new Error('Engine not initialized.');
    }

    const targetSize = previewMode ? 600 : 1200;
    const bgTarget =
      req.bgColor === BackgroundColorOption.OffWhite ? 242 : 255;

    const subjectLayer = this.getOrBuildSubjectLayer(req.edgeMode);

    const canvas = createCanvas(targetSize, targetSize);
    const ctx = canvas.getContext('2d')!;

    if (!previewMode) {
      ctx.fillStyle = `rgb(${bgTarget}, ${bgTarget}, ${bgTarget})`;
      ctx.fillRect(0, 0, targetSize, targetSize);
    } else {
      ctx.clearRect(0, 0, targetSize, targetSize);
    }

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    const { scale, targetEyeYOnCanvas, panXPx, panYPx, totalTiltDeg } =
      this.getTransformParams(req, targetSize);

    ctx.save();
    ctx.translate(targetSize / 2 + panXPx, targetEyeYOnCanvas + panYPx);
    ctx.rotate((totalTiltDeg * Math.PI) / 180);
    ctx.scale(scale, scale);
    ctx.translate(-this._masterEyeX, -this._masterEyeY);
    ctx.drawImage(subjectLayer, 0, 0);
    ctx.restore();

    return canvas;
  }

  /**
   * Renders a scaled preview of the chosen print sheet (or 2x2 digital single)
   * complete with lab-grade outer corner crop marks.
   */
  public renderSheetPreview(
    req: ProcessingRequest,
    maxPreviewHeight = 1100
  ): HTMLCanvasElement {
    const spec = getLayoutSpec(req.layout);
    const previewScale = Math.min(1, maxPreviewHeight / spec.heightPx);
    const dpi = Math.round(600 * previewScale);
    const tilePx = Math.round(2 * dpi);

    // Render base 2x2 at exact preview tile size for sharp display
    const fullBase = this.renderFrame(req, false);
    const scaledBase = createCanvas(tilePx, tilePx);
    const bCtx = scaledBase.getContext('2d')!;
    bCtx.imageSmoothingEnabled = true;
    bCtx.imageSmoothingQuality = 'high';
    bCtx.drawImage(fullBase, 0, 0, tilePx, tilePx);

    if (req.layout === PrintLayout.Digital2x2) {
      return scaledBase;
    }
    return this.generateSheet(scaledBase, req.layout, dpi);
  }

  public async generateHighResPrint(req: ProcessingRequest): Promise<Blob> {
    if (!this._masterOriginal) {
      throw new Error('Engine not initialized.');
    }

    const base2x2 = this.renderFrame(req, false);
    const dpi = 600;
    const finalSheet =
      req.layout === PrintLayout.Digital2x2
        ? base2x2
        : this.generateSheet(base2x2, req.layout, dpi);

    const quality = req.layout === PrintLayout.Digital2x2 ? 0.92 : 0.98;
    return canvasToJpegBlobWithDpi(finalSheet, quality, dpi);
  }

  /**
   * Builds the foreground RGBA subject layer preserving 100% original skin tones
   * (zero auto-gamma distortion) with Hermite edge feathering and local foreground
   * color decontamination (de-fringing) along semi-transparent edges.
   */
  private getOrBuildSubjectLayer(edgeMode: EdgeFeatherMode): HTMLCanvasElement {
    const key = `${edgeMode}`;
    if (this._cachedSubjectCanvas && this._cachedSubjectKey === key) {
      return this._cachedSubjectCanvas;
    }

    const orig = this._masterOrigData!;
    const alpha = this._masterAlphaData!;
    const w = orig.width;
    const h = orig.height;

    const low = edgeMode === EdgeFeatherMode.Soft ? 0.14 : 0.38;
    const high = edgeMode === EdgeFeatherMode.Soft ? 0.86 : 0.62;
    const range = high - low;

    const subjectCanvas = createCanvas(w, h);
    const ctx = subjectCanvas.getContext('2d')!;
    const out = ctx.createImageData(w, h);
    const src = orig.data;
    const dst = out.data;

    // Step 1: Compute smoothstepped alpha and copy untouched original RGB
    const smoothedAlpha = new Float32Array(w * h);
    for (let p = 0, idx = 0; p < w * h; p++, idx += 4) {
      const val = alpha[p] / 255.0;
      let a = 0;
      if (val >= high) {
        a = 1;
      } else if (val > low) {
        const t = (val - low) / range;
        a = t * t * (3 - 2 * t);
      }
      smoothedAlpha[p] = a;

      dst[idx] = src[idx];
      dst[idx + 1] = src[idx + 1];
      dst[idx + 2] = src[idx + 2];
      dst[idx + 3] = Math.round(a * 255);
    }

    // Step 2: Edge color de-fringing on the semi-transparent transition band (0.01 < a < 0.96)
    // Samples nearby solid foreground pixels (alpha >= 0.92) to replace contaminated backdrop halo RGB.
    const stepBase = clamp(Math.round(Math.min(w, h) / 260), 2, 6);
    const offsets: Array<[number, number]> = [
      [-stepBase, 0],
      [stepBase, 0],
      [0, -stepBase],
      [0, stepBase],
      [-stepBase, -stepBase],
      [stepBase, -stepBase],
      [-stepBase, stepBase],
      [stepBase, stepBase],
      [-stepBase * 2, 0],
      [stepBase * 2, 0],
      [0, -stepBase * 2],
      [0, stepBase * 2],
      [-stepBase * 2, -stepBase * 2],
      [stepBase * 2, -stepBase * 2],
      [-stepBase * 2, stepBase * 2],
      [stepBase * 2, stepBase * 2],
    ];

    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const p = row + x;
        const a = smoothedAlpha[p];
        if (a <= 0.01 || a >= 0.96) continue;

        let sumR = 0;
        let sumG = 0;
        let sumB = 0;
        let sumW = 0;

        for (let k = 0; k < offsets.length; k++) {
          const nx = x + offsets[k][0];
          const ny = y + offsets[k][1];
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const np = ny * w + nx;
          const na = smoothedAlpha[np];
          if (na >= 0.92) {
            const distWeight = k < 8 ? 2.0 : 1.0;
            const nIdx = np * 4;
            sumR += src[nIdx] * distWeight;
            sumG += src[nIdx + 1] * distWeight;
            sumB += src[nIdx + 2] * distWeight;
            sumW += distWeight;
          }
        }

        if (sumW > 0) {
          const fgR = sumR / sumW;
          const fgG = sumG / sumW;
          const fgB = sumB / sumW;
          // Stronger decontamination the more transparent the halo pixel is
          const blend = clamp((1 - a) * 0.92, 0, 0.92);
          const idx = p * 4;
          dst[idx] = Math.round(src[idx] * (1 - blend) + fgR * blend);
          dst[idx + 1] = Math.round(src[idx + 1] * (1 - blend) + fgG * blend);
          dst[idx + 2] = Math.round(src[idx + 2] * (1 - blend) + fgB * blend);
        }
      }
    }

    ctx.putImageData(out, 0, 0);
    this._cachedSubjectKey = key;
    this._cachedSubjectCanvas = subjectCanvas;
    return subjectCanvas;
  }

  private detectFaceWithMediaPipe(
    proxy: HTMLCanvasElement
  ): DetectedFaceGeometry | null {
    if (!this._faceLandmarker) return null;
    try {
      const result = runWithTfliteLogFilterSync(() =>
        this._faceLandmarker!.detect(proxy)
      );
      if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
        return null;
      }

      const lm = result.faceLandmarks[0];
      const w = proxy.width;
      const h = proxy.height;

      // Average eye contour landmarks (or iris landmarks 468/473 if present)
      let leftEyeX = 0;
      let leftEyeY = 0;
      if (lm.length > 473) {
        leftEyeX = lm[468].x * w;
        leftEyeY = lm[468].y * h;
      } else {
        for (const i of LEFT_EYE_INDICES) {
          leftEyeX += lm[i].x * w;
          leftEyeY += lm[i].y * h;
        }
        leftEyeX /= LEFT_EYE_INDICES.length;
        leftEyeY /= LEFT_EYE_INDICES.length;
      }

      let rightEyeX = 0;
      let rightEyeY = 0;
      if (lm.length > 473) {
        rightEyeX = lm[473].x * w;
        rightEyeY = lm[473].y * h;
      } else {
        for (const i of RIGHT_EYE_INDICES) {
          rightEyeX += lm[i].x * w;
          rightEyeY += lm[i].y * h;
        }
        rightEyeX /= RIGHT_EYE_INDICES.length;
        rightEyeY /= RIGHT_EYE_INDICES.length;
      }

      // Ensure leftEye is truly viewer-left (smaller X)
      if (leftEyeX > rightEyeX) {
        [leftEyeX, rightEyeX] = [rightEyeX, leftEyeX];
        [leftEyeY, rightEyeY] = [rightEyeY, leftEyeY];
      }

      const eyeX = (leftEyeX + rightEyeX) / 2;
      const eyeY = (leftEyeY + rightEyeY) / 2;
      const eyeSlopeDeg =
        (Math.atan2(rightEyeY - leftEyeY, rightEyeX - leftEyeX) * 180) / Math.PI;

      const chinY = lm[152].y * h;
      const foreheadY = lm[10].y * h;
      const faceLeft = Math.min(lm[234].x, lm[454].x) * w;
      const faceRight = Math.max(lm[234].x, lm[454].x) * w;
      const faceWidth = Math.max(20, faceRight - faceLeft);

      return {
        found: true,
        eyeX,
        eyeY,
        chinY,
        foreheadY,
        faceWidth,
        eyeSlopeDeg,
        faceLeft,
        faceRight,
        faceTop: foreheadY,
        faceBottom: chinY,
        source: 'ai',
      };
    } catch {
      return null;
    }
  }

  private computeMaskWithMediaPipe(
    proxy: HTMLCanvasElement,
    targetWidth: number,
    targetHeight: number,
    proxyFace: DetectedFaceGeometry
  ): Uint8ClampedArray | null {
    if (!this._imageSegmenter) return null;

    let result: ReturnType<ImageSegmenter['segment']> | null = null;
    try {
      result = runWithTfliteLogFilterSync(() =>
        this._imageSegmenter!.segment(proxy)
      );
      const masks = result.confidenceMasks;
      if (!masks || masks.length === 0) return null;

      // MediaPipe Selfie Segmenter: index 1 is foreground when 2 masks exist, else index 0
      const maskObj = masks.length >= 2 ? masks[1] : masks[0];
      const rawFloats = maskObj.getAsFloat32Array();
      const mw = maskObj.width;
      const mh = maskObj.height;

      // Sanity check orientation: ensure detected face center has higher confidence than top corners
      const fx = clamp(Math.round((proxyFace.eyeX / proxy.width) * mw), 0, mw - 1);
      const fy = clamp(Math.round((proxyFace.eyeY / proxy.height) * mh), 0, mh - 1);
      const faceConf = rawFloats[fy * mw + fx];
      const cornerConf =
        (rawFloats[2 * mw + 2] + rawFloats[2 * mw + (mw - 3)]) * 0.5;
      const invert = proxyFace.found && faceConf < 0.35 && cornerConf > 0.65;

      // Write float confidence mask to a small canvas with subtle 1px anti-alias smoothing
      const maskSmall = createCanvas(mw, mh);
      const mCtx = maskSmall.getContext('2d')!;
      const mImg = mCtx.createImageData(mw, mh);
      for (let i = 0; i < mw * mh; i++) {
        const v = invert ? 1 - rawFloats[i] : rawFloats[i];
        const m = clamp(Math.round(v * 255), 0, 255);
        const p = i * 4;
        mImg.data[p] = m;
        mImg.data[p + 1] = m;
        mImg.data[p + 2] = m;
        mImg.data[p + 3] = 255;
      }
      mCtx.putImageData(mImg, 0, 0);

      // Upscale cleanly to full master resolution using high-quality bicubic interpolation
      const fullMask = createCanvas(targetWidth, targetHeight);
      const fCtx = fullMask.getContext('2d', { willReadFrequently: true })!;
      fCtx.imageSmoothingEnabled = true;
      fCtx.imageSmoothingQuality = 'high';
      fCtx.drawImage(maskSmall, 0, 0, targetWidth, targetHeight);

      const fullData = fCtx.getImageData(0, 0, targetWidth, targetHeight).data;
      const out = new Uint8ClampedArray(targetWidth * targetHeight);
      for (let i = 0, p = 0; i < out.length; i++, p += 4) {
        out[i] = fullData[p];
      }
      return out;
    } catch {
      return null;
    } finally {
      if (result) {
        try {
          result.close();
        } catch {
          // Ignore cleanup errors
        }
      }
    }
  }

  private detectFaceFallback(proxy: HTMLCanvasElement): DetectedFaceGeometry {
    const w = proxy.width;
    const h = proxy.height;
    const ctx = proxy.getContext('2d', { willReadFrequently: true })!;
    const img = ctx.getImageData(0, 0, w, h);
    const data = img.data;

    const scale = Math.min(1, 160 / Math.max(w, h));
    const gw = Math.max(24, Math.round(w * scale));
    const gh = Math.max(24, Math.round(h * scale));
    const skin = new Uint8Array(gw * gh);
    const lum = new Float32Array(gw * gh);

    for (let gy = 0; gy < gh; gy++) {
      const sy = Math.min(h - 1, Math.floor(gy / scale));
      for (let gx = 0; gx < gw; gx++) {
        const sx = Math.min(w - 1, Math.floor(gx / scale));
        const idx = (sy * w + sx) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];

        const yVal = 0.299 * r + 0.587 * g + 0.114 * b;
        const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
        const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;

        lum[gy * gw + gx] = yVal;
        skin[gy * gw + gx] =
          yVal > 45 && cb >= 77 && cb <= 127 && cr >= 133 && cr <= 175 && r > b
            ? 1
            : 0;
      }
    }

    const visited = new Uint8Array(gw * gh);
    const queue = new Int32Array(gw * gh);
    let bestScore = 0;
    let bestBox = { minX: 0, maxX: 0, minY: 0, maxY: 0 };

    const maxSearchY = Math.floor(gh * 0.85);
    for (let y = Math.floor(gh * 0.05); y < maxSearchY; y++) {
      for (let x = Math.floor(gw * 0.1); x < Math.floor(gw * 0.9); x++) {
        const start = y * gw + x;
        if (!skin[start] || visited[start]) continue;

        let head = 0;
        let tail = 0;
        queue[tail++] = start;
        visited[start] = 1;

        let minX = x;
        let maxX = x;
        let minY = y;
        let maxY = y;
        let area = 0;

        while (head < tail) {
          const cur = queue[head++];
          const cx = cur % gw;
          const cy = (cur - cx) / gw;
          area++;
          if (cx < minX) minX = cx;
          if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy;
          if (cy > maxY) maxY = cy;

          const neighbors = [cur - 1, cur + 1, cur - gw, cur + gw];
          for (let n = 0; n < 4; n++) {
            const ni = neighbors[n];
            if (ni < 0 || ni >= gw * gh || visited[ni] || !skin[ni]) continue;
            const nx = ni % gw;
            if (Math.abs(nx - cx) > 1) continue;
            visited[ni] = 1;
            queue[tail++] = ni;
          }
        }

        const bw = maxX - minX + 1;
        const bh = maxY - minY + 1;
        const aspect = bh / Math.max(1, bw);
        const fill = area / (bw * bh);
        const cxNorm = (minX + maxX) / (2 * gw);
        const cyNorm = (minY + maxY) / (2 * gh);

        if (
          area < gw * gh * 0.015 ||
          area > gw * gh * 0.65 ||
          aspect < 0.7 ||
          aspect > 2.4 ||
          fill < 0.35
        ) {
          continue;
        }

        const centerWeight =
          Math.exp(-Math.pow((cxNorm - 0.5) / 0.3, 2)) *
          Math.exp(-Math.pow((cyNorm - 0.42) / 0.35, 2));
        const score = area * fill * centerWeight;
        if (score > bestScore) {
          bestScore = score;
          bestBox = { minX, maxX, minY, maxY };
        }
      }
    }

    if (bestScore <= 0) {
      return {
        found: false,
        eyeX: w / 2,
        eyeY: h * 0.38,
        chinY: h * 0.65,
        foreheadY: h * 0.22,
        faceWidth: w * 0.35,
        eyeSlopeDeg: 0,
        faceLeft: w * 0.32,
        faceRight: w * 0.68,
        faceTop: h * 0.22,
        faceBottom: h * 0.65,
        source: 'fallback',
      };
    }

    const bw = bestBox.maxX - bestBox.minX + 1;
    let maxY = bestBox.maxY;
    if (maxY - bestBox.minY + 1 > bw * 1.48) {
      maxY = Math.min(gh - 1, Math.round(bestBox.minY + bw * 1.38));
    }
    const bh = maxY - bestBox.minY + 1;

    const eyeStart = Math.round(bestBox.minY + bh * 0.28);
    const eyeEnd = Math.round(bestBox.minY + bh * 0.5);
    const leftEyeX1 = Math.round(bestBox.minX + bw * 0.18);
    const leftEyeX2 = Math.round(bestBox.minX + bw * 0.42);
    const rightEyeX1 = Math.round(bestBox.minX + bw * 0.58);
    const rightEyeX2 = Math.round(bestBox.minX + bw * 0.82);

    let bestEyeY = bestBox.minY + bh * 0.38;
    let minEyeLum = Infinity;
    for (let ey = eyeStart; ey <= eyeEnd; ey++) {
      let sum = 0;
      let cnt = 0;
      for (let ex = leftEyeX1; ex <= leftEyeX2; ex++) {
        sum += lum[ey * gw + ex];
        cnt++;
      }
      for (let ex = rightEyeX1; ex <= rightEyeX2; ex++) {
        sum += lum[ey * gw + ex];
        cnt++;
      }
      if (cnt > 0 && sum / cnt < minEyeLum) {
        minEyeLum = sum / cnt;
        bestEyeY = ey;
      }
    }

    const inv = 1 / scale;
    return {
      found: true,
      eyeX: ((bestBox.minX + bestBox.maxX) / 2) * inv,
      eyeY: bestEyeY * inv,
      chinY: maxY * inv,
      foreheadY: bestBox.minY * inv,
      faceWidth: bw * inv,
      eyeSlopeDeg: 0,
      faceLeft: bestBox.minX * inv,
      faceRight: bestBox.maxX * inv,
      faceTop: bestBox.minY * inv,
      faceBottom: maxY * inv,
      source: 'fallback',
    };
  }

  private computeFallbackMask(
    source: HTMLCanvasElement,
    face: DetectedFaceGeometry
  ): Uint8ClampedArray {
    const S = 320;
    const small = createCanvas(S, S);
    const sCtx = small.getContext('2d', { willReadFrequently: true })!;
    sCtx.imageSmoothingEnabled = true;
    sCtx.imageSmoothingQuality = 'high';
    sCtx.drawImage(source, 0, 0, S, S);

    const img = sCtx.getImageData(0, 0, S, S);
    const d = img.data;
    const scaleX = S / source.width;
    const scaleY = S / source.height;

    const seeds: Array<[number, number, number]> = [];
    const addPatch = (x0: number, y0: number, pw: number, ph: number) => {
      let r = 0;
      let g = 0;
      let b = 0;
      let n = 0;
      for (let y = y0; y < Math.min(S, y0 + ph); y++) {
        for (let x = x0; x < Math.min(S, x0 + pw); x++) {
          const i = (y * S + x) * 4;
          r += d[i];
          g += d[i + 1];
          b += d[i + 2];
          n++;
        }
      }
      if (n > 0) seeds.push([r / n, g / n, b / n]);
    };

    for (let x = 0; x < S; x += 40) addPatch(x, 0, 16, 12);
    for (let y = 16; y < Math.floor(S * 0.68); y += 36) {
      addPatch(0, y, 10, 16);
      addPatch(S - 10, y, 10, 16);
    }

    const minSeedDist = (r: number, g: number, b: number): number => {
      let best = Infinity;
      for (let k = 0; k < seeds.length; k++) {
        const dr = r - seeds[k][0];
        const dg = g - seeds[k][1];
        const db = b - seeds[k][2];
        const dist = Math.sqrt(dr * dr + dg * dg + db * db);
        if (dist < best) best = dist;
      }
      return best;
    };

    let sumDist = 0;
    let borderCnt = 0;
    for (let x = 0; x < S; x += 2) {
      const i = x * 4;
      sumDist += minSeedDist(d[i], d[i + 1], d[i + 2]);
      borderCnt++;
    }
    const avgBorderNoise = borderCnt > 0 ? sumDist / borderCnt : 12;
    const colorThreshold = clamp(28 + avgBorderNoise * 1.45, 32, 68);
    const localStepThreshold = clamp(16 + avgBorderNoise * 0.65, 16, 36);

    const fcx = face.eyeX * scaleX;
    const fcy = ((face.faceTop + face.faceBottom) / 2) * scaleY;
    const frx = Math.max(
      24,
      ((face.faceRight - face.faceLeft) / 2) * scaleX * 0.92
    );
    const fry = Math.max(
      32,
      ((face.faceBottom - face.faceTop) / 2) * scaleY * 1.05
    );
    const chinY = face.chinY * scaleY;

    const isProtected = (x: number, y: number): boolean => {
      const dx = (x - fcx) / frx;
      const dy = (y - fcy) / fry;
      if (dx * dx + dy * dy <= 0.85) return true;
      if (y > chinY + 12 && Math.abs(x - fcx) < frx * 1.05) return true;
      return false;
    };

    const bgFlag = new Uint8Array(S * S);
    const queue = new Int32Array(S * S);
    let head = 0;
    let tail = 0;

    const enqueueIfBg = (x: number, y: number) => {
      const idx = y * S + x;
      if (bgFlag[idx] || isProtected(x, y)) return;
      const p = idx * 4;
      if (minSeedDist(d[p], d[p + 1], d[p + 2]) <= colorThreshold * 1.2) {
        bgFlag[idx] = 1;
        queue[tail++] = idx;
      }
    };

    for (let x = 0; x < S; x++) enqueueIfBg(x, 0);
    for (let y = 0; y < Math.floor(S * 0.78); y++) {
      enqueueIfBg(0, y);
      enqueueIfBg(S - 1, y);
    }

    const dirs = [-1, 1, -S, S];
    while (head < tail) {
      const cur = queue[head++];
      const cx = cur % S;
      const cp = cur * 4;
      const cr = d[cp];
      const cg = d[cp + 1];
      const cb = d[cp + 2];

      for (let k = 0; k < 4; k++) {
        const ni = cur + dirs[k];
        if (ni < 0 || ni >= S * S || bgFlag[ni]) continue;
        const nx = ni % S;
        if (Math.abs(nx - cx) > 1) continue;
        const ny = (ni - nx) / S;
        if (isProtected(nx, ny)) continue;

        const np = ni * 4;
        const nr = d[np];
        const ng = d[np + 1];
        const nb = d[np + 2];

        const seedDist = minSeedDist(nr, ng, nb);
        const stepDist = Math.sqrt(
          (nr - cr) * (nr - cr) +
            (ng - cg) * (ng - cg) +
            (nb - cb) * (nb - cb)
        );

        if (
          seedDist <= colorThreshold ||
          (stepDist <= localStepThreshold && seedDist <= colorThreshold * 1.55)
        ) {
          bgFlag[ni] = 1;
          queue[tail++] = ni;
        }
      }
    }

    const alphaF = new Float32Array(S * S);
    for (let i = 0; i < S * S; i++) {
      alphaF[i] = bgFlag[i] ? 0 : 1;
    }

    const temp = new Float32Array(S * S);
    const radius = 2;
    for (let pass = 0; pass < 2; pass++) {
      for (let y = 0; y < S; y++) {
        const row = y * S;
        for (let x = 0; x < S; x++) {
          let sum = 0;
          let cnt = 0;
          for (let dx = -radius; dx <= radius; dx++) {
            const nx = x + dx;
            if (nx >= 0 && nx < S) {
              sum += alphaF[row + nx];
              cnt++;
            }
          }
          temp[row + x] = sum / cnt;
        }
      }
      for (let x = 0; x < S; x++) {
        for (let y = 0; y < S; y++) {
          let sum = 0;
          let cnt = 0;
          for (let dy = -radius; dy <= radius; dy++) {
            const ny = y + dy;
            if (ny >= 0 && ny < S) {
              sum += temp[ny * S + x];
              cnt++;
            }
          }
          alphaF[y * S + x] = sum / cnt;
        }
      }
    }

    const maskSmall = createCanvas(S, S);
    const mCtx = maskSmall.getContext('2d')!;
    const mImg = mCtx.createImageData(S, S);
    for (let i = 0; i < S * S; i++) {
      const m = clamp(Math.round(alphaF[i] * 255), 0, 255);
      const p = i * 4;
      mImg.data[p] = m;
      mImg.data[p + 1] = m;
      mImg.data[p + 2] = m;
      mImg.data[p + 3] = 255;
    }
    mCtx.putImageData(mImg, 0, 0);

    const fullMask = createCanvas(source.width, source.height);
    const fCtx = fullMask.getContext('2d', { willReadFrequently: true })!;
    fCtx.imageSmoothingEnabled = true;
    fCtx.imageSmoothingQuality = 'high';
    fCtx.drawImage(maskSmall, 0, 0, source.width, source.height);

    const fullData = fCtx.getImageData(0, 0, source.width, source.height).data;
    const result = new Uint8ClampedArray(source.width * source.height);
    for (let i = 0, p = 0; i < result.length; i++, p += 4) {
      result[i] = fullData[p];
    }
    return result;
  }

  /**
   * Resolution-independent hair top scanner: scales scan width proportionally to
   * detected face width and ignores isolated single-pixel noise.
   */
  private detectHairTop(
    mask: Uint8ClampedArray,
    width: number,
    height: number,
    face: DetectedFaceGeometry
  ): number {
    const halfScan = Math.max(12, Math.round(face.faceWidth * 0.42));
    const scanL = clamp(Math.floor(face.eyeX - halfScan), 0, width - 1);
    const scanR = clamp(Math.floor(face.eyeX + halfScan), scanL, width - 1);
    const spanWidth = Math.max(1, scanR - scanL + 1);
    const minFgCount = Math.max(2, Math.round(spanWidth * 0.05));

    const startY = clamp(Math.floor(face.eyeY), 0, height - 1);
    const eyeToChin = Math.max(20, face.chinY - face.eyeY);
    const yLim = Math.max(0, Math.floor(face.eyeY - eyeToChin * 1.45));

    let top = Math.min(startY, Math.floor(face.foreheadY));
    let consecutiveEmpty = 0;
    const maxGap = Math.max(4, Math.round(eyeToChin * 0.06));

    for (let y = startY; y >= yLim; y--) {
      let fgCount = 0;
      const row = y * width;
      for (let x = scanL; x <= scanR; x++) {
        if (mask[row + x] > 115) fgCount++;
      }
      if (fgCount >= minFgCount) {
        top = y;
        consecutiveEmpty = 0;
      } else if (y < face.foreheadY) {
        consecutiveEmpty++;
        if (consecutiveEmpty > maxGap) break;
      }
    }

    return top;
  }

  /**
   * Generates a multi-photo print sheet with clean lab-grade outer corner crop
   * marks placed inside the 0.05" bleed gutter (zero ink drawn on photo edges).
   */
  private generateSheet(
    base2x2: HTMLCanvasElement,
    layout: PrintLayout,
    dpi: number
  ): HTMLCanvasElement {
    const spec = getLayoutSpec(layout);
    const t = base2x2.width;
    const gutter = Math.max(8, Math.round(dpi * 0.05)); // 0.05 inch bleed safe gutter

    const w = Math.round(spec.widthInches * dpi);
    const h = Math.round(spec.heightInches * dpi);
    const cols = spec.cols;
    const rows = spec.rows;

    const sX = Math.floor((w - (cols * t + (cols - 1) * gutter)) / 2);
    const sY = Math.floor((h - (rows * t + (rows - 1) * gutter)) / 2);

    const sheet = createCanvas(w, h);
    const ctx = sheet.getContext('2d')!;
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, w, h);

    // Draw all 2x2 photo tiles first
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const pX = sX + x * (t + gutter);
        const pY = sY + y * (t + gutter);
        ctx.drawImage(base2x2, pX, pY);
      }
    }

    // Draw clean outer corner crop marks strictly outside photo boundaries
    const gap = Math.max(2, Math.round(dpi * 0.008)); // ~5px clearance from photo edge at 600 DPI
    const tickLen = Math.max(5, Math.min(Math.floor(gutter * 0.48), Math.round(dpi * 0.032)));
    const strokeW = Math.max(1, dpi / 380); // ~1.5px hairline at 600 DPI

    ctx.strokeStyle = '#8A8A8F';
    ctx.lineWidth = strokeW;
    ctx.lineCap = 'square';
    ctx.beginPath();

    const drawCornerMarks = (x: number, y: number, size: number) => {
      const right = x + size;
      const bottom = y + size;

      // Top-left corner
      ctx.moveTo(x - gap - tickLen, y);
      ctx.lineTo(x - gap, y);
      ctx.moveTo(x, y - gap - tickLen);
      ctx.lineTo(x, y - gap);

      // Top-right corner
      ctx.moveTo(right + gap, y);
      ctx.lineTo(right + gap + tickLen, y);
      ctx.moveTo(right, y - gap - tickLen);
      ctx.lineTo(right, y - gap);

      // Bottom-left corner
      ctx.moveTo(x - gap - tickLen, bottom);
      ctx.lineTo(x - gap, bottom);
      ctx.moveTo(x, bottom + gap);
      ctx.lineTo(x, bottom + gap + tickLen);

      // Bottom-right corner
      ctx.moveTo(right + gap, bottom);
      ctx.lineTo(right + gap + tickLen, bottom);
      ctx.moveTo(right, bottom + gap);
      ctx.lineTo(right, bottom + gap + tickLen);
    };

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const pX = sX + x * (t + gutter);
        const pY = sY + y * (t + gutter);
        drawCornerMarks(pX, pY, t);
      }
    }

    ctx.stroke();
    return sheet;
  }
}
