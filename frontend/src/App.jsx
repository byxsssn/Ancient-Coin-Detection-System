import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Download,
  Image as ImageIcon,
  Loader2,
  ScanSearch,
  SlidersHorizontal,
  Upload,
} from "lucide-react";

const DEFAULT_CONFIDENCE = 0.5;

function formatConfidence(value) {
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value) {
  return Number(value).toFixed(1);
}

function App() {
  const fileInputRef = useRef(null);
  const [health, setHealth] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [confidence, setConfidence] = useState(DEFAULT_CONFIDENCE);
  const [result, setResult] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;
    fetch("/api/health")
      .then((response) => response.json())
      .then((data) => {
        if (isMounted) {
          setHealth(data);
        }
      })
      .catch(() => {
        if (isMounted) {
          setHealth({ status: "offline" });
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const selectedDetection = result?.detections?.[selectedIndex] ?? null;
  const displayImage = result?.annotatedImage || previewUrl;

  const classSummary = useMemo(() => {
    if (!result?.summary?.byClass) {
      return [];
    }
    return Object.entries(result.summary.byClass);
  }, [result]);

  function chooseFile(nextFile) {
    if (!nextFile) {
      return;
    }
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setResult(null);
    setSelectedIndex(-1);
    setError("");
  }

  async function runDetection() {
    if (!file) {
      setError("请先选择图片。");
      return;
    }

    setIsDetecting(true);
    setError("");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("confidence", confidence.toString());

    try {
      const response = await fetch("/api/detect", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || "检测失败。");
      }
      setResult(payload);
      setSelectedIndex(payload.detections.length > 0 ? 0 : -1);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "检测失败。");
      setResult(null);
      setSelectedIndex(-1);
    } finally {
      setIsDetecting(false);
    }
  }

  function downloadResult() {
    if (!result?.annotatedImage) {
      return;
    }
    const link = document.createElement("a");
    link.href = result.annotatedImage;
    link.download = "coin_detection_result.jpg";
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>古钱币智能检测</h1>
          <p>YOLOv8s-768 / Web 工作台</p>
        </div>
        <div className={`service-chip ${health?.model?.available ? "ready" : "warn"}`}>
          {health?.model?.available ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
          <span>{health?.model?.available ? "模型就绪" : "模型待检查"}</span>
        </div>
      </header>

      <div className="workspace">
        <aside className="control-panel">
          <section className="tool-section">
            <div className="section-title">
              <ImageIcon size={18} />
              <span>图片</span>
            </div>
            <input
              ref={fileInputRef}
              className="file-input"
              type="file"
              accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp"
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />
            <button className="secondary-button" type="button" onClick={() => fileInputRef.current?.click()}>
              <Upload size={18} />
              <span>{file ? file.name : "选择图片"}</span>
            </button>
          </section>

          <section className="tool-section">
            <div className="section-title">
              <SlidersHorizontal size={18} />
              <span>置信度 {formatConfidence(confidence)}</span>
            </div>
            <input
              className="confidence-slider"
              type="range"
              min="0.1"
              max="0.9"
              step="0.01"
              value={confidence}
              onChange={(event) => setConfidence(Number(event.target.value))}
            />
            <div className="slider-scale">
              <span>10%</span>
              <span>90%</span>
            </div>
          </section>

          <div className="action-row">
            <button className="primary-button" type="button" onClick={runDetection} disabled={isDetecting || !file}>
              {isDetecting ? <Loader2 className="spin" size={18} /> : <ScanSearch size={18} />}
              <span>{isDetecting ? "检测中" : "开始检测"}</span>
            </button>
            <button className="icon-button" type="button" onClick={downloadResult} disabled={!result?.annotatedImage} title="下载结果">
              <Download size={18} />
            </button>
          </div>

          {error && (
            <div className="alert">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}

          <section className="results-section">
            <div className="section-title">
              <ScanSearch size={18} />
              <span>检测结果</span>
            </div>
            <div className="result-count">{result ? `${result.summary.total} 个目标` : "等待检测"}</div>
            <div className="result-list">
              {result?.detections?.map((item, index) => (
                <button
                  className={`result-item ${index === selectedIndex ? "active" : ""}`}
                  key={`${item.className}-${index}`}
                  type="button"
                  onClick={() => setSelectedIndex(index)}
                >
                  <span>{item.className}</span>
                  <strong>{formatConfidence(item.confidence)}</strong>
                </button>
              ))}
              {result && result.detections.length === 0 && <p className="empty-text">未检测到古钱币。</p>}
              {!result && <p className="empty-text">选择图片后运行检测。</p>}
            </div>
          </section>
        </aside>

        <main className="image-workspace">
          {displayImage ? (
            <img className="preview-image" src={displayImage} alt="检测预览" />
          ) : (
            <div className="empty-preview">
              <ImageIcon size={38} />
              <span>选择一张图片开始</span>
            </div>
          )}
        </main>

        <aside className="detail-panel">
          <section className="summary-box">
            <span className="summary-label">汇总</span>
            <strong>{result ? result.summary.total : 0}</strong>
            <span>目标数量</span>
          </section>

          <section className="class-breakdown">
            {classSummary.length > 0 ? (
              classSummary.map(([name, count]) => (
                <div className="breakdown-row" key={name}>
                  <span>{name}</span>
                  <strong>{count}</strong>
                </div>
              ))
            ) : (
              <p className="empty-text">暂无类别统计。</p>
            )}
          </section>

          <section className="knowledge-box">
            <span className="summary-label">科普说明</span>
            {selectedDetection ? (
              <>
                <h2>{selectedDetection.className}</h2>
                <p>{selectedDetection.knowledge}</p>
                <dl>
                  <div>
                    <dt>置信度</dt>
                    <dd>{formatConfidence(selectedDetection.confidence)}</dd>
                  </div>
                  <div>
                    <dt>位置</dt>
                    <dd>
                      x {formatNumber(selectedDetection.box.x1)} / y {formatNumber(selectedDetection.box.y1)}
                    </dd>
                  </div>
                  <div>
                    <dt>尺寸</dt>
                    <dd>
                      {formatNumber(selectedDetection.box.width)} x {formatNumber(selectedDetection.box.height)}
                    </dd>
                  </div>
                </dl>
              </>
            ) : (
              <p className="empty-text">点击检测结果查看详情。</p>
            )}
          </section>
        </aside>
      </div>
    </div>
  );
}

export default App;

