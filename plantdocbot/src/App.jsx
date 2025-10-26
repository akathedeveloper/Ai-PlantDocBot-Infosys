import { useState } from "react";
import axios from "axios";

function App() {
  const [textInput, setTextInput] = useState("");
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [output, setOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("text");

  const handleTextPrediction = async () => {
    if (!textInput.trim()) return;
    setLoading(true);
    setOutput(null);

    try {
      const res = await axios.post("http://127.0.0.1:8000/text-prediction", {
        input: textInput,
      });
      setOutput(res.data);
    } catch (err) {
      setOutput({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  const handleImagePrediction = async () => {
    if (!image) return;
    setLoading(true);
    setOutput(null);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await axios.post("http://127.0.0.1:8000/image-prediction", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setOutput(res.data);
    } catch (err) {
      setOutput({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Helper function to parse confidence value
  const getConfidenceWidth = (confidence) => {
    if (!confidence) return "0%";
    // If it's already a percentage string like "95%", use it directly
    if (typeof confidence === "string" && confidence.includes("%")) {
      return confidence;
    }
    // If it's a decimal like 0.95, convert to percentage
    if (typeof confidence === "number" && confidence <= 1) {
      return `${confidence * 100}%`;
    }
    // If it's a number like 95, add % sign
    if (typeof confidence === "number") {
      return `${confidence}%`;
    }
    return confidence;
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-white/80 backdrop-blur-xl border-b border-neutral-200 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <svg className="w-6 h-6 text-emerald-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/>
              <path d="M2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
            <span className="text-lg font-medium text-neutral-900">PlantDoc</span>
          </div>
          <div className="flex items-center gap-6">
            <a href="#" className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors">Docs</a>
            <a href="#" className="text-sm text-neutral-600 hover:text-neutral-900 transition-colors">API</a>
            <button className="text-sm px-4 py-2 bg-neutral-900 text-white rounded-md hover:bg-neutral-800 transition-colors">
              Sign In
            </button>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl md:text-6xl font-medium text-neutral-900 tracking-tight mb-6">
            AI-powered plant<br />disease detection
          </h1>
          <p className="text-xl text-neutral-600 max-w-2xl mx-auto mb-4">
            Diagnose plant diseases instantly with computer vision. Upload an image or describe symptoms.
          </p>
          <div className="flex items-center justify-center gap-2 text-sm text-neutral-500">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
            <span>System operational</span>
          </div>
        </div>
      </section>

      {/* Main Interface */}
      <section className="pb-24 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="bg-white border border-neutral-200 rounded-lg overflow-hidden shadow-sm">
            {/* Tabs */}
            <div className="flex border-b border-neutral-200">
              <button 
                onClick={() => setActiveTab("text")}
                className={`flex-1 px-6 py-4 text-sm font-medium transition-colors ${
                  activeTab === "text" 
                    ? "text-neutral-900 border-b-2 border-neutral-900" 
                    : "text-neutral-500 hover:text-neutral-900"
                }`}
              >
                Text Input
              </button>
              <button 
                onClick={() => setActiveTab("image")}
                className={`flex-1 px-6 py-4 text-sm font-medium transition-colors ${
                  activeTab === "image" 
                    ? "text-neutral-900 border-b-2 border-neutral-900" 
                    : "text-neutral-500 hover:text-neutral-900"
                }`}
              >
                Image Upload
              </button>
            </div>

            {/* Content */}
            <div className="p-8">
              {/* Text Input Tab */}
              {activeTab === "text" && (
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-neutral-900 mb-3">
                      Describe the symptoms
                    </label>
                    <textarea
                      value={textInput}
                      onChange={(e) => setTextInput(e.target.value)}
                      placeholder="Yellow spots on tomato leaves with brown edges..."
                      className="w-full px-4 py-3 border border-neutral-300 rounded-md focus:outline-none focus:ring-2 focus:ring-neutral-900 focus:border-transparent resize-none h-32 text-neutral-900 placeholder-neutral-400 font-light"
                    />
                  </div>

                  <button
                    onClick={handleTextPrediction}
                    disabled={loading || !textInput.trim()}
                    className="w-full px-6 py-3 bg-neutral-900 text-white text-sm font-medium rounded-md hover:bg-neutral-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    {loading ? "Analyzing..." : "Analyze"}
                  </button>
                </div>
              )}

              {/* Image Upload Tab */}
              {activeTab === "image" && (
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-neutral-900 mb-3">
                      Upload image
                    </label>
                    <div className="relative">
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleImageChange}
                        className="hidden"
                        id="file-upload"
                      />
                      <label
                        htmlFor="file-upload"
                        className="block w-full h-48 border-2 border-dashed border-neutral-300 rounded-md hover:border-neutral-400 transition-colors cursor-pointer"
                      >
                        {imagePreview ? (
                          <div className="relative w-full h-full p-4">
                            <img
                              src={imagePreview}
                              alt="Preview"
                              className="w-full h-full object-contain"
                            />
                            <button
                              onClick={(e) => {
                                e.preventDefault();
                                setImage(null);
                                setImagePreview(null);
                              }}
                              className="absolute top-6 right-6 w-6 h-6 bg-neutral-900 text-white rounded-full flex items-center justify-center hover:bg-neutral-800 transition-colors"
                            >
                              ×
                            </button>
                          </div>
                        ) : (
                          <div className="flex flex-col items-center justify-center h-full">
                            <svg className="w-8 h-8 text-neutral-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                            </svg>
                            <span className="text-sm text-neutral-600">Drop image here or click to upload</span>
                            <span className="text-xs text-neutral-400 mt-1">PNG, JPG up to 10MB</span>
                          </div>
                        )}
                      </label>
                    </div>
                  </div>

                  <button
                    onClick={handleImagePrediction}
                    disabled={loading || !image}
                    className="w-full px-6 py-3 bg-neutral-900 text-white text-sm font-medium rounded-md hover:bg-neutral-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    {loading ? "Analyzing..." : "Analyze"}
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Results */}
          {output && (
            <div className="mt-6 bg-white border border-neutral-200 rounded-lg p-8 shadow-sm">
              {output.error ? (
                <div className="flex items-start gap-3">
                  <svg className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                  <div>
                    <p className="text-sm font-medium text-neutral-900">Error</p>
                    <p className="text-sm text-neutral-600 mt-1">{output.error}</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-6">
                  <div>
                    <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider mb-2">Diagnosis</p>
                    <h3 className="text-2xl font-medium text-neutral-900">{output.label || output.prediction}</h3>
                  </div>

                  {output.confidence && (
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-neutral-600">Confidence</span>
                        <span className="text-sm font-medium text-neutral-900">{output.confidence}</span>
                      </div>
                      <div className="w-full h-1 bg-neutral-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-neutral-900 transition-all duration-500"
                          style={{ width: getConfidenceWidth(output.confidence) }}
                        />
                      </div>
                    </div>
                  )}

                  {output.recommendation && (
                    <div className="pt-6 border-t border-neutral-200">
                      <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider mb-3">Recommendation</p>
                      <p className="text-sm text-neutral-700 leading-relaxed">{output.recommendation}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      {/* Features */}
      <section className="py-24 px-6 border-t border-neutral-200">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-3 gap-12">
            <div>
              <div className="w-10 h-10 bg-neutral-100 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-5 h-5 text-neutral-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-base font-medium text-neutral-900 mb-2">Instant detection</h3>
              <p className="text-sm text-neutral-600 leading-relaxed">
                Get results in seconds with our advanced ML models trained on thousands of plant diseases.
              </p>
            </div>
            <div>
              <div className="w-10 h-10 bg-neutral-100 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-5 h-5 text-neutral-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-base font-medium text-neutral-900 mb-2">High accuracy</h3>
              <p className="text-sm text-neutral-600 leading-relaxed">
                Proven accuracy across multiple crop types and disease classifications.
              </p>
            </div>
            <div>
              <div className="w-10 h-10 bg-neutral-100 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-5 h-5 text-neutral-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
              </div>
              <h3 className="text-base font-medium text-neutral-900 mb-2">Expert recommendations</h3>
              <p className="text-sm text-neutral-600 leading-relaxed">
                Receive actionable treatment plans based on the detected disease.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-neutral-200">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-neutral-500">
            © 2025 PlantDoc. Built by Adhiraj.
          </p>
          <div className="flex items-center gap-8">
            <a href="#" className="text-sm text-neutral-500 hover:text-neutral-900 transition-colors">Documentation</a>
            <a href="#" className="text-sm text-neutral-500 hover:text-neutral-900 transition-colors">API</a>
            <a href="#" className="text-sm text-neutral-500 hover:text-neutral-900 transition-colors">Privacy</a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
