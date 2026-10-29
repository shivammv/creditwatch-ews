import React, { useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Database,
  Download,
  FileSpreadsheet,
  FileText,
  Info,
  RefreshCw,
  Table,
  UploadCloud,
  X
} from 'lucide-react';
import { uploadAndValidateFile } from '../api';

interface DataUploadPageProps {
  onNavigateToWatchlist: () => void;
}

export const DataUploadPage: React.FC<DataUploadPageProps> = ({ onNavigateToWatchlist }) => {
  const [file, setFile] = useState<File | null>(null);
  const [validating, setValidating] = useState<boolean>(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState<boolean>(false);

  const handleFileChange = (selected: File) => {
    setFile(selected);
    setResult(null);
    setError(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleValidate = async () => {
    if (!file) return;
    try {
      setValidating(true);
      setError(null);
      const res = await uploadAndValidateFile(file);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Validation failed');
    } finally {
      setValidating(false);
    }
  };

  const handleDownloadSample = () => {
    // Generate sample CSV for download
    const sampleCSV = `company_name,sector,financial_year,total_assets,gross_profit,ebit,sales,net_income,total_liabilities,current_assets,current_liabilities,cash,equity,retained_earnings,financial_expenses,cogs,receivables,depreciation,cfo
Sample Steel Ltd,Steel & Metals,FY24,12500,2400,1100,10500,-350,9200,4300,5100,220,3300,800,980,8100,1800,450,620
Solvent Tech Corp,IT & Tech,FY24,8400,3200,2100,9200,1650,1800,5200,1200,2400,6600,4500,60,6000,1400,180,2100`;

    const blob = new Blob([sampleCSV], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'creditwatch_sample_financials.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="badge badge-primary">Data Ingest Engine</span>
            <span className="badge badge-neutral">CSV & XLSX Support</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold text-slate-900 mt-2 tracking-tight">
            Financial Statement Ingestion & Schema Validator
          </h1>
          <p className="text-sm text-slate-600 mt-1 max-w-3xl">
            Upload institutional corporate balance sheets, trial balances, or P&L tables. CreditWatch verifies
            schema integrity, maps 39 portable financial ratios, identifies missing line items, and previews data.
          </p>
        </div>

        <button
          onClick={handleDownloadSample}
          className="btn btn-secondary flex items-center gap-2 text-xs shrink-0"
        >
          <Download className="w-3.5 h-3.5" />
          Download Sample CSV Template
        </button>
      </div>

      {/* Drag & Drop Area */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`card border-2 border-dashed transition p-8 text-center flex flex-col items-center justify-center ${
          dragOver ? 'border-brand-600 bg-brand-50/50' : 'border-slate-300 hover:border-slate-400 bg-slate-50/50'
        }`}
      >
        <UploadCloud className="w-12 h-12 text-slate-400 mb-3" />
        <h3 className="font-bold text-base text-slate-900">
          Drag and drop your financial statement file here
        </h3>
        <p className="text-xs text-slate-500 mt-1 mb-4">
          Supports <strong>.csv</strong> and <strong>.xlsx</strong> files up to 25MB (BSE/NSE formats or standard corporate filings)
        </p>

        <div className="flex items-center gap-3">
          <label className="btn btn-primary text-xs cursor-pointer">
            Browse Local File
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              className="hidden"
              onChange={(e) => e.target.files && handleFileChange(e.target.files[0])}
            />
          </label>
        </div>

        {file && (
          <div className="mt-4 p-3 bg-white rounded-lg border border-slate-200 flex items-center gap-3 text-xs font-semibold text-slate-800 shadow-sm">
            <FileSpreadsheet className="w-4 h-4 text-brand-700" />
            <span>{file.name}</span>
            <span className="text-slate-400">({(file.size / 1024).toFixed(1)} KB)</span>
            <button
              onClick={() => {
                setFile(null);
                setResult(null);
              }}
              className="text-slate-400 hover:text-red-600"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>

      {/* Action Button */}
      {file && !result && (
        <div className="flex justify-end">
          <button
            onClick={handleValidate}
            disabled={validating}
            className="btn btn-primary text-xs flex items-center gap-2"
          >
            {validating ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                Validating Schema & Checking Mappings...
              </>
            ) : (
              <>
                Validate & Map Schema <ArrowRight className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-5 text-red-800 text-xs">
          <div className="flex items-center gap-2 font-bold text-sm">
            <AlertCircle className="w-4 h-4 text-red-600" />
            <span>Validation Error</span>
          </div>
          <p className="mt-1">{error}</p>
        </div>
      )}

      {/* Validation Results & Preview */}
      {result && (
        <div className="space-y-6">
          {/* Summary Status Card */}
          <div className={`card border-l-4 ${
            result.success ? 'border-l-emerald-600' : 'border-l-amber-500'
          }`}>
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  {result.success ? (
                    <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                  ) : (
                    <AlertTriangle className="w-5 h-5 text-amber-500" />
                  )}
                  <h3 className="text-base font-bold text-slate-900">
                    {result.success ? 'Schema Validation Succeeded' : 'Schema Requires Attention'}
                  </h3>
                </div>
                <p className="text-xs text-slate-600 mt-1">
                  File: <strong>{result.filename}</strong> | Found <strong>{result.rows_count}</strong> rows & <strong>{result.columns_found.length}</strong> columns.
                </p>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={onNavigateToWatchlist}
                  className="btn btn-primary text-xs flex items-center gap-2"
                >
                  Proceed to Portfolio Scoring <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Recommendation & Warnings */}
            <div className="mt-4 pt-3 border-t border-slate-100 text-xs space-y-1">
              <div className="font-semibold text-slate-800">
                Analysis Engine Recommendation:
              </div>
              <p className="text-slate-600">{result.recommendation}</p>
            </div>
          </div>

          {/* Missing Fields or Validation Errors */}
          {result.missing_required && result.missing_required.length > 0 && (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs">
              <div className="font-bold text-amber-900 flex items-center gap-1.5 mb-1">
                <AlertTriangle className="w-4 h-4 text-amber-600" />
                Missing Core Fields ({result.missing_required.length}):
              </div>
              <p className="text-amber-800 mb-2">
                The following mandatory financial fields were not identified in the uploaded header. LightGBM can process missing ratios natively, but risk coverage is improved when provided:
              </p>
              <div className="flex flex-wrap gap-1.5">
                {result.missing_required.map((col: string) => (
                  <span key={col} className="px-2 py-0.5 rounded bg-white border border-amber-300 font-mono text-[11px] text-amber-900">
                    {col}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Mapped Financial Features */}
          <div className="card">
            <h4 className="text-sm font-bold text-slate-900 mb-2 pb-2 border-b border-slate-100">
              Auto-Mapped Financial Fields ({Object.keys(result.mapped_features || {}).length})
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              {Object.entries(result.mapped_features || {}).map(([feature, mappedCol]) => (
                <div key={feature} className="p-2 bg-slate-50 rounded border border-slate-200">
                  <span className="font-semibold text-brand-800 block">{feature}</span>
                  <span className="text-[11px] text-slate-500 font-mono">mapped to &quot;{String(mappedCol)}&quot;</span>
                </div>
              ))}
            </div>
          </div>

          {/* Data Preview Table */}
          {result.preview && result.preview.length > 0 && (
            <div className="card p-0 overflow-hidden">
              <div className="py-3 px-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                  Raw Ingestion Preview (First 5 Rows)
                </h4>
                <span className="text-[11px] text-slate-500 font-medium">Verify units (INR Crores / Lakhs)</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-100 text-slate-600 font-semibold border-b border-slate-200">
                    <tr>
                      {Object.keys(result.preview[0]).map((col) => (
                        <th key={col} className="py-2.5 px-3 whitespace-nowrap">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-medium font-mono text-[11px]">
                    {result.preview.map((row: any, idx: number) => (
                      <tr key={idx} className="hover:bg-slate-50">
                        {Object.values(row).map((val: any, vIdx: number) => (
                          <td key={vIdx} className="py-2 px-3 whitespace-nowrap text-slate-700">
                            {val !== null ? String(val) : '—'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
