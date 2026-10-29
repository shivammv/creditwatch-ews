import React, { useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Award,
  BookOpen,
  Building2,
  CheckCircle2,
  ChevronRight,
  Cpu,
  FileSpreadsheet,
  FileText,
  HelpCircle,
  LayoutDashboard,
  Layers,
  Menu,
  Shield,
  ShieldAlert,
  Sliders,
  Users,
  X
} from 'lucide-react';

import { CompanyAnalysisPage } from './pages/CompanyAnalysisPage';
import { DashboardPage } from './pages/DashboardPage';
import { DataUploadPage } from './pages/DataUploadPage';
import { ForensicsPage } from './pages/ForensicsPage';
import { MethodologyPage } from './pages/MethodologyPage';
import { ModelPerformancePage } from './pages/ModelPerformancePage';
import { WatchlistPage } from './pages/WatchlistPage';

export type TabType =
  | 'dashboard'
  | 'company'
  | 'forensics'
  | 'watchlist'
  | 'model'
  | 'upload'
  | 'methodology';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [selectedCompany, setSelectedCompany] = useState<string | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

  const handleSelectCompany = (companyName: string) => {
    setSelectedCompany(companyName);
    setActiveTab('company');
  };

  const navItems = [
    { id: 'dashboard' as TabType, label: 'Executive Dashboard', icon: LayoutDashboard },
    { id: 'company' as TabType, label: 'Company Risk Dossier', icon: Building2 },
    { id: 'forensics' as TabType, label: 'Forensic Accounting', icon: ShieldAlert },
    { id: 'watchlist' as TabType, label: 'Portfolio Watchlist', icon: Users },
    { id: 'model' as TabType, label: 'Model Benchmarking', icon: Cpu },
    { id: 'upload' as TabType, label: 'Data Ingestion', icon: FileSpreadsheet },
    { id: 'methodology' as TabType, label: 'Methodology & Policy', icon: BookOpen },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
      {/* Enterprise Top Navigation Bar */}
      <header className="bg-slate-900 border-b border-slate-800 text-white sticky top-0 z-40 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Brand Logo & Platform Title */}
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center text-white shadow-md">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-extrabold text-base tracking-tight text-white">CreditWatch</span>
                  <span className="px-1.5 py-0.2 rounded text-[10px] font-bold bg-brand-500/30 text-brand-300 border border-brand-500/40 uppercase">
                    EWS
                  </span>
                </div>
                <div className="text-[10px] text-slate-400 font-medium tracking-wide">
                  Corporate Credit Distress & Forensic Risk Analytics
                </div>
              </div>
            </div>

            {/* Desktop Navigation Tabs */}
            <nav className="hidden lg:flex items-center gap-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                      isActive
                        ? 'bg-brand-700 text-white shadow-sm'
                        : 'text-slate-300 hover:text-white hover:bg-slate-800'
                    }`}
                  >
                    <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-brand-300' : 'text-slate-400'}`} />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>

            {/* Live System Status Pill */}
            <div className="hidden sm:flex items-center gap-3">
              <div className="flex items-center gap-2 bg-slate-800/80 border border-slate-700/60 px-3 py-1.5 rounded-full text-xs">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-slate-300 font-medium text-[11px]">
                  Model <strong className="text-white">v2.1</strong> Calibrated
                </span>
              </div>
            </div>

            {/* Mobile Hamburger Button */}
            <div className="lg:hidden flex items-center">
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="p-2 rounded-md text-slate-400 hover:text-white hover:bg-slate-800"
              >
                {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Navigation Drawer */}
        {mobileMenuOpen && (
          <div className="lg:hidden bg-slate-900 border-b border-slate-800 px-4 pt-2 pb-4 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    setActiveTab(item.id);
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm font-semibold transition ${
                    isActive
                      ? 'bg-brand-700 text-white'
                      : 'text-slate-300 hover:text-white hover:bg-slate-800'
                  }`}
                >
                  <Icon className="w-4 h-4 text-slate-400" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </div>
        )}
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'dashboard' && (
          <DashboardPage
            onSelectCompany={handleSelectCompany}
            onNavigateToWatchlist={() => setActiveTab('watchlist')}
          />
        )}

        {activeTab === 'company' && (
          <CompanyAnalysisPage initialCompany={selectedCompany} />
        )}

        {activeTab === 'forensics' && (
          <ForensicsPage onSelectCompany={handleSelectCompany} />
        )}

        {activeTab === 'watchlist' && (
          <WatchlistPage onSelectCompany={handleSelectCompany} />
        )}

        {activeTab === 'model' && <ModelPerformancePage />}

        {activeTab === 'upload' && (
          <DataUploadPage onNavigateToWatchlist={() => setActiveTab('watchlist')} />
        )}

        {activeTab === 'methodology' && <MethodologyPage />}
      </main>

      {/* Institutional Enterprise Footer */}
      <footer className="bg-white border-t border-slate-200 mt-auto py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-bold text-slate-800">CreditWatch EWS Platform</span>
            <span>— Corporate Financial Distress & Fraud Risk Screening</span>
          </div>

          <div className="flex items-center gap-4 text-slate-400">
            <span>Asymmetric Cost 20:1 FN:FP</span>
            <span>•</span>
            <span>Bayes Cutoff 4.76%</span>
            <span>•</span>
            <span>5% Review Cap Enforced</span>
            <span>•</span>
            <span>Zeckathon 2026</span>
          </div>
        </div>
      </footer>
    </div>
  );
};
export default App;
