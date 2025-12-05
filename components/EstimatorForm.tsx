import React, { useState, useEffect, useCallback } from 'react';
import { ProjectType, QualityLevel, ProjectInputs, FeasibilityResult } from '../types';
import { Loader2, CheckCircle2, AlertTriangle, ArrowRight, ShieldCheck, ShieldAlert, BadgeCheck } from 'lucide-react';

interface Props {
  onEstimate: (inputs: ProjectInputs) => void;
  onFeasibilityCheck: (inputs: ProjectInputs) => Promise<any>;
  isChecking: boolean;
  feasibilityData: FeasibilityResult | null;
}

const EstimatorForm: React.FC<Props> = ({ onEstimate, onFeasibilityCheck, isChecking, feasibilityData }) => {
  const [inputs, setInputs] = useState<ProjectInputs>({
    type: ProjectType.RESIDENTIAL,
    quality: QualityLevel.STANDARD,
    location: '',
    sizeSqFt: 1000,
    budgetLimit: 5000000, // Default to a realistic number like 50 Lakhs
    timelineMonths: 6,
    manpower: 5
  });

  // Debounce logic for auto-feasibility
  useEffect(() => {
    const timer = setTimeout(() => {
      // Only check if we have minimum viable inputs
      if (inputs.location.length > 2 && inputs.sizeSqFt > 0 && inputs.budgetLimit > 0) {
        onFeasibilityCheck(inputs);
      }
    }, 1500); // 1.5s debounce

    return () => clearTimeout(timer);
  }, [inputs.location, inputs.sizeSqFt, inputs.budgetLimit, inputs.type, inputs.quality, inputs.timelineMonths, inputs.manpower]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setInputs(prev => ({
      ...prev,
      [name]: name === 'sizeSqFt' || name === 'budgetLimit' || name === 'timelineMonths' || name === 'manpower' 
        ? Number(value) 
        : value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onEstimate(inputs);
  };

  const getVerdictStyles = (verdict: string) => {
    switch (verdict) {
      case 'Realistic':
        return { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-700', icon: ShieldCheck };
      case 'Excessive':
        return { bg: 'bg-blue-50', border: 'border-blue-200', text: 'text-blue-700', icon: BadgeCheck };
      case 'Insufficient':
      default:
        return { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', icon: ShieldAlert };
    }
  };

  const verdictConfig = feasibilityData ? getVerdictStyles(feasibilityData.budgetVerdict) : null;
  const VerdictIcon = verdictConfig?.icon;

  return (
    <form onSubmit={handleSubmit} className="bg-white p-6 rounded-xl shadow-lg border border-slate-100">
      <h2 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
        <span className="bg-blue-600 w-2 h-6 rounded-full"></span>
        Project Details
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Project Type</label>
          <select 
            name="type" 
            value={inputs.type} 
            onChange={handleChange}
            className="w-full px-4 py-2 bg-white text-slate-900 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          >
            {Object.values(ProjectType).map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Quality Level</label>
          <select 
            name="quality" 
            value={inputs.quality} 
            onChange={handleChange}
            className="w-full px-4 py-2 bg-white text-slate-900 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          >
            {Object.values(QualityLevel).map(q => <option key={q} value={q}>{q}</option>)}
          </select>
        </div>

        <div className="md:col-span-2">
          <label className="block text-sm font-medium text-slate-700 mb-1">Location (City, State/Country)</label>
          <input 
            type="text" 
            name="location" 
            value={inputs.location} 
            onChange={handleChange}
            placeholder="e.g., Mumbai, India"
            className="w-full px-4 py-2 bg-white text-slate-900 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          />
          <p className="text-xs text-slate-500 mt-1">AI fetches local market rates for this location.</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Size (Sq Ft)</label>
          <input 
            type="number" 
            name="sizeSqFt" 
            value={inputs.sizeSqFt} 
            onChange={handleChange}
            className="w-full px-4 py-2 bg-white text-slate-900 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Budget Limit</label>
          <input 
            type="number" 
            name="budgetLimit" 
            value={inputs.budgetLimit} 
            onChange={handleChange}
            className="w-full px-4 py-2 bg-white text-slate-900 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Timeline (Months)</label>
          <input 
            type="number" 
            name="timelineMonths" 
            value={inputs.timelineMonths} 
            onChange={handleChange}
            className="w-full px-4 py-2 bg-white text-slate-900 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Available Manpower</label>
          <input 
            type="number" 
            name="manpower" 
            value={inputs.manpower} 
            onChange={handleChange}
            className="w-full px-4 py-2 bg-white text-slate-900 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          />
        </div>
      </div>

      <div className="mt-8 transition-all duration-300 ease-in-out">
        {isChecking ? (
          <div className="flex items-center gap-2 text-slate-500 p-4 bg-slate-50 rounded-lg border border-slate-100">
            <Loader2 className="animate-spin w-5 h-5" />
            <span className="text-sm font-medium">Analyzing market feasibility in real-time...</span>
          </div>
        ) : feasibilityData && verdictConfig && VerdictIcon ? (
          <div className={`p-4 rounded-lg border transition-all ${verdictConfig.bg} ${verdictConfig.border} mb-4`}>
            <div className="flex items-start gap-3">
              <VerdictIcon className={`w-6 h-6 mt-1 ${verdictConfig.text}`} />
              <div>
                <h3 className="font-bold text-slate-900 flex items-center gap-2">
                  Budget is {feasibilityData.budgetVerdict}
                </h3>
                <p className="text-sm mt-1 font-medium opacity-90">
                  {feasibilityData.budgetVerdict === 'Realistic' 
                    ? "Great! Your budget aligns with market rates." 
                    : feasibilityData.budgetVerdict === 'Excessive'
                    ? "Your budget is significantly higher than market average. You can easily afford premium quality."
                    : "Budget appears too low for this location and size."
                  }
                </p>
                <ul className="text-sm mt-2 space-y-1">
                  {feasibilityData.issues.slice(0, 2).map((s: string, i: number) => (
                    <li key={i} className="text-slate-700 opacity-80">• {s}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ) : null}
      </div>

      <div className="mt-4 pt-4 border-t border-slate-200 flex justify-end">
        <button 
          type="submit"
          className="w-full md:w-auto px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-bold shadow-md shadow-blue-200 flex items-center justify-center gap-2"
        >
          Generate Detailed Estimate <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </form>
  );
};

export default EstimatorForm;