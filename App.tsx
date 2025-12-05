
import React, { useState } from 'react';
import { ProjectInputs, EstimationResult, FeasibilityResult, ProjectType, QualityLevel } from './types';
import EstimatorForm from './components/EstimatorForm';
import ResultsDashboard from './components/ResultsDashboard';
import ChatInterface from './components/ChatInterface';
import { checkProjectFeasibility, generateConstructionEstimate } from './services/geminiService';
import { LayoutDashboard, MessageSquare, HardHat } from 'lucide-react';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'estimator' | 'chat'>('estimator');
  const [loading, setLoading] = useState(false);
  const [estimationResult, setEstimationResult] = useState<EstimationResult | null>(null);
  
  // Feasibility State
  const [feasibilityChecking, setFeasibilityChecking] = useState(false);
  const [feasibilityData, setFeasibilityData] = useState<FeasibilityResult | null>(null);
  
  // Keep track of current inputs for the dashboard comparison
  const [currentInputs, setCurrentInputs] = useState<ProjectInputs>({
    type: ProjectType.RESIDENTIAL,
    quality: QualityLevel.STANDARD,
    location: '',
    sizeSqFt: 1000,
    budgetLimit: 0,
    timelineMonths: 6,
    manpower: 5
  });

  const handleFeasibilityCheck = async (inputs: ProjectInputs) => {
    setFeasibilityChecking(true);
    setCurrentInputs(inputs);
    try {
      const result = await checkProjectFeasibility(inputs);
      setFeasibilityData(result);
    } catch (error) {
      console.error("Feasibility error", error);
    } finally {
      setFeasibilityChecking(false);
    }
  };

  const handleEstimate = async (inputs: ProjectInputs) => {
    setLoading(true);
    setEstimationResult(null);
    setCurrentInputs(inputs);
    try {
      const result = await generateConstructionEstimate(inputs);
      setEstimationResult(result);
    } catch (error) {
      console.error(error);
      alert("Estimation failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-blue-600 p-2 rounded-lg">
              <HardHat className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-xl font-bold text-slate-800 tracking-tight">BuildSmart <span className="text-blue-600">AI</span></h1>
          </div>
          
          <nav className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg">
            <button 
              onClick={() => setActiveTab('estimator')}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'estimator' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-600 hover:text-slate-900'}`}
            >
              <div className="flex items-center gap-2">
                <LayoutDashboard className="w-4 h-4" /> Estimator
              </div>
            </button>
            <button 
              onClick={() => setActiveTab('chat')}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'chat' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-600 hover:text-slate-900'}`}
            >
              <div className="flex items-center gap-2">
                <MessageSquare className="w-4 h-4" /> Assistant
              </div>
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {activeTab === 'estimator' && (
          <div className="space-y-8">
            <div className="max-w-3xl mx-auto text-center mb-8">
              <h2 className="text-3xl font-bold text-slate-900 mb-2">Intelligent Cost Estimation</h2>
              <p className="text-slate-500">
                Powered by AI. Get real-time market insights, 
                feasibility scores, and detailed breakdowns for any global location.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              <div className="lg:col-span-4 space-y-6">
                <EstimatorForm 
                  onEstimate={handleEstimate}
                  onFeasibilityCheck={handleFeasibilityCheck}
                  isChecking={feasibilityChecking}
                  feasibilityData={feasibilityData}
                />
              </div>

              <div className="lg:col-span-8">
                 {loading ? (
                   <div className="h-full min-h-[400px] flex flex-col items-center justify-center bg-white rounded-xl border border-slate-100 shadow-sm p-8 text-center">
                     <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
                     <h3 className="text-lg font-semibold text-slate-800">Analyzing Project...</h3>
                     <p className="text-slate-500 max-w-md mt-2">
                       AI is using "Thinking Mode" to calculate complex costs, analyze risks, and fetch local market data. This may take a moment.
                     </p>
                   </div>
                 ) : estimationResult ? (
                   <ResultsDashboard 
                     result={estimationResult} 
                     location={currentInputs.location}
                     userBudget={currentInputs.budgetLimit}
                   />
                 ) : (
                   <div className="h-full min-h-[400px] flex flex-col items-center justify-center bg-white rounded-xl border border-dashed border-slate-300 p-8 text-center">
                     <div className="bg-slate-50 p-4 rounded-full mb-4">
                       <LayoutDashboard className="w-8 h-8 text-slate-400" />
                     </div>
                     <h3 className="text-lg font-medium text-slate-900">No Estimate Generated Yet</h3>
                     <p className="text-slate-500 max-w-sm mt-1">
                       Fill out the project details on the left and click "Generate" to see a comprehensive AI analysis.
                     </p>
                   </div>
                 )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'chat' && (
          <div className="max-w-2xl mx-auto">
            <ChatInterface />
          </div>
        )}

      </main>

      <footer className="bg-white border-t border-slate-200 mt-auto py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-slate-500">
          &copy; {new Date().getFullYear()} BuildSmart AI Estimator
        </div>
      </footer>
    </div>
  );
};

export default App;
