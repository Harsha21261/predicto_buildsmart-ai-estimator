import React from 'react';
import { EstimationResult } from '../types';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, BarChart, XAxis, YAxis, Bar, CartesianGrid } from 'recharts';
import { TrendingUp, AlertOctagon, Lightbulb, MapPin, Wallet, CheckCircle2 } from 'lucide-react';

interface Props {
  result: EstimationResult;
  location: string;
  userBudget: number; // passed from parent to compare
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#6366f1'];

const ResultsDashboard: React.FC<Props> = ({ result, location, userBudget }) => {
  // Safe access to arrays and numbers
  const risks = result.risks || [];
  const breakdown = result.breakdown || [];
  const cashflow = result.cashflow || [];
  const tips = result.efficiencyTips || [];
  
  const estimatedCost = result.totalEstimatedCost || 0;
  const budgetLimit = userBudget || 0;
  const currency = result.currencySymbol || '$';

  // Budget Calculations
  const budgetDiff = budgetLimit - estimatedCost; // Positive = Savings, Negative = Deficit
  const isWithinBudget = budgetDiff >= 0;
  
  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      
      {/* --- Top Stats Section: Clean & Explicit --- */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Card 1: Estimated Cost (The Answer) */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col justify-between relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Wallet className="w-16 h-16 text-blue-600" />
          </div>
          <div>
            <p className="text-xs font-bold text-slate-500 uppercase tracking-wider">Total Estimated Cost</p>
            <h2 className="text-3xl font-extrabold text-slate-900 mt-2 tracking-tight">
              {currency} {estimatedCost.toLocaleString()}
            </h2>
          </div>
          <div className="mt-4 flex items-center gap-2 text-xs text-slate-500 bg-slate-50 p-2 rounded w-fit">
             <MapPin className="w-3 h-3" />
             <span>Based on {location} rates</span>
          </div>
        </div>

        {/* Card 2: Budget Analysis (The Comparison) */}
        <div className={`p-6 rounded-xl shadow-sm border flex flex-col justify-between ${isWithinBudget ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
          <div>
            <div className="flex justify-between items-start">
               <p className={`text-xs font-bold uppercase tracking-wider ${isWithinBudget ? 'text-green-700' : 'text-red-700'}`}>
                 {isWithinBudget ? 'Budget Surplus' : 'Budget Deficit'}
               </p>
               {isWithinBudget 
                 ? <div className="bg-green-100 p-1.5 rounded-full"><TrendingUp className="w-4 h-4 text-green-700" /></div>
                 : <div className="bg-red-100 p-1.5 rounded-full"><AlertOctagon className="w-4 h-4 text-red-700" /></div>
               }
            </div>
            <h2 className={`text-3xl font-extrabold mt-2 ${isWithinBudget ? 'text-green-800' : 'text-red-800'}`}>
              {isWithinBudget ? '+' : '-'}{currency} {Math.abs(budgetDiff).toLocaleString()}
            </h2>
          </div>
          
          {/* Explicit comparison line to resolve "messy" confusion */}
          <div className="mt-4 pt-3 border-t border-black/5 flex justify-between items-center text-sm">
             <span className="text-slate-600 font-medium opacity-80">Your Limit:</span>
             <span className="font-bold text-slate-900">{currency} {budgetLimit.toLocaleString()}</span>
          </div>
        </div>

        {/* Card 3: AI Confidence */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col justify-between">
          <div>
            <p className="text-xs font-bold text-slate-500 uppercase tracking-wider">AI Confidence Score</p>
            <div className="flex items-end gap-2 mt-2">
              <h2 className={`text-3xl font-extrabold ${result.confidenceScore > 75 ? 'text-green-600' : result.confidenceScore > 50 ? 'text-yellow-600' : 'text-red-600'}`}>
                {result.confidenceScore}%
              </h2>
              <span className="text-sm text-slate-400 mb-1 font-medium">accuracy probability</span>
            </div>
          </div>
          <div className="mt-4">
             <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
               <div 
                 className={`h-full transition-all duration-1000 ${result.confidenceScore > 75 ? 'bg-green-500' : result.confidenceScore > 50 ? 'bg-yellow-500' : 'bg-red-500'}`} 
                 style={{ width: `${result.confidenceScore}%` }}
               />
             </div>
             <p className="text-xs text-slate-400 mt-2 truncate" title={result.confidenceReason}>{result.confidenceReason}</p>
          </div>
        </div>

      </div>

      {/* --- Charts Section --- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Cost Breakdown */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2">
               <div className="w-1 h-6 bg-blue-500 rounded-full"></div>
               Cost Breakdown
            </h3>
          </div>
          
          {breakdown.length > 0 ? (
            <div className="flex flex-col md:flex-row items-center gap-6">
              <div className="h-64 w-full md:w-1/2">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={breakdown}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={4}
                      dataKey="cost"
                    >
                      {breakdown.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip 
                      contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                      formatter={(value: number) => [`${currency} ${value.toLocaleString()}`, 'Cost']} 
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="w-full md:w-1/2 space-y-3 max-h-64 overflow-y-auto pr-2 custom-scrollbar">
                {breakdown.map((item, index) => (
                  <div key={index} className="flex justify-between items-center text-sm p-2 hover:bg-slate-50 rounded-lg transition-colors group">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: COLORS[index % COLORS.length] }}></div>
                      <span className="text-slate-600 font-medium group-hover:text-slate-900">{item.category}</span>
                    </div>
                    <span className="font-bold text-slate-800">{currency} {item.cost.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-slate-400 bg-slate-50 rounded-lg border border-dashed border-slate-200">
              <p>No breakdown details available.</p>
            </div>
          )}
        </div>

        {/* Cashflow */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2">
              <div className="w-1 h-6 bg-green-500 rounded-full"></div>
              Projected Cashflow
            </h3>
          </div>
          {cashflow.length > 0 ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={cashflow} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis 
                    dataKey="month" 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{fill: '#64748b', fontSize: 12}}
                    dy={10}
                    label={{ value: 'Month', position: 'insideBottom', offset: -5, fill: '#94a3b8', fontSize: 10 }} 
                  />
                  <YAxis 
                    axisLine={false} 
                    tickLine={false}
                    tick={{fill: '#64748b', fontSize: 12}}
                    tickFormatter={(val) => `${val/1000}k`} 
                  />
                  <RechartsTooltip 
                    cursor={{fill: '#f1f5f9'}}
                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                    formatter={(value: number) => [`${currency} ${value.toLocaleString()}`, 'Amount']}
                  />
                  <Bar dataKey="amount" fill="#3b82f6" radius={[4, 4, 0, 0]} maxBarSize={50} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-slate-400 bg-slate-50 rounded-lg border border-dashed border-slate-200">
              <p>No cashflow data available.</p>
            </div>
          )}
        </div>
      </div>

      {/* --- Risks & Tips Section --- */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
            <AlertOctagon className="w-5 h-5 text-red-500" /> Risk Analysis
          </h3>
          <div className="space-y-3">
            {risks.length > 0 ? risks.map((risk, i) => (
              <div key={i} className="p-4 bg-slate-50 rounded-lg border border-slate-100 hover:border-slate-200 transition-colors">
                <div className="flex justify-between items-start mb-2">
                  <span className="font-semibold text-slate-900">{risk.risk}</span>
                  <span className={`text-[10px] uppercase font-bold px-2 py-1 rounded-full ${
                    risk.impact === 'High' ? 'bg-red-100 text-red-700' : 
                    risk.impact === 'Medium' ? 'bg-yellow-100 text-yellow-700' : 'bg-blue-100 text-blue-700'
                  }`}>
                    {risk.impact} Impact
                  </span>
                </div>
                <p className="text-sm text-slate-600 leading-relaxed"><span className="font-medium text-slate-700">Mitigation:</span> {risk.mitigation}</p>
              </div>
            )) : <p className="text-slate-500 text-sm">No significant risks identified.</p>}
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-yellow-500" /> Smart Efficiency Tips
          </h3>
          <ul className="space-y-3">
            {tips.length > 0 ? tips.map((tip, i) => (
              <li key={i} className="flex items-start gap-3 text-sm text-slate-700 p-3 bg-yellow-50/50 rounded-lg border border-yellow-100/50">
                <div className="mt-0.5 bg-yellow-100 p-1 rounded-full text-yellow-600">
                  <CheckCircle2 className="w-3 h-3" />
                </div>
                <span className="leading-relaxed">{tip}</span>
              </li>
            )) : <p className="text-slate-500 text-sm">No tips available.</p>}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ResultsDashboard;